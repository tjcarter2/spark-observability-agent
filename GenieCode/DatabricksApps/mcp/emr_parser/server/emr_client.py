"""EMR Persistent UI Client for Spark History Server access."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


class EMRPersistentUIClient:
    """Client for managing EMR Persistent App UI and HTTP sessions."""

    def __init__(self, cred_name: str, cluster_arn: str):
        """
        Initialize the EMR client.

        Args:
            cred_name: Databricks service credential name for AWS access
            cluster_arn: EMR cluster ARN
        """

        W = WorkspaceClient()
        tmpcreds = W.credentials.generate_temporary_service_credential(credential_name=cred_name)

        # Parse region from cluster ARN (arn:aws:elasticmapreduce:REGION:ACCOUNT:cluster/ID)
        region = cluster_arn.split(':')[3]

        emr_client = boto3.client(
            'emr',
            region_name=region,
            aws_access_key_id=tmpcreds.aws_temp_credentials.access_key_id,
            aws_secret_access_key=tmpcreds.aws_temp_credentials.secret_access_key,
            aws_session_token=tmpcreds.aws_temp_credentials.session_token
        )

        self.emr_client = emr_client
        self.emr_cluster_arn = cluster_arn

        self.session = requests.Session()
        self.persistent_ui_id: Optional[str] = None
        self.presigned_url: Optional[str] = None
        self.presigned_url_ready: bool = False
        self.base_url: Optional[str] = None
        self.api_base: Optional[str] = None
        self.timeout: int = 30
        self._attempt_cache: Dict[str, Optional[str]] = {}

    def create_persistent_app_ui(self) -> Dict:
        """
        Create a persistent app UI for the given cluster.

        Returns:
            Response from create-persistent-app-ui API call

        Raises:
            ClientError: If the API call fails
        """
        logger.info(f"Creating persistent app UI for cluster: {self.emr_cluster_arn}")

        try:
            response = self.emr_client.create_persistent_app_ui(
                TargetResourceArn=self.emr_cluster_arn
            )

            self.persistent_ui_id = response.get("PersistentAppUIId")
            runtime_role_enabled = response.get("RuntimeRoleEnabledCluster", False)

            logger.info("Persistent App UI created successfully")
            logger.debug(f"Persistent UI ID: {self.persistent_ui_id}")
            logger.debug(f"Runtime Role Enabled: {runtime_role_enabled}")

            return response

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(
                f"Failed to create persistent app UI: {error_code} - {error_message}"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating persistent app UI: {str(e)}")
            raise

    def describe_persistent_app_ui(self) -> Dict:
        """
        Describe the persistent app UI.

        Returns:
            Response from describe-persistent-app-ui API call

        Raises:
            ValueError: If no persistent UI ID is available
            ClientError: If the API call fails
        """
        if not self.persistent_ui_id:
            raise ValueError("No persistent UI ID available. Create one first.")

        logger.debug(f"Describing persistent app UI: {self.persistent_ui_id}")

        try:
            response = self.emr_client.describe_persistent_app_ui(
                PersistentAppUIId=self.persistent_ui_id
            )

            ui_details = response.get("PersistentAppUI", {})
            logger.debug(f"Status: {ui_details.get('PersistentAppUIStatus')}")

            return response

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(
                f"Failed to describe persistent app UI: {error_code} - {error_message}"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error describing persistent app UI: {str(e)}")
            raise

    def get_presigned_url(self, ui_type: str = "SHS") -> str:
        """
        Get presigned URL for the persistent app UI.

        Args:
            ui_type: Type of UI ('SHS' for Spark History Server)

        Returns:
            Presigned URL string

        Raises:
            ValueError: If no persistent UI ID is available
            ClientError: If the API call fails
        """
        if not self.persistent_ui_id:
            raise ValueError("No persistent UI ID available. Create one first.")

        logger.debug(
            f"Getting presigned URL for persistent app UI: {self.persistent_ui_id}"
        )

        try:
            response = self.emr_client.get_persistent_app_ui_presigned_url(
                PersistentAppUIId=self.persistent_ui_id, PersistentAppUIType=ui_type
            )

            self.presigned_url_ready = response.get("PresignedURLReady", False)
            self.presigned_url = response.get("PresignedURL")

            parsed_url = urlparse(self.presigned_url)
            self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/shs"
            self.api_base = f"{self.base_url}/api/v1"

            logger.info(f"Presigned URL obtained - Base URL: {self.base_url}")

            return self.presigned_url

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"Failed to get presigned URL: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting presigned URL: {str(e)}")
            raise

    def setup_http_session(self) -> requests.Session:
        """
        Set up HTTP session with proper headers and cookie management.

        Returns:
            Configured requests.Session object

        Raises:
            ValueError: If no presigned URL is available
        """
        if not self.presigned_url:
            raise ValueError("No presigned URL available. Get one first.")

        logger.debug("Setting up HTTP session with cookie management")

        self.session.headers.update(
            {
                "User-Agent": "EMR-Spark-History-MCP/1.0",
                "Accept": "application/json,text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

        try:
            logger.debug("Making initial request to establish session")
            response = self.session.get(
                self.presigned_url, timeout=self.timeout, allow_redirects=True
            )
            response.raise_for_status()

            logger.info(
                f"HTTP session established - {len(self.session.cookies)} cookie(s)"
            )

            return self.session

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to establish HTTP session: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error setting up HTTP session: {str(e)}")
            raise

    def initialize(self) -> Tuple[str, requests.Session]:
        """
        Initialize the EMR Persistent UI client.

        Creates a persistent app UI, verifies its status, gets a presigned URL,
        and sets up an HTTP session. Waits for ATTACHED status if STARTING.

        Returns:
            Tuple containing the base URL and configured session

        Raises:
            ValueError: If the persistent UI status is not ATTACHED after waiting
        """
        self.create_persistent_app_ui()

        max_wait_time = 180
        wait_interval = 10
        total_waited = 0
        ui_status = ""

        while total_waited < max_wait_time:
            describe_response = self.describe_persistent_app_ui()
            ui_status = describe_response.get("PersistentAppUI", {}).get(
                "PersistentAppUIStatus"
            )

            if ui_status == "ATTACHED":
                break
            elif ui_status == "STARTING":
                logger.info(
                    f"EMR Persistent UI status is {ui_status}, waiting for ATTACHED..."
                )
                time.sleep(wait_interval)
                total_waited += wait_interval
            else:
                raise ValueError(
                    f"EMR Persistent UI status is {ui_status}, expected ATTACHED or STARTING"
                )

        if ui_status != "ATTACHED":
            raise ValueError(
                f"EMR Persistent UI status is still {ui_status} after {total_waited}s"
            )

        self.get_presigned_url()
        self.setup_http_session()

        return self.base_url, self.session

    def _resolve_app_path(self, app_id: str, attempt_id: str = None) -> str:
        """
        Build the API path segment for an application, including attempt ID.

        The Spark History Server REST API requires attempt ID for multi-attempt apps:
          /api/v1/applications/{appId}/{attemptId}/...

        Args:
            app_id: Spark application ID
            attempt_id: Explicit attempt ID. If None, auto-resolves the latest.

        Returns:
            Path segment like "applications/{appId}/{attemptId}"
        """
        if attempt_id:
            return f"applications/{app_id}/{attempt_id}"

        # Auto-resolve: fetch app metadata and use the latest attempt
        if app_id not in self._attempt_cache:
            try:
                app_info = self._make_api_request(f"applications/{app_id}")
                attempts = app_info.get("attempts", [])
                if attempts:
                    # Use the most recent attempt (first in list, sorted by recency)
                    self._attempt_cache[app_id] = attempts[0].get("attemptId")
                else:
                    self._attempt_cache[app_id] = None
            except Exception:
                self._attempt_cache[app_id] = None

        cached_attempt = self._attempt_cache[app_id]
        if cached_attempt:
            return f"applications/{app_id}/{cached_attempt}"
        return f"applications/{app_id}"

    def _make_api_request(self, endpoint: str) -> Any:
        """Make an API request to the Spark History Server."""
        if not self.api_base:
            raise ValueError("Client not initialized. Call initialize() first.")

        url = f"{self.api_base}/{endpoint}"
        response = self.session.get(url, allow_redirects=True, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_applications(self) -> List[Dict]:
        """
        Get all Spark applications from the history server.

        Returns:
            List of application dictionaries
        """
        return self._make_api_request("applications")

    def get_application(self, app_id: str, attempt_id: str = None) -> Dict:
        """
        Get details for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            Application details dictionary
        """
        return self._make_api_request(self._resolve_app_path(app_id, attempt_id))

    def get_jobs(self, app_id: str, attempt_id: str = None) -> List[Dict]:
        """
        Get all jobs for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            List of job dictionaries
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/jobs"
        )

    def get_job(self, app_id: str, job_id: int, attempt_id: str = None) -> Dict:
        """
        Get details for a specific job.

        Args:
            app_id: Spark application ID
            job_id: Job ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            Job details dictionary
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/jobs/{job_id}"
        )

    def get_stages(self, app_id: str, attempt_id: str = None) -> List[Dict]:
        """
        Get all stages for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            List of stage dictionaries
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/stages"
        )

    def get_stage(
        self, app_id: str, stage_id: int, stage_attempt_id: int = 0,
        app_attempt_id: str = None
    ) -> Dict:
        """
        Get details for a specific stage attempt.

        Args:
            app_id: Spark application ID
            stage_id: Stage ID
            stage_attempt_id: Stage attempt ID (default 0)
            app_attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            Stage details dictionary
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, app_attempt_id)}/stages/{stage_id}/{stage_attempt_id}"
        )

    def get_stage_task_summary(
        self, app_id: str, stage_id: int, stage_attempt_id: int = 0,
        app_attempt_id: str = None
    ) -> Dict:
        """
        Get task summary metrics for a specific stage.

        Args:
            app_id: Spark application ID
            stage_id: Stage ID
            stage_attempt_id: Stage attempt ID (default 0)
            app_attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            Task summary metrics dictionary
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, app_attempt_id)}/stages/{stage_id}/{stage_attempt_id}/taskSummary"
        )

    def get_stage_tasks(
        self, app_id: str, stage_id: int, stage_attempt_id: int = 0,
        app_attempt_id: str = None
    ) -> List[Dict]:
        """
        Get all tasks for a specific stage.

        Args:
            app_id: Spark application ID
            stage_id: Stage ID
            stage_attempt_id: Stage attempt ID (default 0)
            app_attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            List of task dictionaries
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, app_attempt_id)}/stages/{stage_id}/{stage_attempt_id}/taskList"
        )

    def get_executors(self, app_id: str, attempt_id: str = None) -> List[Dict]:
        """
        Get all executors for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            List of executor dictionaries
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/executors"
        )

    def get_all_executors(self, app_id: str, attempt_id: str = None) -> List[Dict]:
        """
        Get all executors (including dead) for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            List of executor dictionaries
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/allexecutors"
        )

    def get_sql_queries(self, app_id: str, attempt_id: str = None) -> List[Dict]:
        """
        Get all SQL queries for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            List of SQL query dictionaries
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/sql"
        )

    def get_sql_query(self, app_id: str, execution_id: int, attempt_id: str = None) -> Dict:
        """
        Get details for a specific SQL query execution.

        Args:
            app_id: Spark application ID
            execution_id: SQL execution ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            SQL query details dictionary
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/sql/{execution_id}"
        )

    def get_environment(self, app_id: str, attempt_id: str = None) -> Dict:
        """
        Get Spark environment/configuration for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            Environment configuration dictionary
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/environment"
        )

    def get_storage_rdd(self, app_id: str, attempt_id: str = None) -> List[Dict]:
        """
        Get RDD storage information for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            List of RDD storage dictionaries
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/storage/rdd"
        )

    def get_streaming_statistics(self, app_id: str, attempt_id: str = None) -> Dict:
        """
        Get streaming statistics for a specific application.

        Args:
            app_id: Spark application ID
            attempt_id: Application attempt ID. If None, auto-resolves the latest.

        Returns:
            Streaming statistics dictionary
        """
        return self._make_api_request(
            f"{self._resolve_app_path(app_id, attempt_id)}/streaming/statistics"
        )
