#!/usr/bin/env python3
"""
Spark SQL Node Metrics MCP Server

Exposes Spark SQL node metrics extraction as MCP tools for performance tuning.
"""

import json
import time
import base64
import requests
from typing import Any

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("spark-sql-metrics")


def safe_b64_decode(val: str | None) -> str | None:
    """Safely decode a base64-encoded value, or return as-is if already decoded."""
    if not val:
        return val
    try:
        if not val.startswith("http"):
            return base64.b64decode(val).decode('utf-8')
        return val
    except Exception:
        return val


@mcp.tool()
def get_sql_node_metrics(cluster_id: str) -> str:
    """
    Extract SQL node metrics from a Databricks Spark cluster.
    
    This tool retrieves detailed SQL execution metrics from the Spark History Server,
    which can be used to analyze query performance and identify optimization opportunities.
    
    Args:
        cluster_id: The Databricks cluster ID to extract metrics from (e.g., "0311-123456-abc123")
    
    Returns:
        JSON string containing SQL execution data including:
        - Query execution plans
        - Node-level metrics (rows, bytes, time)
        - Shuffle statistics
        - Stage breakdown
        
    Use this data to identify:
        - Data skew issues
        - Expensive operations (sorts, joins)
        - Spill to disk problems
        - Shuffle bottlenecks
    """
    try:
        from databricks.sdk import WorkspaceClient
        
        # Initialize Databricks client
        w = WorkspaceClient()

        # Resolve data plane URL from secrets
        try:
            raw_dp_url_val = w.secrets.get_secret(scope="shscreds", key="dpurl").value
            dp_url = safe_b64_decode(raw_dp_url_val)
            if dp_url:
                dp_url = dp_url.strip().rstrip('/')
            else:
                return json.dumps({"error": "Data plane URL is empty"})
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/dpurl - {str(e)}"})

        # Get auth cookies from secrets
        try:
            raw_cookie = w.secrets.get_secret(scope="shscreds", key="cookies").value
            cookies = {'DATAPLANE_DOMAIN_DBAUTH': safe_b64_decode(raw_cookie)}
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/cookies - {str(e)}"})

        # Get cluster info and build Spark UI URL
        c_info = w.clusters.get(cluster_id=cluster_id)
        ctx_id = str(c_info.spark_context_id)
        if not ctx_id.startswith("driver-"):
            ctx_id = f"driver-{ctx_id}"

        api_base = f"{dp_url}/sparkui/{cluster_id}/{ctx_id}/api/v1/applications"

        # Fetch application data (with retry for loading state)
        max_retries = 10
        retry_count = 0
        apps_raw_txt = requests.get(api_base, cookies=cookies, timeout=30).text
        
        while "Loading historical Spark UI" in apps_raw_txt and retry_count < max_retries:
            time.sleep(30)
            apps_raw_txt = requests.get(api_base, cookies=cookies, timeout=30).text
            retry_count += 1

        if "Loading historical Spark UI" in apps_raw_txt:
            return json.dumps({"error": "Spark UI still loading after max retries"})

        apps_raw_json = json.loads(apps_raw_txt)
        if not apps_raw_json:
            return json.dumps({"error": "No applications found"})
            
        app_id = apps_raw_json[0]["id"]

        # Fetch SQL metrics
        sql_url = f"{api_base}/{app_id}/sql"
        raw_sql_data = requests.get(sql_url, cookies=cookies, timeout=60).text

        return raw_sql_data

    except Exception as e:
        return json.dumps({"error": f"Failed to extract SQL metrics: {str(e)}"})


@mcp.tool()
def get_sql_execution_details(cluster_id: str, execution_id: int) -> str:
    """
    Get detailed metrics for a specific SQL execution.
    
    Args:
        cluster_id: The Databricks cluster ID
        execution_id: The SQL execution ID to get details for
    
    Returns:
        JSON string containing detailed execution metrics including node-level statistics
    """
    try:
        from databricks.sdk import WorkspaceClient
        
        w = WorkspaceClient()

        # Resolve data plane URL
        try:
            raw_dp_url_val = w.secrets.get_secret(scope="shscreds", key="dpurl").value
            dp_url = safe_b64_decode(raw_dp_url_val)
            if dp_url:
                dp_url = dp_url.strip().rstrip('/')
            else:
                return json.dumps({"error": "Data plane URL is empty"})
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/dpurl - {str(e)}"})

        # Get auth cookies
        try:
            raw_cookie = w.secrets.get_secret(scope="shscreds", key="cookies").value
            cookies = {'DATAPLANE_DOMAIN_DBAUTH': safe_b64_decode(raw_cookie)}
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/cookies - {str(e)}"})

        # Get cluster info
        c_info = w.clusters.get(cluster_id=cluster_id)
        ctx_id = str(c_info.spark_context_id)
        if not ctx_id.startswith("driver-"):
            ctx_id = f"driver-{ctx_id}"

        api_base = f"{dp_url}/sparkui/{cluster_id}/{ctx_id}/api/v1/applications"

        # Get app ID
        apps_raw_txt = requests.get(api_base, cookies=cookies, timeout=30).text
        apps_raw_json = json.loads(apps_raw_txt)
        if not apps_raw_json:
            return json.dumps({"error": "No applications found"})
        app_id = apps_raw_json[0]["id"]

        # Fetch specific SQL execution details
        sql_detail_url = f"{api_base}/{app_id}/sql/{execution_id}"
        raw_detail_data = requests.get(sql_detail_url, cookies=cookies, timeout=60).text

        return raw_detail_data

    except Exception as e:
        return json.dumps({"error": f"Failed to get execution details: {str(e)}"})


@mcp.tool()
def get_stages_metrics(cluster_id: str) -> str:
    """
    Get all stage metrics from a Spark application.
    
    Stage metrics provide insights into task distribution, shuffle sizes, 
    and potential data skew issues.
    
    Args:
        cluster_id: The Databricks cluster ID
    
    Returns:
        JSON string containing stage-level metrics useful for tuning
    """
    try:
        from databricks.sdk import WorkspaceClient
        
        w = WorkspaceClient()

        # Resolve data plane URL
        try:
            raw_dp_url_val = w.secrets.get_secret(scope="shscreds", key="dpurl").value
            dp_url = safe_b64_decode(raw_dp_url_val)
            if dp_url:
                dp_url = dp_url.strip().rstrip('/')
            else:
                return json.dumps({"error": "Data plane URL is empty"})
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/dpurl - {str(e)}"})

        # Get auth cookies
        try:
            raw_cookie = w.secrets.get_secret(scope="shscreds", key="cookies").value
            cookies = {'DATAPLANE_DOMAIN_DBAUTH': safe_b64_decode(raw_cookie)}
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/cookies - {str(e)}"})

        # Get cluster info
        c_info = w.clusters.get(cluster_id=cluster_id)
        ctx_id = str(c_info.spark_context_id)
        if not ctx_id.startswith("driver-"):
            ctx_id = f"driver-{ctx_id}"

        api_base = f"{dp_url}/sparkui/{cluster_id}/{ctx_id}/api/v1/applications"

        # Get app ID
        apps_raw_txt = requests.get(api_base, cookies=cookies, timeout=30).text
        apps_raw_json = json.loads(apps_raw_txt)
        if not apps_raw_json:
            return json.dumps({"error": "No applications found"})
        app_id = apps_raw_json[0]["id"]

        # Fetch stages with details
        stages_url = f"{api_base}/{app_id}/stages?details=true"
        raw_stages_data = requests.get(stages_url, cookies=cookies, timeout=60).text

        return raw_stages_data

    except Exception as e:
        return json.dumps({"error": f"Failed to get stages metrics: {str(e)}"})


@mcp.tool()
def get_jobs_metrics(cluster_id: str) -> str:
    """
    Get all job metrics from a Spark application.
    
    Job metrics show the overall execution flow and help identify 
    long-running or failed jobs.
    
    Args:
        cluster_id: The Databricks cluster ID
    
    Returns:
        JSON string containing job-level metrics
    """
    try:
        from databricks.sdk import WorkspaceClient
        
        w = WorkspaceClient()

        # Resolve data plane URL
        try:
            raw_dp_url_val = w.secrets.get_secret(scope="shscreds", key="dpurl").value
            dp_url = safe_b64_decode(raw_dp_url_val)
            if dp_url:
                dp_url = dp_url.strip().rstrip('/')
            else:
                return json.dumps({"error": "Data plane URL is empty"})
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/dpurl - {str(e)}"})

        # Get auth cookies
        try:
            raw_cookie = w.secrets.get_secret(scope="shscreds", key="cookies").value
            cookies = {'DATAPLANE_DOMAIN_DBAUTH': safe_b64_decode(raw_cookie)}
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/cookies - {str(e)}"})

        # Get cluster info
        c_info = w.clusters.get(cluster_id=cluster_id)
        ctx_id = str(c_info.spark_context_id)
        if not ctx_id.startswith("driver-"):
            ctx_id = f"driver-{ctx_id}"

        api_base = f"{dp_url}/sparkui/{cluster_id}/{ctx_id}/api/v1/applications"

        # Get app ID
        apps_raw_txt = requests.get(api_base, cookies=cookies, timeout=30).text
        apps_raw_json = json.loads(apps_raw_txt)
        if not apps_raw_json:
            return json.dumps({"error": "No applications found"})
        app_id = apps_raw_json[0]["id"]

        # Fetch jobs
        jobs_url = f"{api_base}/{app_id}/jobs"
        raw_jobs_data = requests.get(jobs_url, cookies=cookies, timeout=60).text

        return raw_jobs_data

    except Exception as e:
        return json.dumps({"error": f"Failed to get jobs metrics: {str(e)}"})


@mcp.tool()
def get_environment_info(cluster_id: str) -> str:
    """
    Get Spark environment and configuration information.
    
    This includes Spark configuration, system properties, and runtime information
    useful for understanding the execution context.
    
    Args:
        cluster_id: The Databricks cluster ID
    
    Returns:
        JSON string containing environment configuration
    """
    try:
        from databricks.sdk import WorkspaceClient
        
        w = WorkspaceClient()
        
        # Resolve data plane URL
        try:
            raw_dp_url_val = w.secrets.get_secret(scope="shscreds", key="dpurl").value
            dp_url = safe_b64_decode(raw_dp_url_val)
            if dp_url:
                dp_url = dp_url.strip().rstrip('/')
            else:
                return json.dumps({"error": "Data plane URL is empty"})
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/dpurl - {str(e)}"})

        # Get auth cookies
        try:
            raw_cookie = w.secrets.get_secret(scope="shscreds", key="cookies").value
            cookies = {'DATAPLANE_DOMAIN_DBAUTH': safe_b64_decode(raw_cookie)}
        except Exception as e:
            return json.dumps({"error": f"Missing secret: shscreds/cookies - {str(e)}"})

        # Get cluster info
        c_info = w.clusters.get(cluster_id=cluster_id)
        ctx_id = str(c_info.spark_context_id)
        if not ctx_id.startswith("driver-"):
            ctx_id = f"driver-{ctx_id}"

        api_base = f"{dp_url}/sparkui/{cluster_id}/{ctx_id}/api/v1/applications"

        # Get app ID
        apps_raw_txt = requests.get(api_base, cookies=cookies, timeout=30).text
        apps_raw_json = json.loads(apps_raw_txt)
        if not apps_raw_json:
            return json.dumps({"error": "No applications found"})
        app_id = apps_raw_json[0]["id"]

        # Fetch environment info
        env_url = f"{api_base}/{app_id}/environment"
        raw_env_data = requests.get(env_url, cookies=cookies, timeout=60).text

        return raw_env_data

    except Exception as e:
        return json.dumps({"error": f"Failed to get environment info: {str(e)}"})


if __name__ == "__main__":
    mcp.run()
