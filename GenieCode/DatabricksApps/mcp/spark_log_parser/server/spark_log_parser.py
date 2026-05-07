#!/usr/bin/env python3
"""
Spark Event Log Parser

Parses raw Spark event logs (JSON Lines format) and extracts:
- Job metrics
- Stage metrics
- SQL execution metrics

Output schemas match the specified Spark DataFrame schemas.

RSTODO credential name as arg, parse JSON get_s3_client
"""

import json
import boto3
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from databricks.sdk.core import Config
from flask import request
import requests
from requests.auth import HTTPBasicAuth
from databricks.sdk import WorkspaceClient


_s3_client = None

def get_s3_client(cred_name: str):
    """Get or create S3 client from config.json."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    
    W = WorkspaceClient()
    cred_name = cred_name
    tmpcreds = W.credentials.generate_temporary_service_credential(credential_name = cred_name)
    
    _s3_client = boto3.client(
        's3',
        aws_access_key_id=tmpcreds.aws_temp_credentials.access_key_id,
        aws_secret_access_key=tmpcreds.aws_temp_credentials.secret_access_key,
        aws_session_token=tmpcreds.aws_temp_credentials.session_token
    )
    return _s3_client


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse an S3 path into bucket name and prefix."""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    path_without_scheme = s3_path[5:]
    slash_idx = path_without_scheme.find('/')
    if slash_idx == -1:
        return path_without_scheme, ""
    
    bucket_name = path_without_scheme[:slash_idx]
    prefix = path_without_scheme[slash_idx + 1:]
    return bucket_name, prefix


@dataclass
class JobMetric:
    jobId: str
    name: str = ""
    description: str = ""
    submissionTime: str = ""
    completionTime: str = ""
    stageIds: list[str] = field(default_factory=list)
    jobGroup: str = ""
    jobTags: list[str] = field(default_factory=list)
    status: str = ""
    numTasks: float = 0.0
    numActiveTasks: float = 0.0
    numCompletedTasks: float = 0.0
    numSkippedTasks: float = 0.0
    numFailedTasks: float = 0.0
    numKilledTasks: float = 0.0
    numActiveStages: float = 0.0
    numCompletedStages: float = 0.0
    numSkippedStages: float = 0.0
    numFailedStages: float = 0.0


@dataclass
class StageMetric:
    stageId: str
    attemptId: str = "0"
    name: str = ""
    description: str = ""
    submissionTime: str = ""
    completionTime: str = ""
    status: str = ""
    numTasks: float = 0.0
    numCompletedTasks: float = 0.0
    numSkippedTasks: float = 0.0
    numFailedTasks: float = 0.0
    numCompletedStages: float = 0.0
    numSkippedStages: float = 0.0
    numFailedStages: float = 0.0
    memoryBytesSpilled: int = 0
    diskBytesSpilled: int = 0
    inputBytes: int = 0
    inputRecords: int = 0
    outputBytes: int = 0
    outputRecords: int = 0
    shuffleReadBytes: int = 0
    shuffleReadRecords: int = 0
    shuffleWriteBytes: int = 0
    shuffleWriteRecords: int = 0


@dataclass
class NodeMetric:
    name: str
    value: str


@dataclass
class MetricDefinition:
    name: str
    accumulatorId: int
    metricType: str


@dataclass
class PlanNode:
    nodeId: int
    nodeName: str
    metrics: list[NodeMetric] = field(default_factory=list)
    metricDefinitions: list[MetricDefinition] = field(default_factory=list)


@dataclass
class SqlMetric:
    id: int
    status: str = ""
    description: str = ""
    planDescription: str = ""
    submissionTime: str = ""
    duration: int = 0
    successJobIds: list[str] = field(default_factory=list)
    failedJobIds: str = ""
    nodes: list[PlanNode] = field(default_factory=list)
    accumulatorValues: dict = field(default_factory=dict)
    _submissionTimeMs: int = 0


def timestamp_to_iso(timestamp_ms: int) -> str:
    """Convert millisecond timestamp to ISO format string."""
    if not timestamp_ms:
        return ""
    return datetime.fromtimestamp(timestamp_ms / 1000).isoformat()


def extract_accumulator_value(
    accumulables: list[dict], metric_name: str
) -> int:
    """Extract a specific metric value from accumulables list."""
    for acc in accumulables:
        if acc.get("Name", "").lower() == metric_name.lower():
            val = acc.get("Value", 0)
            if isinstance(val, str):
                try:
                    return int(val)
                except ValueError:
                    return 0
            return int(val) if val else 0
    return 0


def parse_stage_metrics_from_accumulables(
    accumulables: list[dict],
) -> dict:
    """Extract stage-level metrics from accumulables."""
    metric_mappings = {
        "internal.metrics.memoryBytesSpilled": "memoryBytesSpilled",
        "internal.metrics.diskBytesSpilled": "diskBytesSpilled",
        "internal.metrics.input.bytesRead": "inputBytes",
        "internal.metrics.input.recordsRead": "inputRecords",
        "internal.metrics.output.bytesWritten": "outputBytes",
        "internal.metrics.output.recordsWritten": "outputRecords",
        "internal.metrics.shuffle.read.localBytesRead": "shuffleReadBytesLocal",
        "internal.metrics.shuffle.read.remoteBytesRead": "shuffleReadBytesRemote",
        "internal.metrics.shuffle.read.recordsRead": "shuffleReadRecords",
        "internal.metrics.shuffle.write.bytesWritten": "shuffleWriteBytes",
        "internal.metrics.shuffle.write.recordsWritten": "shuffleWriteRecords",
    }

    metrics = {
        "memoryBytesSpilled": 0,
        "diskBytesSpilled": 0,
        "inputBytes": 0,
        "inputRecords": 0,
        "outputBytes": 0,
        "outputRecords": 0,
        "shuffleReadBytes": 0,
        "shuffleReadRecords": 0,
        "shuffleWriteBytes": 0,
        "shuffleWriteRecords": 0,
    }

    shuffle_read_local = 0
    shuffle_read_remote = 0

    for acc in accumulables:
        name = acc.get("Name", "")
        val = acc.get("Value", 0)
        if isinstance(val, str):
            try:
                val = int(val)
            except ValueError:
                val = 0

        if name in metric_mappings:
            target_key = metric_mappings[name]
            if target_key == "shuffleReadBytesLocal":
                shuffle_read_local = val
            elif target_key == "shuffleReadBytesRemote":
                shuffle_read_remote = val
            elif target_key in metrics:
                metrics[target_key] = val

    metrics["shuffleReadBytes"] = shuffle_read_local + shuffle_read_remote

    return metrics


def parse_sql_plan_nodes(plan_description: str) -> list[PlanNode]:
    """
    Parse plan description to extract node information.
    This is a simplified parser that extracts node IDs and names from the plan.
    Returns unique nodes by ID (first occurrence wins).
    """
    import re

    nodes = []
    seen_ids: set[int] = set()
    if not plan_description:
        return nodes

    node_pattern = re.compile(r"^(.+?)\s*\((\d+)\)")

    lines = plan_description.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("=="):
            continue

        cleaned = line.lstrip("+- *:| ")

        match = node_pattern.match(cleaned)
        if match:
            node_name = match.group(1).strip()
            node_id_str = match.group(2)
            if node_id_str.isdigit() and node_name:
                node_id = int(node_id_str)
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    node = PlanNode(
                        nodeId=node_id, nodeName=node_name, metrics=[]
                    )
                    nodes.append(node)

    return nodes


def parse_spark_plan_info(
    plan_info: dict,
    nodes: list[PlanNode] = None,
    seen_ids: set[int] = None,
) -> list[PlanNode]:
    """
    Recursively parse sparkPlanInfo tree to extract nodes with metric definitions.
    Returns unique nodes by ID (first occurrence wins).
    """
    if nodes is None:
        nodes = []
    if seen_ids is None:
        seen_ids = set()

    if not plan_info:
        return nodes

    node_name = plan_info.get("nodeName", "")
    explain_id = plan_info.get("explainId")
    metrics_raw = plan_info.get("metrics", [])

    if explain_id is not None and node_name:
        node_id = int(explain_id)
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            metric_defs = []
            for m in metrics_raw:
                if isinstance(m, dict) and "name" in m and "accumulatorId" in m:
                    metric_defs.append(
                        MetricDefinition(
                            name=m.get("name", ""),
                            accumulatorId=m.get("accumulatorId", 0),
                            metricType=m.get("metricType", ""),
                        )
                    )

            node = PlanNode(
                nodeId=node_id,
                nodeName=node_name,
                metrics=[],
                metricDefinitions=metric_defs,
            )
            nodes.append(node)

    for child in plan_info.get("children", []):
        parse_spark_plan_info(child, nodes, seen_ids)

    return nodes


class SparkLogParser:
    """Parser for Spark event logs."""

    def __init__(self):
        self.jobs: dict[str, JobMetric] = {}
        self.stages: dict[str, StageMetric] = {}
        self.sql_executions: dict[int, SqlMetric] = {}
        self.job_sql_mapping: dict[str, int] = {}
        self.stage_task_counts: dict[str, dict] = {}
        self.sql_job_associations: dict[int, set] = {}
        self.global_accum_values: dict[int, int] = {}
        self.global_accum_by_name: dict[str, int] = {}
        self.sql_accum_by_name: dict[int, dict[str, int]] = {}
        self.job_stage_submissions: dict[str, set] = {}
        self.completed_stages: set = set()
        self.submitted_stages: set = set()
        self.stage_first_job: dict[str, str] = {}
        self.skipped_stages_counted: set = set()
        self.stage_referencing_jobs: dict[str, list[str]] = {}
        self.stage_skip_claimed_by: dict[str, list[str]] = {}
        self._job_stage_tasks: dict[str, dict[str, int]] = {}
        self._stage_first_completion: dict[str, int] = {}
        self._job_start_times: dict[str, int] = {}
        self._completed_job_statuses: dict[str, str] = {}

    def parse_file(self, obj_name: str, bucket_name: str, s3_client=None) -> None:
        """Parse a single event log file."""
        if s3_client is None:
            s3_client = get_s3_client()
        
        response = s3_client.get_object(Bucket=bucket_name, Key=obj_name)
        object_content = response['Body'].read().decode('utf-8')

        line_num = 0
        for line in object_content.splitlines():
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                self._process_event(event)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")

    def parse_directory(self, directory: str, bucket_name: str, prefix: str, s3_client=None) -> None:
        """Parse all JSON files in a directory."""
        if s3_client is None:
            s3_client = get_s3_client()
        
        objects_list = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents")
        if not objects_list:
            raise FileNotFoundError(f"No objects found in s3://{bucket_name}/{prefix}")
        for obj in objects_list:
            obj_name = obj["Key"]
            print(f"Parsing: {obj_name}")
            self.parse_file(str(obj_name), bucket_name, s3_client)
        self._finalize_sql_job_associations()    
    


    def _finalize_sql_job_associations(self) -> None:
        """Finalize SQL-job associations after all events are processed.
        
        This handles cases where events appear out-of-order in the log file
        (e.g., job ends before SQL start event is written to log).
        """
        for exec_id, job_ids in self.sql_job_associations.items():
            sql = self.sql_executions.get(exec_id)
            if sql:
                for job_id in job_ids:
                    status = self._completed_job_statuses.get(job_id)
                    if status == "JobSucceeded":
                        if job_id not in sql.successJobIds:
                            sql.successJobIds.append(job_id)
                    elif status:
                        if sql.failedJobIds:
                            if job_id not in sql.failedJobIds:
                                sql.failedJobIds += f",{job_id}"
                        else:
                            sql.failedJobIds = job_id

    def _process_event(self, event: dict) -> None:
        """Process a single Spark event."""
        event_type = event.get("Event", "")

        if event_type == "SparkListenerJobStart":
            self._handle_job_start(event)
        elif event_type == "SparkListenerJobEnd":
            self._handle_job_end(event)
        elif event_type == "SparkListenerStageSubmitted":
            self._handle_stage_submitted(event)
        elif event_type == "SparkListenerStageCompleted":
            self._handle_stage_completed(event)
        elif event_type == "SparkListenerTaskEnd":
            self._handle_task_end(event)
        elif (
            event_type
            == "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart"
        ):
            self._handle_sql_start(event)
        elif (
            event_type
            == "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd"
        ):
            self._handle_sql_end(event)
        elif (
            event_type
            == "org.apache.spark.sql.execution.ui.SparkListenerSQLAdaptiveExecutionUpdate"
        ):
            self._handle_sql_adaptive_update(event)
        elif (
            event_type
            == "org.apache.spark.sql.execution.ui.SparkListenerDriverAccumUpdates"
        ):
            self._handle_driver_accum_updates(event)
        elif (
            event_type
            == "org.apache.spark.sql.execution.ui.SparkListenerSQLAdaptiveSQLMetricUpdates"
        ):
            self._handle_sql_metric_updates(event)

    def _handle_job_start(self, event: dict) -> None:
        """Handle SparkListenerJobStart event."""
        job_id = str(event.get("Job ID", ""))
        submission_time_ms = event.get("Submission Time", 0)
        submission_time = timestamp_to_iso(submission_time_ms)

        self._job_start_times[job_id] = submission_time_ms

        stage_infos = event.get("Stage Infos", [])
        stage_ids = [str(s.get("Stage ID", "")) for s in stage_infos]

        job_name = ""
        if stage_infos:
            job_name = stage_infos[0].get("Stage Name", "")

        total_tasks = sum(s.get("Number of Tasks", 0) for s in stage_infos)

        properties = event.get("Properties", {})
        description = properties.get("spark.job.description", "")
        job_group = properties.get("spark.jobGroup.id", "")
        job_tags_str = properties.get("spark.job.tags", "")
        job_tags = [t.strip() for t in job_tags_str.split(",") if t.strip()] if job_tags_str else []

        sql_exec_id_str = properties.get("spark.sql.execution.id")
        if sql_exec_id_str is not None:
            try:
                sql_exec_id = int(sql_exec_id_str)
                if sql_exec_id not in self.sql_job_associations:
                    self.sql_job_associations[sql_exec_id] = set()
                self.sql_job_associations[sql_exec_id].add(job_id)
            except (ValueError, TypeError):
                pass

        self.job_stage_submissions[job_id] = set()

        self._job_stage_tasks[job_id] = {}
        for stage_info in event.get("Stage Infos", []):
            sid = str(stage_info.get("Stage ID", ""))
            num_tasks = stage_info.get("Number of Tasks", 0)
            self._job_stage_tasks[job_id][sid] = num_tasks

        for stage_id in stage_ids:
            if stage_id not in self.stage_first_job:
                self.stage_first_job[stage_id] = job_id
            if stage_id not in self.stage_referencing_jobs:
                self.stage_referencing_jobs[stage_id] = []
            self.stage_referencing_jobs[stage_id].append(job_id)

        self.jobs[job_id] = JobMetric(
            jobId=job_id,
            name=job_name,
            description=description,
            submissionTime=submission_time,
            stageIds=stage_ids,
            jobGroup=job_group,
            jobTags=job_tags,
            numTasks=float(total_tasks),
        )

    def _should_count_stage_as_skipped(
        self, stage_id: str, job_id: str, job_completion_time_ms: int = 0
    ) -> bool:
        """Determine if a stage should be counted as skipped for a job.
        
        SHS Logic:
        - If stage was submitted (ran), not skipped
        - If only one job references the stage, count as skipped
        - If multiple jobs reference the stage:
          - If there are still OTHER incomplete jobs → skipped for this job
          - If this is the LAST job to complete → NOT skipped
        """
        if stage_id in self.submitted_stages:
            return False

        referencing_jobs = self.stage_referencing_jobs.get(stage_id, [])
        
        if len(referencing_jobs) <= 1:
            return True

        has_incomplete = False
        for other_job_id in referencing_jobs:
            if other_job_id != job_id:
                other_job = self.jobs.get(other_job_id)
                if other_job and not other_job.completionTime:
                    has_incomplete = True
                    break

        if has_incomplete:
            return True
        else:
            return False

    def _handle_job_end(self, event: dict) -> None:
        """Handle SparkListenerJobEnd event."""
        job_id = str(event.get("Job ID", ""))
        completion_time = timestamp_to_iso(event.get("Completion Time", 0))
        result = event.get("Job Result", {})
        status = result.get("Result", "UNKNOWN")

        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.completionTime = completion_time
            job.status = status

            completion_time_ms = event.get("Completion Time", 0)
            for stage_id in job.stageIds:
                if stage_id not in self._stage_first_completion:
                    self._stage_first_completion[stage_id] = completion_time_ms

            completed_stages = 0
            failed_stages = 0
            skipped_stages = 0
            completed_tasks = 0
            failed_tasks = 0
            skipped_stage_tasks = 0

            stages_submitted_for_job = self.job_stage_submissions.get(job_id, set())
            stage_task_counts = getattr(self, '_job_stage_tasks', {}).get(job_id, {})

            for stage_id in job.stageIds:
                stage_found = False

                for key in self.stages:
                    if key.startswith(f"{stage_id}_"):
                        stage = self.stages[key]
                        stage_found = True

                        if stage_id in stages_submitted_for_job:
                            if stage.status == "COMPLETE":
                                completed_stages += 1
                                completed_tasks += stage.numCompletedTasks
                                failed_tasks += stage.numFailedTasks
                            elif stage.status == "FAILED":
                                failed_stages += 1
                                completed_tasks += stage.numCompletedTasks
                                failed_tasks += stage.numFailedTasks
                        else:
                            if self._should_count_stage_as_skipped(
                                stage_id, job_id, completion_time_ms
                            ):
                                skipped_stages += 1
                                skipped_stage_tasks += stage.numTasks
                                if stage_id not in self.stage_skip_claimed_by:
                                    self.stage_skip_claimed_by[stage_id] = []
                                self.stage_skip_claimed_by[stage_id].append(job_id)
                        break

                if not stage_found:
                    if stage_id not in self.submitted_stages:
                        if self._should_count_stage_as_skipped(
                            stage_id, job_id, completion_time_ms
                        ):
                            skipped_stages += 1
                            skipped_stage_tasks += stage_task_counts.get(stage_id, 0)
                            if stage_id not in self.stage_skip_claimed_by:
                                self.stage_skip_claimed_by[stage_id] = []
                            self.stage_skip_claimed_by[stage_id].append(job_id)

            skipped_tasks = skipped_stage_tasks

            job.numCompletedStages = float(completed_stages)
            job.numFailedStages = float(failed_stages)
            job.numSkippedStages = float(skipped_stages)
            job.numCompletedTasks = float(completed_tasks)
            job.numFailedTasks = float(failed_tasks)
            job.numSkippedTasks = float(max(0, skipped_tasks))

            self._completed_job_statuses[job_id] = status

    def _handle_stage_submitted(self, event: dict) -> None:
        """Handle SparkListenerStageSubmitted event."""
        stage_info = event.get("Stage Info", {})
        stage_id = str(stage_info.get("Stage ID", ""))
        attempt_id = str(stage_info.get("Stage Attempt ID", "0"))
        stage_key = f"{stage_id}_{attempt_id}"

        submission_time = timestamp_to_iso(stage_info.get("Submission Time", 0))
        stage_name = stage_info.get("Stage Name", "")
        num_tasks = stage_info.get("Number of Tasks", 0)

        self.submitted_stages.add(stage_id)

        for job_id, job in self.jobs.items():
            if stage_id in job.stageIds and not job.completionTime:
                if job_id not in self.job_stage_submissions:
                    self.job_stage_submissions[job_id] = set()
                self.job_stage_submissions[job_id].add(stage_id)

        if stage_key not in self.stages:
            self.stages[stage_key] = StageMetric(
                stageId=stage_id,
                attemptId=attempt_id,
                name=stage_name,
                submissionTime=submission_time,
                numTasks=float(num_tasks),
            )
        else:
            stage = self.stages[stage_key]
            if not stage.submissionTime:
                stage.submissionTime = submission_time
            if not stage.name:
                stage.name = stage_name

    def _handle_stage_completed(self, event: dict) -> None:
        """Handle SparkListenerStageCompleted event."""
        stage_info = event.get("Stage Info", {})
        stage_id = str(stage_info.get("Stage ID", ""))
        attempt_id = str(stage_info.get("Stage Attempt ID", "0"))
        stage_key = f"{stage_id}_{attempt_id}"

        completion_time = timestamp_to_iso(stage_info.get("Completion Time", 0))
        submission_time = timestamp_to_iso(stage_info.get("Submission Time", 0))
        stage_name = stage_info.get("Stage Name", "")
        num_tasks = stage_info.get("Number of Tasks", 0)
        failure_reason = stage_info.get("Failure Reason", "")

        status = "FAILED" if failure_reason else "COMPLETE"

        self.completed_stages.add(stage_id)

        accumulables = stage_info.get("Accumulables", [])
        metrics = parse_stage_metrics_from_accumulables(accumulables)

        if stage_key not in self.stages:
            self.stages[stage_key] = StageMetric(
                stageId=stage_id,
                attemptId=attempt_id,
                name=stage_name,
                submissionTime=submission_time,
                completionTime=completion_time,
                status=status,
                numTasks=float(num_tasks),
                **metrics,
            )
        else:
            stage = self.stages[stage_key]
            stage.completionTime = completion_time
            stage.status = status
            if not stage.submissionTime:
                stage.submissionTime = submission_time
            if not stage.name:
                stage.name = stage_name

            for key, val in metrics.items():
                setattr(stage, key, val)

        task_counts = self.stage_task_counts.get(stage_key, {})
        stage = self.stages[stage_key]
        stage.numCompletedTasks = float(task_counts.get("completed", 0))
        stage.numFailedTasks = float(task_counts.get("failed", 0))

    def _handle_task_end(self, event: dict) -> None:
        """Handle SparkListenerTaskEnd event to track task counts."""
        stage_id = str(event.get("Stage ID", ""))
        attempt_id = str(event.get("Stage Attempt ID", "0"))
        stage_key = f"{stage_id}_{attempt_id}"

        task_info = event.get("Task Info", {})
        task_end_reason = event.get("Task End Reason", {})
        reason = task_end_reason.get("Reason", "")

        if stage_key not in self.stage_task_counts:
            self.stage_task_counts[stage_key] = {
                "completed": 0,
                "failed": 0,
                "killed": 0,
            }

        counts = self.stage_task_counts[stage_key]
        if reason == "Success":
            counts["completed"] += 1
        elif task_info.get("Failed", False):
            counts["failed"] += 1
        elif task_info.get("Killed", False):
            counts["killed"] += 1

        accumulables = task_info.get("Accumulables", [])
        for acc in accumulables:
            if acc.get("Metadata") == "sql":
                accum_id = acc.get("ID")
                name = acc.get("Name", "")
                value = acc.get("Value", 0)
                if accum_id is not None:
                    if isinstance(value, str):
                        try:
                            value = int(value)
                        except ValueError:
                            continue
                    self.global_accum_values[accum_id] = value
                    if name:
                        self.global_accum_by_name[name] = value

    def _handle_sql_start(self, event: dict) -> None:
        """Handle SparkListenerSQLExecutionStart event."""
        exec_id = event.get("executionId", 0)
        description = event.get("description", "")
        submission_time_ms = event.get("time", 0)
        submission_time = timestamp_to_iso(submission_time_ms)

        spark_plan_info = event.get("sparkPlanInfo", {})
        nodes = parse_spark_plan_info(spark_plan_info)

        self.sql_executions[exec_id] = SqlMetric(
            id=exec_id,
            description=description,
            submissionTime=submission_time,
            status="RUNNING",
            nodes=nodes,
            _submissionTimeMs=submission_time_ms,
        )

        if exec_id not in self.sql_job_associations:
            self.sql_job_associations[exec_id] = set()

    def _handle_sql_end(self, event: dict) -> None:
        """Handle SparkListenerSQLExecutionEnd event."""
        exec_id = event.get("executionId", 0)
        end_time = event.get("time", 0)
        error_message = event.get("errorMessage", "")

        status = "FAILED" if error_message else "COMPLETED"

        if exec_id in self.sql_executions:
            sql = self.sql_executions[exec_id]
            sql.status = status

            if sql._submissionTimeMs and end_time:
                sql.duration = end_time - sql._submissionTimeMs

    def _handle_sql_adaptive_update(self, event: dict) -> None:
        """Handle SparkListenerSQLAdaptiveExecutionUpdate event."""
        exec_id = event.get("executionId", 0)
        plan_description = event.get("physicalPlanDescription", "")

        if exec_id in self.sql_executions:
            sql = self.sql_executions[exec_id]
            sql.planDescription = plan_description
            new_nodes = parse_sql_plan_nodes(plan_description)
            if new_nodes:
                sql.nodes = new_nodes

    def _handle_driver_accum_updates(self, event: dict) -> None:
        """Handle SparkListenerDriverAccumUpdates event."""
        exec_id = event.get("executionId", 0)
        accum_updates = event.get("accumUpdates", [])

        if exec_id in self.sql_executions:
            sql = self.sql_executions[exec_id]
            for update in accum_updates:
                if isinstance(update, list) and len(update) >= 2:
                    accum_id = update[0]
                    value = update[1]
                    sql.accumulatorValues[accum_id] = value

    def _handle_sql_metric_updates(self, event: dict) -> None:
        """Handle SparkListenerSQLAdaptiveSQLMetricUpdates event."""
        exec_id = event.get("executionId", 0)
        sql_plan_metrics = event.get("sqlPlanMetrics", [])

        if exec_id not in self.sql_accum_by_name:
            self.sql_accum_by_name[exec_id] = {}

        for metric in sql_plan_metrics:
            if isinstance(metric, dict):
                accum_id = metric.get("accumulatorId")
                name = metric.get("name", "")
                if accum_id is not None and name:
                    self.sql_accum_by_name[exec_id][name] = accum_id

    def get_job_metrics(self) -> list[dict]:
        """Return job metrics as a list of dictionaries."""
        return [
            {
                "jobId": j.jobId,
                "name": j.name,
                "description": j.description,
                "submissionTime": j.submissionTime,
                "completionTime": j.completionTime,
                "stageIds": j.stageIds,
                "jobGroup": j.jobGroup,
                "jobTags": j.jobTags,
                "status": j.status,
                "numTasks": j.numTasks,
                "numActiveTasks": j.numActiveTasks,
                "numCompletedTasks": j.numCompletedTasks,
                "numSkippedTasks": j.numSkippedTasks,
                "numFailedTasks": j.numFailedTasks,
                "numKilledTasks": j.numKilledTasks,
                "numActiveStages": j.numActiveStages,
                "numCompletedStages": j.numCompletedStages,
                "numSkippedStages": j.numSkippedStages,
                "numFailedStages": j.numFailedStages,
            }
            for j in self.jobs.values()
        ]

    def get_stage_metrics(self) -> list[dict]:
        """Return stage metrics as a list of dictionaries."""
        return [
            {
                "stageId": s.stageId,
                "attemptId": s.attemptId,
                "name": s.name,
                "description": s.description,
                "submissionTime": s.submissionTime,
                "completionTime": s.completionTime,
                "status": s.status,
                "numTasks": s.numTasks,
                "numCompletedTasks": s.numCompletedTasks,
                "numSkippedTasks": s.numSkippedTasks,
                "numFailedTasks": s.numFailedTasks,
                "numCompletedStages": s.numCompletedStages,
                "numSkippedStages": s.numSkippedStages,
                "numFailedStages": s.numFailedStages,
                "memoryBytesSpilled": s.memoryBytesSpilled,
                "diskBytesSpilled": s.diskBytesSpilled,
                "inputBytes": s.inputBytes,
                "inputRecords": s.inputRecords,
                "outputBytes": s.outputBytes,
                "outputRecords": s.outputRecords,
                "shuffleReadBytes": s.shuffleReadBytes,
                "shuffleReadRecords": s.shuffleReadRecords,
                "shuffleWriteBytes": s.shuffleWriteBytes,
                "shuffleWriteRecords": s.shuffleWriteRecords,
            }
            for s in self.stages.values()
        ]

    def get_sql_metrics(self) -> list[dict]:
        """Return SQL metrics as a list of dictionaries."""
        result = []
        for s in self.sql_executions.values():
            sql_name_mapping = self.sql_accum_by_name.get(s.id, {})

            nodes_output = []
            for n in s.nodes:
                node_metrics = []
                for metric_def in n.metricDefinitions:
                    accum_id = metric_def.accumulatorId
                    metric_name = metric_def.name
                    value = None

                    if accum_id in s.accumulatorValues:
                        value = s.accumulatorValues[accum_id]
                    elif accum_id in self.global_accum_values:
                        value = self.global_accum_values[accum_id]
                    elif metric_name in sql_name_mapping:
                        mapped_id = sql_name_mapping[metric_name]
                        if mapped_id in s.accumulatorValues:
                            value = s.accumulatorValues[mapped_id]
                        elif mapped_id in self.global_accum_values:
                            value = self.global_accum_values[mapped_id]
                    elif metric_name in self.global_accum_by_name:
                        value = self.global_accum_by_name[metric_name]

                    if value is not None:
                        node_metrics.append(
                            {"name": metric_name, "value": str(value)}
                        )
                for m in n.metrics:
                    node_metrics.append({"name": m.name, "value": m.value})

                nodes_output.append(
                    {
                        "nodeId": n.nodeId,
                        "nodeName": n.nodeName,
                        "metrics": node_metrics,
                    }
                )

            result.append(
                {
                    "id": s.id,
                    "status": s.status,
                    "description": s.description,
                    "planDescription": s.planDescription,
                    "submissionTime": s.submissionTime,
                    "duration": s.duration,
                    "successJobIds": s.successJobIds,
                    "failedJobIds": s.failedJobIds,
                    "nodes": nodes_output,
                }
            )
        return result


def main():
    """Main entry point for the parser."""
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Parse Spark event logs and extract metrics"
    )
    arg_parser.add_argument(
        "input",
        nargs="?",
        default="events",
        help="Input file or directory (default: events)",
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory for JSON files (default: output)",
    )
    arg_parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format (default: json)",
    )

    args = arg_parser.parse_args()

    with open('config.json', 'r') as file:
        configjson = json.load(file)


    ACCESS_KEY = configjson['aws_access_key_id']
    SECRET_KEY = configjson['aws_secret_access_key']
    SESSION_TOKEN = configjson['aws_session_token']

    s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    aws_session_token=SESSION_TOKEN)


    parser = SparkLogParser()

    input_path = args.input
    s3init = input_path.find("s3://")
    s3initlen = s3init + 5
    secondslash = input_path.find("/", s3initlen)
    bucket_name = input_path[s3initlen:secondslash]

    lastslash = input_path.rfind('/')
    prefix = input_path[lastslash + 1:]

    is_file_binary = prefix.find('.')




    if is_file_binary > -1:
        print(f"Parsing file: {input_path}")
        parser.parse_file(str(input_path) , bucket_name)
        parser._finalize_sql_job_associations()
    elif is_file_binary == -1 :
        print(f"Parsing directory: {input_path}")
        parser.parse_directory(str(input_path), bucket_name, prefix)
    else:
        print(f"Error: {args.input} does not exist")
        return 1

    #output_dir = Path(args.output)
    #output_dir.mkdir(parents=True, exist_ok=True)

    job_metrics = parser.get_job_metrics()
    stage_metrics = parser.get_stage_metrics()
    sql_metrics = parser.get_sql_metrics()

    print(f"\nExtracted metrics:")
    print(f"  - Jobs: {len(job_metrics)}")
    print(f"  - Stages: {len(stage_metrics)}")
    print(f"  - SQL Executions: {len(sql_metrics)}")

    #if args.format == "json":
    #    with open(output_dir / "job_metrics.json", "w") as f:
    #        json.dump(job_metrics, f, indent=2)
    #    with open(output_dir / "stage_metrics.json", "w") as f:
    #        json.dump(stage_metrics, f, indent=2)
    #    with open(output_dir / "sql_metrics.json", "w") as f:
    #        json.dump(sql_metrics, f, indent=2)
    #else:
    #    with open(output_dir / "job_metrics.jsonl", "w") as f:
    #        for m in job_metrics:
    #            f.write(json.dumps(m) + "\n")
    #    with open(output_dir / "stage_metrics.jsonl", "w") as f:
    #        for m in stage_metrics:
    #            f.write(json.dumps(m) + "\n")
    #    with open(output_dir / "sql_metrics.jsonl", "w") as f:
    #        for m in sql_metrics:
    #            f.write(json.dumps(m) + "\n")

    #print(f"\nOutput written to: {output_dir}/")
    #print(f"  - job_metrics.{args.format}")
    #print(f"  - stage_metrics.{args.format}")
    #print(f"  - sql_metrics.{args.format}")

    return 0


if __name__ == "__main__":
    exit(main())
