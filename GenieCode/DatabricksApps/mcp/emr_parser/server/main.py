"""MCP Server for Spark History Log Parsing from EMR clusters."""

import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from emr_client import EMRPersistentUIClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Spark History MCP",
    instructions=(
        "Analyze Spark applications from EMR cluster history servers. "
        "Use list_spark_applications first to discover apps, then drill into "
        "jobs, stages, executors, SQL queries, and performance issues."
    ),
    host="0.0.0.0",
    port=8000,
    stateless_http=True,
)

_emr_client: Optional[EMRPersistentUIClient] = None


def get_client(cred_name: str, cluster_arn: str) -> EMRPersistentUIClient:
    """Get or create the EMR client singleton."""
    global _emr_client
    if _emr_client is None or not getattr(_emr_client, 'api_base', None):
        _emr_client = None
        client = EMRPersistentUIClient(cred_name, cluster_arn)
        try:
            client.initialize()
        except Exception:
            _emr_client = None
            raise
        _emr_client = client
    return _emr_client


def format_response(data: Any, success: bool = True, error: str = None) -> str:
    """Format response as JSON string."""
    response = {"success": success, "data": data}
    if error:
        response["error"] = error
    return json.dumps(response, indent=2, default=str)


def _get_quantile_value(metric_data, index: int, default=0):
    """
    Safely extract a value from a task summary metric.

    The Spark History Server taskSummary endpoint returns metrics in two possible formats:
    1. A list of quantile values: [5th, 25th, 50th, 75th, 95th]
    2. A dict with named keys: {"min": ..., "median": ..., "max": ...}

    Args:
        metric_data: The metric data (list or dict)
        index: For list format, the index to access. Common indices:
               0=min/5th, 1=25th, 2=median/50th, 3=75th, 4=max/95th
        default: Default value if extraction fails

    Returns:
        The extracted numeric value or the default
    """
    if metric_data is None:
        return default
    if isinstance(metric_data, list):
        if len(metric_data) > index:
            return metric_data[index] or default
        return default
    if isinstance(metric_data, dict):
        # Fallback for dict format (some Spark versions)
        key_map = {0: "min", 2: "median", 4: "max"}
        key = key_map.get(index, str(index))
        return metric_data.get(key, default)
    return default


@mcp.tool()
def list_spark_applications(cred_name: str, cluster_arn: str) -> str:
    """
    List all Spark applications from the EMR cluster's history server.
    Returns application IDs, names, start/end times, and status.
    Use this first to discover available applications for analysis.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
    """
    client = get_client(cred_name, cluster_arn)
    apps = client.get_applications()
    return format_response(apps)


@mcp.tool()
def get_application_summary(
    cred_name: str, cluster_arn: str, app_id: str, app_attempt_id: str = None
) -> str:
    """
    Get a comprehensive summary of a Spark application including
    job counts, stage counts, executor info, and duration.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID (e.g., 'application_1234567890123_0001')
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    app = client.get_application(app_id, attempt_id=app_attempt_id)
    jobs = client.get_jobs(app_id, attempt_id=app_attempt_id)
    stages = client.get_stages(app_id, attempt_id=app_attempt_id)
    executors = client.get_executors(app_id, attempt_id=app_attempt_id)

    summary = {
        "application": app,
        "job_count": len(jobs),
        "jobs_succeeded": sum(1 for j in jobs if j.get("status") == "SUCCEEDED"),
        "jobs_failed": sum(1 for j in jobs if j.get("status") == "FAILED"),
        "stage_count": len(stages),
        "stages_completed": sum(1 for s in stages if s.get("status") == "COMPLETE"),
        "stages_failed": sum(1 for s in stages if s.get("status") == "FAILED"),
        "executor_count": len(executors),
        "total_cores": sum(e.get("totalCores", 0) for e in executors),
        "total_memory_gb": sum(e.get("maxMemory", 0) for e in executors) / (1024**3),
    }
    return format_response(summary)


@mcp.tool()
def get_application_jobs(
    cred_name: str, cluster_arn: str, app_id: str, app_attempt_id: str = None
) -> str:
    """
    Get all jobs for a Spark application with execution metrics.
    Returns job IDs, status, stage info, and timing details.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    jobs = client.get_jobs(app_id, attempt_id=app_attempt_id)
    return format_response(jobs)


@mcp.tool()
def get_application_stages(
    cred_name: str, cluster_arn: str, app_id: str, app_attempt_id: str = None
) -> str:
    """
    Get all stages for a Spark application with detailed metrics.
    Returns stage IDs, input/output records, shuffle metrics, task counts, and timing.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    stages = client.get_stages(app_id, attempt_id=app_attempt_id)
    return format_response(stages)


@mcp.tool()
def get_stage_details(
    cred_name: str, cluster_arn: str, app_id: str, stage_id: int,
    stage_attempt_id: int = 0, app_attempt_id: str = None
) -> str:
    """
    Get detailed metrics for a specific stage including task summary
    statistics (min, median, max, percentiles) for duration, GC time,
    shuffle read/write, and memory spill.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        stage_id: Stage ID number
        stage_attempt_id: Stage attempt ID (default 0)
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    stage = client.get_stage(app_id, stage_id, stage_attempt_id, app_attempt_id=app_attempt_id)
    try:
        task_summary = client.get_stage_task_summary(
            app_id, stage_id, stage_attempt_id, app_attempt_id=app_attempt_id
        )
        stage["taskSummary"] = task_summary
    except Exception as e:
        logger.warning(f"Could not get task summary: {e}")
    return format_response(stage)


@mcp.tool()
def get_stage_tasks(
    cred_name: str, cluster_arn: str, app_id: str, stage_id: int,
    stage_attempt_id: int = 0, app_attempt_id: str = None
) -> str:
    """
    Get individual task-level metrics for a stage. Returns detailed
    timing, shuffle, and memory metrics per task. Use for diagnosing
    data skew and straggler tasks.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        stage_id: Stage ID number
        stage_attempt_id: Stage attempt ID (default 0)
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    tasks = client.get_stage_tasks(app_id, stage_id, stage_attempt_id, app_attempt_id=app_attempt_id)
    return format_response(tasks)


@mcp.tool()
def get_executors(
    cred_name: str, cluster_arn: str, app_id: str,
    include_dead: bool = False, app_attempt_id: str = None
) -> str:
    """
    Get executor metrics for a Spark application. Returns memory usage,
    disk usage, active tasks, completed tasks, and GC time per executor.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        include_dead: Include dead/removed executors
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    if include_dead:
        executors = client.get_all_executors(app_id, attempt_id=app_attempt_id)
    else:
        executors = client.get_executors(app_id, attempt_id=app_attempt_id)
    return format_response(executors)


@mcp.tool()
def get_sql_queries(
    cred_name: str, cluster_arn: str, app_id: str, app_attempt_id: str = None
) -> str:
    """
    Get all SQL queries executed by a Spark application. Returns
    query text, execution plan, duration, and job IDs.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    sql_queries = client.get_sql_queries(app_id, attempt_id=app_attempt_id)
    return format_response(sql_queries)


@mcp.tool()
def get_sql_query_details(
    cred_name: str, cluster_arn: str, app_id: str, execution_id: int,
    app_attempt_id: str = None
) -> str:
    """
    Get detailed metrics for a specific SQL query including
    physical plan, metrics, and associated jobs/stages.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        execution_id: SQL execution ID
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    sql_detail = client.get_sql_query(app_id, execution_id, attempt_id=app_attempt_id)
    return format_response(sql_detail)


@mcp.tool()
def get_spark_configuration(
    cred_name: str, cluster_arn: str, app_id: str, app_attempt_id: str = None
) -> str:
    """
    Get Spark configuration and environment for an application.
    Returns spark.* properties, classpath, JVM args, and system properties.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    env = client.get_environment(app_id, attempt_id=app_attempt_id)
    return format_response(env)


@mcp.tool()
def get_storage_info(
    cred_name: str, cluster_arn: str, app_id: str, app_attempt_id: str = None
) -> str:
    """
    Get RDD and DataFrame storage/caching information. Returns
    cached data sizes, memory vs disk usage, and partition counts.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    storage = client.get_storage_rdd(app_id, attempt_id=app_attempt_id)
    return format_response(storage)


@mcp.tool()
def analyze_performance_issues(
    cred_name: str, cluster_arn: str, app_id: str, app_attempt_id: str = None
) -> str:
    """
    Perform comprehensive performance analysis on a Spark application.
    Analyzes stages for skew, spill, GC issues, and shuffle bottlenecks.
    Returns prioritized list of issues with recommendations.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    analysis = _analyze_performance(client, app_id, app_attempt_id)
    return format_response(analysis)


@mcp.tool()
def get_full_application_report(
    cred_name: str, cluster_arn: str, app_id: str,
    include_tasks: bool = False, app_attempt_id: str = None
) -> str:
    """
    Generate a complete diagnostic report for a Spark application.
    Includes application summary, jobs, stages, executors, SQL queries,
    configuration, and performance analysis.

    Args:
        cred_name: Databricks service credential name for AWS access
        cluster_arn: EMR cluster ARN
        app_id: Spark application ID
        include_tasks: Include task-level details (can be very large)
        app_attempt_id: Application attempt ID. If None, auto-resolves the latest.
    """
    client = get_client(cred_name, cluster_arn)
    report = _generate_full_report(client, app_id, include_tasks, app_attempt_id)
    return format_response(report)


def _analyze_performance(
    client: EMRPersistentUIClient, app_id: str, app_attempt_id: str = None
) -> Dict[str, Any]:
    """Analyze application for common performance issues."""
    issues = []
    recommendations = []

    stages = client.get_stages(app_id, attempt_id=app_attempt_id)
    executors = client.get_executors(app_id, attempt_id=app_attempt_id)

    for stage in stages:
        stage_id = stage.get("stageId")
        status = stage.get("status")

        if status != "COMPLETE":
            continue

        try:
            task_summary = client.get_stage_task_summary(
                app_id, stage_id, 0, app_attempt_id=app_attempt_id
            )
        except Exception:
            continue

        # The taskSummary API returns metrics as lists of quantile values:
        # [5th percentile, 25th, 50th (median), 75th, 95th percentile]
        # Use _get_quantile_value to safely extract by index.
        duration_metrics = task_summary.get("executorRunTime") if isinstance(task_summary, dict) else None
        gc_metrics = task_summary.get("jvmGcTime") if isinstance(task_summary, dict) else None
        spill_metrics = task_summary.get("memoryBytesSpilled") if isinstance(task_summary, dict) else None
        disk_spill = task_summary.get("diskBytesSpilled") if isinstance(task_summary, dict) else None

        if duration_metrics:
            max_dur = _get_quantile_value(duration_metrics, 4)  # 95th percentile / max
            median_dur = _get_quantile_value(duration_metrics, 2)  # 50th percentile / median

            if max_dur > 0 and median_dur > 0:
                skew_ratio = max_dur / median_dur
                if skew_ratio > 10:
                    issues.append({
                        "type": "DATA_SKEW",
                        "severity": "HIGH",
                        "stage_id": stage_id,
                        "details": f"Max task duration ({max_dur}ms) is {skew_ratio:.1f}x median ({median_dur}ms)",
                    })
                    recommendations.append({
                        "issue_type": "DATA_SKEW",
                        "stage_id": stage_id,
                        "recommendation": (
                            "Consider salting keys, using adaptive query execution, "
                            "or repartitioning data to reduce skew."
                        ),
                    })

        if gc_metrics and duration_metrics:
            median_gc = _get_quantile_value(gc_metrics, 2)  # median GC time
            median_dur = _get_quantile_value(duration_metrics, 2)  # median duration
            gc_ratio = median_gc / median_dur if median_dur > 0 else 0
            if gc_ratio > 0.1:
                issues.append({
                    "type": "HIGH_GC_TIME",
                    "severity": "MEDIUM",
                    "stage_id": stage_id,
                    "details": f"GC time is {gc_ratio*100:.1f}% of task duration",
                })
                recommendations.append({
                    "issue_type": "HIGH_GC_TIME",
                    "stage_id": stage_id,
                    "recommendation": (
                        "Increase executor memory, tune GC settings, "
                        "or reduce data size per task."
                    ),
                })

        if spill_metrics or disk_spill:
            mem_spill = _get_quantile_value(spill_metrics, 4) if spill_metrics else 0  # max spill
            disk_spill_val = _get_quantile_value(disk_spill, 4) if disk_spill else 0  # max disk spill

            if mem_spill > 0 or disk_spill_val > 0:
                issues.append({
                    "type": "MEMORY_SPILL",
                    "severity": "MEDIUM" if disk_spill_val == 0 else "HIGH",
                    "stage_id": stage_id,
                    "details": f"Memory spill: {mem_spill/(1024**2):.1f}MB, Disk spill: {disk_spill_val/(1024**2):.1f}MB",
                })
                recommendations.append({
                    "issue_type": "MEMORY_SPILL",
                    "stage_id": stage_id,
                    "recommendation": (
                        "Increase spark.memory.fraction or executor memory, "
                        "or reduce partition sizes."
                    ),
                })

        input_records = stage.get("inputRecords", 0)
        output_records = stage.get("outputRecords", 0)
        num_tasks = stage.get("numTasks", 1)

        if num_tasks > 0:
            records_per_task = (input_records + output_records) / num_tasks
            if records_per_task < 1000 and num_tasks > 200:
                issues.append({
                    "type": "TOO_MANY_SMALL_TASKS",
                    "severity": "LOW",
                    "stage_id": stage_id,
                    "details": f"{num_tasks} tasks with ~{records_per_task:.0f} records each",
                })
                recommendations.append({
                    "issue_type": "TOO_MANY_SMALL_TASKS",
                    "stage_id": stage_id,
                    "recommendation": (
                        "Consider coalescing partitions or increasing "
                        "spark.sql.shuffle.partitions appropriately."
                    ),
                })

    total_gc = sum(e.get("totalGCTime", 0) for e in executors)
    total_duration = sum(e.get("totalDuration", 0) for e in executors)
    if total_duration > 0 and total_gc / total_duration > 0.1:
        issues.append({
            "type": "CLUSTER_WIDE_GC",
            "severity": "HIGH",
            "details": f"Cluster-wide GC is {total_gc/total_duration*100:.1f}% of total executor time",
        })
        recommendations.append({
            "issue_type": "CLUSTER_WIDE_GC",
            "recommendation": (
                "Consider G1GC, increase executor memory, or tune "
                "spark.memory.fraction and spark.memory.storageFraction."
            ),
        })

    return {
        "app_id": app_id,
        "issues_found": len(issues),
        "issues": sorted(issues, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.get("severity"), 3)),
        "recommendations": recommendations,
    }


def _generate_full_report(
    client: EMRPersistentUIClient, app_id: str,
    include_tasks: bool = False, app_attempt_id: str = None
) -> Dict[str, Any]:
    """Generate comprehensive application report."""
    report = {
        "app_id": app_id,
        "application": None,
        "jobs": [],
        "stages": [],
        "executors": [],
        "sql_queries": [],
        "environment": None,
        "storage": [],
        "performance_analysis": None,
    }

    try:
        report["application"] = client.get_application(app_id, attempt_id=app_attempt_id)
    except Exception as e:
        logger.warning(f"Could not get application details: {e}")

    try:
        report["jobs"] = client.get_jobs(app_id, attempt_id=app_attempt_id)
    except Exception as e:
        logger.warning(f"Could not get jobs: {e}")

    try:
        stages = client.get_stages(app_id, attempt_id=app_attempt_id)
        for stage in stages:
            stage_data = {"stage": stage}
            try:
                stage_data["task_summary"] = client.get_stage_task_summary(
                    app_id, stage["stageId"], 0, app_attempt_id=app_attempt_id
                )
            except Exception:
                pass

            if include_tasks:
                try:
                    stage_data["tasks"] = client.get_stage_tasks(
                        app_id, stage["stageId"], 0, app_attempt_id=app_attempt_id
                    )
                except Exception:
                    pass

            report["stages"].append(stage_data)
    except Exception as e:
        logger.warning(f"Could not get stages: {e}")

    try:
        report["executors"] = client.get_all_executors(app_id, attempt_id=app_attempt_id)
    except Exception as e:
        logger.warning(f"Could not get executors: {e}")

    try:
        report["sql_queries"] = client.get_sql_queries(app_id, attempt_id=app_attempt_id)
    except Exception as e:
        logger.warning(f"Could not get SQL queries: {e}")

    try:
        report["environment"] = client.get_environment(app_id, attempt_id=app_attempt_id)
    except Exception as e:
        logger.warning(f"Could not get environment: {e}")

    try:
        report["storage"] = client.get_storage_rdd(app_id, attempt_id=app_attempt_id)
    except Exception as e:
        logger.warning(f"Could not get storage info: {e}")

    try:
        report["performance_analysis"] = _analyze_performance(client, app_id, app_attempt_id)
    except Exception as e:
        logger.warning(f"Could not analyze performance: {e}")

    return report


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Spark History MCP Server")
    arg_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    arg_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )

    args = arg_parser.parse_args()

    if args.transport == "streamable-http":
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        _original_config_init = uvicorn.Config.__init__

        def _config_init_with_cors(self, app, *args, **kwargs):
            if hasattr(app, "add_middleware"):
                app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["https://e2-demo-field-eng.cloud.databricks.com"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            _original_config_init(self, app, *args, **kwargs)

        uvicorn.Config.__init__ = _config_init_with_cors

        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
