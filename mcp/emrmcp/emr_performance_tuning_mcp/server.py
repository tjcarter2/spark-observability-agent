"""MCP Server for Spark History Log Parsing from EMR clusters."""

import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from spark_history_mcp.config.config import ServerConfig, SparkAppConfig
from spark_history_mcp.emr_client import EMRPersistentUIClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

server = Server("spark-history-mcp")

_emr_client: Optional[EMRPersistentUIClient] = None
_server_config: Optional[ServerConfig] = None


def get_client() -> EMRPersistentUIClient:
    """Get or create the EMR client singleton."""
    global _emr_client, _server_config
    if _emr_client is None:
        if _server_config is None:
            _server_config = ServerConfig()
        _emr_client = EMRPersistentUIClient(_server_config)
        _emr_client.initialize()
    return _emr_client


def format_response(data: Any, success: bool = True, error: str = None) -> str:
    """Format response as JSON string."""
    response = {"success": success, "data": data}
    if error:
        response["error"] = error
    return json.dumps(response, indent=2, default=str)


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools for Spark History analysis."""
    return [
        Tool(
            name="list_spark_applications",
            description=(
                "List all Spark applications from the EMR cluster's history server. "
                "Returns application IDs, names, start/end times, and status. "
                "Use this first to discover available applications for analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_application_summary",
            description=(
                "Get a comprehensive summary of a Spark application including "
                "job counts, stage counts, executor info, and duration. "
                "Useful for initial assessment of application health."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID (e.g., 'application_1234567890123_0001')",
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="get_application_jobs",
            description=(
                "Get all jobs for a Spark application with execution metrics. "
                "Returns job IDs, status, stage info, and timing details. "
                "Helpful for identifying slow or failed jobs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="get_application_stages",
            description=(
                "Get all stages for a Spark application with detailed metrics. "
                "Returns stage IDs, input/output records, shuffle metrics, "
                "task counts, and timing. Essential for identifying bottlenecks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="get_stage_details",
            description=(
                "Get detailed metrics for a specific stage including task summary "
                "statistics (min, median, max, percentiles) for duration, GC time, "
                "shuffle read/write, and memory spill. Critical for tuning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                    "stage_id": {
                        "type": "integer",
                        "description": "Stage ID number",
                    },
                    "attempt_id": {
                        "type": "integer",
                        "description": "Stage attempt ID (default 0)",
                        "default": 0,
                    },
                },
                "required": ["app_id", "stage_id"],
            },
        ),
        Tool(
            name="get_stage_tasks",
            description=(
                "Get individual task-level metrics for a stage. Returns detailed "
                "timing, shuffle, and memory metrics per task. Use for diagnosing "
                "data skew and straggler tasks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                    "stage_id": {
                        "type": "integer",
                        "description": "Stage ID number",
                    },
                    "attempt_id": {
                        "type": "integer",
                        "description": "Stage attempt ID (default 0)",
                        "default": 0,
                    },
                },
                "required": ["app_id", "stage_id"],
            },
        ),
        Tool(
            name="get_executors",
            description=(
                "Get executor metrics for a Spark application. Returns memory usage, "
                "disk usage, active tasks, completed tasks, and GC time per executor. "
                "Use for resource utilization analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                    "include_dead": {
                        "type": "boolean",
                        "description": "Include dead/removed executors",
                        "default": False,
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="get_sql_queries",
            description=(
                "Get all SQL queries executed by a Spark application. Returns "
                "query text, execution plan, duration, and job IDs. Essential "
                "for SQL-based workload analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="get_sql_query_details",
            description=(
                "Get detailed metrics for a specific SQL query including "
                "physical plan, metrics, and associated jobs/stages. "
                "Use for deep-dive SQL optimization."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                    "execution_id": {
                        "type": "integer",
                        "description": "SQL execution ID",
                    },
                },
                "required": ["app_id", "execution_id"],
            },
        ),
        Tool(
            name="get_spark_configuration",
            description=(
                "Get Spark configuration and environment for an application. "
                "Returns spark.* properties, classpath, JVM args, and system "
                "properties. Critical for configuration tuning recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="get_storage_info",
            description=(
                "Get RDD and DataFrame storage/caching information. Returns "
                "cached data sizes, memory vs disk usage, and partition counts. "
                "Use for cache optimization analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="analyze_performance_issues",
            description=(
                "Perform comprehensive performance analysis on a Spark application. "
                "Analyzes stages for skew, spill, GC issues, and shuffle bottlenecks. "
                "Returns prioritized list of issues with recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="get_full_application_report",
            description=(
                "Generate a complete diagnostic report for a Spark application. "
                "Includes application summary, jobs, stages, executors, SQL queries, "
                "configuration, and performance analysis. Comprehensive but large output."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Spark application ID",
                    },
                    "include_tasks": {
                        "type": "boolean",
                        "description": "Include task-level details (can be very large)",
                        "default": False,
                    },
                },
                "required": ["app_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute MCP tool calls."""
    try:
        client = get_client()

        if name == "list_spark_applications":
            apps = client.get_applications()
            result = format_response(apps)

        elif name == "get_application_summary":
            app_id = arguments["app_id"]
            app = client.get_application(app_id)
            jobs = client.get_jobs(app_id)
            stages = client.get_stages(app_id)
            executors = client.get_executors(app_id)

            summary = {
                "application": app,
                "job_count": len(jobs),
                "jobs_succeeded": sum(1 for j in jobs if j.get("status") == "SUCCEEDED"),
                "jobs_failed": sum(1 for j in jobs if j.get("status") == "FAILED"),
                "stage_count": len(stages),
                "stages_completed": sum(
                    1 for s in stages if s.get("status") == "COMPLETE"
                ),
                "stages_failed": sum(1 for s in stages if s.get("status") == "FAILED"),
                "executor_count": len(executors),
                "total_cores": sum(e.get("totalCores", 0) for e in executors),
                "total_memory_gb": sum(
                    e.get("maxMemory", 0) for e in executors
                ) / (1024**3),
            }
            result = format_response(summary)

        elif name == "get_application_jobs":
            app_id = arguments["app_id"]
            jobs = client.get_jobs(app_id)
            result = format_response(jobs)

        elif name == "get_application_stages":
            app_id = arguments["app_id"]
            stages = client.get_stages(app_id)
            result = format_response(stages)

        elif name == "get_stage_details":
            app_id = arguments["app_id"]
            stage_id = arguments["stage_id"]
            attempt_id = arguments.get("attempt_id", 0)

            stage = client.get_stage(app_id, stage_id, attempt_id)
            try:
                task_summary = client.get_stage_task_summary(app_id, stage_id, attempt_id)
                stage["taskSummary"] = task_summary
            except Exception as e:
                logger.warning(f"Could not get task summary: {e}")

            result = format_response(stage)

        elif name == "get_stage_tasks":
            app_id = arguments["app_id"]
            stage_id = arguments["stage_id"]
            attempt_id = arguments.get("attempt_id", 0)
            tasks = client.get_stage_tasks(app_id, stage_id, attempt_id)
            result = format_response(tasks)

        elif name == "get_executors":
            app_id = arguments["app_id"]
            include_dead = arguments.get("include_dead", False)

            if include_dead:
                executors = client.get_all_executors(app_id)
            else:
                executors = client.get_executors(app_id)

            result = format_response(executors)

        elif name == "get_sql_queries":
            app_id = arguments["app_id"]
            sql_queries = client.get_sql_queries(app_id)
            result = format_response(sql_queries)

        elif name == "get_sql_query_details":
            app_id = arguments["app_id"]
            execution_id = arguments["execution_id"]
            sql_detail = client.get_sql_query(app_id, execution_id)
            result = format_response(sql_detail)

        elif name == "get_spark_configuration":
            app_id = arguments["app_id"]
            env = client.get_environment(app_id)
            result = format_response(env)

        elif name == "get_storage_info":
            app_id = arguments["app_id"]
            storage = client.get_storage_rdd(app_id)
            result = format_response(storage)

        elif name == "analyze_performance_issues":
            app_id = arguments["app_id"]
            analysis = await _analyze_performance(client, app_id)
            result = format_response(analysis)

        elif name == "get_full_application_report":
            app_id = arguments["app_id"]
            include_tasks = arguments.get("include_tasks", False)
            report = await _generate_full_report(client, app_id, include_tasks)
            result = format_response(report)

        else:
            result = format_response(None, success=False, error=f"Unknown tool: {name}")

        return [TextContent(type="text", text=result)]

    except Exception as e:
        logger.exception(f"Error executing tool {name}")
        error_result = format_response(None, success=False, error=str(e))
        return [TextContent(type="text", text=error_result)]


async def _analyze_performance(
    client: EMRPersistentUIClient, app_id: str
) -> Dict[str, Any]:
    """Analyze application for common performance issues."""
    issues = []
    recommendations = []

    stages = client.get_stages(app_id)
    executors = client.get_executors(app_id)

    for stage in stages:
        stage_id = stage.get("stageId")
        status = stage.get("status")

        if status != "COMPLETE":
            continue

        try:
            task_summary = client.get_stage_task_summary(app_id, stage_id, 0)
        except Exception:
            continue

        duration_metrics = task_summary.get("executorRunTime", {})
        gc_metrics = task_summary.get("jvmGcTime", {})
        shuffle_read = task_summary.get("shuffleReadBytes", {})
        shuffle_write = task_summary.get("shuffleWriteBytes", {})
        spill_metrics = task_summary.get("memoryBytesSpilled", {})
        disk_spill = task_summary.get("diskBytesSpilled", {})

        if duration_metrics:
            min_dur = duration_metrics.get("min", 0)
            max_dur = duration_metrics.get("max", 0)
            median_dur = duration_metrics.get("median", 0)

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

        if gc_metrics:
            max_gc = gc_metrics.get("max", 0)
            median_gc = gc_metrics.get("median", 0)
            if duration_metrics:
                median_dur = duration_metrics.get("median", 1)
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
            mem_spill = spill_metrics.get("max", 0) if spill_metrics else 0
            disk_spill_val = disk_spill.get("max", 0) if disk_spill else 0

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


async def _generate_full_report(
    client: EMRPersistentUIClient, app_id: str, include_tasks: bool = False
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
        report["application"] = client.get_application(app_id)
    except Exception as e:
        logger.warning(f"Could not get application details: {e}")

    try:
        report["jobs"] = client.get_jobs(app_id)
    except Exception as e:
        logger.warning(f"Could not get jobs: {e}")

    try:
        stages = client.get_stages(app_id)
        for stage in stages:
            stage_data = {"stage": stage}
            try:
                stage_data["task_summary"] = client.get_stage_task_summary(
                    app_id, stage["stageId"], 0
                )
            except Exception:
                pass

            if include_tasks:
                try:
                    stage_data["tasks"] = client.get_stage_tasks(
                        app_id, stage["stageId"], 0
                    )
                except Exception:
                    pass

            report["stages"].append(stage_data)
    except Exception as e:
        logger.warning(f"Could not get stages: {e}")

    try:
        report["executors"] = client.get_all_executors(app_id)
    except Exception as e:
        logger.warning(f"Could not get executors: {e}")

    try:
        report["sql_queries"] = client.get_sql_queries(app_id)
    except Exception as e:
        logger.warning(f"Could not get SQL queries: {e}")

    try:
        report["environment"] = client.get_environment(app_id)
    except Exception as e:
        logger.warning(f"Could not get environment: {e}")

    try:
        report["storage"] = client.get_storage_rdd(app_id)
    except Exception as e:
        logger.warning(f"Could not get storage info: {e}")

    try:
        report["performance_analysis"] = await _analyze_performance(client, app_id)
    except Exception as e:
        logger.warning(f"Could not analyze performance: {e}")

    return report


def main():
    """Run the MCP server."""
    import asyncio

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
