#!/usr/bin/env python3
"""
Spark Log Parser MCP Server

Exposes Spark event log parsing and tuning recommendations as MCP tools.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent))

from spark_log_parser import SparkLogParser, get_s3_client, parse_s3_path
from tuning_advisor import SparkTuningAdvisor

mcp = FastMCP(
    "Spark Log Parser",
    instructions="Parse Spark event logs and get comprehensive tuning recommendations. Use parse_spark_logs to get an overview, get_tuning_recommendations for optimization suggestions, and analyze_sql_execution for detailed query analysis.",
    host="0.0.0.0",
    port=8000,
    stateless_http=True,
)


def _parse_logs(path: str, cred_name: str) -> tuple[SparkLogParser, str]:
    """Helper to parse logs from a path."""
    parser = SparkLogParser()
    source = path

    if path.startswith("s3://"):
        s3_client = get_s3_client(cred_name)
        bucket_name, prefix = parse_s3_path(path)
        
        is_file = '.' in prefix.split('/')[-1] if prefix else False
        
        if is_file:
            print(f"Parsing S3 file: {path}")
            parser.parse_file(prefix, bucket_name, s3_client)
            parser._finalize_sql_job_associations()
        else:
            print(f"Parsing S3 directory: {path}")
            parser.parse_directory(path, bucket_name, prefix, s3_client)
    else:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if input_path.is_file():
            print(f"Parsing local file: {path}")
            with open(input_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        parser._process_event(event)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
            parser._finalize_sql_job_associations()
        else:
            print(f"Parsing local directory: {path}")
            for file_path in sorted(input_path.glob("*.json")):
                print(f"Parsing: {file_path}")
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            parser._process_event(event)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line {line_num}: {e}")
            parser._finalize_sql_job_associations()

    return parser, source


@mcp.tool()
def parse_spark_logs(path: str, cred_name: str) -> dict[str, Any]:
    """
    Parse Spark event logs from a file or directory.

    Args:
        path: Path to a Spark event log file (.json) or directory containing log files
        cred_name: Name of service credentials for S3 access

    Returns:
        Summary of parsed metrics including job, stage, and SQL execution counts
    """
    parser, source = _parse_logs(path, cred_name)

    job_metrics = parser.get_job_metrics()
    stage_metrics = parser.get_stage_metrics()
    sql_metrics = parser.get_sql_metrics()

    job_statuses = {}
    for j in job_metrics:
        status = j.get("status", "Unknown")
        job_statuses[status] = job_statuses.get(status, 0) + 1

    stage_statuses = {}
    for s in stage_metrics:
        status = s.get("status", "Unknown")
        stage_statuses[status] = stage_statuses.get(status, 0) + 1

    sql_statuses = {}
    for s in sql_metrics:
        status = s.get("status", "Unknown")
        sql_statuses[status] = sql_statuses.get(status, 0) + 1

    total_shuffle_read = sum(s.get("shuffleReadBytes", 0) for s in stage_metrics)
    total_shuffle_write = sum(s.get("shuffleWriteBytes", 0) for s in stage_metrics)
    total_input = sum(s.get("inputBytes", 0) for s in stage_metrics)
    total_output = sum(s.get("outputBytes", 0) for s in stage_metrics)

    return {
        "source": source,
        "summary": {
            "jobs": {
                "total": len(job_metrics),
                "by_status": job_statuses,
            },
            "stages": {
                "total": len(stage_metrics),
                "by_status": stage_statuses,
            },
            "sql_executions": {
                "total": len(sql_metrics),
                "by_status": sql_statuses,
            },
        },
        "io_summary": {
            "total_input_bytes": total_input,
            "total_output_bytes": total_output,
            "total_shuffle_read_bytes": total_shuffle_read,
            "total_shuffle_write_bytes": total_shuffle_write,
            "total_input_gb": round(total_input / (1024**3), 2),
            "total_shuffle_gb": round((total_shuffle_read + total_shuffle_write) / (1024**3), 2),
        },
    }


@mcp.tool()
def get_job_metrics(path: str, cred_name: str) -> list[dict[str, Any]]:
    """
    Get detailed job metrics from Spark event logs.

    Args:
        path: Path to a Spark event log file or directory
        cred_name: Name of service credentials for S3 access

    Returns:
        List of job metrics including jobId, status, task counts, and stage information
    """
    parser, _ = _parse_logs(path, cred_name)
    return parser.get_job_metrics()


@mcp.tool()
def get_stage_metrics(path: str, cred_name: str) -> list[dict[str, Any]]:
    """
    Get detailed stage metrics from Spark event logs.

    Args:
        path: Path to a Spark event log file or directory
        cred_name: Name of service credentials for S3 access

    Returns:
        List of stage metrics including I/O stats, shuffle metrics, and spill data
    """
    parser, _ = _parse_logs(path, cred_name)
    return parser.get_stage_metrics()


@mcp.tool()
def get_sql_metrics(path: str, cred_name: str) -> list[dict[str, Any]]:
    """
    Get detailed SQL execution metrics from Spark event logs.

    Args:
        path: Path to a Spark event log file or directory
        cred_name: Name of service credentials for S3 access

    Returns:
        List of SQL metrics including execution plans, node metrics, and timing
    """
    parser, _ = _parse_logs(path, cred_name)
    return parser.get_sql_metrics()


@mcp.tool()
def get_tuning_recommendations(path: str, cred_name: str) -> dict[str, Any]:
    """
    Analyze Spark event logs and provide comprehensive tuning recommendations.

    Args:
        path: Path to a Spark event log file or directory
        cred_name: Name of service credentials for S3 access

    Returns:
        Tuning recommendations categorized by severity and type
    """
    parser, source = _parse_logs(path, cred_name)

    job_metrics = parser.get_job_metrics()
    stage_metrics = parser.get_stage_metrics()
    sql_metrics = parser.get_sql_metrics()

    advisor = SparkTuningAdvisor(job_metrics, stage_metrics, sql_metrics)
    recommendations = advisor.analyze()
    summary = advisor.get_summary()

    return {
        "source": source,
        "summary": summary,
        "recommendations": [
            {
                "category": r.category,
                "severity": r.severity,
                "title": r.title,
                "description": r.description,
                "current_value": r.current_value,
                "recommended_action": r.recommended_action,
                "config_suggestion": r.config_suggestion,
            }
            for r in recommendations
        ],
    }


@mcp.tool()
def get_tuning_report(path: str, cred_name: str) -> str:
    """
    Generate a formatted tuning report from Spark event logs.

    Args:
        path: Path to a Spark event log file or directory
        cred_name: Name of service credentials for S3 access

    Returns:
        Markdown-formatted tuning report with prioritized recommendations
    """
    parser, source = _parse_logs(path, cred_name)

    job_metrics = parser.get_job_metrics()
    stage_metrics = parser.get_stage_metrics()
    sql_metrics = parser.get_sql_metrics()

    advisor = SparkTuningAdvisor(job_metrics, stage_metrics, sql_metrics)
    advisor.analyze()

    header = f"# Spark Performance Analysis\n\n**Source:** {source}\n\n"
    header += f"**Jobs:** {len(job_metrics)} | **Stages:** {len(stage_metrics)} | **SQL Executions:** {len(sql_metrics)}\n\n"
    header += "---\n"

    return header + advisor.format_recommendations()


@mcp.tool()
def analyze_sql_execution(path: str, execution_id: int, cred_name: str) -> dict[str, Any]:
    """
    Get detailed analysis of a specific SQL execution.

    Args:
        path: Path to a Spark event log file or directory
        execution_id: The SQL execution ID to analyze
        cred_name: Name of service credentials for S3 access

    Returns:
        Detailed SQL execution metrics including plan nodes and their metrics
    """
    parser, _ = _parse_logs(path, cred_name)
    sql_metrics = parser.get_sql_metrics()

    target = None
    for sql in sql_metrics:
        if sql.get("id") == execution_id:
            target = sql
            break

    if not target:
        return {"error": f"SQL execution {execution_id} not found"}

    nodes_with_metrics = [n for n in target.get("nodes", []) if n.get("metrics")]
    total_metrics = sum(len(n.get("metrics", [])) for n in target.get("nodes", []))

    node_types = {}
    for n in target.get("nodes", []):
        name = n.get("nodeName", "").split()[0]
        if name:
            node_types[name] = node_types.get(name, 0) + 1

    return {
        "execution_id": execution_id,
        "status": target.get("status"),
        "duration_ms": target.get("duration"),
        "description": target.get("description", "")[:200],
        "plan_summary": {
            "total_nodes": len(target.get("nodes", [])),
            "nodes_with_metrics": len(nodes_with_metrics),
            "total_metrics_collected": total_metrics,
            "node_types": node_types,
        },
        "success_job_ids": target.get("successJobIds", []),
        "failed_job_ids": target.get("failedJobIds", ""),
        "nodes": target.get("nodes", []),
    }


@mcp.tool()
def export_metrics(path: str, cred_name: str, output_dir: str, format: str = "json") -> dict[str, str]:
    """
    Export parsed metrics to files.

    Args:
        path: Path to a Spark event log file or directory
        cred_name: Name of service credentials for S3 access
        output_dir: Directory to write output files
        format: Output format - 'json' or 'jsonl'


    Returns:
        Paths to the exported files
    """
    parser, _ = _parse_logs(path, cred_name)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    job_metrics = parser.get_job_metrics()
    stage_metrics = parser.get_stage_metrics()
    sql_metrics = parser.get_sql_metrics()

    files = {}
    ext = format if format in ["json", "jsonl"] else "json"

    if format == "jsonl":
        with open(output_path / f"job_metrics.{ext}", "w") as f:
            for m in job_metrics:
                f.write(json.dumps(m) + "\n")
        with open(output_path / f"stage_metrics.{ext}", "w") as f:
            for m in stage_metrics:
                f.write(json.dumps(m) + "\n")
        with open(output_path / f"sql_metrics.{ext}", "w") as f:
            for m in sql_metrics:
                f.write(json.dumps(m) + "\n")
    else:
        with open(output_path / f"job_metrics.{ext}", "w") as f:
            json.dump(job_metrics, f, indent=2)
        with open(output_path / f"stage_metrics.{ext}", "w") as f:
            json.dump(stage_metrics, f, indent=2)
        with open(output_path / f"sql_metrics.{ext}", "w") as f:
            json.dump(sql_metrics, f, indent=2)

    return {
        "job_metrics": str(output_path / f"job_metrics.{ext}"),
        "stage_metrics": str(output_path / f"stage_metrics.{ext}"),
        "sql_metrics": str(output_path / f"sql_metrics.{ext}"),
    }



@mcp.prompt()
def analyze_spark_performance(log_path: str) -> str:
    """
    Generate a prompt for comprehensive Spark performance analysis.

    Args:
        log_path: Path to Spark event logs
    """
    return f"""Please analyze the Spark event logs at '{log_path}' and provide:

1. **Overview**: Parse the logs and summarize the job/stage/SQL execution counts
2. **Performance Issues**: Identify any performance bottlenecks or failures
3. **Tuning Recommendations**: Get tuning recommendations and prioritize them
4. **Specific Optimizations**: For any high-priority issues, suggest specific configuration changes

Use these tools in order:
1. parse_spark_logs('{log_path}') - Get overview
2. get_tuning_recommendations('{log_path}') - Get recommendations
3. For detailed investigation, use get_job_metrics, get_stage_metrics, or get_sql_metrics

Provide a clear, actionable summary of findings."""


@mcp.prompt()
def investigate_slow_queries(log_path: str, threshold_seconds: int = 300) -> str:
    """
    Generate a prompt for investigating slow SQL queries.

    Args:
        log_path: Path to Spark event logs
        threshold_seconds: Duration threshold in seconds for "slow" queries
    """
    return f"""Please investigate slow SQL queries in the Spark logs at '{log_path}'.

1. Parse the SQL metrics using get_sql_metrics('{log_path}')
2. Identify queries taking longer than {threshold_seconds} seconds
3. For each slow query, use analyze_sql_execution to get detailed node metrics
4. Look for common patterns:
   - Data skew in joins
   - Excessive shuffle
   - Missing broadcast hints
   - Inefficient aggregations

Provide specific optimization suggestions for each slow query found."""


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Spark Log Parser MCP Server")
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
        # Patch uvicorn.Config.__init__ to inject CORS middleware into the
        # Starlette app that the MCP SDK creates internally. This is needed
        # because mcp.run() doesn't expose the app for direct modification.
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
