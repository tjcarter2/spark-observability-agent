# Spark Log Parser MCP Server

An MCP (Model Context Protocol) server that parses Spark event logs and provides comprehensive tuning recommendations. This allows AI agents to analyze Spark job performance and suggest optimizations.

## Features

- **Parse Spark Event Logs**: Extract job, stage, and SQL execution metrics from raw Spark event logs
- **Tuning Recommendations**: Get prioritized recommendations for performance optimization
- **Detailed Analysis**: Drill down into specific SQL executions and their plan nodes
- **Export Capabilities**: Export parsed metrics to JSON or JSONL format

## Installation

### Using uv (Recommended)

```bash
cd spark_mcp_server
uv sync
```

### Using pip

```bash
cd spark_mcp_server
pip install -e .
```

## Running the Server

### Stdio Transport (for Claude Desktop, Cursor, etc.)

```bash
uv run python server.py
# or
python server.py --transport stdio
```

### HTTP Transport (for web-based clients)

```bash
uv run python server.py --transport streamable-http --port 8000
```

Then connect to `http://localhost:8000/mcp` using the MCP Inspector or other HTTP clients.

## Available Tools

### `parse_spark_logs`
Parse Spark event logs and get a summary of metrics.

**Input:**
- `path`: Path to a Spark event log file or directory

**Output:** Summary including job/stage/SQL counts and I/O statistics

### `get_job_metrics`
Get detailed job metrics including status, task counts, and stage information.

### `get_stage_metrics`
Get detailed stage metrics including I/O stats, shuffle metrics, and spill data.

### `get_sql_metrics`
Get detailed SQL execution metrics including execution plans and node-level metrics.

### `get_tuning_recommendations`
Analyze logs and provide categorized tuning recommendations.

**Categories:**
- Job Reliability
- Stage Reliability
- Task Reliability
- Memory Management
- Shuffle Optimization
- Data Skew
- SQL Performance
- Parallelism

### `get_tuning_report`
Generate a formatted Markdown report with prioritized recommendations.

### `analyze_sql_execution`
Get detailed analysis of a specific SQL execution by ID.

**Input:**
- `path`: Path to Spark event logs
- `execution_id`: The SQL execution ID to analyze

### `export_metrics`
Export parsed metrics to files.

**Input:**
- `path`: Path to Spark event logs
- `output_dir`: Directory for output files
- `format`: 'json' or 'jsonl'

## Available Prompts

### `analyze_spark_performance`
Generates a comprehensive analysis prompt for Spark performance investigation.

### `investigate_slow_queries`
Generates a prompt for investigating slow SQL queries.

## Configuration for Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "spark-log-parser": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/spark_mcp_server", "python", "server.py"],
      "env": {}
    }
  }
}
```

## Configuration for Cursor

Add to your Cursor MCP settings:

```json
{
  "mcpServers": {
    "spark-log-parser": {
      "command": "python",
      "args": ["/path/to/spark_mcp_server/server.py"],
      "env": {}
    }
  }
}
```

## Example Usage

### Using MCP Inspector

```bash
# Start the server
uv run python server.py --transport streamable-http

# In another terminal, launch the inspector
npx -y @modelcontextprotocol/inspector
```

Connect to `http://localhost:8000/mcp` in the inspector.

### Sample Tool Calls

```python
# Parse logs and get summary
parse_spark_logs("/path/to/events")

# Get tuning recommendations
get_tuning_recommendations("/path/to/events")

# Analyze specific SQL execution
analyze_sql_execution("/path/to/events", 0)

# Export metrics
export_metrics("/path/to/events", "/path/to/output", "json")
```

## Tuning Recommendations

The server analyzes metrics and provides recommendations for:

| Issue | Detection | Recommendation |
|-------|-----------|----------------|
| Job Failures | Jobs with non-success status | Review logs, check OOM errors |
| Stage Retries | Stages with attemptId > 0 | Enable speculation |
| Data Spill | Memory/disk bytes spilled | Increase executor memory |
| Large Shuffle | High shuffle read/write | Use broadcast joins, enable AQE |
| Data Skew | Skew indicators in joins | Salting, AQE skew join |
| Long Queries | Duration > 5 minutes | Review execution plans |
| High Task Count | > 10000 tasks per stage | Reduce partition count |

## License

MIT
