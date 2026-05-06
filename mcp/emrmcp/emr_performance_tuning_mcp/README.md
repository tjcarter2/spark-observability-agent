# Spark History MCP Server

An MCP (Model Context Protocol) server that parses Spark history logs from EMR clusters and returns structured JSON data. Designed to help AI agents provide comprehensive tuning recommendations for inefficient Spark jobs.

## Features

- **EMR Integration**: Connects to EMR clusters via Persistent App UI for secure Spark History Server access
- **Comprehensive Metrics**: Extracts application, job, stage, task, executor, and SQL query metrics
- **Performance Analysis**: Built-in analysis for common issues like data skew, GC overhead, and memory spill
- **AI-Ready Output**: Returns structured JSON optimized for LLM consumption and tuning recommendations

## Available Tools

| Tool | Description |
|------|-------------|
| `list_spark_applications` | List all Spark applications from the history server |
| `get_application_summary` | Get comprehensive summary with job/stage/executor counts |
| `get_application_jobs` | Get all jobs with execution metrics |
| `get_application_stages` | Get all stages with detailed shuffle/IO metrics |
| `get_stage_details` | Get stage details with task summary statistics |
| `get_stage_tasks` | Get individual task-level metrics (for skew analysis) |
| `get_executors` | Get executor metrics (memory, cores, GC time) |
| `get_sql_queries` | Get SQL queries with execution plans |
| `get_sql_query_details` | Get detailed SQL query metrics |
| `get_spark_configuration` | Get Spark configuration and environment |
| `get_storage_info` | Get RDD/DataFrame caching information |
| `analyze_performance_issues` | Automated analysis with prioritized issues and recommendations |
| `get_full_application_report` | Generate complete diagnostic report |

## Installation

### From Source

```bash
# Clone the repository
cd emrmcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Using pip

```bash
pip install -r requirements.txt
```

## Configuration

The server is configured via environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `SPARK_HISTORY_EMR_CLUSTER_ARN` | Yes | ARN of the EMR cluster (e.g., `arn:aws:elasticmapreduce:us-east-1:123456789:cluster/j-ABC123DEF`) |
| `SPARK_HISTORY_TIMEOUT` | No | HTTP request timeout in seconds (default: 30) |
| `SPARK_HISTORY_AWS_PROFILE` | No | AWS profile name for authentication |
| `SPARK_HISTORY_AWS_REGION` | No | AWS region (auto-detected from ARN if not provided) |

### Example `.env` file

```bash
SPARK_HISTORY_EMR_CLUSTER_ARN=arn:aws:elasticmapreduce:us-east-1:123456789012:cluster/j-XXXXXXXXXXXXX
SPARK_HISTORY_TIMEOUT=30
SPARK_HISTORY_AWS_PROFILE=my-aws-profile
```

## AWS Authentication

The server uses boto3 for AWS authentication. Ensure you have valid credentials configured via one of:

1. **AWS Profile** (recommended for local development):
   ```bash
   aws configure --profile my-profile
   export SPARK_HISTORY_AWS_PROFILE=my-profile
   ```

2. **Environment Variables**:
   ```bash
   export AWS_ACCESS_KEY_ID=your-key
   export AWS_SECRET_ACCESS_KEY=your-secret
   ```

3. **IAM Role** (for EC2/ECS/Lambda)

### Required IAM Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "elasticmapreduce:CreatePersistentAppUI",
                "elasticmapreduce:DescribePersistentAppUI",
                "elasticmapreduce:GetPersistentAppUIPresignedURL"
            ],
            "Resource": "arn:aws:elasticmapreduce:*:*:cluster/*"
        }
    ]
}
```

## Running the Server

### Standalone

```bash
# Using the module
python -m spark_history_mcp

# Or using the installed script
spark-history-mcp
```

### With Cursor IDE

Add to your Cursor MCP settings (`~/.cursor/mcp.json` or workspace `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "spark-history": {
      "command": "python",
      "args": ["-m", "spark_history_mcp"],
      "cwd": "/path/to/emrmcp",
      "env": {
        "SPARK_HISTORY_EMR_CLUSTER_ARN": "arn:aws:elasticmapreduce:us-east-1:123456789012:cluster/j-XXXXXXXXXXXXX"
      }
    }
  }
}
```

## Usage Examples

### List Applications
```
Tool: list_spark_applications
Arguments: {}
```

### Analyze a Specific Application
```
Tool: analyze_performance_issues
Arguments: {"app_id": "application_1234567890123_0001"}
```

### Get Full Report
```
Tool: get_full_application_report
Arguments: {"app_id": "application_1234567890123_0001", "include_tasks": false}
```

## Typical Workflow for Tuning Recommendations

1. **List applications** to find the job you want to analyze
2. **Get application summary** for a quick health check
3. **Analyze performance issues** for automated issue detection
4. **Deep dive** into specific stages or SQL queries based on findings
5. **Review configuration** to identify tuning opportunities

## Response Format

All tools return JSON with the following structure:

```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

On error:

```json
{
  "success": false,
  "data": null,
  "error": "Error message"
}
```

## Performance Analysis

The `analyze_performance_issues` tool automatically detects:

- **Data Skew**: Tasks with max duration >10x median
- **High GC Time**: GC consuming >10% of task duration
- **Memory Spill**: Memory/disk spill during shuffle
- **Too Many Small Tasks**: Excessive task count with low record counts
- **Cluster-wide GC**: Overall GC overhead across all executors

Each issue includes severity (HIGH/MEDIUM/LOW) and specific recommendations.

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Formatting

```bash
black spark_history_mcp/
ruff check spark_history_mcp/
```

### Type Checking

```bash
mypy spark_history_mcp/
```

## Architecture

```
spark_history_mcp/
├── __init__.py          # Package initialization
├── __main__.py          # Entry point for `python -m`
├── server.py            # MCP server implementation
├── emr_client.py        # EMR Persistent UI client
└── config/
    ├── __init__.py
    └── config.py        # Configuration models
```

## Troubleshooting

### "No persistent UI ID available"
Ensure your EMR cluster ARN is correct and the cluster is running.

### "EMR Persistent UI status is STARTING"
The server waits up to 3 minutes for the UI to become ATTACHED. If this persists, check EMR cluster status.

### Authentication Errors
Verify your AWS credentials have the required EMR permissions listed above.

### Timeout Errors
Increase `SPARK_HISTORY_TIMEOUT` for large applications with many stages/tasks.

## License

MIT
