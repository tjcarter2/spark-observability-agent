In part three of this blog series we introduced an MCP server designed to parse raw spark logs in the cloud and provide [comprehensive tuning recommendations](https://medium.com/@techvaquero/agent-00111-spark-observability-agent-e325e0a90e27). These logs can be emitted by any spark engine, whether it is Databricks, EMR, OSS, etc. In addition, we introduced an MCP server that can leverage a managed spark history server to provide tuning recommendations for Databricks jobs, to allow for more seamless local development. 

In part four of this blog series we will demonstrate how to leverage custom MCP servers hosted as Databricks Apps to provide comprehensive tuning recommendations for both EMR spark jobs and OSS spark jobs using Genie Code as an interface. This approach is particularly valuable when determining which spark jobs would experience the highest ROI from a cost/ performance perspective if they were executed on Databricks runtime instead. 

## Spark log parser MCP Server

Implementation steps (estimated < 30 minutes)

- Create Databricks App to host [custom MCP server](https://github.com/tjcarter2/spark-observability-agent/tree/main/GenieCode/DatabricksApps/mcp)
- Create service credential using an IAM role that has read access to where raw spark logs are stored/ access to subset of EMR APIs

You can now incorporate the hosted MCP server into Genie Code using the [instructions here](https://docs.databricks.com/aws/en/genie-code/mcp#add-mcp-servers-to-genie-code-manually). Some example questions that Genie Code can address include:

i) Hi I need help tuning a spark job. Please use MCP server spark-log-parser to fetch sql execution metrics for input path {{path}} using credential name {{credname}} and provide tuning recommendations, thanks!

ii) i) Hi I need help tuning a spark job. Please use MCP server emr-parser to fetch sql execution metrics for cluster ARN {{emrClusterArn}} using credential name {{credname}} and provide tuning recommendations, thanks!

ii) Based on the execution metrics, what % of agg runtime would you expect to fall within photon? What cost/perf improvements should be expected from leveraging photon rather than EMR/OSS?


### Create Databricks App to host custom MCP server:

Manual approach:
- Create a git folder (https://docs.databricks.com/aws/en/repos/) that points to this repo
- Create Databricks App via the UI
- Update the Databricks App to have source pointing to the appropriate MCP within the GenieCode directory

Automated approach:
- Download this repo locally
- Synch local repo to Databricks workspace
- Deploy to Databricks apps using CLI or Terraform with command like below
- databricks apps deploy mcp-emr-parser --source-code-path /Workspace/{{desiredpath}}


### Create service credential using an IAM role that has read access to where raw spark logs are stored/ access to subset of EMR APIs

- Create IAM role with appropriate permissions via Terraform using [aws_iam_role resource](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role). Apporpriate trust policy and permissions can be found in helper repo here.

- Create service credential using [databricks_credential resource](https://registry.terraform.io/providers/databricks/databricks/latest/docs/resources/credential)
