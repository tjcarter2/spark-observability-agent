## Key Features
- Discovers EMR or Databricks clusters and reads Spark history data.
- Extracts and processes application, job, stage, and SQL query metrics.
- Produces Spark DataFrames for further analysis and optional S3 output.
- Creates LLM tools to facilitate performance optimization via natural language interface with frontier model.
- Designed for scalable, multi-cluster profiling to aid performance tuning.

## Spark Observability Agent

The ETL scripts and LLM tools/UDFs in this package consolidate and expose spark history server metric data so that they can be leveraged by traditional SQL analysts and LLMs. There are two different frameworks for leveraging this solution. The first framework entails leveraging the tools defined in AgentDDLProd to ‘live fetch’ spark history server metrics for a specific spark cluster. This framework is preferable if you want to conduct deep dive analysis for a small number of spark jobs via a natural language interface with a frontier model.  The second framework entails running an ETL script on a consistent cadence and analyzing the spark history server metric data with traditional SQL analysis or a Databricks Genie Room. This framework is preferable if you want to analyze performance across dozens or hundreds of spark jobs.

### Framework One– LLM Tools + live data fetching 

Implementation steps (estimated about 15 minutes for secret creation and helper script execution)

- Create secrets in your databricks workspace for token, workspace URL, dataplane URL, and cookies (if you want to live fetch spark history server metrics for Databricks spark jobs, feel free to use shsutils helper)
- Run agentconnprod and agentddlprod

<img width="1073" height="504" alt="shsutils" src="https://github.com/user-attachments/assets/1313520b-f9cc-4bbe-ae81-77b3d26413b7" />

After implementation steps are complete you should see the following tools/UDFs within your sink schema:
- Getappid, getexecutor, getslowestjobs, getslowestsql, getsloweststages, getsparkcontext, getstage, listappsraw, listshsenvraw, listshsexeuctorsraw, listshsjobsraw, listshssqlraw, listshsstagesraw, listshstasksraw

You can now reference these tools via Databricks AI playground, or some other open source interface. Some example questions the frontier models can address include:

i) What stages are causing bottlenecks for cluster_id {{cluster_id}}?

ii) What sql queries are causing bottlenecks for cluster_id {{cluster_id}}?

iii) What spark configs did I leverage for cluster_id {{cluster_id}}?

<img width="938" height="705" alt="pguno" src="https://github.com/user-attachments/assets/942bd3a5-62c7-483e-b148-a391ae862841" />

<img width="879" height="702" alt="pgdos" src="https://github.com/user-attachments/assets/0c18beef-ada9-4ec7-865b-db8d5c51df64" />

<img width="961" height="700" alt="pgtres" src="https://github.com/user-attachments/assets/51f0e244-5670-4de5-bd96-1087a3a28913" />

### Framework Two– Scalable ETL + Genie

Implementation steps (estimated about 15 minutes for secret creation and then minutes to hours for the ETL depending on how many terminated spark clusters are associated with your workspace)

- Create secrets in your databricks workspace for token, workspace URL, dataplane URL, and cookies (if you want to extract spark history server metrics for Databricks spark jobs, feel free to use shsutils helper)
- Run databricks_spark_profiler or emr_spark_profiler (depending on which system you want to extract spark history server metric from)
- Run efficiency_analysis
- Run emr_photon_analysis (if you want to determine which EMR jobs are most likely to benefit from photon)

After implementation steps are complete you should see the following tables within your sink schema:
- Applications, cluster_summaries, executors, ineffjobagg, ineffjobraw, jobs, photonanalysis, sql, stages, task_summaries, tasks

You can now reference these tables for traditional SQL analysis. Further you can create a genie room that references the tables in the sink schema for text to SQL analysis. Some example questions the Genie can address include:

i) What are my most inefficient spark jobs?

ii) What stages are causing bottlenecks for cluster_id {{cluster_id}}?

iii) What sql queries are causing bottlenecks for cluster_id {{cluster_id}}?

<img width="1257" height="619" alt="genieuno" src="https://github.com/user-attachments/assets/bc6861f3-f209-40b8-8733-7e98d7c25306" />

<img width="1258" height="628" alt="geniedos" src="https://github.com/user-attachments/assets/2b1e1662-3db8-4d1b-83ed-a21986ef5545" />

<img width="1268" height="545" alt="genietres" src="https://github.com/user-attachments/assets/a16c8e6e-c557-4c2f-9767-376888095456" />

## Databricks Authentication Setup

You'll need to create secrets for four values to authenticate with Databricks.

### 1. Access Token & Workspace URL

- Generate a Databricks **personal access token (PAT)** from your workspace settings (`User Settings > Developer > Access Tokens`).
- Create a secret for the **access token**.
- Create a secret for your **workspace URL** (e.g., `https://<your-workspace>.cloud.databricks.com`).

### 2. Data Plane URL

1. Navigate to any **completed job** in your workspace.
2. Open the **Spark UI** and click **"Open in new tab"**.
3. Copy the URL from the browser address bar — it should contain `dp-` in the hostname.
4. Create a secret for this URL - ending with `.com`.

### 3. Data Plane Auth Cookie (`DATAPLANE_DOMAIN_DBAUTH`)

1. From the same Spark UI tab opened in Step 2:
   - Open browser **DevTools** (`F12` or `Cmd+Option+I`).
   - Go to the **Application** tab → **Cookies**.
   - Find and copy the value of the `DATAPLANE_DOMAIN_DBAUTH` cookie.
2. Create a secret for this value.

<img width="2559" height="1367" alt="AuthInstructions" src="https://github.com/user-attachments/assets/03b5ea74-e6bf-481b-bf17-7e3db3f0ea7e" />

## Configurable Environment Variables Databricks ETL
- **`timeout_seconds`**: Timeout for requests (default: `300`).
- **`max_applications`**: Maximum number of applications to analyze per cluster.
- **`environment`**: Set to `dev` or `prod`.
- **`catalog_name`**: Catalog (required).
- **`schema_name`**: Set explictly, or default is spark_observability.
- **`volume_name`**: Set explictly, or default is profiler_logs_volume.
- **`max_clusters`**: Maximum number of clusters to analyze (default: `5`).
- **`batch_size`**: Clusters to process concurrently (default `10`)
- **`batch_delay_seconds`**: Delay Between Batches (default `2`).
- **`max_endpoint_failures`**: Max Endpoint Failures per Endpoint Type (default `3`).
- **`include_tasks`**: Set to true if you want to include task level metrics (defult false).

Note that for our own internal runs, we set env to prod and keep all the defaults.

## AWS IAM Permissions
*   **elasticmapreduce**:
    *   `ListClusters`
    *   `DescribeCluster`
    *   `ListSteps`
    *   `DescribeStep`
    *   `CreatePersistentAppUI`
    *   `GetPersistentAppUIPresignedURL`
    *   `ListInstanceGroups`
    *   `ListInstanceFleets`
*   **s3**:
    *   `PutObject`
    *   `GetObject`
    *   `ListBucket`
*   **sts**:
    *   `GetCallerIdentity`

## Configurable Environment Variables EMR ETL
- **`aws_region`**: AWS region (e.g., `us-east-1`).
- **`emr_cluster_arn`**: Specify a cluster ARN, or leave blank to auto-discover.
- **`timeout_seconds`**: Timeout for requests (default: `300`).
- **`max_applications`**: Maximum number of applications to analyze per cluster.
- **`environment`**: Set to `dev` or `prod`.
- **`s3_output_path`**: S3 path for results (required in `prod` environment).
- **`cluster_states`**: Cluster states to analyze (e.g., `TERMINATED`, `WAITING`, both, or `ALL`).
- **`cluster_name_filter`**: Optional substring filter for cluster names.
- **`max_clusters`**: Maximum number of clusters to analyze (default: `5`).
- **`created_after_date` / `created_before_date`**: Optional date range filter (format: `YYYY-MM-DD`).
- **`persistent_ui_timeout_seconds`**: Timeout for the persistent UI to become ready.
- **`max_cluster_threads`**: Maximum number of concurrent clusters to analyze.
- **`max_app_threads`**: Maximum number of concurrent application analyses per cluster.

## EMR specific caveat

For best accuracy and performance for the EMR spark observability profiler, we recommend settting the spark configs AWS recommends here (https://docs.aws.amazon.com/emr/latest/ManagementGuide/app-history-spark-UI.html) to allow for efficient parsing of large event logs. Also note that the persistent spark UI can only be leveraged when raw spark logs are written to HDFS (the default behavior for EMR). If they are written to s3 or elsewhere, the script will not work. 

## Contribution Guide

TBD


