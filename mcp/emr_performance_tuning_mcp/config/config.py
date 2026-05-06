"""Server configuration for Spark History MCP."""

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Configuration for the Spark History MCP Server."""

    emr_cluster_arn: str = Field(
        ...,
        description="ARN of the EMR cluster to connect to",
    )
    timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
    )
    aws_region: Optional[str] = Field(
        default=None,
        description="AWS region (extracted from ARN if not provided)",
    )
    aws_profile: Optional[str] = Field(
        default=None,
        description="AWS profile name for authentication",
    )

    model_config = {
        "env_prefix": "SPARK_HISTORY_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    def get_region(self) -> str:
        """Extract region from ARN or return configured region."""
        if self.aws_region:
            return self.aws_region
        # Extract region from ARN: arn:aws:elasticmapreduce:REGION:account:cluster/id
        return self.emr_cluster_arn.split(":")[3]


class SparkAppConfig(BaseModel):
    """Configuration for a specific Spark application query."""

    app_id: Optional[str] = Field(
        default=None,
        description="Specific application ID to query (None for all)",
    )
    include_sql: bool = Field(
        default=True,
        description="Include SQL query metrics",
    )
    include_jobs: bool = Field(
        default=True,
        description="Include job-level metrics",
    )
    include_stages: bool = Field(
        default=True,
        description="Include stage-level metrics",
    )
    include_executors: bool = Field(
        default=True,
        description="Include executor metrics",
    )
    include_environment: bool = Field(
        default=False,
        description="Include Spark environment/configuration",
    )
