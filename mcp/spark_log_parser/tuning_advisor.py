"""
Spark Tuning Advisor

Provides comprehensive tuning recommendations based on parsed Spark metrics,
including task-level summary analysis for skew, partition sizing, and throughput.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TuningRecommendation:
    category: str
    severity: str  # "high", "medium", "low", "info"
    title: str
    description: str
    current_value: str
    recommended_action: str
    config_suggestion: Optional[str] = None


class SparkTuningAdvisor:
    """Analyzes Spark metrics and provides tuning recommendations."""

    def __init__(
        self,
        job_metrics: list[dict],
        stage_metrics: list[dict],
        sql_metrics: list[dict],
        task_summary_metrics: list[dict] = None,
    ):
        self.job_metrics = job_metrics
        self.stage_metrics = stage_metrics
        self.sql_metrics = sql_metrics
        self.task_summary_metrics = task_summary_metrics or []
        self.recommendations: list[TuningRecommendation] = []

    def analyze(self) -> list[TuningRecommendation]:
        """Run all analyses and return recommendations."""
        self.recommendations = []

        self._analyze_job_failures()
        self._analyze_stage_failures()
        self._analyze_task_skew()
        self._analyze_spill()
        self._analyze_shuffle()
        self._analyze_data_skew()
        self._analyze_sql_performance()
        self._analyze_parallelism()

        # Task summary-based analyses
        if self.task_summary_metrics:
            self._analyze_task_duration_skew()
            self._analyze_partition_size()
            self._analyze_processing_rate()

        return self.recommendations

    def _analyze_job_failures(self) -> None:
        """Check for job failures."""
        failed_jobs = [j for j in self.job_metrics if j.get("status") != "JobSucceeded"]
        total_jobs = len(self.job_metrics)

        if failed_jobs:
            failure_rate = len(failed_jobs) / total_jobs * 100 if total_jobs > 0 else 0
            self.recommendations.append(
                TuningRecommendation(
                    category="Job Reliability",
                    severity="high" if failure_rate > 10 else "medium",
                    title="Job Failures Detected",
                    description=f"{len(failed_jobs)} out of {total_jobs} jobs failed ({failure_rate:.1f}% failure rate).",
                    current_value=f"{len(failed_jobs)} failed jobs",
                    recommended_action="Review failed job logs for root cause. Common issues include OOM errors, data skew, or external service failures.",
                )
            )

    def _analyze_stage_failures(self) -> None:
        """Check for stage failures and retries."""
        failed_stages = [s for s in self.stage_metrics if s.get("status") == "FAILED"]
        retry_stages = [s for s in self.stage_metrics if int(s.get("attemptId", 0)) > 0]

        if failed_stages:
            self.recommendations.append(
                TuningRecommendation(
                    category="Stage Reliability",
                    severity="high",
                    title="Stage Failures Detected",
                    description=f"{len(failed_stages)} stages failed during execution.",
                    current_value=f"{len(failed_stages)} failed stages",
                    recommended_action="Check executor logs for OOM, shuffle fetch failures, or task failures. Consider increasing executor memory or reducing partition sizes.",
                )
            )

        if retry_stages:
            self.recommendations.append(
                TuningRecommendation(
                    category="Stage Reliability",
                    severity="medium",
                    title="Stage Retries Detected",
                    description=f"{len(retry_stages)} stages required retry attempts.",
                    current_value=f"{len(retry_stages)} retried stages",
                    recommended_action="Investigate transient failures. Consider enabling speculation for straggler tasks.",
                    config_suggestion="spark.speculation=true",
                )
            )

    def _analyze_task_skew(self) -> None:
        """Analyze task completion patterns for skew."""
        for stage in self.stage_metrics:
            num_tasks = stage.get("numTasks", 0)
            completed = stage.get("numCompletedTasks", 0)
            failed = stage.get("numFailedTasks", 0)

            if num_tasks > 0 and failed > 0:
                failure_rate = failed / num_tasks * 100
                if failure_rate > 5:
                    self.recommendations.append(
                        TuningRecommendation(
                            category="Task Reliability",
                            severity="high" if failure_rate > 20 else "medium",
                            title=f"High Task Failure Rate in Stage {stage['stageId']}",
                            description=f"{failed}/{num_tasks} tasks failed ({failure_rate:.1f}%).",
                            current_value=f"{failure_rate:.1f}% failure rate",
                            recommended_action="Check for data corruption, OOM errors, or network issues. Consider increasing task retries.",
                            config_suggestion="spark.task.maxFailures=8",
                        )
                    )

    def _analyze_spill(self) -> None:
        """Analyze memory and disk spill metrics."""
        total_memory_spill = sum(s.get("memoryBytesSpilled", 0) for s in self.stage_metrics)
        total_disk_spill = sum(s.get("diskBytesSpilled", 0) for s in self.stage_metrics)

        if total_memory_spill > 0 or total_disk_spill > 0:
            spill_gb = (total_memory_spill + total_disk_spill) / (1024**3)
            severity = "high" if spill_gb > 10 else "medium" if spill_gb > 1 else "low"

            self.recommendations.append(
                TuningRecommendation(
                    category="Memory Management",
                    severity=severity,
                    title="Data Spill Detected",
                    description=f"Total spill: {spill_gb:.2f} GB (Memory: {total_memory_spill / (1024**3):.2f} GB, Disk: {total_disk_spill / (1024**3):.2f} GB).",
                    current_value=f"{spill_gb:.2f} GB spilled",
                    recommended_action="Increase executor memory or reduce partition sizes to avoid spilling to disk.",
                    config_suggestion="spark.executor.memory (increase by 50-100%) or spark.sql.shuffle.partitions (increase to reduce per-partition data)",
                )
            )

    def _analyze_shuffle(self) -> None:
        """Analyze shuffle read/write patterns."""
        total_shuffle_read = sum(s.get("shuffleReadBytes", 0) for s in self.stage_metrics)
        total_shuffle_write = sum(s.get("shuffleWriteBytes", 0) for s in self.stage_metrics)

        shuffle_total_gb = (total_shuffle_read + total_shuffle_write) / (1024**3)

        if shuffle_total_gb > 100:
            self.recommendations.append(
                TuningRecommendation(
                    category="Shuffle Optimization",
                    severity="high",
                    title="Large Shuffle Volume",
                    description=f"Total shuffle data: {shuffle_total_gb:.1f} GB (Read: {total_shuffle_read / (1024**3):.1f} GB, Write: {total_shuffle_write / (1024**3):.1f} GB).",
                    current_value=f"{shuffle_total_gb:.1f} GB shuffle",
                    recommended_action="Consider reducing shuffle by using broadcast joins for small tables, or enabling adaptive query execution.",
                    config_suggestion="spark.sql.adaptive.enabled=true; spark.sql.adaptive.coalescePartitions.enabled=true",
                )
            )
        elif shuffle_total_gb > 10:
            self.recommendations.append(
                TuningRecommendation(
                    category="Shuffle Optimization",
                    severity="info",
                    title="Moderate Shuffle Volume",
                    description=f"Total shuffle data: {shuffle_total_gb:.1f} GB.",
                    current_value=f"{shuffle_total_gb:.1f} GB shuffle",
                    recommended_action="Monitor shuffle performance. Consider AQE for automatic optimization.",
                    config_suggestion="spark.sql.adaptive.enabled=true",
                )
            )

    def _analyze_data_skew(self) -> None:
        """Analyze data skew indicators from SQL metrics."""
        for sql in self.sql_metrics:
            for node in sql.get("nodes", []):
                node_name = node.get("nodeName", "")
                metrics = {m["name"]: m["value"] for m in node.get("metrics", [])}

                if "SortMergeJoin" in node_name or "ShuffledHashJoin" in node_name:
                    skew_indicator = metrics.get("Cancellation Is Skewed", "0")
                    if skew_indicator != "0":
                        self.recommendations.append(
                            TuningRecommendation(
                                category="Data Skew",
                                severity="high",
                                title=f"Join Skew Detected in SQL {sql['id']}",
                                description=f"Skew detected in {node_name} (Node {node['nodeId']}).",
                                current_value="Skewed join detected",
                                recommended_action="Consider salting keys, using broadcast join, or enabling AQE skew join optimization.",
                                config_suggestion="spark.sql.adaptive.skewJoin.enabled=true",
                            )
                        )

    def _analyze_sql_performance(self) -> None:
        """Analyze SQL execution performance."""
        long_queries = []
        failed_queries = []

        for sql in self.sql_metrics:
            duration_sec = sql.get("duration", 0) / 1000
            status = sql.get("status", "")

            if status == "FAILED":
                failed_queries.append(sql)
            elif duration_sec > 300:  # > 5 minutes
                long_queries.append((sql, duration_sec))

        if failed_queries:
            self.recommendations.append(
                TuningRecommendation(
                    category="SQL Reliability",
                    severity="high",
                    title="Failed SQL Queries",
                    description=f"{len(failed_queries)} SQL queries failed.",
                    current_value=f"{len(failed_queries)} failed queries",
                    recommended_action="Review query plans and error messages. Check for data issues or resource constraints.",
                )
            )

        if long_queries:
            avg_duration = sum(d for _, d in long_queries) / len(long_queries)
            self.recommendations.append(
                TuningRecommendation(
                    category="SQL Performance",
                    severity="medium",
                    title="Long Running Queries",
                    description=f"{len(long_queries)} queries took over 5 minutes (avg: {avg_duration:.0f}s).",
                    current_value=f"Avg duration: {avg_duration:.0f}s",
                    recommended_action="Review execution plans for optimization opportunities. Consider partitioning, caching, or query restructuring.",
                )
            )

    def _analyze_parallelism(self) -> None:
        """Analyze parallelism and partition settings."""
        task_counts = [s.get("numTasks", 0) for s in self.stage_metrics]

        if task_counts:
            min_tasks = min(task_counts)
            max_tasks = max(task_counts)
            avg_tasks = sum(task_counts) / len(task_counts)

            if max_tasks > 10000:
                self.recommendations.append(
                    TuningRecommendation(
                        category="Parallelism",
                        severity="medium",
                        title="Very High Task Count",
                        description=f"Maximum task count: {max_tasks}. This may cause scheduler overhead.",
                        current_value=f"Max: {max_tasks}, Avg: {avg_tasks:.0f}",
                        recommended_action="Consider reducing partition count for stages with very high task counts.",
                        config_suggestion="spark.sql.shuffle.partitions (reduce if > 2000)",
                    )
                )

            if min_tasks < 10 and max_tasks > 1000:
                self.recommendations.append(
                    TuningRecommendation(
                        category="Parallelism",
                        severity="low",
                        title="Variable Parallelism",
                        description=f"Task counts vary significantly (min: {min_tasks}, max: {max_tasks}).",
                        current_value=f"Range: {min_tasks} - {max_tasks}",
                        recommended_action="Enable AQE to dynamically adjust partitions based on data size.",
                        config_suggestion="spark.sql.adaptive.enabled=true; spark.sql.adaptive.coalescePartitions.enabled=true",
                    )
                )

    # =========================================================================
    # Task Summary-Based Analyses (using per-stage quantile distributions)
    # =========================================================================

    def _analyze_task_duration_skew(self) -> None:
        """Detect task-level data skew based on task duration distribution.
        
        Flags stages where:
        - Median (p50) task duration > 2x the min (p5) task duration
          (indicates some partitions have significantly more data)
        - Max (p95) task duration > 2x the median (p50) task duration
          (indicates straggler tasks / heavy partitions at the tail)
        """
        for summary in self.task_summary_metrics:
            stage_id = summary.get("stageId", "?")
            attempt_id = summary.get("attemptId", "0")
            num_tasks = summary.get("numTasks", 0)
            duration_quantiles = summary.get("duration", [])

            if len(duration_quantiles) < 5 or num_tasks < 10:
                continue

            # quantiles: [p5, p25, p50, p75, p95]
            p5 = duration_quantiles[0]   # min proxy
            p25 = duration_quantiles[1]
            p50 = duration_quantiles[2]  # median
            p75 = duration_quantiles[3]
            p95 = duration_quantiles[4]  # max proxy

            # Check median vs min skew: p50 > 2 * p5
            if p5 > 0 and p50 > 2 * p5:
                skew_ratio = p50 / p5
                self.recommendations.append(
                    TuningRecommendation(
                        category="Task Duration Skew",
                        severity="high" if skew_ratio > 10 else "medium",
                        title=f"Task Duration Skew (Median vs Min) in Stage {stage_id}",
                        description=(
                            f"Stage {stage_id} (attempt {attempt_id}, {num_tasks} tasks): "
                            f"Median task duration ({p50:.0f}ms) is {skew_ratio:.1f}x the p5 duration ({p5:.0f}ms). "
                            f"Distribution: p5={p5:.0f}ms, p25={p25:.0f}ms, p50={p50:.0f}ms, p75={p75:.0f}ms, p95={p95:.0f}ms."
                        ),
                        current_value=f"Median/Min ratio: {skew_ratio:.1f}x",
                        recommended_action=(
                            "Data is unevenly distributed across partitions. "
                            "Consider repartitioning by a more evenly distributed key, "
                            "enabling AQE skew join handling, or salting skewed keys."
                        ),
                        config_suggestion="spark.sql.adaptive.skewJoin.enabled=true; spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes=256m",
                    )
                )

            # Check tail skew: p95 > 2 * p50
            if p50 > 0 and p95 > 2 * p50:
                tail_ratio = p95 / p50
                self.recommendations.append(
                    TuningRecommendation(
                        category="Task Duration Skew",
                        severity="high" if tail_ratio > 5 else "medium",
                        title=f"Straggler Tasks (Tail Skew) in Stage {stage_id}",
                        description=(
                            f"Stage {stage_id} (attempt {attempt_id}, {num_tasks} tasks): "
                            f"p95 task duration ({p95:.0f}ms) is {tail_ratio:.1f}x the median ({p50:.0f}ms). "
                            f"The slowest tasks are significantly lagging behind the majority."
                        ),
                        current_value=f"p95/Median ratio: {tail_ratio:.1f}x",
                        recommended_action=(
                            "Tail tasks are processing disproportionately large partitions. "
                            "Consider increasing partition count, enabling speculation for stragglers, "
                            "or investigating data skew in join/group-by keys."
                        ),
                        config_suggestion="spark.speculation=true; spark.sql.shuffle.partitions (increase to reduce per-partition data)",
                    )
                )

    def _analyze_partition_size(self) -> None:
        """Detect excessive partition sizes based on task input metrics.
        
        Flags stages where the p75 or p95 input bytes per task exceeds 512 MB,
        indicating that individual partitions are too large for efficient processing.
        """
        PARTITION_SIZE_THRESHOLD_BYTES = 512 * 1024 * 1024  # 512 MB

        for summary in self.task_summary_metrics:
            stage_id = summary.get("stageId", "?")
            attempt_id = summary.get("attemptId", "0")
            num_tasks = summary.get("numTasks", 0)

            if num_tasks < 2:
                continue

            # Check input bytes (for read-heavy stages)
            input_metrics = summary.get("inputMetrics", {})
            input_bytes_quantiles = input_metrics.get("bytesRead", [])

            # Check shuffle read bytes (for shuffle-heavy stages)
            shuffle_read_metrics = summary.get("shuffleReadMetrics", {})
            shuffle_read_quantiles = shuffle_read_metrics.get("readBytes", [])

            # Evaluate input partition sizes
            if len(input_bytes_quantiles) >= 5:
                p75_input = input_bytes_quantiles[3]
                p95_input = input_bytes_quantiles[4]
                p50_input = input_bytes_quantiles[2]

                if p95_input > PARTITION_SIZE_THRESHOLD_BYTES:
                    p95_mb = p95_input / (1024 * 1024)
                    p50_mb = p50_input / (1024 * 1024)
                    self.recommendations.append(
                        TuningRecommendation(
                            category="Partition Sizing",
                            severity="high" if p95_input > 1024 * 1024 * 1024 else "medium",
                            title=f"Excessive Input Partition Size in Stage {stage_id}",
                            description=(
                                f"Stage {stage_id} (attempt {attempt_id}, {num_tasks} tasks): "
                                f"p95 input size per task is {p95_mb:.0f} MB (median: {p50_mb:.0f} MB). "
                                f"Partitions exceeding 512 MB can cause GC pressure, spill, and OOM risks."
                            ),
                            current_value=f"p95 input: {p95_mb:.0f} MB/task",
                            recommended_action=(
                                "Increase the number of partitions to reduce per-task data volume. "
                                "For file scans, consider setting spark.sql.files.maxPartitionBytes. "
                                "For shuffles, increase spark.sql.shuffle.partitions."
                            ),
                            config_suggestion="spark.sql.files.maxPartitionBytes=128m; spark.sql.shuffle.partitions (increase 2-4x)",
                        )
                    )
                elif p75_input > PARTITION_SIZE_THRESHOLD_BYTES:
                    p75_mb = p75_input / (1024 * 1024)
                    self.recommendations.append(
                        TuningRecommendation(
                            category="Partition Sizing",
                            severity="medium",
                            title=f"Large Input Partitions in Stage {stage_id}",
                            description=(
                                f"Stage {stage_id}: p75 input size per task is {p75_mb:.0f} MB. "
                                f"A significant portion of tasks are processing partitions near or above 512 MB."
                            ),
                            current_value=f"p75 input: {p75_mb:.0f} MB/task",
                            recommended_action=(
                                "Consider reducing max partition bytes or increasing partition count to keep "
                                "task-level data below 256-512 MB for optimal memory utilization."
                            ),
                            config_suggestion="spark.sql.files.maxPartitionBytes=128m",
                        )
                    )

            # Evaluate shuffle partition sizes
            if len(shuffle_read_quantiles) >= 5:
                p75_shuffle = shuffle_read_quantiles[3]
                p95_shuffle = shuffle_read_quantiles[4]
                p50_shuffle = shuffle_read_quantiles[2]

                if p95_shuffle > PARTITION_SIZE_THRESHOLD_BYTES:
                    p95_mb = p95_shuffle / (1024 * 1024)
                    p50_mb = p50_shuffle / (1024 * 1024)
                    self.recommendations.append(
                        TuningRecommendation(
                            category="Partition Sizing",
                            severity="high" if p95_shuffle > 1024 * 1024 * 1024 else "medium",
                            title=f"Excessive Shuffle Partition Size in Stage {stage_id}",
                            description=(
                                f"Stage {stage_id} (attempt {attempt_id}, {num_tasks} tasks): "
                                f"p95 shuffle read per task is {p95_mb:.0f} MB (median: {p50_mb:.0f} MB). "
                                f"Large shuffle partitions indicate skewed keys or insufficient partition count."
                            ),
                            current_value=f"p95 shuffle read: {p95_mb:.0f} MB/task",
                            recommended_action=(
                                "Increase spark.sql.shuffle.partitions, enable AQE coalescing, "
                                "or investigate skewed join/group-by keys."
                            ),
                            config_suggestion="spark.sql.shuffle.partitions (increase 2-4x); spark.sql.adaptive.enabled=true",
                        )
                    )

    def _analyze_processing_rate(self) -> None:
        """Detect slow task processing rates.
        
        Flags stages where the effective processing rate (input MB / duration seconds)
        falls below 1 MB/second at the median, indicating:
        - Inefficient transformations or UDFs
        - Excessive CPU overhead relative to data volume
        - Network or I/O bottlenecks
        
        Only analyzes stages with >= MIN_TASKS tasks and median duration >= 1 second
        to avoid noisy alerts from trivially small or short-lived stages.
        """
        RATE_THRESHOLD_MB_PER_SEC = 1.0  # 1 MB/s minimum expected throughput
        MIN_TASKS = 50  # Minimum task count for statistically meaningful rate analysis
        MIN_MEDIAN_DURATION_MS = 1000  # Ignore stages with median task duration < 1s

        for summary in self.task_summary_metrics:
            stage_id = summary.get("stageId", "?")
            attempt_id = summary.get("attemptId", "0")
            num_tasks = summary.get("numTasks", 0)
            duration_quantiles = summary.get("duration", [])

            if num_tasks < MIN_TASKS or len(duration_quantiles) < 5:
                continue

            # Get median duration (ms) and input bytes
            p50_duration_ms = duration_quantiles[2]
            if p50_duration_ms < MIN_MEDIAN_DURATION_MS:
                continue

            p50_duration_sec = p50_duration_ms / 1000.0

            # Determine data volume: prefer input bytes, fall back to shuffle read
            input_metrics = summary.get("inputMetrics", {})
            input_bytes_quantiles = input_metrics.get("bytesRead", [])
            shuffle_read_metrics = summary.get("shuffleReadMetrics", {})
            shuffle_read_quantiles = shuffle_read_metrics.get("readBytes", [])

            p50_input_bytes = 0
            data_source = ""
            if len(input_bytes_quantiles) >= 5 and input_bytes_quantiles[2] > 0:
                p50_input_bytes = input_bytes_quantiles[2]
                data_source = "input"
            elif len(shuffle_read_quantiles) >= 5 and shuffle_read_quantiles[2] > 0:
                p50_input_bytes = shuffle_read_quantiles[2]
                data_source = "shuffle read"

            if p50_input_bytes <= 0:
                continue

            p50_input_mb = p50_input_bytes / (1024 * 1024)
            processing_rate_mb_s = p50_input_mb / p50_duration_sec

            if processing_rate_mb_s < RATE_THRESHOLD_MB_PER_SEC:
                self.recommendations.append(
                    TuningRecommendation(
                        category="Processing Rate",
                        severity="high" if processing_rate_mb_s < 0.1 else "medium",
                        title=f"Low Processing Rate in Stage {stage_id}",
                        description=(
                            f"Stage {stage_id} (attempt {attempt_id}, {num_tasks} tasks): "
                            f"Median processing rate is {processing_rate_mb_s:.3f} MB/s "
                            f"({data_source}: {p50_input_mb:.1f} MB median / {p50_duration_sec:.1f}s median duration). "
                            f"Expected minimum: {RATE_THRESHOLD_MB_PER_SEC} MB/s."
                        ),
                        current_value=f"{processing_rate_mb_s:.3f} MB/s",
                        recommended_action=(
                            "Tasks are processing data too slowly. Potential causes: "
                            "expensive UDFs or transformations, Python UDFs (consider Pandas UDFs), "
                            "excessive object creation causing GC pressure, or I/O throttling. "
                            "Profile the workload to identify CPU-bound operations."
                        ),
                        config_suggestion="Consider Photon engine, vectorized Pandas UDFs, or reducing transformation complexity",
                    )
                )

    def get_summary(self) -> dict:
        """Get a summary of all recommendations."""
        by_severity = {"high": [], "medium": [], "low": [], "info": []}
        by_category = {}

        for rec in self.recommendations:
            by_severity[rec.severity].append(rec)
            if rec.category not in by_category:
                by_category[rec.category] = []
            by_category[rec.category].append(rec)

        return {
            "total_recommendations": len(self.recommendations),
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "by_category": {k: len(v) for k, v in by_category.items()},
            "high_priority": [
                {"title": r.title, "description": r.description, "action": r.recommended_action}
                for r in by_severity["high"]
            ],
        }

    def format_recommendations(self) -> str:
        """Format recommendations as a readable report."""
        if not self.recommendations:
            return "No tuning recommendations. Your Spark job appears well-optimized!"

        lines = ["# Spark Tuning Recommendations\n"]

        by_severity = {"high": [], "medium": [], "low": [], "info": []}
        for rec in self.recommendations:
            by_severity[rec.severity].append(rec)

        severity_labels = {
            "high": "🔴 HIGH PRIORITY",
            "medium": "🟡 MEDIUM PRIORITY",
            "low": "🟢 LOW PRIORITY",
            "info": "ℹ️ INFORMATIONAL",
        }

        for severity in ["high", "medium", "low", "info"]:
            recs = by_severity[severity]
            if recs:
                lines.append(f"\n## {severity_labels[severity]}\n")
                for rec in recs:
                    lines.append(f"### {rec.title}")
                    lines.append(f"**Category:** {rec.category}")
                    lines.append(f"**Current:** {rec.current_value}")
                    lines.append(f"\n{rec.description}\n")
                    lines.append(f"**Recommendation:** {rec.recommended_action}")
                    if rec.config_suggestion:
                        lines.append(f"\n**Suggested Config:**\n```\n{rec.config_suggestion}\n```")
                    lines.append("")

        return "\n".join(lines)
