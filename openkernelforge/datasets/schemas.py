"""Simple dataset schema constants."""

DATASET_FILES = [
    "sft_raw.jsonl",
    "repair.jsonl",
    "optimization.jsonl",
    "rejected.jsonl",
]

BASE_REQUIRED_FIELDS = [
    "task_id",
    "task_name",
    "task_description",
    "prompt",
    "raw_response",
    "candidate_code",
    "policy_result",
    "verification_result",
    "benchmark_result",
    "failure_type",
    "short_reason",
    "source_run_dir",
    "prompt_path",
    "response_path",
    "candidate_path",
    "created_at",
    "target_type",
]
