from openkernelforge.reports.failure_taxonomy import (
    CORRECT_BUT_SLOW,
    NUMERICAL_MISMATCH,
    POLICY_REJECTED_TORCH_FALLBACK,
    classify_candidate_record,
)


def test_failure_taxonomy_policy_rejection():
    record = {
        "policy_passed": False,
        "policy_rejection_reason": "obvious_torch_fallback:direct_add",
        "verification_passed": False,
    }
    result = classify_candidate_record(record)
    assert result.failure_type == POLICY_REJECTED_TORCH_FALLBACK
    assert result.suggested_dataset_split == "rejected"


def test_failure_taxonomy_correctness_mismatch():
    record = {
        "policy_passed": True,
        "verification_passed": False,
        "verification_summary": {"first_error_type": "values_not_close"},
    }
    result = classify_candidate_record(record)
    assert result.failure_type == NUMERICAL_MISMATCH
    assert result.suggested_dataset_split == "repair"


def test_failure_taxonomy_correct_but_slow():
    record = {
        "policy_passed": True,
        "verification_passed": True,
        "benchmark_summary": {"speedup_vs_eager": 0.5},
    }
    result = classify_candidate_record(record)
    assert result.failure_type == CORRECT_BUT_SLOW
    assert result.suggested_dataset_split == "optimization"
