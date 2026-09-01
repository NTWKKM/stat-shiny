from utils.state_machine import (
    ResultState,
    compute_input_fingerprint,
    get_result_state,
    render_stale_warning_banner,
    render_state_badge,
)


def test_compute_input_fingerprint_deterministic():
    p1 = {"outcome": "dead", "covariates": ["age", "sex"], "firth": True}
    p2 = {"firth": True, "covariates": ["sex", "age"], "outcome": "dead"}
    assert compute_input_fingerprint(p1) == compute_input_fingerprint(p2)


def test_compute_input_fingerprint_changes():
    p1 = {"outcome": "dead", "covariates": ["age", "sex"]}
    p2 = {"outcome": "dead", "covariates": ["age", "sex", "bmi"]}
    assert compute_input_fingerprint(p1) != compute_input_fingerprint(p2)


def test_get_result_state_empty():
    state = get_result_state(
        has_run=False,
        is_computing=False,
        current_fingerprint=None,
        last_run_fingerprint=None,
        has_required_inputs=False,
    )
    assert state == ResultState.EMPTY


def test_get_result_state_configuring():
    state = get_result_state(
        has_run=False,
        is_computing=False,
        current_fingerprint="abc",
        last_run_fingerprint=None,
        has_required_inputs=True,
    )
    assert state == ResultState.CONFIGURING


def test_get_result_state_computing():
    state = get_result_state(
        has_run=True,
        is_computing=True,
        current_fingerprint="abc",
        last_run_fingerprint="abc",
    )
    assert state == ResultState.COMPUTING


def test_get_result_state_fresh():
    state = get_result_state(
        has_run=True,
        is_computing=False,
        current_fingerprint="abc",
        last_run_fingerprint="abc",
    )
    assert state == ResultState.FRESH


def test_get_result_state_stale():
    state = get_result_state(
        has_run=True,
        is_computing=False,
        current_fingerprint="xyz",
        last_run_fingerprint="abc",
    )
    assert state == ResultState.STALE


def test_compute_input_fingerprint_nested_dict():
    p1 = {"model": {"penalty": "l2", "alpha": 0.1, "nested": {"b": 2, "a": 1}}}
    p2 = {"model": {"nested": {"a": 1, "b": 2}, "alpha": 0.1, "penalty": "l2"}}
    assert compute_input_fingerprint(p1) == compute_input_fingerprint(p2)


def test_get_result_state_missing_required_inputs_after_run():
    # If user has run the model, but clears required inputs (has_required_inputs=False),
    # it must return EMPTY rather than FRESH
    state = get_result_state(
        has_run=True,
        is_computing=False,
        current_fingerprint=None,
        last_run_fingerprint=None,
        has_required_inputs=False,
    )
    assert state == ResultState.EMPTY


def test_get_result_state_missing_required_inputs_with_differing_fingerprints():
    # Missing required inputs must take precedence over STALE state even when fingerprints differ
    state = get_result_state(
        has_run=True,
        is_computing=False,
        current_fingerprint="new_fp_after_clearing_var",
        last_run_fingerprint="old_fp_when_run",
        has_required_inputs=False,
    )
    assert state == ResultState.EMPTY


def test_render_state_badge():
    fresh_badge = render_state_badge(ResultState.FRESH)
    assert "Synchronized" in str(fresh_badge)
    stale_badge = render_state_badge(ResultState.STALE)
    assert "Inputs Changed" in str(stale_badge)


def test_render_stale_warning_banner():
    banner = render_stale_warning_banner()
    assert "Inputs Modified" in str(banner)
