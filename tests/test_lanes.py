from src.triagelane_ct.lanes import assign_lane


def test_critical_lane():
    result = assign_lane(
        0.90,
        0.70,
        "epidural",
    )

    assert result["lane"] == "critical"
    assert result["abstain"] is False


def test_urgent_lane():
    result = assign_lane(
        0.71,
        0.60,
        "epidural",
    )

    assert result["lane"] == "urgent"
    assert result["abstain"] is False


def test_expedited_lane():
    result = assign_lane(
        0.50,
        0.40,
        "epidural",
    )

    assert result["lane"] == "expedited"
    assert result["abstain"] is False


def test_routine_lane():
    result = assign_lane(
        0.20,
        0.20,
        "epidural",
    )

    assert result["lane"] == "routine"
    assert result["abstain"] is False


def test_boundary_abstention():
    result = assign_lane(
        0.85,
        0.70,
        "epidural",
    )

    assert result["abstain"] is True
    assert result["in_abstention_tray"] is True