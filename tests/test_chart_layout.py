"""Chart labels remain horizontal without changing the displayed observations."""
import pandas as pd
import pytest

from app import activity_chart, category_chart


def records(spec):
    return spec["datasets"][spec["data"]["name"]]


@pytest.mark.parametrize("labels,values", [
    (["consolidator", "distributor", "peripheral"], [4, 7, 2]),
    (["Baseline", "Remove top 1", "Remove top 20"], [1877, 1782, 1342]),
    ([0, 1, 2, 3, 4], [81, 472, 462, 789, 444]),
])
def test_category_chart_keeps_order_values_and_horizontal_labels(labels, values):
    spec = category_chart(pd.Series(values, index=labels),
        category_title="Category", value_title="Accounts").to_dict()
    assert spec["width"] == "container"
    assert spec["encoding"]["y"]["sort"] is None
    assert spec["encoding"]["y"]["title"] is None
    assert spec["encoding"]["y"]["axis"]["labelOverlap"] is False
    for name in ("x", "y"):
        assert spec["encoding"][name]["axis"]["labelAngle"] == 0
    assert records(spec) == [{"category": str(label), "value": value}
                             for label, value in zip(labels, values)]


def test_priority_interval_labels_are_serializable_and_get_enough_vertical_space():
    bins = pd.cut(pd.Series([.15, .20, .30, .70, .95]), bins=12).value_counts(sort=False)
    spec = category_chart(bins, category_title="Priority range", value_title="Accounts").to_dict()
    assert len(records(spec)) == 12
    assert sum(row["value"] for row in records(spec)) == 5
    assert all(isinstance(row["category"], str) for row in records(spec))
    assert spec["height"] >= 12 * 28


def test_resilience_fraction_is_displayed_as_percent_without_rescaling_values():
    spec = category_chart(pd.Series([1., .7], index=["Baseline", "Remove top 20"]),
        category_title="Scenario", value_title="Share of baseline", value_format=".0%").to_dict()
    assert spec["encoding"]["x"]["axis"]["format"] == ".0%"
    assert [row["value"] for row in records(spec)] == [1., .7]


def test_daily_chart_uses_horizontal_dates_and_top_legend_with_exact_amounts():
    daily = pd.DataFrame({"Incoming KZT": [12500.5, 0.], "Outgoing KZT": [0., 7500.]},
                         index=pd.to_datetime(["2026-07-01", "2026-07-31"]))
    spec = activity_chart(daily).to_dict()
    assert spec["width"] == "container"
    assert spec["encoding"]["x"]["axis"]["labelAngle"] == 0
    assert spec["encoding"]["x"]["axis"]["format"] == "%d %b"
    assert spec["encoding"]["x"]["axis"]["tickCount"] == 5
    assert spec["encoding"]["color"]["legend"]["direction"] == "horizontal"
    assert spec["encoding"]["color"]["legend"]["orient"] == "top"
    assert spec["encoding"]["y"]["title"] is None
    assert [row["amount_kzt"] for row in records(spec)] == [12500.5, 0., 0., 7500.]
