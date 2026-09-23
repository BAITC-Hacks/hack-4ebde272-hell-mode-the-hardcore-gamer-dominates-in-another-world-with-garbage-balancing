"""Date-granularity activity features; no intraday order is inferred."""

from __future__ import annotations

import pandas as pd


def temporal_features(nodes: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Return date-level activity features indexed by gid.

    ``same_day_flow_ratio`` is the share of a node's outgoing amount occurring
    on dates when it also received funds. ``relay_2d_ratio`` is the share of
    inbound-active dates with any outgoing activity on D, D+1, or D+2. These
    are date-level overlaps only; they make no claim about within-day ordering.
    ``peak_day_share`` is the busiest date's share of total inbound plus
    outbound activity amount.
    """
    gids = nodes["gid"].tolist()
    if transactions.empty:
        result = pd.DataFrame(index=pd.Index(gids, name="gid"))
        result["active_days"] = 0
        result["same_day_flow_ratio"] = float("nan")
        result["relay_2d_ratio"] = float("nan")
        result["max_in_senders_day"] = 0
        result["peak_day_share"] = float("nan")
        return result

    tx = transactions
    incoming = tx.groupby(["dst", "date"], sort=False).agg(
        in_kzt=("sum_kzt", "sum"), in_senders=("src", "nunique")
    )
    outgoing = tx.groupby(["src", "date"], sort=False).agg(out_kzt=("sum_kzt", "sum"))
    incoming.index.names = ["gid", "date"]
    outgoing.index.names = ["gid", "date"]
    in_dates = {gid: set(group.index.get_level_values("date")) for gid, group in incoming.groupby(level="gid")}
    out_dates = {gid: set(group.index.get_level_values("date")) for gid, group in outgoing.groupby(level="gid")}
    daily = incoming.join(outgoing, how="outer").fillna(0.0)
    daily["in_senders"] = daily["in_senders"].astype(int)

    rows = []
    for gid in gids:
        in_set = in_dates.get(gid, set())
        out_set = out_dates.get(gid, set())
        active = in_set | out_set
        if active:
            node_daily = daily.xs(gid, level="gid")
            day_amount = node_daily["in_kzt"] + node_daily["out_kzt"]
            total_amount = float(day_amount.sum())
            peak_share = float(day_amount.max() / total_amount) if total_amount > 0 else float("nan")
            max_senders = int(node_daily["in_senders"].max())
        else:
            peak_share = float("nan")
            max_senders = 0
        if out_set:
            out_amount = float(outgoing.xs(gid, level="gid")["out_kzt"].sum())
            same_day_out = float(sum(
                amount for (node, day), amount in outgoing["out_kzt"].items()
                if node == gid and day in in_set
            ))
            same_day_ratio = same_day_out / out_amount if out_amount else float("nan")
        else:
            same_day_ratio = float("nan")
        relay_ratio = (
            sum(any(day + pd.Timedelta(days=offset) in out_set for offset in range(3)) for day in in_set) / len(in_set)
            if in_set else float("nan")
        )
        rows.append((gid, len(active), same_day_ratio, relay_ratio, max_senders, peak_share))

    return pd.DataFrame(
        rows,
        columns=["gid", "active_days", "same_day_flow_ratio", "relay_2d_ratio", "max_in_senders_day", "peak_day_share"],
    ).set_index("gid")
