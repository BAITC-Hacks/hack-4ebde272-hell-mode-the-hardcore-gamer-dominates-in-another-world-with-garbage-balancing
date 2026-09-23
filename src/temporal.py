"""Deterministic date-level observations, never intraday or same-money claims.

Inputs are normalized by :mod:`src.schema`. Missing outgoing observations at
hop four and missing seed inflows invalidate inbound/outbound ratio signals.
Positive route patterns describe observed date co-occurrence only.
"""

from __future__ import annotations

import json
from itertools import islice, product
from math import fsum

import numpy as np
import pandas as pd


# Limits apply independently to each node and each route/return search. Counts
# are lower bounds when the corresponding ``*_truncated`` column is true.
PATTERN_PAIR_LIMIT = 512
PATTERN_DAY_CHECK_LIMIT = 20_000
PATTERN_EVIDENCE_LIMIT = 3
PATTERN_DATE_LIMIT = 3
FOLLOWUP_DAYS = 2
SIMILAR_AMOUNT_RELATIVE_TOLERANCE = 0.05


def _json(records: list[dict]) -> str:
    """Encode bounded evidence with stable keys and exact string identifiers."""
    return json.dumps(records, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pattern_features(gids: list[int], tx: pd.DataFrame) -> pd.DataFrame:
    """Observe bounded A→B→C patterns, reciprocal dates and amount repetition.

    A route is attached to its intermediate node B. It requires three distinct
    nodes and >=2 distinct dates D with A→B on D and B→C on D/D+1/D+2.
    A temporal return candidate is attached to the receiving node B and needs
    >=1 date D with A→B followed by B→A on D/D+1/D+2. Same-day observations
    have unknown ordering. Neither pattern identifies the same money.

    At most 512 lexicographically ordered counterparty pairs and 20,000
    inbound-date probes are examined per node per pattern type. Truncation
    indicates lower-bound counts; the top three patterns retain at most three
    supporting date pairs each. Amount totals are observed activity totals on
    matching dates, not matched/traced transfer amounts.

    Exact repeated outgoing amounts require >=2 transaction rows over the
    entire observed window. Similar-amount groups are deterministic greedy
    groups on each outgoing date, starting at the smallest ungrouped amount
    and accepting values <=1.05 times that amount; a candidate group needs
    >=3 rows and >=2 distinct destinations. Shares divide matching rows by all
    observed outgoing rows. These are descriptive observations, not a finding
    of splitting, and say nothing about transfers absent from the input.
    """
    pair_days: dict[tuple[int, int], dict[pd.Timestamp, float]] = {}
    incoming: dict[int, list[int]] = {}
    outgoing: dict[int, list[int]] = {}
    pair_daily = tx.groupby(["src", "dst", "date"], sort=True)["sum_kzt"].sum()
    for (src, dst), group in pair_daily.groupby(level=["src", "dst"], sort=True):
        src, dst = int(src), int(dst)
        pair_days[src, dst] = {day: float(amount) for (_, _, day), amount in group.items()}
        incoming.setdefault(dst, []).append(src)
        outgoing.setdefault(src, []).append(dst)
    outgoing_rows = {int(gid): group for gid, group in tx.groupby("src", sort=True)}
    offsets = tuple(pd.Timedelta(days=offset) for offset in range(FOLLOWUP_DAYS + 1))

    def support(in_days: dict, out_days: dict, budget: int) -> tuple[list, int, bool]:
        matches = []
        used = 0
        ordered_days = sorted(in_days)
        for day in ordered_days:
            if used >= budget:
                break
            used += 1
            for offset in offsets:
                out_day = day + offset
                if out_day in out_days:
                    matches.append((day, out_day))
                    break  # earliest observed outgoing date for this inbound date
        return matches, used, used < len(ordered_days)

    def evidence_record(src: int, via: int, dst: int, matches: list, in_days: dict, out_days: dict) -> dict:
        return {
            "src": str(src), "via": str(via), "dst": str(dst),
            "support_days": len(matches),
            "date_pairs": [[a.date().isoformat(), b.date().isoformat()] for a, b in matches[:PATTERN_DATE_LIMIT]],
            "in_kzt_on_support_dates": fsum(in_days[a] for a, _ in matches),
            "out_kzt_on_matched_dates": fsum(out_days[b] for b in sorted({b for _, b in matches})),
        }

    rows = []
    for gid in gids:
        record = {"gid": gid}
        sources = sorted(source for source in incoming.get(gid, []) if source != gid)
        destinations = sorted(destination for destination in outgoing.get(gid, []) if destination != gid)
        candidates = ((a, c) for a, c in product(sources, destinations) if a != c)
        selected = list(islice(candidates, PATTERN_PAIR_LIMIT + 1))
        route_truncated = len(selected) > PATTERN_PAIR_LIMIT
        route_records = []
        checks = 0
        for position, (src, dst) in enumerate(selected[:PATTERN_PAIR_LIMIT]):
            matches, used, partial = support(pair_days[src, gid], pair_days[gid, dst], PATTERN_DAY_CHECK_LIMIT - checks)
            checks += used
            route_truncated |= partial
            if len(matches) >= 2:
                route_records.append(evidence_record(src, gid, dst, matches, pair_days[src, gid], pair_days[gid, dst]))
            if checks >= PATTERN_DAY_CHECK_LIMIT:
                route_truncated |= position + 1 < min(len(selected), PATTERN_PAIR_LIMIT)
                break
        route_records.sort(key=lambda row: (-row["support_days"], int(row["src"]), int(row["dst"])))
        record.update(
            repeated_route_count=len(route_records),
            repeated_route_max_support_days=max((r["support_days"] for r in route_records), default=0),
            repeated_route_evidence=_json(route_records[:PATTERN_EVIDENCE_LIMIT]),
            repeated_route_truncated=bool(route_truncated),
        )

        reciprocal = sorted(set(sources) & set(destinations))
        return_truncated = len(reciprocal) > PATTERN_PAIR_LIMIT
        return_records = []
        checks = 0
        for position, src in enumerate(reciprocal[:PATTERN_PAIR_LIMIT]):
            matches, used, partial = support(pair_days[src, gid], pair_days[gid, src], PATTERN_DAY_CHECK_LIMIT - checks)
            checks += used
            return_truncated |= partial
            if matches:
                return_records.append(evidence_record(src, gid, src, matches, pair_days[src, gid], pair_days[gid, src]))
            if checks >= PATTERN_DAY_CHECK_LIMIT:
                return_truncated |= position + 1 < min(len(reciprocal), PATTERN_PAIR_LIMIT)
                break
        return_records.sort(key=lambda row: (-row["support_days"], int(row["src"])))
        record.update(
            temporal_return_count=len(return_records),
            temporal_return_max_support_days=max((r["support_days"] for r in return_records), default=0),
            temporal_return_evidence=_json(return_records[:PATTERN_EVIDENCE_LIMIT]),
            temporal_return_truncated=bool(return_truncated),
        )

        node_rows = outgoing_rows.get(gid)
        repeated_count = similar_count = similar_group_count = 0
        amount_records = []
        n_out = 0 if node_rows is None else len(node_rows)
        if n_out:
            amount_counts = node_rows.groupby("sum_kzt", sort=True).size()
            repeated_count = int(amount_counts[amount_counts >= 2].sum())
            for amount, count in amount_counts[amount_counts >= 2].items():
                amount_records.append({"kind": "exact_repetition", "amount_kzt": float(amount), "n_tx": int(count)})
            for day, day_rows in node_rows.groupby("date", sort=True):
                values = list(day_rows.sort_values(["sum_kzt", "dst"], kind="stable")[["sum_kzt", "dst"]].itertuples(index=False, name=None))
                start = 0
                while start < len(values):
                    end = start + 1
                    bound = values[start][0] * (1.0 + SIMILAR_AMOUNT_RELATIVE_TOLERANCE)
                    while end < len(values) and values[end][0] <= bound:
                        end += 1
                    group = values[start:end]
                    recipients = sorted({int(dst) for _, dst in group})
                    if len(group) >= 3 and len(recipients) >= 2:
                        similar_count += len(group)
                        similar_group_count += 1
                        amount_records.append({
                            "kind": "same_day_similar_amounts", "date": day.date().isoformat(),
                            "n_tx": len(group), "n_recipients": len(recipients),
                            "min_kzt": float(group[0][0]), "max_kzt": float(group[-1][0]),
                            "recipient_gids": [str(dst) for dst in recipients[:PATTERN_EVIDENCE_LIMIT]],
                        })
                    start = end
        amount_records.sort(key=lambda row: (-row["n_tx"], row["kind"], row.get("date", ""), row.get("amount_kzt", row.get("min_kzt", 0.0))))
        record.update(
            outgoing_repeated_amount_tx_count=repeated_count,
            outgoing_repeated_amount_tx_share=repeated_count / n_out if n_out else np.nan,
            outgoing_similar_amount_tx_count=similar_count,
            outgoing_similar_amount_tx_share=similar_count / n_out if n_out else np.nan,
            similar_amount_group_count=similar_group_count,
            amount_pattern_evidence=_json(amount_records[:PATTERN_EVIDENCE_LIMIT]),
        )
        rows.append(record)
    columns = ["gid", "repeated_route_count", "repeated_route_max_support_days", "repeated_route_evidence", "repeated_route_truncated", "temporal_return_count", "temporal_return_max_support_days", "temporal_return_evidence", "temporal_return_truncated", "outgoing_repeated_amount_tx_count", "outgoing_repeated_amount_tx_share", "outgoing_similar_amount_tx_count", "outgoing_similar_amount_tx_share", "similar_amount_group_count", "amount_pattern_evidence"]
    return pd.DataFrame(rows, columns=columns).set_index("gid")


def temporal_features(nodes: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Return gid-indexed, deterministic date-level activity and evidence.

    The observation end is the latest transaction date in the input, not an
    assumed month end. The relay denominator contains only inbound dates D
    with D+2 <= that end. Relay counts remain observed diagnostics for seeds
    and boundary nodes, but both relay and same-day ratios are invalid/NaN
    for those nodes. A non-boundary, non-seed node with a complete inbound
    date and no subsequent outgoing observation legitimately has relay zero.

    Same-day flow divides outgoing amount on inbound-active dates by all
    observed outgoing amount. Zero denominators are NaN. Peak-day share divides
    the largest daily (in+out) amount by observed (in+out) total; it is NaN for
    seeds and inactive nodes. For depth-4 nodes the peak share describes the
    sampled activity only; it does not imply complete outgoing coverage.

    All counts/dates/amounts describe observed rows. A self-transfer contributes
    once on each side of activity accounting. Peak dates break ties earliest.
    Evidence strings are bounded stable JSON, with identifiers encoded as exact
    decimal strings. No ratio or pattern claims intraday order or traces funds.
    """
    nodes = nodes.sort_values("gid", kind="stable")
    gids = [int(gid) for gid in nodes["gid"]]
    # Canonical floating summation order makes row-permutation results stable.
    tx = transactions.sort_values(["src", "dst", "date", "sum_kzt"], kind="stable")
    incoming = tx.groupby(["dst", "date"], sort=True).agg(in_kzt=("sum_kzt", "sum"), in_senders=("src", "nunique"))
    outgoing = tx.groupby(["src", "date"], sort=True).agg(out_kzt=("sum_kzt", "sum"))
    incoming.index.names = ["gid", "date"]
    outgoing.index.names = ["gid", "date"]
    daily = incoming.join(outgoing, how="outer").fillna(0.0).sort_index()
    grouped_daily = {int(gid): group.droplevel("gid") for gid, group in daily.groupby(level="gid", sort=True)}
    observation_start = tx["date"].min() if not tx.empty else pd.NaT
    observation_end = tx["date"].max() if not tx.empty else pd.NaT
    offsets = tuple(pd.Timedelta(days=offset) for offset in range(FOLLOWUP_DAYS + 1))
    rows = []
    for node in nodes[["gid", "depth", "is_seed"]].itertuples(index=False):
        gid = int(node.gid)
        node_daily = grouped_daily.get(gid)
        in_set: set = set()
        out_set: set = set()
        peak_share, same_day_ratio, relay_ratio = np.nan, np.nan, np.nan
        peak_amount = same_day_out = out_amount = 0.0
        peak_date = max_senders_date = pd.NaT
        max_senders = 0
        if node_daily is not None:
            in_set = set(node_daily.index[node_daily["in_kzt"] > 0])
            out_set = set(node_daily.index[node_daily["out_kzt"] > 0])
            amounts = node_daily["in_kzt"] + node_daily["out_kzt"]
            peak_amount = float(amounts.max())
            peak_date = amounts.idxmax()
            total_amount = float(amounts.sum())
            peak_share = peak_amount / total_amount if total_amount else np.nan
            max_senders = int(node_daily["in_senders"].max())
            if max_senders:
                max_senders_date = node_daily["in_senders"].idxmax()
            out_amount = float(node_daily["out_kzt"].sum())
            same_day_out = float(node_daily.loc[node_daily.index.isin(in_set), "out_kzt"].sum())
        eligible_days = sorted(day for day in in_set if day + offsets[-1] <= observation_end)
        matched_days = sum(any(day + offset in out_set for offset in offsets) for day in eligible_days)
        shared_reason = "seed_inbound_incomplete" if node.is_seed else "boundary_outbound_incomplete" if node.depth == 4 else ""
        same_day_reason = shared_reason or ("no_outgoing_activity" if not out_amount else "")
        relay_reason = shared_reason or ("no_inbound_activity" if not in_set else "no_complete_followup_window" if not eligible_days else "")
        if not same_day_reason:
            same_day_ratio = same_day_out / out_amount
        if not relay_reason:
            relay_ratio = matched_days / len(eligible_days)
        if node.is_seed:
            peak_share = np.nan
        rows.append({
            "gid": gid, "active_days": len(in_set | out_set),
            "same_day_flow_ratio": same_day_ratio, "relay_2d_ratio": relay_ratio,
            "max_in_senders_day": max_senders, "peak_day_share": peak_share,
            "inbound_active_days": len(in_set), "outbound_active_days": len(out_set),
            "same_day_flow_valid": not bool(same_day_reason), "same_day_flow_invalid_reason": same_day_reason,
            "relay_2d_eligible_days": len(eligible_days), "relay_2d_matched_days": matched_days,
            "relay_2d_censored_days": len(in_set) - len(eligible_days),
            "relay_2d_valid": not bool(relay_reason), "relay_2d_invalid_reason": relay_reason,
            "temporal_observation_start": observation_start, "temporal_observation_end": observation_end,
            "peak_activity_date": peak_date, "peak_activity_kzt": peak_amount,
            "max_in_senders_date": max_senders_date, "peak_day_share_valid": bool(pd.notna(peak_share)),
        })
    columns = ["gid", "active_days", "same_day_flow_ratio", "relay_2d_ratio", "max_in_senders_day", "peak_day_share", "inbound_active_days", "outbound_active_days", "same_day_flow_valid", "same_day_flow_invalid_reason", "relay_2d_eligible_days", "relay_2d_matched_days", "relay_2d_censored_days", "relay_2d_valid", "relay_2d_invalid_reason", "temporal_observation_start", "temporal_observation_end", "peak_activity_date", "peak_activity_kzt", "max_in_senders_date", "peak_day_share_valid"]
    result = pd.DataFrame(rows, columns=columns).set_index("gid")
    for column in ("temporal_observation_start", "temporal_observation_end", "peak_activity_date", "max_in_senders_date"):
        # Seconds are not a Parquet timestamp unit; choose milliseconds so
        # loading the artifact preserves the exact public dtype on all inputs.
        result[column] = pd.to_datetime(result[column]).astype("datetime64[ms]")
    result = result.join(_pattern_features(gids, tx))
    # Explicit types keep the public schema stable even for zero input nodes.
    integer_columns = [
        "active_days", "max_in_senders_day", "inbound_active_days",
        "outbound_active_days", "relay_2d_eligible_days", "relay_2d_matched_days",
        "relay_2d_censored_days", "repeated_route_count",
        "repeated_route_max_support_days", "temporal_return_count",
        "temporal_return_max_support_days", "outgoing_repeated_amount_tx_count",
        "outgoing_similar_amount_tx_count", "similar_amount_group_count",
    ]
    boolean_columns = [
        "same_day_flow_valid", "relay_2d_valid", "peak_day_share_valid",
        "repeated_route_truncated", "temporal_return_truncated",
    ]
    float_columns = [
        "same_day_flow_ratio", "relay_2d_ratio", "peak_day_share",
        "peak_activity_kzt", "outgoing_repeated_amount_tx_share",
        "outgoing_similar_amount_tx_share",
    ]
    string_columns = [
        "same_day_flow_invalid_reason", "relay_2d_invalid_reason",
        "repeated_route_evidence", "temporal_return_evidence", "amount_pattern_evidence",
    ]
    result = result.astype({
        **dict.fromkeys(integer_columns, "int64"),
        **dict.fromkeys(boolean_columns, "bool"),
        **dict.fromkeys(float_columns, "float64"),
        **dict.fromkeys(string_columns, "string"),
    })
    result.index = pd.Index(result.index, dtype="int64", name="gid")
    return result
