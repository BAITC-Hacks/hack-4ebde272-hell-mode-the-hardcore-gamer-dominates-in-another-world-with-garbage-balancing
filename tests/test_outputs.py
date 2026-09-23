import pandas as pd

from src.exports import write_exports
from validate_submission import validate_submission


def test_export_schema_and_submission_validation(tmp_path):
    rows = []
    for gid in range(2248):
        rows.append({"gid": gid, "role": "peripheral", "role_score": 0.4,
                     "cluster_id": 0, "priority_score": (2248-gid)/2248,
                     "evidence": f"in_deg={gid % 3}, out_deg=0", "depth": 1,
                     "out_deg": 0, "truncated_by_depth": False, "is_seed": gid < 8,
                     "in_deg": 0, "out_deg": 0, "priority_seed": False})
    frame = pd.DataFrame(rows)
    clusters = pd.DataFrame([{"cluster_id": 0, "n_nodes": 2248, "n_seed": 8,
                              "sum_kzt_internal": 0.0, "top_gids": "0,1,2,3,4",
                              "hypothesis": "sparse peripheral community"}])
    write_exports(frame, clusters, tmp_path)
    output = pd.read_csv(tmp_path / "nodes_roles.csv")
    assert {"gid", "role", "role_score", "cluster_id", "priority_score", "evidence"}.issubset(output.columns)
    assert validate_submission(tmp_path, source_nodes=pd.DataFrame({"gid": range(2248)}))
