"""Amendment 19b: inf-guard sensitivity (own server, 18 h cap) re-runs of inf-crashed Qwen rows (run in a container on CAIR).

python gen_amend19.py <manifest_dir> <sessions_dir>
Appends the new rows to A17_CAP-1.csv / A19_CAP-2.csv / A20_CAP-3.csv and writes S1027m.csv (Q4 server
of the holder S1027k, port 8044) and S1027n.csv (Q122 server of S1027j, port 8042).
"""
import csv
import json
import sys
from pathlib import Path

MAN, SESS = Path(sys.argv[1]), Path(sys.argv[2])
MANIFEST = {"A17": "A17_CAP-1.csv", "A19": "A19_CAP-2.csv", "A20": "A20_CAP-3.csv"}
# (session, new_row, source_row, uniform host port or None for mixed rows)
PLAN = [
    *[("S1028a", f"A17-5{i:02d}", f"A17-{j}", "8046")
      for i, j in zip(range(8, 13), [434, 436, 438, 431, 432])],
    *[("S1028b", f"A17-5{i:02d}", f"A17-{j}", "8047")
      for i, j in zip(range(13, 21), range(442, 450))],
]
CAP_HOURS = "18"


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


sources = {}
for f in SESS.glob("S1027*.csv"):
    _, rows = read(f)
    for row in rows:
        sources[row["row_id"]] = row

new_by_manifest: dict[str, list[dict]] = {k: [] for k in MANIFEST}
sessions: dict[str, list[dict]] = {}
for order, (sid, new_id, src_id, port) in enumerate(PLAN, start=1):
    src = dict(sources[src_id])
    assert "covtype" not in src["dataset_id"].lower()
    ov = json.loads(src["env_overrides"] or "{}")
    ov["ROGUE_ONE_INF_GUARD"] = "1"
    src.update(
        row_id=new_id,
        session_id=sid,
        order_key=str(order),
        image_tag=src["image_tag"].replace("-qfix5", "-qfix6"),
        env_overrides=json.dumps(dict(sorted(ov.items())), separators=(",", ":")),
        cap_hours=CAP_HOURS,
        notes=f"Amendment 19b: inf-guard sensitivity (own server, 18 h cap) re-run of {src_id} (exploratory; {src_id} stays of record)",
    )
    if port:
        src.update(port_S=port, port_E=port, port_T=port)
    sessions.setdefault(sid, []).append(src)
    new_by_manifest[new_id[:3]].append(src)

for abl, fname in MANIFEST.items():
    path = MAN / fname
    fields, rows = read(path)
    have = {r["row_id"] for r in rows}
    add = [dict(r, session_id="", order_key="") for r in new_by_manifest[abl] if r["row_id"] not in have]
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields, lineterminator="\n").writerows(add)
    print(fname, "+", len(add))

for sid, rows in sessions.items():
    fields, _ = read(next(SESS.glob("S1027a.csv")))
    with open(SESS / f"{sid}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(sid, len(rows), "rows")
