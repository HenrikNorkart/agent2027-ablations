"""Amendment 19c: inf-guard sensitivity re-runs of the inf-crashed A19 (Q122) rows on dedicated servers (run in a container on CAIR).

python gen_amend19c.py <manifest_dir> <sessions_dir>
Appends the new rows to A19_CAP-2.csv and writes S1029a.csv: main Q122 server on GPUs 1,2 (port 8048),
with the mixed rows' small roles on the holders S1029b (Q4, port 8049) and S1029c (Q0.8, port 8050).
"""
import csv
import json
import sys
from pathlib import Path

MAN, SESS = Path(sys.argv[1]), Path(sys.argv[2])
SID = "S1029a"
# (new_row, source_row)
PLAN = [
    *[(f"A19-5{i:02d}", f"A19-{j}") for i, j in zip(range(3, 8), [409, 410, 411, 413, 414])],
    ("A19-508", "A19-421"),
]
# old server port -> new: Q122 (S1027j 8042, S1027i 8041) -> 8048, Q4 (S1027k) -> 8049, Q0.8 (S1027l) -> 8050
PORTS = {"8042": "8048", "8041": "8048", "8044": "8049", "8045": "8050"}
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

new_rows = []
for order, (new_id, src_id) in enumerate(PLAN, start=1):
    src = dict(sources[src_id])
    assert "covtype" not in src["dataset_id"].lower()
    ov = json.loads(src["env_overrides"] or "{}")
    ov["ROGUE_ONE_INF_GUARD"] = "1"
    if "LLM_PORT" in ov:
        ov["LLM_PORT"] = PORTS[ov["LLM_PORT"]]
    src.update(
        row_id=new_id,
        session_id=SID,
        order_key=str(order),
        image_tag=src["image_tag"].replace("-qfix5", "-qfix6"),
        env_overrides=json.dumps(dict(sorted(ov.items())), separators=(",", ":")),
        cap_hours=CAP_HOURS,
        notes=f"Amendment 19c: inf-guard sensitivity (own servers, 18 h cap) re-run of {src_id} (exploratory; {src_id} stays of record)",
        **{k: PORTS[src[k]] for k in ("port_S", "port_E", "port_T")},
    )
    new_rows.append(src)

path = MAN / "A19_CAP-2.csv"
fields, rows = read(path)
have = {r["row_id"] for r in rows}
add = [dict(r, session_id="", order_key="") for r in new_rows if r["row_id"] not in have]
with open(path, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=fields, lineterminator="\n").writerows(add)
print(path.name, "+", len(add))

fields, _ = read(next(SESS.glob("S1027a.csv")))
with open(SESS / f"{SID}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(new_rows)
print(SID, len(new_rows), "rows")
