"""Trace parsing for the A7 flow study (EXPLORATORY, post hoc).

One cycle = (run, fold_run_id, iteration_in_fold). Each LLM call record carries role/stage/fold/iteration
fields, so cycle attribution is read off the record, not inferred. Within a cycle we rebuild an ordered event
list from each call's OUTPUT message (`message`) and look up the tool result by tool_call_id in any later
call's input history (`messages`, role=='tool').
"""
import ast
import glob
import gzip
import json
import os
import re

# Raw run outputs (llm_traces.jsonl.gz etc.) and synp_<task>.json task metadata are not part of the
# release; point MAST_DATA / SYNP_META at local copies to re-run the pipeline.
MAST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(MAST, '..', '..', '..', '..'))
DATA = os.environ.get('MAST_DATA', os.path.join(MAST, 'mast_data'))
META = os.environ.get('SYNP_META', os.path.join(MAST, 'synp_meta'))
EPIS = os.path.join(REPO, 'docs', 'agent2027', 'results', 'final_20261112', 'episodes_final.csv')

TASKS = ['p2', 'p3', 'p4', 'p5', 'p6a', 'p6b']


def list_runs():
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, '*', '*', 'tasks', '*', '*', 'output'))):
        parts = p.replace('\\', '/').split('/')
        session, row = parts[-6], parts[-5]
        task = parts[-2].replace('synp_', '')
        arm = 'full' if row.startswith('A05') else 'single'
        out.append(dict(session=session, row_id=row, task=task, arm=arm, path=p))
    return out


def load_meta(task):
    with open(os.path.join(META, f'synp_{task}.json'), encoding='utf-8') as f:
        return json.load(f)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            try:
                return json.loads(x)
            except Exception:
                return []
    return []


def load_records(path):
    recs = []
    with gzip.open(os.path.join(path, 'llm_traces.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            recs.append(r)
    recs.sort(key=lambda r: r['seq'])
    return recs


def tool_results(recs):
    """tool_call_id -> tool result content, from all input histories."""
    res = {}
    for r in recs:
        msgs = r.get('messages')
        if isinstance(msgs, str):
            msgs = _as_list(msgs)
        for m in msgs or []:
            if isinstance(m, dict) and m.get('role') == 'tool' and m.get('tool_call_id'):
                res.setdefault(m['tool_call_id'], m.get('content'))
    return res


def parse_args(s):
    if isinstance(s, dict):
        return s
    try:
        a = json.loads(s)
    except Exception:
        try:
            a = ast.literal_eval(s)
        except Exception:
            return {'_raw': s}
    if isinstance(a, dict) and isinstance(a.get('args'), dict):
        a = dict(a['args'], **{k: v for k, v in a.items() if k != 'args'})
    return a if isinstance(a, dict) else {'_raw': s}


ERR_RE = re.compile(r'(An error occurred while running the tool|Error executing|Traceback|\bError:|Exception|failed|'
                    r'not found in|Invalid JSON|could not|KeyError|ValueError|NameError|TypeError)', re.I)


def is_error(content):
    if content is None:
        return False
    c = str(content)[:600]
    if c.startswith('Attribute ') and 'added to df_features' in c:
        return False
    return bool(ERR_RE.search(c))


def cycle_events(recs):
    """dict (fold, iter) -> list of events (ordered by seq). Each event: role, stage, phase, kind, name, args, result,
    error, seq, sub (True for WebSearchAgent sub-agent calls)."""
    tr = tool_results(recs)
    cyc = {}
    for r in recs:
        key = (r.get('fold_run_id'), r.get('iteration_in_fold'))
        sub = r.get('agent_name') == 'WebSearchAgent'
        base = dict(role=r.get('role'), stage=r.get('stage'), phase=r.get('phase'), seq=r['seq'], sub=sub,
                    index=r.get('index'), usage=r.get('usage') or {}, session=r.get('session_name'))
        ev = cyc.setdefault(key, [])
        m = r.get('message')
        if not m:
            ev.append(dict(base, kind='nomsg', name='', args={}, result=None, error=True))
            continue
        tcs = m.get('tool_calls')
        if isinstance(tcs, str):
            tcs = _as_list(tcs)
        if tcs:
            for i, t in enumerate(tcs):
                fn = t.get('function', {})
                res = tr.get(t.get('id'))
                ev.append(dict(base, kind='tool', name=fn.get('name', ''), args=parse_args(fn.get('arguments', '')),
                               result=res, error=is_error(res), first_in_call=(i == 0)))
        else:
            ev.append(dict(base, kind='text', name='', args={}, result=None, error=False,
                           content=m.get('content') or ''))
    return cyc


def load_jsonl(p):
    out = []
    if not os.path.exists(p):
        return out
    with open(p, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l:
                out.append(json.loads(l))
    return out
