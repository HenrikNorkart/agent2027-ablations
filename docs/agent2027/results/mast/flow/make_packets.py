"""RQ2 annotation packets (48 cycles; P2, P4, P6b x arm x 8). EXPLORATORY. Sample fixed with seed 42."""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flowlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
PK = os.path.join(STUDY, 'packets')
LIMIT = 25000

d = pd.read_csv(os.path.join(HERE, 'cycles_analysis.csv'))
rng = np.random.default_rng(42)
picks = []
for task in ['p2', 'p4', 'p6b']:
    for arm in ['full', 'single']:
        x = d[(d.task == task) & (d.arm == arm)]
        # never-recovered folds (official rec_any == 0 and no E_phi_code in our replay), iterations 1-8
        neg = x[(x.rec_any == 0) & (x.first_E.isna()) & x.iteration.between(1, 8)]
        runs = list(neg.row_id.unique())
        rng.shuffle(runs)
        chosen = []
        i = 0
        while len(chosen) < 4 and runs:
            r = runs[i % len(runs)]
            cand = neg[(neg.row_id == r) & ~neg.set_index(['row_id', 'fold', 'iteration']).index.isin(chosen)]
            cand = cand[~cand.fold.isin([c[1] for c in chosen if c[0] == r])] if len(cand[~cand.fold.isin([c[1] for c in chosen if c[0] == r])]) else cand
            if len(cand):
                row = cand.iloc[rng.integers(len(cand))]
                chosen.append((row.row_id, int(row.fold), int(row.iteration)))
            i += 1
            if i > 200:
                break
        for c in chosen:
            picks.append(dict(task=task, arm=arm, row_id=c[0], fold=c[1], iteration=c[2], stratum='never_recovered'))
        # recovering folds: first E_phi_code cycle or the one just before it
        pos = x[x.first_E.notna()][['row_id', 'fold', 'first_E']].drop_duplicates()
        runs = list(pos.row_id.unique())
        rng.shuffle(runs)
        chosen = []
        i = 0
        while len(chosen) < 4 and runs:
            r = runs[i % len(runs)]
            cand = pos[(pos.row_id == r) & ~pos.fold.isin([c[1] for c in chosen if c[0] == r])]
            if len(cand):
                row = cand.iloc[rng.integers(len(cand))]
                fe = int(row.first_E)
                which = 'first_E' if (fe == 0 or rng.random() < 0.5) else 'before_first_E'
                it = fe if which == 'first_E' else fe - 1
                chosen.append((row.row_id, int(row.fold), it, which))
            i += 1
            if i > 200:
                break
        for c in chosen:
            picks.append(dict(task=task, arm=arm, row_id=c[0], fold=c[1], iteration=c[2], stratum='recovering:' + c[3]))

P = pd.DataFrame(picks)
P = P.sample(frac=1.0, random_state=42).reset_index(drop=True)
P['unit_id'] = [f'U{i + 1:02d}' for i in range(len(P))]

# ------------------------------------------------------------------ blinding
SUBS = [
    (r'set_phased\w*', '[session]'), (r'\bA(05|16)-\d{3}\b', '[run]'), (r'\bS10\d\d[a-z]\b', '[session]'),
    (r'#{0,3}\s*PHASE\s*\d/4\s*[—-]+\s*\w+\.?', ''), (r'\bphases?\b', 'step'), (r'\bPhases?\b', 'Step'),
    (r'\bPHASES?\b', 'STEP'),
    (r'\bPLAN\b', 'PLANNER'), (r'\bBUILD\b', 'CODER'), (r'\bASSESS/PRUNE\b', 'EVALUATOR'), (r'\bASSESS\b', 'EVALUATOR'),
    (r'\bPRUNE\b', 'EVALUATOR'),
    (r'Scientist\s*Agent|ScientistAgent|\bScientist\b', 'Planner'), (r'\bscientist\b', 'planner'),
    (r'Extractor\s*Agent|ExtractorAgent|\bExtractor\b', 'Coder'), (r'\bextractor\b', 'coder'),
    (r'Tester\s*Agent|TesterAgent|\bTester\b', 'Evaluator'), (r'\btester\b', 'evaluator'),
    (r'WebSearchAgent', 'search helper'),
    (r'chatcmpl-tool-[0-9a-f]+', '[id]'),
]


def blind(s):
    s = '' if s is None else str(s)
    for a, b in SUBS:
        s = re.sub(a, b, s)
    return s


def trunc(s, n):
    s = '' if s is None else str(s)
    return s if len(s) <= n else s[:n] + f' …[truncated {len(s) - n} chars]'


def fmt_args(a, keep_code=False, n=600):
    out = []
    for k, v in a.items():
        if keep_code and k in ('command', 'code', 'name'):
            out.append(f'{k}:\n```python\n{v}\n```' if k != 'name' else f'name: {v}')
        else:
            out.append(f'{k}: {trunc(v, n)}')
    return '\n'.join(out)


def render(run, key, phi_txt):
    recs = F.load_records(run['path'])
    cyc = F.cycle_events(recs)
    ev = cyc[key]
    secs = {}
    # PLANNER
    sev = [e for e in ev if e['role'] == 'scientist' and not e['sub']]
    txts = [e for e in sev if e['kind'] == 'text' and (e.get('content') or '').strip()]
    focus = txts[-1]['content'] if txts else ''
    L = []
    for e in sev:
        if e['kind'] == 'tool':
            L.append(f"- TOOL CALL `{e['name']}`\n```\n{trunc(fmt_args(e['args'], n=500), 900)}\n```\n  RESULT: {trunc(e['result'], 700)}")
        elif e['kind'] == 'text' and e.get('content') != focus and (e.get('content') or '').strip():
            L.append(f"- TEXT: {trunc(e['content'], 800)}")
    secs['P'] = ('\n'.join(L), focus)
    # CODER
    L = []
    for e in [e for e in ev if e['role'] == 'extractor' and not e['sub']]:
        if e['kind'] == 'tool' and e['name'] == 'append_new_attribute':
            a = dict(e['args'])
            code = a.pop('command', a.pop('_raw', ''))
            nm = a.pop('name', '')
            rest = '; '.join(f'{k}: {trunc(v, 250)}' for k, v in a.items())
            L.append(f"- append_new_attribute `{nm}`\n```python\n{code}\n```\n  rationale: {rest}\n  RESULT: {trunc(e['result'], 300)}")
        elif e['kind'] == 'tool':
            L.append(f"- TOOL CALL `{e['name']}`: {trunc(fmt_args(e['args'], n=300), 500)}\n  RESULT: {trunc(e['result'], 300)}")
        elif e['kind'] == 'text' and (e.get('content') or '').strip():
            L.append(f"- TEXT: {trunc(e['content'], 300)}")
        elif e['kind'] == 'nomsg':
            L.append('- (failed LLM call, no output)')
    secs['C'] = '\n'.join(L)
    # EVALUATOR
    L = []
    for e in [e for e in ev if e['role'] == 'tester']:
        tag = 'assessment' if e['stage'] == 'assessment' else 'pruning'
        if e['kind'] == 'tool':
            if e['name'] == 'attribute_pruning_tool':
                L.append(f"- [{tag}] PRUNE `{json.dumps(e['args'], ensure_ascii=False)[:1500]}`\n  RESULT: {trunc(e['result'], 400)}")
            else:
                L.append(f"- [{tag}] TOOL CALL `{e['name']}`: {trunc(fmt_args(e['args'], n=400), 600)}\n  RESULT: {trunc(e['result'], 400)}")
        elif e['kind'] == 'text' and (e.get('content') or '').strip():
            L.append(f"- [{tag}] TEXT:\n{trunc(e['content'], 6000 if tag == 'assessment' else 2500)}")
    secs['E'] = '\n'.join(L)
    return secs


def assemble(uid, phi_txt, secs, scale=1.0):
    p_body, focus = secs['P']
    c, e = secs['C'], secs['E']
    def cap(s, n):
        n = int(n * scale)
        return s if len(s) <= n else s[:n] + f'\n…[section truncated, {len(s) - n} chars omitted]'
    out = [f'# Annotation unit {uid}\n',
           f'**Planted hypothesis phi (for the annotator; the agents were not told):** `{phi_txt}`\n',
           'Roles: PLANNER proposes a focus, CODER writes features, EVALUATOR assesses and prunes. Each section shows only what that role produced (its tool calls, tool results and text).\n',
           '## PLANNER turn\n', cap(blind(p_body), 6000), '\n### Final focus handed to the CODER\n', blind(focus),
           '\n## CODER turn\n', cap(blind(c), 11000), '\n## EVALUATOR turn\n', cap(blind(e), 6500)]
    return '\n'.join(out)


def main():
    os.makedirs(PK, exist_ok=True)
    runs = {r['row_id']: r for r in F.list_runs()}
    key_rows = []
    for _, p in P.iterrows():
        run = runs[p.row_id]
        meta = F.load_meta(p.task)
        phi_txt = meta['phi_pandas_expr']
        secs = render(run, (p.fold, p.iteration), phi_txt)
        scale = 1.0
        txt = assemble(p.unit_id, phi_txt, secs, scale)
        while len(txt) > LIMIT and scale > 0.3:
            scale -= 0.1
            txt = assemble(p.unit_id, phi_txt, secs, scale)
        with open(os.path.join(PK, f'{p.unit_id}.md'), 'w', encoding='utf-8') as f:
            f.write(txt)
        c = d[(d.row_id == p.row_id) & (d.fold == p.fold) & (d.iteration == p.iteration)].iloc[0]
        key_rows.append(dict(unit_id=p.unit_id, arm=p.arm, row_id=p.row_id, session=run['session'], task=p.task,
                             fold=p.fold, iteration=p.iteration, stratum=p.stratum,
                             recovered_fold=int(c.rec_any), rec_sel_fold=int(c.rec_sel), E_phi_code_cycle=int(c.E_phi_code),
                             focus_phi=int(c.focus_phi), S_reason_phi=int(c.S_reason_phi), packet_chars=len(txt),
                             context_visible=('CODER saw only the focus message (+ its own system prompt); EVALUATOR saw the metric table / '
                                              'assessment markdown; PLANNER saw focus history + test results via tools'
                                              if p.arm == 'full' else
                                              'single shared conversation: CODER saw the PLANNER\'s tool calls, tool results and focus of this '
                                              'cycle; EVALUATOR saw PLANNER + CODER turns of this cycle')))
    pd.DataFrame(key_rows).to_csv(os.path.join(STUDY, 'packets_key.csv'), index=False)
    print(pd.DataFrame(key_rows).groupby(['task', 'arm', 'stratum']).size())
    print('max chars', max(k['packet_chars'] for k in key_rows))


if __name__ == '__main__':
    main()
