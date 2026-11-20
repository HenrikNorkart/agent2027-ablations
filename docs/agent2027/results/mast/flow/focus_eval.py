"""Value-based check of the hand-off text: does the focus (or the Scientist's pre-focus text/notes) contain an
evaluable formula that recovers phi? (EXPLORATORY)

Formula candidates: backtick spans and `name = expr` fragments. Unicode operators are normalised (× · -> *, ÷ -> /,
^ -> **, ² -> **2), column names are bound to the raw columns, `name = expr` definitions are bound sequentially so
that `stress = a / b` then `stress * c` resolves. Each candidate is scored with the replay detector criterion.
Adds columns focus_phi_exact, S_phi_exact (scientist's own pre-focus text, incl. notes) to cycles_analysis.csv.
"""
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flowlib as F
import replay as RP

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = s.replace('×', '*').replace('·', '*').replace('÷', '/').replace('²', '**2').replace('−', '-').replace(' ', ' ')
    s = re.sub(r'(?<=[\w\)])\s*\^\s*', '**', s)
    s = s.replace('π', 'pi')
    return s


DF_REF = re.compile(r"df(?:_features)?\[['\"]([^'\"]+)['\"]\]")


def candidates(text):
    t = norm(text or '')
    spans = re.findall(r'`([^`\n]{3,300})`', t)
    spans += re.findall(r'(?m)^[^\n=]{0,80}?([A-Za-z_]\w*\s*=\s*[^\n]{3,300})$', t)
    out = []
    for s in spans:
        s = DF_REF.sub(lambda m: m.group(1), s).strip().rstrip('.,;')
        out.append(s)
    return out


def evaluate(text, raw, cols, phi, family, rawnum, best_raw):
    ns = {c: raw[c].to_numpy() for c in raw.columns if re.match(r'^[A-Za-z_]\w*$', c)}
    ns.update(np=np, log=np.log, log1p=np.log1p, sqrt=np.sqrt, sin=np.sin, cos=np.cos, pi=np.pi, exp=np.exp, abs=np.abs,
              sign=np.sign)
    # hyphenated raw names (P5) -> underscore aliases
    alias = {c: c.replace('-', '_') for c in raw.columns if '-' in c}
    for c, a in alias.items():
        ns[a] = raw[c].to_numpy()
    hit = 0
    for s in candidates(text):
        for c, a in alias.items():
            s = s.replace(c, a)
        m = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)$', s)
        name, expr = (m.group(1), m.group(2)) if m else (None, s)
        if '=' in expr.replace('==', '').replace('>=', '').replace('<=', '').replace('!=', ''):
            continue
        try:
            v = eval(expr, {'__builtins__': {}}, ns)
            v = np.asarray(v, float)
            if v.shape != (len(raw),):
                continue
        except Exception:
            continue
        if name:
            ns[name] = v
        rho = RP.spearman_abs(v, phi)
        n = RP.nmi(v, phi) if family == 'rule' else 0
        if rho >= 0.95 or (family == 'rule' and n >= 0.9):
            mono = any(RP.spearman_abs(v, cv) >= 0.8 and RP.weakly_monotone_of(v, cv) for cv in rawnum.values())
            if not mono and (n >= 0.9 or rho >= best_raw + 0.01):
                hit = 1
                break
    return hit


def main():
    d = pd.read_csv(os.path.join(HERE, 'cycles_analysis.csv'))
    runs = {r['row_id']: r for r in F.list_runs()}
    res = []
    cache = {}
    for rid in d.row_id.unique():
        run = runs[rid]
        task = run['task']
        meta = F.load_meta(task)
        if task not in cache:
            raw, _, numeric_raw = RP.build_raw(run['path'])
            phi = RP.phi_values(meta, raw)
            rawnum = {c: raw[c].to_numpy(float) for c in numeric_raw}
            best = max(RP.spearman_abs(v, phi) for v in rawnum.values())
            cache[task] = (raw, phi, rawnum, best)
        raw, phi, rawnum, best = cache[task]
        cols = list(meta['phi']['cols'].values())
        fam = meta['detector']['family']
        cyc = F.cycle_events(F.load_records(run['path']))
        for key, ev in cyc.items():
            if key[0] is None:
                continue
            sev = [e for e in ev if e['role'] == 'scientist' and not e['sub']]
            txt = [e for e in sev if e['kind'] == 'text' and (e.get('content') or '').strip()]
            focus = txt[-1]['content'] if txt else ''
            pre = '\n'.join(('\n'.join(str(v) for v in e['args'].values()) if e['kind'] == 'tool' else (e.get('content') or ''))
                            for e in sev if not (e['kind'] == 'text' and e.get('content') == focus))
            res.append(dict(row_id=rid, fold=key[0], iteration=key[1],
                            focus_phi_exact=evaluate(focus, raw, cols, phi, fam, rawnum, best),
                            S_phi_exact=evaluate(pre, raw, cols, phi, fam, rawnum, best)))
        print(rid, flush=True)
    r = pd.DataFrame(res)
    d = d.drop(columns=[c for c in ['focus_phi_exact', 'S_phi_exact'] if c in d.columns]).merge(r, on=['row_id', 'fold', 'iteration'], how='left')
    d.to_csv(os.path.join(HERE, 'cycles_analysis.csv'), index=False)
    c = pd.read_csv(os.path.join(HERE, 'cycles.csv'))
    c = c.drop(columns=[x for x in ['focus_phi_exact', 'S_phi_exact'] if x in c.columns]).merge(r, on=['row_id', 'fold', 'iteration'], how='left')
    c.to_csv(os.path.join(HERE, 'cycles.csv'), index=False)
    print(d.groupby(['task', 'arm'])[['focus_phi_exact', 'S_phi_exact', 'E_phi_code']].mean().round(3))


if __name__ == '__main__':
    main()
