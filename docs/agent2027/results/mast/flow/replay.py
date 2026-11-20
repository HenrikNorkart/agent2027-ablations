"""Replay every append_new_attribute call of one run on the raw columns recovered from
df_entities_attributes_final.csv, and score each feature against phi with the repo detector's criterion
(|Spearman| >= 0.95; rule family P3/P5 also NMI >= 0.9; raw columns / monotone single-column transforms /
no gain over the best raw column (+0.01) excluded). EXPLORATORY.

Differences from the registered detector (recovery_detector.py): all 2000 rows instead of the fold's training
rows; raw NaNs are 0 (seed columns were filled with 0); the label column is unavailable (random stand-in), so
label-dependent code (target encodings) is evaluated on noise -- irrelevant to phi, which is label-free.

usage: python -X utf8 -I replay.py <row_id>   -> writes flow/replay/<row_id>.csv
"""
import os
import re
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import flowlib as F

warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'replay')
BLOCK = re.compile(r'\bimport\b|open\(|\bos\.|\bsys\.|subprocess|eval\(|exec\(|__|to_csv|to_pickle|to_parquet|'
                   r'\.write|shutil|requests|socket|\bglob\b|while\s', re.I)


def rank(x):
    return pd.Series(x).rank(method='average').to_numpy()


def clean_pair(f, phi):
    f = np.asarray(f, float)
    phi = np.asarray(phi, float)
    ok = np.isfinite(f) & np.isfinite(phi)
    if ok.mean() < 0.9 or ok.sum() < 10:
        return None
    return f[ok], phi[ok]


def spearman_abs(f, phi):
    p = clean_pair(f, phi)
    if p is None:
        return 0.0
    a, b = rank(p[0]), rank(p[1])
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return 0.0 if den == 0 else abs(float(a @ b) / den)


def _disc(x, bins=10):
    u = np.unique(x)
    if u.size <= bins:
        return np.searchsorted(u, x)
    e = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(e, x, side='right')


def _H(c):
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())


def nmi(f, phi):
    p = clean_pair(f, phi)
    if p is None:
        return 0.0
    ca, cb = _disc(p[0]), _disc(p[1])
    na, nb = int(ca.max()) + 1, int(cb.max()) + 1
    j = np.bincount(ca * nb + cb, minlength=na * nb).reshape(na, nb).astype(float)
    ha, hb = _H(j.sum(1)), _H(j.sum(0))
    if ha == 0 or hb == 0:
        return 0.0
    return float(max(0.0, ha + hb - _H(j.ravel())) / (0.5 * (ha + hb)))


def weakly_monotone_of(f, c):
    ok = np.isfinite(f) & np.isfinite(c)
    if ok.sum() < 3:
        return False
    f, c = f[ok], c[ok]
    o = np.argsort(c, kind='stable')
    cs, fs = c[o], f[o]
    gid = np.cumsum(np.concatenate(([True], cs[1:] != cs[:-1]))) - 1
    tol = 1e-9 * max(1.0, float(np.max(np.abs(fs))))
    gmin = np.full(gid[-1] + 1, np.inf)
    gmax = np.full(gid[-1] + 1, -np.inf)
    np.minimum.at(gmin, gid, fs)
    np.maximum.at(gmax, gid, fs)
    if np.any(gmax - gmin > tol):
        return False
    d = np.diff(gmin)
    return bool(np.all(d >= -tol) or np.all(d <= tol))


SEED_RE = re.compile(r"df\.groupby\('id'\)\['([^']+)'\]\.first\(\)")


def build_raw(path):
    ex = pd.read_csv(os.path.join(path, 'df_attributes_explanations_final.csv'))
    ent = pd.read_csv(os.path.join(path, 'df_entities_attributes_final.csv'))
    seeds = ex[ex['Seeded'] == True]
    raw = pd.DataFrame({'id': ent['id'].values})
    seed_feats = {}
    numeric_raw = []
    for _, s in seeds.iterrows():
        cmd = str(s['Pandas Command'])
        m = SEED_RE.search(cmd)
        if not m:
            continue
        col = m.group(1)
        vals = ent[s['Attribute']].values
        seed_feats[s['Attribute']] = vals
        mm = re.search(r"\.map\((\{.*?\})\)", cmd)
        if mm:
            mp = eval(mm.group(1), {})  # dict literal from our own seed table
            inv = {v: k for k, v in mp.items()}
            raw[col] = [inv.get(int(v), None) if v == v else None for v in vals]
        else:
            raw[col] = vals
            numeric_raw.append(col)
    rng = np.random.default_rng(0)
    for lab in ['target', 'label', 'y', 'alarm', 'class', 'default', 'outcome']:
        if lab not in raw.columns:
            raw[lab] = rng.integers(0, 2, len(raw))
    df_feat0 = pd.DataFrame({'id': ent['id'].values})
    for k, v in seed_feats.items():
        df_feat0[k] = v
    return raw, df_feat0, numeric_raw


def phi_values(meta, raw):
    expr = meta['phi_pandas_expr']
    return np.asarray(pd.eval(expr, local_dict={'df': raw, 'np': np}) if False else
                      eval(expr, {'np': np, 'pd': pd}, {'df': raw}), float)


def run_code(code, raw, dff):
    import textwrap
    code = textwrap.dedent(code).strip()
    ns = {'df': raw.copy(), 'df_features': dff.copy(), 'pd': pd, 'np': np}
    try:
        compile(code, '<f>', 'exec')
    except IndentationError:
        code = '\n'.join(l.strip() for l in code.split('\n'))
    exec(code, ns)
    res = ns.get('result')
    if res is None:
        raise ValueError('no result')
    tmp = dff[['id']].copy()
    tmp['__r'] = res
    return pd.to_numeric(tmp['__r'], errors='coerce').to_numpy(float)


def main(row_id):
    run = [r for r in F.list_runs() if r['row_id'] == row_id][0]
    meta = F.load_meta(run['task'])
    family = meta['detector']['family']
    raw, dff0, numeric_raw = build_raw(run['path'])
    phi = phi_values(meta, raw)
    rawnum = {c: raw[c].to_numpy(float) for c in numeric_raw}
    raw_rho = {c: spearman_abs(v, phi) for c, v in rawnum.items()}
    best_raw = max(raw_rho.values()) if raw_rho else 0.0
    recs = F.load_records(run['path'])
    cyc = F.cycle_events(recs)
    rows = []
    t0 = time.time()
    folds = sorted({k[0] for k in cyc if k[0] is not None})
    for fold in folds:
        dff = dff0.copy()
        keys = sorted(k for k in cyc if k[0] == fold and k[1] is not None)
        for key in keys:
            for e in cyc[key]:
                if e['kind'] != 'tool' or e['name'] != 'append_new_attribute':
                    continue
                a = e['args']
                name = str(a.get('name', a.get('attribute_name', '')))
                code = a.get('command', a.get('pandas_command', a.get('_raw', '')))
                code = str(code)
                res = str(e['result'])
                if 'added to df_features' in res:
                    status = 'accepted'
                elif 'restored to Active' in res:
                    status = 'restored'
                elif 'already exists' in res:
                    status = 'exists'
                else:
                    status = 'error'
                row = dict(row_id=row_id, fold=fold, iteration=key[1], seq=e['seq'], role=e['role'],
                           name=name, status=status, code=code, result=res[:300], eval_ok=0, eval_err='',
                           rho=np.nan, nmi=np.nan, crit=0, rec=0, excl='')
                chk = re.sub(r'^\s*(import (numpy|pandas|math)( as \w+)?(, *(numpy|pandas|math)( as \w+)?)*|from (numpy|pandas|math|scipy[\w.]*|sklearn[\w.]*) import [\w, ]+|import (scipy|sklearn)[\w.]*( as \w+)?)\s*$', '', code, flags=re.M)
                if BLOCK.search(chk):
                    row['eval_err'] = 'blocked'
                else:
                    try:
                        v = run_code(code, raw, dff)
                        row['eval_ok'] = 1
                        rho = spearman_abs(v, phi)
                        row['rho'] = rho
                        n = nmi(v, phi) if family == 'rule' else np.nan
                        row['nmi'] = n
                        crit = rho >= 0.95 or (family == 'rule' and n >= 0.9)
                        row['crit'] = int(crit)
                        if crit:
                            excl = ''
                            if name in numeric_raw:
                                excl = 'raw_name'
                            else:
                                for c, cv in rawnum.items():
                                    if spearman_abs(v, cv) >= 0.8 and weakly_monotone_of(v, cv):
                                        excl = 'monotone:' + c
                                        break
                            if not excl and rho >= 0.95 and not (family == 'rule' and n >= 0.9) \
                                    and rho < best_raw + 0.01:
                                excl = 'no_gain_over_raw'
                            row['excl'] = excl
                            row['rec'] = int(not excl)
                        if status in ('accepted',) and name:
                            dff[name] = v
                    except Exception as ex:
                        row['eval_err'] = (type(ex).__name__ + ': ' + str(ex))[:150]
                rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, f'{row_id}.csv'), index=False)
    print(row_id, run['task'], len(out), 'eval_ok', out['eval_ok'].mean().round(3) if len(out) else 0,
          'rec', int(out['rec'].sum()) if len(out) else 0, 'best_raw_rho', round(best_raw, 3),
          f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
