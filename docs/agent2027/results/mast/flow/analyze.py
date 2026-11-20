"""RQ1 analyses on flow/cycles.csv (EXPLORATORY, post hoc; run-clustered bootstrap B=2000, seed 42)."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
B = 2000
d = pd.read_csv(os.path.join(HERE, 'cycles.csv'))
d = d.sort_values(['row_id', 'fold', 'iteration'])
# first discovery per fold
first_e = d[d.E_phi_code == 1].groupby(['row_id', 'fold']).iteration.min().rename('first_E')
d = d.merge(first_e, on=['row_id', 'fold'], how='left')
d['pre_disc'] = (d.first_E.isna() | (d.iteration <= d.first_E)).astype(int)
d.to_csv(os.path.join(HERE, 'cycles_analysis.csv'), index=False)
TASKS = ['p2', 'p3', 'p4', 'p5', 'p6a', 'p6b']


def cond(x, a, b):
    den = (x[b] == 1).sum() if b else len(x)
    if den == 0:
        return np.nan
    num = ((x[a] == 1) & (x[b] == 1)).sum() if b else (x[a] == 1).sum()
    return num / den


def boot(df, fn, seed=42):
    runs = df.row_id.unique()
    groups = {r: df[df.row_id == r] for r in runs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        s = rng.choice(runs, len(runs), replace=True)
        out.append(fn(pd.concat([groups[r] for r in s])))
    out = np.array(out, float)
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)


def boot_diff(df, fn, seed=42):
    """single - full, resampling runs within each arm."""
    rng = np.random.default_rng(seed)
    g = {a: {r: df[df.row_id == r] for r in df[df.arm == a].row_id.unique()} for a in ['full', 'single']}
    out = []
    for _ in range(B):
        v = {}
        for a in g:
            rs = list(g[a])
            s = rng.choice(rs, len(rs), replace=True)
            v[a] = fn(pd.concat([g[a][r] for r in s]))
        out.append(v['single'] - v['full'])
    out = np.array(out, float)
    return np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5)


def fmt(p, ci, num=None, den=None):
    s = 'NA' if p != p else f'{p:.3f}'
    if num is not None:
        s = f'{num}/{den} = ' + s
    return s + (f' [{ci[0]:.3f}, {ci[1]:.3f}]' if ci and ci[0] == ci[0] else ' [NA]')


STEPS = [
    ('P(S_reason_phi)', 'S_reason_phi', None),
    ('P(S_cols_all)', 'S_cols_all', None),
    ('P(focus_phi | S_reason_phi)', 'focus_phi', 'S_reason_phi'),
    ('P(focus_phi | not S_reason_phi)', 'focus_phi', 'not_S'),
    ('P(focus_phi)', 'focus_phi', None),
    ('P(E_phi_code | focus_phi)', 'E_phi_code', 'focus_phi'),
    ('P(E_phi_code | not focus_phi)', 'E_phi_code', 'not_F'),
    ('P(E_phi_code | focus_phi_cols)', 'E_phi_code', 'focus_phi_cols'),
    ('P(E_phi_code)', 'E_phi_code', None),
]


def prep(x):
    x = x.copy()
    x['S_cols_all'] = (x.S_cols_k == x.n_phi_cols).astype(int)
    x['not_S'] = 1 - x.S_reason_phi
    x['not_F'] = 1 - x.focus_phi
    return x


def funnel(sub_name, dd):
    lines = [f'\n### Funnel ({sub_name})\n',
             '| task | arm | runs | cycles | ' + ' | '.join(s[0] for s in STEPS) + ' |',
             '|' + '---|' * (4 + len(STEPS))]
    rows = []
    for t in TASKS:
        for arm in ['full', 'single']:
            x = prep(dd[(dd.task == t) & (dd.arm == arm)])
            if not len(x):
                continue
            cells = []
            for lab, a, b in STEPS:
                p = cond(x, a, b)
                num = int(((x[a] == 1) & (x[b] == 1)).sum()) if b else int((x[a] == 1).sum())
                den = int((x[b] == 1).sum()) if b else len(x)
                ci = boot(x, lambda z, a=a, b=b: cond(z, a, b))
                cells.append(fmt(p, ci, num, den))
                rows.append(dict(subset=sub_name, task=t, arm=arm, step=lab, num=num, den=den, p=p, lo=ci[0], hi=ci[1]))
            lines.append(f'| {t} | {arm} | {x.row_id.nunique()} | {len(x)} | ' + ' | '.join(cells) + ' |')
    return lines, rows


def fold_level(dd):
    f = dd.groupby(['arm', 'task', 'row_id', 'fold']).agg(
        E=('E_phi_code', 'max'), E_att=('E_phi_attempt', 'max'), S=('S_reason_phi', 'max'), F=('focus_phi', 'max'),
        T=('T_mentions_phi_feat', 'max'), rec_sel=('rec_sel', 'max'), rec_any=('rec_any', 'max'),
        Tpr=('T_pruned_phi', 'max')).reset_index()
    return f


def main():
    out = ['# RQ1 analysis tables (EXPLORATORY, post hoc; not part of the pre-registered Holm family)\n',
           'Uncertainty: 95% percentile intervals, bootstrap over runs (B=2000, seed 42), runs resampled within arm x task.\n']
    allrows = []
    l, r = funnel('all cycles', d)
    out += l
    allrows += r
    l, r = funnel('pre-discovery cycles: fold not yet phi-coded, up to and including the first E_phi_code cycle', d[d.pre_disc == 1])
    out += l
    allrows += r
    pd.DataFrame(allrows).to_csv(os.path.join(HERE, 'funnel.csv'), index=False)

    # differences single-full on key steps (pre-discovery)
    out.append('\n### Single minus Full on funnel steps (pre-discovery cycles), run-bootstrap 95% CI\n')
    out.append('| task | step | full | single | diff [CI] |\n|---|---|---|---|---|')
    dp = d[d.pre_disc == 1]
    for t in ['p2', 'p4', 'p6b']:
        x = prep(dp[dp.task == t])
        for lab, a, b in STEPS:
            pf = cond(x[x.arm == 'full'], a, b)
            ps = cond(x[x.arm == 'single'], a, b)
            ci = boot_diff(x, lambda z, a=a, b=b: cond(z, a, b))
            out.append(f'| {t} | {lab} | {pf:.3f} | {ps:.3f} | {ps - pf:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |')

    # fold level
    f = fold_level(d)
    f.to_csv(os.path.join(HERE, 'folds.csv'), index=False)
    out.append('\n### Fold level (episode = run x fold)\n')
    out.append('| task | arm | folds | any S_reason_phi | any focus_phi | any E_phi_code | rec_any (official) | rec_sel (official) | P(rec_sel given E_phi_code) | P(T mentions phi feat given E) | P(T pruned a phi feat given E) |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for t in TASKS:
        for arm in ['full', 'single']:
            x = f[(f.task == t) & (f.arm == arm)]
            if not len(x):
                continue
            e = x[x.E == 1]
            def pr(col, xx):
                return f'{int(xx[col].sum())}/{len(xx)}'
            sel = boot(x, lambda z: z[z.E == 1].rec_sel.mean() if (z.E == 1).any() else np.nan)
            out.append(f'| {t} | {arm} | {len(x)} | {pr("S", x)} | {pr("F", x)} | {pr("E", x)} | {pr("rec_any", x)} | {pr("rec_sel", x)} | '
                       f'{pr("rec_sel", e)} [{sel[0]:.2f}, {sel[1]:.2f}] | {pr("T", e)} | {pr("Tpr", e)} |')

    # first-mention per fold
    out.append('\n### First cycle (iteration_in_fold, 0-9) at which each stage carries phi, per fold\n')
    out.append('Among folds where the stage ever carries phi; mean / median; and ordering relative to first E_phi_code.\n')
    out.append('| task | arm | stage | folds with stage | mean first iter | median | among E-folds: stage before first E | same cycle | after / never |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    fm_rows = []
    for t in TASKS:
        for arm in ['full', 'single']:
            x = d[(d.task == t) & (d.arm == arm)]
            if not len(x):
                continue
            fe = x[x.E_phi_code == 1].groupby(['row_id', 'fold']).iteration.min()
            for st in ['S_cols_all', 'S_reason_phi', 'focus_phi_cols', 'focus_phi', 'E_phi_code', 'T_mentions_phi_feat']:
                xx = prep(x)
                fs = xx[xx[st] == 1].groupby(['row_id', 'fold']).iteration.min()
                before = same = after = 0
                for k, v in fe.items():
                    if st == 'E_phi_code':
                        continue
                    s = fs.get(k)
                    if s is None or s != s or s > v:
                        after += 1
                    elif s < v:
                        before += 1
                    else:
                        same += 1
                fm_rows.append(dict(task=t, arm=arm, stage=st, n=len(fs), mean=fs.mean(), median=fs.median()))
                out.append(f'| {t} | {arm} | {st} | {len(fs)} | {fs.mean():.2f} | {fs.median() if len(fs) else float("nan"):.1f} | '
                           + (f'{before} | {same} | {after} |' if st != 'E_phi_code' else '- | - | - |'))
    pd.DataFrame(fm_rows).to_csv(os.path.join(HERE, 'first_mention.csv'), index=False)

    # focus specificity
    out.append('\n### Focus specificity (all cycles)\n')
    out.append('| task | arm | cycles | mean focus chars | mean raw-column names in focus | P(focus names >=1 phi col) | P(all phi cols) | P(phi op words) | P(explicit formula linking phi cols) |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for t in TASKS + ['ALL']:
        for arm in ['full', 'single']:
            x = d[(d.arm == arm) & ((d.task == t) | (t == 'ALL'))]
            out.append(f'| {t} | {arm} | {len(x)} | {x.focus_len.mean():.0f} | {x.focus_ncols.mean():.2f} | {(x.focus_cols_k > 0).mean():.3f} | '
                       f'{x.focus_phi_cols.mean():.3f} | {x.focus_op.mean():.3f} | {x.focus_formula.mean():.3f} |')
    out.append('\nSingle minus Full, run-bootstrap 95% CI (all tasks pooled, runs resampled within arm):\n')
    for col in ['focus_len', 'focus_ncols', 'focus_formula', 'focus_phi_cols', 'focus_op']:
        ci = boot_diff(d, lambda z, c=col: z[c].mean())
        diff = d[d.arm == 'single'][col].mean() - d[d.arm == 'full'][col].mean()
        out.append(f'- {col}: {diff:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]')

    # effort / errors
    out.append('\n### Effort and errors per cycle (means)\n')
    cols = ['S_n_calls', 'S_n_search_sub_calls', 'E_n_calls', 'E_n_append', 'E_n_accepted', 'tok_compl_S', 'tok_compl_E',
            'tok_compl_T', 'tok_prompt_E', 'err_S', 'err_E', 'err_T', 'T_n_pruned']
    out.append('| task | arm | ' + ' | '.join(cols) + ' |')
    out.append('|' + '---|' * (2 + len(cols)))
    for t in TASKS:
        for arm in ['full', 'single']:
            x = d[(d.task == t) & (d.arm == arm)]
            out.append(f'| {t} | {arm} | ' + ' | '.join(f'{x[c].mean():.1f}' for c in cols) + ' |')
    open(os.path.join(HERE, 'tables.md'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
