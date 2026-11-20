"""Build flow/cycles.csv: one row per run x fold x iteration (EXPLORATORY, post hoc)."""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flowlib as F

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- keyword sets (documented in README)
DIV = r"ratio|divid|quotient|\s/\s|[a-z0-9_`'\)\]]\s?/\s?[a-z0-9_`'\(]|÷|\bper[ -](unit|area|cross|section|length|metre|meter)|normali[sz]"
MUL = r"product|multipl|\s\*\s|[a-z0-9_`'\)\]]\s?\*\s?[a-z0-9_`'\(]|×|\btimes\b|interaction"
OPS_TEXT = {
    # op = list of regex groups that must ALL match (AND), or any of 'complete' terms
    'p2': dict(all=[DIV, MUL], complete=r"elongation|deformation|deflection|axial (displacement|extension)|PL\s*/\s*A"),
    'p6a': dict(all=[DIV, MUL], complete=r"reynolds"),
    'p6b': dict(all=[DIV, MUL], complete=None),
    'p4': dict(all=[r"\bsin\b|\bsine\b|sinus|\bcos\b|cosine|fourier|periodic|cyclic|circadian|diurnal|24[- ]?h\b|24[- ]hour|"
                    r"hour[- _]of[- _](the[- _])?day|time[- _]of[- _]day|daily[- _]cycle|%\s?24|mod(ulo)?\s?24"], complete=None),
    'p3': dict(all=[DIV, r"threshold|cut-?off|>=|≥|\s>\s|exceed|above|\bflag|indicator|binary|high[- ]", r"self[_ -]?employ"],
               complete=None),
    'p5': dict(all=[r"torque|moment|lever|weight\s*[×*x]\s*distance|product|multipl",
                    r"squared|\*\*\s*2|\^\s*2|²|\bsquare|pow\(", r"\bsign\b|np\.sign|differen|minus|compar|heavier|balance|tip"],
               complete=None),
}
OPS_CODE = {
    'p2': [r"/|\.div\(|truediv", r"\*|\.mul\(|\.prod\(|multiply"],
    'p6a': [r"/|\.div\(|truediv", r"\*|\.mul\(|\.prod\(|multiply"],
    'p6b': [r"/|\.div\(|truediv", r"\*|\.mul\(|\.prod\(|multiply"],
    'p4': [r"np\.sin|\bsin\(", r"24|pi\s*/\s*12|0\.26179"],
    'p3': [r"/|\.div\(", r">=|>|\.ge\(|\.gt\(", r"self_employed"],
    'p5': [r"\*\*\s*2|\.pow\(\s*2|np\.square|\*\s*\w+\s*\*", r"-|np\.sign|\.sub\(|subtract|>|<"],
}


def col_regex(c):
    parts = re.split(r"[-_ ]", c.lower())
    return r"(?<![a-z0-9])" + r"[-_ ]?".join(re.escape(p) for p in parts) + r"(?![a-z0-9])"


def k_cols(text, cols):
    t = (text or '').lower()
    return sum(1 for c in cols if re.search(col_regex(c), t))


def op_text(text, task):
    t = (text or '').lower()
    o = OPS_TEXT[task]
    if o['complete'] and re.search(o['complete'], t, re.I):
        return 1
    return int(all(re.search(g, t, re.I) for g in o['all']))


def op_code(code, task):
    return int(all(re.search(g, code or '', re.I) for g in OPS_CODE[task]))


def formula_in_text(text, cols, task):
    """An explicit operator expression linking >=2 phi columns (P4: sin(...hours...) / 2*pi*h/24)."""
    t = (text or '').lower()
    if task == 'p4':
        return int(bool(re.search(r"(sin|cos)\s*\(?[^\n]{0,40}" + col_regex(cols[0]), t)) or
                   bool(re.search(col_regex(cols[0]) + r"[^\n]{0,15}(/|%)\s*24", t)) or
                   bool(re.search(r"2\s*[*·]?\s*(π|pi)[^\n]{0,30}" + col_regex(cols[0]), t)))
    rx = [col_regex(c) for c in cols]
    n = 0
    for i, a in enumerate(rx):
        for j, b in enumerate(rx):
            if i < j and (re.search(a + r"[`'\]\)\s]{0,4}\s*[*/×÷x·]\s*[`'\[\(\s]{0,4}" + b, t) or
                          re.search(b + r"[`'\]\)\s]{0,4}\s*[*/×÷x·]\s*[`'\[\(\s]{0,4}" + a, t)):
                n += 1
    return int(n >= 1)


def ev_text(e):
    if e['kind'] == 'text':
        return e.get('content') or ''
    a = e['args']
    return ' '.join(str(v) for v in a.values())


def raw_columns(meta):
    return list(meta.get('column_roles', {}).keys())


def main():
    epi = pd.read_csv(F.EPIS)
    epi = epi[epi.row_id.str.match(r'A(05|16)-')]
    rows = []
    for run in F.list_runs():
        task, rid = run['task'], run['row_id']
        meta = F.load_meta(task)
        cols = list(meta['phi']['cols'].values())
        rawcols = raw_columns(meta)
        n = len(cols)
        recs = F.load_records(run['path'])
        cyc = F.cycle_events(recs)
        rp = pd.read_csv(os.path.join(HERE, 'replay', f'{rid}.csv'))
        rp['code'] = rp['code'].fillna('')
        focus_log = {(d['fold_run_id'], d['iteration_in_fold']): d for d in F.load_jsonl(os.path.join(run['path'], 'focus_log.jsonl'))}
        it_log = {(d['fold_run_id'], d['iteration_in_fold']): d for d in F.load_jsonl(os.path.join(run['path'], 'iteration_log.jsonl'))}
        best = pd.read_csv(os.path.join(run['path'], 'fold_aggregation_best_iterations.csv'))
        sel_iter = dict(zip(best['test_fold_id'], best['iteration_in_fold']))
        ep = epi[epi.row_id == rid].set_index('fold')
        phi_active = {}  # fold -> set of active phi-feature names
        phi_names_ever = {}
        for key in sorted(k for k in cyc if k[0] is not None and k[1] is not None):
            fold, it = key
            ev = cyc[key]
            act = phi_active.setdefault(fold, set())
            ever = phi_names_ever.setdefault(fold, set())
            phi_before = int(len(act) > 0)
            # --- Scientist (own calls; WebSearchAgent sub-agent excluded)
            sev = [e for e in ev if e['role'] == 'scientist' and not e['sub']]
            stexts = [e for e in sev if e['kind'] == 'text' and (e.get('content') or '').strip()]
            focus = stexts[-1]['content'] if stexts else ''
            fl = focus_log.get(key, {}).get('focus', '')
            if not focus:
                focus = fl
            pre = [e for e in sev if not (e['kind'] == 'text' and (e.get('content') or '') == focus)]
            s_txt = '\n'.join(ev_text(e) for e in pre)
            s_search_q = '\n'.join(str(e['args'].get('query', '')) for e in sev if e['name'] == 'search_tool')
            # --- Extractor (replay table)
            rk = rp[(rp.fold == fold) & (rp.iteration == it)]
            acc = rk[rk.status == 'accepted']
            e_phi_code = int(((acc.rec == 1)).any())
            e_phi_attempt = int((rk.rec == 1).any())
            e_crit_any = int((rk.crit == 1).any())
            cols_k_code = [k_cols(c, cols) for c in acc.code]
            e_cols_max = max(cols_k_code) if cols_k_code else 0
            e_phi_cols = int(e_cols_max == n)
            e_phi_op_static = int(any(k == n and op_code(c, task) for k, c in zip(cols_k_code, acc.code)))
            phi_new = set(acc[acc.rec == 1].name.astype(str))
            # restored phi features (name previously phi)
            restored = set(rk[(rk.status == 'restored') & rk.name.astype(str).isin(ever)].name.astype(str))
            act |= phi_new | restored
            ever |= phi_new
            eev = [e for e in ev if e['role'] == 'extractor']
            # --- Tester
            tev = [e for e in ev if e['role'] == 'tester']
            t_assess = [e.get('content') or '' for e in tev if e['stage'] == 'assessment' and e['kind'] == 'text']
            t_assess = t_assess[-1] if t_assess else ''
            if not t_assess:
                p = os.path.join(run['path'], 'tester_reports', f"tester_report_iteration_{ev[0]['index']}.md")
                if os.path.exists(p):
                    t_assess = open(p, encoding='utf-8').read()
            t_prune_txt = [e.get('content') or '' for e in tev if e['stage'] == 'pruning' and e['kind'] == 'text']
            t_prune_txt = t_prune_txt[-1] if t_prune_txt else ''
            pruned = []
            for e in tev:
                if e['name'] == 'attribute_pruning_tool' and not e['error']:
                    lst = e['args'].get('attribute_names_list') or e['args'].get('attributes') or []
                    if isinstance(lst, str):
                        lst = re.findall(r"[\w\-]+", lst)
                    pruned += [str(x) for x in lst]
            phi_cand = act | phi_new
            t_mentions = int(any(re.search(r"(?<![\w])" + re.escape(nm) + r"(?![\w])", t_assess) for nm in phi_cand)) if phi_cand else 0
            t_pruned_phi = int(any(nm in pruned for nm in phi_cand))
            act -= set(pruned)
            phi_after = int(len(act) > 0)
            # --- tokens / calls / errors
            def tok(role, field):
                return int(sum((e['usage'] or {}).get(field, 0) or 0 for e in {id(x): x for x in ev}.values()
                               if e['role'] == role and (e['kind'] != 'tool' or e.get('first_in_call', True))))
            def ncalls(role, sub=None):
                return len({e['seq'] for e in ev if e['role'] == role and (sub is None or e['sub'] == sub)})
            err = {r: sum(1 for e in ev if e['role'] == r and e['kind'] in ('tool', 'nomsg') and e['error']) for r in
                   ['scientist', 'extractor', 'tester']}
            itl = it_log.get(key, {}).get('tokens', {})
            o = ep.loc[fold] if fold in ep.index else None
            rows.append(dict(
                arm=run['arm'], task=task, row_id=rid, session=run['session'], fold=fold, iteration=it,
                index=ev[0]['index'],
                rec_sel=int(o['rec_sel']) if o is not None else np.nan, rec_any=int(o['rec_any']) if o is not None else np.nan,
                selected_iter=sel_iter.get(fold, np.nan),
                n_phi_cols=n,
                S_cols_k=k_cols(s_txt, cols), S_op=op_text(s_txt, task),
                S_search_cols_k=k_cols(s_search_q, cols), S_search_op=op_text(s_search_q, task),
                S_n_calls=ncalls('scientist', False), S_n_search_sub_calls=ncalls('scientist', True),
                focus_cols_k=k_cols(focus, cols), focus_op=op_text(focus, task), focus_formula=formula_in_text(focus, cols, task),
                focus_len=len(focus), focus_ncols=k_cols(focus, rawcols),
                focus_matches_log=int(fl.strip()[:200] == focus.strip()[:200]) if fl else np.nan,
                E_n_append=len(rk), E_n_accepted=len(acc), E_n_eval_ok=int(rk.eval_ok.sum()),
                E_phi_code=e_phi_code, E_phi_attempt=e_phi_attempt, E_crit_any=e_crit_any,
                E_cols_max=e_cols_max, E_phi_cols=e_phi_cols, E_phi_op_static=e_phi_op_static,
                E_phi_names=';'.join(sorted(phi_new)), E_n_calls=ncalls('extractor', False),
                phi_active_before=phi_before, phi_active_after=phi_after,
                T_mentions_phi_feat=t_mentions, T_pruned_phi=t_pruned_phi, T_n_pruned=len(pruned),
                T_assess_cols_k=k_cols(t_assess, cols), T_assess_op=op_text(t_assess, task), T_assess_len=len(t_assess),
                tok_prompt_S=tok('scientist', 'prompt_tokens'), tok_compl_S=tok('scientist', 'completion_tokens'),
                tok_prompt_E=tok('extractor', 'prompt_tokens'), tok_compl_E=tok('extractor', 'completion_tokens'),
                tok_prompt_T=tok('tester', 'prompt_tokens'), tok_compl_T=tok('tester', 'completion_tokens'),
                itlog_compl_S=itl.get('scientist', {}).get('compl'), itlog_compl_E=itl.get('extractor', {}).get('compl'),
                itlog_compl_T=itl.get('tester', {}).get('compl'),
                err_S=err['scientist'], err_E=err['extractor'], err_T=err['tester'],
                n_nomsg=sum(1 for e in ev if e['kind'] == 'nomsg'),
            ))
        print(rid, task, 'cycles', sum(1 for r in rows if r['row_id'] == rid), flush=True)
    d = pd.DataFrame(rows)
    d['S_reason_phi'] = ((d.S_cols_k == d.n_phi_cols) & (d.S_op == 1)).astype(int)
    d['focus_phi'] = ((d.focus_cols_k == d.n_phi_cols) & (d.focus_op == 1)).astype(int)
    d['focus_phi_cols'] = (d.focus_cols_k == d.n_phi_cols).astype(int)
    d['is_selected_iter'] = (d.iteration == d.selected_iter).astype(int)
    d.to_csv(os.path.join(HERE, 'cycles.csv'), index=False)
    print(d.shape)


if __name__ == '__main__':
    main()
