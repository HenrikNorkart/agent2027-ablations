import pandas as pd,math
exec(open('kappa.py',encoding='utf-8').read().split("out=[]")[0])
key=pd.read_csv('packets_key.csv').set_index('unit_id')
modes=[c for c in A.columns if c.startswith('FM')]+['MS']
C=(A[modes]&B[modes]).astype(int)   # consensus = both annotators
U=(A[modes]|B[modes]).astype(int)
C=C.join(key[['arm','task','stratum','row_id']]); U=U.join(key[['arm','task','stratum']])
C['HF']=((A.HF+B.HF)/2)
def fisher(a,n1,b,n2):
    # two-sided Fisher exact
    from math import comb
    K=a+b;N=n1+n2
    p=lambda x: comb(n1,x)*comb(n2,K-x)/comb(N,K)
    p0=p(a); return sum(p(x) for x in range(max(0,K-n2),min(K,n1)+1) if p(x)<=p0+1e-12)
rows=[]
for m in modes+['HF']:
    r={'mode':m}
    for arm in ['full','single']:
        s=C[C.arm==arm]; r[arm]= f"{s[m].sum():.0f}/{len(s)}" if m!='HF' else round(s[m].mean(),2)
        if m!='HF': r[arm+'_union']=f"{U[U.arm==arm][m].sum()}/{len(s)}"
    if m!='HF':
        a=C[C.arm=='full'][m].sum(); b=C[C.arm=='single'][m].sum(); r['p_fisher']=round(fisher(a,24,b,24),3)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print('\nby stratum (consensus):'); print(C.groupby(['stratum','arm'])[['FM-1.3','FM-1.5','FM-2.6','FM-3.3','MS']].sum().to_string())
print(C.groupby('arm').size(), C.groupby(['arm','task']).size().unstack())
print('runs per arm', C.groupby('arm').row_id.nunique())
C.to_csv('mast_consensus.csv')
