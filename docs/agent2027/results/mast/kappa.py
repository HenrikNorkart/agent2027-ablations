import json,glob,pandas as pd,numpy as np
def load(d):
    rows=[]
    for f in sorted(glob.glob(d+'/*.jsonl')):
        for l in open(f,encoding='utf-8'):
            if l.strip(): x=json.loads(l); rows.append({'unit':x['unit_id'],**x['labels'],'MS':x['missed_signal'],'HF':x['handoff_fidelity']})
    return pd.DataFrame(rows).set_index('unit').sort_index()
A=load('annot_A'); B=load('annot_B'); print(len(A),len(B))
modes=[c for c in A.columns if c.startswith('FM')]+['MS']
def kappa(a,b):
    po=(a==b).mean(); pa=a.mean(); pb=b.mean(); pe=pa*pb+(1-pa)*(1-pb)
    return (po-pe)/(1-pe) if pe<1 else np.nan, po
out=[]
for m in modes:
    k,po=kappa(A[m],B[m]); out.append((m,int(A[m].sum()),int(B[m].sum()),int((A[m]&B[m]).sum()),round(po,3),round(k,3) if k==k else None))
K=pd.DataFrame(out,columns=['mode','A','B','both','agree','kappa']); print(K.to_string(index=False))
a=A[[c for c in modes if c.startswith('FM')]].values.ravel(); b=B[[c for c in modes if c.startswith('FM')]].values.ravel()
print('pooled FM kappa',kappa(pd.Series(a),pd.Series(b)))
print('HF weighted agreement exact',(A.HF==B.HF).mean())
key=pd.read_csv('packets_key.csv'); print(key.columns.tolist()); print(key.head(3).to_string())
