import pandas as pd
d = 'in7xx/'
for base, add in (('rows_final.csv', 'rows_7xx.csv'), ('episodes_final.csv', 'ep_7xx.csv'),
                  ('funnel_episodes_5xx.csv', 'funnel_7xx/funnel_episodes.csv')):
    a = pd.read_csv(d + base); b = pd.read_csv(add)
    miss = set(b.columns) - set(a.columns); assert not miss, (base, miss)
    assert not set(b.row_id) & set(a.row_id), (base, 'dup rows')
    pd.concat([a, b], ignore_index=True).to_csv(d + base, index=False); print(base, len(a), '+', len(b))
