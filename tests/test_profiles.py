import pandas as pd, yaml
from metaderm.reasoning.consistency import build_profile_map

def test_inconsistent_profile_reliability_zero():
    cfg=yaml.safe_load(open('configs/base.yaml',encoding='utf-8'))
    base={c:cfg['data']['concepts'][c][0] for c in cfg['data']['concepts']}
    df=pd.DataFrame([{**base,'binary_label':0},{**base,'binary_label':1}])
    pm=build_profile_map(df,cfg); item=next(iter(pm.values()))
    assert item['inconsistent'] is True
    assert item['reliability']==0
