import numpy as np, yaml
from metaderm.reasoning.metacognition import metacognitive_features, reliability_score, assign_actions

def test_reliability_prefers_confident_agreeing_case():
    cfg=yaml.safe_load(open('configs/base.yaml',encoding='utf-8'))
    cps={
      'pigment_network':np.array([[.98,.01,.01],[1/3,1/3,1/3]]),
      'streaks':np.array([[.98,.01,.01],[1/3,1/3,1/3]]),
      'pigmentation':np.array([[.96,.01,.01,.01,.01],[.2,.2,.2,.2,.2]]),
      'regression_structures':np.array([[.97,.01,.01,.01],[.25,.25,.25,.25]]),
      'dots_and_globules':np.array([[.98,.01,.01],[1/3,1/3,1/3]]),
      'blue_whitish_veil':np.array([[.99,.01],[.5,.5]]),
      'vascular_structures':np.array([[.93,.01,.01,.01,.01,.01,.01,.01],[.125]*8]),
    }
    f=metacognitive_features(cps,np.array([.05,.5]),np.array([.06,.9]),np.array([1.,0.]))
    r=reliability_score(f,cfg)
    assert r[0]>r[1]
    a=assign_actions(r,0.25,0.75)
    assert a[0] in {'predict','query'}
