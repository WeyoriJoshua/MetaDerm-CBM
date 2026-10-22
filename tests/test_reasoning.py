import numpy as np, yaml
from metaderm.reasoning.seven_point import SoftSevenPointReasoner
from metaderm.constants import CONCEPT_COLUMNS

def cfg():
    return yaml.safe_load(open('configs/base.yaml',encoding='utf-8'))

def test_high_risk_profile_scores_higher():
    c=cfg(); r=SoftSevenPointReasoner(c); probs={}
    for name in CONCEPT_COLUMNS:
        k=len(c['data']['concepts'][name]); probs[name]=np.zeros((2,k)); probs[name][:,0]=1
    # force suspicious state for sample 1 wherever available
    for name in CONCEPT_COLUMNS:
        states=c['data']['concepts'][name]; suspicious=c['reasoning']['criteria'][name]['suspicious_states']
        probs[name][1,:]=0; probs[name][1,states.index(suspicious[0])]=1
    out=r.predict(probs)
    assert out['score'][1] > out['score'][0]
    assert out['probability'][1] > out['probability'][0]
