from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from system.runtime import (
    CONCEPTS,
    apply_expert_corrections,
    concept_reasoning_impact,
    concept_state_labels,
    concept_summary_table,
    gradcam,
    infer_image,
    load_frozen_artifacts,
)
from system.input_guard import assess_input, build_or_load_reference

UI_VERSION = 'MetaDerm-CBM Clinical Workstation v1.4'

st.set_page_config(
    page_title='MetaDerm-CBM | Clinical Workstation',
    page_icon='🩺',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(r'''
<style>
:root{--ink:#112235;--muted:#667085;--bg:#F5F7FA;--surface:#FFF;--line:#E3E9EF;--navy:#0B2A41;--teal:#0B7C74;--green:#18794E;--amber:#A85B00;--red:#B42318;--shadow:0 10px 30px rgba(16,24,40,.055)}
html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.stApp{background:var(--bg);color:var(--ink)}
header[data-testid="stHeader"]{display:none!important;height:0!important;min-height:0!important}
[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
#MainMenu,footer{visibility:hidden}.block-container{max-width:1540px;padding-top:.35rem!important;padding-bottom:3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#081F31 0%,#0C3048 58%,#0A4D49 100%)}
[data-testid="stSidebar"] *{color:#F8FAFC}[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 18px;background:white;border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);margin:0 0 14px}
.brand{display:flex;align-items:center;gap:12px}.logo{width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,#0B2A41,#0B7C74);color:white;font-size:1.2rem;font-weight:900;box-shadow:0 7px 18px rgba(11,124,116,.22)}
.brand-title{font-size:1.05rem;font-weight:900;color:var(--navy);line-height:1.1}.brand-sub{font-size:.74rem;color:var(--muted);margin-top:3px}.sys-pill{padding:7px 11px;border-radius:999px;background:#ECF8F5;color:#0B6D66;border:1px solid #BCE4DE;font-size:.73rem;font-weight:850}
.hero{position:relative;overflow:hidden;background:linear-gradient(118deg,#0A253A 0%,#123F5D 58%,#0B7C74 100%);border-radius:24px;padding:28px 30px;color:white;box-shadow:0 16px 40px rgba(12,45,72,.14);margin-bottom:16px}.hero:after{content:"";position:absolute;width:280px;height:280px;border-radius:50%;right:-80px;top:-150px;background:rgba(255,255,255,.08)}
.hero .eyebrow{font-size:.68rem;font-weight:850;letter-spacing:.14em;text-transform:uppercase;color:#A6DDD8}.hero h1{font-size:2rem;line-height:1.12;letter-spacing:-.035em;margin:7px 0}.hero p{font-size:.94rem;line-height:1.55;color:rgba(255,255,255,.82);margin:0;max-width:940px}
.safety{display:flex;gap:10px;align-items:flex-start;background:#FFF9EE;border:1px solid #F1D7A8;color:#714A14;border-radius:13px;padding:11px 14px;font-size:.82rem;margin-bottom:15px}
.card{background:white;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:var(--shadow)}.section-title{font-size:1.08rem;font-weight:900;color:var(--navy);margin-bottom:3px}.section-sub{font-size:.82rem;color:var(--muted);margin-bottom:12px}
.metric{background:white;border:1px solid var(--line);border-radius:16px;padding:15px 16px;min-height:105px;box-shadow:0 5px 16px rgba(16,24,40,.04)}.metric .label{font-size:.69rem;text-transform:uppercase;letter-spacing:.07em;font-weight:850;color:#667085}.metric .value{font-size:1.55rem;font-weight:900;color:#0B2A41;margin-top:8px}.metric .note{font-size:.72rem;color:#667085;margin-top:5px}
.workflow{display:grid;grid-template-columns:repeat(6,1fr);gap:7px;margin:10px 0 16px}.workflow div{background:white;border:1px solid var(--line);border-radius:11px;padding:9px 7px;text-align:center;font-size:.71rem;font-weight:800;color:#667085}.workflow .active{background:#EAF8F6;border-color:#AFE1DB;color:#0B746D}
.status{display:inline-flex;align-items:center;border-radius:999px;padding:8px 12px;font-size:.76rem;font-weight:900;letter-spacing:.02em}.predict,.pass{background:#EAF7F0;color:#18794E;border:1px solid #B9E2CA}.query,.review{background:#FFF4E5;color:#A85B00;border:1px solid #FFD39A}.abstain,.reject{background:#FEEFED;color:#B42318;border:1px solid #F9C2BD}
.gate-pass,.gate-review,.gate-reject{border-radius:18px;padding:18px;border:1px solid var(--line);background:white;box-shadow:var(--shadow)}.gate-pass{border-left:5px solid #18794E}.gate-review{border-left:5px solid #A85B00}.gate-reject{border-left:5px solid #B42318}.gate-title{font-size:1rem;font-weight:900;color:#0B2A41}.gate-text{font-size:.84rem;color:#596579;line-height:1.55;margin-top:6px}
.concept{background:white;border:1px solid var(--line);border-radius:15px;padding:14px;margin-bottom:9px}.concept .name{font-size:.69rem;text-transform:uppercase;letter-spacing:.065em;font-weight:850;color:#667085}.concept .state{font-size:.98rem;font-weight:900;color:#12324B;margin-top:4px}.concept .meta{font-size:.72rem;color:#667085;margin-top:5px}
.callout{background:#EDF6FC;border:1px solid #C8E0F0;border-left:4px solid #356D91;border-radius:12px;padding:12px 14px;font-size:.82rem;color:#35536A}.audit{background:#F8FAFC;border:1px solid var(--line);border-radius:14px;padding:13px;font-size:.81rem}
div[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:6px}div[data-testid="stTabs"] [data-baseweb="tab"]{background:white;border:1px solid var(--line);border-radius:11px 11px 0 0;font-weight:800;padding:0 16px}div[data-testid="stTabs"] [aria-selected="true"]{color:#0B7C74;border-bottom:3px solid #0B7C74}
[data-testid="stFileUploader"]{background:white;border:1px dashed #A8B8C7;border-radius:16px;padding:7px}.stButton button,.stDownloadButton button{border-radius:11px;font-weight:800}
@media(max-width:980px){.workflow{grid-template-columns:1fr 1fr}.hero h1{font-size:1.65rem}}
</style>
''', unsafe_allow_html=True)


def repo_root():
    return Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner='Loading frozen MetaDerm-CBM models...')
def load_system():
    d7 = os.environ.get('METADERM_DERM7PT_ROOT', '').strip()
    if not d7:
        raise RuntimeError('METADERM_DERM7PT_ROOT is not set. Launch through run_system.ps1.')
    return load_frozen_artifacts(repo_root(), Path(d7), seed=2026), Path(d7)


@st.cache_resource(show_spinner='Preparing dermoscopic input-safety reference...')
def load_guard(_artifacts, derm7pt_root):
    cache = repo_root() / 'results' / 'system_safety' / 'input_guard_reference_seed2026.npz'
    return build_or_load_reference(_artifacts, derm7pt_root, cache, max_reference=256, k=5, quantile=0.995)


def action_name(a):
    return {'predict':'PREDICT','query':'QUERY','abstain':'ABSTAIN'}.get(str(a).strip().lower(), str(a).upper())


def action_class(a):
    return {'PREDICT':'predict','QUERY':'query','ABSTAIN':'abstain'}.get(action_name(a),'query')


def metric(label, value, note):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)


def probbar(label, value, note):
    v=float(np.clip(value,0,1))
    st.markdown(f'<div style="margin:10px 0 14px"><div style="display:flex;justify-content:space-between;font-size:.8rem"><b style="color:#344054">{label}</b><b style="color:#0B2A41">{v:.1%}</b></div><div style="height:8px;background:#E9EEF3;border-radius:999px;margin-top:7px;overflow:hidden"><div style="height:100%;width:{v*100:.1f}%;background:linear-gradient(90deg,#0B7C74,#29A49A);border-radius:999px"></div></div><div style="font-size:.7rem;color:#667085;margin-top:5px">{note}</div></div>', unsafe_allow_html=True)


def overlay(image, cam):
    img=image.convert('RGB').resize((cam.shape[1],cam.shape[0]))
    fig,ax=plt.subplots(figsize=(6.2,5.0)); ax.imshow(img); ax.imshow(cam,cmap='jet',alpha=.42); ax.axis('off'); fig.tight_layout(pad=0); return fig


st.markdown(f'<div class="topbar"><div class="brand"><div class="logo">M</div><div><div class="brand-title">MetaDerm-CBM</div><div class="brand-sub">Metacognitive clinical AI research workstation</div></div></div><div class="sys-pill">● Frozen research model • v1.4</div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">Interpretable • selective • expert-aware</div><h1>Skin Lesion Decision-Support Workstation</h1><p>A cognitive concept-bottleneck workflow that separates image perception, semantic clinical concepts, structured reasoning, self-assessed reliability and selective action before optional expert concept verification.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="safety"><b>Research prototype.</b><span>Not a medical device and not intended for autonomous diagnosis. Histopathology and qualified clinical assessment remain definitive where indicated.</span></div>', unsafe_allow_html=True)

try:
    artifacts, derm7pt_root = load_system()
except Exception as exc:
    st.error(str(exc)); st.stop()

with st.sidebar:
    st.markdown('## MetaDerm-CBM'); st.caption('Clinical Workstation v1.4'); st.divider()
    page=st.radio('Navigation',['Analyze case','System & safety'],label_visibility='collapsed')
    st.divider(); st.markdown('### Frozen model'); st.write('**Backbone:** EfficientNet-B3'); st.write('**Reference seed:** 2026'); st.write(f'**Temperature:** {artifacts.blackbox_temperature:.4f}'); st.write(f'**τ low:** {artifacts.tau_low:.3f}'); st.write(f'**τ high:** {artifacts.tau_high:.3f}'); st.write(f'**Compute:** {artifacts.device.type.upper()}'); st.divider(); st.caption('The input-domain gate is a post-publication system safety layer and does not alter the manuscript results.')

if page == 'System & safety':
    st.markdown('### System & safety architecture')
    st.markdown('<div class="card"><div class="section-title">Two-stage safety architecture</div><div class="section-sub">The input gate operates before the frozen diagnostic pathway.</div><b>Stage 1 — Input suitability gate</b><br>Basic image checks plus one-class embedding distance to a deterministic Derm7pt reference bank. REVIEW and REJECT suppress diagnostic output.<br><br><b>Stage 2 — Frozen MetaDerm-CBM</b><br>Calibrated image classifier → seven semantic concepts → soft clinical reasoner → profile/reliability metacognition → PREDICT / QUERY / ABSTAIN.</div>', unsafe_allow_html=True)
    st.info('The gate is a defensive engineering feature, not a separately validated OOD detector. Dedicated OOD datasets are required before reporting rejection sensitivity/specificity.')
    st.stop()

st.markdown('<div class="section-title">Analyze a dermoscopic case</div><div class="section-sub">The image is screened for domain suitability before any diagnostic output is generated.</div>', unsafe_allow_html=True)
left,right=st.columns([1.3,1])
with left:
    uploaded=st.file_uploader('Dermoscopic image',type=['jpg','jpeg','png','bmp','tif','tiff'],label_visibility='collapsed')
with right:
    case_id=st.text_input('Case ID',placeholder='CASE-001'); case_note=st.text_input('Case note',placeholder='Optional research note')

if uploaded is None:
    st.markdown('<div class="card" style="text-align:center;padding:50px 20px"><div style="font-size:2.2rem">◉</div><div style="font-size:1.12rem;font-weight:900;color:#0B2A41;margin-top:9px">Ready for a dermoscopic image</div><div style="font-size:.84rem;color:#667085;margin-top:6px">Non-dermoscopic or strongly out-of-domain images are screened before diagnosis.</div></div>', unsafe_allow_html=True); st.stop()

if not case_id.strip(): case_id=Path(uploaded.name).stem.upper()
image=Image.open(uploaded).convert('RGB')

try:
    guard_ref=load_guard(artifacts,derm7pt_root)
    with st.spinner('Checking image-domain suitability...'):
        guard=assess_input(artifacts,image,guard_ref)
except Exception as exc:
    st.error('The input-safety reference could not be initialized. Diagnostic analysis has been withheld.'); st.exception(exc); st.stop()

gate_class={'PASS':'gate-pass','REVIEW':'gate-review','REJECT':'gate-reject'}[guard.status]
display_distance = 'n/a' if not np.isfinite(guard.distance) else f'{guard.distance:.4f}'
st.markdown(f'<div class="{gate_class}"><div style="display:flex;justify-content:space-between;gap:12px;align-items:center"><div class="gate-title">Input suitability gate</div><div class="status {guard.status.lower()}">{guard.status}</div></div><div class="gate-text">{guard.reason}</div><div style="font-size:.72rem;color:#667085;margin-top:8px">Domain-affinity score: {guard.domain_score:.3f} • embedding distance: {display_distance} • reference envelope: {guard.threshold:.4f} • n={guard.reference_size}</div></div>', unsafe_allow_html=True)

if guard.status != 'PASS':
    a,b=st.columns([.8,1.2])
    with a: st.image(image,caption='Uploaded image',use_container_width=True)
    with b:
        if guard.status=='REJECT': st.error('No melanoma probability, clinical concepts, or Grad-CAM output has been generated for this image.')
        else: st.warning('Diagnostic analysis is paused for this near-boundary input. Verify that it is a valid dermoscopic image and upload a clearer image if necessary.')
        st.markdown('**Why this behaviour matters:** closed-set classifiers can otherwise produce confident-looking outputs for unrelated images. The safety layer suppresses those outputs before they are presented as clinically meaningful.')
    st.stop()

with st.spinner('Running frozen MetaDerm-CBM diagnostic pathway...'):
    result=infer_image(artifacts,image)
concepts=concept_summary_table(artifacts,result['concept_probs']); impacts=concept_reasoning_impact(artifacts,result['concept_probs'])

st.markdown('<div class="workflow"><div class="active">1. Input gate</div><div class="active">2. Perception</div><div class="active">3. Concepts</div><div class="active">4. Reasoning</div><div class="active">5. Reliability</div><div class="active">6. Selective action</div></div>', unsafe_allow_html=True)
st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px"><div><div style="font-size:1.03rem;font-weight:900;color:#0B2A41">Case {case_id}</div><div style="font-size:.72rem;color:#667085">{uploaded.name}</div></div><div class="status {action_class(result["action"])}">{action_name(result["action"])}</div></div>', unsafe_allow_html=True)

overview,concept_tab,verify_tab,xai_tab,audit_tab=st.tabs(['Clinical overview','Concept profile','Expert verification','Visual explanation','Audit'])
with overview:
    a,b=st.columns([.9,1.35],gap='large')
    with a:
        st.image(image,caption=f'Case {case_id}',use_container_width=True)
        if case_note: st.caption(case_note)
    with b:
        st.markdown('<div class="section-title">Decision summary</div>',unsafe_allow_html=True)
        x,y,z=st.columns(3)
        with x: metric('Black-box',f'{result["blackbox_prob"]:.1%}','Calibrated melanoma probability')
        with y: metric('Reasoner',f'{result["reason_prob"]:.1%}','Concept-based clinical reasoning')
        with z: metric('Reliability',f'{result["reliability"]:.1%}','Frozen metacognitive score')
        st.markdown('<br>',unsafe_allow_html=True)
        probbar('Direct classifier',result['blackbox_prob'],'Frozen EfficientNet-B3 output'); probbar('Concept-based reasoner',result['reason_prob'],'Seven-point semantic pathway'); probbar('Profile reliability',result['profile_reliability'],'Concept-profile consistency signal')
        gap=abs(result['blackbox_prob']-result['reason_prob'])
        st.markdown(f'<div class="callout"><b>{action_name(result["action"])}</b><br>Neural–reasoner probability gap: {gap:.3f}. The frozen metacognitive model reports both pathways transparently.</div>',unsafe_allow_html=True)

with concept_tab:
    st.markdown('<div class="section-title">Seven-concept semantic profile</div><div class="section-sub">Predicted concept states with confidence and uncertainty.</div>',unsafe_allow_html=True)
    cols=st.columns(2)
    for i,row in concepts.iterrows():
        with cols[i%2]:
            st.markdown(f'<div class="concept"><div class="name">{row["concept"].replace("_"," ").title()}</div><div class="state">{row["predicted_state"]}</div><div class="meta">Confidence {row["confidence"]:.1%} • Normalized entropy {row["entropy"]:.3f}</div></div>',unsafe_allow_html=True)
    impact_df=pd.DataFrame([{'Concept':x['concept'].replace('_',' ').title(),'Reasoning impact':x['max_abs_reasoning_impact']} for x in impacts]).set_index('Concept')
    st.markdown('<div class="section-title">Decision-critical concept ranking</div>',unsafe_allow_html=True); st.bar_chart(impact_df)
    if impacts: st.markdown(f'<div class="callout"><b>Priority verification candidate:</b> {impacts[0]["concept"].replace("_"," ").title()}</div>',unsafe_allow_html=True)

with verify_tab:
    st.markdown('<div class="section-title">Expert-in-the-loop concept verification</div><div class="section-sub">Modify concept states and recompute structured reasoning without changing the image classifier.</div>',unsafe_allow_html=True)
    corrections={}; cs=st.columns(2)
    for i,concept in enumerate(CONCEPTS):
        with cs[i%2]:
            labels=concept_state_labels(artifacts,concept); row=concepts[concepts['concept']==concept].iloc[0]
            choice=st.selectbox(concept.replace('_',' ').title(),['Keep model prediction']+labels,key=f'corr_{concept}',help=f'Model: {row["predicted_state"]} ({row["confidence"]:.1%})')
            if choice!='Keep model prediction': corrections[concept]=choice
    if corrections:
        corrected=apply_expert_corrections(artifacts,result['concept_probs'],result['blackbox_prob'],corrections); st.divider(); q1,q2,q3,q4=st.columns(4)
        with q1: metric('Original reasoner',f'{result["reason_prob"]:.1%}','Before verification')
        with q2: metric('Updated reasoner',f'{corrected["reason_prob"]:.1%}',f'Δ {corrected["reason_prob"]-result["reason_prob"]:+.1%}')
        with q3: metric('Updated reliability',f'{corrected["reliability"]:.1%}',f'Δ {corrected["reliability"]-result["reliability"]:+.1%}')
        with q4: metric('Updated action',action_name(corrected['action']),'After verification')
        st.caption('The publication intervention used oracle-simulated concept correction; this panel demonstrates the intended expert workflow.')
    else: st.info('No expert concept corrections selected.')

with xai_tab:
    st.markdown('<div class="section-title">Post-hoc spatial localization</div><div class="section-sub">Grad-CAM is a qualitative localization aid, not a causal clinical explanation.</div>',unsafe_allow_html=True)
    l,r=st.columns([.9,1.1])
    with l: st.image(image,caption='Original dermoscopic image',use_container_width=True)
    with r:
        if st.button('Generate Grad-CAM',type='primary',use_container_width=True):
            with st.spinner('Computing spatial attribution...'): gc=gradcam(artifacts,image)
            st.pyplot(overlay(image,gc['cam']),clear_figure=True,use_container_width=True); st.markdown(f'<div class="audit"><b>Target layer:</b> {gc["layer"]}<br><b>Calibrated melanoma probability:</b> {gc["calibrated_prob"]:.1%}</div>',unsafe_allow_html=True)
        else: st.markdown('<div class="card" style="text-align:center;padding:44px 18px"><div style="font-size:1.7rem">◎</div><b>Grad-CAM is generated on demand</b><div style="font-size:.78rem;color:#667085;margin-top:5px">This avoids unnecessary GPU computation.</div></div>',unsafe_allow_html=True)

with audit_tab:
    audit={'ui_version':UI_VERSION,'case_id':case_id,'source_file':uploaded.name,'input_gate':guard.to_dict(),'model_seed':2026,'backbone':artifacts.cfg['model']['backbone'],'blackbox_probability':result['blackbox_prob'],'reasoner_probability':result['reason_prob'],'reliability':result['reliability'],'profile_reliability':result['profile_reliability'],'action':action_name(result['action']),'tau_low':artifacts.tau_low,'tau_high':artifacts.tau_high,'blackbox_temperature':artifacts.blackbox_temperature,'generated_at':datetime.utcnow().isoformat()+'Z'}
    st.json(audit); st.download_button('Download audit JSON',data=json.dumps(audit,indent=2),file_name=f'{case_id}_metaderm_audit.json',mime='application/json',use_container_width=True)
