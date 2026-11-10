"""Generate manuscript Tables 1–3 from PRIVATE cohort/prediction data.

Never transcribe numeric manuscript values. Output aggregate-only tables and audit logs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from python.common import compute_metrics, patient_bootstrap, percentile_summary

TUMOURS = ['All (tumor-agnostic)','Lung','Breast','Colorectal','Pancreatic',
           'Gastric','Prostate','Endometrial','Melanoma','Renal Cell Carcinoma','Head & Neck']
METRICS = {'cyclic':['weighted_f1','accuracy'], 'bert':['weighted_f1','accuracy','macro_auroc']}
PROB_COLS = ['prob_class_1','prob_class_2','prob_class_3plus']

def fmt(estimate, lower, upper, *, percent=False):
    if not all(np.isfinite([estimate,lower,upper])):return 'NA'
    if percent: return f'{estimate:.1f} ({lower:.1f}, {upper:.1f})'
    return f'{estimate:.3f} ({lower:.3f}, {upper:.3f})'

def validate_interval(estimate,lower,upper,*,metric):
    if not all(np.isfinite([estimate,lower,upper])):
        raise ValueError(f'{metric}: non-finite point or CI')
    if lower > upper:raise ValueError(f'{metric}: reversed CI bounds')
    # A percentile CI need not always contain the point estimate, especially with
    # skewed/bootstrap distributions. Flag for review rather than rejecting.
    return lower <= estimate <= upper

def _groups(frame):
    if frame['patient_id'].isna().any():raise ValueError('Missing patient ID')
    return {p:g for p,g in frame.groupby('patient_id',sort=False)}

def _sample(groups,rng):
    ids=list(groups)
    return pd.concat([groups[ids[j]] for j in rng.integers(len(ids),size=len(ids))],ignore_index=True)

def _scores(frame,model):
    probs=frame[PROB_COLS].to_numpy(float) if model=='bert' else None
    return compute_metrics(frame['true_class'].astype(int).to_numpy(),
                           frame['pred_class'].astype(int).to_numpy(),
                           probs,include_auroc=model=='bert')

def evaluate_pair(in_domain,out_domain,model,*,n_boot=1000,seed=123):
    """Independent patient bootstrap; return domain metrics and relative change summaries."""
    if model not in METRICS:raise ValueError('model must be cyclic or bert')
    for name,df in [('in',in_domain),('out',out_domain)]:
        needed={'patient_id','true_class','pred_class'} | (set(PROB_COLS) if model=='bert' else set())
        if not needed.issubset(df):raise ValueError(f'{name} missing {sorted(needed-set(df))}')
        if df.empty:raise ValueError(f'{name} dataset is empty')
    groups_in,groups_out=_groups(in_domain),_groups(out_domain)
    point_in,point_out=_scores(in_domain,model),_scores(out_domain,model)
    rng=np.random.default_rng(seed)
    dist_in=[];dist_out=[];dist_change=[]
    for i in range(n_boot):
        x=_scores(_sample(groups_in,rng),model)
        y=_scores(_sample(groups_out,rng),model)
        dist_in.append(x);dist_out.append(y)
        dist_change.append({k:(y[k]-x[k])/x[k]*100 if x[k] != 0 else np.nan
                            for k in ['weighted_f1','accuracy']})
    results={}
    audit=[]
    for domain,point,dist in [('in',point_in,dist_in),('out',point_out,dist_out)]:
        boot=pd.DataFrame(dist)
        for metric in METRICS[model]:
            valid=boot[metric].dropna()
            lo,hi=valid.quantile([.025,.975]) if len(valid) else (np.nan,np.nan)
            est=point[metric]
            if len(valid)!=n_boot: audit.append(f'{domain} {metric}: {len(valid)}/{n_boot} valid replicates')
            if np.isfinite(est) and np.isfinite(lo) and np.isfinite(hi):
                if not validate_interval(est,lo,hi,metric=f'{domain} {metric}'):
                    audit.append(f'{domain} {metric}: point outside percentile CI; review, not necessarily error')
            results[(domain,metric)]=(est,lo,hi)
    changes=pd.DataFrame(dist_change)
    for metric in ['weighted_f1','accuracy']:
        a,b=point_in[metric],point_out[metric]
        est=(b-a)/a*100 if a else np.nan
        valid=changes[metric].dropna()
        lo,hi=valid.quantile([.025,.975]) if len(valid) else (np.nan,np.nan)
        if len(valid)!=n_boot:audit.append(f'change {metric}: {len(valid)}/{n_boot} valid replicates')
        if np.isfinite(est) and np.isfinite(lo) and np.isfinite(hi):
            if not validate_interval(est,lo,hi,metric=f'change {metric}'):
                audit.append(f'change {metric}: point outside percentile CI; review')
        results[('change',metric)]=(est,lo,hi)
    return results,audit

def make_performance_table(prediction_dir:Path,model:str,n_boot=1000,seed=123):
    """Expected files: {dataset_1|dataset_2}_{cyclic|bert}_{tumour_slug}_predictions_PRIVATE.csv.

    'all' denotes tumour-agnostic; missing tumour-specific BC files produce em dashes.
    """
    rows=[];audits=[]
    for idx,tumour in enumerate(TUMOURS):
        slug=('all' if idx==0 else tumour.lower().replace(' ','_').replace('&','and').replace('-','_'))
        paths={d:prediction_dir/f'{d}_{model}_{slug}_predictions_PRIVATE.csv'
               for d in ['dataset_1','dataset_2']}
        frames={d:pd.read_csv(p) for d,p in paths.items() if p.exists()}
        row={'Model':tumour}
        if not frames:
            audits.append(f'{model}/{tumour}: no prediction inputs')
            continue
        for domain,df in frames.items():
            if 'tumour_type' in df and idx>0:
                actual=set(df['tumour_type'].dropna().astype(str))
                if actual != {tumour}:audits.append(f'{model}/{tumour}/{domain}: tumour label mismatch {actual}')
        # Separate domain metrics when only one dataset is present.
        for domain,frame in frames.items():
            rng=np.random.default_rng(seed+idx)
            group=_groups(frame)
            pt=_scores(frame,model)
            boots=pd.DataFrame([_scores(_sample(group,rng),model) for _ in range(n_boot)])
            for metric in METRICS[model]:
                vals=boots[metric].dropna()
                lo,hi=vals.quantile([.025,.975]) if len(vals) else (np.nan,np.nan)
                row[f'{domain} {metric}']=fmt(pt[metric],lo,hi)
                if len(vals)!=n_boot:audits.append(f'{model}/{tumour}/{domain}/{metric}: {len(vals)} valid bootstraps')
                if np.isfinite(pt[metric]) and np.isfinite(lo) and np.isfinite(hi) and not validate_interval(pt[metric],lo,hi,metric=metric):
                    audits.append(f'{model}/{tumour}/{domain}/{metric}: point outside CI')
        if len(frames)==2:
            in_d,out_d=('dataset_2','dataset_1') if model=='cyclic' else ('dataset_1','dataset_2')
            pair,alerts=evaluate_pair(frames[in_d],frames[out_d],model,n_boot=n_boot,seed=seed+idx)
            audits.extend(f'{model}/{tumour}: {a}' for a in alerts)
            for metric in ['weighted_f1','accuracy']:
                row[f'percentage change {metric}']=fmt(*pair[('change',metric)],percent=True)
        for domain in ['dataset_1','dataset_2']:
            for metric in METRICS[model]:row.setdefault(f'{domain} {metric}','—')
        for metric in ['weighted_f1','accuracy']:row.setdefault(f'percentage change {metric}','—')
        rows.append(row)
    return pd.DataFrame(rows),audits

def _median_iqr(x):
    x=pd.to_numeric(x,errors='coerce').dropna()
    if x.empty:return 'NA'
    q=x.quantile([.25,.5,.75])
    return f'{q.loc[.5]:.0f} [{q.loc[.25]:.0f}-{q.loc[.75]:.0f}]'

def make_table1(cohorts:dict[str,pd.DataFrame]):
    """One row per medication, with diagnosis/follow-up dates and full (not collapsed) LoT.

    """
    result=[]
    for domain,df in cohorts.items():
        required=['patient_id','tumour_type','line_number','diagnosis_date','last_followup_date']
        missing=set(required)-set(df)
        if missing:raise ValueError(f'{domain}: missing {sorted(missing)}')
        for tumour in TUMOURS:
            sub=df if tumour==TUMOURS[0] else df.loc[df['tumour_type']==tumour]
            if sub.empty:continue
            work=sub.copy()
            work['diagnosis_date']=pd.to_datetime(work['diagnosis_date'],errors='coerce')
            work['last_followup_date']=pd.to_datetime(work['last_followup_date'],errors='coerce')
            work['line_number']=pd.to_numeric(work['line_number'],errors='coerce')
            per=work.groupby('patient_id',sort=False).agg(
                medication_records=('patient_id','size'),lines=('line_number','max'),
                diagnosis=('diagnosis_date','min'),followup=('last_followup_date','max'))
            months=(per['followup']-per['diagnosis']).dt.days/30.4375
            result.append({'Cohort':domain,'Tumour':tumour,'Patients n':len(per),
                           'Medication records per patient median [IQR]':_median_iqr(per['medication_records']),
                           'Lines of therapy median [IQR]':_median_iqr(per['lines']),
                           'Diagnosis to follow-up months median [IQR]':_median_iqr(months)})
    return pd.DataFrame(result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--dataset_1-cohort');p.add_argument('--dataset_2-cohort')
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=123)
    a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    audit=[]
    for model,table in [('cyclic','table2_cyclic'),('bert','table3_bert')]:
        df,notes=make_performance_table(Path(a.predictions_dir),model,a.bootstrap,a.seed)
        df.to_csv(out/f'{table}_aggregate.csv',index=False)
        audit.extend(notes)
    if a.dataset_1_cohort and a.dataset_2_cohort:
        make_table1({'Dataset 1':pd.read_csv(a.dataset_1_cohort),
                     'Dataset 2':pd.read_csv(a.dataset_2_cohort)}).to_csv(out/'table1_aggregate.csv',index=False)
    (out/'audit.txt').write_text('\n'.join(audit) if audit else 'No audit flags.\n',encoding='utf-8')
    print(f'Aggregate tables written to {out}; {len(audit)} audit flags')
if __name__=='__main__':main()
