"""New offline analysis; requires original archived observations."""
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_outputs'
FIELDS=['grounded_correctness','evidence_passage_correctness','hallucination_absence','appropriate_abstention','semantic_consistency','conflict_resolved_correctly']

def rows(name):
    return [json.loads(l) for l in (ROOT/'data'/name).read_text().splitlines() if l.strip()]

def main():
    p1={(r['query_id'],r['condition']):r for r in rows('jury_pass1.jsonl')}
    p2={(r['query_id'],r['condition']):r for r in rows('jury_pass2.jsonl')}
    answers=rows('openweb_answers.jsonl')
    search=rows('search_results.jsonl')
    qids=sorted({r['query_id'] for r in search})
    expected={(q,c) for q in qids for c in ['OW-G','OW-Q']}
    if len(qids)!=45 or set(p1)!=expected or set(p2)!=expected:
        raise ValueError('Require 45 questions and exactly 90 matched records per jury pass.')
    if len(answers)!=90 or {(r['query_id'],r['condition']) for r in answers}!=expected:
        raise ValueError('Answer pairs/count mismatch')
    if not all(type(p[k].get(f)) is int and p[k][f] in (0,1) for p in [p1,p2] for k in expected for f in FIELDS):
        raise ValueError('Scores must be binary integers')
    keys=sorted(expected)
    a=np.array([p1[k][f] for k in keys for f in FIELDS])
    b=np.array([p2[k][f] for k in keys for f in FIELDS])
    totals1={k:sum(p1[k][f] for f in FIELDS) for k in keys}
    totals2={k:sum(p2[k][f] for f in FIELDS) for k in keys}
    comp={k:(totals1[k]+totals2[k])/2 for k in keys}
    result={'status':'new offline reanalysis of archived records','n_questions':45}
    result['reliability']={'raw_binary_agreement':float(np.mean(a==b)), 'cohens_kappa_pooled':float(cohen_kappa_score(a,b)), 'exact_composite_agreement':float(np.mean([totals1[k]==totals2[k] for k in keys])), 'pearson_composite':float(stats.pearsonr([totals1[k] for k in keys],[totals2[k] for k in keys]).statistic)}
    result['conditions']={}
    for c in ['OW-G','OW-Q']:
        ks=[k for k in keys if k[1]==c]
        mean=lambda f,k:(p1[k][f]+p2[k][f])/2
        gc=[mean('grounded_correctness',k)>=.5 for k in ks]
        ec=[mean('evidence_passage_correctness',k)>=.5 for k in ks]
        ar=[r for r in answers if r['condition']==c]
        result['conditions'][c]={'composite_mean':float(np.mean([comp[k] for k in ks])), 'composite_sd':float(np.std([comp[k] for k in ks],ddof=1)), 'field_rates':{f:float(np.mean([mean(f,k) for k in ks])) for f in FIELDS}, 'declared_abstention_rate':float(np.mean([r['abstained'] for r in ar])), 'any_evidence_rate':float(np.mean([r['evidence_present'] for r in ar])), 'joint_provenance':{'correct_conclusion_correct_citation':sum(g and e for g,e in zip(gc,ec)), 'correct_conclusion_wrong_citation':sum(g and not e for g,e in zip(gc,ec)), 'wrong_or_absent_conclusion':sum(not g for g in gc)}}
    g=np.array([comp[(q,'OW-G')] for q in qids]); z=np.array([comp[(q,'OW-Q')] for q in qids]); d=z-g
    sd=float(d.std(ddof=1)); se=sd/np.sqrt(len(d))
    t=stats.ttest_rel(z,g)
    result['paired_OWQ_minus_OWG']={'mean_difference':float(d.mean()),'ci95_normal_approximation':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],'paired_t':float(t.statistic),'paired_t_p':float(t.pvalue),'wilcoxon_p':float(stats.wilcoxon(z,g).pvalue) if np.any(d) else 1.0,'cohens_dz':float(d.mean()/sd) if sd else None}
    gv=np.array([r['generic_visible']>0 for r in search]); iv=np.array([r['inst_visible']>0 for r in search])
    onlyg=int(np.sum(gv & ~iv)); onlyi=int(np.sum(~gv & iv)); n=onlyg+onlyi
    result['visibility']={'generic_rate':float(gv.mean()),'qualified_rate':float(iv.mean()),'generic_only':onlyg,'qualified_only':onlyi,'mcnemar_exact_p':float(stats.binomtest(onlyg,n,.5).pvalue) if n else 1.0,'mean_hits_generic':float(np.mean([r['generic_visible'] for r in search])),'mean_hits_qualified':float(np.mean([r['inst_visible'] for r in search]))}
    benchmark=ROOT/'results/per_response_composite_scores.csv'
    if benchmark.is_file():
        with benchmark.open(encoding='utf-8-sig',newline='') as f: br=list(csv.DictReader(f))
        result['external_benchmark_descriptive_only']={c:float(np.mean([float(r['composite_score']) for r in br if r['condition']==c])) for c in ['vector','hybrid']}
    OUT.mkdir(exist_ok=True)
    (OUT/'openweb_reanalysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'openweb_composite_scores.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['query_id','condition','pass1_composite','pass2_composite','composite_mean'])
        for k in keys: w.writerow([*k,totals1[k],totals2[k],comp[k]])
    print('Wrote offline reanalysis; compare with archived results before claiming reproduction.')
if __name__=='__main__': main()
