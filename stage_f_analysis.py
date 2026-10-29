"""Stage F: statistical analysis.

Computes inter-pass reliability, rubric field rates, visibility with McNemar's
exact test, the paired OW-Q versus OW-G comparison, descriptive differences from
the previously reported institutional-assistant scores, category means, and the
joint conclusion-citation table.

Outputs: data/outputs/results.json, data/outputs/openweb_composite_scores.csv
"""
import csv
import json

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import cohen_kappa_score

import config

OUT = config.OUTPUTS
FIELDS = config.RUBRIC_FIELDS
CATS = config.CATEGORIES
R = {}

p1 = {(r["query_id"], r["condition"]): r for r in map(json.loads, open(f"{OUT}/jury_pass1.jsonl"))}
p2 = {(r["query_id"], r["condition"]): r for r in map(json.loads, open(f"{OUT}/jury_pass2.jsonl"))}
answers = [json.loads(l) for l in open(f"{OUT}/openweb_answers.jsonl")]
search = {r["query_id"]: r for r in map(json.loads, open(f"{OUT}/search_results.jsonl"))}
keys = [(a["query_id"], a["condition"]) for a in answers]

# ---------- 1. inter-pass reliability -------------------------------------------------
a1 = [p1[k][f] for k in keys for f in FIELDS]
a2 = [p2[k][f] for k in keys for f in FIELDS]
comp1 = {k: sum(p1[k][f] for f in FIELDS) for k in keys}
comp2 = {k: sum(p2[k][f] for f in FIELDS) for k in keys}
R["reliability"] = {
    "n_binary_judgments": len(a1),
    "cohens_kappa_pooled": round(cohen_kappa_score(a1, a2), 4),
    "raw_agreement_binary": round(float(np.mean(np.array(a1) == np.array(a2))), 4),
    "exact_composite_agreement": round(float(np.mean([comp1[k] == comp2[k] for k in keys])), 4),
    "pearson_r_composite": round(float(stats.pearsonr([comp1[k] for k in keys], [comp2[k] for k in keys])[0]), 4),
}
comp = {k: (comp1[k] + comp2[k]) / 2 for k in keys}

# ---------- 2. field-level rates by condition ----------------------------------------
R["field_rates"] = {}
for cond in ("OW-G", "OW-Q"):
    ks = [k for k in keys if k[1] == cond]
    R["field_rates"][cond] = {f: round(float(np.mean([(p1[k][f] + p2[k][f]) / 2 for k in ks])), 4) for f in FIELDS}
    R["field_rates"][cond]["composite_mean"] = round(float(np.mean([comp[k] for k in ks])), 4)
    R["field_rates"][cond]["composite_sd"] = round(float(np.std([comp[k] for k in ks], ddof=1)), 4)

# ---------- 3. declared abstention & evidence availability ---------------------------
R["behaviour"] = {}
for cond in ("OW-G", "OW-Q"):
    rs = [a for a in answers if a["condition"] == cond]
    R["behaviour"][cond] = {
        "n": len(rs),
        "declared_abstention_rate": round(float(np.mean([a["abstained"] for a in rs])), 4),
        "any_regulation_evidence_rate": round(float(np.mean([a["evidence_present"] for a in rs])), 4),
        "mean_evidence_chunks": round(float(np.mean([a["n_evidence_chunks"] for a in rs])), 3),
    }

# ---------- 4. source visibility (GEO metric) + McNemar -----------------------------
gvis = [1 if search[q]["generic_visible"] > 0 else 0 for q in sorted(search)]
ivis = [1 if search[q]["inst_visible"] > 0 else 0 for q in sorted(search)]
tbl = [[sum(1 for g, i in zip(gvis, ivis) if g == 1 and i == 1), sum(1 for g, i in zip(gvis, ivis) if g == 1 and i == 0)],
       [sum(1 for g, i in zip(gvis, ivis) if g == 0 and i == 1), sum(1 for g, i in zip(gvis, ivis) if g == 0 and i == 0)]]
mc = mcnemar(tbl, exact=True)
R["visibility"] = {
    "n_queries": len(gvis),
    "generic_visibility_rate": round(float(np.mean(gvis)), 4),
    "institution_qualified_visibility_rate": round(float(np.mean(ivis)), 4),
    "contingency_[[gi,g_noti],[notg_i,neither]]": tbl,
    "mcnemar_exact_p": float(mc.pvalue), "mcnemar_statistic": float(mc.statistic),
    "mean_visible_hits_generic": round(float(np.mean([search[q]["generic_visible"] for q in search])), 3),
    "mean_visible_hits_qualified": round(float(np.mean([search[q]["inst_visible"] for q in search])), 3),
}
by_cat_vis = {}
for c in CATS:
    qs_ = [q for q in search if search[q]["category"] == c]
    by_cat_vis[c] = {"n": len(qs_),
                     "generic": round(float(np.mean([1 if search[q]["generic_visible"] > 0 else 0 for q in qs_])), 4),
                     "qualified": 1.0}
R["visibility"]["by_category"] = by_cat_vis

# ---------- 5. paired OW-G vs OW-Q (same jury) -------------------------------------
def paired(x, y, label):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y - x
    t, tp = stats.ttest_rel(y, x)
    nz = d[d != 0]
    if len(nz) > 0:
        w, wp = stats.wilcoxon(y, x, zero_method="wilcox")
    else:
        w, wp = float("nan"), 1.0
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0
    pw = TTestPower().power(effect_size=abs(dz), nobs=len(d), alpha=0.05, alternative="two-sided") if dz else float("nan")
    try:
        need = TTestPower().solve_power(effect_size=abs(dz), power=0.8, alpha=0.05, alternative="two-sided") if abs(dz) > 1e-6 else float("inf")
    except Exception:
        need = float("nan")
    return {"label": label, "n_pairs": len(d), "mean_a": round(float(x.mean()), 4), "sd_a": round(float(x.std(ddof=1)), 4),
            "mean_b": round(float(y.mean()), 4), "sd_b": round(float(y.std(ddof=1)), 4),
            "mean_diff_b_minus_a": round(float(d.mean()), 4),
            "ci95": [round(float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))), 4),
                     round(float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d))), 4)],
            "paired_t": round(float(t), 4), "paired_t_p": round(float(tp), 5),
            "wilcoxon_p": round(float(wp), 5), "cohens_dz": round(dz, 4),
            "achieved_power": None if np.isnan(pw) else round(float(pw), 4),
            "n_for_80pct_power": None if (need != need or need == float("inf")) else int(np.ceil(need))}

qids = sorted({k[0] for k in keys})
R["ow_g_vs_ow_q"] = {"pooled": paired([comp[(q, "OW-G")] for q in qids], [comp[(q, "OW-Q")] for q in qids], "OW-Q minus OW-G, pooled")}
for c in CATS:
    cq = [q for q in qids if search[q]["category"] == c]
    R["ow_g_vs_ow_q"][c] = paired([comp[(q, "OW-G")] for q in cq], [comp[(q, "OW-Q")] for q in cq], f"OW-Q minus OW-G, {c}")

# ---------- 6. institutional assistant (previously reported, different jury) --------
# Reported for descriptive context only; evaluator equivalence is not established.
agent = {}
for r in csv.DictReader(open(config.ASSISTANT_SCORES_CSV, encoding="utf-8-sig")):
    agent[(r["query_id"], r["condition"])] = float(r["composite_score"])
R["agent_benchmark"] = {}
for acond in ("vector", "hybrid"):
    av = [agent[(q, acond)] for q in qids]
    for owcond in ("OW-G", "OW-Q"):
        ow = [comp[(q, owcond)] for q in qids]
        R["agent_benchmark"][f"{owcond}_vs_agent_{acond}"] = paired(av, ow, f"{owcond} minus agent-{acond}")
R["agent_published_means"] = {c: round(float(np.mean([agent[(q, c)] for q in qids])), 4) for c in ("vector", "hybrid")}

# ---------- 7. per-category composite means table ---------------------------------
R["category_means"] = {}
for c in CATS:
    cq = [q for q in qids if search[q]["category"] == c]
    R["category_means"][c] = {
        "n": len(cq),
        "OW-G": round(float(np.mean([comp[(q, "OW-G")] for q in cq])), 3),
        "OW-Q": round(float(np.mean([comp[(q, "OW-Q")] for q in cq])), 3),
        "agent_vector": round(float(np.mean([agent[(q, "vector")] for q in cq])), 3),
        "agent_hybrid": round(float(np.mean([agent[(q, "hybrid")] for q in cq])), 3),
    }

# ---------- 8. the provenance-fidelity cross-tab ---------------------------------
R["provenance_fidelity"] = {}
for cond in ("OW-G", "OW-Q"):
    ks = [k for k in keys if k[1] == cond]
    gc = [(p1[k]["grounded_correctness"] + p2[k]["grounded_correctness"]) / 2 >= 0.5 for k in ks]
    ec = [(p1[k]["evidence_passage_correctness"] + p2[k]["evidence_passage_correctness"]) / 2 >= 0.5 for k in ks]
    R["provenance_fidelity"][cond] = {
        "correct_conclusion_correct_citation": sum(1 for g, e in zip(gc, ec) if g and e),
        "correct_conclusion_wrong_citation": sum(1 for g, e in zip(gc, ec) if g and not e),
        "wrong_or_absent_conclusion": sum(1 for g in gc if not g),
        "n": len(ks),
    }

with open(f"{OUT}/results.json", "w") as f:
    json.dump(R, f, indent=2)

# composite score CSV for release
with open(f"{OUT}/openweb_composite_scores.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["query_id", "category", "condition", "pass1_composite", "pass2_composite", "composite_mean"])
    for k in keys:
        w.writerow([k[0], search[k[0]]["category"], k[1], comp1[k], comp2[k], comp[k]])

print("Wrote", OUT / "results.json")
