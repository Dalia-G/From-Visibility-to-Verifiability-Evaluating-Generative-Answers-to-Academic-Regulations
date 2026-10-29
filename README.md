# From Visibility to Verifiability: Evaluating Generative Answers to Academic Regulations

Replication package for the paper:

> D. Gamal, A. Sedky, and S. Mesbah, "From visibility to verifiability: Evaluating generative answers to academic regulations," College of Computing and Information Technology, Arab Academy for Science, Technology and Maritime Transport (AASTMT), Alexandria, Egypt.

The study poses 45 adversarial academic-regulation queries to an author-implemented open-web retrieval-augmented generation pipeline under two conditions:

| Code | Condition |
| --- | --- |
| OW-G | Generic student phrasing |
| OW-Q | Institution-qualified phrasing (query prefixed with "AASTMT") |

The queries cover three failure modes, with 15 queries each: cross-document conflict, rule-precedence override, and multi-hop exception. Responses are scored with a six-field binary rubric in two independent jury passes. Previously reported scores of an institutional assistant on the same queries are included for descriptive comparison only, because they were obtained with a differently configured jury.

## Repository contents

```
src/
  config.py               Paths and shared settings
  providers.py            Search, fetch, and LLM adapters (keys read from environment)
  stage_ab_search.py      A: query construction; B: web search and visibility
  stage_cd_answer.py      C: evidence extraction; D: answer generation
  stage_e_score.py        E: two-pass rubric scoring (requires ground truth, not included)
  stage_f_analysis.py     F: statistical analysis
  make_figures.py         Figures 1 and 2
data/
  inputs/
    queries.csv                     45 queries (query_id, category, query_text)
    assistant_composite_scores.csv  Previously reported institutional-assistant composites
  outputs/
    search_queries.jsonl            Distilled student search strings
    search_results.jsonl            Search results per query and condition (URL, title, domain)
    openweb_answers.jsonl           90 generated responses
    jury_pass1.jsonl                Rubric scores, pass 1
    jury_pass2.jsonl                Rubric scores, pass 2
    openweb_composite_scores.csv    Composite score per response
    results.json                    All reported statistics
figures/
  fig1_visibility.png
  fig2_composite.png
```

## What is not included

- **Regulation text.** The institutional regulation documents and the text extracted from fetched pages are not redistributed. Search results are released as URL, title, and domain only.
- **Ground truth.** The expert-authored governing articles and resolutions are withheld so the query set can continue to be used as a held-out benchmark. For the same reason, the jury's free-text rationales, which refer to the governing articles, are omitted from the released scores. Researchers may request the ground truth from the corresponding author.

## Reproducing the reported results

The analysis and figures can be reproduced from the released outputs without any API access:

```bash
pip install -r requirements.txt
cd src
python stage_f_analysis.py
python make_figures.py
```

This regenerates `data/outputs/results.json` and both figures. The values match those reported in the paper.

## Rerunning the pipeline

Stages A to D query live web search and LLM services. Copy `.env.example` to `.env`, fill in your own credentials, and export them:

```bash
cp .env.example .env
# edit .env, then:
set -a; source .env; set +a
cd src
python stage_ab_search.py
python stage_cd_answer.py
python stage_e_score.py      # requires data/inputs/ground_truth.csv
python stage_f_analysis.py
python make_figures.py
```

Notes:

- `providers.py` contains reference adapters for a Brave Search API client and an OpenAI-compatible chat-completions endpoint. Any equivalent provider can be used by editing that file.
- The published run used a hosted web search API and a hosted schema-constrained LLM service that did not expose a model identifier. Search indexes and model behavior change over time, so a rerun will not reproduce the released outputs exactly. Use the files in `data/outputs/` to verify the reported statistics.
- Running stages A to E overwrites the released files in `data/outputs/`. Work on a copy if you want to keep them.

## Pipeline settings

| Setting | Value |
| --- | --- |
| Search results per query and condition | 10 (visibility measured over all 10) |
| Results fetched for evidence | Top 3 per query and condition |
| Text kept per source | 3,000 characters |
| Evidence passed to generation | Up to 12,000 characters |
| Institutional match | URL, title, or domain contains `aast.edu`, `aastmt`, or `arab academy for science` |
| Rubric fields | Grounded correctness, evidence-passage correctness, hallucination absence, appropriate abstention, semantic consistency, failure mode resolved correctly |

