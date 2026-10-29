"""Stage E: two-pass LLM-jury scoring.

Each of the 90 open-web responses is scored twice against the six-field
binary rubric, using the expert-authored ground truth.

Requires data/inputs/ground_truth.csv (not distributed; see README) with
columns: query_id, governing_articles, resolution, failure_mode_tested.

Outputs: data/outputs/jury_pass1.jsonl, data/outputs/jury_pass2.jsonl
"""
import csv
import json

import config
import providers

SCHEMA = {
    "type": "object",
    "properties": {
        "grounded_correctness": {"type": "integer", "description": "1 if the substantive conclusion matches the authoritative resolution, else 0"},
        "evidence_passage_correctness": {"type": "integer", "description": "1 if the articles/clauses cited match the authoritative governing articles, else 0"},
        "hallucination_absence": {"type": "integer", "description": "1 if no invented article numbers, rules, or institutional facts appear, else 0"},
        "appropriate_abstention": {"type": "integer", "description": "1 if the system answered when it had the governing rule OR abstained when it did not, else 0"},
        "semantic_consistency": {"type": "integer", "description": "1 if the response is internally consistent and non-contradictory, else 0"},
        "conflict_resolved_correctly": {"type": "integer", "description": "1 if the specific tested failure mode (conflict, override precedence, or multi-hop exception chain) was actually resolved rather than sidestepped, else 0"},
    },
    "required": ["grounded_correctness", "evidence_passage_correctness", "hallucination_absence",
                 "appropriate_abstention", "semantic_consistency", "conflict_resolved_correctly"],
}

INSTRUCTION = (
    "You are a strict evaluator of a university student-portal assistant's answer about academic "
    "regulations. Compare the system response against the authoritative ground truth. Score each of "
    "the six rubric fields as exactly 1 (satisfied) or 0 (not satisfied), applying these rules:\n"
    "- evidence_passage_correctness = 1 ONLY if the cited article/clause identifiers correspond to the "
    "authoritative governing articles. Citing a plausible but different article number scores 0.\n"
    "- hallucination_absence = 0 if the response asserts an article number, threshold, or institutional "
    "rule that is not in the authoritative resolution and not verifiable from it.\n"
    "- appropriate_abstention = 1 if the system correctly abstained when it lacked the governing rule, or "
    "correctly answered when it had it. A confident answer without the governing rule scores 0. A refusal "
    "when the governing rule was available also scores 0.\n"
    "- conflict_resolved_correctly = 0 if the system merely restated related rules without deciding which "
    "one governs.\n"
    "A response that abstains cannot receive grounded_correctness = 1."
)



def main():
    if not config.GROUND_TRUTH_CSV.exists():
        raise SystemExit("Ground truth file not found. See README, 'Ground truth'.")
    gt = {r["query_id"]: r for r in csv.DictReader(open(config.GROUND_TRUTH_CSV, encoding="utf-8-sig"))}
    qs = {r["query_id"]: r for r in csv.DictReader(open(config.QUERIES_CSV, encoding="utf-8-sig"))}
    answers = [json.loads(l) for l in open(config.OUTPUTS / "openweb_answers.jsonl", encoding="utf-8")]

    items = []
    for a in answers:
        g = gt[a["query_id"]]
        items.append({
            "student_question": qs[a["query_id"]]["query_text"],
            "authoritative_governing_articles": g["governing_articles"],
            "authoritative_resolution": g["resolution"],
            "failure_mode_tested": g["failure_mode_tested"],
            "system_response": a["answer"],
            "articles_the_system_cited": a["cited_articles"],
            "system_declared_abstention": a["abstained"],
        })

    for p in (1, 2):
        path = config.OUTPUTS / f"jury_pass{p}.jsonl"
        scores = providers.llm_json_many(INSTRUCTION, items, SCHEMA)
        ok = 0
        with open(path, "w", encoding="utf-8") as f:
            for a, s in zip(answers, scores):
                if s is None:
                    s = {k: None for k in config.RUBRIC_FIELDS}
                else:
                    ok += 1
                    s = {k: s.get(k) for k in config.RUBRIC_FIELDS}
                f.write(json.dumps({"query_id": a["query_id"], "category": a["category"],
                                    "condition": a["condition"], "pass": p, **s}) + "\n")
        print(f"Pass {p}: scored {ok}/{len(items)}")


if __name__ == "__main__":
    main()
