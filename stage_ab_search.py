"""Stages A and B: build search strings and run open-web retrieval.

A. Each scenario question is reduced to the short keyword query a student
   would type (OW-G). The OW-Q string is the same query prefixed with the
   institution's acronym.
B. Both strings are sent to the search API. Authoritative-source visibility
   is counted over the full result list.

Outputs: data/outputs/search_queries.jsonl, data/outputs/search_results.jsonl
"""
import csv
import json

import config
import providers

DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "search_query": {"type": "string", "description": "3-7 word keyword web search query"},
        "topic": {"type": "string", "description": "the regulatory topic in 6 words or fewer"},
    },
    "required": ["search_query", "topic"],
}
DISTILL_INSTRUCTION = (
    "You are given a student's academic-regulation scenario question at an Egyptian private "
    "university (Arab Academy for Science, Technology and Maritime Transport, AASTMT, Alexandria). "
    "Produce the short keyword web-search query a student would realistically type to find the "
    "governing rule. Use 3-7 words, no quotation marks, no boolean operators, no site: filters. "
    "Do not include the word AASTMT unless the scenario is meaningless without it."
)


def is_institutional(hit):
    text = f"{hit['url']} {hit.get('title') or ''} {hit.get('domain') or ''}".lower()
    return any(t in text for t in config.INSTITUTION_TOKENS)


def main():
    queries = list(csv.DictReader(open(config.QUERIES_CSV, encoding="utf-8-sig")))
    items = [{"query_id": q["query_id"], "scenario": q["query_text"]} for q in queries]
    results = providers.llm_json_many(DISTILL_INSTRUCTION, items, DISTILL_SCHEMA)

    distilled = {}
    for it, res in zip(items, results):
        distilled[it["query_id"]] = res or {"search_query": it["scenario"][:60], "topic": "unknown"}

    with open(config.OUTPUTS / "search_queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({**q, **distilled[q["query_id"]]}, ensure_ascii=False) + "\n")

    rows = []
    for q in queries:
        qid = q["query_id"]
        sq = distilled[qid]["search_query"]
        inst_sq = f"{config.INSTITUTION_PREFIX} {sq}"
        hits = {}
        for name, s in (("generic", sq), ("inst", inst_sq)):
            try:
                hits[name] = providers.web_search(s, count=config.N_SEARCH_RESULTS)
            except Exception as e:
                print(qid, name, "search failed:", e)
                hits[name] = []
        rows.append({
            "query_id": qid,
            "category": q["category"],
            "search_query": sq,
            "inst_search_query": inst_sq,
            "n_generic": len(hits["generic"]),
            "n_inst": len(hits["inst"]),
            "generic_visible": sum(map(is_institutional, hits["generic"])),
            "inst_visible": sum(map(is_institutional, hits["inst"])),
            "generic_top": [{k: h[k] for k in ("url", "title", "domain")} for h in hits["generic"][:5]],
            "inst_top": [{k: h[k] for k in ("url", "title", "domain")} for h in hits["inst"][:5]],
        })
        print(qid, "generic:", rows[-1]["generic_visible"], "institution-qualified:", rows[-1]["inst_visible"])

    with open(config.OUTPUTS / "search_results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gv = sum(r["generic_visible"] > 0 for r in rows)
    iv = sum(r["inst_visible"] > 0 for r in rows)
    print(f"Visibility: generic {gv}/{len(rows)}, institution-qualified {iv}/{len(rows)}")


if __name__ == "__main__":
    main()
