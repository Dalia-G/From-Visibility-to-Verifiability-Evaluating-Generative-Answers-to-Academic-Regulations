"""Stages C and D: evidence extraction and answer generation.

C. The top three results per query and condition are fetched. An extraction
   pass keeps only text stating academic-regulation rules; pages without such
   text are discarded. Each source is truncated to 3,000 characters.
D. Answers are generated only from the retrieved evidence, with article
   citations or an explicit abstention.

Output: data/outputs/openweb_answers.jsonl
"""
import json

import config
import providers

EXTRACT_INSTRUCTION = (
    "Extract any text stating academic-regulation rules on credit-hour load limits, academic "
    "probation, GPA thresholds, absence/attendance limits, course repetition, withdrawal, "
    "graduation requirements, appeals or grade rules. Quote article/clause numbers verbatim "
    "where present. If the page contains no such regulation text, set text to NO_REGULATION_TEXT."
)
EXTRACT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

ANSWER_INSTRUCTION = (
    "You are a university student-portal assistant answering a question about academic regulations. "
    "Answer ONLY from the retrieved evidence provided. Cite the specific article or clause numbers "
    "that govern your answer. If the retrieved evidence does not contain the governing rule, you MUST "
    "abstain: set abstained=true and state plainly that the authoritative regulation was not found, "
    "rather than answering from general knowledge about universities. Never invent article numbers."
)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_articles": {"type": "string"},
        "abstained": {"type": "boolean"},
    },
    "required": ["answer", "cited_articles", "abstained"],
}


def main():
    rows = [json.loads(l) for l in open(config.OUTPUTS / "search_results.jsonl", encoding="utf-8")]
    qmeta = {}
    for l in open(config.OUTPUTS / "search_queries.jsonl", encoding="utf-8"):
        r = json.loads(l)
        qmeta[r["query_id"]] = r

    index = []
    for r in rows:
        for cond, key in (("OW-G", "generic_top"), ("OW-Q", "inst_top")):
            for h in r[key][:config.N_FETCHED]:
                index.append((r["query_id"], cond, h["url"], h["domain"], h["title"]))
    urls = list(dict.fromkeys(u for _, _, u, _, _ in index))
    print("Unique URLs to fetch:", len(urls))

    raw = {u: providers.fetch_text(u) for u in urls}
    payloads = [{"url": u, "page_text": raw[u][:20000]} for u in urls]
    extracted = providers.llm_json_many(EXTRACT_INSTRUCTION, payloads, EXTRACT_SCHEMA)
    pages = {u: ((e or {}).get("text", "") if raw[u] else "") for u, e in zip(urls, extracted)}

    evidence = {}
    for qid, cond, url, domain, title in index:
        txt = pages.get(url, "")
        if not txt or txt.strip().startswith("NO_REGULATION_TEXT"):
            continue
        evidence.setdefault((qid, cond), []).append(
            f"[SOURCE: {title} | {domain} | {url}]\n{txt[:config.MAX_CHARS_PER_SOURCE]}")

    items, keys = [], []
    for r in rows:
        for cond in ("OW-G", "OW-Q"):
            chunks = evidence.get((r["query_id"], cond), [])
            items.append({
                "question": qmeta[r["query_id"]]["query_text"],
                "retrieved_evidence": "\n\n".join(chunks)[:config.MAX_EVIDENCE_CHARS] if chunks
                else "NO EVIDENCE RETRIEVED",
            })
            keys.append((r["query_id"], r["category"], cond, len(chunks)))

    answers = providers.llm_json_many(ANSWER_INSTRUCTION, items, ANSWER_SCHEMA)
    ok = 0
    with open(config.OUTPUTS / "openweb_answers.jsonl", "w", encoding="utf-8") as f:
        for (qid, cat, cond, n), out in zip(keys, answers):
            if out is None:
                out = {"answer": "GENERATION_ERROR", "cited_articles": "", "abstained": True}
            else:
                ok += 1
            f.write(json.dumps({"query_id": qid, "category": cat, "condition": cond,
                                "n_evidence_chunks": n, "evidence_present": n > 0, **out},
                               ensure_ascii=False) + "\n")
    print(f"Answers generated: {ok}/{len(items)}")


if __name__ == "__main__":
    main()
