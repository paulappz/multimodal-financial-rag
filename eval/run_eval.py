"""Score a running finrag server (finrag serve) against an eval set.

  python eval/run_eval.py eval/custodian_q3.json --top-k 6                 # retrieval only, no LLM calls
  python eval/run_eval.py eval/custodian_q3.json --top-k 12 --answers      # also ask Claude and score answers
  python eval/run_eval.py ... --scope                                      # restrict search to the item's reports

Metrics
- retrieval hit: a retrieved chunk comes from a page that prints the expected figure
- answer correct: the answer states the expected figure (0.5% tolerance; thousands, millions,
  billions and naira/kobo are all accepted), or says the reports don't cover it for the
  not-in-reports questions
- citation correct: a cited chunk comes from a page that prints the expected figure
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

NUM = re.compile(r"\d[\d,]*\.?\d*")


def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)


def on_page(chunk: dict, pages: dict[str, list[int]]) -> bool:
    return chunk["page"] in pages.get(chunk["source"], [])


def states_value(text: str, value: int, unit: str) -> bool:
    if unit == "k":  # naira thousands; answers may say thousands, millions or billions
        scales = [1, 1e-3, 1e-6, 1e3]
    elif unit == "kobo":  # 827 kobo or N8.27
        scales = [1, 0.01]
    else:
        scales = [1, 1e-6, 1e-9]
    for raw in NUM.findall(text):
        x = float(raw.replace(",", "").rstrip("."))
        for s in scales:
            target = value * s
            if target and abs(x - target) / target < 0.005:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("evalset")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--scope", action="store_true", help="limit retrieval to the reports named in each item")
    ap.add_argument("--out")
    args = ap.parse_args()

    items = json.load(open(args.evalset))
    rows = []
    for item in items:
        body = {"top_k": args.top_k}
        if args.scope and "pages" in item:
            body["documents"] = sorted(item["pages"])
        row = {"id": item["id"]}
        if "pages" in item:
            hits = post(args.base, "/api/search", {"query": item["question"], **body})["results"]
            row["retrieval_hit"] = any(on_page(h, item["pages"]) for h in hits)
        if args.answers:
            t = time.time()
            ans = post(args.base, "/api/ask", {"question": item["question"], **body})
            row["seconds"] = round(time.time() - t, 1)
            row["answer"] = ans["answer"]
            if item.get("not_in_reports"):
                row["answer_correct"] = bool(ans["insufficient_context"])
            else:
                row["answer_correct"] = states_value(ans["answer"], item["value"], item["unit"])
                cited = [c for c in ans["context"] if c["n"] in ans["cited"]]
                row["citation_correct"] = any(on_page(c, item["pages"]) for c in cited)
        rows.append(row)
        print(row["id"], {k: v for k, v in row.items() if k not in ("id", "answer")}, flush=True)

    def rate(key: str) -> str:
        vals = [r[key] for r in rows if key in r]
        return f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)" if vals else "-"

    summary = {"top_k": args.top_k, "scope": args.scope, "retrieval_hit": rate("retrieval_hit"),
               "answer_correct": rate("answer_correct"), "citation_correct": rate("citation_correct")}
    print("\nSUMMARY", json.dumps(summary))
    if args.out:
        json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
