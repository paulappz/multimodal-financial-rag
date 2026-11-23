"""Build eval/custodian_q3.json: questions with figures copied from the reports and the pages that contain them.

Run: python eval/build_custodian.py <dir with Custodian-Q3-2026.pdf and Custodian-Q3-2025.pdf>
Expected figures are typed in from each report's executive summary (page 2); the acceptable
pages are found by searching every page for the exact printed figure.
"""

import json
import sys
from pathlib import Path

import pymupdf

Q26, Q25 = "Custodian-Q3-2026.pdf", "Custodian-Q3-2025.pdf"

# (id, question, expected value, unit, documents that print it)
# unit: "k" = naira thousands, "kobo", "count"
ITEMS = [
    ("rev26", "What was Custodian's group gross revenue for the nine months ended 30 September 2026?", 255414006, "k", [Q26]),
    ("isr26", "What was Custodian's group insurance service revenue for the nine months ended 30 September 2026?", 115848850, "k", [Q26]),
    ("int26", "How much interest income did the Custodian group earn in the nine months to September 2026?", 110142303, "k", [Q26]),
    ("fee26", "What were the group's fees and commission income for the nine months ended 30 September 2026?", 16584105, "k", [Q26]),
    ("pbt26", "What was Custodian's group profit before taxation for the nine months ended 30 September 2026?", 66200792, "k", [Q26]),
    ("tax26", "What was the group income tax expense for the nine months ended 30 September 2026?", 8606102, "k", [Q26]),
    ("pat26", "What was Custodian's group profit after tax for the nine months ended 30 September 2026?", 57594690, "k", [Q26]),
    ("tci26", "What was the group's total comprehensive income for the nine months ended 30 September 2026?", 64140041, "k", [Q26]),
    ("mgx26", "What were the group's management expenses for the nine months to September 2026?", 29949785, "k", [Q26]),
    ("ix26", "What was the group's interest expense for the nine months ended 30 September 2026?", 50774192, "k", [Q26]),
    ("ta26", "What were Custodian's group total assets at 30 September 2026?", 1398272006, "k", [Q26]),
    ("ta25d", "What were Custodian's group total assets at 31 December 2025?", 1064733159, "k", [Q26]),
    ("eq26", "What was equity attributable to owners of the parent at 30 September 2026 for the group?", 235185330, "k", [Q26]),
    ("cash26", "How much cash and cash equivalents did the group hold at 30 September 2026?", 193807109, "k", [Q26]),
    ("icl26", "What were the group's insurance contract liabilities at 30 September 2026?", 338841610, "k", [Q26]),
    ("fa26", "What was the value of the group's financial assets at 30 September 2026?", 754425717, "k", [Q26]),
    ("eps26", "What were Custodian's group basic earnings per share for the nine months ended 30 September 2026?", 827, "kobo", [Q26]),
    ("cpat26", "What was the profit for the period of Custodian Investment Plc as a separate company (not the group) for the nine months to September 2026?", 23811429, "k", [Q26]),
    ("cdiv26", "How much dividend income did Custodian Investment Plc (the company) receive in the nine months ended 30 September 2026?", 23912202, "k", [Q26]),
    ("pat25", "What was the group profit after tax for the nine months ended 30 September 2025?", 45649246, "k", [Q26, Q25]),
    ("shares", "How many ordinary shares does Custodian Investment Plc have in issue?", 5881864195, "count", [Q26, Q25]),
    ("rev25", "What was Custodian's group gross revenue for the nine months ended 30 September 2025?", 173846699, "k", [Q26, Q25]),
    ("isr25", "What was the group insurance service revenue for the nine months ended 30 September 2025?", 121708949, "k", [Q26, Q25]),
    ("pbt24", "What was the group profit before taxation for the nine months ended 30 September 2024?", 52447247, "k", [Q25]),
    ("ta25", "What were the group's total assets at 30 September 2025?", 501758836, "k", [Q25]),
    ("eps24", "What were group basic earnings per share for the nine months ended 30 September 2024?", 750, "kobo", [Q25]),
    ("eq25", "What was group equity attributable to owners of the parent at 30 September 2025?", 163282978, "k", [Q25]),
]
NOT_IN_REPORTS = [
    ("seg26", "What profit did Custodian's pension business (Crusader Sterling Pensions) make in the nine months to September 2026?"),
    ("div27", "What dividend per share will Custodian pay for the 2027 financial year?"),
    ("fcst", "What profit will Custodian make in 2027?"),
]


def pages_with(doc: pymupdf.Document, value: int) -> list[int]:
    needle = f"{value:,}"
    return [i + 1 for i in range(len(doc)) if needle in doc[i].get_text()]


def main(pdf_dir: Path) -> None:
    docs = {name: pymupdf.open(pdf_dir / name) for name in (Q26, Q25)}
    out = []
    for qid, question, value, unit, names in ITEMS:
        where = {n: pages_with(docs[n], value) for n in names}
        where = {n: p for n, p in where.items() if p}
        if not where:
            sys.exit(f"{qid}: {value:,} not found in {names}; fix the expected value")
        out.append({"id": qid, "question": question, "value": value, "unit": unit, "pages": where})
    out += [{"id": qid, "question": q, "not_in_reports": True} for qid, q in NOT_IN_REPORTS]
    dest = Path(__file__).with_name("custodian_q3.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} questions -> {dest}")
    for item in out:
        print(item["id"], item.get("pages", "not in reports"))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
