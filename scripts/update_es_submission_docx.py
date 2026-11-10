from __future__ import annotations

import csv
import shutil
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission" / "manuscript_source.md"
PACKAGE = ROOT / "submission" / "FINAL_SUBMISSION_SAJ"
CONTROL_TABLE = ROOT / "outputs" / "tables" / "control_optimisation.csv"
PARETO_FIGURE = ROOT / "outputs" / "figures" / "figure5_pareto_frontier.png"
TRANSITION_FIGURE = ROOT / "outputs" / "figures" / "figure2_transition_intervals.png"


def source_paragraphs() -> dict[str, str]:
    wanted = {
        "Automated actuarial workflows create": "abstract",
        "For the reproducible demonstration": "methods",
        "The selected budget-feasible design": "results",
    }
    found: dict[str, str] = {}
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        for prefix, name in wanted.items():
            if line.startswith(prefix):
                found[name] = line
    if set(found) != set(wanted.values()):
        raise RuntimeError(f"Missing manuscript paragraphs: {set(wanted.values()) - set(found)}")
    return found


def leading_controls() -> list[list[str]]:
    with CONTROL_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["feasible"].lower() == "true"]
    rows.sort(key=lambda row: float(row["posterior_probability_optimal"]), reverse=True)
    result = []
    for row in rows[:10]:
        result.append([
            row["placement"].zfill(6),
            f'{float(row["annual_cost"]):.0f}',
            f'{float(row["posterior_mean_silent_probability"]):.3f}',
            f'{float(row["posterior_probability_risk_constraint"]):.1%}',
            f'{float(row["posterior_mean_es_995"]):.3e}',
            f'{float(row["posterior_probability_optimal"]):.1%}',
            "Yes" if row["selected"].lower() == "true" else "No",
        ])
    return result


def replace_updated_images(path: Path) -> None:
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp) / path.name
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
            temp_path, "w", zipfile.ZIP_DEFLATED
        ) as target:
            for item in source.infolist():
                replacements = {
                    "word/media/image3.png": PARETO_FIGURE,
                    "word/media/image5.png": TRANSITION_FIGURE,
                }
                data = replacements[item.filename].read_bytes() if item.filename in replacements else source.read(item.filename)
                target.writestr(item, data)
        shutil.copy2(temp_path, path)


def update(path: Path) -> None:
    replacements = source_paragraphs()
    prefixes = {
        "Automated actuarial workflows create": replacements["abstract"],
        "For the reproducible demonstration": replacements["methods"],
        "The selected budget-feasible design": replacements["results"],
    }
    document = Document(path)
    replaced = set()
    for paragraph in document.paragraphs:
        for prefix, text in prefixes.items():
            if paragraph.text.startswith(prefix):
                paragraph.text = text
                replaced.add(prefix)
    if replaced != set(prefixes):
        raise RuntimeError(f"Could not replace all paragraphs in {path.name}")

    matching_tables = [candidate for candidate in document.tables if candidate.cell(0, 0).text == "Placement"]
    if len(matching_tables) != 1:
        raise RuntimeError(f"Could not identify Table 7 in {path.name}")
    table = matching_tables[0]
    headers = ["Placement", "Cost", "Mean p(E)", "Pr(constraint)", "Mean ES99.5", "Pr(optimal)", "Selected"]
    values = [headers, *leading_controls()]
    if len(table.rows) != len(values) or len(table.columns) != len(headers):
        raise RuntimeError(f"Unexpected Table 7 dimensions in {path.name}")
    for row, row_values in zip(table.rows, values):
        for cell, value in zip(row.cells, row_values):
            cell.text = value
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)
                    if row is table.rows[0]:
                        run.bold = True
    document.save(path)
    replace_updated_images(path)


def main() -> None:
    for name in ["01_main_manuscript_anonymous.docx", "02_manuscript_with_author_details.docx"]:
        update(PACKAGE / name)


if __name__ == "__main__":
    main()
