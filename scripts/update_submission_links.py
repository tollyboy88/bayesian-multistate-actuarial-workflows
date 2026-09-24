from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document


PAPER1_REPO = "https://github.com/tollyboy88/agentic-actuarial-risk-testbed"
PAPER2_REPO = "https://github.com/tollyboy88/bayesian-multistate-actuarial-workflows"
DATA_DOI = "https://doi.org/10.5281/zenodo.22821051"


def replace_paragraph(paragraph, text: str, bold_lead: str | None = None) -> None:
    paragraph.clear()
    if bold_lead:
        run = paragraph.add_run(bold_lead)
        run.bold = True
        paragraph.add_run(text[len(bold_lead):])
    else:
        paragraph.add_run(text)


def update_document(path: Path) -> None:
    document = Document(path)
    for paragraph in document.paragraphs:
        text = paragraph.text
        if text.startswith("Code availability."):
            if "withheld" in text:
                replacement = (
                    "Code availability. The separate Paper 2 methodology repository, fixed outputs, tests and "
                    "Bayesian visualisation are withheld during double-anonymized review. The applied Paper 1 "
                    "testbed repository is cited separately."
                )
            else:
                replacement = (
                    f"Code availability. The separate Paper 2 methodology code, fixed outputs, tests and Bayesian "
                    f"visualisation are available at {PAPER2_REPO}. The applied Paper 1 testbed is maintained "
                    f"independently at {PAPER1_REPO}."
                )
            replace_paragraph(paragraph, replacement, "Code availability.")
        elif text.startswith("Adetola, A. A., and O. Akinade. 2026. Agentic Actuarial Testbed"):
            replace_paragraph(
                paragraph,
                f"Adetola, A. A., and O. Akinade. 2026. Bayesian Multistate Actuarial Workflows, version 1.0.0. GitHub. {PAPER2_REPO}.",
            )
        elif text.startswith("Anonymous. 2026. Agentic Actuarial Testbed"):
            replace_paragraph(
                paragraph,
                "Anonymous. 2026. Bayesian Multistate Actuarial Workflows, version 1.0.0. GitHub. [repository link withheld during double-anonymized review].",
            )
        elif text.startswith("Code repository:"):
            replace_paragraph(paragraph, f"Code repository: {PAPER2_REPO}")
        elif text.startswith("Data are archived at") and "reproducible code" in text:
            replace_paragraph(
                paragraph,
                f"Data are archived at {DATA_DOI}; the separate methodology code, fixed outputs and Bayesian "
                f"visualisation are at {PAPER2_REPO}. We request the standard non-open-access route, for which "
                "the journal states there is no APC.",
            )
        elif text.startswith("Run `python scripts/reproduce_saj.py`"):
            replace_paragraph(
                paragraph,
                "Run `python scripts/reproduce.py`, followed by `python -m pytest`. Start the Bayesian Workflow "
                "Laboratory with `streamlit run dashboard/app.py`.",
            )
        elif text.startswith("Python 3.11 or newer. Install the repository"):
            replace_paragraph(
                paragraph,
                "Python 3.11 or newer. Clone the separate Paper 2 repository and install `.[test,dashboard]`.",
            )
        elif text.startswith("Paper 2 consumes the frozen"):
            replace_paragraph(
                paragraph,
                "The repository contains a frozen stage-level transition table derived from the shared open data "
                "archive. Paper 1 and Paper 2 cite the same Zenodo DOI but use separate codebases and visual interfaces.",
            )
        elif text.startswith("Dataset:") and "Code:" in text:
            replace_paragraph(paragraph, f"Dataset: {DATA_DOI}. Paper 2 code: {PAPER2_REPO}.")
    document.core_properties.subject = f"Paper 2 methodology repository: {PAPER2_REPO}"
    document.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_directory", type=Path)
    args = parser.parse_args()
    for path in sorted(args.submission_directory.glob("*.docx")):
        if path.name.startswith("~$"):
            continue
        update_document(path)


if __name__ == "__main__":
    main()
