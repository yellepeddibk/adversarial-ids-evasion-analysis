# Final Project Report

This directory contains the final written report for the adversarial IDS project.

## Files

- `final_project_report.tex` — LaTeX source for the report
- `final_project_report.pdf` — rendered PDF for submission or sharing

## Notes

- The report keeps the same overall structure as the midterm report, but updates the work as completed rather than in-progress.
- A dedicated future work section has been added to address possible extensions and improvements.

## Build

```bash
mkdir -p docs/reports/build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/reports/build docs/reports/final_project_report.tex
cp docs/reports/build/final_project_report.pdf docs/reports/final_project_report.pdf
```