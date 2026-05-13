# Documentation and Project Assets

This directory contains public-facing project materials, including presentation visuals and the final project report.

## Structure

- `scripts/` — Python scripts that generate slide visuals with matplotlib
- `visuals/` — Rendered PNG assets sized for widescreen slides
- `reports/` — Final report source, companion notes, and rendered PDF

## Presentation Visuals

| Asset | File | Purpose |
|---|---|---|
| Title slide | `visuals/slide01_title.png` | Project title card with authors |
| Motivation slide | `visuals/slide02_motivation.png` | IDS evasion motivation diagram |
| Objectives slide | `visuals/slide03_objectives.png` | Attack-defense workflow summary |
| Limitations slide | `visuals/slide14_challenges.png` | Constraints and deployment limitations |

## Final Report

- `reports/final_project_report.tex` — LaTeX source for the final report
- `reports/final_project_report.pdf` — compiled PDF artifact
- `reports/README.md` — quick build notes and report contents

## Regenerate Visuals

```bash
python docs/scripts/generate_presentation_visuals.py
```

Outputs are written to `docs/visuals/`.

## Build the Final Report

```bash
mkdir -p docs/reports/build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/reports/build docs/reports/final_project_report.tex
cp docs/reports/build/final_project_report.pdf docs/reports/final_project_report.pdf
```
