# Presentation Visuals

Generated graphics for the midterm presentation deck.

## Structure

- `scripts/` — Python scripts (matplotlib) that generate the PNG visuals
- `visuals/` — Output PNGs at 200 DPI, sized 16:9 for slide use

## Bhargav's slides

| Slide | File | Purpose |
|-------|------|---------|
| 1 — Title | `visuals/slide01_title.png` | Title card with authors |
| 2 — Motivation | `visuals/slide02_motivation.png` | Normal/attack/perturbed traffic flow through IDS |
| 3 — Objectives | `visuals/slide03_objectives.png` | 5-step attack–defense pipeline |
| 14 — Limitations | `visuals/slide14_challenges.png` | 5 challenges in two-column layout |

## Regenerating

```bash
python presentation/scripts/generate_bhargav_visuals.py
```

Outputs are written to `presentation/visuals/`.
