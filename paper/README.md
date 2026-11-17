# Revised Overleaf project

Open `main.tex` as the main document and select **pdfLaTeX** in Overleaf. The bundled IEEEtran class and bibliography style allow compilation without additional uploads.

- `main.tex`: revised English manuscript with Chinese drafting comments.
- `references.bib`: corrected bibliography, 20 entries.
- `main.pdf`: compiled manuscript, 11 pages.
- `supplementary.tex` / `supplementary.pdf`: eight supplementary figures preserving supplied records. Select this source separately to compile the supplement.
- `revision_report_cn.md`: Chinese revision report, including unresolved evidence and reproducibility issues.
- `figures/`: original assets and revised figure compositions.
- `scripts/build_figures.py`: optional figure assembly (PyMuPDF, Pillow, reportlab). The generated figures are already included; Overleaf does not need Python.
- `source_archive/`: previous sources retained for comparison. These are historical files, not the revised submission.
- `verification.json`: completed mathematical and artifact checks.

Local build:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

The supplied package did not include planner implementation or raw trial logs. Revision does not rerun the 450 trials or establish end-to-end hardware validation. See the Chinese report for the sensing-reference and paired-jerk discrepancies.

IEEEtran distribution sources (licenses and original headers preserved):

- Class v1.8b: https://github.com/nikcleju/quarto-ieeetran/blob/main/IEEEtran.cls
- Bibliography style v1.14: https://github.com/otvam/IEEEtran_file_url_doi_link_bibtex/blob/master/IEEEtran.bst

Author information remains anonymous as in the supplied manuscript.

Layout update (2026-09-06): Fig. 2 and Fig. 3 remain complete. The rolling figure highlights three source-artwork windows; all seven are in S8. The former handoff figure is S7. Regenerate the revised artwork with `python scripts/refine_layout_figures.py` after the original figure assembly if needed.

Evidence restored: main Fig. 4 includes 3-D and side views; Fig. 6 includes angle, full residual, and local-detail panels for both angles. Run `python scripts/restore_evidence_figures.py` last when rebuilding artwork. Main manuscript remains 11 pages.
