# AI workflow record

Date: 8 October 2026 (Asia/Shanghai).

This is a concise factual record, not a verbatim transcript. The user asked the
agent to complete the assignment strictly, put submission-only files in UPLOAD,
and publish the code in a newly created GitHub repository separate from prior work.

1. Extracted the one-page assignment PDF using pypdf and rendered it with Poppler.
   Confirmed six required report elements and the explicit AI-only coding rule.
2. Searched the web for `2024 2025 Bayesian last layer classification Laplace
   approximation computer vision paper`, then `Identity Curvature Laplace github
   arxiv`. Checked WACV 2025 publication metadata, read the ICLA preprint and
   official postprocessor. Other surfaced candidates included Variational
   Linearized Laplace Approximation (ICML 2024) and Last Layer Empirical Bayes
   (ICLR 2025 workshop). Selection was based on scope and implementability, not
   on a claim that ICLA is universally best.
3. Chose CPU-only digit recognition and a separate held-out-class detection task.
   Prespecified five seeds, stratified partitions, four methods, pixel corruption,
   precision grids and metrics in the experiment script.
4. Wrote the entire implementation. No student-written code was required.
5. An initial 600-iteration execution produced optimizer convergence warnings.
   Increased the cap to 3,000 and reran every task/seed. All final runs stopped
   without convergence warnings. Report tables use only the final run.
6. Computed saved metrics and found substantial underconfidence for evidence-tuned
   ICLA. Kept the result rather than selecting favorable test cases or adjusting
   the protocol to improve test scores. Every validation fit chose the upper
   precision boundary, which is disclosed.
7. Wrote independent numerical and results checks. Verified 60 per-run rows and
   12 aggregate rows. Stored verification.json and generated figures from raw
   saved predictions. The illustrative plots use seed 0, not the best seed.
8. Drafted the English report with algorithm details, implementation choices,
   genuine experimental results, limitations, AI-assisted reflection, references
   and a source-code link. Built the final PDF and inspected rendered pages.
9. Created a separate public repository dedicated to this assignment. The local
   UPLOAD directory is reserved for the final PDF, since the assignment asks for
   one Word or PDF report and a source webpage link inside it.

## Deliberate differences from the original paper

The assignment uses small digits images and an MLP rather than ResNet benchmarks.
The final mode is frozen during evidence tuning. Precision is optimized by a
bounded SciPy scalar minimizer. ICLA-Val is an extra comparison, explicitly
separate from evidence-tuned ICLA. The probabilistic predictive link is an
approximation, and no paper-level performance reproduction is claimed.

## Interpretation boundaries

The five seeds change split and initialization together. Repeated splits overlap,
and the OOD images are reused across seeds. Standard deviations are descriptive;
there is no significance or confidence-interval claim. Better OOD ranking does
not imply better calibrated class probabilities. Validation selection at the
upper grid boundary should be interpreted as near-deterministic prediction.
