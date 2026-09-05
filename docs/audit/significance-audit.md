# Extended significance audit

The saved prediction scores remain unchanged, and no new training fault was found. The significance arithmetic reproduces correctly. However, the entire significance package should not be described as assumption-free or fully robust: several secondary MAE/equivalence claims are sensitive to monthly correlation and multiplicity, and some mean improvements are inconsistent across the three seeds.

## What was reproduced

- Running the corrected repository significance script on the final saved error archive produces a text file identical to the published `significance_FINAL_v3.txt`, including the printed confidence intervals and TOST results. No printed number changes at the published precision.
- Independent SciPy calculations agree with all 27 DM-MSE, DM-MAE, paired-t, Wilcoxon and TOST calculations to floating-point precision. The largest independent TOST p-value difference is 2.48e-8 and the largest confidence-bound difference is 1.76e-8; neither affects a decision. With h=1, the implemented DM/HLN calculation is algebraically a one-sample t-test on the 24 monthly mean loss differences.
- All 224 final loss arrays have the expected 30,087 × 24 shape, contain only finite values, and are nonnegative.
- All 72 arrays for the 36 backfill runs match the separately archived source dump exactly. All 36 source metadata entries and the Census configuration match too. Both downloaded error archives were checked against their hosted SHA-256 identities.

## Monthly correlation and multiple comparisons

The existing calculation uses lag-zero variance. The saved monthly loss differences show substantial serial dependence: a descriptive three-lag Ljung–Box screen gives unadjusted p<0.05 in 24/27 MSE pairs and 22/27 MAE pairs. This screen is not a bandwidth-selection rule or proof of a particular model of dependence. It does show that the lag-zero assumption deserves scrutiny.

For sensitivity, I used Bartlett/Newey–West covariance with lags 1, 3 and 6, retaining the t(23) reference and the finite-sample scaling that agrees with the original calculation at lag zero. These are approximate sensitivity analyses on only 24 months, not replacement p-values selected after seeing which result is preferred. See the [statsmodels HAC documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html) for the covariance method.

- Holm correction across the 27 comparisons separately for each endpoint, without changing covariance, changes no significance or equivalence decisions.
- Treating all 108 difference tests as a single family changes one original decision: identity vs blind Transformer 2-D MAE goes from p=0.01249 to 0.07492. The paper should define its testing families; neither choice should be selected to obtain a preferred outcome.
- All 27 MSE differences remain significant with HAC lags 1, 3 and 6, including Holm adjustment within the 27 MSE comparisons.
- Combining HAC and Holm does weaken secondary claims. At lag 3, the following previously significant results exceed 0.05:

| Comparison / endpoint | Original p | HAC lag 3 p | HAC lag 3 + Holm(27) p |
|---|---:|---:|---:|
| S4ND-4D vs S4 flat / MAE | 0.00168752 | 0.0428207 | 0.214104 |
| identity vs blind (Tr 2D) / MAE | 0.0124861 | 0.0800419 | 0.320167 |
| fa_sm vs ASA (GRU 2D) / MAE | 0.000908696 | 0.0186814 | 0.112088 |
| fa_local vs ASA (transformer 3D) / TOST | 0.000278604 | 0.0199285 | 0.39857 |

The first two MAE comparisons and the Transformer 3-D equivalence claim already lose significance with lag 1 plus Holm. The GRU softmax-vs-ASA MAE comparison loses it with lags 3 and 6 plus Holm. Without multiplicity correction, identity MAE crosses 0.05 at lags 3 and 6, and S4ND-vs-flat MAE crosses at lag 6. None of these changes the underlying MSE or MAE values or their observed direction.

TOST here means equivalence within a stated margin of 5% of the reference model’s mean MSE. The numerical result is conditional on that margin; the saved artifacts cannot establish that it was chosen before examining results. A nonsignificant difference is not evidence of equivalence. [Holm correction reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

## Variation across seeds and across series

The main significance suite first averages loss arrays across three seeds, then treats months or series as its testing units. It therefore does not directly establish that an architecture’s advantage persists across new random training seeds. Fourteen of the 27 MSE comparisons have mixed signs across the three saved seeds.

Two especially important examples (negative means fa_local has lower MSE than ASA):

| Comparison | Seed 947 | Seed 732 | Seed 619 |
|---|---:|---:|---:|
| fa_local vs ASA (lstm 3D) | -0.317887 | +0.031431 | +0.032861 |
| fa_local vs ASA (lstm 4D) | +0.025528 | -0.376440 | +0.043912 |

In each of these two cells, fa_local wins on one seed and loses on two, despite having a better three-seed average. The archived `FINAL_RESULTS.md` statement that seed spread was a bad proxy for significance should not be interpreted as dismissing this training variability: the tests answer a different question. Keep the means and seed spread visible; say “lower mean over the three trained seeds,” not “consistently superior across seeds.” Testing that stronger claim would require an explicitly designed seed-level analysis and potentially additional seeds, not redoing all existing training.

The per-series Wilcoxon and paired-t tests also do not model dependence between series sharing states, commodities and flows. The [SciPy Wilcoxon documentation](https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.stats.wilcoxon.html) specifies independent differences and a symmetric-difference null. Their arithmetic is correct, but the 30,087 series should not automatically be treated as independent replicates. The monthly aggregate analysis accommodates contemporaneous cross-series dependence by treating the month as the unit; its temporal dependence is what the HAC sensitivity examines.

## Four supplementary provenance entries

All four previously flagged error-dump entries are present unchanged in the original backfill source archive, in both loss arrays and metadata. This rules out a merge mismatch for them. Their recorded checkpoint hashes also match valid main-matrix checkpoint hashes, as established in the first audit.

The documentary gaps remain: four exact supplementary completion records are absent; three dump fingerprints match the supplementary manifest, while `lstm_asa_4d_embeddings_s619` still lacks a matching archived declaration for its dump fingerprint. Its saved log shows training and checkpoint writes but does not supply the missing completion binding. No records were reconstructed or altered. This is a limitation of the archived chain of evidence, not evidence that its reported loss arrays changed.

## Reporting action

Keep the completed checkpoints and the checked prediction metrics. Reproduce the existing significance table as calculated, but report the serial-correlation/multiplicity sensitivity and qualify the affected MAE/equivalence claims. Keep seed variability visible and distinguish an average over these three runs from a claim about retraining reliability. No numerical evidence found here justifies blanket retraining.

This audit reviewed the repository and saved published artifacts; the current paper manuscript was not present, so its exact claims have not been certified. The static inference-cost panel is separate from the 243 incomplete training-cost totals identified earlier.

Files: `significance_recomputed.txt` (reproduced table), `significance_extended.csv` and `.json` (all sensitivities and independent formula checks), `source_merge_audit.json` (source-array validation), and the corresponding `check_*.py` scripts.
