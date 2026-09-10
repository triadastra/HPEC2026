# Completed-training audit

The completed main matrix does not need blanket retraining because of the ten findings fixed in commit 294dee0. The saved evidence supports retaining the trained checkpoints and the checked prediction metrics. Compute-cost labels need correction; there are also a few supplementary provenance gaps.

Audited the published snapshot [51e064a2f8e1](https://huggingface.co/Celsia/HPEC2026/tree/51e064a2f8e1af1ff867794b5ce8aeeade4692cd), last updated 2026-09-04. This is an audit of the saved snapshot, not the current state of the training VMs. No training, deployment, or publication was performed.

## Direct checks

| Session | Declared | Checkpoints with matching completion fingerprint and hosted SHA-256 | Saved curves |
|---|---:|---:|---:|
| Main matrix | 477 | 477 | 477 |
| Base LR search | 195 | 195 | 195 |
| FA LR search | 25 | 25 | 25 |
| Exp 7 | 27 | 27 | 27 |
| Exp 8 | 27 | 27 | 27 |

- All 477 main run names match their manifest: no missing or undeclared root checkpoints. All 108 main runs requiring fresh model sessions have matching session evidence. No nonfinite validation losses were found in the 477 saved curves.
- Checkpoint hashes were compared to Hugging Face LFS SHA-256 object identities at the pinned snapshot. The checkpoint binaries were not all downloaded or executed. The final prediction-error archive was downloaded and its full SHA-256 was verified.
- All 111 rows in the published per-run metrics table exactly reproduce their MSE and MAE from the final saved prediction errors (30,087 series × 24 months). This checks arithmetic and artifact consistency; it is not a fresh inference run.
- Recomputed all 27 Wilcoxon comparisons using the corrected tied-rank variance. No p<0.05 decision changes. Maximum absolute p-value change: 2.33655e-12. The other reported test formulas were not changed by this fix.

## Impact of the ten fixes on these completed runs

| Finding | Observed impact | Required action |
|---|---|---|
| Publication failure handling, remote withdrawal, missing source merge, manifest membership (4 findings) | Main published matrix has complete matching saved provenance and no extra root checkpoints. | Keep main runs. Supplementary archive exceptions are listed below. |
| Training FLOP undercount | 243 main records call affected costs measured: GRU 99, LSTM 99, S4 27, S4ND 18. | Mark these training-cost totals partial/lower bounds or remeasure with complete formulas. Their predictions do not change. Static inference-cost tables are a separate measurement. |
| Mixed successful/divergent seeds in LR selection | Base and FA searches each use seed 947 only; neither reports divergence. All 30 selected arm/model rates reproduce from the curves. | No retraining caused by this bug. |
| CSV category vocabulary and tabular target alignment (2 findings) | All 477 main declarations, both LR searches, and Exp 7/8 use the Census NPZ pipeline. | These CSV-path fixes do not affect these runs. |
| Wilcoxon tie correction | All 27 significance decisions unchanged; p-value differences at most 2.34e-12. | Update statistical output if desired; no retraining needed. |
| Stale last.pth | Evaluation selects best.pth, which is the published checkpoint checked above. | No effect on those best-checkpoint predictions. Fix matters for last-checkpoint use/resumption. |

## Remaining qualifications

- The base LR selection was already explicitly unofficial (official=false). Its unstable primary cells are aggregate S4, rolling Mamba-2, and rolling S4; two additional transfer probes are unstable. These are pre-existing tuning limitations, not damage introduced by the recent code fixes. The FA selection is official=true.
- Supplementary archive directories contain five best.pth files without colocated completion records and 15 completion-record backups without colocated checkpoints. These are outside the fully validated main matrix. Do not treat every recursively counted file as a distinct validated run.
- Of the final error archive’s 111 run metadata entries, 107 match a saved completion record in both fingerprint and checkpoint hash. Four LSTM entries lack that exact completion pairing. All four error-archive checkpoint hashes match valid main-matrix checkpoint hashes, supporting the numerical results; three also match their supplementary manifest fingerprints and input fingerprints. The fourth, lstm_asa_4d_embeddings_s619, still lacks matching saved declaration provenance for the dump fingerprint. Restore/reconcile the supplementary provenance before claiming all 111 dump entries have a complete audit chain. Do not fabricate replacement records.
- The audit does not prove that nothing was withdrawn on the VMs after the pinned publication snapshot, nor independently reproduce training or inference. It does establish completion and saved-artifact consistency for the main matrix.

## Evidence files

- `audit.json`: session counts, manifest/hash checks, cost classifications and archive exceptions.
- `prediction_audit.json`: all old/new Wilcoxon results, metric comparisons, and supplementary provenance details.
- `hf_inventory.json` and `tree.json`: pinned hosted inventory and LFS object hashes.
- `audit.py` and `check_predictions.py`: local checks used. Full downloaded metadata and the error archive remain under `/tmp/hpec_completed_audit/snapshot`.

## Extended statistical checks

See [significance-audit.md](significance-audit.md). The earlier statement that the tie correction changes no decisions remains true. The broader correlation/multiplicity sensitivity does qualify three MAE claims and one equivalence claim; all 27 MSE differences survive the tested settings. The four supplementary entries also match their original source dump exactly, although the archival completion gaps remain.
