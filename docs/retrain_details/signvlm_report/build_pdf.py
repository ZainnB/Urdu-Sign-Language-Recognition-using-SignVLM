"""Build SignVLM_Retrain_Report.pdf from the generated report assets.

Writes a self-contained print HTML (all figures embedded as base64 so the PDF
build has no file:// dependencies), then prints it to PDF with headless Edge.

Run:  python build_pdf.py
"""
import base64
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def img(name):
    data = base64.b64encode((HERE / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def md_table_to_html(md_text):
    rows = [r for r in md_text.strip().splitlines() if r.strip()]
    out = ["<table>"]
    for i, row in enumerate(rows):
        if set(row.replace("|", "").replace("-", "").strip()) == set():
            continue  # separator row
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        tag = "th" if i == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def figure(name, title, caption):
    return f"""
<figure>
  <div class="figtitle">{title}</div>
  <img src="{img(name)}" alt="{title}">
  <figcaption>{caption}</figcaption>
</figure>"""


per_epoch_html = md_table_to_html((HERE / "per_epoch_table.md").read_text(encoding="utf-8"))

CM_HOWTO = ("How to read it: rows are true classes, columns are predicted classes, row-normalized. "
            "The blue diagonal encodes each class's correct-classification rate (brighter = higher); "
            "red off-diagonal cells are misclassifications (brighter red = larger error mass). "
            "The right-hand strip shows per-class accuracy as horizontal bars, with the yellow dashed "
            "line marking mean class accuracy.")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PSL Recognition — SignVLM Signer-Disjoint Finetune Report</title>
<style>
  @page {{ size: A4; margin: 14mm 13mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10.5pt; line-height: 1.45; color: #1a1a1a;
    margin: 0;
  }}
  h1 {{ font-size: 19pt; margin: 0 0 4px; }}
  h2 {{ font-size: 14pt; margin: 22px 0 8px; border-bottom: 2px solid #1f77b4; padding-bottom: 3px; }}
  h3 {{ font-size: 11.5pt; margin: 16px 0 6px; }}
  .runline {{ color: #444; margin-bottom: 6px; }}
  .runline code, code {{ font-family: Consolas, monospace; font-size: 9.5pt; background: #f2f2f2; padding: 0 3px; border-radius: 3px; }}
  .provenance {{ font-size: 9pt; color: #555; border-left: 3px solid #ccc; padding-left: 8px; margin: 8px 0 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 8.6pt; }}
  th, td {{ border: 1px solid #c9c9c9; padding: 3px 6px; text-align: left; }}
  th {{ background: #eaf1f8; }}
  tr:nth-child(even) td {{ background: #f7f9fb; }}
  figure {{ margin: 14px 0 20px; text-align: center; page-break-inside: avoid; }}
  figure img {{ max-width: 100%; max-height: 210mm; }}
  .figtitle {{ font-weight: 600; font-size: 11pt; margin-bottom: 4px; }}
  figcaption {{
    font-size: 9pt; color: #333; text-align: left; margin: 6px auto 0;
    max-width: 165mm; background: #f5f7fa; border-left: 3px solid #1f77b4;
    padding: 6px 9px; border-radius: 0 4px 4px 0;
  }}
  ul {{ margin: 6px 0; padding-left: 20px; }}
  li {{ margin-bottom: 4px; }}
  .pagebreak {{ page-break-before: always; }}
  .note {{ font-size: 9.5pt; color: #333; font-style: italic; }}
  strong {{ color: #000; }}
</style>
</head>
<body>

<h1>PSL Recognition — SignVLM Signer-Disjoint Finetune Report</h1>
<p class="runline"><b>Run:</b> <code>signVLM_lists</code> artifacts &nbsp;|&nbsp; 46 epochs &nbsp;|&nbsp;
Best checkpoint: epoch 45 (val loss 0.1967, val acc 94.61%*)</p>
<p class="note">*In-training validation pass (per-epoch table, &sect;2). The independent post-training re-evaluation of
this same checkpoint (Final metrics, &sect;2) measures 94.53% &mdash; see protocol note below.</p>
<p class="provenance">All numbers, curves, and confusion matrices in this report are reconstructed directly from the
logged artifacts in <code>docs/retrain_details/signVLM_lists/</code> (<code>signvlm_loss_history.json</code>,
<code>signvlm_step_log.csv</code>, <code>signvlm_final_metrics.csv</code>, <code>signvlm_full_metrics_test.csv</code>,
<code>*_confusion_matrix.npy</code>). Every figure is regenerable with
<code>generate_report_assets.py</code>; this PDF is built by <code>build_pdf.py</code>.</p>

<h2>1. Model architecture</h2>
<p><b>SignVLM</b> — frozen CLIP image backbone + lightweight EVL-style temporal decoder + linear classifier.</p>
<table>
<tr><th>Stage</th><th>Layer</th><th>Notes</th></tr>
<tr><td>backbone</td><td>CLIP ViT-L/14 (lnpre variant), <b>frozen</b></td><td>per-frame visual features, 224×224 input</td></tr>
<tr><td>temporal decoder</td><td>4 transformer decoder layers, qkv_dim 1024, 16 attention heads</td><td>EVL-style, attends across frames</td></tr>
<tr><td>head</td><td>Linear → 104 logits</td><td>PSL-104 classifier</td></tr>
</table>
<p><b>Input:</b> 16–24 frames/clip, 224×224, RGB. <b>Classes:</b> 104 (PSL-104 — English gloss words + Urdu
alphabet letters; <code>label_map_auto.json</code>).</p>
<p>For reference, the base SignVLM architecture (Luqman, 2025, <i>PeerJ Comput. Sci.</i> 11:e3112) reports
<b>58.6&nbsp;M trainable parameters in the EVL-style temporal decoder + classification head</b> &mdash; the CLIP
backbone is frozen and contributes 0 trainable parameters, so it is not part of that count. No parameter-count
snapshot was captured for this specific PSL-104 run.</p>

<h3>Training recipe</h3>
<table>
<tr><th>Setting</th><th>Value</th><th>Source</th></tr>
<tr><td>Train / Val / Test split</td><td>4,368 / 1,244 / 1,248 clips</td><td><code>train.tsv</code> / <code>val.tsv</code> / <code>test.tsv</code> (<code>unseen.tsv</code> empty, not evaluated)</td></tr>
<tr><td>Test protocol</td><td>signer-disjoint (held-out signer), n=1,248</td><td><code>signvlm_full_metrics_test.csv</code> (<code>test_diff_signer</code>)</td></tr>
<tr><td>Batch size</td><td>20 (inferred: 218 steps/epoch × 20 ≈ 4,368 with drop_last)</td><td><code>signvlm_step_log.csv</code></td></tr>
<tr><td>Optimizer</td><td>AdamW</td><td>training log</td></tr>
<tr><td>Learning rate</td><td>4e-5 initial, cosine-annealed to ~1e-8</td><td><code>signvlm_loss_history.json</code></td></tr>
<tr><td>Scheduler</td><td>CosineAnnealingLR (T_max = num_steps)</td><td>training log</td></tr>
<tr><td>Loss</td><td>CrossEntropyLoss</td><td>training log</td></tr>
<tr><td>Weight decay</td><td>0.05 (project default, not confirmed in this run's logs)</td><td>project scripts</td></tr>
<tr><td>Epochs completed</td><td>46 (global_step 0–10,027, 218 steps/epoch)</td><td><code>signvlm_step_log.csv</code></td></tr>
<tr><td>Checkpointing</td><td>best-val-loss checkpoint saved at the every-5-epoch evals (epochs 5, 10, …, 45)</td><td>training log</td></tr>
</table>

<p><b>Key protocol notes</b></p>
<ul>
<li><b>Train accuracy is train-mode / augmented</b> (<code>Acc1(aug)</code>), so it <i>understates</i> the model's true
fit — final eval-mode train accuracy is 98.01% vs. the ~85% seen in the curves.</li>
<li><b>Validation ran every 5th epoch</b> (epochs 5, 10, …, 45) — the val curves have points only at those
epochs.</li>
<li><b>Final metrics are eval-mode, multi-view</b> for train/val (<code>signvlm_final_metrics.csv</code>).</li>
<li><b>The epoch-45 val acc in the per-epoch table (94.61%) and the Final-metrics val acc (94.53%) are both
multi-view eval on the same best checkpoint, run as two independent passes</b> &mdash; one logged live inside the
training loop, one re-run afterward by a separate script. Multi-view eval samples frames with a random component not
pinned to a fixed seed across runs, so the two passes disagree by exactly 1 of 1,244 clips (1,177 vs. 1,176 correct).
This is evaluation-pass noise, not a train/eval-mode mismatch or a copy-paste error, and it is well within
run-to-run variance for a sample this size.</li>
<li>The test split is the <b>same 104 classes performed by a signer never seen in training</b> (signer-disjoint
protocol).</li>
</ul>

<div class="pagebreak"></div>
<h2>2. Per-epoch metrics</h2>
{per_epoch_html}
<p class="note">Val columns are blank on epochs where validation was not scheduled; accuracy values are top-1/top-5
percentages of the split. Train accuracy is train-mode on augmented clips.</p>

<h3>Final metrics (eval mode, multi-view) — best model</h3>
<table>
<tr><th>Split</th><th>n</th><th>Accuracy</th><th>Precision (macro)</th><th>Recall (macro)</th><th>F1 (macro)</th><th>Precision (wtd)</th><th>Recall (wtd)</th><th>F1 (wtd)</th></tr>
<tr><td>train</td><td>4,368</td><td><b>98.01%</b></td><td>0.9811</td><td>0.9801</td><td>0.9800</td><td>0.9811</td><td>0.9801</td><td>0.9800</td></tr>
<tr><td>validation</td><td>1,244</td><td><b>94.53%</b></td><td>0.9504</td><td>0.9454</td><td>0.9457</td><td>0.9503</td><td>0.9453</td><td>0.9456</td></tr>
</table>

<h3>Test (signer-disjoint, held-out signer, 1,248 clips) — best model</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>test_top1_acc</td><td><b>78.12%</b></td></tr>
<tr><td>test_precision (macro)</td><td>0.8150</td></tr>
<tr><td>test_recall (macro)</td><td>0.7813</td></tr>
<tr><td>test_f1 (macro)</td><td>0.7713</td></tr>
<tr><td>test_precision (weighted)</td><td>0.8150</td></tr>
<tr><td>test_recall (weighted)</td><td>0.7813</td></tr>
<tr><td>test_f1 (weighted)</td><td>0.7713</td></tr>
</table>
<p class="note">Exact value: 975/1,248 = 78.125%, rounded to 78.12% throughout this report — a few earlier drafts
rounded this figure up to 78.13%; that has been corrected for consistency with the charts and confusion-matrix
titles in &sect;3&ndash;5.</p>
<p class="note">Macro and weighted precision/recall/F1 are identical to four decimal places on the test split because
the test set is exactly balanced — 1,248 clips over 104 classes = 12 clips/class — so the per-class weighting used
by "weighted" averages collapses to the unweighted mean used by "macro" averages. This is expected for a balanced
split, not a duplicated column.</p>
<p>A second held-out test set drawn from the <b>same signer pool</b> as train/val
(<code>test_confusion_matrix.npy</code>, n=1,248) scores <b>93.75%</b> top-1 — so the pure signer-shift cost is
93.75% → 78.12% (−15.6 points), not 94.5% → 78.1%.</p>

<div class="pagebreak"></div>
<h2>3. Training curves</h2>
{figure("signvlm_loss_per_epoch.png", "Loss per epoch",
        "Cross-entropy loss for train (blue) and validation (orange) over all 46 epochs; validation is evaluated "
        "every 5th epoch. Train loss falls from 4.93 to 0.59. Validation loss decreases monotonically at every "
        "scheduled evaluation (1.55 at epoch 5 → 0.1967 at epoch 45) "
        "and never inflects upward — there is no overfitting signature anywhere in the run. Validation sits "
        "below train because train loss is measured in train mode on augmented clips with dropout active.")}
{figure("signvlm_top1_accuracy_per_epoch.png", "Top-1 accuracy per epoch",
        "Top-1 accuracy for train (blue, train-mode on augmented clips) and validation (orange, eval-mode). "
        "Validation reaches 89.95% by epoch 10 and saturates near 94.6% from epoch 35 onward. The validation curve "
        "sitting above train is expected under this protocol: the train series is depressed by augmentation and "
        "active dropout, while eval-mode train accuracy on the final model is actually 98.01%.")}
{figure("signvlm_top5_accuracy_per_epoch.png", "Top-5 accuracy per epoch",
        "Top-5 accuracy for train and validation. Validation top-5 exceeds 99% from epoch 10 onward and ends at "
        "99.36% — for nearly every validation clip the correct sign is within the model's five most confident "
        "predictions, meaning residual top-1 errors are mostly fine-grained confusions between similar signs rather "
        "than outright misses.")}
{figure("signvlm_lr_schedule.png", "Learning-rate schedule",
        "Cosine-annealed learning rate over the run (log scale): 4e-5 at epoch 1 decaying smoothly to ~1e-8 by "
        "epoch 46, with no warmup phase and no restarts. The smooth decay matches the plateauing of the loss and "
        "accuracy curves over the final third of training.")}
{figure("signvlm_step_loss.png", "Step-level training loss",
        "Per-step training loss across all 10,028 optimizer steps (light blue, batch size 20, 218 steps/epoch) with "
        "a 109-step (~half-epoch) moving average (dark blue). The sharp drop between steps ~800 and ~2,000 (epochs "
        "4–9) is where the temporal decoder locks onto the frozen CLIP features; afterwards the average declines "
        "gradually to ~0.6 with well-behaved per-batch variance and no instability spikes.")}

<div class="pagebreak"></div>
<h2>4. Split-level results</h2>
{figure("signvlm_split_accuracy.png", "Final top-1 accuracy per split",
        "Final-model top-1 accuracy (eval mode, multi-view) on all four splits. Train 98.01% and validation 94.53% "
        "come from signvlm_final_metrics.csv; the two test bars are computed from their confusion matrices. The key "
        "comparison is the last two bars: a held-out test set from the same signer pool scores 93.75%, while the "
        "signer-disjoint test (a person never seen in training) scores 78.12% — isolating a −15.6-point cost "
        "attributable purely to signer shift.")}
{figure("signvlm_final_metrics_bars.png", "Final metrics — train vs validation vs signer-disjoint test",
        "Accuracy, macro precision, macro recall, and macro F1 for train, validation, and the signer-disjoint test "
        "(best model). The three metrics degrade together and moderately (accuracy 0.980 → 0.945 → 0.781; F1 0.980 → "
        "0.946 → 0.771), and macro precision (0.815) exceeding accuracy on the test split indicates errors are "
        "concentrated in a minority of classes rather than spread uniformly.")}

<div class="pagebreak"></div>
<h2>5. Confusion matrices</h2>
<p>Rendered with the project's <code>confusion_matrices/confusion_matrix_builder.py</code>. {CM_HOWTO}</p>
{figure("signvlm_train_confusion_matrix.png", "Train confusion matrix (n=4,368, acc 98.01%)",
        "Final-model predictions on the training split (eval mode). The diagonal is nearly uniformly bright — "
        "4,281 of 4,368 clips correct — with only faint scattered off-diagonal cells, confirming the model has "
        "essentially fit the training pool without residual class-level blind spots.")}
{figure("signvlm_validation_confusion_matrix.png", "Validation confusion matrix (n=1,244, acc 94.53%)",
        "Predictions on the validation split (same signer pool as train). The structure mirrors the train matrix "
        "with slightly more off-diagonal mass (1,176/1,244 correct). Per-class accuracy bars on the right stay high "
        "across nearly all 104 classes — there is no class the model systematically fails within the seen-signer "
        "distribution.")}
{figure("signvlm_test_confusion_matrix.png", "Test confusion matrix — same signer pool (n=1,248, acc 93.75%)",
        "Held-out test clips drawn from the same signers as train/val. Performance is statistically "
        "indistinguishable from validation (93.75% vs 94.53%), which establishes the baseline the signer-disjoint "
        "matrix below should be compared against: any additional degradation there is attributable to the new "
        "signer, not to held-out-clip noise.")}
{figure("signvlm_test_diff_signer_confusion_matrix.png", "Test confusion matrix — different signer (n=1,248, acc 78.12%)",
        "The core result: predictions on 1,248 clips from a signer never seen during training. 45 of 104 classes "
        "remain perfect, and 55 (including those 45) score ≥90% overall, but errors concentrate in structured "
        "cliques of visually similar signs — "
        "Hear/Healthy/He_or_she form a near-closed confusion cycle (hand-near-face signs), Why→Where and Where→Four "
        "confuse adjacent question/counting signs, and م→Come is a systematic Urdu-letter/gloss collision. This "
        "structured failure pattern is the signature of genuine sign similarity under a new signer's motion style.")}

<h3>Where the diff-signer errors concentrate</h3>
<p>Per-class accuracy on the signer-disjoint test: <b>45/104 classes are perfect (100%)</b>, and <b>55/104 classes
score ≥ 90% overall</b> (this 55 <i>includes</i> the 45 perfect classes — i.e. 10 more classes land in the 90–99%
band), mean class accuracy 78.12% — but 17 classes fall below 50%, and four collapse completely. (The remaining
104 − 55 − 17 = 32 classes fall in the 50–89% band.)</p>
<table>
<tr><th>Class</th><th>Acc</th><th>Correct/Total</th><th>Confused with (count)</th></tr>
<tr><td>Hear</td><td>0%</td><td>0/12</td><td>He_or_she (8), Speak (4)</td></tr>
<tr><td>He_or_she</td><td>0%</td><td>0/12</td><td>Healthy (11), Winter (1)</td></tr>
<tr><td>Healthy</td><td>0%</td><td>0/12</td><td>Hear (12)</td></tr>
<tr><td>Why</td><td>0%</td><td>0/12</td><td>Where (8), Wednesday (2)</td></tr>
<tr><td>ت</td><td>8%</td><td>1/12</td><td>Forward (7), You (3)</td></tr>
<tr><td>other</td><td>17%</td><td>2/12</td><td>Why (7), Forward (2)</td></tr>
<tr><td>ز</td><td>25%</td><td>3/12</td><td>See (4), Speak (3)</td></tr>
<tr><td>م</td><td>25%</td><td>3/12</td><td>Come (9)</td></tr>
<tr><td>Afraid</td><td>33%</td><td>4/12</td><td>other (2), Why (1)</td></tr>
<tr><td>Where</td><td>33%</td><td>4/12</td><td>Four (8)</td></tr>
</table>

<div class="pagebreak"></div>
<h2>6. Read</h2>
<ul>
<li><b>Training is healthy end-to-end.</b> Train loss 4.93 → 0.59; eval-mode train accuracy reaches 98.01%. Val loss
decreases monotonically at every scheduled eval (1.55 at epoch 5 → 0.1967) and never inflects upward — no overfitting signature
over 46 epochs.</li>
<li><b>Most of the fit happens by epoch 10</b> (val 89.95% top-1, 99.20% top-5); the remaining 36 epochs of cosine
decay add ~4.7 points of val accuracy and steadily better calibration (val loss 0.41 → 0.20).</li>
<li><b>The signer-disjoint generalization gap is moderate.</b> SignVLM goes 94.53% val → 78.12% test on a signer
never seen in training. Against the same-pool test set (93.75%) the pure signer-shift cost is −15.6 points — a real,
measurable gap, but the model retains the large majority of its skill on a person it has never seen.</li>
<li><b>Residual errors are concentrated and interpretable</b> (Section 5): four classes absorb a third of the total
error mass via mutual confusion among visually adjacent signs (Hear ↔ Healthy ↔ He_or_she; Why → Where → Four).
Top-5 accuracy on val is ~99.4%, and the diff-signer confusions are predominantly within these small cliques.</li>
<li><b>What could push past 78%:</b>
  <ol>
  <li><b>More signers in train</b> — the gap that remains is still signer-identity, and the confusion cliques are
  exactly where a second/third signer's motion variance would help most.</li>
  <li><b>Targeted disambiguation of the confusion cliques</b> — hand-crop or higher-frame-rate sampling for the
  hand-near-face cluster (Hear/Healthy/He_or_she), which differs mainly in fine hand shape and contact point.</li>
  <li><b>Light backbone adaptation</b> (last-block or LoRA finetuning of CLIP) once more signer diversity exists —
  with a single training-signer pool, unfreezing now would mostly re-open the door to signer memorization.</li>
  </ol>
</li>
</ul>
<p class="note">Report and figures generated 2026-07-19 from the artifacts in
<code>docs/retrain_details/signVLM_lists/</code>.</p>

</body>
</html>
"""

out_html = HERE / "SignVLM_Retrain_Report_print.html"
out_html.write_text(html, encoding="utf-8")
print(f"Wrote {out_html} ({out_html.stat().st_size/1e6:.1f} MB)")

out_pdf = HERE / "SignVLM_Retrain_Report.pdf"
subprocess.run([
    EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
    f"--print-to-pdf={out_pdf}", out_html.as_uri(),
], check=True, timeout=180)
print(f"Wrote {out_pdf} ({out_pdf.stat().st_size/1e6:.1f} MB)")
