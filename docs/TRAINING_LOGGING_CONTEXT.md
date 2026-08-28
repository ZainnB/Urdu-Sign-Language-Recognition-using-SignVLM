# Training Metric Logging — Context Handoff

This document describes every layer of per-step / per-epoch / per-run data logging built for the
SignVLM training pipeline, so it can be handed to a different LLM/model as a reference for
building equivalent logging on a new training run. It covers what is logged, at what cadence, to
which files, in what format, and the specific traps that shaped the current design (Drive-write
latency, GPU-sync cost, resume-safety, `None`-handling for skipped validation epochs).

The pipeline runs on Google Colab with the dataset/checkpoints on Google Drive (slow, network
FUSE mount) and local disk (`/content`, fast but ephemeral) available as scratch. That constraint
— **Drive writes are slow enough to stall a training step if done every step** — is the reason the
logging is split into a fast local tier and a periodically-synced Drive tier, described below.

## Logging layers, cadence, and destination

| Layer | Cadence | Written to | Format |
|---|---|---|---|
| Per-step CSV row | every optimizer step | local disk, synced to Drive once/epoch | CSV |
| Per-step console line | every `print_freq` steps (10) | stdout only | text |
| Per-epoch history entry | every epoch | in-process dict → JSON on Drive (every epoch) + embedded in checkpoint | JSON / `.pth` |
| Per-epoch console summary | every epoch | stdout only | text |
| Validation loss/acc | every `EVAL_EVERY_N_EPOCHS` (5) | folded into the per-epoch history entry (`None` on skipped epochs) | JSON |
| Validation confusion + F1 | every `CONFUSION_EVERY_N_EPOCHS` (currently `0` = disabled) | Drive `.npy` | NumPy array |
| Epoch checkpoint | every `SAVE_EVERY_N_EPOCHS` (1) | Drive `.pth` | PyTorch state dict + embedded history |
| Training curves plot | once, after training loop finishes (Cell 7b) | Drive `.png` | matplotlib figure |
| Final precision/recall/F1 table | once, after training loop finishes (Cell 7c) | Drive `.csv` | CSV |
| Multi-split eval + confusion matrices | on demand (Cell 8) | Drive `.npy` per split | NumPy array |
| Standalone full metrics + confusion | on demand, runnable in a fresh runtime (Cell 10) | Drive `.csv` + `.npy` + `.png` per split | CSV / NumPy / PNG |

## 1. Per-step CSV log — local-write, Drive-sync-per-epoch

**Why this exists:** logging every optimizer step (potentially thousands per epoch) directly to a
Drive-mounted path would put a slow FUSE network write on the hot training path, stalling every
single step. The fix: write the per-step CSV to local scratch disk (`/content/...`) on every step
(cheap), and copy the whole file to Drive **once per epoch**, not once per step.

```python
_step_log_local  = Path("/content/signvlm_step_log.csv")   # hot path target
_step_log_drive  = Path(LIST_DIR) / "signvlm_step_log.csv" # synced destination, once/epoch
_STEP_LOG_FIELDS = ["global_step", "epoch", "step_in_epoch", "lr", "loss", "acc1", "acc5"]
```

- **Resume-safety:** if a Drive copy already exists from a previous session (e.g. after a Colab
  runtime recycle) but the fresh runtime's local scratch is empty, the Drive copy is pulled local
  *first*, so appends continue the same file instead of starting a second, disconnected log:
  ```python
  if _step_log_drive.exists() and _step_log_drive.stat().st_size > 0 and not _step_log_local.exists():
      shutil.copy(_step_log_drive, _step_log_local)
  ```
- The CSV writer is opened once in append mode (`"a"`) for the life of the training loop; the
  header is written only if the local file is new/empty, so re-running the cell after a resume
  doesn't duplicate headers mid-file.
- **Flush cadence is separate from sync cadence:** the local file is `flush()`-ed every 10 steps
  (cheap — still local disk, not Drive) so a crash loses at most ~10 steps of rows, while the Drive
  copy only happens once per epoch (expensive — a full-file copy over FUSE).
- The Drive sync is wrapped in a `try/except` that prints a warning and continues on failure rather
  than crashing training — a transient Drive hiccup should not kill an otherwise-healthy training
  run; it just retries the sync at the next epoch boundary.
- The file handle is closed, flushed, and **force-synced to Drive one last time** in a `finally`
  block around the whole training loop, so even a mid-epoch crash doesn't lose the tail of the log.

Fields logged per step: `global_step` (monotonic across the whole run, `epoch * steps_per_epoch +
step_in_epoch`), `epoch` (1-indexed), `step_in_epoch` (1-indexed), `lr` (current, post-scheduler-step
value), `loss` (this step's loss, 6 decimal places), `acc1`/`acc5` (this step's own top-1/top-5 hit
rate, not a running average).

## 2. Per-step console line (throttled)

Printed only every `print_freq` (10) global steps, not every step, to keep console output usable
on long runs:

```
Epoch 3/15 step 40/156 batch 30.120s wait 0.412s compute 29.708s lr 0.000038 loss 2.104318 acc1 45.00% acc5 72.50%
```

- `batch` / `wait` / `compute` split lets you see, at a glance, whether a run is data-I/O-bound
  (`wait` dominates — investigate the cache/DataLoader) or compute-bound (`compute` dominates —
  normal). This is a cheap diagnostic that costs only two `datetime.now()` calls per step.
- The printed `loss`/`acc1`/`acc5` values are **synced across all distributed workers** via
  `dist.all_reduce` right before printing (world_size=1 on Colab, so this is a no-op there, but
  keeps the code correct for multi-GPU). This sync happens **only** at the print cadence, not every
  step — see §4 for why per-step syncs are avoided.

## 3. Per-epoch history dict → JSON (+ embedded in checkpoint)

A single in-process dict accumulates one entry per epoch across the whole run (including across
resumes — see §6):

```python
def new_history():
    return {
        "epoch": [], "lr": [],
        "train_loss": [], "train_acc1": [], "train_acc5": [],
        "val_loss": [], "val_acc1": [], "val_acc5": [],
    }
```

plus `eval_train_loss` / `eval_train_acc1` / `eval_train_acc5` keys added via `.setdefault(...)` (a
now-disabled optional eval-mode train-accuracy pass — see §7).

- Every value list is **the same length as `epoch`**, including epochs where validation didn't run
  — those slots are `None`, not omitted. This keeps every list index-aligned with `epoch`, so
  downstream plotting/analysis code never has to reconcile mismatched list lengths; it just filters
  `None`s out at read time (see §8's `_valid_xy` helper).
- Saved to a Drive JSON file **every epoch**, unconditionally — not just on checkpoint epochs:
  ```python
  _history_path = Path(LIST_DIR) / "signvlm_loss_history.json"
  def _save_history_json():
      with open(_history_path, "w", encoding="utf-8") as f:
          json.dump(history, f, indent=2)
  ```
  This is the artifact you'd tail/inspect mid-run to see how training is progressing without
  touching the (much larger) checkpoint files.
- **Also embedded inside every checkpoint** (`save_epoch_checkpoint` writes `"history": history`
  into the `.pth`) — belt-and-suspenders redundancy: the standalone JSON could be stale or missing
  in a weird partial-write scenario, but the checkpoint's copy is written atomically alongside the
  model weights it corresponds to, so `.pth` files are self-describing (a checkpoint alone tells
  you the full loss/accuracy history up to that epoch, no external file needed).
- The JSON is re-saved again at the checkpoint-save boundary too (redundant with the unconditional
  per-epoch save, kept deliberately as a second belt-and-suspenders point in case someone changes
  the per-epoch save to be conditional later).

## 4. GPU-sync minimization inside the per-step loop

A specific performance trap worth carrying forward: every `.item()` / `.sum().item()` call on a
CUDA tensor is a **hard synchronization point** — it blocks the CPU from queuing further GPU work
until the current kernel queue drains. Doing this multiple times per micro-batch (e.g. once each
for loss, hit@1, hit@5, repeated per gradient-accumulation split) serializes what should be an
async pipeline and measurably slows training.

Fix used here: accumulate loss/hit-counts as GPU tensors across the *whole* gradient-accumulation
inner loop (`torch.zeros((), device=data.device)` accumulators), and pull them to CPU with exactly
**one** `.tolist()` call at the end of the step:

```python
hit1_t = torch.zeros((), device=data.device)
hit5_t = torch.zeros((), device=data.device)
loss_t = torch.zeros((), device=data.device)
for j in range(args.batch_split):
    ...
    hit1_t += (...).sum()
    hit5_t += (...).sum()
    loss_t += loss.detach() / args.batch_split
hit1, hit5, loss_value = torch.stack([hit1_t, hit5_t, loss_t]).tolist()  # single sync
```

**Lesson for a new pipeline:** if you need a per-step scalar for logging (which this pipeline does
— both the per-step CSV and the periodic console print need one), you can't avoid *a* sync per
step, but you can and should avoid *multiple* syncs per step by batching every scalar you need into
one tensor and pulling it out with a single `.item()`/`.tolist()` call.

## 5. Per-epoch console summary line

```
== Epoch 3/15 done in 0:32:10.412000 (eta 6:42:15) lr 0.000038 train_loss(aug) 2.451203 train_acc1(aug) 41.20% train_acc5(aug) 68.90% | val_loss 2.108845 val_acc1 48.30% val_acc5 74.10%
```

- `(aug)` explicitly labels train metrics as computed **in training mode, with augmentation
  active** — i.e. these numbers are not directly comparable to validation accuracy (no
  augmentation, eval mode) and shouldn't be read as "the model is worse on train than val," which
  would otherwise be a confusing artifact to anyone reading the log without this label.
- `eta` is computed from the *actual* elapsed wall-clock time of completed epochs since resume,
  extrapolated to remaining epochs — not a fixed estimate — so it self-corrects if epoch time
  drifts (e.g. Colab throttling).
- The `| val_...` segment is appended conditionally — only present on epochs where validation
  actually ran that epoch (see `EVAL_EVERY_N_EPOCHS` below); this is the human-readable console
  mirror of the same `None`-on-skip rule used in the history dict (§3).

## 6. Validation cadence, checkpoint cadence, and confusion cadence — independent knobs

Three separate, independently-tunable cadences, set in Cell 6:

```python
SAVE_EVERY_N_EPOCHS = 1        # checkpoint after every epoch
EVAL_EVERY_N_EPOCHS = 5        # validation is the expensive part (bs=1, multi-view) -> less often
CONFUSION_EVERY_N_EPOCHS = 0   # confusion matrix + full classification report; 0 = disabled
```

**Why these are decoupled rather than one "log every N epochs" knob:** checkpointing is cheap on
this dataset size and buys precise resume — no reason to make it rare. Validation is measured to
cost ~82% of a whole epoch's wall time (batch_size=1, multi-view evaluation, run silently with no
progress bar) purely to produce a monitoring metric that doesn't feed back into training at all —
so it's deliberately made rarer than checkpointing. Confusion-matrix + full `classification_report`
generation is rarer still (or fully off) because it's the most expensive of the three and mainly
useful for periodic spot-checks, not every eval.

**Lesson for a new pipeline:** don't couple "how often do I save a resumable checkpoint" to "how
often do I run the expensive full-validation pass" to "how often do I generate the even-more-expensive
full confusion matrix / classification report." Each has a different cost and a different purpose
(resume safety vs. progress monitoring vs. deep diagnostic), and tying them together either makes
checkpointing too rare (risking lost resume progress) or makes the expensive diagnostics run far
more often than anyone actually looks at them.

## 7. A disabled logging pass, kept as a flag (not deleted)

An "eval-mode train accuracy" pass — re-running the model over the *entire* train split in eval
mode (no augmentation) purely to report a more paper-comparable train accuracy number — was found
to be the single largest eval-time cost on a T4 GPU (re-evaluating thousands of train samples every
time validation ran). It doesn't affect training in any way; it's a reporting-only metric. Rather
than deleting the code, it's gated behind a flag left in place and set to `False`, with a comment
explaining exactly why:

```python
# Eval-mode train accuracy (paper-quality) -- DISABLED
# Was re-running the model over all 4,368 train samples every eval epoch (the single
# biggest eval cost on a T4, cold-cache or not). Purely a reporting metric -- does not
# affect training -- so it is cut entirely. Flip EVAL_TRAIN_ENABLED to True to restore.
EVAL_TRAIN_ENABLED = False
```

The history dict still has `eval_train_loss`/`eval_train_acc1`/`eval_train_acc5` keys and the
plotting cell (§8) still has an axis for it — they just stay empty/`None` while the flag is off, so
flipping it back on later requires no code changes anywhere else, only this one flag.

**Lesson:** when you cut an expensive-but-sometimes-useful metric for performance reasons, prefer a
named, commented, defaulted-off flag over deleting the code — it documents *why* the metric isn't
running (so a future reader doesn't wonder if it's a bug) and makes turning it back on a one-line
change instead of a re-implementation.

## 8. Post-training artifacts (run once, not per-epoch)

**Cell 7b — training curves plot.** Reads the history JSON (or the in-memory `history` dict if
still in scope), produces a 3-panel figure (loss / augmented-mode accuracy / eval-mode accuracy),
annotates the best validation top-1 epoch directly on the accuracy plot, and saves one PNG to
Drive (`signvlm_training_curves.png`). Every series uses the same "filter out `None` before
plotting" helper:

```python
def _valid_xy(xs, ys):
    xv = [x for x, y in zip(xs, ys) if y is not None]
    yv = [y for y in ys if y is not None]
    return xv, yv
```
so validation (which only has a point every `EVAL_EVERY_N_EPOCHS`) plots correctly against the
denser train-loss curve without needing matching-length arrays.

**Cell 7c — final precision/recall/F1 table.** Runs the *final* trained model, in eval mode,
multi-view, once each over train and validation, computes accuracy + macro/weighted/micro
precision/recall/F1 via `sklearn.metrics.precision_recall_fscore_support`, prints an aligned
console table, and saves one CSV (`signvlm_final_metrics.csv`) to Drive. This is the "paper-quality
numbers" companion to the per-epoch augmented-mode numbers logged during training.

**Cell 8 — multi-split evaluation with confusion matrices.** Runs eval sequentially over
train → validation → test → unseen (each optional via a flag), **explicitly releasing each split's
DataLoader and dataset object (`del` + `gc.collect()` + `torch.cuda.empty_cache()`) before building
the next split's loader** — evaluating four splits back-to-back would otherwise accumulate RAM
across four separate loader/dataset instances that no longer need to coexist. For each split, saves
a `.npy` confusion matrix to Drive (`<split>_confusion_matrix.npy`) via a shared helper,
`print_confusion_and_f1`, that also prints macro/weighted/micro F1 and (optionally, gated by a
`fast_confusion_print` flag) the full `sklearn.classification_report` and raw matrix — gated because
printing a full report + matrix for every split, every run, is a lot of console noise most runs
don't need.

**Cell 10 — standalone full metrics, runnable in a fresh runtime.** The most complete artifact: for
each of train/validation/test_diff_signer, saves *both* a `.npy` confusion matrix *and* a rendered
`.png` heatmap (`imshow` with a colorbar, saved at 150 dpi), then a single combined CSV
(`signvlm_full_metrics_train_val_test.csv`) with accuracy + macro/weighted/micro precision/recall/F1
for every split, plus an aligned console table. This cell is deliberately independent of the
training loop's in-memory state (it rebuilds the model, reloads the latest checkpoint, and
regenerates its own eval loaders from scratch) — see `docs/DATA_INGESTION_CONTEXT.md` §10 for the
"any cell should be runnable standalone in a fresh runtime" design principle this follows.

## 9. Full artifact manifest (everything written, and by which cell)

| File | Written by | Cadence | Purpose |
|---|---|---|---|
| `/content/signvlm_step_log.csv` (local) | Cell 7 | every step | hot-path per-step log, pre-Drive-sync |
| `<LIST_DIR>/signvlm_step_log.csv` | Cell 7 | synced once/epoch | durable per-step log |
| `<LIST_DIR>/signvlm_loss_history.json` | Cell 7 | every epoch | per-epoch train/val loss+acc history |
| `<CHECKPOINT_DIR>/signvlm_epoch-<N>.pth` | Cell 7 | every `SAVE_EVERY_N_EPOCHS` | resumable checkpoint + embedded history |
| `<LIST_DIR>/validation_confusion_matrix.npy` | Cell 7 (if `CONFUSION_EVERY_N_EPOCHS>0`) | periodic during training | mid-training validation confusion matrix |
| `<LIST_DIR>/signvlm_training_curves.png` | Cell 7b | once, post-training | loss/accuracy curves |
| `<LIST_DIR>/signvlm_final_metrics.csv` | Cell 7c | once, post-training | final-model P/R/F1, train+val |
| `<LIST_DIR>/<split>_confusion_matrix.npy` | Cell 8 | on demand, per split | per-split confusion matrix (train/val/test/unseen) |
| `<LIST_DIR>/<split>_confusion_matrix.npy` + `.png` | Cell 10 | on demand, per split | standalone confusion matrix + heatmap |
| `<LIST_DIR>/signvlm_full_metrics_train_val_test.csv` | Cell 10 | on demand | standalone combined metrics table |

`LIST_DIR` and `CHECKPOINT_DIR` are both under Drive (`MyDrive/FYP/Models/...`) — see
`docs/DATA_INGESTION_CONTEXT.md` §2 for why nothing log/checkpoint-related is ever written *only*
to ephemeral `/content` (with the sole, deliberate exception of the per-step CSV's hot-path copy,
which always has a Drive-synced counterpart).

## 10. Principles to carry into a new pipeline's logging design

1. **Split logging into a fast/local tier and a periodically-synced durable tier** whenever the
   durable destination (Drive, S3, a network share) is too slow to write on every step. Sync on a
   cadence (once per epoch is usually right), not per step — and always sync one final time in a
   `finally` block so a crash doesn't lose the tail.
2. **Make cadences for checkpointing / validation / expensive diagnostics independent knobs**, not
   one shared "log every N" setting — they have different costs and different purposes.
3. **Represent "this metric didn't run this epoch" as `None` at that epoch's index, not as a
   missing/shorter list.** Keeps every series index-aligned with the epoch axis; filter `None`s out
   only at the point of use (plotting, "find best epoch," etc.).
4. **Embed the run history inside checkpoints, not only in a side file.** A checkpoint should be
   self-describing — loadable in a fresh runtime with its own full history attached, not dependent
   on a separate JSON file surviving alongside it.
5. **Batch GPU-tensor scalars into one accumulator and pull them out with a single `.item()`/
   `.tolist()` call per step**, if you need per-step logging values — avoid one sync per metric.
6. **Label metrics with the mode they were computed in** (`train_loss(aug)` vs eval-mode) directly
   in both the console output and the plot legends — augmented/train-mode numbers and eval-mode
   numbers are not comparable, and unlabeled logs invite someone to misread the gap as a bug.
7. **When you cut an expensive metric for performance, gate it behind a named, commented,
   defaulted-off flag — don't delete the code.** Document the specific cost that justified cutting
   it, so a future reader knows it's a deliberate tradeoff, not an oversight.
8. **Explicitly release DataLoader/dataset objects between sequential eval passes** (`del` +
   `gc.collect()` + `torch.cuda.empty_cache()`) rather than holding several splits' loaders in
   memory simultaneously when they're only ever used one at a time.
9. **Design the "final metrics" and "standalone eval" cells to be independently runnable in a fresh
   process/runtime**, rebuilding model/checkpoint/loaders from persisted state rather than assuming
   any earlier cell's in-memory globals are present — this is what lets you check metrics on a
   single split (e.g. a held-out test set) without re-running the entire training pipeline.
