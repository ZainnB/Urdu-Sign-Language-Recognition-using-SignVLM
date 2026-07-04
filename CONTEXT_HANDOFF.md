# SignVLM Notebook — Output Format Handoff

## Task
Update **Cell 7** of `notebooks/SignVLM_Colab_Training.ipynb` so its training output matches the ViViT-style format shown below. No other cells need to change.

---

## Repo
- **Repo:** `c:\ARK\Urdu-Sign-Language-Recognition-using-SignVLM` (branch: `training_thru_tensors`)
- **Target file:** `notebooks/SignVLM_Colab_Training.ipynb`
- **Target cell:** cell id `067d31b0`, index 16 in `nb['cells']`

---

## Desired Output Format (match this exactly)

```
===== Epoch 1/15 =====
Training: 100%|██████████| 156/156 [1:18:51<00:00, 30.33s/it]
[TRAIN] Loss: 4.2718, Acc: 0.1284, Prec: 0.1598, Recall: 0.1284, F1: 0.1278
[VAL]   Loss: 3.7814, Acc: 0.3137, Prec: 0.3849, Recall: 0.3137, F1: 0.2951
[INFO] ✓ New best model saved! Val Loss: 3.7814
[INFO] Checkpoint saved: /content/drive/MyDrive/.../signvlm_epoch-1.pth

===== Epoch 2/15 =====
Training: 100%|██████████| 156/156 [1:18:24<00:00, 30.16s/it]
[TRAIN] Loss: 3.3621, Acc: 0.4489, Prec: 0.4726, Recall: 0.4489, F1: 0.4437
[VAL]   Loss: 3.1398, Acc: 0.5132, Prec: 0.6022, Recall: 0.5132, F1: 0.5074
[INFO] ✓ New best model saved! Val Loss: 3.1398
[INFO] Checkpoint saved: /content/drive/MyDrive/.../signvlm_epoch-2.pth

...

============================================================
EVALUATING ON TEST SET
============================================================

[TEST RESULTS]
  Loss:      1.4398
  Accuracy:  0.8750 (87.50%)
  Precision: 0.8924
  Recall:    0.8750
  F1 Score:  0.8736
============================================================
```

---

## What the Current Cell 7 Produces (wrong)

- No epoch header
- Raw `print()` every N steps instead of tqdm bar
- End-of-epoch line: `== Epoch 1/N done in ... lr X train_loss X train_acc1 X% train_acc5 X% | val_loss X val_acc1 X%`
- No Precision / Recall / F1 in train or val summaries
- No `[INFO] ✓ New best model saved!` message
- No `[INFO] Checkpoint saved:` message
- No inline test evaluation (test eval is a separate Cell 8)

---

## Changes Required to Cell 7

### 1. New imports at top of cell
```python
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
```
(`sklearn` is already installed in Cell 1; `tqdm` is pre-installed in Colab.)

### 2. Update `evaluate_with_loss` — add Prec/Recall/F1 return values
Current signature returns `(loss, acc1, acc5)`.
New signature must return `(loss, acc1, acc5, prec, rec, f1)`.

Collect `all_preds` and `all_labels` lists during the eval loop, then compute:
```python
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
```

### 3. Training loop changes
- Add `best_val_loss = float("inf")` before the epoch loop
- Print `===== Epoch {epoch+1}/{NUM_EPOCHS} =====` at the start of each epoch
- Replace the inner `for i, (data, labels) in enumerate(train_loader):` with a `tqdm` wrapped version:
  ```python
  pbar = tqdm(train_loader, desc="Training", total=STEPS_PER_EPOCH)
  for data, labels in pbar:
      ...
      pbar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
  ```
- Remove the old `if global_step % args.print_freq == 0: print(...)` block
- Accumulate `train_preds_all` and `train_labels_all` during the batch loop (from `logits.topk(1)`)
- After the epoch, compute train Prec/Recall/F1 from accumulated lists
- Print: `[TRAIN] Loss: X.XXXX, Acc: X.XXXX, Prec: X.XXXX, Recall: X.XXXX, F1: X.XXXX`
- Print: `[VAL]   Loss: X.XXXX, Acc: X.XXXX, Prec: X.XXXX, Recall: X.XXXX, F1: X.XXXX`
- Best model logic (save + print message) when `val_loss < best_val_loss`
- Fallback periodic save when not best

### 4. Inline test evaluation at the end (after the epoch loop)
```python
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET")
print("=" * 60)
_test_loader = make_eval_split_loader(test_tsv, TEST_DIR)
test_loss, test_acc1, test_acc5, test_prec, test_rec, test_f1 = evaluate_with_loss(model, _test_loader, criterion)
del _test_loader
print(f"\n[TEST RESULTS]")
print(f"  Loss:      {test_loss:.4f}")
print(f"  Accuracy:  {test_acc1:.4f} ({test_acc1 * 100:.2f}%)")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")
print("=" * 60)
```

---

## How to Edit the Notebook File

The `.ipynb` is JSON. Use Python to load, replace `nb['cells'][16]['source']`, and save:

```python
import json

path = r'notebooks/SignVLM_Colab_Training.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][16]
assert cell['id'] == '067d31b0'   # safety check

# set cell['source'] to the new lines list and save
...

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
```

---

## Key Facts About the Existing Code

- `STEPS_PER_EPOCH`, `NUM_EPOCHS`, `EVAL_EVERY_N_EPOCHS`, `SAVE_EVERY_N_EPOCHS`, `CONFUSION_EVERY_N_EPOCHS` — set in **Cell 6**
- `args.batch_split`, `args.fp16` — set in Cell 6
- `save_epoch_checkpoint(epoch, history)` and `_epoch_ckpt_path(epoch)` — defined in Cell 6
- `make_eval_split_loader(tsv, root)` — defined in Cell 4
- `collect_predictions`, `print_confusion_and_f1` — defined in Cell 3 (metrics helpers)
- `train_loader`, `val_loader`, `test_tsv`, `TEST_DIR` — defined in Cell 4
- `history` dict and `resume_epoch` — set by `resume_epoch_checkpoint()` in Cell 6
- `optimizer`, `lr_sched`, `loss_scaler`, `criterion`, `model` — defined in Cell 6
- `LIST_DIR` — set in Cell 2

---

## Notes
- `evaluate_with_loss` currently only computes loss/acc for the val loop. In the new version it must also return Prec/Recall/F1 (collected during the per-sample loop, not reduced via dist).
- Train Prec/Recall/F1 are computed in **train mode** from accumulated batch predictions (not eval mode) — this matches the ViViT approach.
- The `dist.all_reduce` sync for loss/hit counts should remain for correctness on multi-GPU; the preds lists are local-only (world_size=1 on Colab).
- Cell 7b (loss curve plotting) and Cell 7c (final metrics table) do **not** need changes.
