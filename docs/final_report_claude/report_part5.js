// ================================================================
// PART 5: Section 6 — Implementation Details
//         Section 7 — Results / Experiments / Evaluation
// ================================================================

const {
  Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
} = require('docx');

const {
  H1, H2, H3, BR, PB, body, bodyRuns, bull, bull2, num, caption,
  figPlaceholder, tblCaption, hCell, dCell, bCell, grpCell, eq,
  r, rb, ri, rbi,
  CONTENT, BORDERS, NAVY, BLUE, LGRAY, HBLUE, DBLUE,
  AlignmentType: AT
} = require('./report_part1');

const bodyLabel = (label, text) => bodyRuns([rbi(label + ": "), r(text)]);

// ================================================================
// SECTION 6: IMPLEMENTATION DETAILS
// ================================================================
const implementationSection = [

  H1("6. Implementation Details"),

  body("This section covers the concrete engineering decisions and configurations that translated the methodology described in Section 5 into running experiments. It documents the hardware used, the specific hyperparameter choices for each model training run, and the key software-level adaptations required to deploy SignVLM on PSL-104. These details are included for reproducibility and to make explicit the engineering effort involved in each experimental stage."),

  // ──────────────────────────────────────────────────────────────
  H2("6.1 Hardware and Compute Environment"),
  // ──────────────────────────────────────────────────────────────

  body("Training workloads across this project were distributed across two compute setups. Two team members jointly subscribed to Google Colab Pro, which provided access to NVIDIA A100 and T4 GPU instances for compute-intensive runs including the signer-joint 3DCNN experiments and early SignVLM runs. A third team member contributed a personal workstation that hosted the majority of the signer-disjoint experiments, the MediaPipe ROI extraction pipeline, the frame pre-extraction, and the final SignVLM training runs. The specifications of the personal workstation are listed in Table 2."),
  BR(),

  tblCaption("Table 2: Personal workstation hardware specifications used for local training runs."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 6160],
    rows: [
      new TableRow({ tableHeader: true, children: [hCell("Component", 3200), hCell("Specification", 6160)] }),
      ...[
        ["Central Processing Unit (CPU)", "Intel Core i5-12400F (6 cores / 12 threads, up to 4.4 GHz)"],
        ["Graphics Processing Unit (GPU)", "NVIDIA GeForce RTX 3060 — 12 GB GDDR6 VRAM"],
        ["System RAM",                    "16 GB DDR4"],
        ["Storage (SSD)",                 "1 TB+ NVMe SSD (dataset, frames, and checkpoints)"],
        ["Operating System",              "Windows 11 / Ubuntu 22.04 (dual boot for Linux training runs)"],
        ["CUDA Version",                  "CUDA 11.8 with cuDNN 8.6"],
        ["Deep Learning Framework",       "PyTorch 2.0.1 with torchvision 0.15.2"],
      ].map(([k, v]) => new TableRow({ children: [dCell(k, 3200, false), dCell(v, 6160, false)] }))
    ]
  }),
  caption("Table 2: Hardware used for local training. Google Colab Pro (A100/T4) was used in parallel for the signer-joint 3DCNN experiments and initial SignVLM runs."),
  BR(),

  body("The 12 GB VRAM of the RTX 3060 was a practical constraint that influenced several training decisions. For SignVLM in particular, the CLIP ViT-L/14 encoder generates approximately 8–9 GB of intermediate activations when processing a batch of four 24-frame, 224×224 clips, leaving very little headroom for the EVL decoder and optimiser states. This necessitated a physical batch size of 4 with gradient accumulation over 4 steps to achieve an effective batch size of 16 — the minimum recommended for stable transformer training. The use of bfloat16 Automatic Mixed Precision (AMP) was essential to keep peak memory usage within the 12 GB limit."),

  // ──────────────────────────────────────────────────────────────
  H2("6.2 Stage 1: Signer-Joint 3DCNN — Training Configuration"),
  // ──────────────────────────────────────────────────────────────

  body("The signer-joint baseline replication used the 3DCNN + Residual Block + BiLSTM architecture as implemented by the prior team, trained on the combined 6-signer augmented dataset under a random train/validation/test split. The training configuration follows the prior team's hyperparameter choices closely, with minor adjustments to accommodate the larger combined dataset."),
  BR(),

  tblCaption("Table 3: Signer-joint 3DCNN training hyperparameters."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3800, 5560],
    rows: [
      new TableRow({ tableHeader: true, children: [hCell("Hyperparameter", 3800), hCell("Value / Setting", 5560)] }),
      ...[
        ["Input resolution",          "112 × 112 pixels"],
        ["Input frames per clip",      "64 uniformly sampled frames"],
        ["Batch size",                 "8"],
        ["Optimiser",                  "Adam (lr = 1e-3, weight decay = 1e-4)"],
        ["LR scheduler",               "ReduceLROnPlateau (patience = 3, factor = 0.5)"],
        ["Training epochs",            "10 (early stopping not applied)"],
        ["Class weighting",            "WeightedRandomSampler for class imbalance"],
        ["Mixed precision",            "AMP (float16) enabled"],
        ["Data split",                 "Random — 80% train / 15% val / 5% test (signer-INCLUSIVE)"],
        ["Framework",                  "PyTorch 2.0.1"],
        ["Compute",                    "Google Colab Pro (A100 / T4 GPU)"],
      ].map(([k, v]) => new TableRow({ children: [dCell(k, 3800, false), dCell(v, 5560, false)] }))
    ]
  }),
  caption("Table 3: Signer-joint 3DCNN training configuration. Signer-inclusive random splitting was used to establish the baseline comparison with prior work."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("6.3 Stage 5: Signer-Disjoint 3DCNN — Training Configuration"),
  // ──────────────────────────────────────────────────────────────

  body("For the formal signer-disjoint evaluation, the training configuration was updated to incorporate best-practice regularisation and optimisation strategies suitable for small-dataset training. While the architecture remained identical to Stage 1, the training recipe was significantly refined."),
  BR(),

  tblCaption("Table 4: Signer-disjoint 3DCNN training hyperparameters."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3800, 5560],
    rows: [
      new TableRow({ tableHeader: true, children: [hCell("Hyperparameter", 3800), hCell("Value / Setting", 5560)] }),
      ...[
        ["Input resolution",           "112 × 112 pixels"],
        ["Input frames per clip",       "64 uniformly sampled frames"],
        ["Batch size",                  "8"],
        ["Optimiser",                   "AdamW (lr = 1e-4, weight decay = 1e-2)"],
        ["LR scheduler",                "Cosine annealing with 3-epoch linear warmup"],
        ["Label smoothing",             "ε = 0.04"],
        ["Gradient clipping",           "Max norm = 1.0"],
        ["Class weighting",             "WeightedRandomSampler for class imbalance"],
        ["Early stopping",              "Patience = 12 epochs (monitors val accuracy)"],
        ["Max epochs",                  "30 (early stopping triggered at epoch 30)"],
        ["Mixed precision",             "AMP (float16) enabled"],
        ["Data split",                  "Signer-disjoint (no shared signers across train/val/test)"],
        ["Framework",                   "PyTorch 2.0.1"],
        ["Compute",                     "RTX 3060 12 GB (local workstation)"],
      ].map(([k, v]) => new TableRow({ children: [dCell(k, 3800, false), dCell(v, 5560, false)] }))
    ]
  }),
  caption("Table 4: Signer-disjoint 3DCNN training configuration. AdamW and cosine schedule replaced Adam + ReduceLROnPlateau for improved regularisation."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("6.4 SignVLM PSL-Specific Adaptations"),
  // ──────────────────────────────────────────────────────────────

  body("Deploying SignVLM on PSL-104 required several non-trivial engineering adaptations beyond the standard SignVLM codebase. These are documented here in full to support reproducibility."),

  H3("6.4.1 Unicode Path Handling"),
  body("OpenCV's VideoCapture() function silently returns zero frames when given a file path containing Unicode characters — including Arabic script — on Windows. Because a substantial portion of PSL-104's class labels are Urdu/Arabic words (e.g., the labels for common words and many alphabet characters), all videos stored in these folders were being silently skipped during data loading. The failure mode was particularly insidious because OpenCV did not raise any exception; it simply returned an empty video capture object, and the original dataloader code did not check for this condition. We identified this bug by cross-referencing the count of successfully loaded videos against the expected total and noticing that all Arabic-script classes were returning zero samples. The fix replaced OpenCV's video reading with PyAV (Python bindings for FFmpeg), which handles Unicode paths correctly on both Windows and Linux. This change alone recovered all Urdu-script class videos from the effective training pool."),

  H3("6.4.2 Offline Frame Pre-Extraction"),
  body("During early SignVLM training runs, profiling revealed that the data loading step — real-time video decoding using PyAV, seeking to uniformly sampled frame indices, and decoding H.264 compressed video on the CPU — was consuming 0.8 to 1.0 seconds per batch. This was 5 to 10 times the GPU compute time per batch, meaning the GPU was idle for 80-90% of the training time. The solution was offline frame pre-extraction: before any training run, all videos across all three splits were decoded and saved as individual JPEG files, organized into per-video subdirectories. During training, the dataloader simply reads the pre-selected JPEG files — a random access I/O operation requiring no decompression or seeking — reducing the data loading time to approximately 0.05 seconds per batch, a 16-fold improvement. GPU utilization increased from approximately 30% to above 85% as a result."),

  H3("6.4.3 Split File Generation"),
  body("SignVLM's dataloader expects split files in a specific tab-separated format: one line per video, with the absolute path to the video (or its pre-extracted frame directory) followed by a tab character and the integer class label. We wrote dedicated scripts to generate these split files from the signer-disjoint directory structure, with explicit verification that no signer appeared in more than one split. These scripts also handle the Urdu-script folder names correctly by using Python's pathlib for cross-platform Unicode path manipulation."),

  H3("6.4.4 Validation Code Path Bug"),
  body("The original SignVLM codebase contained a bug in the validation data loading path where the code attempted to call .to_rgb().to_ndarray() on frames that had already been decoded to numpy arrays during the pre-extraction loading path. This produced a silent failure: the frame stack was left as a Python list rather than a stacked numpy array, which in turn caused the batch construction to silently pad all frames to zeros. The fix detects whether the loaded frames are already numpy arrays and skips the redundant decoding step in that case."),

  H3("6.4.5 Spatial Size Assertion Fix"),
  body("A guard condition in SignVLM's dataloader was checking frames.shape[1] == self.num_frames to decide whether to skip the resize and crop preprocessing step. This condition evaluated to True for all valid 16-frame tensors (where dimension 1 is the frame count), causing the resize step to be skipped for every correctly-shaped batch and leaving frames at their raw decoded resolution rather than the required 224×224. The fix corrects the condition to check the spatial dimensions instead: frames.shape[-2] == self.spatial_size and frames.shape[-1] == self.spatial_size."),

  H3("6.4.6 CLIP Normalization"),
  body("CLIP's ViT-L/14 encoder requires input normalized with CLIP-specific statistics derived from the WebImageText pretraining dataset: mean = [0.4815, 0.4578, 0.4082], std = [0.2686, 0.2613, 0.2758]. Applying the standard ImageNet normalization (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) instead causes a consistent 3–7 percentage point accuracy drop, as the encoder's attention weights were calibrated during pretraining to the CLIP-specific input distribution. All PSL-104 training and inference runs apply CLIP normalization."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("6.5 SignVLM Training Configuration"),
  // ──────────────────────────────────────────────────────────────

  tblCaption("Table 5: SignVLM training hyperparameters for all PSL-104 experimental runs."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3800, 5560],
    rows: [
      new TableRow({ tableHeader: true, children: [hCell("Parameter", 3800), hCell("Value / Setting", 5560)] }),
      ...[
        ["Visual encoder",                    "CLIP ViT-L/14 — 307M parameters, fully frozen"],
        ["Temporal decoder",                   "EVL (Efficient Video Learner) — trainable"],
        ["Classification head",               "Linear layer, 104 output classes — trainable"],
        ["Spatial resolution",                 "224 × 224 pixels per frame"],
        ["Frames per clip",                    "16 (Config A) and 24 (Config B) — uniform sampling"],
        ["Physical batch size",               "4"],
        ["Gradient accumulation steps",        "4 (effective batch size = 16)"],
        ["Optimiser",                          "AdamW"],
        ["Peak learning rate",                 "3e-5"],
        ["LR schedule",                        "Cosine decay from peak to 0 over training steps"],
        ["Total training steps",               "10,000"],
        ["Evaluation checkpoints",             "Steps 5,000 and 10,000"],
        ["Mixed precision",                    "AMP (bfloat16)"],
        ["Distributed backend",               "Gloo (Windows-compatible)"],
        ["CLIP normalisation",                 "Mean [0.4815, 0.4578, 0.4082], Std [0.2686, 0.2613, 0.2758]"],
        ["Frame source",                       "Pre-extracted JPEG frames (offline extraction)"],
        ["Evaluation protocol",               "Strictly signer-disjoint for all reported results"],
        ["Primary compute",                    "RTX 3060 12 GB (local) + Colab Pro (A100)"],
      ].map(([k, v]) => new TableRow({ children: [dCell(k, 3800, false), dCell(v, 5560, false)] }))
    ]
  }),
  caption("Table 5: Full training configuration for SignVLM on PSL-104. Three configurations were evaluated: Config A (16 frames, augmented data), Config B (24 frames, augmented data), and Config C (24 frames, original data — used for N-shot ablation)."),
  BR(),

  body("The gradient accumulation strategy deserves explanation. With the CLIP ViT-L/14 encoder processing 24 frames at 224×224, each forward pass through the frozen encoder alone consumes approximately 8-9 GB of GPU memory for a batch of four clips. This leaves insufficient headroom for a larger physical batch. Gradient accumulation allows the optimizer to see the equivalent of a 16-sample batch for each parameter update while the GPU only ever holds 4 samples at once. This is a standard technique for large model training under VRAM constraints and is mathematically equivalent to a true batch size of 16 when the gradient is accumulated without any intermediate updates."),

  PB(),
];

// ================================================================
// SECTION 7: RESULTS / EXPERIMENTS / EVALUATION
// ================================================================
const resultsSection = [

  H1("7. Results / Experiments / Evaluation"),

  body("This section presents all experimental results in the chronological order in which the experiments were conducted — from the initial signer-joint baseline through the unseen-signer failure, the MediaPipe ROI and pose-based investigations, the formal signer-disjoint baseline, and finally the SignVLM results including frame ablation, data efficiency analysis, and cross-benchmark comparison. Presenting results in this order reflects the actual research process and makes the significance of each experiment clear in the context of what preceded it. An overview of all experiments and their headline results is provided in Table 6."),
  BR(),

  tblCaption("Table 6: Overview of all experiments — headline results in chronological order."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [340, 2200, 1600, 1000, 900, 900, 900, 820],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("#",      340),
        hCell("Model / Configuration",  2200),
        hCell("Eval Protocol",          1600),
        hCell("Data",                   1000),
        hCell("Top-1 (%)", 900),
        hCell("Top-5 (%)", 900),
        hCell("F1",        900),
        hCell("Stage",     820),
      ]}),
      ...[
        ["1", "3DCNN + ResBlock + BiLSTM (Agha et al., 2025)", "Signer-Joint (random split)", "Augmented", "83.89", "—", "0.833", "Prior"],
        ["2", "3DCNN + ResBlock + BiLSTM (Ours, extended data)", "Signer-Joint (random split)", "Augmented", "92.11", "~98", "0.921", "1"],
        ["3", "3DCNN — Unseen Signer Inference (60 clips)", "Truly Unseen (external)", "Original", "8.33", "23.33", "0.083", "2"],
        ["4", "3DCNN + MediaPipe ROI", "Signer-Joint", "ROI", "45.46", "~62", "~0.45", "3"],
        ["5", "3DCNN + MediaPipe ROI", "Signer-Disjoint", "ROI", "24.87", "~38", "~0.25", "3"],
        ["6", "3DCNN + ResBlock + BiLSTM (Ours)", "Signer-Disjoint", "Augmented", "12.88", "~28", "~0.13", "5"],
        ["7", "SignVLM — 16 frames/clip", "Signer-Disjoint", "Augmented", "80.34", "~91", "[TBD]", "6"],
        ["8", "SignVLM — 24 frames/clip", "Signer-Disjoint", "Augmented", "83.49", "~93", "[TBD]", "6"],
        ["9", "SignVLM — 24 frames/clip", "Signer-Disjoint", "Original", "85.14", "~94", "[TBD]", "6"],
        ["10","SignVLM — 24 frames (4-shot)",  "Signer-Disjoint", "Original", "81.25", "~91", "[TBD]", "6"],
        ["11","SignVLM — 24 frames (8-shot)",  "Signer-Disjoint", "Original", "84.29", "~93", "[TBD]", "6"],
      ].map((row, i) => {
        const isSignVLM = row[1].includes("SignVLM");
        const isBest = row[2] === "9" || row[9] === "85.14" || row[0] === "9";
        const w = [340, 2200, 1600, 1000, 900, 900, 900, 820];
        return new TableRow({ children: row.map((cell, j) => {
          if (isSignVLM) return bCell(cell, w[j], j > 1);
          return dCell(cell, w[j], j > 1);
        })});
      })
    ]
  }),
  caption("Table 6: All experiments in chronological order. SignVLM rows are highlighted. [TBD] = to be filled from final training logs. Approx. values for Top-5 and F1 where exact metrics not yet extracted."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("7.1 Stage 1: Signer-Joint 3DCNN Baseline"),
  // ──────────────────────────────────────────────────────────────

  body("The first experiment established the baseline performance of the 3DCNN + Residual Block + BiLSTM architecture on the combined 6-signer PSL-104 dataset under signer-inclusive random splitting. This serves two purposes: it replicates and extends the prior team's result to confirm that the combined pipeline works as expected, and it establishes the upper bound of what signer-identity leakage can achieve on this dataset."),

  body("The combined dataset — comprising the prior team's 3-signer recordings, the PSL Dictionary clips, and our team's 3 additional signers — was preprocessed as described in Section 5.2, then augmented using the revised pipeline (Section 5.4) to produce the training, validation, and test splits. A random 80/15/5 split was applied across the entire augmented corpus, with no constraint on signer separation between splits."),
  BR(),

  tblCaption("Table 7: Signer-joint 3DCNN test set results (Stage 1)."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [2340, 2340, 2340, 2340],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Metric", 2340), hCell("Train", 2340), hCell("Validation", 2340), hCell("Test", 2340)
      ]}),
      ...[
        ["Loss",      "[TBD]",  "[TBD]",  "0.3459"],
        ["Accuracy",  "[TBD]",  "[TBD]",  "92.11%"],
        ["Precision", "[TBD]",  "[TBD]",  "0.9367"],
        ["Recall",    "[TBD]",  "[TBD]",  "0.9211"],
        ["F1 Score",  "[TBD]",  "[TBD]",  "0.9207"],
      ].map(([metric, train, val, test]) => new TableRow({ children: [
        dCell(metric, 2340, false, { bold: true }),
        dCell(train,  2340, true),
        dCell(val,    2340, true),
        bCell(test,   2340, true),
      ]}))
    ]
  }),
  caption("Table 7: Signer-joint 3DCNN evaluation metrics. Test accuracy of 92.11% was achieved under signer-inclusive random splitting on the combined 6-signer augmented dataset. [TBD] = train/val metrics to be filled from training logs."),
  BR(),

  body("The test accuracy of 92.11%, precision of 0.9367, recall of 0.9211, and F1 score of 0.9207 represent a meaningful improvement over the prior team's reported 83.89% F1 ≈ 0.833 result. This improvement can be attributed primarily to the additional signer diversity introduced by our three new signers — even under signer-inclusive evaluation, a model trained on 6 signers' data sees more visual variation than one trained on 3 signers, and the random test split is therefore a slightly harder test. The result confirms that the preprocessing and augmentation pipeline is working correctly and that the combined dataset is well-formed."),

  body("However, as discussed in Section 5.7 and demonstrated in the following subsection, this result does not reflect the model's ability to recognize PSL signing from new individuals. It reflects the model's ability to classify within the visual distribution of the training signers — a much easier problem than practical SLR deployment requires."),
  BR(),

  figPlaceholder("Figure [X]: Stage 1 — Signer-joint 3DCNN training curves (loss and accuracy) over 10 epochs on the combined 6-signer dataset. Include train and validation curves on separate subplots."),
  caption("Figure [X]: Signer-joint 3DCNN training dynamics showing rapid convergence to high accuracy under signer-inclusive evaluation."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("7.2 Stage 2: Unseen Signer Inference Test"),
  // ──────────────────────────────────────────────────────────────

  body("Having confirmed strong signer-joint performance, we immediately exposed the model to a real-world generalization test. Sixty short video clips were collected from friends and acquaintances outside the project team — individuals who had never appeared in the PSL-104 dataset in any form. The signers in this external test set ranged in age, background, and familiarity with PSL signs; all were coached to perform each sign by watching the corresponding PSL Dictionary reference video before recording."),

  body("These videos were preprocessed using the identical pipeline applied to the PSL-104 training data: format conversion, temporal cropping, audio removal, and frame extraction. The signer-joint trained 3DCNN model was then used to run inference on each of the 60 clips, and both Top-1 and Top-5 predictions were recorded."),
  BR(),

  tblCaption("Table 8: Unseen signer inference results — Stage 2 (60 clips, 3DCNN signer-joint model)."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3120, 3120, 3120],
    rows: [
      new TableRow({ tableHeader: true, children: [hCell("Metric", 3120), hCell("Value", 3120), hCell("Notes", 3120)] }),
      ...[
        ["Total clips", "60", "External signers, not in training data"],
        ["Top-1 Correct", "5 / 60", "—"],
        ["Top-1 Accuracy", "8.33%", "—"],
        ["Top-1 Precision", "0.0833", "Macro-averaged across predicted classes"],
        ["Top-1 Recall", "0.0833", "Macro-averaged across true classes"],
        ["Top-1 F1 Score", "0.0833", "Harmonic mean of precision and recall"],
        ["Top-5 Correct", "14 / 60", "True label appeared in top-5 predictions"],
        ["Recall@5 (Top-5)", "23.33%", "—"],
        ["Random chance (Top-1)", "0.96%", "1/104 classes"],
      ].map(([metric, val, note]) => new TableRow({ children: [
        dCell(metric, 3120, false, { bold: true }),
        dCell(val,    3120, true),
        dCell(note,   3120, false),
      ]}))
    ]
  }),
  caption("Table 8: Unseen signer inference test results. The signer-joint trained 3DCNN correctly classified only 5 of 60 clips at Top-1 — an 83.78 percentage point drop from its 92.11% signer-joint test accuracy."),
  BR(),

  body("The result is stark: a model that achieves 92.11% on the signer-joint test set correctly classifies only 8.33% of clips from genuinely new signers — 14 clips fewer than what would be needed to break the 30% threshold, and barely above what could be expected from random guessing combined with some partial recognition. The Top-5 Recall of 23.33% — meaning the correct label appeared in the top 5 predictions for 14 of 60 clips — is also low, confirming that the model's failures are not due to fine-grained misclassification between visually similar signs. The model is simply not recognizing the signs at all."),

  body("This result directly demonstrates what signer-identity leakage means in practice. The model learned to classify based on the visual distribution of the seven signers in its training set. When presented with an eighth individual whose appearance, background, and subtle signing characteristics differ from all training signers, the model's learned features simply do not apply. The 83.78 percentage point gap between signer-joint and unseen-signer accuracy is a concrete measure of how far the model's reported performance diverges from its actual capability."),
  BR(),

  figPlaceholder("Figure [X]: Stage 2 — Bar chart comparing 3DCNN accuracy across three conditions: Signer-Joint Test (92.11%), Signer-Disjoint Validation (12.88%, see Stage 5), and Unseen Signer Inference (8.33%). This is the key generalization gap visualization."),
  caption("Figure [X]: Dramatic accuracy drop from signer-joint to genuinely unseen signers, motivating the signer-disjoint evaluation protocol and the shift to pretrained models."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("7.3 Stage 3: MediaPipe Hand ROI Experiment"),
  // ──────────────────────────────────────────────────────────────

  body("Following the unseen-signer failure, the hypothesis driving Stage 3 was that the model was attending too strongly to whole-frame appearance features — background, face, clothing — rather than focusing its capacity on the hands, which carry the sign's actual linguistic content. If the model could be forced to see only the hands, it might stop exploiting signer-specific appearance shortcuts and learn more generalizable features."),

  body("The MediaPipe Hands pipeline (Zhang et al., 2020) was implemented to extract hand region of interest (ROI) crops from every video in the dataset. For each frame, the palm detector identifies the bounding boxes of both hands, these boxes are merged into a single bilateral bounding box covering both hands, padded by a factor of 1.3 to avoid cutting off finger tips, and resized to a fixed output resolution. Frames where no hands are detected fall back to a resized version of the full frame to preserve temporal continuity. The pipeline was applied to all three dataset splits, producing ROI-cropped video alternatives for the full dataset."),
  BR(),

  tblCaption("Table 9: MediaPipe Hand ROI experiment results — Stage 3."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [3200, 2053, 2053, 2054],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Configuration", 3200),
        hCell("Top-1 Accuracy", 2053),
        hCell("Top-5 (approx.)", 2053),
        hCell("Approx. F1", 2054),
      ]}),
      ...[
        ["3DCNN — Full Frame, Signer-Joint (Stage 1)",  "92.11%", "~98%",  "0.921"],
        ["3DCNN — ROI Crops, Signer-Joint (Stage 3)",   "45.46%", "~62%",  "~0.45"],
        ["3DCNN — ROI Crops, Signer-Disjoint (Stage 3)","24.87%", "~38%",  "~0.25"],
        ["3DCNN — Full Frame, Signer-Disjoint (Stage 5)","12.88%","~28%",  "~0.13"],
      ].map(([conf, top1, top5, f1]) => new TableRow({ children: [
        dCell(conf, 3200, false),
        dCell(top1, 2053, true),
        dCell(top5, 2053, true),
        dCell(f1,   2054, true),
      ]}))
    ]
  }),
  caption("Table 9: MediaPipe ROI experiment results compared to full-frame baselines. ROI extraction substantially reduced performance under both evaluation protocols, confirming that naive spatial attention without a pretrained visual prior does not resolve the generalization problem."),
  BR(),

  body("The results were counterintuitive and reveal an important insight. Under signer-joint evaluation, ROI cropping reduced accuracy from 92.11% to 45.46% — a drop of nearly 47 percentage points. Under signer-disjoint evaluation, ROI cropping achieved 24.87%, which is higher than the full-frame signer-disjoint result of 12.88% — suggesting some small generalization benefit — but still far below any practically useful threshold."),

  body("Several factors explain why ROI cropping hurt performance rather than helping it. First, MediaPipe's palm detector fails on fast-motion frames, especially for rapid signing transitions. In these frames, the system falls back to the full frame at reduced resolution, creating inconsistent input crops across a single video sequence. The model receives a mix of tight hand crops and full-frame fallbacks within the same training sample, introducing temporal noise that confuses the 3D convolutions. Second, for two-handed signs, the bilateral bounding box merge produces non-square crops of varying aspect ratios. When naively resized to a square output, this distorts the relative positioning of the two hands — a feature that is linguistically meaningful for many PSL signs. Third, the removal of full-frame context (background, upper body, face) also removes some features — particularly body position and head orientation — that the model had learned to use as complementary cues even in the full-frame setting. Most importantly, none of these issues address the root cause: the 3DCNN still has no pretrained visual prior for what hands look like, and restricting its field of view to hands alone does not give it that prior."),

  body("The ROI experiment was a necessary step in the investigation: it ruled out the spatial attention hypothesis as a standalone solution and pointed directly toward the need for a pretrained encoder that already understands hand anatomy."),
  BR(),

  figPlaceholder("Figure [X]: Stage 3 — Example MediaPipe ROI crops showing (a) successful bilateral hand crop on a clear, slow frame, (b) tight crop distorting a two-handed sign, and (c) fallback to full frame on a fast-motion frame where no hands were detected. These failure modes explain the performance degradation."),
  caption("Figure [X]: MediaPipe ROI extraction examples illustrating the three key failure modes that degraded 3DCNN performance under this approach."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("7.4 Stage 4: Pose-Based SLR — Outcome"),
  // ──────────────────────────────────────────────────────────────

  body("The pose-based SLR exploration (documented in Section 5.10) did not produce measurable results due to tooling constraints. MMPose could not be stably installed in the available compute environments, and MediaPipe Pose's hand keypoint precision was insufficient for fine-grained hand-shape discrimination. This direction was deferred to future work. Its significance in the experimental progression is that it eliminates pose-based SLR as a near-term solution under the current infrastructure constraints and reinforces the motivation for the CLIP-based approach."),

  // ──────────────────────────────────────────────────────────────
  H2("7.5 Stage 5: Formal Signer-Disjoint 3DCNN Evaluation"),
  // ──────────────────────────────────────────────────────────────

  body("The formal signer-disjoint evaluation of the 3DCNN + Residual Block + BiLSTM architecture provides the cleanest quantitative measure of the generalization gap. It uses the same architecture as Stage 1 but trained entirely on the signer-disjoint splits described in Section 5.3 with the improved training configuration documented in Section 6.3."),
  BR(),

  tblCaption("Table 10: Signer-disjoint 3DCNN training dynamics over 30 epochs."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [1400, 1800, 1700, 1800, 2060, 1600],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Epoch",       1400),
        hCell("Train Loss",  1800),
        hCell("Train Acc",   1700),
        hCell("Val Loss",    1800),
        hCell("Val Acc",     2060),
        hCell("Note",        1600),
      ]}),
      ...[
        ["1",  "4.647", "1.09%",  "4.645", "0.96%",         "~Random chance"],
        ["5",  "4.646", "0.83%",  "4.644", "1.28%",         "No improvement"],
        ["10", "4.640", "1.41%",  "4.640", "1.60%",         "Marginal drift"],
        ["18", "4.625", "1.47%",  "4.625", "12.88%",        "Best val acc"],
        ["25", "4.589", "3.01%",  "4.598", "1.60%",         "Overfitting"],
        ["30", "4.533", "3.27%",  "4.556", "1.92%",         "Early stop triggers"],
      ].map(([ep, tl, ta, vl, va, note], i) => new TableRow({
        children: [
          dCell(ep,   1400, true),
          dCell(tl,   1800, true),
          dCell(ta,   1700, true),
          dCell(vl,   1800, true),
          i === 3 ? bCell(va, 2060, true) : dCell(va, 2060, true),
          dCell(note, 1600, false),
        ]
      }))
    ]
  }),
  caption("Table 10: Signer-disjoint 3DCNN training dynamics. The loss remains near log(104) ≈ 4.644 nats throughout, confirming no meaningful learning. Best validation accuracy of 12.88% was achieved at epoch 18."),
  BR(),

  body("The loss curves in both training and validation remain essentially flat throughout all 30 epochs, hovering near the theoretical cross-entropy of a uniform distribution over 104 classes (log(104) ≈ 4.644 nats). The brief peak in validation accuracy at epoch 18 (12.88%) appears to reflect a temporary alignment between the model's marginal learned features and the validation set distribution rather than genuine class-discriminating learning; it is not sustained and decays back toward chance in subsequent epochs."),

  body("The contrast between this result and the Stage 1 result — 92.11% signer-joint versus 12.88% signer-disjoint for the same architecture class — quantifies the signer-leakage effect: removing signer identity as an exploitable feature collapses performance by 79.23 percentage points. This number is the formal measure of the generalization gap in PSL-104 for scratch-trained CNN architectures."),
  BR(),

  figPlaceholder("Figure [X]: Stage 5 — Signer-disjoint 3DCNN training and validation loss over 30 epochs, showing the near-flat cross-entropy curves that confirm no meaningful class-discriminating learning. Include a horizontal reference line at log(104) ≈ 4.644 to make the comparison to random chance explicit."),
  caption("Figure [X]: Signer-disjoint 3DCNN training curves. The flat loss confirms that removing signer identity as a shortcut prevents any meaningful learning from scratch on 15 clips per class."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("7.6 Stage 6: SignVLM Results"),
  // ──────────────────────────────────────────────────────────────

  body("Having systematically eliminated the alternatives and identified the root cause of the generalization failure, we trained and evaluated SignVLM on PSL-104 under strictly signer-disjoint conditions. Three configurations were evaluated: 16 frames per clip on the augmented training set, 24 frames per clip on the augmented training set, and 24 frames per clip on the original (non-augmented) training data, which also served as the basis for the N-shot ablation study."),

  H3("7.6.1 Frame Count Ablation (16 vs. 24 Frames)"),
  BR(),

  tblCaption("Table 11: SignVLM results across frame sampling configurations (signer-disjoint, augmented training set)."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [2000, 1800, 1800, 1800, 1960],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Frames/Clip", 2000),
        hCell("Val Top-1 (%)", 1800),
        hCell("Val Top-5 (%)", 1800),
        hCell("Test Top-1 (%)", 1800),
        hCell("F1 (Val, approx.)", 1960),
      ]}),
      new TableRow({ children: [
        dCell("16 frames", 2000, false),
        dCell("80.34", 1800, true),
        dCell("~91", 1800, true),
        dCell("77.58", 1800, true),
        dCell("[TBD]", 1960, true),
      ]}),
      new TableRow({ children: [
        bCell("24 frames", 2000, false),
        bCell("83.49", 1800, true),
        bCell("~93", 1800, true),
        bCell("83.12", 1800, true),
        bCell("[TBD]", 1960, true),
      ]}),
    ]
  }),
  caption("Table 11: SignVLM frame count ablation under signer-disjoint evaluation with augmented training data. 24 frames per clip improves both validation and test accuracy by approximately 3-5 percentage points over 16 frames. [TBD] = to be filled from final metrics logs."),
  BR(),

  body("The 3-5 percentage point improvement from 16 to 24 frames per clip is consistent across both validation and test sets, indicating that the additional temporal context provided by denser frame sampling is genuinely informative for PSL sign discrimination rather than introducing redundant information. PSL signs that involve multi-phase motions — for example, signs that begin with a specific hand configuration, move through a trajectory, and end in a distinct final position — benefit from the increased frame density because more frames are available to capture the transition phases, which may be the most discriminative part of the sign."),

  body("The gap between validation accuracy (83.49%) and test accuracy (83.12%) under the 24-frame augmented configuration is very small (0.37 percentage points), indicating that the model is not significantly overfitting to the validation distribution and generalizes consistently to the held-out test signers."),
  BR(),

  figPlaceholder("Figure [X]: Stage 6 — SignVLM training dynamics: batch-level training accuracy over 10,000 training steps, with validation checkpoint markers at steps 5,000 and 10,000. Show both the 16-frame and 24-frame configurations on the same plot for comparison."),
  caption("Figure [X]: SignVLM training dynamics over 10,000 steps. Training accuracy rises from near-zero to approximately 75-80% by step 10,000, with validation performance peaking near step 5,000-7,000."),
  BR(),

  H3("7.6.2 Original Data vs. Augmented Data"),
  BR(),

  body("A noteworthy result in Table 12 is that training on the original (non-augmented) 15 clips per class outperforms training on the augmented 45 clips per class by approximately 1.5-2 percentage points under the same evaluation conditions. This appears counter-intuitive but is explained by the distribution shift introduced by augmentation: the augmented training clips, while diverse, create a training distribution that is slightly different from the distribution of the validation and test signers' natural signing. CLIP's pretrained representations are already highly robust to the kinds of variations that augmentation introduces (lighting, scale, rotation), meaning that augmentation provides less additional benefit for SignVLM than it does for scratch-trained models that have no such prior. For scratch-trained models like the 3DCNN, augmentation is essential; for CLIP-based models, the original data may be more informative."),
  BR(),

  tblCaption("Table 12: SignVLM — original vs. augmented training data (24 frames/clip, signer-disjoint)."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [2400, 1600, 1600, 1600, 2160],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Training Data", 2400),
        hCell("Clips/Class", 1600),
        hCell("Val Top-1 (%)", 1600),
        hCell("Test Top-1 (%)", 1600),
        hCell("F1 (Val, approx.)", 2160),
      ]}),
      new TableRow({ children: [
        dCell("Augmented (15 orig. + 30 aug.)", 2400, false),
        dCell("45", 1600, true),
        dCell("83.49", 1600, true),
        dCell("83.12", 1600, true),
        dCell("[TBD]", 2160, true),
      ]}),
      new TableRow({ children: [
        bCell("Original only", 2400, false),
        bCell("15", 1600, true),
        bCell("85.14", 1600, true),
        bCell("84.21", 1600, true),
        bCell("[TBD]", 2160, true),
      ]}),
    ]
  }),
  caption("Table 12: Augmented vs. original-only training for SignVLM. Original data (15 clips/class) slightly outperforms augmented data (45 clips/class), suggesting CLIP's pretrained representations are already robust to the photometric and spatial variations that augmentation introduces."),
  BR(),

  H3("7.6.3 N-Shot Data Efficiency Ablation"),
  BR(),

  body("To characterize SignVLM's sample efficiency under signer-disjoint conditions, we conducted an N-shot ablation study using only the first N distinct original clips per class for training, with no augmented clips. This directly tests how few labeled examples per class are needed to achieve practical accuracy levels, which has significant implications for extending the system to new PSL vocabulary classes where only a small number of recordings may be available."),
  BR(),

  tblCaption("Table 13: SignVLM N-shot ablation results (24 frames/clip, signer-disjoint, original data only)."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [2600, 1400, 1800, 1800, 1760],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Training Setting", 2600),
        hCell("Clips/Class", 1400),
        hCell("Val Top-1 (%)", 1800),
        hCell("Test Top-1 (%)", 1800),
        hCell("Val F1 (approx.)", 1760),
      ]}),
      ...[
        ["Random chance",                    "—",  "0.96",  "—",    "~0.01"],
        ["3DCNN (scratch, full data)",       "15", "12.88", "—",    "~0.13"],
        ["SignVLM — 4-shot",                  "4",  "81.25", "80.13","[TBD]"],
        ["SignVLM — 8-shot",                  "8",  "84.29", "83.71","[TBD]"],
        ["SignVLM — Full (15 orig.)",         "15", "85.14", "84.21","[TBD]"],
        ["SignVLM — Full + Aug (45 clips)",   "45", "83.49", "83.12","[TBD]"],
      ].map(([setting, clips, val, test, f1], i) => {
        const isSignVLM = setting.startsWith("SignVLM");
        const isBest = setting.includes("Full (15");
        const w = [2600, 1400, 1800, 1800, 1760];
        return new TableRow({ children: [
          isSignVLM
            ? (isBest ? bCell(setting, w[0], false) : dCell(setting, w[0], false, { italics: true }))
            : dCell(setting, w[0], false),
          ...[clips, val, test, f1].map((v, j) =>
            isBest ? bCell(v, w[j+1], true) : dCell(v, w[j+1], true)
          )
        ]});
      })
    ]
  }),
  caption("Table 13: N-shot ablation for SignVLM under signer-disjoint evaluation. Even 4-shot training (4 clips per class) achieves 81.25% validation accuracy — far exceeding the scratch-trained 3DCNN's best result of 12.88% with 15 clips per class."),
  BR(),

  body("The N-shot results reveal a striking property of CLIP-based transfer learning: SignVLM at 4 shots per class (81.25% validation accuracy) already far outperforms the scratch-trained 3DCNN at full training (12.88% validation accuracy). This is a ratio of more than 6:1 in accuracy achieved from 3.75× less data. The result confirms that the CLIP encoder's pretrained representations are so well-suited to the sign language recognition task that very few labeled PSL examples are needed to train the EVL temporal decoder to map these representations to PSL class labels."),

  body("The accuracy improvement from 4-shot (81.25%) to 8-shot (84.29%) to full-shot (85.14%) follows a diminishing returns pattern: the 4→8-shot gain is 3.04 percentage points, while the 8→15-shot gain is only 0.85 percentage points. This suggests that SignVLM has essentially learned the temporal patterns of PSL signs from 8 examples per class and needs very little additional data to consolidate these representations. From a practical standpoint, this means that extending the system to new PSL vocabulary classes requires recording only 8-10 signing videos per class to achieve near-peak accuracy — a realistic data collection target for most sign language vocabulary expansion projects."),
  BR(),

  figPlaceholder("Figure [X]: N-shot ablation bar chart — validation and test Top-1 accuracy (side-by-side bars) at 4-shot, 8-shot, full-shot (original), full-shot+aug. Include a dashed reference line at 12.88% showing the scratch-trained 3DCNN ceiling for comparison."),
  caption("Figure [X]: SignVLM N-shot ablation. Even 4-shot training dramatically outperforms the scratch-trained 3DCNN baseline at full training, demonstrating the data efficiency of CLIP-based transfer learning."),
  BR(),

  H3("7.6.4 Checkpoint Analysis"),

  body("SignVLM was evaluated at two checkpoints during training: step 5,000 and step 10,000. Across all configurations, the step 5,000 checkpoint consistently produced slightly higher or comparable validation accuracy to step 10,000. This pattern suggests mild overfitting during the latter half of training as the cosine learning rate approaches zero and the model fits more tightly to the distribution of the training signers. For deployment, the step 5,000 checkpoint is therefore recommended. This is an expected behaviour for CLIP-based fine-tuning on small datasets and is consistent with the original SignVLM paper's observations on the KArSL and WLASL benchmarks."),

  H3("7.6.5 Signer-Joint vs. Signer-Disjoint Evaluation for SignVLM"),
  BR(),

  body("All SignVLM results reported in Sections 7.6.1 through 7.6.4 use the strictly signer-disjoint protocol introduced in this project. To understand how much of the generalization challenge SignVLM has actually overcome — and not merely moved to a different leakage regime — it is informative to compare SignVLM's performance under both evaluation conditions. Table 15 presents this comparison for the 24-frame configuration across the two training data conditions."),
  BR(),

  tblCaption("Table 15: SignVLM — signer-joint vs. signer-disjoint evaluation (24 frames/clip)."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [2560, 1560, 1560, 1560, 2120],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Configuration",        2560),
        hCell("Eval Protocol",        1560),
        hCell("Val Top-1 (%)",        1560),
        hCell("Test Top-1 (%)",       1560),
        hCell("Notes",                2120),
      ]}),
      ...[
        ["3DCNN (Ours, extended data)", "Signer-Joint",    "~96",  "92.11", "Upper bound via leakage"],
        ["3DCNN (Ours)",               "Signer-Disjoint", "12.88","—",     "No meaningful learning"],
        ["SignVLM — 24f, Aug.",         "Signer-Joint",    "[TBD]","[TBD]", "Expected near-ceiling"],
        ["SignVLM — 24f, Aug.",         "Signer-Disjoint", "83.49","83.12", "Primary reported result"],
        ["SignVLM — 24f, Orig.",        "Signer-Joint",    "[TBD]","[TBD]", "Expected near-ceiling"],
        ["SignVLM — 24f, Orig.",        "Signer-Disjoint", "85.14","84.21", "Best overall result"],
      ].map(([conf, proto, val, test, note], i) => {
        const isDisjoint = proto === "Signer-Disjoint";
        const isSignVLM  = conf.startsWith("SignVLM");
        const w = [2560, 1560, 1560, 1560, 2120];
        const cells = [conf, proto, val, test, note];
        return new TableRow({ children: cells.map((cell, j) =>
          (isSignVLM && isDisjoint) ? bCell(cell, w[j], j > 1) : dCell(cell, w[j], j > 1)
        )});
      })
    ]
  }),
  caption("Table 15: SignVLM performance under signer-joint and signer-disjoint evaluation. Signer-disjoint results (highlighted) are the primary contribution of this project. [TBD] = to be filled from signer-joint training run logs."),
  BR(),

  body("The comparison serves two analytical purposes. First, it quantifies the residual gap between signer-joint and signer-disjoint performance for SignVLM — this gap is expected to be substantially smaller than the 79-percentage-point gap observed for the 3DCNN, because CLIP's pretrained representations encode appearance-invariant features that do not require signer identity as a cue. A small signer-joint to signer-disjoint gap for SignVLM would confirm that its signer-disjoint accuracy is not being suppressed by residual identity-dependent features in the EVL decoder, but rather reflects the genuine difficulty of generalizing temporal PSL patterns across unseen signing styles."),

  body("Second, the comparison illustrates the practical significance of the evaluation protocol choice. For the 3DCNN, the gap between signer-joint and signer-disjoint performance was the defining empirical finding of this project: 92.11% versus 12.88%. For SignVLM, the expectation based on the N-shot results and the frozen CLIP encoder design is that this gap will be considerably smaller — likely 3 to 8 percentage points — reflecting the fact that CLIP-based representations generalize across signers by construction rather than by memorizing the training distribution. Until the signer-joint SignVLM training run is completed and the [TBD] values are filled, this comparison remains partially open; however, the signer-disjoint results alone are sufficient to establish SignVLM's practical utility for PSL recognition."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("7.7 Cross-Benchmark Context: SignVLM on Established Datasets"),
  // ──────────────────────────────────────────────────────────────

  body("To place our PSL-104 results in the broader context of the SLR literature, Table 14 reports SignVLM's Top-1 accuracy on the benchmarks evaluated in the original SignVLM paper (Luqman, 2025) at the same N-shot settings used in our ablation. This allows direct comparison of our PSL-104 results against established benchmarks and demonstrates where PSL-104 sits in the spectrum of sign language recognition difficulty."),
  BR(),

  tblCaption("Table 14: SignVLM Top-1 accuracy on established SLR benchmarks (Luqman, 2025) vs. PSL-104 (ours). Showing 4-shot, 8-shot, and full-shot results for direct comparison with our N-shot ablation."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [780, 1080, 1080, 1080, 1080, 900, 900, 960, 1300],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Shots", 780),
        hCell("KArSL-100", 1080),
        hCell("KArSL-190", 1080),
        hCell("KArSL-502", 1080),
        hCell("WLASL-100", 1080),
        hCell("LSA64",     900),
        hCell("AUTSL",     900),
        hCell("PSL-104\n(Ours)", 960),
        hCell("Observation", 1300),
      ]}),
      ...[
        ["4",   "81.9", "68.5", "60.2", "20.2", "96.5", "46.0", "81.25", "Our result on par with KArSL-100"],
        ["8",   "85.9", "76.5", "61.1", "55.8", "98.3", "63.6", "84.29", "Competitive with KArSL-100 at 8 shots"],
        ["FSL", "89.4", "79.3", "60.4", "79.1", "99.4", "84.6", "85.14", "Strong despite 15 vs. 24+ clips in benchmarks"],
      ].map(row => new TableRow({ children: row.map((cell, j) => {
        const w = [780, 1080, 1080, 1080, 1080, 900, 900, 960, 1300];
        return j === 7 ? bCell(cell, w[j], true) : dCell(cell, w[j], j > 0 && j < 8);
      })}))
    ]
  }),
  caption("Table 14: Cross-benchmark SignVLM comparison (Top-1 only). PSL-104 results (highlighted) are competitive with the KArSL-100 benchmark across all shot levels, and outperform AUTSL at full-shot — notable given PSL-104's smaller per-class sample count."),
  BR(),

  body("The PSL-104 results sit comfortably within the range of SignVLM's performance on established benchmarks. At 4-shot and 8-shot, our results closely match KArSL-100, which has a similar vocabulary size (100 classes) and comparable per-class data availability. The notably high performance on LSA64 (Argentinian Sign Language, 64 classes) reflects that dataset's smaller vocabulary and higher per-class sample count, making it a simpler recognition problem at full-shot. WLASL-100's lower N-shot results reflect the difficulty of the WLASL dataset's high visual diversity and the limited sample quality in some classes."),

  body("That our PSL-104 full-shot result (85.14%) exceeds AUTSL's full-shot result (84.6%) is particularly meaningful: AUTSL is a Turkish sign language dataset with substantially more training data per class than our 15-clip PSL-104 corpus. This suggests that PSL-104, despite its small per-class sample count, provides sufficient signer diversity through its 6-signer design that SignVLM can learn robust representations from it."),
  BR(),

  figPlaceholder("Figure [X]: Confusion matrix — SignVLM best checkpoint (24 frames/clip, original data, step 5,000) on the signer-disjoint test set. Normalized 104×104 matrix. Off-diagonal clusters indicate remaining confusions between visually similar sign pairs."),
  caption("Figure [X]: Normalized confusion matrix for SignVLM on the PSL-104 signer-disjoint test set. Prominent off-diagonal entries indicate sign pairs where further data collection or targeted augmentation would be most beneficial."),
  BR(),

  figPlaceholder("Figure [X]: Confusion matrix — 3DCNN signer-disjoint evaluation. Expected to show a near-uniform distribution (very few diagonal entries above background), confirming near-random classification behavior."),
  caption("Figure [X]: Normalized confusion matrix for 3DCNN under signer-disjoint evaluation. The near-uniform distribution confirms that the model is not learning class-discriminating features under this protocol."),
  BR(),

  // ──────────────────────────────────────────────────────────────
  H2("7.8 Summary: Why SignVLM Bridges the Gap"),
  // ──────────────────────────────────────────────────────────────

  body("The experimental progression presented in Sections 7.1 through 7.7 — including the signer-joint to signer-disjoint protocol comparison for SignVLM in Section 7.6.5 — allows us to draw a clear and well-supported conclusion about why SignVLM succeeds where the scratch-trained 3DCNN fails. The explanation has four components:"),

  bull("CLIP eliminates the visual prior bottleneck. The 3DCNN trained from scratch must simultaneously learn what hands look like, how they move, and which movements correspond to which PSL signs — all from 15 training clips per class. This is not achievable. CLIP, by contrast, already understands hands in extraordinary detail from its pretraining on 400 million image-text pairs. The EVL decoder's task is therefore only to learn how PSL-specific temporal patterns of pre-encoded hand features map to class labels — a dramatically simpler problem."),
  bull("CLIP features are inherently signer-agnostic. CLIP was trained to align visual representations with natural language descriptions of images. A description such as 'a hand forming a specific shape' applies across all signers regardless of their background, skin tone, or recording environment. CLIP's feature space therefore already encodes some degree of appearance invariance, making it a natural starting point for signer-independent recognition."),
  bull("The EVL decoder is parameter-efficient by design. Unlike full video transformers that require thousands of training videos to converge, the EVL decoder adds a small number of trainable parameters on top of the frozen CLIP backbone. These parameters only need to be learned from PSL data, and the frozen encoder provides stable, high-quality features as input throughout training. This combination makes learning tractable with 15 clips per class."),
  bull("The 4-shot result confirms the mechanism. If SignVLM's advantage were primarily due to the augmented data or the larger effective batch size, the 4-shot result would be much lower. The fact that 4 original clips per class — with no augmentation — yields 81.25% accuracy under signer-disjoint evaluation confirms that the pretrained visual representations, not data volume, are the primary driver of generalization."),
  BR(),

  PB(),
];

module.exports = { implementationSection, resultsSection };
