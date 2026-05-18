// ================================================================
// PART 3: Section 5 — Methodology / System Design
// The largest section: dataset two-cohort story, preprocessing,
// augmentation pipeline comparison, model architectures, and
// the full 6-stage experimental progression.
// ================================================================

const {
  Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType, PageBreak,
  LevelFormat
} = require('docx');

const {
  H1, H2, H3, BR, PB, body, bodyRuns, bull, bull2, num, caption,
  figPlaceholder, tblCaption, hCell, dCell, bCell, grpCell, eq,
  r, rb, ri, rbi,
  CONTENT, BORDERS, NAVY, BLUE, LGRAY, HBLUE, DBLUE, BORDERS_NONE,
  AlignmentType: AT
} = require('./part1');

// ── helper: italic + bold label inline ───────────────────────
const bodyLabel = (label, text) => bodyRuns([
  rbi(label + ": "),
  r(text),
]);

// ================================================================
const methodologySection = [

  H1("5. Methodology / System Design"),

  body("This section describes the complete methodology followed in this project, from the construction and preprocessing of the PSL-104 dataset through the design and evaluation of all model architectures. The methodology is presented in the order in which the work was actually carried out: dataset construction (Section 5.1), the data preprocessing pipeline (Section 5.2), the augmentation strategy and how it compares to the prior team's approach (Section 5.3), the signer-disjoint splitting protocol (Section 5.4), and then the full experimental progression that led from a well-performing signer-joint baseline through a series of failures and investigations to the final SignVLM solution (Sections 5.5 through 5.10). Presenting the methodology in this narrative order rather than in idealized retrospective order is deliberate — each stage of the experimental progression informed the next, and the reasoning behind architectural choices cannot be fully understood without first understanding what had been tried and what had failed."),

  // ─────────────────────────────────────────────────────────────
  H2("5.1 Dataset Construction: PSL-104 (A Two-Cohort Story)"),
  // ─────────────────────────────────────────────────────────────

  body("The PSL-104 dataset was not built by a single team in a single semester. It is the result of work carried out across two consecutive FYP cohorts, and understanding its construction requires understanding what each team contributed and why."),

  H3("5.1.1 Phase 1: The Prior FYP Cohort's Contribution (Agha et al., 2025)"),

  body("The prior FYP cohort — Agha Fardeen, Virkha Kumari, and Tania Saleh — established the foundational corpus that our work builds on. Their first step was to obtain the official PSL reference material: they scraped the PSL Dictionary hosted at psl.org.pk, which is maintained by the Pakistan Sign Language organization and constitutes the only publicly available, authority-endorsed PSL video resource. The dictionary provides one to two short video clips per sign, each demonstrating the sign performed by a single professional signer in a controlled, neutral studio environment. These studio recordings served as the authoritative reference for correct sign form across all 104 classes."),

  body("Recognizing that one or two clips from a single signer is entirely insufficient for training any deep learning model — let alone evaluating its generalization — the prior team additionally recorded their own videos. Each of the three team members recorded three videos per sign label across all 104 classes, contributing nine original team-recorded clips per sign. Combined with the PSL Dictionary clips, this brought the total original video count to approximately ten to eleven clips per sign. All team-recorded videos were performed in front of a neutral background at a standardized recording distance, and the sign content was verified against the PSL Dictionary reference videos for correctness."),

  body("The prior team also designed and implemented the first augmentation pipeline for the dataset, expanding the ten original clips per sign to sixty augmented clips per sign, and then split the augmented dataset randomly into training (80%), validation (15%), and test (5%) partitions. This yielded 4,992 training examples, 832 validation examples, and 416 test examples across 104 classes. The prior team then trained three deep learning architectures on this dataset — ViViT, ResNet-18 + LSTM, and their proposed 3DCNN + Residual Block + BiLSTM — with the 3DCNN architecture achieving the best generalization at 83.89% test accuracy."),

  body("This prior work was genuinely substantial: it established the vocabulary, produced the first multi-clip PSL video corpus, designed the augmentation pipeline, and demonstrated that deep learning models could achieve strong performance on PSL under the evaluation conditions used. The critical limitation — one that the prior team correctly acknowledged — was that the evaluation used random splitting across a 3-signer dataset, meaning the same signers' videos appeared in both training and test sets. This signer-inclusive evaluation is the industry-standard practice in small-dataset SLR work and does not imply any error or oversight by the prior team. It is, however, the specific limitation that our FYP-II work sets out to address."),

  H3("5.1.2 Phase 2: Our FYP-II Extension"),

  body("Our FYP-II team's first major contribution was extending the signer diversity of the dataset from three to six distinct individuals. We recruited three new signers, none of whom had appeared in the prior team's recordings, and asked each new signer to record three videos per sign across all 104 classes. The recording protocol was kept consistent with the prior team's approach — each signer performed the sign in full view of the camera with a clear start and end, at a distance of approximately two meters from the lens — while deliberately varying the recording environment for each signer. Specifically, each of the three new signers recorded across different rooms, backgrounds, and lighting conditions (natural daylight, fluorescent indoor lighting, and mixed ambient lighting), ensuring that the combined dataset would contain genuine environmental diversity rather than a single controlled studio setting."),

  body("Adding three signers contributed an additional nine original videos per sign, bringing the combined total to eighteen to nineteen original clips per sign across six distinct signers. This is the first PSL video dataset with sufficient signer diversity to support a strictly signer-disjoint train-validation-test split, which is the foundational requirement for evaluating whether a model can genuinely generalize to new individuals."),

  body("Our second major data contribution was re-scraping and reprocessing the PSL Dictionary. While the prior team had also obtained PSL Dictionary videos, our team independently re-downloaded the original .mov source files to ensure we had the highest-quality originals for the preprocessing pipeline described in Section 5.2. The complete PSL-104 dataset — combining Phase 1 and Phase 2 recordings with the PSL Dictionary videos — therefore contains videos from six team members, one professional studio signer, and a variety of recording environments, representing substantially more real-world diversity than any prior PSL dataset."),

  body("Table 2 summarizes the composition of the combined PSL-104 dataset before augmentation, breaking down the contribution of each phase to the total original clip count."),

  BR(),
  tblCaption("Table 2: PSL-104 dataset composition before augmentation. Each count is per-sign; totals are across all 104 classes."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [2800, 1700, 1700, 1700, 1460],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Source",             2800),
        hCell("Signers",            1700),
        hCell("Videos/Sign",        1700),
        hCell("Total Clips",        1700),
        hCell("Environment",        1460),
      ]}),
      new TableRow({ children: [
        dCell("PSL Dictionary (psl.org.pk)",           2800),
        dCell("1 (professional)",                       1700, true),
        dCell("1–2",                                    1700, true),
        dCell("~104–208",                               1700, true),
        dCell("Controlled studio",                      1460),
      ]}),
      new TableRow({ children: [
        dCell("Prior FYP cohort (Agha et al., 2025)",  2800),
        dCell("3 team members",                         1700, true),
        dCell("3 each → 9 total",                       1700, true),
        dCell("~936",                                   1700, true),
        dCell("Neutral background",                     1460),
      ]}),
      new TableRow({ children: [
        dCell("Our FYP-II team (new signers)",          2800),
        dCell("3 new signers",                          1700, true),
        dCell("3 each → 9 total",                       1700, true),
        dCell("~936",                                   1700, true),
        dCell("Varied (3 environments × 3 signers)",   1460),
      ]}),
      new TableRow({ children: [
        bCell("PSL-104 Total (combined)",               2800, false),
        bCell("6 + 1 studio",                           1700, true),
        bCell("~18–19/sign",                            1700, true),
        bCell("~1,872",                                 1700, true),
        bCell("Diverse",                                1460, true),
      ]}),
    ]
  }),
  BR(),

  // ─────────────────────────────────────────────────────────────
  H2("5.2 Data Preprocessing Pipeline"),
  // ─────────────────────────────────────────────────────────────

  body("The raw videos collected across both FYP cohorts required substantial preprocessing before they could be used for model training. The prior team's recordings were already in .mp4 format and relatively clean, but the PSL Dictionary source files and a portion of the new recordings came in .mov format with audio tracks, variable frame rates, and significant amounts of temporal noise at the start and end of each clip. A consistent, high-quality preprocessing pipeline was essential not only for model performance but for the validity of signer-disjoint splitting — inconsistent clip lengths or presence of pre-sign and post-sign blank frames could mislead temporal models and introduce confounds between signers based on their recording habits rather than their signing style."),

  H3("5.2.1 Format Standardization and Audio Removal"),

  body("All raw video files were first converted to a consistent .mp4 format using FFmpeg. The PSL Dictionary .mov files in particular required this conversion step, as the .mov container format is not universally supported by Python video processing libraries and its internal compression (ProRes or H.264 with AAC audio) is less efficient for deep learning data pipelines. Conversion to .mp4 with H.264 video encoding standardized the container format across all clips without any loss of visual quality."),

  body("Audio tracks were stripped from all videos during the conversion step. Audio is entirely irrelevant to sign language recognition from video, as the recognition task depends purely on visual information — hand configurations, body posture, and motion. Retaining audio would add unnecessary file size to the dataset and could in principle introduce unintended correlations between audio artifacts and class labels in certain model architectures that process raw video files. Removing audio from all clips at the source ensures the dataset is fully visual."),

  body("Label folder organization was also verified and corrected at this stage. The PSL Dictionary organizes signs using Urdu-script folder names, which introduced a significant technical complication: OpenCV's cv2.VideoCapture() function silently fails on file paths containing Unicode characters (including Arabic and Urdu script), returning an empty capture object without raising any exception. This means that all Urdu-script class videos would be silently skipped during any training pipeline that uses OpenCV for video loading — a silent data loss failure that would be very difficult to detect without explicit per-class video count verification. We resolved this by replacing all OpenCV-based video reading with PyAV (Python bindings for the FFmpeg library), which handles Unicode file paths correctly on both Windows and Linux. All split file generation scripts also use Python's pathlib.Path rather than the os module for the same reason."),

  H3("5.2.2 Temporal Cropping"),

  body("Temporal cropping is the preprocessing step that had the single largest impact on data quality and model training stability. In every raw video recording — whether from the PSL Dictionary or from team-recorded clips — the signing motion does not begin immediately at frame zero and does not end at the last frame. Instead, there is typically a preparatory phase at the start where the signer is positioning their hands and settling into the signing posture, and a wind-down phase at the end where the signer lowers their hands back to a resting position. These pre-sign and post-sign frames are visually uninformative: they show neutral hand positions that are identical across many different signs and many different signers."),

  body("The presence of these blank transition frames is particularly damaging for temporal models. A 3D-CNN or LSTM processing a clip that spends the first 20% of its frames on blank hand positioning and the last 15% on the hands returning to rest will necessarily attend to and partially model this non-informative content. Because the transition postures are similar across many signs, this reduces the discriminative signal available to the model and adds noise to the temporal feature representations. In worst cases, a model might learn to key on the timing of transition frames relative to the signing window as a spurious feature, which would generalize poorly across signers who have different preparation speeds."),

  body("We addressed this by manually reviewing and temporally cropping every video in the dataset to a window of three to four seconds containing only the active signing motion. The cropping was performed frame-by-frame for each clip, with the crop boundaries set to the moment the signer's hands first begin to form the sign and the moment they complete the final position of the sign. This was carried out semi-automatically: a custom script identified candidate crop boundaries based on motion energy thresholding (frames where pixel-level change exceeded a threshold), and a team member then reviewed and corrected the boundaries for each clip. The final cropped clips range from approximately 90 to 120 frames at 30 fps, capturing the complete signing motion with minimal pre- and post-sign content."),

  body("This temporal cleaning step is not present in the prior team's pipeline — they used the full raw video files as-is, relying on the model to learn to ignore non-informative frames. Our contribution of temporally cropping the entire dataset therefore represents a meaningful improvement in data quality that benefits all model architectures trained on PSL-104, not just the specific architectures we evaluate in this project."),

  H3("5.2.3 Frame Pre-Extraction for Training Efficiency"),

  body("The final preprocessing step — offline frame pre-extraction — was motivated by a severe training bottleneck encountered during early SignVLM training experiments. During initial runs, profiling revealed that the data loading time per batch (data_time) was dominating total iteration time by a factor of five to ten, meaning the GPU was idle for the vast majority of each training step while the CPU decoded video frames on-the-fly. The bottleneck was H.264 video decoding: even using PyAV with multiple DataLoader worker processes, the CPU overhead of seeking to random frame positions within compressed video files and decompressing them in real time made each data loading step take approximately 0.8 seconds per batch — far exceeding the 0.4-second GPU forward and backward pass time."),

  body("The solution was to pre-extract all video frames to individual JPEG files on disk before training begins, so that the data loader reads pre-decompressed image files rather than decoding video at training time. JPEG reads are random-access I/O operations on small files, which are approximately sixteen times faster than the equivalent H.264 video seek-and-decode operation at our video resolution. After implementing frame pre-extraction, data_time dropped from approximately 0.8 seconds per batch to approximately 0.05 seconds per batch, GPU utilization increased from roughly 30% to 85-90%, and total iteration time fell from approximately 5 seconds per step to approximately 1.2 seconds per step."),

  body("The pre-extracted frames were organized as label_folder/video_name/frame_XXXXXX.jpg, with frame indices zero-padded to six digits. This structure matches the expected input format of SignVLM's data loader when configured in frames_available mode. All 1,976 original videos across all three dataset splits were pre-extracted, producing a total of approximately 180,000 to 240,000 JPEG frame files across the dataset. The frame extraction script also resolved the Unicode path issue described above by using PyAV rather than OpenCV, ensuring that Urdu-script class directories were processed correctly."),

  // ─────────────────────────────────────────────────────────────
  H2("5.3 Signer-Disjoint Data Splitting"),
  // ─────────────────────────────────────────────────────────────

  body("The construction of training, validation, and test splits is the methodological decision with the most significant impact on the validity of the reported results. In prior PSL SLR work — including the prior FYP cohort's study — this splitting was performed randomly across the entire augmented dataset, which means that videos from the same signer could and did appear in both the training set and the test set. As demonstrated in Section 2 and quantified in our experimental results (Section 7), this signer-inclusive splitting procedure allows models to learn signer-identity cues and use them as shortcuts to achieve high test accuracy without genuinely learning sign-discriminating features."),

  body("Our splitting procedure enforces strict signer disjointness across all three partitions. The six signers in PSL-104 were assigned exclusively to one of the three splits, such that no signer's videos appear in more than one split. The formal constraint satisfied by our splits is:"),

  eq("S_train ∩ S_val = ∅,    S_train ∩ S_test = ∅,    S_val ∩ S_test = ∅"),

  body("where S_train, S_val, and S_test denote the sets of signers whose videos appear in the training, validation, and test splits respectively. This mirrors the Leave-One-Signer-Out (LOSO) evaluation protocol that is recommended as the minimum requirement for meaningful SLR results in the broader sign language recognition literature [12, 16, 17]."),

  body("Under this assignment, the training split contains recordings from four of the six signers (contributing fifteen original videos per class), the validation split contains recordings from one signer (contributing three original videos per class), and the test split contains recordings from the remaining signer (contributing one video per class plus PSL Dictionary clips). The PSL Dictionary videos — recorded by the professional studio signer — are allocated to the test split only, as they represent the most controlled and standardized recordings and serve as a meaningful held-out evaluation set. This allocation is consistent across all experiments in the project."),

  body("Split files were generated programmatically by a dedicated script (prepare_psl_splits.py) that maps each absolute video path to its integer class label, with signer assignment encoded by directory structure. These split files are then consumed directly by both the 3DCNN and SignVLM data loaders, ensuring consistency across all training runs. The split assignments are fixed across all experiments — we do not re-split between runs, ensuring that every model is evaluated on exactly the same held-out signer videos."),

  body("It is important to note that augmentation was applied separately to each split after the signer-disjoint splitting was performed — not before. Augmenting before splitting would allow augmented variants of a training signer's video to appear in the validation or test set, which would partially defeat the purpose of signer-disjoint evaluation by leaking appearance information across splits. Applying augmentation within each split after assignment ensures strict separation. The augmentation applied to validation and test splits is used to produce statistically richer evaluation pools for per-class accuracy estimation and confusion matrix analysis; it does not expose the model to any additional training information."),

  // ─────────────────────────────────────────────────────────────
  H2("5.4 Video Augmentation Pipeline: Prior Approach vs. Our Redesign"),
  // ─────────────────────────────────────────────────────────────

  body("Data augmentation is essential for any deep learning system trained on a small corpus. With only fifteen original training videos per class in our signer-disjoint training split, a model trained without augmentation would massively overfit — memorizing the specific appearance of the four training signers rather than learning generalizable sign features. Augmentation artificially increases the apparent diversity of the training set by applying random transformations to existing clips, forcing the model to become invariant to the specific variations introduced."),

  body("The prior FYP cohort designed and implemented an augmentation pipeline that was spatial-only: it applied eleven different image-level transformations (rotation, scaling, brightness adjustment, Gaussian noise, and similar operations) frame-by-frame to individual video frames. This produced six augmented variants per original video, expanding the per-sign clip count from ten to sixty. While this approach effectively increased frame-level visual diversity and was appropriate for the ViViT, ResNet+LSTM, and 3DCNN architectures the prior team evaluated, it had a structural limitation: by treating each frame independently, it did not introduce any temporal diversity into the augmented clips. Every augmented clip had the same number of frames in the same temporal order as the original, with only per-frame pixel variations applied. The temporal structure — the timing, pacing, and sequencing of the signing motion — was identical across all augmented variants of a given original clip."),

  body("This matters because temporal variation is a real and significant source of inter-signer variation in the real world. Different signers perform the same sign at different speeds, with different amounts of hesitation or momentum, and with slightly different timing of individual sub-movements within the sign. A model that has only seen augmented versions of a sign with identical temporal structure may be poorly calibrated for this real-world temporal variability."),

  body("Our augmentation pipeline redesign addresses this limitation by adding temporal augmentations alongside the existing spatial and photometric transforms. The complete pipeline applies one to two randomly selected transforms per video, drawn from three categories:"),

  H3("5.4.1 Spatial Augmentations (retained from prior pipeline, parameters updated)"),

  bull("Lower-body crop (crop_ratio = 0.2): Removes the bottom 20% of the frame area, keeping the focus on the upper body, hands, and face. This simulates slight differences in camera framing and signer height relative to the camera."),
  bull("Scale zoom in/out (factor ∈ {0.8, 1.2}): Applies a random zoom factor, simulating variation in the distance between the signer and the camera. Zoom-in crops to a central region while zoom-out adds border padding."),
  bull("Rotation (angle ~ Uniform(−10°, +10°)): Applies a small random rotation, simulating slight camera tilt or variation in the signer's body lean. Rotation is kept within ±10° to preserve sign legibility."),

  H3("5.4.2 Photometric Augmentations (retained from prior pipeline)"),

  bull("Brightness adjustment (factor ~ Uniform(0.7, 1.4)): Multiplies pixel brightness by a random factor, simulating variation in ambient light intensity between different recording environments."),
  bull("Brightness-Saturation-Hue (BSH) transform: Adjusts HSV color channels independently, simulating differences in camera white balance, sensor response across different devices, and variation in skin tone representation under different lighting."),
  bull("Gaussian noise injection: Adds pixel-level noise drawn from a Gaussian distribution, simulating camera sensor noise particularly visible in lower-light recordings."),

  H3("5.4.3 Temporal Augmentations (new in our pipeline)"),

  bull("Frame jitter (temporal offset ~ Uniform(1, 3) frames): Applies a small random temporal shift to the start of the sampled frame sequence, simulating minor variation in exactly when within the clip the active signing motion begins. This increases robustness to imprecise temporal cropping boundaries."),
  bull("Random frame drop (drop rate ~ Uniform(0.05, 0.15)): Randomly removes up to 15% of frames from the clip and resamples the remaining frames to maintain the target clip length. This simulates dropped frames in lower-quality video feeds and, more importantly, introduces variation in the effective playback speed of the sign — compressing or stretching the temporal structure of the signing motion within the clip window."),

  body("Table 3 provides a direct comparison of the prior team's augmentation pipeline and our redesigned pipeline, highlighting the additions and the effect on dataset statistics."),

  BR(),
  tblCaption("Table 3: Comparison of augmentation pipeline: prior FYP-I approach vs. our FYP-II redesign."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [2200, 1580, 1580, 2000, 2000],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Transform",                2200),
        hCell("FYP-I Pipeline",           1580),
        hCell("FYP-II Pipeline",          1580),
        hCell("Purpose",                  2000),
        hCell("Diversity Introduced",     2000),
      ]}),
      // Spatial group
      new TableRow({ children: [
        grpCell("Spatial Transforms", 2200, 1),
        grpCell("", 1580),
        grpCell("", 1580),
        grpCell("", 2000),
        grpCell("", 2000),
      ]}),
      ...[
        ["Lower-body crop",    "Yes",     "Yes (refined)", "Focus on hands/torso",           "Camera framing variation"],
        ["Scale zoom in/out",  "Yes",     "Yes",           "Signer distance variation",       "Camera proximity"],
        ["Rotation (±10°)",    "Yes",     "Yes",           "Camera tilt / body lean",         "Orientation robustness"],
      ].map(row => new TableRow({ children: row.map((c, i) => dCell(c, [2200,1580,1580,2000,2000][i], false)) })),
      // Photometric group
      new TableRow({ children: [
        grpCell("Photometric Transforms", 2200, 1),
        grpCell("", 1580),
        grpCell("", 1580),
        grpCell("", 2000),
        grpCell("", 2000),
      ]}),
      ...[
        ["Brightness adjust",  "Yes",     "Yes",           "Lighting intensity",              "Indoor/outdoor lighting"],
        ["BSH (Hue/Sat/Bri)",  "Yes",     "Yes",           "Camera/white balance variation",  "Device and skin tone diversity"],
        ["Gaussian noise",     "Yes",     "Yes",           "Sensor noise simulation",         "Low-light robustness"],
      ].map(row => new TableRow({ children: row.map((c, i) => dCell(c, [2200,1580,1580,2000,2000][i], false)) })),
      // Temporal group — NEW
      new TableRow({ children: [
        grpCell("Temporal Transforms (NEW in FYP-II)", 2200, 1),
        grpCell("", 1580),
        grpCell("", 1580),
        grpCell("", 2000),
        grpCell("", 2000),
      ]}),
      ...[
        ["Frame jitter (1–3 frames)",  "No",  "Yes",  "Clip start boundary variation",    "Sign onset timing robustness"],
        ["Random frame drop (5–15%)",  "No",  "Yes",  "Effective speed variation",         "Signing pace / speed diversity"],
      ].map(row => new TableRow({ children: row.map((c, i) => {
        const w = [2200,1580,1580,2000,2000][i];
        if (i === 1 && row[1] === "No") return dCell(row[i], w, false, { color: "999999" });
        if (i === 2 && row[2] === "Yes" && row[1] === "No") return dCell(row[i], w, false, { bold: true, color: "1F6E38" });
        return dCell(c, w, false);
      }) })),
      // Summary row
      new TableRow({ children: [
        bCell("Net effect on dataset",    2200, false),
        dCell("Spatial diversity only;\n10 → 60 clips/sign",  1580, false),
        bCell("Spatial + temporal diversity;\n15 → 45 clips/sign (training split)", 1580, false),
        bCell("Broader distribution coverage", 2000, false),
        bCell("More realistic inter-signer variation", 2000, false),
      ]}),
    ]
  }),
  BR(),

  body("The net effect of the temporal augmentations is meaningful even though the raw clip counts after augmentation are similar: the training split expands from 1,560 original clips to 4,680 augmented clips (thirty augmented variants per class), compared to the prior team's sixty variants from ten originals. More importantly, the distribution of the augmented training data now covers temporal variation — signing pace, clip boundary sensitivity — in addition to spatial and photometric variation, producing a more realistic approximation of the variability a deployed model would encounter across new signers. Table 4 summarizes the final dataset statistics after augmentation."),

  BR(),
  tblCaption("Table 4: PSL-104 dataset statistics after signer-disjoint splitting and augmentation."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [1800, 1200, 1680, 1680, 1200, 1800],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("Split",         1800),
        hCell("Classes",       1200),
        hCell("Original Clips",1680),
        hCell("After Aug.",    1680),
        hCell("Avg./Class",    1200),
        hCell("Signers",       1800),
      ]}),
      ...[
        ["Training",    "104",  "1,560",  "4,680",  "45",  "4 signers"],
        ["Validation",  "104",  "312",    "1,248",  "12",  "1 signer"],
        ["Test",        "104",  "104",    "1,040",  "10",  "1 signer + PSL Dict."],
      ].map(row => new TableRow({ children: row.map((c, i) => dCell(c, [1800,1200,1680,1680,1200,1800][i], true)) })),
      new TableRow({ children: [
        bCell("Total", 1800),
        bCell("104", 1200),
        bCell("1,976", 1680),
        bCell("6,968", 1680),
        bCell("—", 1200),
        bCell("6 total (disjoint)", 1800),
      ]}),
    ]
  }),
  BR(),

  // ─────────────────────────────────────────────────────────────
  H2("5.5 Model Architectures"),
  // ─────────────────────────────────────────────────────────────

  body("Two model architectures were used throughout the experimental progression: the 3DCNN + Residual Block + BiLSTM architecture inherited and extended from the prior FYP cohort, and SignVLM, a large pretrained visual-temporal model that we adapted to PSL-104. This section describes both architectures in detail."),

  H3("5.5.1 Baseline: 3DCNN + Residual Block + BiLSTM"),

  body("The 3DCNN + Residual Block + BiLSTM architecture was originally proposed by the prior FYP cohort and forms our controlled baseline for all comparison experiments. The architecture is designed to jointly extract spatiotemporal features from video clips and model the temporal dependencies across the frame sequence, without any reliance on pretrained weights. Its design was inspired by two independent prior works: the residual 3D CNN framework of Chen et al. (2023) [DA-R3DCNN], which demonstrated that adding residual connections to 3D convolutional backbones improves gradient flow and enables deeper spatiotemporal feature learning; and the C3D-BiLSTM architecture with Multi-Head Attention proposed by Dey et al. (2024) for Wh-question sign recognition in ASL, which showed that bidirectional LSTMs more effectively capture signs where the temporal direction of the motion carries meaning."),

  body("The architecture processes video clips as five-dimensional tensors of shape (Batch, Channels, Time, Height, Width). It consists of three sequential stages:"),

  bodyLabel("Stage 1 — Spatial Feature Extraction", "Three stacked 3D convolutional blocks extract local spatiotemporal patterns from the input volume. Each block applies a Conv3D layer followed by Batch Normalization and ReLU activation. MaxPool3D layers between blocks progressively reduce spatial and temporal resolution while increasing feature depth. Following the convolutional blocks, two Residual Blocks — each containing two 3×3×3 convolutions with a skip connection and batch normalization — enable deeper feature learning without gradient vanishing. Adaptive Average Pooling then collapses the spatial dimensions to a fixed-length temporal feature sequence of shape (Batch, Time', 128), regardless of the input clip's spatial resolution."),

  bodyLabel("Stage 2 — Temporal Modelling", "The temporal feature sequence is processed by a two-layer Bidirectional LSTM with hidden size 256 per direction, producing a 512-dimensional representation at the final timestep. The bidirectional design means the LSTM processes the temporal sequence in both the forward (first frame to last) and backward (last frame to first) directions simultaneously, allowing it to capture both the motion trajectory of a sign and the final hand configuration that often carries discriminative information. Dropout (rate 0.3) is applied to the LSTM output for regularization."),

  bodyLabel("Stage 3 — Classification", "A linear layer projects the 512-dimensional LSTM output to 104 class logits. The total trainable parameter count is approximately 2.5 million, making the architecture computationally lightweight relative to transformer-based alternatives."),

  body("The architecture is trained from random initialization in all experiments — there are no pretrained weights of any kind. This is an important distinction from SignVLM, and it is the primary reason the architecture struggles under signer-disjoint evaluation: without any pretrained visual prior, the model must learn to recognize hand shapes and body configurations entirely from the limited PSL-104 training corpus."),

  BR(),
  figPlaceholder("Figure 2: 3DCNN + Residual Block + BiLSTM architecture diagram. Three-stage pipeline: stacked Conv3D blocks → two Residual Blocks with skip connections → two-layer BiLSTM → linear classifier. Input shape (B, 3, T, H, W); output 104-class logits."),
  BR(),

  H3("5.5.2 Primary Model: SignVLM"),

  body("SignVLM (Luqman, 2025) [6] is a two-component visual-temporal architecture that decouples spatial feature extraction from temporal modelling. This decoupling is the key architectural property that makes SignVLM effective in low-resource settings: it allows the spatial encoder — which requires enormous amounts of training data to learn meaningful representations — to be fully pretrained on an external corpus, while only the temporal decoder needs to be learned from the limited PSL training data."),

  bodyLabel("CLIP Visual Encoder (Frozen)", "Each video frame is independently processed by CLIP's Vision Transformer ViT-L/14, which is a Large-capacity ViT model (307 million parameters) pretrained by OpenAI on 400 million internet image-text pairs using a contrastive learning objective. The ViT-L/14 configuration divides each 224×224 input image into a 14×14 grid of non-overlapping 16-pixel patches and processes these through 24 transformer layers. The [CLS] classification token produced at the output of the final transformer layer encodes a 1,024-dimensional global representation of the frame that is rich in fine-grained spatial information about hand configuration, body posture, and scene context. This encoder is kept fully frozen throughout all PSL-104 training — its weights are never updated. The frozen encoder serves as a powerful, fixed feature extractor whose representations are signer-agnostic: because CLIP was trained on diverse internet imagery covering an enormous range of people, skin tones, backgrounds, and environments, its representations do not encode identity-specific appearance features, only semantic visual content."),

  bodyLabel("EVL Temporal Decoder (Trainable)", "The sequence of per-frame CLIP [CLS] tokens — one 1,024-dimensional vector per frame — is passed to the Efficient Video Learning (EVL) decoder proposed by Lin et al. (2022) [8]. The EVL decoder is a lightweight transformer decoder that models temporal dependencies across the frame sequence through four mechanisms: temporal convolutions that capture short-range motion patterns, cross-frame attention that relates distant frames in the sequence, multi-head self-attention (with 16 attention heads) over learnable decoder query tokens, and positional encodings that preserve temporal ordering. The decoder produces a fixed-dimensional temporal representation that aggregates information across all frames, which is then passed through Layer Normalization and a Dropout layer before the final linear classifier."),

  bodyLabel("Classification Head", "A linear projection from the EVL decoder output dimension to 104 class logits. During training, only the EVL decoder and this classification head have trainable parameters — the CLIP encoder contributes no gradients and requires no backward computation through its layers, substantially reducing training memory requirements and allowing effective batch accumulation on consumer-grade hardware."),

  BR(),
  figPlaceholder("Figure 3: SignVLM two-component architecture. Frozen CLIP ViT-L/14 extracts per-frame CLS token embeddings (1024-dim). The trainable EVL temporal decoder aggregates these across the T-frame sequence using temporal convolutions, cross-frame attention, and multi-head self-attention. A linear classifier produces 104-class logits."),
  BR(),

  body("The key advantage of SignVLM over scratch-trained architectures in this setting is precisely this property: CLIP's visual prior eliminates the need to learn what hands look like from PSL training data. The model only needs to learn which temporal patterns of CLIP-encoded features correspond to which PSL signs — a much simpler function to fit from fifteen clips per class."),

  // ─────────────────────────────────────────────────────────────
  H2("5.6 Experimental Progression Overview"),
  // ─────────────────────────────────────────────────────────────

  body("Our experimental work proceeded through six stages, each motivated by the findings of the previous stage. Rather than presenting experiments in isolation, we describe them here in the order they were conducted, explaining the hypothesis behind each experiment and what it revealed. This narrative is important because the experimental progression itself is a contribution of this project — it provides the first systematic empirical demonstration of the generalization gap in PSL SLR and a principled path toward closing it."),

  body("Table 5 provides an overview of all six experimental stages and their key results; the following subsections describe each stage in detail."),

  BR(),
  tblCaption("Table 5: Experimental progression overview. All results are Top-1 accuracy. Eval. protocol indicates whether signer-inclusive (random split) or signer-disjoint evaluation was used. F1 is macro-averaged across 104 classes."),
  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [380, 1620, 1440, 1440, 900, 900, 900, 1180],
    rows: [
      new TableRow({ tableHeader: true, children: [
        hCell("#",           380),
        hCell("Experiment",  1620),
        hCell("Model",       1440),
        hCell("Eval. Protocol", 1440),
        hCell("Top-1 Acc.", 900),
        hCell("Top-5 Acc.", 900),
        hCell("F1",         900),
        hCell("Key Finding",1180),
      ]}),
      ...[
        ["1", "Signer-Joint Baseline",       "3DCNN",   "Signer-inclusive",       "92.11%", "—",     "0.9207", "Pipeline confirmed; baseline extended"],
        ["2", "Real-World Unseen Signers",   "3DCNN",   "Truly unseen (60 clips)","8.33%",  "23.33%","0.0833", "Generalization gap exposed empirically"],
        ["3a","MediaPipe ROI (signer-joint)","3DCNN+ROI","Signer-inclusive",      "45.46%", "~62%",  "~0.45",  "ROI cropping degrades, not improves"],
        ["3b","MediaPipe ROI (disjoint)",    "3DCNN+ROI","Signer-disjoint",       "24.87%", "~38%",  "~0.25",  "No improvement from attention fix alone"],
        ["4", "Pose-Based SLR Exploration",  "N/A",     "N/A",                   "—",      "—",     "—",      "Deferred: tooling blockers"],
        ["5", "Signer-Disjoint Baseline",    "3DCNN",   "Signer-disjoint",       "12.88%", "—",     "~0.13",  "Gap formally quantified: 83.89→12.88%"],
        ["6a","SignVLM (16 frames)",         "SignVLM",  "Signer-disjoint",       "80.34%", "~91%",  "[TBD]",  "Gap largely closed with pretrained prior"],
        ["6b","SignVLM (24 frames)",         "SignVLM",  "Signer-disjoint",       "83.49%", "~93%",  "[TBD]",  "Best augmented-data result"],
        ["6c","SignVLM (24fr, orig. only)",  "SignVLM",  "Signer-disjoint",       "85.14%", "~94%",  "[TBD]",  "Best overall result (full-shot, no aug.)"],
      ].map((row, idx) => {
        const isLast = idx >= 5;
        const w = [380, 1620, 1440, 1440, 900, 900, 900, 1180];
        if (isLast) return new TableRow({ children: row.map((c, j) => bCell(c, w[j], j > 0)) });
        return new TableRow({ children: row.map((c, j) => dCell(c, w[j], j > 0)) });
      })
    ]
  }),
  BR(),

  // ─────────────────────────────────────────────────────────────
  H2("5.7 Stage 1: Signer-Joint Baseline Replication and Extension"),
  // ─────────────────────────────────────────────────────────────

  body("Before investigating any new architectures or evaluation protocols, our first responsibility was to replicate and extend the prior team's result on the combined PSL-104 dataset. This served two purposes: it confirmed that the preprocessing pipeline and training code were functioning correctly end-to-end, and it established a clear baseline showing the effect of adding three new signers to the training corpus under the same signer-inclusive evaluation conditions used by the prior team."),

  body("The prior team's 3DCNN + Residual Block + BiLSTM architecture was trained on the combined Phase 1 + Phase 2 dataset using a signer-inclusive random split — identical in methodology to the prior team's approach, but now drawing from all six signers rather than three. The same hyperparameters were used as in the prior team's training run: Adam optimizer with an initial learning rate of 1e-3, CrossEntropyLoss, and training for 10 epochs with a ReduceLROnPlateau scheduler. The same augmentation pipeline (prior team's spatial-only pipeline, applied to the combined dataset) was used for fair comparison."),

  body("The results confirmed that the additional signer data produced a meaningful improvement: the signer-joint test accuracy improved from the prior team's 83.89% to 92.11%, with Precision of 0.9367, Recall of 0.9211, F1 of 0.9207, and Test Loss of 0.3459. This improvement is exactly what one would expect when adding more examples: the model had access to more variation in how each sign is performed across different individuals, making it better able to generalize across the visual differences between the six training signers. The result validated that our preprocessing pipeline was correct and that the combined dataset was properly organized."),

  body("It is critical to re-emphasize what this result means: 92.11% accuracy under signer-inclusive evaluation is a strong result, but it reflects the model's ability to classify video clips from signers it has already seen during training. Whether this performance would hold on a genuinely new signer is an entirely separate question — and Stage 2 answers it."),

  BR(),
  figPlaceholder("Figure 4: Stage 1 — Signer-Joint 3DCNN training curves. Loss and Top-1 accuracy across 10 training epochs on the combined 6-signer signer-inclusive dataset. Both training and validation accuracy rise steadily, reaching 92.11% test accuracy at completion."),
  BR(),

  // ─────────────────────────────────────────────────────────────
  H2("5.8 Stage 2: Real-World Unseen Signer Inference Test"),
  // ─────────────────────────────────────────────────────────────

  body("The Stage 1 result was promising in isolation, but it had a fundamental limitation: the model had been evaluated on video clips from the same six signers it was trained on. To determine whether the model had actually learned sign-discriminating features — or had merely learned to recognize the training signers' individual appearances — we conducted a real-world inference test using video clips from individuals who had never appeared in the dataset at all."),

  body("We collected 60 short video clips covering a subset of the 104 PSL sign classes, recorded by friends and acquaintances of the team who had no prior involvement in the dataset. These individuals were asked to perform specific PSL signs based on the same PSL Dictionary reference videos that informed the original recordings, so the sign content was as accurate as the participants could manage without professional sign language training. The 60 clips were preprocessed using the same pipeline as the training data: .mov to .mp4 conversion, audio stripping, temporal cropping to the active signing window, and frame pre-extraction. Inference was then run using the Stage 1 signer-joint trained 3DCNN model without any retraining."),

  body("The results were stark. The model correctly classified only 5 of 60 clips at Top-1 — an accuracy of 8.33% — and 14 of 60 clips at Top-5 — a recall@5 of 23.33%. The complete metrics were: Accuracy 8.33%, Precision 0.0833, Recall 0.0833, F1 0.0833. For context, random chance for a 104-class classification problem would give approximately 0.96% Top-1 accuracy. The model therefore performed only marginally better than chance on genuinely new signers, despite achieving 92.11% on the known-signer test set."),

  body("This 83.78 percentage-point gap between signer-inclusive test accuracy (92.11%) and real-world unseen-signer accuracy (8.33%) is the central empirical finding that motivates everything else in this project. It confirms that the signer-inclusive evaluation that has been used throughout the PSL SLR literature does not reflect real-world model capability, and that any system reporting only signer-inclusive accuracy figures cannot be considered deployable. The model had not learned PSL signs — it had learned to recognize the training signers."),

  body("The failure mode is well-understood in the SLR literature. With approximately fifteen training clips per class, a scratch-trained 3D-CNN has insufficient data to learn robust visual features from first principles. The model is forced to leverage the easiest available discriminative signal, which in a signer-inclusive training corpus is not the sign's gestural content but the individual signer's appearance: their background, their skin tone relative to the background, their hand size, their face, their clothing color. These features are stable within a signer's training clips (they appear in both the training and the signer-inclusive test set), making them effective class discriminators under that evaluation condition. They are completely useless for classifying clips from new individuals."),

  BR(),
  figPlaceholder("Figure 5: Stage 2 — Generalization gap visualization. Bar chart comparing 92.11% signer-joint test accuracy vs. 8.33% unseen-signer inference accuracy for the same model. The 83.78 percentage-point gap quantifies the effect of signer-identity leakage in the training corpus."),
  BR(),

  // ─────────────────────────────────────────────────────────────
  H2("5.9 Stage 3: MediaPipe Hand ROI Extraction"),
  // ─────────────────────────────────────────────────────────────

  body("The diagnosis from Stage 2 suggested a clear hypothesis: the model was attending to whole-frame appearance features rather than focusing on the hands where the sign's linguistic content is actually encoded. If we could force the model to look only at the hand region — by cropping the video to just the bilateral hand bounding box before training — the signer-specific background and body appearance cues would be eliminated from the input, and the model might be forced to learn genuine hand-configuration features instead."),

  body("This hypothesis led us to implement a MediaPipe-based hand Region of Interest (ROI) extraction pipeline. MediaPipe Hands (Zhang et al., 2020) is a real-time hand detection system that operates in two stages: a palm detector that locates each hand's bounding box in the frame, and a landmark regression model that estimates 21 3D keypoints per hand. We used the bounding boxes from the palm detector to define the hand ROI for each frame."),

  body("The ROI extraction pipeline was implemented as follows. For each frame of each video, MediaPipe's palm detector was run to locate the bounding boxes of all detected hands. The bounding boxes of the left and right hands were merged into a single bilateral bounding box covering both hands (their union), to preserve relative hand positioning for two-handed signs. A padding factor of 1.3 was applied to expand the bounding box by 30% in each direction, ensuring that the full hand including fingertips was captured even when the palm detector's bounding box was slightly tight. The padded bilateral bounding box was then used to crop the frame, and the crop was resized to a fixed output resolution. In frames where MediaPipe failed to detect any hands — due to motion blur, fast motion, occlusion, or the hand being temporarily outside the palm detector's operating range — a fallback was applied: the full frame at reduced resolution was written rather than dropping the frame, to preserve temporal continuity in the clip."),

  body("This pipeline was applied to all videos across all three dataset splits, producing a parallel set of ROI-cropped videos stored in separate directories (train_data_roi/, validation_data_roi/, test_data_roi/). The 3DCNN architecture was then retrained from scratch on the ROI-cropped dataset using the same training configuration as Stage 1."),

  body("The results were counterproductive. Under signer-inclusive evaluation, accuracy fell dramatically from 92.11% (full-frame Stage 1) to 45.46% (ROI-cropped). Under signer-disjoint evaluation, accuracy was 24.87% — meaningfully higher than the 12.88% signer-disjoint full-frame result (Stage 5), but far below practical utility. The ROI approach failed to deliver the improvement that was hypothesized."),

  body("Post-hoc analysis identified four reasons for this failure:"),

  num("MediaPipe failure rate under fast motion. Sign language involves rapid hand movements that frequently exceed the frame-to-frame tracking capability of MediaPipe's palm detector. When hands move quickly, the detector loses track and falls back to the full frame, producing inconsistent crops across the clip — some frames show the tight hand ROI, others show the full frame. This temporal inconsistency in the crop region is disruptive to temporal models that expect spatially stable input across frames."),
  num("Aspect ratio distortion for two-handed signs. When both hands are far apart — a common configuration in two-handed signs — the bilateral union bounding box becomes very wide relative to its height. Resizing this to a square or fixed-aspect-ratio output distorts the spatial relationships between the two hands, potentially obscuring the configuration information that distinguishes the sign."),
  num("No pretrained visual prior. The ROI cropping changes the input to the 3DCNN but does not add any new visual knowledge. The model still has to learn what a hand looks like and what different hand configurations mean from fifteen training clips per class — the fundamental data scarcity problem is unchanged. Removing the background shortcut without providing an alternative source of discriminative features simply left the model with a harder learning problem and less discriminative signal."),
  num("Loss of non-manual cues. Sign language relies partly on facial expression and body posture to convey grammatical information. By cropping to the hand region, the model loses access to these non-manual features entirely. Even in the signer-inclusive setting, some discriminative information was being drawn from the whole frame, and losing it produced the large accuracy drop from 92.11% to 45.46%."),

  body("These findings are not discouraging — they are instructive. They reveal that the generalization problem in PSL SLR is not simply a spatial attention problem that can be solved by restricting the model's field of view. The root cause is the absence of a pretrained visual prior: the model needs to start from rich, pre-learned representations of what hands look like and how they move before it can learn sign-specific temporal patterns from fifteen clips per class. This insight directly motivated the adoption of SignVLM as the solution."),

  BR(),
  figPlaceholder("Figure 6: Stage 3 — MediaPipe ROI extraction examples. Left column: original full frames. Right column: corresponding ROI-cropped frames. Rows 1-2 show successful crops on clear, well-lit frames. Rows 3-4 show failure cases: fast-motion frame where MediaPipe lost tracking (fallback to full frame), and two-handed sign where bilateral union crop produces a very wide, distorted aspect ratio."),
  BR(),

  // ─────────────────────────────────────────────────────────────
  H2("5.10 Stage 4: Pose-Based SLR Exploration"),
  // ─────────────────────────────────────────────────────────────

  body("Concurrent with the MediaPipe ROI experiments, we explored a fundamentally different approach to the signer generalization problem: pose-based SLR. The key theoretical motivation is that skeletal pose representations — sequences of 3D joint positions describing the configuration of the signer's hands and body — are by construction independent of appearance features. A pose sequence encodes only the geometry of the signing motion, not the background, skin tone, clothing, or any other identity-specific visual attribute. Models operating in pose space therefore cannot, in principle, exploit signer identity as a classification shortcut, making them inherently more signer-agnostic."),

  body("We explored two architectures from the pose-based SLR literature as potential candidates. SignBERT (Hu et al., 2021) [placeholder-signbert] and its successor SignBERT+ (Hu et al., 2023) [placeholder-signbert+] both propose masked pose reconstruction as a self-supervised pretraining strategy for sign language representations, analogous to BERT's masked language modelling objective applied to skeletal sequences. The pretraining phase, which masks a subset of joints and trains a transformer to predict the masked configurations, produces rich representations of signing motion that can then be fine-tuned on small labeled corpora. This is precisely the kind of pretrained prior we were looking for in the pose domain."),

  body("For pose estimation itself — the step of converting raw RGB video frames to 3D joint keypoints — we evaluated two options. MediaPipe Pose provides a lightweight body landmark estimation pipeline, but its hand keypoint precision under fast motion and partial occlusion degrades significantly, particularly for the 21-point per-hand keypoint map required for fine-grained hand-shape discrimination. MMPose, the pose estimation framework from the OpenMMLab research ecosystem, provides access to state-of-the-art pose estimation models with substantially higher accuracy than MediaPipe under challenging conditions."),

  body("Unfortunately, MMPose could not be stably installed in either the Windows development environment or the Linux compute environment available to the project. The MMPose installation process involves a complex dependency chain — including specific versions of MMCV, MMEngine, and their associated CUDA extensions — that produced irreconcilable configuration conflicts in our hardware and OS combination during the project timeline. Multiple installation attempts with different version pinnings failed to produce a working environment. MediaPipe Pose, while installable, did not provide the keypoint quality needed for hand-shape-discriminating sign recognition in our fast-motion signing videos."),

  body("As a result, the pose-based SLR direction was deferred rather than abandoned. The theoretical argument for pose-based approaches remains strong — indeed, combining CLIP's spatial visual prior with a pose-based temporal representation is a promising direction for future work that may combine the advantages of both approaches. We defer this to Section 10 (Future Work) and note that our MediaPipe experiments from Stage 3 already provided practical evidence of the pose estimation quality problem under fast signing motion."),

  // ─────────────────────────────────────────────────────────────
  H2("5.11 Stage 5: Formal Signer-Disjoint Evaluation of 3DCNN"),
  // ─────────────────────────────────────────────────────────────

  body("Having established the generalization failure empirically through real-world inference (Stage 2), and having determined that spatial attention approaches cannot compensate for the absence of a pretrained visual prior (Stage 3), we proceeded to formally quantify the generalization gap under controlled signer-disjoint evaluation conditions. This stage uses the same 3DCNN + Residual Block + BiLSTM architecture as Stage 1, but trained and evaluated on the strictly signer-disjoint splits described in Section 5.3."),

  body("The training configuration was updated from Stage 1 in several respects, adopting best-practice techniques for training from scratch on small datasets. The Adam optimizer was replaced with AdamW (AdamW decouples weight decay from the gradient update step, providing better regularization), the initial learning rate was reduced from 1e-3 to 1e-4, label smoothing of 0.04 was applied to reduce overconfident softmax probabilities, a cosine learning rate decay schedule with a 3-epoch warmup was used in place of ReduceLROnPlateau, gradient clipping (max norm 1.0) was applied to stabilize LSTM training, and WeightedRandomSampler was used to correct for any minor class imbalances in the training distribution. Mixed-precision training (AMP with float16) was enabled to reduce GPU memory usage and training time. Early stopping was configured with a patience of 12 epochs."),

  body("The model was trained for 30 epochs before early stopping triggered, at which point the best validation accuracy seen during training was 12.88% — achieved at epoch 18. For reference, random chance for a 104-class problem gives 0.96% accuracy, so the model learned meaningfully above chance but far below practical utility. The training and validation loss curves were nearly flat throughout training, hovering around log(104) ≈ 4.644 nats — the theoretical cross-entropy for a uniform distribution over 104 classes — confirming that no meaningful class-discriminating learning occurred under the signer-disjoint condition."),

  body("The contrast between 83.89% (prior team, signer-inclusive) and 12.88% (our team, signer-disjoint) for the same architecture class quantifies exactly the effect of signer leakage in prior PSL SLR work. This is not a critique of the prior team's results — 83.89% is the correct answer to the question their evaluation asked. It is a demonstration that the question their evaluation asked is not the right question for assessing real-world system utility."),

  BR(),
  figPlaceholder("Figure 7: Stage 5 — Signer-Disjoint 3DCNN training curves over 30 epochs. Both training and validation loss remain near the theoretical entropy floor of log(104) ≈ 4.644 nats throughout, confirming that no meaningful class-discriminating learning occurs when signer identity is unavailable as a shortcut. Best validation accuracy: 12.88% at epoch 18."),
  BR(),

  // ─────────────────────────────────────────────────────────────
  H2("5.12 Stage 6: SignVLM — CLIP-Based Visual-Temporal Modelling"),
  // ─────────────────────────────────────────────────────────────

  body("The root cause identified by Stages 1 through 5 is unambiguous: scratch-trained architectures cannot generalize across unseen signers when the training corpus provides only fifteen clips per class, because they lack the visual prior knowledge required to learn sign-discriminating features from such limited data. The solution must therefore involve a model that brings substantial pre-learned visual knowledge about human hands and bodies into the PSL-104 training process, rather than attempting to learn these representations from scratch."),

  body("SignVLM (Luqman, 2025) [6] addresses this problem directly through its frozen CLIP ViT-L/14 encoder. CLIP's pretraining on 400 million image-text pairs produces a visual encoder that already understands human hands in enormous detail — their configurations, the relationships between fingers, the way they move, and the subtle differences between similar-looking hand shapes. This knowledge is available from frame zero of PSL-104 training, without consuming any of the PSL training data. The trainable EVL temporal decoder then only needs to learn to interpret the temporal sequence of CLIP-encoded frame features as PSL signs — a substantially simpler learning problem that is tractable from fifteen clips per class."),

  body("Several non-trivial engineering adaptations were required to deploy SignVLM on PSL-104 beyond what the original SignVLM codebase assumed. These are described in detail in Section 6 (Implementation Details). From a methodological perspective, the key design decisions were:"),

  num("Frame sampling: 24 frames per clip were uniformly sampled from each video. This matches the optimal frame count identified in the original SignVLM paper for datasets with comparable temporal complexity to PSL — longer clips (32 frames) showed marginal or negative returns due to redundant frames, while shorter clips (16 frames) slightly underperformed."),
  num("Frozen encoder: The CLIP ViT-L/14 encoder was kept fully frozen throughout all training. Experiments with partial unfreezing (not reported in the main results) showed training instability when encoder layers were updated on the small PSL corpus."),
  num("Normalization: CLIP requires specific input normalization (mean = [0.4815, 0.4578, 0.4082], std = [0.2686, 0.2613, 0.2758]), applied consistently across all clips in all splits."),
  num("Offline frame pre-extraction: As described in Section 5.2.3, all video frames were pre-extracted to JPEG files before training, reducing data loading time by approximately 16× and maintaining GPU utilization above 85%."),

  body("Two frame sampling configurations of SignVLM were evaluated: 16 frames per clip and 24 frames per clip, both under strictly signer-disjoint evaluation on the augmented training set. A third configuration — 24 frames per clip on the original (non-augmented) training clips — was additionally evaluated as an N-shot data efficiency test. Full results are presented in Section 7."),

  PB(),
];

module.exports = { methodologySection };