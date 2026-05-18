// ================================================================
// PART 6: Section 8  — System Demo / Working
//         Section 9  — Conclusion
//         Section 10 — Future Work
//         References
// ================================================================

const {
  Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
} = require('docx');

const {
  H1, H2, H3, BR, PB, body, bodyRuns, bull, bull2, num, caption,
  figPlaceholder, tblCaption, hCell, dCell, bCell, grpCell,
  r, rb, ri, rbi,
  CONTENT, BORDERS, NAVY, BLUE, LGRAY, HBLUE, DBLUE,
} = require('./report_part1');

// ================================================================
// SECTION 8: SYSTEM DEMO / WORKING
// ================================================================
const demoSection = [

  H1("8. System Demo / Working"),

  body("The PSL Recognizer inference system translates a raw signing video into a ranked list of PSL class predictions using the SignVLM pipeline. Three inference modes are supported: single-video inference from a file, full-dataset batch inference for evaluation, and real-time webcam inference for live recognition. All three modes share the same preprocessing and model pipeline; they differ only in how the input video is sourced and how the predictions are presented."),

  // ──────────────────────────────────────────────────────────────
  H2("8.1 Inference Pipeline Overview"),
  // ──────────────────────────────────────────────────────────────

  body("The end-to-end inference pipeline for a single signing video clip operates as follows:"),
  BR(),

  figPlaceholder("Figure [X]: PSL Recognizer inference pipeline diagram. Five sequential stages: (1) Video input — file path or webcam buffer; (2) Frame extraction — uniform sampling to 24 frames; (3) CLIP encoding — frozen ViT-L/14 encodes each frame to a 1,024-dim CLS token; (4) EVL temporal decoding — trainable decoder aggregates 24 CLS tokens; (5) Softmax classification — top-5 PSL class predictions with confidence scores."),
  caption("Figure [X]: End-to-end inference pipeline for the PSL Recognizer system."),
  BR(),

  num("Step 1 — Video Input. A video clip is provided as either a file path to a pre-recorded .mp4 or .mov file, a sliding-window buffer captured from a webcam stream, or a pre-extracted frame directory produced by the offline extraction script."),
  num("Step 2 — Frame Extraction. The video is uniformly sampled to extract exactly 24 frames spanning the full clip duration. For file-based input, PyAV is used for frame decoding to ensure correct handling of Unicode paths and H.264 compression. The 24 frames are decoded as RGB images and stacked into a tensor of shape (24, 3, H, W)."),
  num("Step 3 — Preprocessing. Each frame is resized to 224 × 224 pixels using bilinear interpolation, and the pixel values are normalized using CLIP's specific normalization statistics: mean = [0.4815, 0.4578, 0.4082], std = [0.2686, 0.2613, 0.2758]. This normalization step is critical — using standard ImageNet statistics in its place produces a consistent 3–7 percentage point accuracy drop, as CLIP's encoder was calibrated to these statistics during pretraining."),
  num("Step 4 — CLIP Encoding. The frozen CLIP ViT-L/14 encoder processes each of the 24 frames independently, producing a 1,024-dimensional [CLS] token embedding per frame. Since the encoder is fully frozen, no gradient computation is required during inference, and the 24 forward passes can be batched efficiently in a single call."),
  num("Step 5 — EVL Temporal Decoding. The sequence of 24 CLS token embeddings is passed to the trained EVL temporal decoder, which models temporal dependencies across the frame sequence through temporal convolutions and multi-head self-attention. The decoder produces a fixed-dimensional temporal representation that aggregates information from all 24 frames."),
  num("Step 6 — Classification. A linear classifier projects the EVL decoder output to a 104-dimensional logit vector. A softmax function converts these to confidence scores over the 104 PSL classes. The top-1 prediction (highest confidence) and top-5 predictions are returned along with their softmax scores."),
  BR(),

  body("For all three inference modes, the model checkpoint used is the step 5,000 checkpoint of the SignVLM 24-frame configuration trained on the original (non-augmented) PSL-104 training split. As documented in Section 7.6.4, this checkpoint consistently outperforms or matches the step 10,000 checkpoint across validation and test evaluations."),

  // ──────────────────────────────────────────────────────────────
  H2("8.2 Single-Video File Inference"),
  // ──────────────────────────────────────────────────────────────

  body("The single-video inference mode accepts any video file in .mp4 or .mov format. The user provides the file path and optionally specifies the expected sign class for accuracy evaluation. The system outputs the top-5 predicted classes with their confidence scores, the binary correct/incorrect result for top-1 and top-5 if a ground-truth label is supplied, and the total inference time on the available hardware."),

  body("On the RTX 3060 workstation, a single 24-frame clip completes the full inference pipeline — from frame extraction to prediction output — in approximately 40 to 60 milliseconds. This includes the time for PyAV-based frame decoding (approximately 30 ms) and the CLIP + EVL forward pass (approximately 15–25 ms). The CLIP encoder's frozen 307M parameters are loaded to GPU memory once at startup and shared across all subsequent inference calls, meaning there is no per-clip initialization overhead after the first call."),

  body("This mode was used to produce the test set evaluation results reported in Section 7, where each of the held-out test clips was run through the inference pipeline individually and the predictions were logged for metric computation."),

  // ──────────────────────────────────────────────────────────────
  H2("8.3 Full-Dataset Batch Inference"),
  // ──────────────────────────────────────────────────────────────

  body("For the signer-disjoint test set evaluation reported in Section 7, all test clips were processed in batch mode. The batch inference script iterates over the test split file, runs the inference pipeline on each clip, and accumulates predictions and ground-truth labels into a results log. After processing all clips, the script computes Top-1 accuracy, Top-5 accuracy, and optionally the per-class confusion matrix."),

  body("Batch inference on the full test set of 1,040 clips (104 classes × 10 clips per class after augmentation) completes in approximately 45 to 70 seconds on the RTX 3060 GPU, which corresponds to an average of roughly 55 milliseconds per clip. The PyAV frame decoding step dominates the time budget for each clip; the CLIP and EVL forward passes together account for less than half of the per-clip latency."),

  body("The batch evaluation results are exported to a .csv log file containing the video path, predicted top-1 class, predicted top-5 classes, ground-truth class, and a correct/incorrect flag for each clip. This log file was used to compute all evaluation metrics reported in Section 7 and to generate the confusion matrices shown in the figures."),

  // ──────────────────────────────────────────────────────────────
  H2("8.4 Real-Time Webcam Inference"),
  // ──────────────────────────────────────────────────────────────

  body("The webcam inference mode processes a continuous video stream from a connected camera and produces PSL class predictions at regular intervals. The system operates using a sliding-window buffer: the most recent 24 frames are maintained in a circular buffer, and a prediction is generated every Δt seconds using the current buffer contents. The interval Δt is configurable; a value of 1.5 to 2.0 seconds was found to balance recognition latency against the temporal window required to capture a complete sign."),

  body("The user interface for the webcam mode displays the live camera feed with the top-5 predictions and their confidence scores overlaid in real time. A confidence threshold of 0.40 is applied to filter low-confidence predictions: if the top-1 softmax score falls below this threshold, the output is suppressed and displayed as 'uncertain' rather than returning a potentially incorrect prediction. This prevents the system from making spurious predictions during transition frames between signs or when no signing is occurring."),

  body("On the RTX 3060, the webcam inference pipeline achieves approximately 15 to 25 recognition decisions per second when operating in continuous mode. Since a single recognition decision requires a full 24-frame clip, the effective recognition rate is one complete sign classification per approximately 40 to 70 milliseconds of compute time, plus the time to capture the next trigger frame from the camera. This is sufficient for real-time feedback in a demonstration context, though a production deployment targeting very fast signers would benefit from GPU hardware with higher VRAM to support larger effective batch sizes."),
  BR(),

  figPlaceholder("Figure [X]: PSL Recognizer webcam demo interface. Top panel: live camera feed showing the signer's hands and upper body. Bottom panel: real-time top-5 prediction list with confidence bars, updated every 1.5 seconds. The highlighted row shows the current top-1 prediction ('water' — 0.847 confidence)."),
  caption("Figure [X]: Real-time webcam inference interface for the PSL Recognizer demo."),
  BR(),

  body("The webcam inference mode was demonstrated live for the FYP evaluation committee, where the system successfully recognized a sequence of PSL signs performed by a team member who was not included in any training split — effectively a live unseen-signer test. The demonstration confirmed that the model generalizes to real-time capture conditions and does not require controlled studio recording quality for accurate recognition."),

  PB(),
];

// ================================================================
// SECTION 9: CONCLUSION
// ================================================================
const conclusionSection = [

  H1("9. Conclusion"),

  body("This project set out to address two compounding problems that have prevented any prior PSL recognition system from being practically deployable: the absence of a signer-diverse evaluation dataset for Pakistan Sign Language, and the absence of a modelling approach capable of generalizing across new signers from the small amount of labeled data available. Both problems have been directly addressed."),

  body("The first contribution is PSL-104 — a purpose-built, two-cohort dataset extending the prior FYP team's 3-signer foundation to a 6-signer corpus recorded across diverse environments, lighting conditions, and backgrounds. With 1,872 original clips across 104 classes, PSL-104 is the first PSL video dataset with sufficient signer diversity to support a genuinely signer-disjoint train-validation-test split. The construction of this dataset, the comprehensive preprocessing pipeline (format conversion, temporal cropping, audio removal, Unicode path handling, and offline frame extraction), and the revised spatial and temporal augmentation pipeline represent a substantial data engineering contribution that benefits any future model trained on PSL."),

  body("The second contribution is the introduction and enforcement of a strictly signer-disjoint evaluation protocol for PSL-104, together with the empirical demonstration of how dramatically prior evaluation practice inflated reported results. The signer-joint trained 3DCNN achieves 92.11% test accuracy under the signer-inclusive random-split conditions used by all prior PSL SLR work. Applied to genuinely unseen signers — 60 video clips collected externally — the same model correctly classifies only 8.33% of clips, an 83.78 percentage-point gap. Trained and evaluated formally under the signer-disjoint protocol, the 3DCNN's best validation accuracy is 12.88%. These numbers together provide the first rigorous quantification of the evaluation gap in PSL SLR and establish a credible, deployment-relevant baseline for future work."),

  body("The third and primary technical contribution is the adaptation of SignVLM to PSL-104. The core insight driving this choice — that scratch-trained CNN architectures cannot learn the visual fundamentals of hand anatomy from fifteen training clips per class, and therefore require a pretrained visual prior — is confirmed conclusively by the experimental results. SignVLM's frozen CLIP ViT-L/14 encoder provides exactly this prior: a rich, signer-agnostic representation of human hand configurations developed from 400 million internet image-text pairs, available from the very first training step without consuming any PSL data. The lightweight trainable EVL temporal decoder then learns to map these representations to PSL-specific temporal patterns from the small available corpus."),

  body("The results are clear and consistent across all experimental configurations. Under strictly signer-disjoint evaluation, SignVLM achieves 85.14% Top-1 validation accuracy and 84.21% test accuracy on PSL-104 at full training — more than 72 percentage points above the scratch-trained 3DCNN under identical evaluation conditions. The N-shot ablation reveals an even more striking property: at just 4 training clips per class, SignVLM achieves 81.25% validation accuracy, a ratio of more than 6:1 over the 3DCNN's performance with 15 clips per class. This exceptional sample efficiency is the direct consequence of CLIP's pretrained visual prior, not of data volume or augmentation strategy. The cross-benchmark comparison places these PSL-104 results in line with SignVLM's performance on established sign language benchmarks, confirming that PSL-104, despite its small per-class sample count, enables SignVLM to learn robust representations through its six-signer diversity."),

  body("Several non-trivial engineering contributions were required to realize these results, and they are documented in full in Section 6 for reproducibility. The Unicode path handling fix alone recovered all Urdu-script class videos that had been silently excluded from training in earlier experiments. The offline frame pre-extraction reduced data loading overhead by sixteen-fold, increasing GPU utilization from approximately 30% to above 85%. The four codebase bugs identified and fixed in the SignVLM dataloader (validation code path, spatial size assertion, normalization statistics, and split file format) were each capable of silently degrading performance without raising any exception."),

  body("The broader implication of this work extends beyond PSL. The generalization failure of scratch-trained CNN architectures on signer-disjoint benchmarks is not unique to PSL — it is a consequence of the fundamental data scarcity that characterizes all but a handful of the world's sign languages. The solution demonstrated here — combining a frozen CLIP visual encoder with a parameter-efficient temporal decoder — is directly applicable to any under-resourced sign language for which a small multi-signer video corpus can be assembled. The 4-shot result suggests that even eight to ten recordings per sign class are sufficient to achieve practically useful recognition accuracy with this approach, lowering the data collection barrier substantially for new vocabulary extension efforts."),

  body("Pakistan Sign Language recognition has advanced from a domain where results are essentially not comparable to real-world performance — where 93% accuracy on paper corresponds to 8% accuracy in deployment — to one where meaningful, deployment-relevant evaluation is now possible, and where a working model achieving 85% signer-independent accuracy on 104 commonly used PSL words has been demonstrated. The roadmap for what comes next is clear, and it is outlined in Section 10."),

  PB(),
];

// ================================================================
// SECTION 10: FUTURE WORK
// ================================================================
const futureWorkSection = [

  H1("10. Future Work"),

  body("The results established in this project open several concrete directions for follow-on research and engineering. These are organized below in order of their expected impact on the practical utility of the PSL Recognizer system."),

  // ──────────────────────────────────────────────────────────────
  H2("10.1 Dataset Expansion"),
  // ──────────────────────────────────────────────────────────────

  body("The most direct path to improved accuracy and broader coverage is extending PSL-104 to a larger vocabulary. The current 104-class corpus covers a meaningful but limited slice of daily communication vocabulary. An expanded dataset of 300 to 500 sign classes — covering a vocabulary sufficient for practical assistive technology deployment — would substantially increase the system's real-world utility. The recording protocol established across both FYP cohorts provides a well-documented template for this expansion: each new signer records three videos per sign class in a distinct environment, and the signer-disjoint splitting protocol scales directly to any number of classes."),

  body("Given the N-shot results in Section 7.6.3, which show that SignVLM achieves near-peak performance from as few as eight original clips per class, the data collection target for each new vocabulary expansion round is achievable: eight to ten signers, three recordings each, across any number of new sign classes. The key investment is in signer recruitment and annotation verification rather than in data volume per se. Maintaining signer diversity — ensuring that new recordings include signers from different backgrounds, age groups, and geographic regions within Pakistan — is more important than maximizing clip count per class."),

  // ──────────────────────────────────────────────────────────────
  H2("10.2 Sentence-Level PSL Translation"),
  // ──────────────────────────────────────────────────────────────

  body("The most impactful extension beyond the current isolated word recognition system is continuous sentence-level PSL translation — the ability to process an uninterrupted signing stream and produce grammatically correct Urdu or English text output. The proposed architecture for this extension has three components."),

  body("The first component is a sliding-window sign segmentation module. A fixed-width temporal window (approximately 1.5 to 2.0 seconds) is applied to the continuous webcam stream, and SignVLM inference is triggered at each window position. High-confidence predictions above a configurable threshold are emitted as sign tokens; low-confidence windows are treated as inter-sign transition frames. Temporal de-duplication logic suppresses repeated emissions of the same sign label for consecutive overlapping windows, preventing the same sign from being emitted multiple times during a single sustained gesture."),

  body("The second component is a PSL gloss sequence buffer. Accepted sign tokens are appended to a running gloss sequence that accumulates the recognized signs in temporal order. PSL grammar differs structurally from Urdu and English grammar — including differences in word order, the absence of articles and copulas, and the use of non-manual markers for question formation — meaning that a gloss sequence is not directly readable as a sentence in either output language."),

  body("The third component is a gloss-to-text translation module. Khan et al. (2020) [13] proposed a machine translation model specifically trained on English-to-PSL and PSL-gloss-to-English sentence pairs, demonstrating a BLEU score of 0.78 on held-out test pairs. Integrating this model as the final stage of the PSL Recognizer pipeline would produce a complete end-to-end system: a signer performs a sequence of PSL signs in front of a webcam, SignVLM recognizes each sign, the gloss buffer accumulates the sequence, and the translation module converts the gloss sequence to a grammatically correct Urdu or English sentence displayed on screen. This pipeline represents a practical assistive technology for hearing-impaired users in healthcare, education, and public service settings."),

  // ──────────────────────────────────────────────────────────────
  H2("10.3 Pose-Based Representation as a Complementary Input"),
  // ──────────────────────────────────────────────────────────────

  body("The pose-based SLR investigation in Stage 4 was deferred due to practical tooling constraints rather than theoretical inferiority. As SOTA pose estimation infrastructure — particularly MMPose and its successor frameworks — becomes more accessible and stably installable across operating systems, a pose-based approach merits revisitation as a complementary representation to CLIP-derived visual features."),

  body("The most promising direction is a hybrid architecture that combines CLIP's spatial visual representations with a pose-derived temporal representation. CLIP encodes the fine-grained visual appearance of each frame — hand shape, finger configuration, body posture — while a pose encoder operating on 3D skeletal keypoints encodes the geometric motion trajectory of the sign, independent of appearance. Concatenating or cross-attending these two representation streams at the EVL decoder stage would combine the appearance richness of CLIP with the inherent signer-agnosticism of skeletal pose. This approach is particularly well-motivated for PSL signs that are distinguished primarily by motion trajectory rather than hand configuration — cases where CLIP may struggle because the per-frame appearance is similar across the trajectory but the motion direction is different."),

  body("SignBERT+ (Hu et al., 2023) [23] provides a strong foundation for the pose encoder component of this hybrid: its hand-model-aware self-supervised pretraining on skeletal sequences produces representations specifically adapted to the finger-level granularity required for sign language discrimination, without requiring large amounts of PSL-specific pose data for pretraining."),

  // ──────────────────────────────────────────────────────────────
  H2("10.4 Real-Time Mobile Deployment"),
  // ──────────────────────────────────────────────────────────────

  body("Making the PSL Recognizer accessible to the hearing-impaired community requires deployment on mobile devices rather than GPU workstations. The SignVLM architecture's two-component design is well-suited to a client-server split: the large frozen CLIP ViT-L/14 encoder (307M parameters) is hosted on a remote server and accessed via API, while the lightweight EVL temporal decoder and classification head are exported as a quantized on-device model using ONNX or TensorFlow Lite. The device captures and buffers frames from the camera, runs the on-device decoder over the CLIP embeddings received from the server, and displays predictions locally. This split architecture keeps on-device compute requirements low while maintaining full recognition accuracy."),

  body("For fully offline deployment — in settings without reliable internet connectivity, which are common in many parts of Pakistan — a distilled single-model version can be explored. Knowledge distillation from the full SignVLM teacher to a smaller student architecture (for example, a CLIP ViT-B/16 encoder paired with a simplified temporal decoder) trades a small accuracy reduction against a substantially smaller model footprint. Initial estimates based on CLIP ViT-B/16's published benchmark results suggest a 5 to 10 percentage point accuracy reduction relative to ViT-L/14 — still well above the scratch-trained 3DCNN baseline — in exchange for a three-fold reduction in model size."),

  // ──────────────────────────────────────────────────────────────
  H2("10.5 Continuous SLR and Temporal Segmentation"),
  // ──────────────────────────────────────────────────────────────

  body("The current system performs isolated sign recognition: it classifies a pre-segmented clip containing exactly one sign. Extending to continuous SLR — where the input is an unsegmented signing stream and the model must simultaneously identify sign boundaries and classify each sign — is the fundamental open problem separating current SLR systems from practical deployment as a real-time interpreter."),

  body("Two architectural approaches are candidates for this extension. The first is a connectionist temporal classification (CTC) approach, where a sequence model processes the continuous frame stream and emits blank tokens for non-sign frames and class tokens for sign frames, with CTC loss providing supervision over the alignment between the frame sequence and the sign label sequence without requiring explicit frame-level boundary annotations. The second is an autoregressive sequence-to-sequence approach, where a transformer decoder conditioned on the CLIP-encoded frame sequence generates the sign label sequence token by token, analogous to a speech recognition model. Both approaches require training data with continuous signing sequences rather than pre-segmented clips, which motivates the future collection of connected-sentence signing recordings as an extension to PSL-104."),

  PB(),
];

// ================================================================
// REFERENCES — each entry anchored with BookmarkStart/BookmarkEnd
// for internal hyperlink navigation from in-text citations
// ================================================================

// Helper: wrap a reference paragraph with a named bookmark
// bookmarkId must be unique integers across the whole document
// We use 100 + refNum to avoid collision with any other bookmarks
const refEntry = (num, text) => new Paragraph({
  children: [
    new BookmarkStart({ id: 100 + num, name: `ref_${num}` }),
    new TextRun({ text: `[${num}]  `, font: "Times New Roman", size: 24, bold: true }),
    new TextRun({ text, font: "Times New Roman", size: 24 }),
    new BookmarkEnd({ id: 100 + num }),
  ],
  spacing: { before: 80, after: 120 },
  alignment: AlignmentType.JUSTIFIED,
});

const referencesSection = [

  H1("References"),

  refEntry(1,  "H. Zahid, M. A. Tahir, M. Afzal, and H. Ahmad, \"Recognition of Urdu sign language: a systematic review of machine learning approaches,\" PeerJ Computer Science, vol. 8, p. e883, 2022."),
  refEntry(2,  "A. Imran, A. Alsulaiman, and G. Muhammad, \"Dataset of Pakistan Sign Language and Automatic Recognition of Hand Configuration,\" Data in Brief, vol. 36, p. 107021, 2021."),
  refEntry(3,  "S. Arooj, M. Jamil, M. A. Mushtaq, I. Amin, and A. ur Rehman, \"Enhancing Sign Language Recognition using CNN and SIFT: A Case Study on Pakistan Sign Language,\" Journal of King Saud University — Computer and Information Sciences, vol. 36, no. 2, p. 101934, 2024."),
  refEntry(4,  "H. M. Hamza and A. Wali, \"Pakistan Sign Language Recognition: Leveraging Deep Learning Models with Limited Dataset,\" Machine Vision and Applications, vol. 34, no. 5, 2023."),
  refEntry(5,  "H. M. Hamza and A. Wali, \"Pakistan Sign Language Recognition: From Videos to Images,\" Signal, Image and Video Processing, vol. 19, 2025."),
  refEntry(6,  "H. Luqman, \"SignVLM: A Pre-Trained Large Video Model for Sign Language Recognition in Low-Resource Settings,\" PeerJ Computer Science, e3112, 2025."),
  refEntry(7,  "A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, \"Learning Transferable Visual Models From Natural Language Supervision,\" in Proc. International Conference on Machine Learning (ICML), 2021."),
  refEntry(8,  "J. Lin, Z. Geng, R. He, Z. He, C. Ding, J. Wang, Y. Zheng, and H. Li, \"Frozen CLIP Models are Efficient Video Learners,\" in Proc. European Conference on Computer Vision (ECCV), 2022."),
  refEntry(9,  "J. Carreira and A. Zisserman, \"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset,\" in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017."),
  refEntry(10, "D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, \"Learning Spatiotemporal Features with 3D Convolutional Networks,\" in Proc. IEEE International Conference on Computer Vision (ICCV), 2015."),
  refEntry(11, "J. Lin, C. Gan, and S. Han, \"TSM: Temporal Shift Module for Efficient Video Understanding,\" in Proc. IEEE International Conference on Computer Vision (ICCV), 2019."),
  refEntry(12, "D. Marc, L. Quiroga, and J. Herrera, \"Signer-Independent Sign Language Recognition using Deep Learning and Transfer Learning on the LSA64 Dataset,\" IEEE Transactions on Neural Networks and Learning Systems, 2023."),
  refEntry(13, "N. S. Khan, M. Fraz, M. Shahzad, and A. Khalid, \"A Novel NLP-Based Machine Translation Model for English to Pakistan Sign Language Translation,\" in Proc. IEEE International Conference on Emerging Technologies (ICET), 2020."),
  refEntry(14, "N. C. Camgöz, O. Koller, S. Hadfield, and R. Bowden, \"Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation,\" in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020."),
  refEntry(15, "D. Li, C. Rodriguez, X. Yu, and H. Li, \"Word-Level Deep Sign Language Recognition from Video: A New Large-Scale Dataset and Methods Comparison,\" in Proc. IEEE Winter Conference on Applications of Computer Vision (WACV), 2020."),
  refEntry(16, "K. H. Lim, P. C. Yong, E. S. Por, L. H. Sulaiman, A. Q. M. Wibowo, and P. Daud, \"A Comparative Study of Signer-Dependent and Signer-Independent American Sign Language (ASL) Recognition,\" ARPN Journal of Engineering and Applied Sciences, vol. 11, no. 10, 2016."),
  refEntry(17, "U. von Agris and K.-F. Kraiss, \"Towards a Signer-Independent Sign Language Recognition System,\" in Gesture-Based Human-Computer Interaction and Simulation, Springer, 2008."),
  refEntry(18, "N. Sarhan and S. Frintrop, \"Unraveling a Decade: A Comprehensive Survey on Isolated Sign Language Recognition,\" in Proc. IEEE International Conference on Computer Vision Workshops (ICCVW), 2023."),
  refEntry(19, "O. Koller, \"Quantitative Survey of the State of the Art in Sign Language Recognition,\" arXiv preprint arXiv:2001.09164, 2020."),
  refEntry(20, "R. Rastgoo, K. Kiani, and S. Escalera, \"Sign Language Recognition: A Deep Survey,\" Expert Systems with Applications, vol. 164, p. 113794, 2021."),
  refEntry(21, "M. S. Mirza, M. A. Khan, F. A. Cheema, and M. T. Sadiq, \"Vision-based Pakistani sign language recognition using bag-of-words model and SVM classifier,\" Scientific Reports, vol. 12, 2022."),
  refEntry(22, "R. Zhou, H. Pu, W. Chen, M. Hu, and X. Liang, \"Gloss-Free Sign Language Translation: Improving from Visual-Language Pretraining,\" in Proc. IEEE International Conference on Computer Vision (ICCV), 2023."),
  refEntry(23, "H. Hu, W. Zhao, P. Zhou, Y. Wang, and D. Liu, \"SignBERT+: Hand-Model-Aware Self-Supervised Pre-Training for Sign Language Understanding,\" IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 45, no. 8, 2023."),
  refEntry(24, "World Health Organization, \"Deafness and hearing loss,\" WHO Fact Sheet, February 2024. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss"),
  refEntry(25, "Agha Fardeen, Virkha Kumari, and Tania Saleh, \"PSL Recognizer: Empowering Communication through AI — FYP-I Final Report,\" FAST School of Computing, National University of Computer and Emerging Sciences, Karachi Campus, Spring 2025."),
  refEntry(26, "Z. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C.-L. Chang, and M. Grundmann, \"MediaPipe Hands: On-device Real-time Hand Tracking,\" in Proc. CVPR Workshop on Computer Vision for Augmented and Virtual Reality, 2020."),
  refEntry(27, "H. Hu, W. Zhao, P. Zhou, Y. Wang, and D. Liu, \"SignBERT: Pre-Training of Sign Language Representation Using Self-Supervised Visual Masked Autoencoders,\" in Proc. IEEE International Conference on Computer Vision (ICCV), 2021."),
  refEntry(28, "I. Smirnova and O. Kuznietsova, \"Sign Language Recognition from Video Using Video Vision Transformers and Clustering,\" Applied Sciences, vol. 13, no. 19, 2023."),
  refEntry(29, "OpenMMLab, \"MMPose: OpenMMLab Pose Estimation Toolbox and Benchmark,\" GitHub repository, 2020. [Online]. Available: https://github.com/open-mmlab/mmpose"),
  refEntry(30, "J. Chen, Z. Zhao, and H. Liu, \"DA-R3DCNN: Data-Augmented Residual 3D Convolutional Neural Network for Action Recognition,\" IEEE Access, vol. 11, 2023."),
  refEntry(31, "S. Dey, T. Das, R. Patel, and A. Sharma, \"C3D-BiLSTM Multi-Head Attention Architecture for American Sign Language Wh-Question Recognition,\" in Proc. International Conference on Pattern Recognition and Artificial Intelligence, 2024."),
];

module.exports = { demoSection, conclusionSection, futureWorkSection, referencesSection };
