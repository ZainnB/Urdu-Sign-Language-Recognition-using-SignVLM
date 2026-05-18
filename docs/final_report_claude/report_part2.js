// ================================================================
// PART 2: Section 4 — Literature Review / Related Work
// ================================================================

const {
  Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType, PageBreak,
  LevelFormat
} = require('docx');

const {
  H1, H2, H3, BR, PB, body, bodyRuns, bull, bull2, num, caption, figPlaceholder,
  tblCaption, hCell, dCell, bCell, grpCell, eq, r, rb, ri, rbi,
  CONTENT, BORDERS, NAVY, BLUE, LGRAY, HBLUE, DBLUE, BORDERS_NONE,
  AlignmentType: AT
} = require('./part1');

// ── SECTION 4: LITERATURE REVIEW ─────────────────────────────
const litReviewSection = [
  H1("4. Literature Review / Related Work"),

  body("The field of Sign Language Recognition has evolved substantially over the past two decades, progressing from hand-crafted feature methods and shallow classifiers to deep convolutional networks, recurrent architectures, and most recently, large pretrained vision-language models. This section surveys the literature relevant to our work across five areas: static PSL and Urdu SLR (Section 4.1), video-based PSL recognition (Section 4.2), CNN and 3D-CNN architectures developed broadly for SLR (Section 4.3), the role of transfer learning and large pretrained models (Section 4.4), and pose-based and hand-attention approaches that we investigated as intermediate strategies (Section 4.5). A comparative summary of all prior PSL and Urdu SLR studies is provided in Table 1."),

  // ── 4.1 Static PSL ───────────────────────────────────────────
  H2("4.1 Static PSL and Urdu Sign Language Recognition"),

  body("The majority of published work on PSL and Urdu Sign Language (USL) recognition operates on static images rather than video. This body of work is valuable as an early foundation but is inherently limited in scope: it captures only the hand configuration component of a sign, entirely ignoring the motion trajectories, non-manual features (facial expression, head tilt, body posture), and temporal dynamics that carry the full linguistic content of most signs."),

  body("Zahid et al. (2022) [1] conducted the most comprehensive systematic review of the field to date, surveying machine learning approaches applied to PSL and USL across a decade of publications. Their review found that nearly all prior systems operate on static images of Urdu alphabet characters, use single-signer datasets with signer-dependent evaluation, and rely on relatively shallow classifiers such as Support Vector Machines (SVMs), k-Nearest Neighbours (k-NN), or early convolutional architectures. This observation is significant because it establishes the structural limitations of the existing corpus: the reported accuracy figures — which are frequently high, reaching into the 90s — are measured under conditions where the model is tested on samples from the same individual it was trained on, and only on static hand shapes rather than full dynamic vocabulary. Neither of these conditions maps to real-world deployment."),

  body("Imran et al. (2021) [2] contributed one of the more structured dataset efforts in this space, presenting a dataset of 1,480 static images covering 37 Urdu alphabet characters, with 40 images per character recorded from a single signer. They also developed a mobile application for real-time alphabet recognition and reported 90% accuracy using a CNN classifier. While the contribution of the dataset and application is meaningful, the single-signer, static-image, and alphabet-only design means the system cannot scale to dynamic word-level PSL recognition."),

  body("Arooj et al. (2024) [3] represent a more recent attempt, proposing a hybrid CNN-SIFT system that additionally makes use of depth data from a Kinect sensor. Their approach applies Scale-Invariant Feature Transform (SIFT) for keypoint extraction — capturing hand margin, size, and finger position — followed by segmentation through thresholding and a CNN classifier for gesture recognition. The reported accuracy on the 37-class PSL alphabet dataset reached 98.74%. This is an impressive figure but comes with the same caveats: single-signer, signer-dependent evaluation, static gestures only, and reliance on Kinect depth data not available in standard deployment scenarios. Additionally, the system was evaluated on images, not video — even the Kinect recordings were treated as individual frame captures rather than temporal sequences."),

  body("Several earlier works — Halim and Abbas (2014), Kanwal et al. (2014), Nasir et al. (2014), Sami et al. (2014), and Husnain et al. (2019) — produced systems following the same general template: static images, Urdu alphabets, SVM or simple neural network classifiers, signer-dependent evaluation, and reported accuracies between 75% and 98%. While these works collectively establish that hand-shape recognition is tractable in controlled single-signer settings, they do not advance the field toward the practical goal of recognizing dynamic PSL vocabulary across multiple signers. The consistent absence of signer-independent evaluation across all these works is the central methodological gap that our project sets out to address."),

  body("One notable non-image contribution in the static category is the Leap Motion-based work of Khan et al. (2015) [placeholder], which used sensor data from a Leap Motion Controller and Artificial Neural Networks to convert PSL gestures into text and speech. While this avoids the image-classification framing, it introduces a hardware dependency that is impractical for most deployment contexts, and the vocabulary coverage and signer diversity were limited. The broader lesson from this body of work is that the PSL SLR problem is not simply one of algorithm design — the foundational data and evaluation infrastructure required to develop and assess generalizable systems has been largely absent."),

  // ── 4.2 Video-Based PSL ──────────────────────────────────────
  H2("4.2 Video-Based PSL Recognition"),

  body("A smaller and more recent body of work has addressed dynamic, video-based PSL recognition — the setting that is the direct focus of our project. The most significant contribution in this category is the work of Hamza and Wali (2023) [4], who were the first to apply established video recognition architectures — specifically C3D, I3D, and the Temporal Shift Module (TSM) — to dynamic PSL recognition at word level. Their best result, 93.33% with C3D, was reported on a dataset of 80 signs with 160 total video samples (2 samples per sign) drawn from the PSL Dictionary — a single-signer corpus."),

  body("On the surface, 93.33% is a compelling result. However, it was obtained using a random train-test split on a 2-sample-per-class, single-signer corpus. As Marc et al. (2023) [12] demonstrated on the LSA64 dataset, and as we demonstrate directly on PSL-104 in this project, random splitting on a single-signer corpus invariably leads to signer leakage: the same individual's videos appear in both training and test, allowing the model to exploit appearance-based shortcuts rather than learning sign content. In our own replication using a structurally similar architecture under signer-disjoint evaluation, performance drops to 12.88% — a gap of over 80 percentage points. This directly quantifies how much signer leakage inflates the results reported by Hamza and Wali (2023)."),

  body("Hamza and Wali (2025) [5] presented a follow-up study that took a different architectural direction. Rather than processing video directly, they converted sign videos into static landmark-trajectory images by computing skeletal joint positions across time and rendering these as RGB visualizations, then applying CNN classifiers to the resulting images. This approach achieved 92.5% accuracy on an expanded 100-sign vocabulary. However, this design choice discards temporal ordering information — two trajectories with the same keypoints but different motion directions would produce identical or near-identical images — and entirely removes non-manual linguistic features such as facial expression, head position, and body posture, which carry grammatical meaning in natural sign language. The approach is also indirectly dependent on accurate pose estimation quality, which degrades under fast motion and partial occlusion."),

  body("Taken together, these works establish that video-based PSL recognition is a tractable problem in signer-dependent settings, but that no prior study has demonstrated robust, signer-independent performance on dynamic PSL word-level recognition. This gap — practical generalizability across unseen signers — is the central problem addressed by our project."),

  // ── 4.3 CNN / 3D-CNN ─────────────────────────────────────────
  H2("4.3 CNN and 3D-CNN Architectures for SLR"),

  body("The dominant architectural paradigm for video-based SLR over the past decade has combined per-frame spatial feature extraction using Convolutional Neural Networks (CNNs) with temporal sequence modelling using Recurrent Neural Networks (RNNs), or alternatively used 3D CNNs that jointly extract spatial and temporal features from video volumes."),

  body("The ResNet family of architectures, particularly ResNet-18 and ResNet-50, have been widely used as frozen or fine-tuned spatial feature extractors within SLR pipelines. Their residual connections mitigate gradient vanishing in deep networks, enabling more stable training. Huang and Chouvatut (2024) [placeholder] demonstrated this approach on the LSA64 dataset, using a pre-trained ResNet-18 backbone to extract per-frame spatial features which are then fed into an LSTM for temporal modelling. They reported 86.25% accuracy on LSA64, outperforming several baseline 3D-CNN approaches, and noted that the use of a pre-trained backbone reduced training time and improved generalization compared to training from scratch."),

  body("3D CNN architectures extend the 2D convolution operation into the temporal dimension, allowing the network to learn spatiotemporal features jointly. The C3D architecture (Tran et al., 2015) [10] was among the first practical 3D CNN models and has been widely used in action recognition and sign language recognition benchmarks. I3D (Inflated 3D ConvNet, Carreira and Zisserman, 2017) [9] further improved on this by inflating pre-trained 2D ImageNet filters into 3D, providing a useful initialization for video models that inherit spatial knowledge from image pretraining. Hamza and Wali (2023) [4] directly applied both C3D and I3D to PSL recognition. The Temporal Shift Module (TSM, Lin et al., 2019) [11] offers a computationally efficient alternative to full 3D convolutions by shifting channel features across time, enabling temporal reasoning at essentially the cost of a 2D CNN."),

  body("The prior FYP cohort (Agha et al., 2025) proposed a hybrid architecture — 3DCNN + Residual Block + BiLSTM — that combines stacked 3D convolution layers, two residual learning blocks for gradient stability, and a two-layer bidirectional LSTM for temporal modelling. This architecture, which forms our direct baseline, achieved 83.89% test accuracy on the prior team's signer-inclusive PSL dataset. The residual blocks are directly inspired by the Data-Augmented Residual 3D CNN (DA-R3DCNN) proposed by Chen et al. (2023) [placeholder], which demonstrated that residual connections within 3D CNN backbones improve action recognition accuracy by enabling deeper networks without gradient degradation. The BiLSTM component is motivated by the C3D-BiLSTM Multi-Head Attention architecture proposed by Dey et al. (2024) for American Sign Language Wh-question recognition, which showed that bidirectional temporal modelling captures both the forward motion of a sign and the return-to-rest trajectory, improving recognition of signs where the ending hand position is linguistically significant."),

  body("Smirnova and Kuznietsova (2023) [placeholder] applied Video Vision Transformers (ViViT) to the LSA64 Argentine Sign Language dataset, additionally exploring a strategy of grouping semantically related signs into clusters to reduce data sparsity. They reported 69.7% Top-1 accuracy for ViViT in this clustered setting, highlighting both the potential of transformer-based architectures for SLR and the importance of dataset design strategies that compensate for limited per-class data. The prior FYP cohort also trained a ViViT model on their PSL dataset, observing strong training performance (~96%) but severe overfitting on the validation set — a pattern consistent with the general observation that large transformer architectures require significantly more data than our corpus provides to generalize."),

  body("A consistent finding across this body of work is that scratch-trained 3D-CNN and transformer architectures require far more data than is available for low-resource sign languages like PSL to achieve robust generalization. Marc et al. (2023) [12] provided systematic evidence for this on the LSA64 dataset, showing that random signer-inclusive splits inflate reported accuracy by 30-50 percentage points compared to signer-disjoint evaluation, and that this gap is consistently large regardless of the specific architecture used. Their finding directly parallels and anticipates our empirical result on PSL-104."),

  // ── 4.4 Transfer Learning and Large Pretrained Models ────────
  H2("4.4 Transfer Learning and Large Pretrained Models for SLR"),

  body("The fundamental bottleneck for SLR in low-resource languages — insufficient annotated data — has motivated a growing body of work exploring transfer learning and large pretrained models as a way to bring external visual knowledge into the sign recognition pipeline."),

  body("The most influential development in this direction has been CLIP (Contrastive Language-Image Pretraining, Radford et al., 2021) [7]. CLIP is a Vision Transformer trained jointly with a language model on 400 million image-text pairs collected from the internet. Its training objective — aligning image and text embeddings of semantically matching pairs in a shared latent space — produces visual representations that are extraordinarily rich and generalized. Crucially for sign language recognition, the internet's visual content includes an enormous variety of images of human hands, arms, and bodies in diverse poses, lighting conditions, skin tones, and environments. CLIP therefore develops fine-grained representations of human hand configurations without any sign-specific supervision, making its features inherently more signer-agnostic than those of models trained specifically on constrained video recognition datasets."),

  body("Lin et al. (2022) [8] demonstrated that a frozen CLIP image encoder can be repurposed for video understanding with minimal additional training through the Efficient Video Learning (EVL) framework. EVL attaches a lightweight temporal decoder — incorporating temporal convolutions, cross-frame attention, and multi-head self-attention — to the sequence of per-frame CLIP feature embeddings. By keeping the CLIP encoder frozen, the model avoids overwriting the rich pretrained representations and only needs to learn the temporal aggregation function from the downstream video dataset. This parameter-efficient design is particularly well-suited to low-data settings."),

  body("Luqman (2025) [6] applied this CLIP+EVL framework specifically to sign language recognition, proposing SignVLM as a unified architecture. SignVLM was evaluated on four sign language datasets — KArSL (Arabic), WLASL-100 (American), LSA64 (Argentine), and AUTSL (Turkish) — under signer-independent evaluation conditions and across multiple few-shot settings. SignVLM achieved state-of-the-art performance on three of the four datasets (KArSL, WLASL-100, and LSA64) and competitive performance on AUTSL. Most importantly for our purposes, SignVLM's performance scaled rapidly with the number of training clips per class: even at 4-shot (4 clips per class), SignVLM achieved 81.9% on KArSL-100 and 96.5% on LSA64, demonstrating that its CLIP-derived visual prior removes the need for large amounts of sign-specific data. This property makes SignVLM directly relevant to PSL, where the available labeled corpus is small and signer diversity is limited."),

  body("The relevance of CLIP's pretraining to sign language can be understood through the specific nature of its training data. Unlike action recognition models such as I3D or SlowFast, which are pretrained on coarse-grained motion categories (sports, cooking, etc.) with little fine-grained hand detail, CLIP's internet pretraining exposes it to the full diversity of human hands in social and cultural contexts — cooking images, crafts, gestures, teaching, photography. Its 14x14 patch grid on a 224x224 input preserves substantially more spatial detail per frame than coarser video models, allowing it to distinguish subtle differences in finger positions that determine the identity of many similar-looking signs."),

  body("Beyond SignVLM, other work has explored CLIP for sign-related understanding. Zhou et al. (2023) [placeholder] proposed CVT-SLR, which uses CLIP's visual representations with contrastive visual-textual learning for sign language recognition, demonstrating that language-aligned visual features provide a natural representation space where sign semantics are better organized. Hu et al. (2023) [placeholder-signbert+] proposed SignBERT+, which extends CLIP-style pretraining to pose-based sign representations through masked pose reconstruction as a self-supervised objective. These works collectively support the hypothesis that pretrained representations — particularly those that align visual content with language — are especially effective for sign language tasks where labeled data is scarce."),

  body("Selvaraj et al. (2022) [placeholder] investigated cross-lingual transfer learning for sign language recognition through the OpenHands library, showing that models pretrained on one sign language can improve performance on others with as little as 2% to 18% accuracy gains depending on the structural similarity of the languages. While PSL is sufficiently distinct from all other sign languages that direct cross-lingual transfer is unlikely to work well, this work motivates the broader principle: visual representations trained on rich, diverse data — whether from another sign language or from the general internet as in CLIP — transfer more effectively than training from scratch on limited target-domain data."),

  // ── 4.5 Pose-Based and Hand-Attention Approaches ─────────────
  H2("4.5 Pose-Based and Hand-Attention Approaches"),

  body("An alternative approach to the signer generalization problem is to eliminate appearance-based features entirely by working in the pose domain. If a model's input is a skeletal representation of the signer — a set of joint positions or keypoints — rather than RGB pixel values, the model cannot exploit identity-specific appearance cues such as skin tone, clothing, or background. Pose-based approaches have attracted substantial interest precisely because they offer a principled path to signer-agnostic recognition."),

  body("SignBERT (Hu et al., 2021) [placeholder-signbert] introduced masked pose reconstruction as a self-supervised pretraining strategy for sign language representations. Inspired by BERT's masked language modelling, SignBERT randomly masks a subset of skeletal joints and trains the model to predict the masked joint positions. This forces the network to learn the structural relationships between joints that characterize different hand configurations and motion patterns, without relying on appearance. SignBERT+ (Hu et al., 2023) [placeholder-signbert+] extended this framework with hand-model-aware pretraining, incorporating anatomical constraints from a parametric hand model (MANO) into the self-supervised objective, improving the quality of the learned representations particularly for signs that involve subtle finger configurations."),

  body("MediaPipe Hands (Zhang et al., 2020) [placeholder-mediapipe] is a widely used real-time hand detection and landmark estimation system developed by Google. It operates in two stages: a palm detector that identifies the bounding box of each hand in the frame, followed by a landmark regression model that estimates 21 3D keypoints per hand at sub-pixel accuracy. MediaPipe is appealing for sign language applications because it is lightweight, runs in real time on standard hardware, and produces structured output that is independent of signer appearance. It has been used in several SLR pipelines as a preprocessing step to extract hand ROI crops or to generate pose-based input representations."),

  body("Graph Convolutional Networks (GCNs) have become the dominant architecture for pose-based SLR, building on the natural graph structure of the human skeleton where joints are nodes and anatomical connections are edges. ST-GCN (Song et al., 2017) [placeholder] first applied spatial-temporal graph convolutions to action recognition, and this approach has since been adapted extensively for SLR. Recent variants include SignGraph (Naz et al., 2023) [placeholder], which reported 72.1% on WLASL-100 using pose data, and the work of Ozdemir et al. (2023) who incorporated multi-cue LSTMs processing hands, body, and face simultaneously in a GCN framework to achieve 90.85% on the AUTSL dataset."),

  body("Pose-based approaches carry an important theoretical advantage: by design, they discard appearance information and operate purely on geometric relationships between body parts, making them signer-agnostic in principle. However, this advantage is contingent on the quality of pose estimation. In practice, current off-the-shelf pose estimators — including MediaPipe — exhibit accuracy degradation under conditions that are common in real signing: fast hand motion, partial occlusion between hands, close proximity of both hands to the face or body, and non-frontal signer orientations. When the pose estimator fails, it either produces inaccurate keypoints or drops frames entirely, introducing noise into the sequence representation that can be more damaging than the appearance features it was intended to eliminate."),

  body("We investigated two pose-related approaches during the development of this project. First, we used MediaPipe Hands to extract bilateral hand bounding box crops from all videos, creating a separate ROI-cropped dataset intended to focus model attention on the hands and reduce background and identity influence. Second, we explored MMPose — a state-of-the-art pose estimation framework from the OpenMMLab ecosystem — as a potential source of higher-quality skeletal representations for a fully pose-based approach. The findings from both investigations are described in the methodology section and inform why we ultimately chose the CLIP-based SignVLM approach. In short: hand ROI cropping hurt rather than helped the scratch-trained 3DCNN, because MediaPipe's failure rate on fast-motion frames introduced inconsistent crops that degraded temporal coherence; and MMPose could not be stably installed in our available computing environment on the hardware and OS configuration we were working with. These practical findings motivate the CLIP-based approach as a more robust alternative — CLIP's large-scale pretraining implicitly learns attention to hands and body structure without requiring an explicit, fragile pose estimation step."),

  // ── 4.6 Summary and Gap ───────────────────────────────────────
  H2("4.6 Summary and Identified Gaps"),

  body("Table 1 provides a comprehensive summary of prior PSL and Urdu SLR work alongside our own results, including input modality, gesture type, number of classes, sample count, dataset, and whether signer-independent (SI) evaluation was enforced. The pattern is consistent: all prior work uses signer-dependent evaluation, all static-image work operates on alphabet recognition only, and the two video-based studies that address dynamic vocabulary do so on the same limited, single-signer corpus (the PSL Dictionary) using random splitting. No prior study has demonstrated robust, signer-independent, dynamic word-level PSL recognition."),

  body("Our work closes this gap through three parallel contributions: extending the dataset to include six signers so that genuine signer-disjoint evaluation is possible, quantifying how large the evaluation gap is in this specific setting (a finding with implications for the credibility of all prior results), and demonstrating that SignVLM — a large pretrained visual-temporal model — achieves practical signer-independent accuracy on this challenging benchmark. Together, these contributions advance PSL SLR from a domain where results are essentially not comparable to real-world performance to one where meaningful, deployment-relevant evaluation is now possible."),

  BR(),

  // ── TABLE 1 ──────────────────────────────────────────────────
  tblCaption("Table 1: Summary of prior PSL and Urdu Sign Language recognition studies. SI = Signer-Independent evaluation enforced. N/A = not reported. † = BLEU score (translation metric, not accuracy)."),

  new Table({
    width: { size: CONTENT, type: WidthType.DXA },
    columnWidths: [1600, 900, 900, 800, 900, 1000, 560, 700],
    rows: [
      // header
      new TableRow({ tableHeader: true, children: [
        hCell("Study",          1600),
        hCell("Input",          900),
        hCell("Type",           900),
        hCell("Classes",        800),
        hCell("Samples",        900),
        hCell("Dataset",        1000),
        hCell("SI?",            560),
        hCell("Acc.",           700),
      ]}),
      // rows
      ...[
        ["Halim & Abbas (2014)",      "Image",    "Static",  "PSL signs",  "N/A",   "N/A",        "No",  "91%"],
        ["Kanwal et al. (2014)",      "Image",    "Static",  "PSL signs",  "N/A",   "N/A",        "No",  "90%"],
        ["Nasir et al. (2014)",       "Image",    "Static",  "PSL signs",  "N/A",   "N/A",        "No",  "97/86%"],
        ["Sami et al. (2014)",        "Image",    "Static",  "37 alpha",   "N/A",   "N/A",        "No",  "75%"],
        ["Husnain et al. (2019)",     "Image",    "Static",  "48 classes", "38,400","N/A",        "No",  "96/98%"],
        ["Imran et al. (2021)",       "Image",    "Static",  "37 alpha",   "1,480", "PSL",        "No",  "90%"],
        ["Arooj et al. (2024)",       "Kinect",   "Static",  "37 classes", "N/A",   "PSL (Urdu)", "No",  "98.7%"],
        ["Hamza & Wali (2023)",       "Video",    "Dynamic", "80 signs",   "160",   "PSL Dict.",  "No",  "93.3%"],
        ["Hamza & Wali (2025)",       "Landmark", "Dynamic", "100 signs",  "N/A",   "PSL Dict.",  "No",  "92.5%"],
        ["Khan et al. (2020)",        "Text",     "NLP/Trans.","Glosses",  "2k",    "PSL",        "No",  "0.78†"],
        ["Agha et al. (2025) [FYP-I]","Video",   "Dynamic", "104 classes","4,992", "PSL-104",    "No",  "83.89%"],
        ["Ours (2026) — 3DCNN",       "Video",   "Dynamic", "104 classes","6,968", "PSL-104",    "Yes", "12.88%"],
        ["Ours (2026) — SignVLM",      "Video",   "Dynamic", "104 classes","6,968", "PSL-104",    "Yes", "85.14%"],
      ].map((row, i) => new TableRow({
        children: row.map((cell, j) => {
          const w = [1600, 900, 900, 800, 900, 1000, 560, 700][j];
          const isOurs = row[0].startsWith("Ours");
          const isBest = row[0].includes("SignVLM");
          if (isBest) return bCell(cell, w, j > 0);
          if (isOurs) return new TableCell({
            borders: BORDERS("2E74B5"),
            width: { size: w, type: WidthType.DXA },
            shading: { fill: "EBF3FB", type: ShadingType.CLEAR },
            margins: { top: 60, bottom: 60, left: 120, right: 120 },
            children: [new Paragraph({
              children: [new TextRun({ text: String(cell), font: "Times New Roman", size: 20, italics: true })],
              alignment: j > 0 ? AT.CENTER : AT.LEFT
            })]
          });
          return dCell(cell, w, j > 0);
        })
      }))
    ]
  }),
  BR(),

  PB(),
];

module.exports = { litReviewSection };
