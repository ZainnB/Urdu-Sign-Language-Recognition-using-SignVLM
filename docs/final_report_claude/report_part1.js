const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, PageNumber, Footer, Header,
  BorderStyle, WidthType, ShadingType, VerticalAlign, PageBreak,
  LevelFormat, UnderlineType
} = require('docx');
const fs = require('fs');

// ── Palette ─────────────────────────────────────────────────
const NAVY   = "1F3864";
const BLUE   = "2E74B5";
const LGRAY  = "F5F5F5";
const HBLUE  = "D6E4F0";
const DBLUE  = "BDD7EE";
const WHITE  = "FFFFFF";

// ── Page / content dimensions (US Letter, 1" margins) ────────
const PAGE_W   = 12240;
const PAGE_H   = 15840;
const MARGIN   = 1440;
const CONTENT  = PAGE_W - 2 * MARGIN; // 9360

// ── Border helpers ────────────────────────────────────────────
const bdr  = (color = "AAAAAA") => ({ style: BorderStyle.SINGLE, size: 1, color });
const bdrS = (color = "2E74B5") => ({ style: BorderStyle.SINGLE, size: 4, color });
const BORDERS  = (c = "AAAAAA") => ({ top: bdr(c), bottom: bdr(c), left: bdr(c), right: bdr(c) });
const BORDERS_NONE = { top: { style: BorderStyle.NONE, size: 0, color: WHITE },
                        bottom: { style: BorderStyle.NONE, size: 0, color: WHITE },
                        left:   { style: BorderStyle.NONE, size: 0, color: WHITE },
                        right:  { style: BorderStyle.NONE, size: 0, color: WHITE } };

// ── TextRun helpers ───────────────────────────────────────────
const r   = (t, opts = {}) => new TextRun({ text: t, font: "Times New Roman", size: 24, ...opts });
const rb  = (t, opts = {}) => new TextRun({ text: t, font: "Times New Roman", size: 24, bold: true, ...opts });
const ri  = (t, opts = {}) => new TextRun({ text: t, font: "Times New Roman", size: 24, italics: true, ...opts });
const rbi = (t, opts = {}) => new TextRun({ text: t, font: "Times New Roman", size: 24, bold: true, italics: true, ...opts });

// ── Paragraph helpers ─────────────────────────────────────────
const BR = () => new Paragraph({ children: [] });

const body = (text, opts = {}) => new Paragraph({
  children: [r(text)],
  spacing: { before: 80, after: 120 },
  alignment: AlignmentType.JUSTIFIED,
  ...opts
});

const bodyRuns = (runs, opts = {}) => new Paragraph({
  children: runs,
  spacing: { before: 80, after: 120 },
  alignment: AlignmentType.JUSTIFIED,
  ...opts
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text, font: "Georgia", size: 32, bold: true, color: NAVY })],
  spacing: { before: 360, after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 4 } }
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  children: [new TextRun({ text, font: "Georgia", size: 26, bold: true, color: NAVY })],
  spacing: { before: 280, after: 140 }
});

const H3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  children: [new TextRun({ text, font: "Times New Roman", size: 24, bold: true, color: "2E4057" })],
  spacing: { before: 200, after: 100 }
});

const PB = () => new Paragraph({ children: [new PageBreak()] });

const bull = (text, bold_prefix = "") => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  children: bold_prefix
    ? [rb(bold_prefix + " "), r(text)]
    : [r(text)],
  spacing: { before: 60, after: 80 },
  alignment: AlignmentType.JUSTIFIED
});

const bull2 = (text) => new Paragraph({
  numbering: { reference: "bullets2", level: 0 },
  children: [r(text)],
  spacing: { before: 40, after: 60 },
  alignment: AlignmentType.JUSTIFIED
});

const num = (text, bold_prefix = "") => new Paragraph({
  numbering: { reference: "numbers", level: 0 },
  children: bold_prefix
    ? [rb(bold_prefix + " "), r(text)]
    : [r(text)],
  spacing: { before: 60, after: 80 },
  alignment: AlignmentType.JUSTIFIED
});

const caption = (text) => new Paragraph({
  children: [ri(text, { color: "555555", size: 20 })],
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 200 }
});

const figPlaceholder = (label) => new Table({
  width: { size: CONTENT, type: WidthType.DXA },
  columnWidths: [CONTENT],
  rows: [new TableRow({ children: [new TableCell({
    borders: BORDERS("AAAAAA"),
    width: { size: CONTENT, type: WidthType.DXA },
    shading: { fill: "F0F4F8", type: ShadingType.CLEAR },
    margins: { top: 600, bottom: 600, left: 200, right: 200 },
    children: [new Paragraph({
      children: [new TextRun({ text: `[ FIGURE PLACEHOLDER: ${label} ]`, font: "Arial", size: 20, italics: true, color: "888888" })],
      alignment: AlignmentType.CENTER
    })]
  })]})],
});

// ── Table helpers ─────────────────────────────────────────────
const tblCaption = (text) => new Paragraph({
  children: [rb(text, { size: 20 })],
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 80 }
});

const hCell = (text, w, span = 1) => new TableCell({
  columnSpan: span,
  borders: BORDERS("2E74B5"),
  width: { size: w, type: WidthType.DXA },
  shading: { fill: DBLUE, type: ShadingType.CLEAR },
  margins: { top: 80, bottom: 80, left: 120, right: 120 },
  children: [new Paragraph({
    children: [new TextRun({ text, font: "Times New Roman", size: 20, bold: true })],
    alignment: AlignmentType.CENTER
  })]
});

const dCell = (text, w, center = false, opts = {}) => new TableCell({
  borders: BORDERS("CCCCCC"),
  width: { size: w, type: WidthType.DXA },
  margins: { top: 60, bottom: 60, left: 120, right: 120 },
  children: [new Paragraph({
    children: [new TextRun({ text: String(text), font: "Times New Roman", size: 20, ...opts })],
    alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT
  })]
});

const bCell = (text, w, center = true) => new TableCell({
  borders: BORDERS("2E74B5"),
  width: { size: w, type: WidthType.DXA },
  shading: { fill: "EBF3FB", type: ShadingType.CLEAR },
  margins: { top: 60, bottom: 60, left: 120, right: 120 },
  children: [new Paragraph({
    children: [new TextRun({ text: String(text), font: "Times New Roman", size: 20, bold: true })],
    alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT
  })]
});

const grpCell = (text, w, span = 1) => new TableCell({
  columnSpan: span,
  borders: BORDERS("999999"),
  width: { size: w, type: WidthType.DXA },
  shading: { fill: "E8EDF2", type: ShadingType.CLEAR },
  margins: { top: 60, bottom: 60, left: 120, right: 120 },
  children: [new Paragraph({
    children: [new TextRun({ text, font: "Times New Roman", size: 20, bold: true, italics: true })],
    alignment: AlignmentType.LEFT
  })]
});

// ── Equation-style display ────────────────────────────────────
const eq = (text) => new Paragraph({
  children: [new TextRun({ text, font: "Courier New", size: 22, italics: true })],
  alignment: AlignmentType.CENTER,
  spacing: { before: 120, after: 120 }
});

// =============================================================
// BUILD DOCUMENT CONTENT
// =============================================================

// ── COVER PAGE ────────────────────────────────────────────────
const coverPage = [
  BR(), BR(), BR(),
  new Paragraph({
    children: [new TextRun({ text: "PSL Recognizer", font: "Georgia", size: 56, bold: true, color: NAVY })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Empowering Communication through AI", font: "Georgia", size: 36, color: BLUE })],
    alignment: AlignmentType.CENTER, spacing: { after: 80 }
  }),
  new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BLUE, space: 1 } },
    children: [], spacing: { after: 240 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "FYP-II Final Project Report", font: "Times New Roman", size: 28, italics: true, color: "444444" })],
    alignment: AlignmentType.CENTER, spacing: { after: 80 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Project Code: F25-115", font: "Times New Roman", size: 24, color: "555555" })],
    alignment: AlignmentType.CENTER, spacing: { after: 400 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Project Supervisor", font: "Georgia", size: 22, bold: true, color: NAVY })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Miss Anum Qureshi", font: "Times New Roman", size: 24 })],
    alignment: AlignmentType.CENTER, spacing: { after: 200 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Project Co-Supervisor", font: "Georgia", size: 22, bold: true, color: NAVY })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Ms. Virkha Kumari (Alumni)", font: "Times New Roman", size: 24 })],
    alignment: AlignmentType.CENTER, spacing: { after: 300 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Project Team", font: "Georgia", size: 22, bold: true, color: NAVY })],
    alignment: AlignmentType.CENTER, spacing: { after: 100 }
  }),
  ...["Muhammad Zain Baig  —  22K-4593", "Aarij Ali  —  22K-4264", "Abdur Rehman Khan  —  22K-4155"].map(name =>
    new Paragraph({ children: [new TextRun({ text: name, font: "Times New Roman", size: 24 })], alignment: AlignmentType.CENTER, spacing: { after: 60 } })
  ),
  BR(),
  new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BLUE, space: 1 } },
    children: [], spacing: { after: 240 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Submitted in partial fulfillment of the requirements for the degree of", font: "Times New Roman", size: 22, italics: true, color: "555555" })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Bachelor of Science in Computer Science", font: "Times New Roman", size: 24, bold: true })],
    alignment: AlignmentType.CENTER, spacing: { after: 300 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "FAST School of Computing", font: "Georgia", size: 26, bold: true, color: NAVY })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "National University of Computer and Emerging Sciences", font: "Times New Roman", size: 22 })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Karachi Campus  |  Spring 2026", font: "Times New Roman", size: 22 })],
    alignment: AlignmentType.CENTER
  }),
];

// ── ABSTRACT ──────────────────────────────────────────────────
const abstractSection = [
  H1("Abstract"),
  bodyRuns([
    r("Sign languages are visually expressed languages that rely on spatial structure and continuous motion over time, and they serve as the primary communication medium for hearing-impaired individuals worldwide. Automated Sign Language Recognition (SLR) — the problem of mapping signed video sequences to their spoken or written language equivalents — has the potential to substantially reduce the communication barrier between the deaf community and the general population. Pakistan Sign Language (PSL), used by an estimated ten million hearing-impaired citizens in Pakistan, remains one of the most underrepresented languages in SLR research. The limited prior work on PSL is largely confined to static alphabet recognition or to dynamic gesture models evaluated under conditions that do not account for signer diversity, making their reported results misleading for real-world deployment. The only publicly available PSL video resource — the official PSL Dictionary — provides just two video clips per sign from a single signer, making it impossible to train or evaluate models that generalize across new individuals."),
  ]),
  BR(),
  bodyRuns([
    r("This project addresses both the data bottleneck and the modelling gap simultaneously. We present "),
    rb("PSL-104"),
    r(", a purpose-built, signer-diverse video dataset of 104 commonly used Urdu words and alphabets covering 70 dynamic gestures, constructed across two consecutive FYP cohorts. The prior cohort established the initial 3-signer corpus and the foundational 3DCNN+ResBlock+BiLSTM architecture. Our FYP-II team extended the dataset to six signers, scraped and processed the official PSL Dictionary videos, performed extensive video preprocessing, and developed a substantially improved augmentation pipeline covering spatial, photometric, and temporal transforms — yielding 1,872 original clips across varied environments and lighting conditions. We introduce and enforce a strictly signer-disjoint evaluation protocol and demonstrate that the random-split methodology prevalent in prior PSL work induces signer leakage that artificially inflates accuracy. Under this protocol, scratch-trained CNN-based models fail to generalize to unseen signers, as evidenced by our own custom architecture achieving 92.11% under signer-inclusive evaluation but collapsing to 8.33% on a real-world test of 60 videos from genuinely new signers. We then adapt "),
    rb("SignVLM"),
    r(" — a large pretrained video model that couples a frozen CLIP ViT-L/14 visual encoder with a trainable Efficient Video Learning (EVL) temporal decoder — to PSL-104. CLIP's pretraining on 400 million image-text pairs provides rich, signer-agnostic spatial representations of hand shape and motion without consuming any PSL training data, while the lightweight EVL decoder learns PSL-specific temporal patterns from our limited corpus. Trained and evaluated on PSL-104 under strictly signer-disjoint conditions, SignVLM achieves strong generalization to unseen signers and high sample efficiency even in low-resource settings, establishing it as a practical solution for low-resource PSL recognition."),
  ]),
  BR(),
  bodyRuns([rb("Keywords: "), ri("Pakistan Sign Language, Sign Language Recognition, Transfer Learning, CLIP, SignVLM, Signer Independence, Signer-Disjoint Evaluation, Video Augmentation, Low-Resource Learning")]),
  PB(),
];

// ── SECTION 1: INTRODUCTION ───────────────────────────────────
const introSection = [
  H1("1. Introduction / Background"),

  body("Communication is the foundation of human social life. For the approximately 70 million deaf and hard-of-hearing individuals worldwide who rely on more than 300 distinct sign languages to express themselves, the inability of the majority hearing population to understand sign language creates a persistent and serious communication barrier [1]. Sign languages are not simplified or encoded versions of spoken languages — they are fully independent linguistic systems with their own vocabulary, grammar, syntax, and phonological structure, conveyed through the coordinated use of hand shape, hand movement, facial expression, and body posture. This rich, spatially encoded nature makes sign language fundamentally different from any form of spoken or written language, and it requires temporal understanding across video frames rather than recognition from a single image."),

  body("Automated Sign Language Recognition (SLR) is the computer vision and machine learning problem of automatically mapping signed video sequences to their corresponding spoken or written language equivalents. A reliable SLR system could serve as a real-time interpreter, supporting hearing-impaired individuals in healthcare settings, classrooms, workplaces, and everyday communication scenarios where sign language interpreters are unavailable. Considerable research progress has been made for resource-rich sign languages, particularly American Sign Language (ASL) and Chinese Sign Language (CSL), largely because large-scale annotated datasets exist for these languages. However, for the vast majority of the world's sign languages — including Pakistan Sign Language — the necessary data infrastructure and standardized evaluation methodology have been almost entirely absent."),

  body("Within SLR research, two important distinctions shape how systems are designed and evaluated. The first is the nature of the signs themselves: static sign recognition treats each sign as a single hand configuration that can be identified from a still image, while dynamic sign recognition requires modelling motion across a video sequence to capture the full gestural content of a sign. The overwhelming majority of real-world signing is dynamic, meaning that a static image classifier is fundamentally unsuited to practical deployment, regardless of how high its accuracy may be on image benchmarks. The second distinction concerns the scope of the recognition task: isolated SLR recognizes one sign at a time from a segmented video clip — essentially a multi-class classification problem over a fixed vocabulary — while continuous SLR must process uninterrupted signing streams and simultaneously identify sign boundaries. Isolated SLR is the foundational problem from which continuous SLR is built, and it is the focus of this project."),

  body("A further, often overlooked distinction in SLR evaluation is whether a system is signer-dependent or signer-independent. A signer-dependent model is trained and tested on video from the same set of signers; it can achieve high accuracy by learning identity-specific visual features such as background, skin tone, hand size, and clothing rather than the sign's actual meaning. A signer-independent model is tested on individuals it has never seen during training, which is the only condition that actually reflects deployment in the real world. Despite being a well-documented methodological concern in the SLR literature [16, 17], the majority of prior PSL work has not enforced signer independence in its evaluation."),

  body("Pakistan Sign Language is the primary mode of communication for Pakistan's estimated ten million hearing-impaired citizens [1]. Crucially, PSL is a linguistically independent system: it bears no direct relationship to Urdu, English, or any other Pakistani language, and it is also structurally distinct from other regional sign languages such as ASL or Indian Sign Language. Consequently, models trained on other sign languages cannot be transferred directly to PSL, making dedicated PSL-specific research essential. Despite this need, the field remains severely underdeveloped. The systematic review by Zahid et al. (2022) [1] found that prior PSL and Urdu SLR work is almost entirely limited to static image recognition of Urdu alphabet letters, using small single-signer datasets evaluated without signer-independent validation. A small number of more recent works have addressed dynamic PSL recognition [4, 5], but as this project demonstrates, their reported results are inflated by the use of signer-inclusive evaluation conditions."),

  body("The root cause of PSL's underrepresentation in SLR research is a data problem. The only publicly available PSL video resource is the official PSL Dictionary maintained by the PSL Organization (psl.org.pk), which provides exactly two video clips per sign, both recorded by the same signer under identical, controlled studio conditions. This is fundamentally insufficient for training or evaluating any model intended to generalize across new signers, environments, or recording conditions."),

  body("This project addresses the data and modelling problems simultaneously, building directly on the foundation established by the prior FYP cohort (Agha Fardeen, Virkha Kumari, and Tania Saleh, 2025) who constructed the initial PSL dataset and proposed the 3DCNN+ResBlock+BiLSTM architecture. Our FYP-II work extends the dataset, substantially revamps the preprocessing and augmentation pipeline, introduces rigorous signer-disjoint evaluation, and demonstrates that a large pretrained visual-temporal model — SignVLM — can bridge the generalization gap that CNN-based scratch-trained models cannot."),

  body("Our work proceeded through a systematic sequence of experiments before arriving at the final solution. We began by replicating and extending the prior team's signer-joint 3DCNN results on the combined dataset, confirming strong performance under that evaluation condition. We then exposed the model to a real-world test using videos from genuinely unseen individuals, revealing a dramatic generalization failure. This led us to investigate two intermediate approaches — MediaPipe-based hand Region of Interest (ROI) extraction and pose-based SLR architectures — before identifying the root cause of the failure as the absence of a strong pretrained visual prior, and adapting SignVLM as the solution. This progression is documented in full throughout this report."),

  body("SignVLM (Luqman, 2025) [6] builds on two key developments: CLIP (Contrastive Language-Image Pretraining, Radford et al. 2021) [7], a Vision Transformer pretrained on 400 million internet image-text pairs that develops rich, fine-grained visual understanding of human hands and body configurations as a byproduct of its general pretraining objective; and the Efficient Video Learner (EVL, Lin et al. 2022) [8], which demonstrated that a frozen CLIP image encoder can be adapted for video understanding by attaching a lightweight trainable temporal decoder. SignVLM applies this CLIP+EVL framework specifically to sign language recognition and has demonstrated state-of-the-art results across multiple sign language benchmarks, with its advantages being most pronounced in low-data settings — precisely the regime that characterizes PSL and most underrepresented sign languages worldwide."),

  H2("1.1 Contributions"),
  body("This project makes the following concrete contributions:"),
  num("PSL-104 Dataset Extension: We extend the prior cohort's 3-signer PSL corpus to a 6-signer dataset by recording 3 new signers across varied environments, lighting conditions, and backgrounds, bringing the total signer count to 6 and enabling genuine signer-disjoint evaluation."),
  num("Comprehensive Data Preprocessing Pipeline: We scrape and integrate the official PSL Dictionary videos, convert and clean all raw recordings, perform temporal cropping to remove pre-sign and post-sign blank frames, and extract frames for efficient training — a substantial engineering contribution that is fully documented for reproducibility."),
  num("Revamped Augmentation Pipeline: We redesign the prior team's spatial-only augmentation pipeline to include temporal augmentations (frame jitter, random frame drop) alongside the spatial transforms, significantly increasing the variety and realism of the augmented training distribution."),
  num("Signer-Disjoint Evaluation Protocol: We introduce and enforce a strictly signer-disjoint evaluation protocol for PSL-104, demonstrating empirically that random-split methodology inflates accuracy by as much as 79 percentage points (92.11% signer-joint vs. 12.88% signer-disjoint for the same architecture class)."),
  num("Systematic Generalization Failure Analysis: We document the full experimental progression from signer-joint success to real-world failure, including unseen-signer inference testing (8.33% Top-1 on 60 clips from new individuals), MediaPipe ROI exploration, and pose-based SLR investigation — providing a thorough analysis of why scratch CNN architectures cannot generalize in this low-resource setting."),
  num("SignVLM on PSL: We adapt and train SignVLM on PSL-104, achieving strong validation and test accuracy under signer-disjoint conditions and demonstrating significant sample efficiency across N-shot experiments."),
  num("Future Roadmap: We outline a concrete path toward sentence-level PSL-to-text translation by integrating SignVLM's word recognition with NLP-based sign-to-text generation methods [13], alongside a dataset expansion plan."),
  BR(),

  body("The rest of this report is structured as follows: Section 2 states the problem formally. Section 3 lists the project objectives. Section 4 reviews the relevant prior literature. Section 5 describes the methodology and system design in full, including the dataset construction, preprocessing, augmentation, and model architectures. Section 6 details the implementation. Section 7 presents all experimental results in the order they were obtained. Section 8 describes the system demo and inference pipeline. Section 9 concludes, and Section 10 outlines future work."),
  PB(),
];

// ── SECTION 2: PROBLEM STATEMENT ─────────────────────────────
const problemSection = [
  H1("2. Problem Statement"),

  body("Existing Pakistan Sign Language recognition systems suffer from a set of compounding problems that, taken together, prevent any current system from being useful in real-world deployment. These problems can be grouped into three interconnected categories:"),

  H2("2.1 The Dataset Problem"),
  body("The only publicly available PSL video resource is the official PSL Dictionary (psl.org.pk), which provides exactly two video samples per sign, both recorded by the same individual under controlled studio conditions. While this resource is valuable as a reference, it is structurally unsuitable as a standalone training corpus for two reasons. First, two clips per class is far below the data quantity required for any deep learning model to learn meaningful, generalizable features — even simple image classifiers typically require hundreds of samples per class for reliable training. Second, and more fundamentally, a single-signer corpus provides no exposure to inter-signer variation. A model trained on one person's signing will learn to associate that person's appearance — their background, their hand size, their skin tone, their recording environment — with class labels, rather than learning the sign's actual motion content. When deployed on a new signer, such a model has no basis for classification because the learned appearance features simply do not match."),
  body("Prior efforts to build PSL datasets have been limited in scope. The prior FYP cohort (Agha et al., 2025) made a meaningful contribution by recording 3 signers across 104 classes, but their dataset still contained too few signers for a split that keeps all training and test signers completely separate. This project addresses this by extending to 6 signers and enforcing strict signer disjointness across all dataset splits."),

  H2("2.2 The Evaluation Problem"),
  body("Prior PSL video recognition studies, including the well-cited work of Hamza and Wali (2023) [4] and the prior FYP cohort's results, evaluate models using random train-test splitting applied across the entire dataset. Under this protocol, videos from the same signer can appear in both the training and test sets. This creates a phenomenon known as signer leakage: the model can learn to recognize the signer's identity from visual appearance cues and use this as a proxy for the sign label, achieving high test accuracy without learning anything about the gestural content of the sign. The result is a reported accuracy figure that looks strong on paper but completely fails to predict how the system will perform in the real world."),
  body("This is not a subtle statistical artefact — the gap between signer-inclusive and signer-independent accuracy can be enormous. In our experiments, the same architecture class achieves 92.11% under signer-inclusive random splitting but only 12.88% under strictly signer-disjoint evaluation on the same dataset. The prior team's reported 83.89% test accuracy similarly reflects signer-inclusive evaluation conditions. This does not mean their architecture was flawed; it means the evaluation conditions did not expose the model to the challenge it will face in deployment."),

  H2("2.3 The Real-World Deployment Problem"),
  body("The practical consequences of the evaluation problem become starkly visible when a signer-inclusive trained model is applied to genuinely new individuals. In a real-world inference test conducted as part of this project, we collected 60 short video clips from friends and acquaintances who had never appeared in the training dataset, preprocessed them identically to the training data, and ran inference using our 3DCNN model that had achieved 92.11% on the signer-joint test set. The model correctly classified only 5 of 60 clips at Top-1 (8.33% accuracy) and 14 of 60 at Top-5 (23.33%). This result, while striking, is not surprising: the model had learned to recognize the training signers, not the signs themselves."),
  body("This test makes clear that any PSL SLR system must be explicitly designed and evaluated for signer independence before it can be considered ready for deployment. The core technical challenge then becomes: how do we build a model that learns sign-discriminating features — motion patterns, hand configurations, temporal dynamics — rather than signer-specific appearance features, given the limited amount of signer-diverse training data available for PSL?"),

  H2("2.4 The Architecture Problem"),
  body("CNN-based architectures trained from random initialization, regardless of how sophisticated their temporal modelling components are, face a fundamental data problem in this setting. Learning what a human hand looks like, how it moves, and which movements correspond to which signs — all from roughly 15 training clips per class — is not a tractable learning problem. These models inevitably fall back on learning the easiest available signal, which in a small signer-inclusive dataset is signer identity. In a signer-disjoint setting, where that shortcut is removed, they converge to near-random performance. This is the root cause of the generalization failure, and it cannot be fixed by architectural refinements alone."),
  body("The solution this project investigates is transfer learning from large pretrained visual models. By starting from a visual encoder already trained on 400 million diverse image-text pairs — and therefore already possessing rich, signer-agnostic representations of human hands in diverse configurations, lighting conditions, and backgrounds — we remove the requirement that the model learn visual fundamentals from PSL data. The PSL-specific training data then only needs to teach the temporal patterns of PSL signs, a much more tractable learning problem with limited data."),
  PB(),
];

// ── SECTION 3: OBJECTIVES ─────────────────────────────────────
const objectivesSection = [
  H1("3. Objectives"),
  body("This project pursues the following concrete objectives, each addressing a specific identified gap in the prior PSL SLR literature:"),
  num("Extend PSL-104 with additional signer diversity by recording 3 new signers across varied environments, lighting conditions, and backgrounds, bringing the total signer count to 6 and enabling genuine signer-disjoint evaluation."),
  num("Scrape, integrate, and preprocess the official PSL Dictionary video data to supplement the team-recorded corpus, ensuring comprehensive coverage of the 104 target sign classes."),
  num("Develop a comprehensive data preprocessing pipeline covering format standardization, temporal noise removal through video cropping, audio stripping, and frame pre-extraction, addressing the data quality issues that contributed to earlier model failures."),
  num("Redesign and extend the prior team's augmentation pipeline from spatial-only transforms to a full spatial, photometric, and temporal augmentation strategy, increasing training data diversity and model robustness."),
  num("Establish and enforce a strictly signer-disjoint evaluation protocol for all experiments, where no signer or augmented variant of their videos appears in more than one dataset split."),
  num("Quantify the generalization gap empirically by evaluating the 3DCNN+ResBlock+BiLSTM architecture under both signer-joint and signer-disjoint conditions, and by conducting a real-world unseen-signer inference test."),
  num("Investigate intermediate approaches to the generalization problem, specifically MediaPipe-based hand ROI extraction and pose-based SLR frameworks, to understand their limitations and motivate the final approach."),
  num("Adapt and train SignVLM (CLIP ViT-L/14 + EVL temporal decoder) on PSL-104 under signer-disjoint evaluation, targeting a validation accuracy that demonstrates practical signer-independent PSL recognition."),
  num("Conduct N-shot ablation experiments to characterize SignVLM's sample efficiency and determine the minimum data requirement for practical accuracy levels."),
  num("Outline a concrete technical roadmap for extending isolated word recognition to sentence-level PSL-to-text translation using NLP-based downstream processing."),
  PB(),
];

module.exports = { coverPage, abstractSection, introSection, problemSection, objectivesSection,
  H1, H2, H3, BR, PB, body, bodyRuns, bull, bull2, num, caption, figPlaceholder,
  tblCaption, hCell, dCell, bCell, grpCell, eq, r, rb, ri, rbi,
  CONTENT, BORDERS, NAVY, BLUE, LGRAY, HBLUE, DBLUE, BORDERS_NONE, bdrS,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign };
