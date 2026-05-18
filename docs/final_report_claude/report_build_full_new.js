const {
  Document, Packer, Paragraph, TextRun,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, Footer
} = require('docx');
const fs = require('fs');

const {
  coverPage, abstractSection, introSection, problemSection, objectivesSection,
  r, rb, ri, BR, PB, NAVY, BLUE, CONTENT
} = require('./report_part1');

const { litReviewSection } = require('./report_part2');
const { methodologySection } = require('./report_part3');
const { implementationSection, resultsSection } = require('./report_part5');

const { demoSection, conclusionSection, futureWorkSection, referencesSection } = require('./report_part6');

const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets",  levels: [{ level: 0, format: LevelFormat.BULLET,  text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720,  hanging: 360 } } } }] },
      { reference: "bullets2", levels: [{ level: 0, format: LevelFormat.BULLET,  text: "\u25E6", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }] },
      { reference: "numbers",  levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720,  hanging: 360 } } } }] },
    ]
  },
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24, color: "000000" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        run: { font: "Georgia", size: 32, bold: true, color: NAVY },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        run: { font: "Georgia", size: 26, bold: true, color: NAVY },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal",
        run: { font: "Times New Roman", size: 24, bold: true, color: "2E4057" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ]
  },
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children: coverPage
    },
    {
      properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      footers: {
        default: new Footer({ children: [new Paragraph({
          children: [
            new TextRun({ text: "PSL Recognizer: Empowering Communication through AI  |  F25-115  |  ", font: "Times New Roman", size: 18, color: "777777" }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 18, color: "777777" })
          ],
          alignment: AlignmentType.CENTER
        })] })
      },
      children: [
        ...abstractSection,
        ...introSection,
        ...problemSection,
        ...objectivesSection,
        ...litReviewSection,
        ...methodologySection,
        ...implementationSection,
        ...resultsSection,
        ...demoSection,
        ...conclusionSection,
        ...futureWorkSection,
        ...referencesSection,
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/mnt/user-data/outputs/PSL_Report_Full_Draft.docx", buf);
  console.log("Done.");
}).catch(err => { console.error(err); process.exit(1); });
