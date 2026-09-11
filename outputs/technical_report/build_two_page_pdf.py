#!/usr/bin/env python3
"""Typeset the compact AIMO technical report as a polished two-page PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "When_Good_Probes_Fail_Xuhang_Ren_submission.pdf"
FIGURE = HERE / "results_overview.png"

PAGE_W, PAGE_H = letter
LEFT = 0.56 * inch
RIGHT = 0.56 * inch
BOTTOM = 0.42 * inch
GAP = 0.18 * inch
USABLE_W = PAGE_W - LEFT - RIGHT
COL_W = (USABLE_W - GAP) / 2


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "PaperBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.65,
            leading=9.2,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#182230"),
            spaceAfter=3.2,
            allowWidows=0,
            allowOrphans=0,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.55,
            leading=9.05,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#182230"),
            leftIndent=5,
            rightIndent=5,
            spaceAfter=4,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=9.4,
            leading=10.8,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#163b65"),
            spaceBefore=4.2,
            spaceAfter=2.4,
            keepWithNext=True,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=6.65,
            leading=7.8,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#334155"),
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=8.2,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceBefore=2,
        ),
    }


STYLES = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def h(number: str, title: str) -> Paragraph:
    label = f"{number}. {title}" if number else title
    return p(label, "heading")


def first_page(canvas, doc) -> None:
    del doc
    canvas.saveState()
    canvas.setTitle("When Good Probes Fail")
    canvas.setAuthor("XUHANG REN")
    canvas.setSubject("AIMO Interpretability Challenge 2026 technical report candidate")
    canvas.setFillColor(colors.HexColor("#102a43"))
    canvas.setFont("Helvetica-Bold", 15.2)
    canvas.drawCentredString(PAGE_W / 2, PAGE_H - 34, "When Good Probes Fail")
    canvas.setFont("Helvetica-Bold", 9.6)
    canvas.drawCentredString(
        PAGE_W / 2,
        PAGE_H - 49,
        "Stress-Testing an Efficient Hidden-State Robustness Predictor",
    )
    canvas.setFillColor(colors.HexColor("#334155"))
    canvas.setFont("Helvetica", 8.0)
    canvas.drawCentredString(
        PAGE_W / 2,
        PAGE_H - 64,
        "XUHANG REN  |  Independent Researcher  |  renxuhang2020@qq.com",
    )
    canvas.setFont("Helvetica-Oblique", 7.2)
    canvas.drawCentredString(
        PAGE_W / 2,
        PAGE_H - 76,
        "AIMO Interpretability Challenge 2026 - Small Models Track - candidate v0.3",
    )
    canvas.setStrokeColor(colors.HexColor("#a9bfd5"))
    canvas.setLineWidth(0.6)
    canvas.line(LEFT, PAGE_H - 84, PAGE_W - RIGHT, PAGE_H - 84)
    draw_footer(canvas, 1)
    canvas.restoreState()


def second_page(canvas, doc) -> None:
    del doc
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#334155"))
    canvas.setFont("Helvetica-Bold", 7.4)
    canvas.drawString(LEFT, PAGE_H - 20, "WHEN GOOD PROBES FAIL")
    canvas.setFont("Helvetica", 7.2)
    canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 20, "XUHANG REN")
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.5)
    canvas.line(LEFT, PAGE_H - 25, PAGE_W - RIGHT, PAGE_H - 25)
    draw_footer(canvas, 2)
    canvas.restoreState()


def draw_footer(canvas, page_number: int) -> None:
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.setFont("Helvetica", 6.8)
    canvas.drawString(LEFT, 16, "Technical report candidate v0.3 - evidence frozen 2026-09-11")
    canvas.drawRightString(PAGE_W - RIGHT, 16, str(page_number))


def result_table() -> Table:
    data = [
        ["Exp.", "Change from V6", "OOF BA", "Decision"],
        ["V6", "Final-token linear vote", "0.7048", "Exploratory pass"],
        ["E6", "Adjacent-layer deltas", "0.7137", "Reject"],
        ["E7", "Mean problem-token state", "0.7186", "Reject"],
        ["E8", "Final problem-token state", "0.6949", "Reject"],
        ["E9", "PCA10 + RBF vote", "0.7126", "Reject"],
        ["E11a", "Multi-view latent drift", "0.6273", "Reject"],
        ["E12", "Counterbalanced metacognition", "0.6363", "Reject"],
    ]
    table = Table(data, colWidths=[25, 125, 43, 64], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163b65")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.25),
                ("LEADING", (0, 0), (-1, -1), 7.2),
                ("ALIGN", (2, 1), (2, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def build_story() -> list:
    story: list = []

    # Page 1, left column.
    story.extend(
        [
            h("", "Abstract"),
            p(
                "We test whether inexpensive hidden-state probes can predict if a "
                "mathematical reasoning model remains correct under meaning-preserving "
                "perturbations. Our V6 submission applies balanced linear probes to nine "
                "fixed layers of DeepSeek-R1-0528-Qwen3-8B and aggregates 225 hard votes. "
                "It achieved 0.705 grouped out-of-fold (OOF) balanced accuracy and 12/19 "
                "Small Track private accuracy with a 4.23 MB bundle. Four registered "
                "refinements produced only small or unstable gains; two mechanism-changing "
                "follow-ups performed near shuffled-label controls. A frozen "
                "audit on ten non-overlapping AIMO problems then found that V6 predicted "
                "every problem robust, scoring 0.20 accuracy against an 0.80 always-negative "
                "baseline. Thus representation separability and a small private-set gain "
                "did not establish OOD reliability. We contribute an auditable low-resource "
                "workflow, controlled negative results, and safeguards for interpretability "
                "research under small, imbalanced datasets.",
                "abstract",
            ),
            h("1", "Motivation"),
            p(
                "The AIMO Interpretability Challenge asks whether a system can distinguish "
                "robust mathematical reasoning from solutions that fail under controlled "
                "counterfactual variation [1]. Probing frozen internal representations is "
                "compute-efficient, but probe accuracy is correlational and may reflect "
                "dataset regularities or probe capacity rather than a mechanism used by the "
                "model [3,4]. We therefore ask whether apparent probe advantage survives "
                "changes in problem source and evaluation distribution."
            ),
            h("2", "Data audit"),
            p(
                "The official aggregate data had 141 rows but only 137 unique problem IDs; "
                "four were exact, label-consistent duplicates. After collapse, labels were "
                "101 non-robust and 36 robust. The official 675-row expanded data still "
                "contained the same 137 problems, so perturbation rows were treated as "
                "repeated measurements rather than independent samples. Every split was "
                "grouped by problem ID, and preprocessing was fit inside training partitions."
            ),
            p(
                "The published binary label was reproduced on all 137 problems by marking a "
                "problem robust only when its maximum absolute accuracy decay across available "
                "perturbations was at most zero. Revisions and canonical content hashes were "
                "frozen before evaluation."
            ),
        ]
    )
    story.append(FrameBreak())

    # Page 1, right column.
    story.extend(
        [
            h("3", "Frozen V6 method"),
            p(
                "For each problem, V6 extracts the final prompt-token state from the frozen "
                "8B checkpoint at layers 4, 8, ..., 36. Each layer receives a balanced L2 "
                "logistic probe with C=0.001 after training-only standardization. Five grouped "
                "folds under five fixed seeds yield 25 fold groups x 9 layers = 225 deployment "
                "votes. The final threshold is a fixed majority vote. The method uses one model "
                "forward pass, no generation or external API during scoring, and a 4.23 MB "
                "submission bundle."
            ),
            h("4", "Validation protocol"),
            p(
                "V6 used repeated grouped OOF validation, both directional source holdouts, a "
                "shuffled-label control, 5,000 problem bootstrap samples, and exact deployment "
                "replay. Because prior exploratory results had been observed, V6 public evidence "
                "is sequentially exploratory, not untouched confirmation."
            ),
            p(
                "E6-E9 each changed one component and were registered before evaluation. "
                "Promotion required at least +0.02 balanced accuracy over V6, a positive paired "
                "bootstrap lower bound, strong performance in each source direction, advantage "
                "over shuffled labels, and nontrivial prediction disagreement. After E10, "
                "E11a tested multi-view latent drift and E12 tested a counterbalanced "
                "robustness/correctness logit readout; both were fail-closed before loading "
                "the untouched AIME/RRB source."
            ),
            h("5", "Public and private results"),
            result_table(),
            Spacer(1, 3),
            p(
                "V6's public confusion was TN=61, FP=40, FN=7, TP=29. Its one-sided bootstrap "
                "95% lower bound was 0.6169 and its advantage over the shuffled-label control "
                "was 0.1189. On 19 private Small Track cases it scored 12/19 with full coverage "
                "and no invalid predictions - the first non-constant private improvement in our "
                "sequence. The sample is too small to establish reliable generalization."
            ),
            p(
                "No successor passed. E6 changed only 2/137 decisions. E7 had the highest OOF "
                "balanced accuracy but asymmetric source transfer (0.612 vs 0.756). E8 weakened "
                "aggregate and transfer evidence. E9 raised ordinary accuracy to 0.708, yet its "
                "paired balanced-accuracy interval crossed zero and one source direction fell to "
                "0.628. E11a was only 0.0337 above its shuffled control. E12 was only 0.0059 "
                "above its control; reversing A/B made correctness margins correlate at "
                "-0.771, exposing answer-position bias. Accuracy and balanced accuracy must "
                "both be reported under the 101:36 imbalance."
            ),
            NextPageTemplate("Second"),
            PageBreak(),
        ]
    )

    # Page 2, full-width figure frame.
    image = Image(str(FIGURE), width=USABLE_W, height=USABLE_W * 470 / 1120)
    story.extend(
        [
            image,
            p(
                "Figure 1. Public grouped validation gives several small apparent gains, but "
                "all registered successors fail promotion and frozen V6 collapses on the "
                "independent ten-problem OOD audit. Error bar: 95% Wilson interval.",
                "caption",
            ),
            FrameBreak(),
        ]
    )

    # Page 2, lower-left column.
    story.extend(
        [
            h("6", "Independent OOD audit"),
            p(
                "E10 evaluated the already frozen V6 artifact on the direct qwen3-8b:low slice "
                "of a separately published ten-problem AIMO sample [2]. Its 64 perturbation rows "
                "yielded eight non-robust and two robust problem labels. The label rule reproduced "
                "all 137 training labels, and overlap was zero by ID and normalized text. Labels "
                "were never supplied to the model or probe."
            ),
            p(
                "Before extraction, we pinned dataset, model, and artifact hashes and required "
                "at least eight two-class problems, accuracy >= 0.60, and >= +0.10 over the best "
                "constant predictor. V6 predicted all ten cases robust. Accuracy was 0.20, "
                "balanced accuracy 0.50, and the 95% Wilson interval [0.057, 0.510]; always-negative "
                "accuracy was 0.80. Median robust-vote fraction was 0.94 and the minimum 0.769, "
                "indicating confident OOD error rather than a small threshold mismatch."
            ),
            p(
                "The audit failed by its registered rule. We did not tune on individual E10 "
                "cases and did not create V8."
            ),
            h("7", "Actionable lessons"),
            p(
                "<b>1.</b> Split at the causal unit: variants of one problem are repeated "
                "measurements. <b>2.</b> Random-label controls are necessary but insufficient. "
                "<b>3.</b> Small gains need paired evidence. <b>4.</b> Source-direction symmetry "
                "is an early warning. <b>5.</b> Confidence under shift distinguishes a threshold "
                "near-miss from extrapolation failure. <b>6.</b> Counterbalance elicited "
                "judgments to reveal position bias. <b>7.</b> Pre-registered stopping rules "
                "turn tempting marginal gains into useful negative results."
            ),
        ]
    )
    story.append(FrameBreak())

    # Page 2, lower-right column.
    story.extend(
        [
            h("8", "Limitations and claim boundary"),
            p(
                "Training used 137 problems from two related MATH sources; private and OOD "
                "samples contained only 19 and ten cases. E10 tests problem transfer for one 8B "
                "checkpoint, not all models or perturbations. Linear separability does not show "
                "that the model uses the probed features causally. E11a/E12 failed before their "
                "independent Stage B. Our defensible claim is negative: final-token states "
                "contain in-distribution robustness signal, but neither this probe family nor "
                "the tested elicited confidence generalizes as a detector of robust reasoning."
            ),
            h("9", "Reproducibility and resources"),
            p(
                "Inputs, registrations, artifacts, and result JSONs are hash-pinned. Activation "
                "extraction used free Kaggle T4 x 2 sessions; probe training, bootstrap, packaging, "
                "and audit ran locally on CPU, incurring zero direct monetary compute cost. Any "
                "future model must start from a distinct mechanistic hypothesis and an untouched "
                "validation source; V6 private outcomes and E10 are frozen against tuning."
            ),
            p(
                "Public code and frozen evidence: "
                "<link href='https://github.com/renxuhangrt-dotcom/aimo-interp-probe-stress-tests' "
                "color='#163b65'>github.com/renxuhangrt-dotcom/<br/>"
                "aimo-interp-probe-stress-tests</link>",
                "small",
            ),
            h("10", "Conclusion"),
            p(
                "An efficient probe showed meaningful public and small-private-set signal, but "
                "refinements failed strict gates, frozen V6 collapsed on a new AIMO source, and "
                "two new mechanisms performed near randomized controls. The gap between "
                "separability, elicited confidence, and transfer is the main result."
            ),
            h("", "References"),
            p(
                "[1] Stefanik et al. <i>AIMO Interpretability Challenge.</i> arXiv:2607.13899, "
                "2026. https://arxiv.org/abs/2607.13899<br/>"
                "[2] AIMO challenge sample-full dataset. https://huggingface.co/datasets/"
                "aimo-interp/aimo-interp-challenge-sample-full<br/>"
                "[3] Belinkov. <i>Probing Classifiers: Promises, Shortcomings, and Advances.</i> "
                "Computational Linguistics, 2022. https://aclanthology.org/2022.cl-1.7/<br/>"
                "[4] Hewitt and Liang. <i>Designing and Interpreting Probes with Control Tasks.</i> "
                "EMNLP-IJCNLP, 2019. https://aclanthology.org/D19-1275/",
                "small",
            ),
        ]
    )
    return story


def build() -> Path:
    if not FIGURE.is_file():
        raise FileNotFoundError(FIGURE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    page1_top = PAGE_H - 91
    page1_frames = [
        Frame(LEFT, BOTTOM, COL_W, page1_top - BOTTOM, id="p1-left", showBoundary=0),
        Frame(LEFT + COL_W + GAP, BOTTOM, COL_W, page1_top - BOTTOM, id="p1-right", showBoundary=0),
    ]

    # Reserve enough room for the image, caption, and the frame's internal padding.
    # If this is too short, ReportLab moves the caption into the first text column
    # and every subsequent explicit FrameBreak shifts by one frame.
    figure_height = USABLE_W * 470 / 1120 + 45
    page2_top = PAGE_H - 31
    lower_top = page2_top - figure_height - 7
    page2_frames = [
        Frame(LEFT, lower_top + 7, USABLE_W, figure_height, id="p2-figure", showBoundary=0),
        Frame(LEFT, BOTTOM, COL_W, lower_top - BOTTOM, id="p2-left", showBoundary=0),
        Frame(LEFT + COL_W + GAP, BOTTOM, COL_W, lower_top - BOTTOM, id="p2-right", showBoundary=0),
    ]

    document = BaseDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=LEFT,
        rightMargin=RIGHT,
        topMargin=0,
        bottomMargin=BOTTOM,
        title="When Good Probes Fail",
        author="XUHANG REN",
        subject="AIMO Interpretability Challenge 2026 technical report candidate",
    )
    document.addPageTemplates(
        [
            PageTemplate(id="First", frames=page1_frames, onPage=first_page),
            PageTemplate(id="Second", frames=page2_frames, onPage=second_page),
        ]
    )
    document.build(build_story())
    return OUTPUT_PDF


if __name__ == "__main__":
    print(build())
