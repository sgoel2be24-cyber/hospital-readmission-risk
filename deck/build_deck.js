/**
 * ML Bubble 2026 — submission deck generator.
 *
 *   node deck/build_deck.js
 *
 * Every number here is copied from reports/metrics.json, model_comparison.csv,
 * ablation_study.csv and fairness_report.csv. Nothing is illustrative.
 */

const PptxGenJS = require("pptxgenjs");
const path = require("path");

// ---------------------------------------------------------------- palette --
const INK = "0B2E3B";      // deep petrol — dominant dark
const DEEP = "12556B";     // mid teal
const TEAL = "2E8FA8";     // light teal
const CORAL = "F2634A";    // accent: risk / flagged
const PAPER = "FFFFFF";
const MIST = "EEF4F6";     // card fill on light slides
const MUTED = "5A7683";
const CHALK = "C9DCE3";    // body text on dark

const HEAD = "Cambria";
const BODY = "Calibri";

const W = 13.333;
const H = 7.5;
const M = 0.62;            // page margin
const CW = W - M * 2;      // content width = 12.093

const REPORTS = path.join(__dirname, "..", "reports", "figures");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Shikhar";
pptx.company = "ML Bubble 2026";
pptx.title = "Hospital 30-Day Readmission Risk Prediction";

// ---------------------------------------------------------------- helpers --
function darkSlide() {
  const s = pptx.addSlide();
  s.background = { color: INK };
  return s;
}

function lightSlide(title, kicker) {
  const s = pptx.addSlide();
  s.background = { color: PAPER };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.34, w: CW, h: 0.26,
      fontFace: BODY, fontSize: 11, bold: true, color: TEAL, charSpacing: 2, margin: 0,
    });
  }
  s.addText(title, {
    x: M, y: kicker ? 0.62 : 0.44, w: CW, h: 0.72,
    fontFace: HEAD, fontSize: 32, bold: true, color: INK, margin: 0, valign: "top",
  });
  return s;
}

/** Rounded card. Returns nothing; add text separately so padding stays explicit. */
function card(s, x, y, w, h, fill) {
  s.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || MIST }, line: { color: fill || MIST, width: 0 },
  });
}

/** Big number + label, left aligned inside a card. */
function stat(s, x, y, w, value, label, valueColor, valueSize) {
  s.addText(value, {
    x, y, w, h: 0.74,
    fontFace: HEAD, fontSize: valueSize || 40, bold: true,
    color: valueColor || INK, margin: 0, valign: "middle",
  });
  s.addText(label, {
    x, y: y + 0.74, w, h: 0.52,
    fontFace: BODY, fontSize: 11.5, color: MUTED, margin: 0, valign: "top",
  });
}

/** Numbered step chip used by the pipeline slide. */
function chip(s, x, y, d, n, color) {
  s.addShape(pptx.ShapeType.ellipse, {
    x, y, w: d, h: d, fill: { color }, line: { color, width: 0 },
  });
  s.addText(String(n), {
    x, y, w: d, h: d,
    fontFace: HEAD, fontSize: 14, bold: true, color: PAPER,
    align: "center", valign: "middle", margin: 0,
  });
}

function footer(s, text) {
  // Bottom edge lands at exactly H - 0.5, holding the 0.5" page margin.
  s.addText(text, {
    x: M, y: H - 0.76, w: CW, h: 0.26,
    fontFace: BODY, fontSize: 9.5, color: "9AB0B9", italic: true, margin: 0,
  });
}

// ============================================================ 1. title =====
{
  const s = darkSlide();
  s.addShape(pptx.ShapeType.ellipse, {
    x: 9.5, y: -1.9, w: 6.4, h: 6.4,
    fill: { color: DEEP, transparency: 45 }, line: { width: 0 },
  });
  s.addShape(pptx.ShapeType.ellipse, {
    x: 11.0, y: 3.6, w: 4.2, h: 4.2,
    fill: { color: TEAL, transparency: 70 }, line: { width: 0 },
  });

  s.addText("ML BUBBLE 2026  ·  TE-BE DESIGN & SOLVE", {
    x: M, y: 1.42, w: 9.2, h: 0.3,
    fontFace: BODY, fontSize: 12, bold: true, color: TEAL, charSpacing: 2.4, margin: 0,
  });
  s.addText("Predicting 30-Day\nHospital Readmission", {
    x: M, y: 1.95, w: 9.0, h: 1.95,
    fontFace: HEAD, fontSize: 46, bold: true, color: PAPER, lineSpacing: 50, margin: 0,
  });
  s.addText(
    "Turning 101,766 discharge records into a ranked call list for a follow-up team that can only reach one patient in five.",
    { x: M, y: 4.12, w: 8.5, h: 0.8, fontFace: BODY, fontSize: 15, color: CHALK, margin: 0 }
  );

  const items = [
    ["40.2%", "of readmissions caught"],
    ["2.05×", "lift over random calling"],
    ["0.238", "PR-AUC vs 0.113 base rate"],
  ];
  items.forEach(([v, l], i) => {
    const x = M + i * 3.05;
    s.addText(v, {
      x, y: 5.28, w: 2.8, h: 0.6,
      fontFace: HEAD, fontSize: 30, bold: true, color: CORAL, margin: 0, valign: "middle",
    });
    s.addText(l, {
      x, y: 5.86, w: 2.8, h: 0.42,
      fontFace: BODY, fontSize: 11, color: CHALK, margin: 0,
    });
  });

  s.addText("Held-out test set · 19,773 encounters · 13,998 patients unseen in training", {
    x: M, y: 6.66, w: 11.0, h: 0.3,
    fontFace: BODY, fontSize: 10.5, color: "7E9AA5", italic: true, margin: 0,
  });
  s.addNotes(
    "The framing is the point: this is not 'who will be readmitted' but 'given we can call 20% of discharges, which 20%'. " +
    "Every modelling decision follows from that. Headline: 40.2% of readmissions caught by calling the top-ranked fifth."
  );
}

// ========================================================= 2. the problem ==
{
  const s = lightSlide("A capacity problem, not a prediction problem", "The question");
  card(s, M, 1.72, 5.62, 4.42, MIST);
  s.addText("What a hospital actually faces", {
    x: M + 0.34, y: 1.98, w: 4.94, h: 0.34,
    fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0,
  });
  s.addText(
    [
      { text: "About 1 in 9 diabetic discharges returns within 30 days.", options: { bullet: true, breakLine: true } },
      { text: "Many are preventable — a phone call, a medication check, an earlier appointment.", options: { bullet: true, breakLine: true } },
      { text: "A discharge-planning team can reach only a fraction of patients per day.", options: { bullet: true, breakLine: true } },
      { text: "So the useful question is which patients get that scarce attention.", options: { bullet: true } },
    ],
    {
      x: M + 0.34, y: 2.44, w: 4.94, h: 3.4,
      fontFace: BODY, fontSize: 14, color: "1F3D49", margin: 0,
      paraSpaceAfter: 10, lineSpacing: 20,
    }
  );

  s.addText("Reframed", {
    x: 6.62, y: 1.86, w: 6.1, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: TEAL, charSpacing: 2, margin: 0,
  });
  s.addText("“We can call 20% of\ntoday's discharges.\nWhich 20%?”", {
    x: 6.62, y: 2.24, w: 6.1, h: 1.72,
    fontFace: HEAD, fontSize: 27, bold: true, color: DEEP, lineSpacing: 34, margin: 0,
  });
  s.addText(
    "That reframing changes what counts as success. The team works down a ranked list, so ranking quality matters more than absolute probability — and the decision threshold comes from budget, not from an arbitrary 0.5 cutoff.",
    { x: 6.62, y: 4.06, w: 6.1, h: 1.1, fontFace: BODY, fontSize: 13.5, color: "1F3D49", margin: 0, lineSpacing: 19 }
  );

  card(s, 6.62, 5.24, 6.1, 0.9, "FBE4E0");
  s.addText("Consequence: the model is optimised for PR-AUC and recall-at-capacity — never accuracy.", {
    x: 6.9, y: 5.24, w: 5.54, h: 0.9,
    fontFace: BODY, fontSize: 12.5, bold: true, color: "9E3423", margin: 0, valign: "middle",
  });
  s.addNotes("Everything downstream — metric choice, threshold, calibration — is a consequence of this reframing.");
}

// ============================================================ 3. dataset ===
{
  const s = lightSlide("Diabetes 130-US Hospitals, 1999–2008", "The data");
  s.addText("UCI ML Repository dataset 296 · 10 years of care across 130 hospitals · downloaded reproducibly via make data", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  const stats = [
    ["101,766", "raw encounters", INK],
    ["71,518", "distinct patients", INK],
    ["50", "raw columns", INK],
    ["11.4%", "readmitted <30 days", CORAL],
  ];
  stats.forEach(([v, l, c], i) => {
    const x = M + i * 3.06;
    card(s, x, 1.9, 2.84, 1.42, MIST);
    stat(s, x + 0.26, 1.98, 2.4, v, l, c, 26);
  });

  s.addText("Target definition", {
    x: M, y: 3.62, w: 5.7, h: 0.32,
    fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0,
  });
  s.addText(
    [
      { text: "readmitted has three values: <30, >30, NO", options: { bullet: true, breakLine: true } },
      { text: "Binarised to <30 vs everything else — the window used by readmission-penalty programmes", options: { bullet: true, breakLine: true } },
      { text: "A return 14 months later is not a discharge-planning failure, so >30 is a negative", options: { bullet: true } },
    ],
    { x: M, y: 4.04, w: 5.7, h: 1.66, fontFace: BODY, fontSize: 13, color: "1F3D49", margin: 0, paraSpaceAfter: 8, lineSpacing: 18 }
  );

  s.addText("Cleaning that changes the label", {
    x: 6.62, y: 3.62, w: 6.1, h: 0.32,
    fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0,
  });
  s.addText(
    [
      { text: "2,423 death / hospice discharges removed — those patients cannot be readmitted, and labelling them 'not readmitted' teaches the model that dying is a good outcome", options: { bullet: true, breakLine: true } },
      { text: "weight dropped (97% missing); examide and citoglipton are constant", options: { bullet: true, breakLine: true } },
      { text: "'?' kept as an explicit Unknown category — missingness here is administrative, and it carries signal", options: { bullet: true } },
    ],
    { x: 6.62, y: 4.04, w: 6.1, h: 1.9, fontFace: BODY, fontSize: 13, color: "1F3D49", margin: 0, paraSpaceAfter: 8, lineSpacing: 18 }
  );

  footer(s, "99,340 encounters and 39 engineered features survive to modelling.");
  s.addNotes(
    "The death/hospice filter is the one cleaning step that changes the label rather than the features. " +
    "Also worth mentioning: max_glu_serum uses the literal string 'None' for 'never tested' — pandas turns that into NaN by default and destroys an informative level."
  );
}

// ======================================================== 4. leakage trap ==
{
  const s = lightSlide("99,340 encounters belong to only 69,987 patients", "The trap");
  s.addText("The single decision that separates an honest result from a meaningless one.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 13, color: MUTED, margin: 0,
  });

  card(s, M, 1.94, 5.86, 3.34, "FBE4E0");
  s.addText("✕   Split by row", {
    x: M + 0.34, y: 2.18, w: 5.18, h: 0.4,
    fontFace: HEAD, fontSize: 18, bold: true, color: "9E3423", margin: 0,
  });
  s.addText(
    [
      { text: "The same patient lands in train and test", options: { bullet: true, breakLine: true } },
      { text: "The model memorises individuals instead of learning transferable risk", options: { bullet: true, breakLine: true } },
      { text: "Every metric comes out inflated", options: { bullet: true, breakLine: true } },
      { text: "One patient appears 40 times", options: { bullet: true } },
    ],
    { x: M + 0.34, y: 2.66, w: 5.18, h: 2.3, fontFace: BODY, fontSize: 13.5, color: "7A2A1C", margin: 0, paraSpaceAfter: 8, lineSpacing: 19 }
  );

  card(s, 6.85, 1.94, 5.86, 3.34, "E2F0E9");
  s.addText("✓   Split by patient_nbr", {
    x: 7.19, y: 2.18, w: 5.18, h: 0.4,
    fontFace: HEAD, fontSize: 18, bold: true, color: "1F5F44", margin: 0,
  });
  s.addText(
    [
      { text: "GroupShuffleSplit for train / val / test", options: { bullet: true, breakLine: true } },
      { text: "GroupKFold for every cross-validation", options: { bullet: true, breakLine: true } },
      { text: "No patient can appear on both sides", options: { bullet: true, breakLine: true } },
      { text: "Enforced by a test, not by convention", options: { bullet: true } },
    ],
    { x: 7.19, y: 2.66, w: 5.18, h: 2.3, fontFace: BODY, fontSize: 13.5, color: "1D4D39", margin: 0, paraSpaceAfter: 8, lineSpacing: 19 }
  );

  card(s, M, 5.34, CW, 1.06, INK);
  s.addText("test_split_never_puts_a_patient_on_both_sides", {
    x: M + 0.36, y: 5.5, w: 6.6, h: 0.36,
    fontFace: "Courier New", fontSize: 13, bold: true, color: "8FD4C8", margin: 0,
  });
  s.addText("Two further leakage paths are closed the same way: identifiers dropped after splitting, unscoreable outcomes removed.", {
    x: M + 0.36, y: 5.86, w: 11.2, h: 0.36,
    fontFace: BODY, fontSize: 12, color: CHALK, margin: 0,
  });

  footer(s, "A materially higher ROC-AUC on this dataset almost always means a leaked split rather than a better model.");
  s.addNotes(
    "This is the slide to linger on. Published results for this dataset sit at 0.64–0.70. If someone reports 0.85, they split by row. " +
    "Our three leakage guards are each covered by a test that fails the build."
  );
}

// =========================================================== 5. pipeline ===
{
  const s = lightSlide("From raw encounter to ranked call list", "Method");

  const steps = [
    ["Clean", "Remove death/hospice\ndischarges. Keep '?' as\nan explicit category."],
    ["Engineer", "2,000+ ICD-9 codes into\n9 chapters. Utilisation\nand regimen-churn features."],
    ["Split", "GroupShuffleSplit on\npatient_nbr. Train / val /\ntest = 63.6k / 16.0k / 19.8k."],
    ["Compare", "6 models under 5-fold\nGroupKFold. Winner picked\non validation PR-AUC."],
    ["Calibrate", "Platt scaling, then a\nthreshold set from\nfollow-up capacity."],
  ];
  const cw = 2.28, gap = 0.19;
  steps.forEach(([t, d], i) => {
    const x = M + i * (cw + gap);
    card(s, x, 1.86, cw, 2.66, MIST);
    chip(s, x + 0.24, 2.08, 0.46, i + 1, i === 4 ? CORAL : DEEP);
    s.addText(t, {
      x: x + 0.24, y: 2.66, w: cw - 0.48, h: 0.34,
      fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0,
    });
    s.addText(d, {
      x: x + 0.24, y: 3.04, w: cw - 0.48, h: 1.6,
      fontFace: BODY, fontSize: 11.5, color: "34525E", margin: 0, lineSpacing: 16,
    });
  });

  card(s, M, 5.12, CW, 1.36, INK);
  s.addText("The test set is read exactly once, at the very end.", {
    x: M + 0.36, y: 5.3, w: 11.4, h: 0.36,
    fontFace: HEAD, fontSize: 16, bold: true, color: PAPER, margin: 0,
  });
  s.addText(
    "Model choice, hyperparameters, the calibration method and the decision threshold are all fixed against validation. That discipline is what makes 0.684 an estimate of future performance rather than the best of a dozen peeks.",
    { x: M + 0.36, y: 5.7, w: 11.4, h: 0.62, fontFace: BODY, fontSize: 12.5, color: CHALK, margin: 0, lineSpacing: 17 }
  );
  s.addNotes("Reproducible end to end: make setup, make data, make tune, make train, make experiments, make explain, make test.");
}

// =================================================== 6. imbalance handling ==
{
  const s = lightSlide("11.4% positives — handled explicitly, not hopefully", "Class imbalance");
  s.addText("Left alone, every model collapses to predicting 'no readmission' for everyone. That scores 88.6% accuracy and catches nothing.", {
    x: M, y: 1.44, w: CW, h: 0.34, fontFace: BODY, fontSize: 13, color: MUTED, margin: 0,
  });

  const rows = [
    ["Logistic regression", "class_weight='balanced'", DEEP],
    ["Random forest", "class_weight='balanced_subsample'", DEEP],
    ["XGBoost", "scale_pos_weight ≈ 7.8", DEEP],
    ["LightGBM", "class_weight='balanced'", CORAL],
    ["MLP", "none available — judged on ranking", MUTED],
  ];
  rows.forEach(([name, mech, c], i) => {
    const y = 1.94 + i * 0.62;
    card(s, M, y, 6.3, 0.52, MIST);
    s.addShape(pptx.ShapeType.ellipse, { x: M + 0.2, y: y + 0.15, w: 0.22, h: 0.22, fill: { color: c }, line: { width: 0 } });
    s.addText(name, {
      x: M + 0.56, y, w: 2.2, h: 0.52,
      fontFace: BODY, fontSize: 12.5, bold: true, color: INK, margin: 0, valign: "middle",
    });
    s.addText(mech, {
      x: M + 2.72, y, w: 3.44, h: 0.52,
      fontFace: "Courier New", fontSize: 10, color: "34525E", margin: 0, valign: "middle",
    });
  });

  card(s, 7.24, 1.94, 5.48, 2.34, "E2F0E9");
  s.addText("SMOTE was measured, not assumed away", {
    x: 7.56, y: 2.16, w: 4.84, h: 0.36,
    fontFace: HEAD, fontSize: 15, bold: true, color: "1F5F44", margin: 0,
  });
  s.addText(
    "Oversampling inside each CV fold scored 0.2194 PR-AUC against 0.2200 for class weighting — no better, and it adds a resampling step to the serving path.\n\nClass weighting kept on the evidence.",
    { x: 7.56, y: 2.58, w: 4.84, h: 1.5, fontFace: BODY, fontSize: 12.5, color: "1D4D39", margin: 0, lineSpacing: 17 }
  );

  card(s, 7.24, 4.44, 5.48, 1.72, "FBE4E0");
  s.addText("The cost of class weighting", {
    x: 7.56, y: 4.64, w: 4.84, h: 0.34,
    fontFace: HEAD, fontSize: 15, bold: true, color: "9E3423", margin: 0,
  });
  s.addText(
    "It deliberately distorts predicted probabilities upward. Fine for ranking, wrong for a number a clinician reads — which is why calibration is a separate step.",
    { x: 7.56, y: 5.02, w: 4.84, h: 1.0, fontFace: BODY, fontSize: 12.5, color: "7A2A1C", margin: 0, lineSpacing: 17 }
  );
  s.addNotes("The MLP exception is informative: because it is unweighted, it ends up the best-calibrated model while ranking worst.");
}

// ================================================== 7. comparative analysis =
{
  const s = lightSlide("Six models, one selection rule", "Comparative analysis");
  s.addText("5-fold GroupKFold on training patients for selection; a single held-out test evaluation at the end.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  s.addChart(
    pptx.ChartType.bar,
    [
      {
        name: "Test PR-AUC",
        labels: ["Dummy", "Logistic reg.", "Random forest", "XGBoost", "LightGBM", "MLP"],
        values: [0.114, 0.213, 0.225, 0.235, 0.238, 0.222],
      },
    ],
    {
      x: M, y: 1.9, w: 6.5, h: 4.24,
      barDir: "col",
      chartColors: [DEEP, DEEP, DEEP, DEEP, CORAL, DEEP],
      varyColors: true,
      showTitle: true, title: "Test PR-AUC  (base rate 0.113)",
      titleFontSize: 13, titleColor: INK, titleFontFace: BODY,
      showValue: true, dataLabelPosition: "outEnd",
      dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFormatCode: "0.000",
      showLegend: false,
      valAxisMinVal: 0, valAxisMaxVal: 0.28,
      catAxisLabelColor: MUTED, catAxisLabelFontSize: 10,
      valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
      valGridLine: { color: "E3EBEE", size: 1 },
      catGridLine: { style: "none" },
    }
  );

  card(s, 7.44, 1.9, 5.28, 2.08, MIST);
  s.addText("Selected: LightGBM", {
    x: 7.74, y: 2.08, w: 4.68, h: 0.36,
    fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "Test PR-AUC 0.238 · ROC-AUC 0.684 · recall 40.2% at 20% capacity. Chosen on validation PR-AUC, before the test set was read.",
    { x: 7.74, y: 2.5, w: 4.68, h: 1.2, fontFace: BODY, fontSize: 12.5, color: "34525E", margin: 0, lineSpacing: 17 }
  );

  card(s, 7.44, 4.14, 5.28, 2.0, "FFF3E6");
  s.addText("Which gaps are real?", {
    x: 7.74, y: 4.32, w: 4.68, h: 0.32,
    fontFace: HEAD, fontSize: 15, bold: true, color: "8A5217", margin: 0,
  });
  s.addText(
    "Paired bootstrap on \u0394 PR-AUC:\n\nvs XGBoost   +0.0022  [\u22120.0017, +0.0058]   not resolved\nvs Random forest   +0.0124  [+0.0064, +0.0189]   resolved\nvs Logistic reg.   +0.0249  [+0.0177, +0.0321]   resolved",
    { x: 7.74, y: 4.7, w: 4.68, h: 1.3, fontFace: BODY, fontSize: 10.5, color: "6B4213", margin: 0, lineSpacing: 14 }
  );

  footer(s, "Boosting genuinely beats bagging and linear models. Which booster wins is not resolved by this data.");
  s.addNotes(
    "If asked why LightGBM over XGBoost: validation PR-AUC, and it is genuinely a coin flip. Saying so is more credible than claiming a 0.0003 win."
  );
}

// ======================================================= 8. accuracy trap ==
{
  const s = darkSlide();
  s.addText("THE METRIC TRAP", {
    x: M, y: 0.72, w: CW, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: TEAL, charSpacing: 2.4, margin: 0,
  });
  s.addText("Our model is 78.2% accurate.\nA model that does nothing is 88.6% accurate.", {
    x: M, y: 1.14, w: 11.6, h: 1.5,
    fontFace: HEAD, fontSize: 33, bold: true, color: PAPER, lineSpacing: 44, margin: 0,
  });

  const cols = [
    ["78.2%", "our model", "40.2% of readmissions caught", CORAL],
    ["88.6%", "“nobody is readmitted”", "0% caught — flags no one", "6E8B96"],
    ["88.8%", "best accuracy at any threshold", "flags 0.4% of patients", "6E8B96"],
  ];
  cols.forEach(([v, l, d, c], i) => {
    const x = M + i * 4.06;
    s.addShape(pptx.ShapeType.roundRect, {
      x, y: 3.0, w: 3.8, h: 2.1, rectRadius: 0.08,
      fill: { color: "16404F" }, line: { width: 0 },
    });
    s.addText(v, {
      x: x + 0.3, y: 3.2, w: 3.2, h: 0.72,
      fontFace: HEAD, fontSize: 34, bold: true, color: c, margin: 0, valign: "middle",
    });
    s.addText(l, {
      x: x + 0.3, y: 3.94, w: 3.2, h: 0.34,
      fontFace: BODY, fontSize: 12.5, bold: true, color: PAPER, margin: 0,
    });
    s.addText(d, {
      x: x + 0.3, y: 4.3, w: 3.2, h: 0.62,
      fontFace: BODY, fontSize: 11.5, color: CHALK, margin: 0, lineSpacing: 15,
    });
  });

  s.addText(
    "With an 11% positive rate, accuracy rewards inaction. Optimising it means optimising toward a model that never flags anyone — so nothing in this project was selected using it.",
    { x: M, y: 5.44, w: 11.6, h: 0.66, fontFace: BODY, fontSize: 14, color: CHALK, margin: 0, lineSpacing: 19 }
  );
  s.addText("This is also why the deployed threshold is 0.166, not 0.5 — at 0.5 the calibrated model flags zero patients.", {
    x: M, y: 6.2, w: 11.6, h: 0.36,
    fontFace: BODY, fontSize: 12.5, italic: true, color: TEAL, margin: 0,
  });
  s.addNotes(
    "Expect the accuracy question from judges. Answer with the 88.6% baseline. Accuracy is reported in metrics.json for completeness; it selected nothing."
  );
}

// ========================================================= 9. calibration ==
{
  const s = lightSlide("Making the number mean what it says", "Calibration");
  s.addText("Class weighting is what makes ranking work and probabilities wrong. Platt scaling fixes the probabilities without touching the ranking.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  s.addImage({ path: path.join(REPORTS, "calibration.png"), x: M, y: 1.9, w: 4.5, h: 4.24 });

  const t = [
    ["", "Brier", "ROC-AUC", "PR-AUC"],
    ["LightGBM, raw", "0.2007", "0.6839", "0.2376"],
    ["+ Platt scaling", "0.0954", "0.6839", "0.2376"],
  ];
  s.addTable(t, {
    x: 5.36, y: 2.06, w: 7.36,
    colW: [2.5, 1.62, 1.62, 1.62],
    rowH: [0.36, 0.42, 0.42],
    fontFace: BODY, fontSize: 12.5, color: "1F3D49",
    border: { type: "solid", color: "DCE7EB", pt: 1 },
    fill: { color: "FFFFFF" },
  });

  s.addText("Brier more than halves. Discrimination is bit-identical.", {
    x: 5.36, y: 3.42, w: 7.36, h: 0.34,
    fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "Platt scaling is strictly monotonic, so it cannot reorder a single patient — the call list is unchanged while the risk numbers become trustworthy. A predicted 18% now means roughly 18%.",
    { x: 5.36, y: 3.82, w: 7.36, h: 0.86, fontFace: BODY, fontSize: 13, color: "34525E", margin: 0, lineSpacing: 18 }
  );

  card(s, 5.36, 4.82, 7.36, 1.32, MIST);
  s.addText("Why not isotonic?", {
    x: 5.66, y: 4.98, w: 6.76, h: 0.32,
    fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "It won validation Brier by 0.0005 but is a step function — it ties scores together and cost 0.004 PR-AUC. The rule: among calibrators within 0.002 Brier of the best, take the one that ranks better.",
    { x: 5.66, y: 5.32, w: 6.76, h: 0.74, fontFace: BODY, fontSize: 12, color: "34525E", margin: 0, lineSpacing: 16 }
  );
  s.addNotes("The uncalibrated curves sitting below the diagonal are the class-weighting distortion made visible.");
}

// ==================================================== 10. operating point ==
{
  const s = lightSlide("The threshold is a budget decision", "Operating point");
  s.addText("Precision and recall as a function of how many discharges the follow-up team can call.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  s.addImage({ path: path.join(REPORTS, "capacity_sweep.png"), x: M, y: 1.88, w: 6.05, h: 4.26 });

  const rows = [
    ["Capacity", "Recall", "Precision"],
    ["10%", "24.4%", "27.7%"],
    ["20%  ← deployed", "40.6%", "23.1%"],
    ["30%", "52.9%", "20.0%"],
    ["40%", "63.9%", "18.1%"],
    ["50%", "72.1%", "16.4%"],
  ];
  s.addTable(rows, {
    x: 7.06, y: 1.98, w: 5.66,
    colW: [2.42, 1.62, 1.62],
    rowH: [0.34, 0.34, 0.34, 0.34, 0.34, 0.34],
    fontFace: BODY, fontSize: 12, color: "1F3D49",
    border: { type: "solid", color: "DCE7EB", pt: 1 },
    fill: { color: "FFFFFF" },
  });

  s.addText("Doubling capacity from 20% to 40% buys 23 more points of recall for 5 points of precision.", {
    x: 7.06, y: 4.12, w: 5.66, h: 0.62,
    fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0, lineSpacing: 19,
  });
  s.addText(
    "Whether that trade is worth taking depends on the cost of a call against the cost of a readmission — a hospital finance question. The model supports either answer without retraining.",
    { x: 7.06, y: 4.84, w: 5.66, h: 0.92, fontFace: BODY, fontSize: 12.5, color: "34525E", margin: 0, lineSpacing: 17 }
  );

  card(s, 7.06, 5.82, 5.66, 0.62, "FBE4E0");
  s.addText("At 20%: 901 true positives, 2,965 false alarms, 1,343 missed.", {
    x: 7.3, y: 5.82, w: 5.18, h: 0.62,
    fontFace: BODY, fontSize: 12, bold: true, color: "9E3423", margin: 0, valign: "middle",
  });
  s.addNotes("Capacity-sweep rows use test-quantile thresholds, so they differ marginally from the deployed 40.2%/23.3%, where the threshold is fixed on validation.");
}

// ==================================================== 11. decision curve ===
{
  const s = lightSlide("Is acting on it better than what they do today?", "Clinical value");
  s.addText("Discrimination says the ranking is good. It does not say that using it beats calling everyone, or calling nobody.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  s.addImage({ path: path.join(REPORTS, "decision_curve.png"), x: M, y: 1.86, w: 6.35, h: 4.32 });

  card(s, 7.3, 1.9, 5.42, 1.44, MIST);
  s.addText("Decision curve analysis", {
    x: 7.58, y: 2.06, w: 4.86, h: 0.32,
    fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "net benefit = TP/N − (FP/N) × pt/(1−pt)\n\npt is the risk at which a call becomes worthwhile — it encodes the cost ratio.",
    { x: 7.58, y: 2.42, w: 4.86, h: 0.86, fontFace: BODY, fontSize: 11.5, color: "34525E", margin: 0, lineSpacing: 15 }
  );

  const rows = [
    ["Policy", "Net benefit"],
    ["Use the model", "+0.0157"],
    ["Call everyone", "−0.0617"],
    ["Call nobody", "0.0000"],
  ];
  s.addTable(rows, {
    x: 7.3, y: 3.52, w: 5.42,
    colW: [3.2, 2.22],
    rowH: [0.34, 0.36, 0.36, 0.36],
    fontFace: BODY, fontSize: 12.5, color: "1F3D49",
    border: { type: "solid", color: "DCE7EB", pt: 1 },
    fill: { color: "FFFFFF" },
  });

  card(s, 7.3, 5.06, 5.42, 1.1, "E2F0E9");
  s.addText("15.6 extra readmissions caught per 1,000 discharges, net of the false alarms they cost.", {
    x: 7.58, y: 5.06, w: 4.86, h: 1.1,
    fontFace: BODY, fontSize: 12.5, bold: true, color: "1D4D39", margin: 0, valign: "middle", lineSpacing: 16,
  });

  footer(s, "Model beats both alternatives for pt between 0.025 and 0.43 — every realistic hospital sits inside that range.");
  s.addNotes(
    "Note that 'call everyone' is worse than doing nothing at this exchange rate: the false-alarm burden swamps the benefit. " +
    "That is precisely the situation a triage model exists to fix. Vickers & Elkin 2006 is the reference."
  );
}

// ========================================================== 11. drivers ====
{
  const s = lightSlide("What the model is actually reading", "Explainability");
  s.addText("Permutation importance on held-out data and SHAP agree on the top of the list.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  s.addImage({ path: path.join(REPORTS, "shap_beeswarm.png"), x: M, y: 1.86, w: 5.5, h: 4.3 });

  const drivers = [
    ["Prior inpatient admissions", "Utilisation history beats anything measured during the stay."],
    ["Where the patient is discharged to", "Home lowers risk; a care facility raises it — a proxy for frailty."],
    ["Depth of prior encounter history", "How often this patient has been through the system before."],
    ["Primary diagnosis group", "Circulatory pushes risk up; respiratory pushes it down."],
  ];
  drivers.forEach(([t, d], i) => {
    const y = 1.94 + i * 1.06;
    card(s, 6.42, y, 6.3, 0.92, MIST);
    s.addShape(pptx.ShapeType.ellipse, { x: 6.68, y: y + 0.27, w: 0.38, h: 0.38, fill: { color: DEEP }, line: { width: 0 } });
    s.addText(String(i + 1), {
      x: 6.68, y: y + 0.27, w: 0.38, h: 0.38,
      fontFace: HEAD, fontSize: 12, bold: true, color: PAPER, align: "center", valign: "middle", margin: 0,
    });
    s.addText(t, {
      x: 7.2, y: y + 0.12, w: 5.3, h: 0.34,
      fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0,
    });
    s.addText(d, {
      x: 7.2, y: y + 0.44, w: 5.3, h: 0.38,
      fontFace: BODY, fontSize: 11.5, color: "34525E", margin: 0,
    });
  });

  card(s, 6.42, 6.2, 6.3, 0.62, "E2F0E9");
  s.addText("Nothing suspicious at the top — no identifier, no date artefact.", {
    x: 6.68, y: 6.2, w: 5.8, h: 0.62,
    fontFace: BODY, fontSize: 12, bold: true, color: "1F5F44", margin: 0, valign: "middle",
  });
  s.addNotes("Per-patient reason codes ship with every score — a nurse needs the three facts behind a number, not the number alone.");
}

// ========================================================= 12. ablation ====
{
  const s = lightSlide("Seven approaches tried. None beat the baseline.", "Ablation study");
  s.addText("All scored with identical 5-fold GroupKFold on training patients. The test set was not read during any of this.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  const rows = [
    ["Variant", "CV PR-AUC", "Δ", "Adopted"],
    ["A — current, 39 features", "0.2200", "—", "baseline"],
    ["B — drop payer_code", "0.2180", "−0.0020", "no"],
    ["C — +10 engineered features", "0.2196", "−0.0005", "no"],
    ["D — B and C combined", "0.2178", "−0.0023", "no"],
    ["E — native categorical splits", "0.2194", "−0.0007", "no"],
    ["F — soft-vote RF+XGB+LGBM", "0.2191", "−0.0009", "no"],
    ["G — SMOTE, not class weighting", "0.2194", "−0.0006", "no"],
  ];
  s.addTable(rows, {
    x: M, y: 1.9, w: 6.86,
    colW: [3.0, 1.34, 1.24, 1.28],
    rowH: [0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34],
    fontFace: BODY, fontSize: 11.5, color: "1F3D49",
    border: { type: "solid", color: "DCE7EB", pt: 1 },
    fill: { color: "FFFFFF" },
  });

  card(s, 7.66, 1.9, 5.06, 2.24, "E2F0E9");
  s.addText("Every delta is inside noise", {
    x: 7.94, y: 2.1, w: 4.5, h: 0.34,
    fontFace: HEAD, fontSize: 15, bold: true, color: "1F5F44", margin: 0,
  });
  s.addText(
    "Fold-to-fold standard deviation is ±0.0069. Every difference above is smaller than that, so the honest reading is “no measurable difference”, not “slightly worse”.",
    { x: 7.94, y: 2.5, w: 4.5, h: 1.5, fontFace: BODY, fontSize: 12.5, color: "1D4D39", margin: 0, lineSpacing: 17 }
  );

  card(s, 7.66, 4.3, 5.06, 1.84, MIST);
  s.addText("Why this matters", {
    x: 7.94, y: 4.48, w: 4.5, h: 0.32,
    fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "Seven independent approaches converging on 0.22 is itself the finding: more modelling sophistication does not help here.\n\nThat is a ceiling on method \u2014 the learning curve shows it is not a ceiling on data.",
    { x: 7.94, y: 4.86, w: 4.5, h: 1.2, fontFace: BODY, fontSize: 12, color: "34525E", margin: 0, lineSpacing: 16 }
  );

  footer(s, "Reproduce with make experiments · reports/ablation_study.csv");
  s.addNotes("Ensembling buying nothing tells you the three tree models make correlated errors — they read the same limited signal.");
}

// ================================================= 13. how sure are we? ====
{
  const s = lightSlide("Is 0.684 real, and is it the ceiling?", "Robustness");
  s.addText("Two questions the headline number depends on but cannot answer on its own.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  s.addImage({ path: path.join(REPORTS, "robustness.png"), x: M, y: 1.84, w: 7.5, h: 2.74 });

  card(s, 8.32, 1.84, 4.4, 2.74, "FBE4E0");
  s.addText("The reported split is the luckiest of seven", {
    x: 8.6, y: 2.02, w: 3.84, h: 0.62,
    fontFace: HEAD, fontSize: 14.5, bold: true, color: "9E3423", margin: 0, lineSpacing: 19,
  });
  s.addText(
    "Across seven seeds the same pipeline averages PR-AUC 0.224 ± 0.010 and ROC-AUC 0.678 ± 0.006.\n\nSeed 42 was fixed before any evaluation, so this is luck rather than selection — but we report it rather than bury it.",
    { x: 8.6, y: 2.72, w: 3.84, h: 1.66, fontFace: BODY, fontSize: 11.5, color: "7A2A1C", margin: 0, lineSpacing: 15 }
  );

  card(s, M, 4.76, 5.98, 1.44, MIST);
  s.addText("95% confidence intervals", {
    x: M + 0.28, y: 4.92, w: 5.42, h: 0.3,
    fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "ROC-AUC 0.684 [0.671, 0.698]   ·   PR-AUC 0.238 [0.213, 0.265]\nRecall 40.2% [37.6%, 42.7%]\n2,000 cluster resamples of patients, not rows.",
    { x: M + 0.28, y: 5.24, w: 5.42, h: 0.88, fontFace: BODY, fontSize: 11.5, color: "34525E", margin: 0, lineSpacing: 15 }
  );

  card(s, 6.86, 4.76, 5.86, 1.44, "FFF3E6");
  s.addText("This corrected one of our own claims", {
    x: 7.14, y: 4.92, w: 5.3, h: 0.3,
    fontFace: HEAD, fontSize: 14, bold: true, color: "8A5217", margin: 0,
  });
  s.addText(
    "We had called 0.22 the information ceiling of the data. The learning curve is still rising — the last 43% of training data bought +0.0049. The ceiling is on method, not on data.",
    { x: 7.14, y: 5.24, w: 5.3, h: 0.88, fontFace: BODY, fontSize: 11.5, color: "6B4213", margin: 0, lineSpacing: 15 }
  );
  s.addNotes(
    "This slide is the answer to 'how do you know that number is real'. It is also the slide where we correct ourselves: " +
    "the ablation shows a method ceiling, the learning curve shows more patients would still help slightly. " +
    "That points at a multi-hospital dataset, not a better booster, as the next real improvement."
  );
}

// ========================================================= 13. fairness ====
{
  const s = lightSlide("The subgroup gap that would block deployment", "Fairness");
  s.addText("Per-subgroup performance at the deployed threshold. Groups under 200 test encounters are suppressed.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  const rows = [
    ["Age band", "n", "Flag rate", "Recall", "Precision", "ROC-AUC"],
    ["Under 40", "1,289", "20.2%", "58.0%", "34.9%", "0.784"],
    ["40–60", "5,376", "14.8%", "39.6%", "27.9%", "0.712"],
    ["60–75", "9,277", "20.8%", "39.7%", "21.8%", "0.673"],
    ["75 and over", "3,831", "23.0%", "35.8%", "19.0%", "0.613"],
  ];
  s.addTable(rows, {
    x: M, y: 1.9, w: 7.2,
    colW: [1.7, 0.95, 1.15, 1.05, 1.25, 1.1],
    rowH: [0.36, 0.38, 0.38, 0.38, 0.38],
    fontFace: BODY, fontSize: 12, color: "1F3D49",
    border: { type: "solid", color: "DCE7EB", pt: 1 },
    fill: { color: "FFFFFF" },
  });

  s.addText("Race and gender gaps are modest. Age is the problem.", {
    x: M, y: 3.86, w: 7.2, h: 0.34,
    fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0,
  });
  s.addText(
    "The two large racial groups get near-identical flag rates and recall. Patients over 75, however, are flagged most often (23.0%) while the model ranks them worst (ROC-AUC 0.613) — so the tool spends disproportionate follow-up capacity on the group where its ordering is least trustworthy.",
    { x: M, y: 4.26, w: 7.2, h: 1.4, fontFace: BODY, fontSize: 13, color: "34525E", margin: 0, lineSpacing: 18 }
  );
  s.addText(
    "Almost everything the model relies on is a proxy for frailty, and nearly all patients over 75 look frail — so the features stop separating within that group.",
    { x: M, y: 5.62, w: 7.2, h: 0.7, fontFace: BODY, fontSize: 12.5, italic: true, color: MUTED, margin: 0, lineSpacing: 17 }
  );

  card(s, 8.1, 1.9, 4.62, 4.42, "FBE4E0");
  s.addText("Before deployment", {
    x: 8.4, y: 2.12, w: 4.02, h: 0.36,
    fontFace: HEAD, fontSize: 16, bold: true, color: "9E3423", margin: 0,
  });
  s.addText(
    [
      { text: "A separate model for over-75s", options: { bullet: true, breakLine: true } },
      { text: "Or an age-stratified threshold", options: { bullet: true, breakLine: true } },
      { text: "Or restrict the tool to under-75s and triage older patients by existing clinical judgement", options: { bullet: true } },
    ],
    { x: 8.4, y: 2.6, w: 4.02, h: 2.0, fontFace: BODY, fontSize: 13, color: "7A2A1C", margin: 0, paraSpaceAfter: 10, lineSpacing: 18 }
  );
  s.addText("This is not fixable by tuning.", {
    x: 8.4, y: 5.6, w: 4.02, h: 0.4,
    fontFace: HEAD, fontSize: 14, bold: true, color: "9E3423", margin: 0,
  });

  s.addNotes(
    "Protected attributes are kept as features on purpose. Dropping race does not remove its influence — it is reconstructable from other columns — it only removes the ability to measure the disparity."
  );
}

// ================================================ 15. the equity trade ====
{
  const s = lightSlide("We tried the obvious fix. It does not fix it.", "Equity experiment");
  s.addText("One capacity threshold per age band, each fitted on validation, so every band flags 20% of its own members.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  s.addImage({ path: path.join(REPORTS, "equity.png"), x: M, y: 1.84, w: 7.2, h: 2.64 });

  const rows = [
    ["Policy", "Recall", "TP"],
    ["Global threshold", "40.2%", "901"],
    ["Age-stratified", "39.7%", "891"],
  ];
  s.addTable(rows, {
    x: 8.06, y: 1.94, w: 4.66,
    colW: [2.34, 1.16, 1.16],
    rowH: [0.34, 0.36, 0.36],
    fontFace: BODY, fontSize: 12, color: "1F3D49",
    border: { type: "solid", color: "DCE7EB", pt: 1 },
    fill: { color: "FFFFFF" },
  });
  s.addText(
    "Stratifying moves calls away from over-75s and toward the 40–60 band, at a cost of 10 true positives.",
    { x: 8.06, y: 3.14, w: 4.66, h: 0.8, fontFace: BODY, fontSize: 12, color: "34525E", margin: 0, lineSpacing: 16 }
  );
  card(s, 8.06, 3.94, 4.66, 0.54, MIST);
  s.addText("75+ ROC-AUC is 0.613 under both policies.", {
    x: 8.3, y: 3.94, w: 4.18, h: 0.54,
    fontFace: BODY, fontSize: 11.5, bold: true, color: INK, margin: 0, valign: "middle",
  });

  card(s, M, 4.7, 5.98, 1.5, INK);
  s.addText("Thresholds move capacity. They cannot change ranking.", {
    x: M + 0.3, y: 4.86, w: 5.42, h: 0.34,
    fontFace: HEAD, fontSize: 14, bold: true, color: PAPER, margin: 0,
  });
  s.addText(
    "The 75+ problem is discrimination, not calibration. No threshold policy repairs it — a real fix has to be upstream: features that separate frailty within an elderly population, or a model fitted for that group.",
    { x: M + 0.3, y: 5.2, w: 5.42, h: 0.9, fontFace: BODY, fontSize: 11, color: CHALK, margin: 0, lineSpacing: 14 }
  );

  card(s, 6.86, 4.7, 5.86, 1.5, "FFF3E6");
  s.addText("And is stratifying even fairer?", {
    x: 7.14, y: 4.86, w: 5.3, h: 0.34,
    fontFace: HEAD, fontSize: 14, bold: true, color: "8A5217", margin: 0,
  });
  s.addText(
    "Patients over 75 have the highest base rate in the data (12.2%). A policy that calls fewer of them equalises exposure but withdraws attention from the highest-risk group. That is a hospital's value judgement, not a metric.",
    { x: 7.14, y: 5.2, w: 5.3, h: 0.9, fontFace: BODY, fontSize: 11, color: "6B4213", margin: 0, lineSpacing: 14 }
  );
  s.addNotes(
    "The honest framing: two definitions of fairness disagree here. Equal flag rate vs attention proportional to risk. " +
    "We measured the trade and handed the choice to the hospital rather than picking one silently."
  );
}

// ======================================================= 14. deployment ====
{
  const s = lightSlide("Decision support, not an autonomous action", "Deployment");
  s.addText("The model produces a ranked call list. A clinician decides what to do with it.", {
    x: M, y: 1.44, w: CW, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTED, margin: 0,
  });

  const cards = [
    ["Nightly batch", "The discharge list becomes a ranked worklist for the follow-up team, written straight into their queue."],
    ["Real-time endpoint", "FastAPI service — /score returns risk, flag, band and disclaimer. Sub-millisecond tree inference."],
    ["One 5 MB artefact", "Preprocessing, booster, calibrator and threshold travel together in a single joblib file."],
  ];
  cards.forEach(([t, d], i) => {
    const x = M + i * 4.06;
    card(s, x, 1.9, 3.8, 1.74, MIST);
    s.addText(t, {
      x: x + 0.28, y: 2.08, w: 3.24, h: 0.34,
      fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0,
    });
    s.addText(d, {
      x: x + 0.28, y: 2.46, w: 3.24, h: 1.0,
      fontFace: BODY, fontSize: 11.5, color: "34525E", margin: 0, lineSpacing: 16,
    });
  });

  card(s, M, 3.82, CW, 1.62, INK);
  s.addText("Two serving bugs that offline metrics could never have caught", {
    x: M + 0.34, y: 3.98, w: 11.4, h: 0.34,
    fontFace: HEAD, fontSize: 15, bold: true, color: PAPER, margin: 0,
  });
  s.addText(
    [
      { text: "Feature engineering required the readmitted label — which does not exist at discharge.", options: { bullet: true, breakLine: true } },
      { text: "prior_encounters was counted within the batch, so a single-row request scored every patient as a first-ever admission — zeroing out the model's strongest feature.", options: { bullet: true } },
    ],
    { x: M + 0.34, y: 4.36, w: 11.4, h: 0.94, fontFace: BODY, fontSize: 12, color: CHALK, margin: 0, paraSpaceAfter: 6, lineSpacing: 16 }
  );

  card(s, M, 5.62, 5.86, 0.98, "E2F0E9");
  s.addText("Serving reuses data_prep.py directly — a second copy of the feature logic is how models quietly degrade in production.", {
    x: M + 0.28, y: 5.62, w: 5.3, h: 0.98,
    fontFace: BODY, fontSize: 11.5, bold: true, color: "1D4D39", margin: 0, valign: "middle", lineSpacing: 15,
  });
  card(s, 6.85, 5.62, 5.86, 0.98, MIST);
  s.addText("Monitor flag rate, realised vs predicted readmission rate, calibration drift and the subgroup gaps — on a schedule, not once at launch.", {
    x: 7.13, y: 5.62, w: 5.3, h: 0.98,
    fontFace: BODY, fontSize: 11.5, color: "34525E", margin: 0, valign: "middle", lineSpacing: 15,
  });
  s.addNotes("The data is from 1999–2008. Recalibration on recent local data is mandatory before any real use, not optional.");
}

// ================================================ 15. limitations & close ==
{
  const s = darkSlide();
  s.addShape(pptx.ShapeType.ellipse, {
    x: 10.2, y: 4.2, w: 5.4, h: 5.4,
    fill: { color: DEEP, transparency: 55 }, line: { width: 0 },
  });

  s.addText("WHAT WE WOULD SAY TO A HOSPITAL", {
    x: M, y: 0.66, w: CW, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: TEAL, charSpacing: 2.4, margin: 0,
  });
  s.addText("Honest about the ceiling", {
    x: M, y: 1.06, w: 9.0, h: 0.68,
    fontFace: HEAD, fontSize: 34, bold: true, color: PAPER, margin: 0,
  });

  const lims = [
    ["Discrimination is modest — and that is the ceiling.", "0.684 sits inside the 0.64–0.70 band published for this dataset. Seven approaches converged there."],
    ["payer_code_Unknown is a top-6 driver.", "Insurance coding is not physiology. It may not transfer to a new hospital — first thing to re-examine on local data."],
    ["The data is 1999–2008.", "Coding standards, drug availability and discharge policy have all moved. Recalibration is mandatory."],
    ["59.8% of readmissions are missed at 20% capacity.", "This reprioritises attention. An unflagged patient is not a safe patient."],
  ];
  lims.forEach(([t, d], i) => {
    const y = 2.0 + i * 1.06;
    s.addShape(pptx.ShapeType.ellipse, { x: M, y: y + 0.1, w: 0.2, h: 0.2, fill: { color: CORAL }, line: { width: 0 } });
    s.addText(t, {
      x: M + 0.42, y, w: 9.3, h: 0.36,
      fontFace: BODY, fontSize: 14, bold: true, color: PAPER, margin: 0,
    });
    s.addText(d, {
      x: M + 0.42, y: y + 0.36, w: 9.3, h: 0.56,
      fontFace: BODY, fontSize: 12, color: CHALK, margin: 0, lineSpacing: 16,
    });
  });

  s.addShape(pptx.ShapeType.roundRect, {
    x: M, y: 6.36, w: CW, h: 0.72, rectRadius: 0.06,
    fill: { color: "16404F" }, line: { width: 0 },
  });
  s.addText("github.com/sgoel2be24-cyber/hospital-readmission-risk", {
    x: M + 0.32, y: 6.36, w: 6.4, h: 0.72,
    fontFace: "Courier New", fontSize: 12.5, bold: true, color: "8FD4C8", margin: 0, valign: "middle",
  });
  s.addText("make setup && make data && make train  ·  17 tests  ·  ~60 s end to end", {
    x: 7.3, y: 6.36, w: 5.4, h: 0.72,
    fontFace: BODY, fontSize: 11.5, color: CHALK, margin: 0, valign: "middle", align: "right",
  });

  s.addNotes(
    "Close on credibility rather than on a number. The limitations slide is the one that separates a model that was measured from a model that was demoed."
  );
}

const out = path.join(__dirname, "..", "ML_Bubble_2026_Readmission_Risk.pptx");
pptx.writeFile({ fileName: out }).then(() => console.log("wrote", out));
