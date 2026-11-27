"""Build the six-page validation report from the committed tables (requires ReportLab)."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
QUICK = ROOT / "results/quick"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def baseline_macro(name):
    rows = read_rows(ROOT / "evidence/baseline_run" / name / "per_query.csv")
    cells = defaultdict(list)
    for row in rows:
        rank = 1 + int(row["n_greater"]) + int(row["n_equal"]) / 2
        assert abs(1 / rank - float(row["rr"])) < 1e-12
        cells[row["relation"], row["direction"]].append(1 / rank)
    return [name.replace("_seed101", "").replace("_", " "), len(cells),
            f"{mean(float(r['rr']) for r in rows):.4f}",
            f"{mean(mean(v) for v in cells.values()):.4f}"]


def build():
    pdf = Canvas(str(ROOT / "docs/Validation_Report.pdf"), pagesize=(612, 792))
    pdf.setTitle("Popularity, Leakage, or Structure? | Team 7 validation audit")
    pdf.setAuthor("Kamil Gainutdinov, Artem Salakhutdinov, Marat Akhmetov")
    margin, width = 84, 444
    y = 712

    def para(text, size=11.5, font="Times-Roman", leading=16, center=False):
        nonlocal y
        style = ParagraphStyle("body", fontName=font, fontSize=size, leading=leading,
                               alignment=TA_CENTER if center else 0)
        block = Paragraph(text, style)
        _, height = block.wrap(width, 700)
        y -= height
        block.drawOn(pdf, margin, y)
        y -= 10

    def heading(text):
        para(text, 14, "Helvetica-Bold", 18)

    def table(headers, body):
        nonlocal y
        grid = Table([headers, *body], hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f4f4")]),
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        w, h = grid.wrap(width, 680)
        assert w <= width
        y -= h
        grid.drawOn(pdf, margin + (width-w)/2, y)
        y -= 16

    def finish():
        nonlocal y
        assert y > 52, (pdf.getPageNumber(), y)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(margin, 38, "Team 7  /  DKR course project  /  Validation report")
        pdf.drawRightString(528, 38, str(pdf.getPageNumber()))
        pdf.showPage()
        y = 712

    pdf.setLineWidth(2.5)
    pdf.line(margin, 727, margin + width, 727)
    para("Popularity, Leakage, or Structure?", 23, "Helvetica-Bold", 28, True)
    para("An evaluation audit of biomedical link prediction on ogbl-biokg", 13, "Times-Italic", 17, True)
    para("Kamil Gainutdinov, Artem Salakhutdinov, Marat Akhmetov<br/>Team 7 · Innopolis University · 26 November 2026", 10.5, "Helvetica", 15, True)
    heading("Abstract")
    para("On a relation-stratified validation sample of 963 triples, a scorer using only the candidate's training frequency achieves mean reciprocal rank (MRR) 0.4782 on official negatives. Replacing those candidates with typed alternatives selected by frequency rank reduces MRR to 0.1192. The paired change is -0.3590 (95% cluster-bootstrap interval -0.3749 to -0.3432). Only 545 of 1,926 queries meet the operational balance threshold used in the saved run. No same-relation reverse training triple was available for the 1,726 applicable queries. Three short learned-model pilots establish reference scores on official negatives. Their performance under frequency matching remains unmeasured.")
    heading("1   Research question")
    para("In a head-completion query (?, r, t), a scorer can use the fixed entity t, the relation r and each candidate head. A simpler strategy ignores t and chooses the candidate that most often appears as a head of r in training. If that strategy ranks the true answer highly, the benchmark rewards information that is available without modelling the pair of endpoints.")
    para("We ask how sensitive the observed scores are to candidate selection, whether an exact reverse edge provides a shortcut, and how learned scorers compare at a common training budget. The unit of inference is a benchmark query. Biological plausibility and clinical utility require different evidence.")
    finish()

    heading("2   Sampling, controls and uncertainty")
    para("OGB's ogbl-biokg benchmark supplies typed entities and relation-specific queries [1]. The training split has 4,762,678 triples. The audit samples up to 20 of the 162,886 validation triples per relation with row seed 71. Each selected triple contributes a head and a tail query. Both directions use 500 candidate negatives of the appropriate entity type. Frequency counts, frequency bins and reverse lookup are derived from training.")
    heading("Candidate sets")
    para("<b>Official.</b> Reuse the supplied candidates. <b>Nearest frequency.</b> Order eligible entities by train count for the same relation and role. Select near the target's rank, taking candidates from both sides when possible. At the extremes the sampler borrows from the other side. <b>Same-bin first.</b> Sample within the target's frequency bin and fill any shortfall from adjacent bins. The two generated protocols exclude triples known positive in train, validation or test. The scorer and trainer receive no test labels for fitting.")
    heading("Ranking and pairing")
    para("Let G count negatives scoring strictly above the positive and E count negatives tied with it. Its mid-rank is 1 + G + E/2 and its reciprocal rank is the inverse of that value. For example, one higher candidate and two ties give rank 3. MRR averages reciprocal ranks across queries. Hits@10 counts the fraction of ranks at most 10. The implementation checks ties and candidate permutation against the official OGB evaluator [2].")
    para("To compare scorers, align row ID and prediction direction, subtract the two reciprocal ranks for each query, and average. Bootstrap resampling uses relation/entity-pair clusters to keep dependent directions together. Same-type pairs are unordered within a relation. Reported 95% intervals use 1,000 resamples and condition on this sample and candidate seed 2024. They exclude variation from retraining or selecting another candidate seed.")
    finish()

    heading("3   Baselines under three candidate protocols")
    main = {(r["model"],r["protocol"]):r for r in read_rows(QUICK / "main_table.csv")}
    models = [("random","Random"),("count_prior","Relation/role count"),("global_degree","Global degree"),("reverse_lookup","Reverse + count")]
    table(["Scorer","Official","Nearest","Bin-first"], [[label, *[f"{float(main[name,p]['rr_mean']):.4f}" for p in ("official","matched_nearest","matched_bin")]] for name,label in models])
    para("Table 1. Mean reciprocal rank on the same 1,926 validation queries. Reverse lookup uses the count prior to break ties. Global degree counts appearances across relations. Per-query records and candidate fingerprints support paired comparisons within each protocol.",10,"Times-Italic",14)
    para("The count prior exceeds global degree by 0.0966 MRR on official negatives and 0.0297 under nearest matching. The nearest-matched count prior still exceeds random overall by 0.1022 [0.0905, 0.1138]. Matching uses the same train counts as the prior. Its score change describes sensitivity to the candidate rule and cannot allocate a fraction of a learned model's performance to popularity.")
    img = ImageReader(str(QUICK / "mrr_by_frequency.png"))
    iw, ih = img.getSize()
    h = width * ih / iw
    y -= h
    pdf.drawImage(img, margin, y, width, h)
    y -= 10
    para("Figure 1. MRR by frequency of the true answer. Error bars show cluster-bootstrap intervals. Group sizes are query counts. The high-frequency stratum accounts for 1,300 of the 1,926 queries.",10,"Times-Italic",14)
    finish()

    heading("4   Balance, weighting and reverse availability")
    para("A query meets the run's balance threshold if at least 25% of its negatives have a count greater than or equal to its target's count. With 500 negatives, at least 125 must tie with or outrank the target. Even if all 125 tie, the mid-rank is at least 63.5. Thus the threshold itself caps count-prior reciprocal rank at 0.01575. It is a property of a query with a particular candidate set. Across nearest-matched queries the mean fraction is 0.201. Only 37.2% of nearest candidates share the target's frequency bin.")
    ci = read_rows(QUICK / "ci_by_stratum.csv")
    groups = {(r["group"],r["model"]):r for r in ci if r["protocol"]=="matched_nearest" and r["breakdown"]=="popularity_matchable" and r["metric"]=="rr"}
    table(["Nearest-frequency subset","Queries","Count MRR","Random MRR"], [[label,groups[k,"count_prior"]["n_queries"],*[f"{float(groups[k,m]['mean']):.5f}" for m in ("count_prior","random")]] for k,label in (("True","Threshold met"),("False","Below threshold"))])
    para("For low-frequency answers the paired count-minus-random estimate is +0.00009 [-0.00204, 0.00211]. For mid-frequency answers it is -0.00057 [-0.00692, 0.00491]. Both intervals include zero. The data do not establish equivalence between the scorers.")
    heading("Sensitivity to query weighting")
    table(["Official scorer","Relation/role cells","Query mean","Equal-cell mean"], [baseline_macro(n) for n in ("random_seed101","count_prior","reverse_lookup")])
    para("Table 2. Recomputed from archived official per-query ranks. Each observed relation/direction cell has equal weight in the last column. Both estimates refer to the sampled queries. Neither weighting reconstructs the frequency of relations in the full validation split.",10,"Times-Italic",14)
    para("Of 1,726 applicable same-type queries, zero had an exact reverse training triple with the same relation. Reverse lookup therefore matched the prior on every observed query. The remaining 200 cross-type queries are inapplicable. With no observed reverse-available group the proposed between-group comparison is undefined. Other possible forms of leakage fall outside this diagnostic.")
    finish()

    heading("5   Learned pilots and the missing comparison")
    pilots = [r for r in read_rows(ROOT / "results/pilot_results.csv") if r["training_steps"]]
    table(["Model","Updates","Seed","Official MRR","Hits@10"], [[r["model"],r["selected_step"],r["seed"],f"{float(r['mrr']):.4f}",f"{float(r['hits10']):.4f}"] for r in pilots])
    para("Table 3. Historical Kaggle T4 validation pilots. The original executed notebooks preserve selected-step MRR and rounded Hits. Trained weights and complete per-query records for these pilots are absent from the archived evidence.",10,"Times-Italic",14)
    para("All three pilots used 128 real scalars per entity, batch size 256, 64 typed training negatives, Adam and learning rate 0.001. A validation check selected step 500. HeadBlind models candidate and relation without conditioning on the opposite endpoint. TransE [3] and ComplEx [4] can use both endpoints. HeadBlind rose from 0.2851 to 0.3606 between steps 250 and 500, TransE from 0.2126 to 0.2408, and ComplEx from 0.0147 to 0.0154. A two-point trajectory gives little evidence about convergence.")
    heading("What the updated implementation tests")
    para("The Kaggle notebook exports the selected scorer's weights, configuration and dataset identity. The validation command restores that scorer, generates candidate sets once, compares it with the baselines and checks row and candidate fingerprints. The per-query output supports paired estimates for each frequency stratum. A synthetic integration test trains a small model for one update and checks restoration and scoring. It contributes no benchmark performance estimate.")
    para("Recovering the original weights would permit a direct rescore of these pilots. A new export-enabled pilot instead produces a new observation with its own seed, selected step and manifest. Before running several seeds, measure one cloud pilot's time and memory, inspect the loss and validate shared-candidate scoring. The common training horizon and final decision rule must be recorded before scoring the locked test set.")
    finish()

    heading("6   Reproducibility, contribution and sources")
    para("The baseline audit entry point is scripts/run_audit.py. A quick run with --download prepares the official data. Use --out with a fresh directory for the tables. To add a saved scorer, pass --checkpoint with its scoring export. The run records row and candidate hashes, seeds, sample size, balance diagnostics and software versions. The repository README contains the complete commands.")
    para("The committed quick tables have identical Git contents in main ac9230d and fix 62bfa53. The latter includes a separate run manifest and regenerated figure. That manifest records Windows, Python 3.13.14 and 467 seconds elapsed. The November baseline tables and learned pilot scores predate this report. The equal-cell summary on page 4 is calculated directly from archived per-query counts. Full validation, multiple training seeds and locked test scoring remain open.")
    heading("Our contribution and AI use")
    para("The team ran the Kaggle pilots and assembled the evaluation evidence. Its contribution is a controlled comparison of candidate protocols, relation/role frequency and reverse baselines, uncertainty estimates and a reproducible checkpoint-to-evaluation path. OGB supplies the benchmark and reference evaluator. TransE and ComplEx follow published model families. The midway report identifies Astra/Sol assistance with research, code and interpretation. This revision used Codex for implementation, tests, review and report preparation. Team members retain responsibility for verification and presentation.")
    heading("References")
    para('[1] Hu et al. Open Graph Benchmark. NeurIPS 2020. <link href="https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg" color="#125566">Benchmark specification</link>.<br/>[2] OGB authors. Link prediction evaluator, ogb 1.3.6. <link href="https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py" color="#125566">Reference implementation</link>.<br/>[3] Bordes et al. Translating Embeddings for Modeling Multi-relational Data. NeurIPS 2013.<br/>[4] Trouillon et al. Complex Embeddings for Simple Link Prediction. ICML 2016. <link href="https://proceedings.mlr.press/v48/trouillon16.html" color="#125566">Proceedings</link>.',10.5,"Times-Roman",14)
    finish()
    pdf.save()


if __name__ == "__main__":
    build()
