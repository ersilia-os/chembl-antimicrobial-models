# How `consensus_score` is built, and what steps 15/16/17 can and cannot tell us

**Date:** 2026-07-27
**Scope:** steps 12b, 14, 15, 16a/b, 17, 18b of this repo
**Status:** findings recorded; two plotting changes made to `16b` — see §9.
**Origin:** written during work on the `ersilia-model-hub-paper` repo, which consumes this
pipeline's outputs; moved here because it documents this pipeline.

Investigation into whether a performance number can be reported for `consensus_score` — the
headline output of the shipped Ersilia pathogen models. **§8 is the standalone summary of
assumptions, risks and open questions**; §§1–7 are the supporting detail.

---

## 1. There is no cross-validation for `consensus_score`

**`consensus_score` only ever exists on DrugBank, which has no activity labels for these
pathogens.** Step 12 runs every trained model over DrugBank to get `prob_rank`; step 14
aggregates those. The labelled training data (`output/07_datasets/`) is read only by steps
07–10, 17 and 18b — the entire consensus chain (12→16) never touches it.

**The contributing models are the step-09.3 full-data retrains, not the fold models.** Even if
labels existed for the training compounds, every one is in-sample for at least one contributing
model, so any AUROC computed there would be leakage-inflated rather than cross-validated.

A leakage-free internal consensus number would require re-running step 09 with **shared fold
assignments per pathogen**, so that a compound held out by model A is also held out by model B.
That is an upstream change and was not pursued.

**Consequence:** the only ground-truth evaluation of `consensus_score` is external validation on
libraries the models never saw — EU OpenScreen and CoAdd, i.e. **step 05 of the
`ersilia-model-hub-paper` repo**.

## 2. How the consensus is assembled (step 14)

`weight[i,m] = mean(w1_m, …, w6_m, w7[i,m])` — seven components, equally weighted
(`W_WEIGHTS = np.ones(7)`), then `score[i] = Σ_m prob_rank[i,m]·weight[i,m] / Σ_m weight[i,m]`.

| | measures | range | notes |
|---|---|---|---|
| w1 | fraction of the negative class that is real data (not added negatives / DUD-E decoys) | 0–1 | 139/193 models at exactly 1.0 |
| w2 | mean CV AUROC, rescaled `(AUROC−0.7)/0.3` | 0–1 | 0 for AUROC ≤ 0.7; no retained model hits 0 |
| w3 | AUPRC vs prevalence baseline: absolute excess (0–0.5) + fold enrichment (0–0.5, saturating at 10×) | 0–1 | |
| w4 | BEDROC vs random baseline, same function | 0–1 | |
| w5 | compound count, piecewise linear (100→0, 1k→0.25, 10k→0.5, 100k→1) | 0–1 | median 0.25 |
| w6 | active count, piecewise linear (50→0, 250→0.25, 1k→0.5, 10k→1) | 0–1 | median 0.29 |
| **w7** | **per-compound**: 0 at or below the model's `decision_cutoff_rank`, linear to 1 at rank 1 | 0–1 | computed at inference, not stored |

`final_weight` in `10_reports.csv` is `mean(w1..w6)` and is **not used by step 14** — step 14
recomputes from the raw weights plus w7.

**w7 is recomputed on every prediction but is not batch-dependent.** `prob_rank` comes from a
frozen out-of-fold ECDF stored in each sub-model artifact (`lazyqsar/artifacts/linear.py:105-112`,
`lazyqsar/agnostic.py:115`), not from a percentile within the submitted batch. A given SMILES
therefore always receives the same `consensus_score`, alone or inside a 100k library.

**Tanh transform.** `f(x) = 0.5 + 0.5·tanh(k(x−0.5))/tanh(k/2)`, applied to restore the IQR that
averaging compresses. `k` is solved numerically **per pathogen and per LOO exclusion** by step 12b
(202 distinct k values in the current run) — there is **no** global `k(M)` meta-curve. k ranges
1.46 (*P. falciparum*) to 5.08 (*C. albicans*), driven by inter-model correlation rather than M.
All 12 fits are `k_star_exact = True`.

The transform is **strictly monotone**, so it changes no ranking and cannot affect AUROC, BEDROC or
EF. It is a presentation step. Only the pathogen's full-consensus `k_star` is baked into the
shipped hub model (`18b:823`).

**Three pathogens have no consensus at all** — campylobacter, ngonorrhoeae and hpylori — because
step 14 requires ≥2 *retained* models. hpylori has two step-09 models but one (`SP_catchall`,
AUROC 0.656) was discarded at step 10.

## 3. What steps 15 and 16 measure

Both run on DrugBank and compare **rankings to rankings**. Step 15 is model vs model (ordered
pairs); step 16 is model vs consensus, in four variants ({LOO, full} × {weighted, unweighted}).

Their "AUROC" columns binarize one model's own top t% as pseudo-labels
(`labels = (scores_b >= cutoff)`, `15:52-56`) and score them with the other array. **No ground
truth is involved.** `docs/CONSENSUS_REPORT.md` files these under "Validation steps", which is
misleading — they measure *agreement*, not accuracy.

The **LOO (exc)** variant is the informative one: it asks whether the other M−1 models can
reconstruct a model's ranking without having seen it. The **full** variant is partly
self-referential, since the model sits inside the consensus it is compared against.

## 4. Empirical findings (run of 2026-07-27)

**Inter-model agreement is low almost everywhere (step 15).** Median pairwise Spearman is
0.08–0.33 in 11 of 12 pathogens; 10 have negative minimum pairs (to −0.47). *P. falciparum* is
the sole exception at 0.530.

**The LOO consensus reconstructs most models poorly (step 16).** Median LOO Spearman 0.24–0.42 in
10 of 12. Of 190 models, 66 score below 0.3 and only 36 above 0.7 — **28 of those 36 are
*P. falciparum*** (plus 5 mtuberculosis, 3 saureus). Nine of twelve pathogens have zero redundant
models.

**Some models are anti-correlated with their LOO consensus**, e.g. `kpneumoniae/SP_0004`
(−0.334), `saureus/DR_0007` (−0.261, `auroc_1pct` 0.174 — the consensus ranks that model's top 1%
*below* average).

**Median top-10 hit overlap between a model and its LOO consensus is 0 in 8 of 12 pathogens.**

**The quality weighting is effectively rank-equivalent to a plain mean.** Correlating the weighted
and unweighted `consensus_score` columns directly: Spearman **0.9940–0.9995** across all 12
pathogens, 87–96 of the top 100 compounds shared, 8–10 of the top 10.

Strictly it *is* a reordering — the weighted mean is not a monotone function of the unweighted one,
so Spearman sits below 1 — but the movement is negligible in practice. On *C. albicans* (11,347
compounds) the median rank shift is **1.1% of the list** (p99 6%, max 11%). The 9 compounds that
enter the weighted top-100 come from unweighted ranks **101–144**, and the 9 that leave land at
103–137; the score spread across ranks 90–110 is **0.0050**, so that boundary is a tie region that
any perturbation reshuffles. Across all 12 pathogens, the deepest a compound is ever promoted into
a top-100 is **rank 221** (smansoni), most 116–180. **No compound anywhere moves substantively.**

Two statistics to avoid here, both misleading: "99.7% of compounds change rank" (vacuous — any
perturbation of continuous scores moves nearly every compound by ≥1 position) and "9 of the top 100
swap" (boundary jitter inside a 0.005-wide score band, not substantive reshuffling).

Do not conflate any of this with the tanh transform, which **is** monotone and leaves the order
bit-identical (Spearman exactly 1.0000000000, verified). The weighting decides the order; the tanh
decides only the spacing.

Mechanically the smallness follows from the arithmetic: `final_weight` spans only 0.26–0.91, so the
heaviest model outweighs the lightest by ~3.5×, and w7 adds at most 1/7.

### 4b. Memorisation test — run 2026-07-27, hypothesis refuted

*P. falciparum* is the only pathogen with high inter-model agreement **and** the largest in-sample
fraction of DrugBank (20.5%), so the obvious hypothesis was that its models had memorised the same
training compounds. Tested by splitting DrugBank per pathogen into **IN** (present in that
pathogen's training data) and **OUT** (never seen), recomputing step-15 pairwise Spearman on each,
with OUT subsampled to |IN| as a size-matched control, and splitting model pairs by data source.

**No effect for *P. falciparum*.** All 1,275 pairs: IN **0.529**, OUT **0.535**, OUT size-matched
**0.530** — a drop of −0.001 on 2,331 in-sample vs 9,016 out-of-sample compounds. Its agreement is
real, not memorised.

**The driver is PubChem redundancy, and it holds out-of-sample:**

| pair type | n pairs | IN | OUT (n-matched) |
|---|---|---|---|
| PubChem + PubChem | 190 | 0.658 | **0.661** |
| PubChem + ChEMBL-DR | 480 | 0.557 | 0.547 |
| PubChem + ChEMBL-SP | 140 | 0.486 | 0.457 |
| ChEMBL-DR + ChEMBL-DR | 276 | 0.474 | 0.480 |
| ChEMBL-DR + ChEMBL-SP | 168 | 0.398 | 0.372 |
| ChEMBL-SP + ChEMBL-SP | 21 | 0.334 | **0.262** |

*P. falciparum* has **20 PubChem models of 51** — far more than any other pathogen (mtuberculosis 9,
saureus 3, calbicans 2, ecoli 1). Its elevated agreement is largely a block of mutually-consistent
PubChem HTS models. Single-point ChEMBL assays are the least consistent of all, at 0.26.

**In-sample inflation is per-pathogen, not systematic — and inversely related to overlap size.**
The pathogen with the largest in-sample fraction shows none, while ecoli at 8.4% shows the largest
drop:

| pathogen | % DrugBank in-sample | IN | OUT (matched) | drop |
|---|---|---|---|---|
| ecoli | 8.4% | 0.331 | 0.118 | **0.213** |
| paeruginosa | 2.3% | 0.212 | 0.101 | **0.111** |
| efaecium | 0.8% | 0.238 | 0.165 | 0.073 |
| saureus | 9.4% | 0.237 | 0.187 | 0.050 |
| calbicans | 10.2% | 0.134 | 0.109 | 0.025 |
| mtuberculosis | 13.1% | 0.136 | 0.138 | −0.002 |
| pfalciparum | 20.5% | 0.529 | 0.530 | −0.001 |

enterobacter (0.226) and abaumannii rest on ~100 in-sample compounds each and should be read as
noise; ecoli's 950 and paeruginosa's 262 are more solid.

**Limitation — the refutation is not airtight.** "IN" means the compound is in the training data of
*some* model for that pathogen, not necessarily **both** models of a given pair. Since a pair only
benefits from memorisation when both members saw the compound, this definition *dilutes* the
effect, so a null result is weaker evidence than it appears. The strict version (restrict each pair
to compounds in both models' training sets, requiring per-model InChIKey lists from
`07_datasets/`) was **not run**.

The test was deliberately kept **outside the codebase** — it is a one-off diagnostic, not a
pipeline step.

## 5. Decisions

**Framing (agreed 2026-07-27).** Steps 15/16 are reported **descriptively**, as ensemble-coherence
and redundancy diagnostics. No causal claim is drawn from them. The low inter-model agreement has
two readings these steps cannot distinguish — (A) genuine assay diversity, complementary signal;
(B) DrugBank sitting outside the applicability domain, so averaging is smoothing noise. Both
predict the same pattern. **Accuracy is established externally, in step 05 of `ersilia-model-hub-paper`, and nowhere
else.**

Never caption a step-15/16 AUROC as performance: a consensus of many correlated models will always
score well on these metrics, as a property of the arithmetic.

**Weighting: parked, and stated precisely.** Two distinct claims that must not be merged.
(a) The weights *do* differentiate models — `final_weight` spans 0.258–0.905, a ~3.5× spread, so
`efaecium/SP_catchall` really is weighted about a quarter of `pfalciparum/504832`. That works as
designed. (b) That differentiation barely reaches the output — the compound ranking is Spearman
0.994–0.9995 against a plain mean. There is no contradiction: a weighted mean over models whose
weights sit inside a 3.5× band, all on the same 0–1 `prob_rank` scale, lands in nearly the same
order as an unweighted one.

The only implication supported by this evidence is narrow: **do not claim the weighting is what
makes the consensus work**, since the same ranking emerges without it. Whether to keep, sharpen or
simplify the scheme is a separate design question this evidence does not settle. No action now.

## 6. Note on the removed `docs/CONSENSUS_REPORT.md`

That file was **deleted on 2026-07-27** after this review found it had drifted substantially from
the code. Recorded here only so the discrepancies are not reintroduced from memory: it described
7 model-level weights plus a per-compound W8 (the code has **6** model-level weights plus w7); w1
as a dataset-specificity table (w1 is the real-negative fraction — `10a:125` notes the old flat
dataset-type weight was removed and the rest renumbered); a global
`k(M) = 2(1+1.156(1−exp(−M/6.47)))` meta-curve (there is none — `12b:103` solves k numerically per
pathogen and per exclusion; for *P. falciparum* the retired curve gives k ≈ 4.31 against a true
1.46); 395 models with pfalciparum at 97 (it is 193 and 52); and "4 models with w3=0 (mean CV
AUROC ≤ 0.70)", where the AUROC weight is now **w2**.

The corrected facts are all stated in §2. `scripts/README.md` is current and agrees with the code.

## 7. Step 17 — data and model quality audit

Reads `07_datasets/`, `07_datasets_metadata.csv`, `10_reports.csv` and `10_discarded_models.csv`;
emits four tables per pathogen (`all_smiles_no_added`, `all_smiles_with_added`, `data_summary`,
`model_summary`) plus a top-level `summary.csv`. Two thresholds, both `src/default.py`:
`FOLD_UNSTABLE_AUROC_STD = 0.05`, `LOW_WEIGHT_THRESHOLD = 0.3`. Reconciles cleanly:
**199 datasets − 6 discarded = 193 models**.

**Label conflicts — and why step 17's number is NOT step 27's.** Step 17 flags a molecule when
`n_active > 0 and (n_inactive + n_added) > 0`, pooling DR and SP and covering ChEMBL **and**
PubChem. Step 27 of `chembl-antimicrobial-tasks` counts a molecule as conflicting only if it
appears in >1 **curated ChEMBL pool** with >1 distinct `bin`, computed **within** DR and within SP
separately. The two therefore differ by 3–8× and must never be quoted interchangeably.

Verified decomposition (2026-07-27, recomputed from `output/stage4/*/2[56]_pools/`):

| pathogen | step 17 | via PubChem | DR↔SP | within-cat | **step 27** | s27 % act | s17 % act |
|---|---|---|---|---|---|---|---|
| pfalciparum | 33,790 | **24,288** | 6,630 | 2,872 | **4,179** | 8.1% | 39.8% |
| mtuberculosis | 5,339 | 2,016 | 2,390 | 933 | **1,155** | 6.9% | 23.0% |
| saureus | 3,843 | 363 | 1,140 | 2,340 | **2,492** | 11.3% | 13.8% |
| calbicans | 3,147 | 2,123 | 490 | 534 | **589** | 7.9% | 33.3% |
| ecoli | 1,807 | 40 | 916 | 851 | **987** | 8.6% | 15.3% |
| paeruginosa | 1,252 | 0 | 651 | 601 | **679** | 11.4% | 21.6% |
| spneumoniae | 652 | 0 | 38 | 614 | **631** | 11.5% | 12.0% |
| kpneumoniae | 443 | 0 | 259 | 184 | **236** | 6.2% | 11.8% |
| efaecium | 133 | 0 | 17 | 116 | **119** | 5.4% | 6.1% |
| abaumannii | 110 | 0 | 56 | 54 | **64** | 4.8% | 8.4% |

Strip the two excluded categories and step 17's residual sits just below step 27's everywhere —
expected, since `07_datasets/` is a subset of the pools. **Neither number is wrong; they answer
different questions.** Step 27: *within the curated ChEMBL pools, per assay category, how
self-consistent is the curation?* → **3–11% of actives**. Step 17: *across everything a pathogen's
models actually train on, how many actives are contradicted somewhere?* → **0–40%**.

The step-17 figure is the modelling-relevant one, because the models do train on the ChEMBL ∪
PubChem union with DR and SP pooled. But its driver is **cross-source and cross-category
disagreement, not curation failure** — quoting 39.8% as "the curation left 40% contradictory
labels" would misrepresent step 27's work. Added negatives and decoys contribute **zero** conflicts
(verified: the `n_added` term in the flag never fires alone).

**DrugBank overlap — 25.4% of the diagnostic substrate is in-sample.** 2,886 of 11,347 DrugBank
compounds appear in at least one pathogen's training set; **pfalciparum alone contributes 2,329
(20.5% of DrugBank)**. Steps 14/15/16 all run on DrugBank.

**Added negatives are concentrated:** spneumoniae 1,908 = **40.6%** of its inactives; enterobacter
11.3%, smansoni 6.8%, pfalciparum 5.0%; six pathogens have none. This feeds w1 directly.

**Flags are advisory — 16 `fold_unstable`, 2 `low_weight`, 0 acted upon; all ship.**
`fold_unstable` = `auroc_std > 0.05` (median across 193 is 0.016), driven almost entirely by too
few positives (rho −0.70 vs `n_positives`). Worst: `smansoni/SP_0001` folds
[0.980, 0.690, 0.985, 0.824, 0.593]; `calbicans/1242` has 121,183 compounds but **19 positives**.
Critically, **`auroc_std` feeds no weight** — `w2` uses the mean only, so abaumannii `DR_0007`
(σ=0.052) and ecoli `DR_0002` (σ=0.003) receive near-identical w2.
`low_weight` = `final_weight < 0.3` (observed range 0.258–0.905, median 0.539) and catches exactly
two, both `SP_catchall` and both also fold-unstable: efaecium (0.258, w6=**0.000**, 25 actives) and
spneumoniae (0.294). The catch-alls are the recurring weak spot — `efaecium/SP_catchall` is also
the model that puts 97.8% of DrugBank above its own decision cutoff.

**Decision-cutoff transfer.** Median fraction of DrugBank above a model's `decision_cutoff_rank` is
0.147, but the tails break w7: 3 models exceed 90% (efaecium `SP_catchall` 97.8%, kpneumoniae
`SP_0004` 92.4%, smansoni `SP_0002` 90.7%) so their bonus is near-constant, and 10 models sit below
1% (incl. `paeruginosa/DR_0001`, n=10,556) so they never receive it and are permanently capped at
6/7 weight. The pattern concentrates in the *last* column of each rank matrix (median 0.245 vs
0.136) because catch-alls are appended last. Reassuringly, DrugBank's median `prob_rank` across all
193 models is **0.496** — no pipeline-wide calibration drift, just individual cases.

## 8. Assumptions, risks and open questions

### Assumptions the design makes

| assumption | status |
|---|---|
| Averaging per-assay models beats any single one | never tested against ground truth |
| ChEMBL and PubChem labels for a pathogen are mutually consistent | cross-source conflict dominates where PubChem is present (72% of pfalciparum's conflicts) — yet PubChem *models* agree with each other more than any other pairing (0.661, §4b). Disagreement is in the labels, not the learned rankings |
| DR and SP assays can be pooled into one model set | 6,630 pfalciparum molecules are active in one category and inactive in the other |
| w1–w6 differentiation propagates to the output ranking | it does not — weighted ≈ unweighted (Spearman 0.994–0.9995). The weights *do* separate models (0.258–0.905); the ranking barely moves |
| `decision_cutoff_rank` transfers to a new library | breaks at both tails (3 models >90%, 10 models <1%) |
| `prob_rank` is comparable across models | it is each model's own training-OOF quantile |
| DrugBank is an independent reference set | 25.4% of it is in training data |
| Mean CV AUROC is a reliable quality signal | 16 models with fold σ > 0.05; worst spans 0.59–0.99 |
| The DrugBank-fitted `k_star` suits other libraries | frozen into the shipped model; ranking-neutral only |

### Risks, worst first

1. **No cross-validated performance for the shipped output, and none obtainable** without re-running
   step 09 with shared fold assignments. Only step 05 can validate.
2. **A quarter of the diagnostic substrate is in-sample** (2,886/11,347). This is *not* what
   explains *P. falciparum* — tested and refuted (§4b) — but it does inflate ecoli (agreement
   0.331 in-sample vs 0.118 out) and paeruginosa (0.212 vs 0.101). Per-pathogen, not systematic.
3. **Training labels contradict at scale across sources** — pfalciparum 39.8% of actives, but
   72% of that is ChEMBL↔PubChem disagreement and most of the rest is DR↔SP. Within a single
   curated ChEMBL category the rate is 3–11% (step 27). Cite the right one for the claim; see §7.
4. **Steps 15/16 read as validation but measure agreement**; a consensus of correlated models scores
   well by construction.
5. **AUROC flatters at these thresholds**: `smansoni/DR_0000` `auroc_1pct` 0.908, yet 1 of its 114
   positives reaches the consensus top-10; median `hit_overlap_10` is 0 in 8 of 12 pathogens.
6. **All quality flags are advisory**; none reduces a weight or excludes a model.
7. **Very thin models ship** (`calbicans/1242`: 19 positives).
8. **Synthetic negatives are concentrated** (spneumoniae 40.6%).
9. **Documentation had drifted severely** — `CONSENSUS_REPORT.md` removed 2026-07-27, see §6.

### Open questions

1. ~~P. falciparum memorisation test~~ — **run 2026-07-27, hypothesis refuted (§4b)**. Remaining:
   the strict per-pair variant (compounds in *both* models' training sets), which would close the
   dilution caveat; and why ecoli/paeruginosa show real in-sample inflation when pfalciparum does
   not.
2. Whether low inter-model agreement is diversity or applicability-domain noise — needs step 05.
3. The 10 inverted models (`saureus/DR_0007`, `auroc_1pct` 0.174) — cross-check against step 05.
4. Whether the weighting should be kept as-is, sharpened, or simplified (parked, §5) — noting
   the evidence shows only that it does not change the ranking, not that it is useless.
5. Whether `fold_unstable` / `low_weight` should act rather than advise.
6. Whether `decision_cutoff_rank` should be recalibrated per prediction library.

## 9. Changes made to `16b` (uncommitted as of 2026-07-27)

`scripts/16b_consensus_results.py` + its `scripts/README.md` entry:

- **Panel [2] extended** to three metric families on one 0–1 axis — `o` AUROC at 0.1/1/5%,
  `^` top-N overlap / N, `s` spearman. Colour encodes depth; the AUROC thresholds and overlap
  depths coincide at n=11,347 (0.1%=12≈top10, 1%=114≈top100, 5%=568≈top500), so circle and triangle
  of the same colour are directly comparable. Two null lines (0.5 for AUROC, 0 for overlap and
  spearman). Y-limit is now data-driven, since spearman goes to −0.33.
- **Histogram `xlim` fixed from `[0.5, 1]` to `[0, 1]`.** The old limit silently hid every pair the
  consensus ranks backwards — 611 of 4,870 pairs overall (12.5%), 29.8% for calbicans — i.e. exactly
  the disagreements the panel exists to show.
- The README entry described a "12-panel dashboard" with an RMSE panel; the script produces 5 panels
  and no RMSE. Corrected.

All 12 PNGs regenerated. **Not committed** — the pipeline was being run the same day, so check for
a concurrent session before committing.

## 10. Downstream consumption

`output/14_consensus/`, `15_recapitulate_models/` and `16_recapitulate_consensus/` were produced
here on 2026-07-27 but are **not** consumed by `ersilia-model-hub-paper`; nothing in that repo reads
them. If any of it is to become a paper figure, a small pre-aggregated summary CSV must be exported
from this repo and pulled through that repo's `scripts/00_download_data.py` — its convention is that
figures are fed from summaries, never from full per-compound tables.
