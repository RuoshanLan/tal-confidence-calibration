# results/

The frozen results the paper is built from. One directory per chain; each
holds its result files and a `MANIFEST.json`:

| field | content |
|---|---|
| `chain` | chain name (= directory name) |
| `command` | the command that produces the results, run from the framework root |
| `config` | config file -> sha256 of its parsed YAML content |
| `params` | (annotation_ego4d only) protocol constants read by other tools |
| `inputs` | the input files under `data/raw/` the chain reads |
| `results` | result file -> sha256; exactly the files in the directory |

## Check

```
python -m tal_calibration.results check
```

## Reproduce

Run the chain's `command` from its `MANIFEST.json`, then compare the run with
the frozen results:

```
python experiments/benchmark/run_benchmark.py --out-dir runs/benchmark
python -m tal_calibration.results compare benchmark runs/benchmark
```

`benchmark_quality`, `quality_uncertainty` and
`quality_uncertainty_multidetector` write one directory per run; pass that
directory, e.g. `runs/quality_uncertainty/<run_id>`. Or run and compare in
one step:

```
python -m tal_calibration.results reproduce CHAIN --work DIR
```

A chain passes when the last line reads `RESULT: PASS`: every result file is
byte-identical, or equal cell by cell with max|delta| = 0.

Every chain recomputes everything from `data/raw` on every run, and no chain
reads another chain's results.

## Replace a chain's results

Copy the run's result files (the manifest's `results` list) into
`results/CHAIN/`, then rewrite the manifest:

```
python -m tal_calibration.results manifest CHAIN
```

## Paper element → result file

For every table, figure and number in the paper and the supplementary
material, the result file and the rows and columns it is read from. Section,
table and figure numbers follow the camera-ready version.

### benchmark and benchmark_quality

The confidence benchmark of Sec. 3 and supp. Sec. B: the sixteen detector–dataset
pairs of Table 1, plus the `num_queries=10` ActivityNet-1.3 TadTR that the Table 1
footnote compares against.

Files are under `results/benchmark/`, except `12_table1_quality_estimation.csv`,
which is under `results/benchmark_quality/`. Conventions used below:

- `population=official_topk` is the Top-$K$ population and `population=lrp_crossfit`
  the LRP operating point.
- *τ-mean* is the mean over the dataset's τ-grid rows (THUMOS14 .3–.7, EPIC and
  Ego4D-MQ .1–.5, ActivityNet-1.3 .5/.75/.95). Table 1 and the supp. tables
  average the TP/FP metrics this way.
- `dataset=anet, model=tadtr` is the `num_queries=10` TadTR and is read only by
  the Table 1 footnote. Table 1's ActivityNet-1.3 TadTR‡ row is
  `model=tadtr_nq100`. Statements over "the sixteen pairs" exclude `anet, tadtr`.

| Paper element | Result file | Columns / rows to read |
|---|---|---|
| Table 1, mAP | `01_pair_summary.csv` | `average_mAP` |
| Table 1, keep% | `10_lrp_operating_point.csv`, `01_benchmark_main.csv` | `crossfit_n_keep` ÷ `n` of `population=official_topk, arm=raw` |
| Table 1, TP% | `01_benchmark_main.csv` | `tp_rate`, `population=lrp_crossfit, arm=raw`, τ-mean |
| Table 1, TP/FP ECE (raw, Iso × Top-$K$, LRP) | `01_benchmark_main.csv` | `ECE`, `arm ∈ {raw, iso}`, both populations, τ-mean |
| Table 1, LaECE$_0$ (raw, Iso × Top-$K$, LRP) | `08_laece_reveal.csv` | `pooled_LaECE0_raw`, `pooled_LaECE0_iso` (× 100), both populations |
| Table 1, AUROC | `01_benchmark_main.csv` | `auroc`, `arm=raw`, both populations, τ-mean |
| Table 1, $R^2$ | `12_table1_quality_estimation.csv` | `r2`, `population=all_predictions`, `eval_population ∈ {official_topk, lrp_crossfit}` |
| Table 1 footnote ‡ (mAP 33.9 → 34.2; Top-$K$ raw ECE 2.04% vs 0.21%) | `01_pair_summary.csv`, `01_benchmark_main.csv` | `dataset=anet`, `model ∈ {tadtr, tadtr_nq100}`: `average_mAP`; `ECE` at `population=official_topk, arm=raw`, τ-mean |
| Data behind Fig. 2 (THUMOS14 reliability diagrams) | `13_reliability_bins.csv` | THUMOS14, τ = 0.5, both populations × `arm ∈ {raw, iso}` × 10 bins: `n`, `share`, `mean_conf`, `tp_rate` |
| Sec. 3 text: 71–89% of THUMOS14 Top-$K$ predictions in the lowest confidence bin | `13_reliability_bins.csv` | `share` at `population=official_topk, arm=raw, bin=0` |
| Sec. 3 text: dense THUMOS14 detectors underconfident by 25.5–28.0 pp; every DETR-style pair overconfident at the operating point | `01_benchmark_main.csv` | `mean_conf − tp_rate`, `population=lrp_crossfit, arm=raw`, τ-mean |
| Sec. 3 text: $R^2$ reaches .80 on Top-$K$ and falls to .13–.21 at the operating point | `12_table1_quality_estimation.csv` | as the Table 1 $R^2$ row; `population=hits_only` gives the matched-only refit |
| Supp. benchmark specification table: ActionFormer mAP 66.8 / 19.5 / 36.6 | `01_pair_summary.csv` | `average_mAP` |
| Supp. specification table: native caps (2,000 / ~900 / 400 → 200) | `02_population_dilution.csv` | `rows_per_video_max` where `is_full_pool=True` |
| Supp. specification table: the output budget moves raw ECE by 1.5–4.3× | `02_population_dilution.csv` | ActionFormer, `arm=raw`: `ECE` at `topk_per_video=5` ÷ `ECE` at the full pool, per dataset |
| Supp. specification table: operating-point anchor τ and five folds | `10_lrp_operating_point.csv` | `crossfit_anchor_tau`, `crossfit_n_folds` |
| Data behind the supp. budget figure (THUMOS14, τ = 0.5) | `02_population_dilution.csv` | `dataset=thumos, arm=raw`: `ECE` against `topk_per_video`; markers at `is_full_pool=True` |
| Supp. complete benchmark table | `01_benchmark_main.csv`, `01_pair_summary.csv`, `08_laece_reveal.csv` | `auroc`, `ECE`, `ACE`, `SCE`, `NLL` at `arm=raw`; `ECE` at `arm ∈ {hist, iso, platt, beta, temp}` (τ-means); `average_mAP`; `LaECE0_macro_raw`, `LaECE0_macro_iso` (× 100) |
| Supp. complete table caption: classes under 50 rows; class-macro vs pooled LaECE$_0$ | `08_laece_reveal.csv` | `n_classes_under_50_rows`, `LaECE0_macro_raw`, `pooled_LaECE0_raw` at `population=lrp_crossfit` |
| Supp. bootstrap table, and the 0/32 and 20/32 counts | `05_bootstrap_ci.csv` | `ECE`, `ECE_lo`, `ECE_hi`, `NLL`, `NLL_lo`, `NLL_hi`, `arm ∈ {raw, iso}`, at each dataset's primary τ |
| Supp. signed-gap table, and the confidence bands and TP-rate spans of the direction-of-miscalibration section | `01_benchmark_main.csv` | `mean_conf`, `tp_rate` and their difference, `arm=raw`, both populations, τ-mean |
| Supp. direction-of-miscalibration section: top-5 confidences .77–.79 vs .48–.50, converged by a budget of 100 | `02_population_dilution.csv` | `mean_conf`, `dataset=thumos, arm=raw`, `topk_per_video ∈ {5, 100}` |

### fragmentation

Audit 2 (Sec. 4.2): official one-to-one greedy matching versus the strict
two-pass fragment-tolerant matching of supp. Algorithm S1, on the 16 Table 1
pairs and two populations: `population=official` is the top-K pool,
`population=lrp_crossfit` the LRP operating point. Every calibrated number is
pooled isotonic (`arm=iso`), fitted out of fold on video-grouped folds.

| Paper element | Result file (under `results/fragmentation/`) | Columns / rows to read |
|---|---|---|
| Table 3, "Iso ECE (%) TP-fit → frag-fit" | `03_matching_summary.csv` | `model=actionformer`, `dataset` in {`epic_verb`, `epic_noun`, `ego4d`}, each population; mean over the dataset's `tau` grid of `ece_cfit_yfrag` → `ece_ffit_yfrag` |
| Table 3, "Rescued (%) at τ=.1" | `fragment_counts.csv` | same rows, `tau=0.1`; `100 * n_rescued / n_predictions` |
| Table 3, per-threshold Δ_match (pp) | `03_matching_summary.csv` | same rows, one per `tau`; `delta_match_ece` |
| Sec. 4.2: Δ_match "at most 0.41 pp … vanishes at τ=.5" (top-K) | `03_matching_summary.csv` | Table 3 top-K rows; `delta_match_ece` |
| Sec. 4.2: "statistically detectable" on top-K; "no entry separates from zero" at LRP | `03_matching_summary.csv` | Table 3 rows; `delta_match_ece_ci_lo`, `delta_match_ece_ci_hi` (95% paired video-clustered bootstrap, B=2000); an entry separates from zero when the interval excludes 0 |
| Sec. 4.2: rescued fragments are "0.17–0.46% of top-K predictions and 0.85–1.44% at the LRP operating point" | `fragment_counts.csv` | the Table 3 "Rescued" cells |
| Fig. 1(b) caption, Sec. 4.2 and supp. Sec. A: 42.9% of official TPs on Ego4D-MQ at τ=.1 (= 1,790 / 4,176) | `fragment_counts.csv` | `dataset=ego4d`, `model=actionformer`, `population=official`, `tau=0.1`; `n_rescued / n_tp_official` |
| Sec. 4.2: below 0.6% on every dataset by τ=.4, zero at τ=.5 | `fragment_counts.csv` | `population=official`; per `dataset`, `sum(n_rescued) / sum(n_tp_official)` over models at `tau=0.4` and `tau=0.5` |
| Data behind Fig. 4 | `fragment_counts.csv` | `population=official`, `dataset != anet`; (a) per `dataset` and `tau`, `100 * sum(n_rescued) / sum(n_tp_official)` over models; (b) the same per model for `actionformer` and `te-tad` on `epic_verb` and `epic_noun` |
| Fig. 4 caption: no strict fragments on ActivityNet-1.3 | `fragment_counts.csv` | `dataset=anet`; `n_rescued` is 0 in every row |
| Data behind Fig. 1(b), and the instance in supp. Sec. A(b) | `fig1b_fragment_example.csv` | every GT (`role=gt`) of video `088ae5ff-ae50-40bd-bea9-7bbdaab0c29b`, class `peel_a_fruit_or_vegetable`, and every prediction (`role=pred`) reaching tIoU ≥ .1 with one of them: `t_start`, `t_end`, `score`, `tp_credit_0.1`, `tp_frag_0.1`, and `tiou_gt0..2` (tIoU with each GT). The drawn GT is `id=1` [2.48, 51.74]; the drawn TP has score 0.244 (tIoU 0.1646 with it), the drawn fragment score 0.220 (tIoU 0.674) |
| Supp. Table S5 | `03_matching_summary.csv` | every pair with `dataset != anet`, each population; mean over the `tau` grid of `ece_cfit_yfrag` → `ece_ffit_yfrag` |
| Supp. Table S5 caption: on ActivityNet-1.3 the two targets coincide and every Δ is zero | `03_matching_summary.csv`, `fragment_counts.csv` | `dataset=anet`: `delta_match_ece` = 0, `n_rescued` = 0 |
| Supp. Sec. C.2: bootstrap intervals over all pairs ("about a quarter of the top-K entries", none at LRP, only the two lowest thresholds) | `03_matching_summary.csv` | `dataset != anet`, `tau < 0.5`; `delta_match_ece_ci_lo`, `delta_match_ece_ci_hi` (11 of 42 top-K entries and 0 LRP entries exclude 0, all at τ ∈ {.1, .2}) |

### annotation and annotation_ego4d

Audit 1 (Sec. 4.1, Table 2, supp. Sec. D) has two chains with the same readouts:
`annotation` builds synthetic alternative annotation worlds for ActivityNet-1.3
from DETAD's published agreement statistics, and `annotation_ego4d` uses the real
annotators of Ego4D-MQ. Rates and calibration errors are stored as fractions and
printed as percentages (x100); `01_transfer_check.csv` is already in percent, and
ICC and tIoU values are printed as stored. In Table 2, row "Top-K" is
`population=full` and row "LRP" is `population=lrp_crossfit`.

#### annotation (ActivityNet-1.3, synthetic worlds)

| Paper element | Result file (under `results/annotation/`) | Columns / rows to read |
|---|---|---|
| Table 2, ActivityNet-1.3 columns, TP/FP ECE rows | `04_claimA_floor.csv` | `claim=A_credit_ece`, `tau=0.5`; Raw / Rel. / Same = `raw_cross` / `cal_pop0_floor` / `same_world_refit` |
| Table 2, ActivityNet-1.3 columns, LaECE₀ rows | `04_claimA_floor.csv` | `claim=A_laece0`; same columns |
| Sec. 4.1, transfer–refit gap at the LRP operating point ("almost nine" pp TP/FP, "about nine" pp LaECE₀) | `04_claimA_floor.csv` | `population=lrp_crossfit`: `cal_pop0_floor − same_world_refit` |
| Sec. 4.1, 45.5% of LRP predictions change TP/FP status; 13.6% of labels differ; TP rate 69.6% → 59.6% ("about ten" pp) | `10_target_shift.csv` | `target=credit`, `population=lrp_crossfit`, `tau=0.5`: `moved_frac`, `differ_frac`, `rel_rate`, `alt_rate`, `rate_drop` |
| Sec. 4.1, across-world SD 0.1546 tIoU; 30.3% annotation-relative variance | `05_claimB_icc.csv` | `stable_def=same_instance`: `sigma_within`; 1 − `icc_corrected` |
| Sec. 4.1, 83.3% of matched predictions stay matched in every world | `05_claimB_icc.csv` | `stable_def=protocol`: `stable_frac` |
| supp. D (transfer check), marginal agreement within 0.29 pp | `01_transfer_check.csv` | max of `gap_pp` |
| supp. D (noise scale from agreement), 25 values of σ_rel in [.02, .80]; agreement floor ≈ .30 | `02_noise_curve.csv` | 25 rows; `mean_median_pairwise_tiou` of the last row |
| supp. D (alternative worlds), mean released-vs-alternative tIoU ≈ .699 | `09_multiseed_ci.csv` | mean of `mean_swap_tiou` over the rows `seed=43..47` |
| supp. D (quality-target changes), 7,360 / 6,132 / 997 / 5,135 predictions; ICC_corr .6971, σ_within .1546, 30.3% | `05_claimB_icc.csv` | `n_hits`; `stable_def=protocol`: `n_stable`, `n_identity_switch`; `stable_def=same_instance`: `n_stable`, `icc_corrected`, `sigma_within` |
| supp. D (sampling uncertainty), ICC interval [.6829, .7094]; .6969 ± .0045 over seeds | `09_multiseed_ci.csv` | `seed=BOOT_ci`, `stat=icc_corrected_same_instance`: `ci_lo_2.5`, `ci_hi_97.5`; `seed=AGG_seeds`: `icc_corrected_same_instance`, `icc_corrected_same_instance_std` |
| supp. D (sampling uncertainty), transferred ECE 9.78 ± 0.34 and LaECE₀ 9.89 ± 0.18; intervals [9.14, 11.18] and [9.19, 10.74] | `09_multiseed_ci.csv` | `seed=AGG_seeds`: `head_credit_floor@0.5_paired_popfit(_std)`, `head_laece0_floor_paired_popfit(_std)`; `seed=BOOT_ci` rows with those `stat` values (`eval_worlds=6-10`) |

The DETAD joint table J printed in supp. D, and the generator settings it states
(ten worlds, five seeds 43–47, 3,000 simulated sets, up to 20 redraws, LRP anchor
τ = .75), are constants and defaults in `lib.py`, not result files.

#### annotation_ego4d (Ego4D-MQ, real annotators)

| Paper element | Result file (under `results/annotation_ego4d/`) | Columns / rows to read |
|---|---|---|
| Table 2, Ego4D-MQ columns, TP/FP ECE rows | `04_claimA_floor.csv` | `claim=A_credit_ece`, `tau=0.1`; Raw / Rel. / Same = `raw_cross` / `cal_pop0_floor` / `same_world_refit` |
| Table 2, Ego4D-MQ columns, LaECE₀ rows | `04_claimA_floor.csv` | `claim=A_laece0`; same columns |
| Sec. 4.1, transfer–refit gap at the LRP operating point ("about seventeen" pp TP/FP, "about nine" pp LaECE₀) | `04_claimA_floor.csv` | `population=lrp_crossfit`: `cal_pop0_floor − same_world_refit` |
| Sec. 4.1, no label changes from FP to TP; TP rate at τ = .1 falls from 32.0% to 14.1% | `10_target_shift.csv` | `target=credit`, `population=lrp_crossfit`, `tau=0.1`: `gain_frac` (= 0), `rel_rate`, `alt_rate` |
| Sec. 4.1, only 11.1% of predictions matched under the union stay matched in every world | `05_claimB_icc.csv` | `stable_def=protocol`: `stable_frac` |
| Sec. 4.1, each clip has up to three annotators | `00_parity.csv` | `clips_with_4plus_annotators` = 0 on both subsets |
| supp. D (single-annotator worlds), the union reproduces the released GT: 1,486 training clips / 13,559 moments, 521 validation clips / 4,296 moments; 470 / 44 / 7 validation clips with 3 / 2 / 1 annotators | `00_parity.csv` | `n_clips`, `n_instances`, `clips_with_3_annotators`, `clips_with_2_annotators`, `clips_with_1_annotator`; `mismatch_ordered` = 0 |
| supp. D, ten worlds for each of five seeds; TP/FP at τ = .1, LRP anchored at τ = .3 | `MANIFEST.json` | `params`: `n_worlds`, `world_seeds`, `audit_tau`, `lrp_anchor_tau` |
| supp. D (sampling uncertainty), 17.91 ± 0.05 and 10.00 ± 0.03 over seeds; intervals [17.06, 18.99] and [9.49, 10.65] | `09_multiseed_ci.csv` | `seed=AGG_seeds`: `credit_transferred(_std)`, `laece0_transferred(_std)`; `seed=BOOT_ci` rows (`eval_worlds=6-10`) |

### quality_uncertainty and quality_uncertainty_multidetector

Sec. 5 and supp. Sec. E: how much of a prediction's localization quality `q`
the exported detector outputs recover, what they omit (the hidden matched
extent, `xi = log(L_g / L_p)`), and conformal quality intervals and floors.
Both chains run the same code on the LRP head. One train-fitted oLRP score
cutoff is fitted per (dataset, seed) at tau=0, the anchor of the target `q`
(the τ=0 one-to-one matched tIoU, exactly 0 off the matched set). Each
result has five video-disjoint 50/25/25 train/calibration/test splits
(seeds 0–4). The paper prints the mean over the five seeds of each
(dataset, detector, arm) cell. Filter `population=lrp` (the only value
present). Coverage is always the video-balanced `coverage_video_macro`; the
pooled `coverage` column is not printed anywhere.

- `quality_uncertainty`: ActionFormer on the five datasets. It also writes
  the readout tables (`11_readout_*.csv`) and the Fig. 1(c) data.
- `quality_uncertainty_multidetector`: the ten detector–dataset pairs
  (ActionFormer ×5, TriDet and TE-TAD on THUMOS14 and EPIC verb, TadTR on
  THUMOS14), without the readout tables. Its ActionFormer rows equal those of
  `quality_uncertainty`.

Arm names: `score_only` (Score), `full_observable` (Full, 45 features),
`post_nms_observable` (Full for TadTR and TE-TAD, 15 features),
`population_marginal` (Feature-free), `trivial_zero` (the [0,0] control),
`matched_obs` / `matched_post_nms_obs` (matched Full), `matched_oracle` /
`matched_post_nms_oracle` (+ξ).

Paths are under `results/quality_uncertainty/summary/` unless marked
**MD**, which means `results/quality_uncertainty_multidetector/summary/`.

| Paper element | Result file | Columns / rows to read |
|---|---|---|
| Table 4, Hit | `11_readout_attribution.csv` | `hit_rate_test`. This is video-macro; the identically named column of `10_point_seeds.csv` is prediction-pooled and is not the printed value |
| Table 4, Full-pop. R² Score | `10_point_seeds.csv` | `arm=score_only`, `r2` |
| Table 4, Full-pop. R² Full | `11_readout_attribution.csv` | `r2_a_full_regressor` (equal to `10_point_seeds.csv`, `arm=full_observable`, `r2`) |
| Table 4, A−B | `11_readout_attribution.csv` | `gap_a_minus_b` |
| Table 4, Matched R² Full / +ξ | `10_mechanism_seeds.csv` | `arm=matched_obs` / `arm=matched_oracle`, `r2` |
| Table 4, Cov. / Wid. | `10_interval_seeds.csv` | `arm=full_observable`, `coverage_video_macro` / `mean_width` |
| Table 4, Floor | `10_floor_seeds.csv` | `arm=full_observable`, `floor_median` |
| Sec. 5 text: A−B at most .019 on four datasets and .059 on THUMOS14; +ξ R² .50–.96; coverage .82–.91; mean widths .52–.92; median floors .516 and .305, zero on the other three | as the Table 4 cells | as the Table 4 cells |
| Sec. 5 text: matched-row R² from −.05 to .12 across all ten pairs | **MD** `10_mechanism_seeds.csv` | `arm` in {`matched_obs`, `matched_post_nms_obs`}, `r2`, mean per (`canonical_dataset`, `detector`) |
| Data behind Fig. 1(c); its caption (3.0 s predictions, ceilings 0.44 and 0.04); supp. Sec. A(c) | `fig1c_hidden_extent.csv` | Two rows, one per drawn EPIC verb `mix` instance (`label_id=10`): `video_id`, `gt_start`, `gt_end`, `gt_dur`, `ceiling`. The class-level columns repeat on both rows: `n_eval_instances` (287), `gt_dur_min` / `gt_dur_max` (0.50–133.35 s), `pred_n` (5,820), `pred_len_median` (2.96 s) |
| Data behind Fig. S2 (floor vs. demanded coverage) | `11_readout_coverage.csv` | Per `dataset`, `arm` in {`population_marginal`, `full_observable`}, grouped by `target_coverage` (.30–.90). Solid curve: `floor_median` mean over seeds, with a min–max band. Dashed curve: `positive_floor_rate`. Hollow marker: `head_status` in {`marginal_constant`, `not_fitted`} on at least 3 of 5 seeds. Dotted line: mean `hit_rate_test`. The `target_coverage=0.8` rows reproduce `10_floor_seeds.csv` |
| Table S7 (population and fitting protocol), printed counts | `10_selection_ledger.csv`, `10_interval_seeds.csv` | LRP cutoff: `partition=train`, `score_cutoff`. Selected test rows / matched: `n_test` / `n_test_hits`. The 212/138/138/521/4,926 videos are the sizes of the annotations' evaluation splits, which the partition covers in full; they are not stored in the results |
| Table S8 (conformal protocol), printed counts | `10_interval_seeds.csv` | Calibration videos: `n_cal_videos`. Calibration rows (786–2,196): `n_cal`. Calibration predictions per video (1.3 on ActivityNet, 65 on EPIC verb): `n_cal / n_cal_videos`. Realized levels (.801–.848): `conformal_level`, range over rows |
| Table S9 (matched predictions, other detectors) | **MD** `10_mechanism_seeds.csv` | `detector` in {`tridet`, `tadtr`, `te-tad`}. Full: `matched_obs` (TriDet) or `matched_post_nms_obs` (TadTR, TE-TAD), `r2`. +ξ: `matched_oracle` / `matched_post_nms_oracle`, `r2`. Gap: `oracle_gap` |
| Table S10 (complete estimation ladder) | **MD** `10_point_seeds.csv`, `10_interval_seeds.csv`, `10_floor_seeds.csv` | One block per (`canonical_dataset`, `detector`). Pt. R²: `r2` of `score_only` and of `full_observable` / `post_nms_observable`. Interval Cov. / Wid.: `coverage_video_macro` / `mean_width`. Floor Med. / Cov.: `floor_median` / `coverage_video_macro`. Rows are the four uncertainty arms. In the block heading, 45 or 15 features comes from `feature_interface`, the native budget from `native_max_per_video` ("→ top-200" when `n_rows_truncated > 0`), and the hit rate is `1 − coverage_video_macro` of the interval `trivial_zero` arm |
| Table S11 (floor vs. prediction population) | `11_readout_population.csv` | `arm=population_marginal`, `status=ok`. Primary: `rule=lrp`. Target h: `rule=hit`, `knob` in {.70, .80, .90}. Columns: `hit_rate_test`, `recall_test`, `floor_median`, `positive_floor_rate` |
| Supp. Sec. E text: EPIC recall below 6% at the .90 target, Ego4D-MQ below 6% at .80 | `11_readout_population.csv` | `recall_test` of the Table S11 rows |
| Table S12 (target-definition robustness) | `11_readout_convention.csv` | `convention=one_to_one` (1:1) and `convention=max` (Max). Columns: `zero_mass_test`, `r2_matched`, `oracle_gap`, `floor_median` |
| Supp. Sec. E text: Max lowers the zero mass by .02–.14; matched R² at most .138; ξ gain at least .482 | `11_readout_convention.csv` | the Table S12 columns, `one_to_one` minus `max` for the zero mass |
| Table S13 (per-split results) | `10_mechanism_seeds.csv`, `10_interval_seeds.csv` | One row per (`dataset`, `seed`), not averaged. Gap / LB: `arm=matched_obs`, `oracle_gap` / `oracle_gap_ci_lo`. Cov. / 95% CI: `arm=full_observable`, `coverage_video_macro` / `coverage_video_macro_ci_lo`, `coverage_video_macro_ci_hi`. Model: `head_status` (`conditional` = C, `marginal_constant` = F) |

Table S6 (feature definitions) and the hyperparameters in Tables S7 and S8
come from the code, not from a result file (paths under
`src/tal_calibration/protocols/quality_uncertainty/`): the feature lists in
`data/observable_features.py` and `data/cloud.py`, the learner settings in
`estimation/estimators.py`, and α = β = .20 in `estimation/conformal.py`.
