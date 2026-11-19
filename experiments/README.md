# experiments/

One entry point per chain. Every run writes to `runs/<chain>/` (gitignored) and
never to `results/`; `results/README.md` describes how to compare a run with the
frozen results. Measured runtimes are in the top-level `README.md`.

## benchmark and benchmark_quality

- Entry points, run from the framework root:
  - `python experiments/benchmark/run_benchmark.py [--out-dir runs/benchmark]` writes the seven CSVs of `results/benchmark/` into the output directory.
  - `python experiments/benchmark/run_benchmark_quality.py [--runs-root runs/benchmark_quality]` writes `<runs-root>/<UTC timestamp>/12_table1_quality_estimation.csv`.
- Reproduce and compare:
  - `python -m tal_calibration.results reproduce benchmark --work DIR`
  - `python -m tal_calibration.results reproduce benchmark_quality --work DIR`
- Raw inputs used (`MANIFEST.json` `inputs`; 22 files for `benchmark`):
  - Annotations: `data/raw/annotations/{thumos/thumos14.json, epic_verb/epic_kitchens_100_verb.json, epic_noun/epic_kitchens_100_noun.json, ego4d/ego4d.json, anet/anet1.3_tsp_filtered.json}`
  - ActionFormer: `data/raw/predictions/actionformer/{thumos, epic_verb, epic_noun, ego4d, anet}/post_nms.pkl`
  - TriDet: `data/raw/predictions/tridet/{thumos, epic_verb, epic_noun}/post_nms.pkl`, `data/raw/predictions/tridet/anet/post_nms_ext.pkl`
  - TemporalMaxer: `data/raw/predictions/temporalmaxer/{thumos, epic_verb}/post_nms.pkl`
  - TadTR: `data/raw/predictions/tadtr/thumos/detection_nms.json`, `data/raw/predictions/tadtr/anet/detection_nms_top2.json` (`num_queries=10`), `data/raw/predictions/tadtr_nq100/anet/detection_nms_top2.json`
  - TE-TAD: `data/raw/predictions/te_tad/{thumos, epic_verb, epic_noun}/t_nms.json`
  - `benchmark_quality` reads the same files except `tadtr/anet/detection_nms_top2.json` (21 files).

### Code map

- `experiments/benchmark/run_benchmark.py`: for each pair, builds the pair context and calls one task per output CSV; checks the expected key grid before writing.
- `experiments/benchmark/run_benchmark_quality.py`: runs the Table 1 $R^2$ cells for the sixteen pairs.
- `src/tal_calibration/protocols/benchmark/protocol.py`: registered pairs and per-video budgets (`OFFICIAL_POPULATION`, `TABLE1_PAIRS`), τ grids and primary τ, the one-to-one matcher, video-grouped out-of-fold calibration and the calibration metrics.
- `src/tal_calibration/protocols/benchmark/common.py`: loads a pair, labels its capped pool, and fits the out-of-fold confidence columns.
- `src/tal_calibration/protocols/benchmark/benchmark_tasks.py`: the pair context, the per-population refit, and one function per CSV: `benchmark` (01), `population_dilution` (02), `bootstrap_ci` (05), `laece_reveal` (08), `lrp_operating_point` (10), `reliability_bins` (13).
- `src/tal_calibration/protocols/benchmark/table1_quality_estimation.py`: the score-only gradient-boosting regressor and its video-grouped out-of-fold $R^2$ (12).
- `src/tal_calibration/populations.py`: the two populations, `official_topk` and the cross-fitted oLRP score threshold `lrp_crossfit`.
- `src/tal_calibration/{calibration, evaluation, matching}.py`: ECE and the netcal calibrators; mAP with the vendored ActionFormer evaluator; tIoU matching.

## fragmentation

- Entry point: `python experiments/fragmentation/run_fragmentation.py` (writes the three CSVs to `runs/fragmentation/`; `--out-dir DIR` to change it).
- Reproduce and compare: `python -m tal_calibration.results reproduce fragmentation --work DIR`
- Raw inputs used (from the MANIFEST `inputs`): the evaluation-split annotations of THUMOS14, EPIC-Kitchens-100 verb and noun, Ego4D-MQ and ActivityNet-1.3 (`data/raw/annotations/`), and the exported post-NMS predictions of the 16 Table 1 pairs (`data/raw/predictions/`: ActionFormer ×5 datasets, TriDet ×4, TemporalMaxer ×2, TE-TAD ×3, TadTR on THUMOS14, and TadTR with `num_queries=100` on ActivityNet-1.3), plus `configs/datasets.yaml`.

### Code map

- `experiments/fragmentation/run_fragmentation.py`: entry point; runs every pair of `TABLE1_PAIRS` on both populations, checks the key grid and writes the three CSVs.
- `src/tal_calibration/protocols/benchmark/fragmentation_tasks.py`: builds the top-K and LRP pools, fits the credit-fitted and frag-fitted isotonic calibrators, and computes the summary rows, the rescued-fragment counts, the paired video bootstrap and the Fig. 1(b) instance.
- `src/tal_calibration/protocols/benchmark/protocol.py`: `label_multi_tau` (official greedy one-to-one matcher and the strict two-pass matcher of Algorithm S1), `oof_calibrate` and `video_folds` (video-grouped 5-fold out-of-fold calibration), and the pair and threshold registries `TABLE1_PAIRS`, `CALIB_TAUS`, `HEADLINE_TAU`.
- `src/tal_calibration/populations.py`: the `lrp_crossfit` operating point. The oLRP-optimal score threshold is cross-fitted on the same video folds and anchored at each dataset's middle grid threshold (`HEADLINE_TAU`: .5 THUMOS14, .75 ActivityNet-1.3, .3 elsewhere); it is fitted once and used across the whole τ grid.
- `src/tal_calibration/calibration.py`: `ece`, pooled ECE with 10 equal-width bins, reported in percent.

## annotation

- Entry point: `python experiments/annotation/run_annotation.py` (writes to `runs/annotation/`)
- Reproduce and compare: `python -m tal_calibration.results reproduce annotation --work DIR`
- Raw inputs used: `data/raw/annotations/anet/anet1.3_tsp_filtered.json` (ActivityNet-1.3 annotations, validation subset), `data/raw/predictions/actionformer/anet/post_nms.pkl` (ActionFormer post-NMS predictions)

### Code map

- `protocols/annotation/lib.py`: loaders, the DETAD constants and boundary-noise generator, the greedy one-to-one matcher (quality and TP/FP targets), crc32 video folds, pooled out-of-fold isotonic calibration, credit ECE and LaECE₀, the two populations (full and cross-fitted LRP), and the video-clustered bootstrap.
- `protocols/annotation/steps.py`: transfer check (01), noise curve (02), alternative worlds and per-world targets.
- `protocols/annotation/analysis.py`: readouts shared with Ego4D-MQ: released-fit vs same-world refit (04), hit-level ICC (05), target shift (10).
- `protocols/annotation/pipeline.py`: runs the reference seed, then the five-seed aggregates and bootstrap intervals (09).
- `experiments/annotation/run_annotation.py`: entry point that writes the CSVs.

## annotation_ego4d

- Entry point: `python experiments/annotation_ego4d/run_annotation_ego4d.py` (writes to `runs/annotation_ego4d/`). It checks the parity of the rebuilt union against `ego4d.json` and the row alignment of the LRP population before writing anything.
- Reproduce and compare: `python -m tal_calibration.results reproduce annotation_ego4d --work DIR`
- Raw inputs used:
  - `data/raw/annotations/ego4d/moments_train.json` and `data/raw/annotations/ego4d/moments_val.json`: the official Ego4D Episodic Memory Moments Queries annotations, v1 (`"version": "1"`, `"date": "220429"`), which keep each annotator's moments separately. Both subsets are needed: validation supplies the worlds, and training is checked to show that the released annotation is the union the detector was trained on.
  - `data/raw/annotations/ego4d/ego4d.json`: the released annotation used for training and evaluation, generated from the two files above by `python -m tal_calibration.data build-ego4d-json` (ActionFormer's annotation rules without the feature-grid alignment; see the top-level README).
  - `data/raw/predictions/actionformer/ego4d/post_nms.pkl`: ActionFormer post-NMS predictions on the validation subset.

### Code map

- `protocols/annotation/ego4d.py`: per-annotator tables from the moments files, the parity check against `ego4d.json` (00), single-annotator worlds (each clip keeps one uniformly drawn annotator), per-world targets, the five-seed and bootstrap readouts (09), and the invariants checked before writing.
- It reuses `lib.py` (matcher, calibrator, metrics, populations at Ego4D's anchor τ) and `analysis.py` (04, 05, 10) from the ActivityNet-1.3 chain.
- `experiments/annotation_ego4d/run_annotation_ego4d.py`: entry point that writes the CSVs.

## quality_uncertainty and quality_uncertainty_multidetector

- Entry points (from the framework root; each run writes
  `<runs-root>/<run_id>/summary/`):
  - `python experiments/quality_uncertainty/run_quality_uncertainty.py --config src/tal_calibration/protocols/quality_uncertainty/configs/formal_all_datasets.yaml --runs-root runs/quality_uncertainty`
  - `python experiments/quality_uncertainty/run_quality_uncertainty.py --config src/tal_calibration/protocols/quality_uncertainty/configs/formal_multi_detector.yaml --runs-root runs/quality_uncertainty_multidetector`
- Reproduce and compare:
  - `python -m tal_calibration.results reproduce quality_uncertainty --work DIR`
  - `python -m tal_calibration.results reproduce quality_uncertainty_multidetector --work DIR`
- Raw inputs used (from the MANIFEST `inputs`):
  - Both chains read the evaluation-split annotations of THUMOS14,
    EPIC-Kitchens-100 verb and noun, Ego4D-MQ and ActivityNet-1.3
    (`data/raw/annotations/`), and ActionFormer's post-NMS and pre-NMS
    exports on all five datasets (`data/raw/predictions/actionformer/*/{post_nms,pre_nms}.pkl`).
  - `quality_uncertainty_multidetector` also reads:
    - TriDet's post-NMS and pre-NMS exports on THUMOS14 and EPIC verb;
    - TadTR's `detection_nms.json` on THUMOS14;
    - TE-TAD's `t_nms.json` on THUMOS14 and EPIC verb.
  - Configs: the chain's YAML and `configs/datasets.yaml`.

### Code map

All under `src/tal_calibration/protocols/quality_uncertainty/`; the entry point is `experiments/quality_uncertainty/run_quality_uncertainty.py`.

- `data/`: everything upstream of an estimand.
  - `prepared_data.py`: loads annotations and predictions, applies the declared per-video top-K, runs the τ=0 one-to-one matching and adds the post-NMS neighbourhood features (`observable_features.py`).
  - `cloud.py`: builds the pre-NMS proposal-cloud features.
  - `splits.py`: draws the seeded 50/25/25 video partition of each dataset's evaluation split.
  - `schemas.py`: column semantics and validation.
- `estimation/population.py` and `head.py`: build the candidate universe (prepared rows joined with the cloud features) and the target `q`, then the evaluated head: partition → train-fitted LRP selection (`operating_point.py`) → shared train/cal/test frames.
- `estimation/arms.py`: the arm registries. Point, interval/floor and matched-subset mechanism arms come in two bundles (45 features for dense detectors, 15 post-NMS features for TadTR and TE-TAD), together with the ban on ground-truth features outside the oracle arm.
- `estimation/estimators.py`: histogram gradient-boosting point and quantile models.
- `estimation/conformal.py`: CQR intervals and one-sided floors with the conformal rank taken over calibration videos.
- `estimation/metrics.py` and `bootstrap.py`: point, interval and floor metrics, and the paired video-cluster bootstrap CIs.
- `driver/`: runs the chain.
  - `tasks.py`: expands a config into one task per (registered dataset–detector arm, seed).
  - `shard.py`: computes the five per-task tables.
  - `contracts.py`: checks them before they are written.
  - `readout.py`: builds the `11_readout_*` tables.
  - `figure_data.py`: writes the Fig. 1(c) data.
  - `driver.py`: writes a fresh run directory.
