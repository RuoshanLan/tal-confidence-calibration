# Calibrating Confidence in Temporal Action Localization

Code and results for the paper *Calibrating Confidence in Temporal Action
Localization: What Detector Outputs Reveal and What Remains Uncertain*, by
Ruoshan Lan, Ehsan Azimi and Mihai Surdeanu, accepted at WACV 2027
([accepted papers](https://wacv.thecvf.com/Conferences/2027/AcceptedPapers)).
The paper will be linked here once it becomes available through CVF Open
Access.

The code recomputes, from detector predictions and dataset annotations, every
number and table printed in the paper and its supplementary material, and the
data behind every figure. `results/` holds those results as frozen CSV files.
One command recomputes a chain from the raw inputs and compares it with
`results/` cell by cell.

## Chains

| Chain | Paper | Results |
|---|---|---|
| `benchmark` | Sec. 3: Table 1, Fig. 2; supp. Sec. B | `results/benchmark/` |
| `benchmark_quality` | Table 1, $R^2$ columns | `results/benchmark_quality/` |
| `fragmentation` | Sec. 4.2 (Audit 2): Table 3, Fig. 4, Fig. 1(b); supp. Sec. A(b), C | `results/fragmentation/` |
| `annotation` | Sec. 4.1 (Audit 1), ActivityNet-1.3: Table 2; supp. Sec. D | `results/annotation/` |
| `annotation_ego4d` | Sec. 4.1 (Audit 1), Ego4D-MQ: Table 2; supp. Sec. D | `results/annotation_ego4d/` |
| `quality_uncertainty` | Sec. 5: Table 4, Fig. 1(c); supp. Sec. A(c), E | `results/quality_uncertainty/` |
| `quality_uncertainty_multidetector` | supp. Sec. E, the ten detector–dataset pairs | `results/quality_uncertainty_multidetector/` |

`results/README.md` maps each table, figure and printed number to the result
file, rows and columns it comes from. `experiments/README.md` describes each
chain's entry point, inputs and code.

## Installation

```bash
micromamba create -n tal_calib -f environment.yml   # or conda
micromamba activate tal_calib
pip install -e .
```

`environment.yml` pins the exact versions the results were produced and
verified with (Python 3.10.19). `netcal` requires PyTorch; `environment.yml`
installs its CUDA 13 build (torch 2.10.0+cu130), the build the results were
produced with. No chain uses the GPU: every chain computes on CPU only.

## Data

All inputs live under `data/raw/`. `data/manifest.json` lists every file the
chains read. After placing the files, check that they are all in place with

```bash
python -m tal_calibration.data verify
```

**Annotations.** These come from the datasets' official distributions; we do
not redistribute them.

- THUMOS14, ActivityNet-1.3 and EPIC-Kitchens-100 (verb and noun): the
  annotation files distributed with
  [ActionFormer](https://github.com/happyharrycn/actionformer_release)
  (`thumos14.json`, `anet1.3_tsp_filtered.json`,
  `epic_kitchens_100_{verb,noun}.json`).
- Ego4D-MQ: the official Episodic Memory Moments Queries annotations v1
  (`moments_train.json`, `moments_val.json`), obtained from
  [Ego4D](https://ego4d-data.org) under its license. Place them in
  `data/raw/annotations/ego4d/` and generate `ego4d.json` from them with

  ```bash
  python -m tal_calibration.data build-ego4d-json
  ```

  It follows the annotation rules of ActionFormer's
  `tools/convert_ego4d_trainval.py` (per clip, the union of every annotator's
  primary moments of at least 0.25 s, times relative to the clip start and
  rounded to 2 decimals) without its feature-grid alignment, since the released
  file pairs with EgoVLP's clip-level features. Running ActionFormer's script
  unmodified shifts most clips, so its output differs from the `ego4d.json` used
  here. The label map is ActionFormer's (`src/tal_calibration/third_party/`). The
  `annotation_ego4d` chain also rebuilds this union from the two raw files and
  checks it against `ego4d.json`.

**Detector predictions.** The chains read the exported predictions of
ActionFormer, TriDet, TemporalMaxer, TadTR and TE-TAD (post-NMS, and pre-NMS
where a chain uses proposal-cloud features) from `data/raw/predictions/`.
They are not distributed with this repository, nor are the detector weights.
Three of the 16 detector-dataset pairs use officially released weights
(ActionFormer on Ego4D-MQ, TadTR on THUMOS14, TriDet on EPIC-Kitchens-100
verb); the other 13 were trained locally with the detectors' released code.
Supp. Sec. B describes how each detector was configured. The exported
ActivityNet-1.3 predictions already carry the classes and scores of the CUHK
video-level classifier (`cuhk_val_simp_share.json`, distributed with
ActionFormer's ActivityNet-1.3 annotations): each proposal is expanded to its
top-2 classes and the scores are combined as described in supp. Sec. B. The
code does not read this file.

## Reproducing the results

```bash
python -m tal_calibration.results check                   # results/ matches its manifests
python -m tal_calibration.results reproduce CHAIN --work DIR
```

`reproduce` runs the chain into `DIR` and compares every result file with
`results/CHAIN/`. A chain passes when the last line reads `RESULT: PASS`.

Measured on a 48-core machine, CPU only, `OMP_NUM_THREADS=8`, three chains
running at a time:

| Chain | Runtime |
|---|---|
| `benchmark` | 77 min |
| `benchmark_quality` | 19 min |
| `fragmentation` | 69 min |
| `annotation` | 13 min |
| `annotation_ego4d` | 10 min |
| `quality_uncertainty` | 35 min |
| `quality_uncertainty_multidetector` | 36 min |

## Repository layout

| Path | Contents |
|---|---|
| `src/tal_calibration/` | the library: calibration, temporal geometry, matching, evaluation, populations, and one protocol package per experiment family (`protocols/{benchmark, annotation, quality_uncertainty}`) |
| `src/tal_calibration/third_party/` | vendored ActionFormer evaluator (MIT; see `THIRD_PARTY_NOTICES.md`) |
| `experiments/` | one entry point per chain |
| `configs/` | dataset configuration |
| `data/` | `manifest.json` (tracked) and `raw/` (the inputs, not tracked) |
| `results/` | frozen results, one directory per chain with its `MANIFEST.json` |
| `runs/` | output of every run (not tracked) |

## Citation

```bibtex
@inproceedings{lan2027calibrating,
  title     = {Calibrating Confidence in Temporal Action Localization: What Detector Outputs Reveal and What Remains Uncertain},
  author    = {Lan, Ruoshan and Azimi, Ehsan and Surdeanu, Mihai},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

MIT; see `LICENSE`. The vendored ActionFormer evaluator keeps its own MIT
license; see `THIRD_PARTY_NOTICES.md`.
