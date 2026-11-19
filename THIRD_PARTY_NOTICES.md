# Third-party notices

## 1. ActionFormer: TAL evaluator (vendored code)

- **File**: `src/tal_calibration/third_party/actionformer_metrics.py`
- **Contents**: `load_gt_seg_from_json`, `compute_average_precision_detection`
  and their helpers (`remove_duplicate_annotations`, `segment_iou`,
  `interpolated_prec_rec`), byte-for-byte identical to the source.
- **Source**: `libs/utils/metrics.py` of
  https://github.com/happyharrycn/actionformer_release, commit
  `61ea7eb9308a568b0cf45e3804830836e30061de`.
- **Upstream attribution**: the source file is "Modified from official
  EPIC-Kitchens action detection evaluation code"
  (https://github.com/epic-kitchens/C2-Action-Detection), whose README states
  that parts of that evaluation code were adapted from the ActivityNet
  evaluation code (https://github.com/activitynet/ActivityNet, file
  `Evaluation/eval_detection.py`, with `segment_iou` and
  `interpolated_prec_rec` in `Evaluation/utils.py`). The EPIC-Kitchens
  repository does not state a license. The ActivityNet repository is
  distributed under the MIT License, reproduced below.
- **License**: MIT (ActionFormer), reproduced below.

## 2. ActionFormer: Ego4D-MQ label map (data file)

- **File**: `src/tal_calibration/third_party/ego4d_label_map.txt`, copied
  unmodified. `python -m tal_calibration.data build-ego4d-json` reads it to map
  the 110 Ego4D-MQ categories to `label_id`.
- **Source**: `tools/ego4d_label_map.txt` of
  https://github.com/happyharrycn/actionformer_release, commit
  `ba971128ec026aeb3da81636d9873571a3692f20`.
- **License**: MIT, reproduced below.

## 3. netcal (dependency, not vendored)

Four of the five calibrators (histogram binning, Platt, beta and temperature
scaling) come from the `netcal` package; isotonic regression comes from
scikit-learn. `netcal` is available at https://github.com/EFS-OpenSource/calibration-framework,
pinned to 1.4.0 and installed by `environment.yml`. It is distributed under the
Apache License 2.0.

## Datasets

The annotations of THUMOS14, ActivityNet-1.3, EPIC-Kitchens-100 and Ego4D-MQ
are not distributed with this repository; obtain them from their official
sources under their respective licenses (see `README.md`).

## ActionFormer license (applies to sections 1 and 2)

```
MIT License

Copyright (c) 2021 University of Wisconsin-Madison

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ActivityNet license (upstream of section 1)

```
The MIT License (MIT)

Copyright (c) 2015 ActivityNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
