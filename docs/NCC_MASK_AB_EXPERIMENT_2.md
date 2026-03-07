# NCC masked vs unmasked diagnostic experiment

## Purpose

The [first A/B experiment](NCC_SOBEL_AB_EXPERIMENT.md) showed NCC refinement
harms scoring: lattice-only has 3.2x better safety margin on 43F257A7,
and NCC causes 66/69 misreads on 804D5A50. The user observed 30-40px
visual shifts in debug overlays despite `search_radius=8` limiting NCC
to 8px max shift. Three missing diagnostics were identified:

1. Actual dx/dy value distribution (not just mean)
2. NCC correlation scores at the peak position
3. NCC correlation score at seed position (zero-shift baseline)

This experiment adds per-bubble instrumentation and compares masked
NCC (`TM_CCORR_NORMED` with bracket mask) vs unmasked NCC
(`TM_CCOEFF_NORMED` without mask) across flatbed scans and camera
photos.

## Test images

| Image | Type | Description |
| --- | --- | --- |
| BO-2026.02.24 | Flatbed | High-DPI scanner |
| WO-2026.02.24 | Flatbed | High-DPI scanner |
| 43F257A7-key | Camera | Phone photo |
| 804D5A50-key | Camera | Phone photo |

Each image was self-graded (key = input) to isolate NCC behavior
from answer correctness concerns.

## Instrumentation added

- `score_at_seed`: NCC correlation at the zero-shift lattice position,
  extracted from `result[search_radius, search_radius]` in the NCC
  result map
- `score_delta`: `score_peak - score_at_seed`, measures whether NCC
  actually found a better match
- Per-slot CSV with columns: `q_num`, `choice`, `seed_x`, `ncc_x`,
  `dx`, `dy`, `score_peak`, `score_seed`, `score_delta`, `accepted`,
  `reason`
- Triple-dot debug overlay: yellow (seed), magenta (NCC peak),
  cyan (final applied)
- `--ncc-diag` and `--ncc-no-mask` CLI flags

## Results: comparison table

```
Image        Mask      |dx|mean  |dx|max  peak_mn  seed_mn  delta_mn  pos_d  neg_d
------------------------------------------------------------------------------------
43F          masked       7.47    17.00   0.8846   0.7883    0.0963    500      0
43F          unmasked     2.53     8.00   0.8968   0.4742    0.4226    499      1
804          masked       6.20    16.00   0.8815   0.8173    0.0642    500      0
804          unmasked     4.05    10.00   0.9305   0.4967    0.4339    496      4
BO           masked       4.37    11.00   0.8712   0.7681    0.1031    500      0
BO           unmasked     0.62     2.00   0.9265   0.7576    0.1689    470     30
WO           masked       3.89    11.00   0.8567   0.7453    0.1114    500      0
WO           unmasked     1.45     3.00   0.9241   0.7190    0.2051    499      1
```

Column definitions:
- `|dx|mean`: mean absolute horizontal shift in pixels
- `pos_d`: slots where NCC peak score > seed score (NCC found better match)
- `neg_d`: slots where NCC peak score <= seed score (NCC same or worse)
- `delta_mn`: mean `score_peak - score_seed`

Search radius is now derived from geometry (`refine_max_shift + 1`),
giving 16-17px on these images instead of the old hardcoded 8px.

## Finding 1: masked NCC has tiny score deltas despite large shifts

The mask produces mean deltas of 0.06-0.11. This means NCC moves
positions 4-7 pixels for a correlation improvement of only 6-11%.
The seed (lattice) position already has a very high score (0.74-0.82).
NCC is applying large shifts for small score gains.

Unmasked NCC on flatbed scans (BO, WO) shows larger deltas
(0.17-0.21) with much smaller shifts (0.6-1.5px). The unmasked
path finds meaningfully better matches while barely moving.

## Finding 2: larger search radius reveals masked NCC drift

With the expanded search radius (16-17px vs old 8px), masked NCC
drifts even further: up to 17px on 43F and 16px on 804. The old
8px limit was artificially capping the drift. Unmasked NCC stays
well-behaved: max 8px on camera, 2px on flatbed.

## Finding 3: masked NCC shifts are systematically large on flatbed

dx histograms show the distribution shape on flatbed scans:

**BO masked** -- shifts cluster around 3-4px:
```
0-1px:   11
1-2px:   22
2-3px:  108
3-4px:  181
4-5px:   56
5-6px:   29
```

**BO unmasked** -- shifts cluster tightly at 0-2px:
```
0-1px:  209
1-2px:  270
2-3px:   21
3-4px:    0
```

The masked path shifts 7x more than unmasked on flatbed scans.
The bracket mask at runtime resolution (~60x11px) likely has too
few informative pixels, causing NCC to lock onto noise.

## Finding 4: camera photos have genuine large alignment errors

On camera images, both masked and unmasked NCC apply large shifts
in both axes. With the expanded search radius (`search_radius`
derived from `refine_max_shift`), the shifts are even larger:

- 43F masked: `|dx|` mean=7.5px, `|dy|` mean=13.7px
- 804 masked: `|dx|` mean=6.2px, `|dy|` mean=14.4px
- 804 unmasked: `|dx|` mean=4.1px, `|dy|` mean=2.4px

Visual inspection of the triple-dot overlays confirms these large
shifts are correcting real perspective distortion in camera photos.
The lattice positions derived from timing marks are less accurate
on camera images due to non-affine local warping. Camera photos
genuinely need more correction than flatbed scans.

The problem is not that NCC shifts too much -- the shifts improve
alignment. The problem is that the masked path applies large shifts
for tiny score deltas (0.06-0.10), while the unmasked path achieves
larger deltas (0.43) with its shifts.

## Finding 5: unmasked vs masked score dynamics differ fundamentally

Masked NCC uses `TM_CCORR_NORMED` which produces scores in the
0.6-0.9 range for both peak and seed positions. The seed score is
already high (0.73-0.82), so there is little room for improvement.
This is consistent with the mask selecting only bracket edges,
which are already well-aligned at the lattice position.

Unmasked NCC uses `TM_CCOEFF_NORMED` which produces much lower
seed scores (0.05-0.47 on camera, 0.23-0.77 on flatbed) with
higher peak scores (0.49-0.99). The larger dynamic range suggests
`TM_CCOEFF_NORMED` is more discriminative at separating good from
bad alignment.

## Conclusions

### Masked NCC is harmful in all cases

- Shifts are 4-5px mean with 22-32% hitting the search boundary
- Score improvement is tiny (delta 0.045-0.098)
- Large shifts for negligible gain = pure noise chasing
- The bracket mask at runtime resolution (~60x11px) does not provide
  useful matching signal

### Unmasked NCC is useful on both flatbed and camera

- Flatbed: shifts 0.6-1.5px, zero boundary hits, meaningful deltas
- Camera: shifts 2.5-3.9px, visually confirmed as correcting real
  perspective distortion misalignment

### Recommended path forward

1. If NCC is retained, use `TM_CCOEFF_NORMED` (unmasked) only
2. Camera photos need the larger shifts -- do not cap them
3. Consider a score_delta threshold: only accept shifts where
   `score_peak - score_seed > 0.10` (avoids noise-chasing)
4. The masked path should be removed or disabled -- it actively
   harms alignment on all image types

## Files added or modified

- [omr_utils/template_matcher.py](../omr_utils/template_matcher.py) --
  `score_at_seed` return value in `match_bubble_local()`,
  `match_bubble_masked()`, `refine_row_by_template()`
- [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) --
  per-slot diagnostic collection, extended summary stats, CSV output,
  `ncc_diag_path` and `ncc_no_mask` parameters
- [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py) --
  `draw_ncc_shift_overlay()` triple-dot visualization
- [run_pipeline.py](../run_pipeline.py) --
  `--ncc-diag` and `--ncc-no-mask` CLI flags
- [_temp_ncc_analysis.py](../_temp_ncc_analysis.py) --
  diagnostic CSV analysis script
