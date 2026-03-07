# NCC vs Sobel A/B scoring experiment

## Purpose

The runtime scoring pipeline applies two sequential x-refinement stages
to lattice-derived bubble positions before measuring fill brightness:

1. **NCC template matching** -- shifts bubble center (x, y) to maximize
   normalized cross-correlation against letter templates
2. **Sobel edge detection** -- detects left/right bracket edges using
   horizontal gradient peaks, then overwrites NCC's x-position

The Sobel stage always overwrites NCC's x-position. It was unknown whether
either stage helps, hurts, or is silently overridden. This experiment
isolates each refinement layer to measure its contribution independently.

## Experiment design

Three refinement modes were tested using the `-r`/`--refine-mode` CLI
argument added to [run_pipeline.py](../run_pipeline.py):

| Mode | NCC stage | Sobel stage | What it tests |
| --- | --- | --- | --- |
| `lattice` | skip | skip | pure geometry baseline |
| `ncc+sobel` | run | run | current default pipeline |
| `ncc` | run | skip | NCC without Sobel override |

Each mode was run on two key images with known ground truth:

- **43F257A7**: Q1-71 each have exactly one filled bubble, Q72-100 all blank,
  only Q28 has answer E
- **804D5A50**: 69 filled answers, 31 blank

Automated checks per mode:
1. Q1-71 accuracy (must read exactly one answer per row, no BLANK/MULTIPLE)
2. Q72-100 blank detection (must read BLANK for all 29 questions)
3. Debug overlay visual inspection (measurement zones vs bracket structure)

## Results: 43F257A7

### Answer accuracy

All three modes produce identical answer letters. Zero errors on both
the filled and blank gates.

```
  Mode            Q1-71    Q72-100    Total
                 errors  false_pos   errors
  lattice             0          0        0
  ncc+sobel           0          0        0
  ncc                 0          0        0
```

### Confidence score separation

Although answers are identical, confidence scores differ substantially.
Lattice-only produces higher filled scores and wider filled/blank
separation, making the system more robust to threshold perturbations.

| Mode | Filled mean | Filled min | Blank max | Gap | Margin |
| --- | --- | --- | --- | --- | --- |
| lattice | 0.496 | 0.206 | 0.040 | 0.166 | 80.6% |
| ncc+sobel | 0.369 | 0.098 | 0.046 | 0.052 | 53.1% |
| ncc | 0.369 | 0.098 | 0.046 | 0.052 | 53.1% |

- **Gap** = filled min - blank max. This is the safety margin between the
  weakest filled question and the noisiest blank question. A larger gap
  means the adaptive threshold has more room to land correctly.
- **Margin** = gap / filled min as a percentage. Lattice-only has 80.6%
  margin vs 53.1% for NCC modes.

The lattice-only filled minimum (0.206) is more than double the NCC
filled minimum (0.098). NCC refinement compresses the score range
for filled bubbles, bringing the weakest filled questions dangerously
close to the blank noise floor.

### Score degradation pattern

Mean per-question score difference between lattice and ncc+sobel
is 0.127 (stdev 0.065). Every single filled question scored lower
with NCC refinement -- not a single question improved. The top 10
largest drops:

| Question | Lattice | NCC+Sobel | Delta | Answer |
| --- | --- | --- | --- | --- |
| Q35 | 0.724 | 0.453 | -0.271 | D |
| Q25 | 0.644 | 0.392 | -0.252 | D |
| Q29 | 0.601 | 0.363 | -0.238 | B |
| Q31 | 0.592 | 0.371 | -0.221 | B |
| Q41 | 0.609 | 0.391 | -0.218 | D |
| Q18 | 0.515 | 0.303 | -0.212 | D |
| Q30 | 0.625 | 0.413 | -0.212 | B |
| Q27 | 0.614 | 0.409 | -0.205 | C |
| Q33 | 0.574 | 0.375 | -0.199 | C |
| Q40 | 0.551 | 0.353 | -0.198 | D |

The largest drops cluster on answers B, C, D -- letters with prominent
vertical strokes. This is consistent with NCC locking onto letter
glyph features rather than bracket edges, shifting the measurement
window off-center.

### NCC diagnostic counters

```
  NCC refinement: 500 slots, 500 accepted (conf>=0.45),
    0 conf-rejected, 0 shift-rejected
  Mean |dx|=4.4px  Mean |dy|=5.6px
```

All 500 slots passed the confidence threshold (>= 0.45) and the
shift limit. The mean shifts of 4.4px horizontal and 5.6px vertical
are large relative to the measurement zone geometry:

- `col_pitch` = 103.2px, so 4.4px is a 4.3% column shift
- `bracket_inner_half` = 32.0px, so 4.4px shifts the fill window
  edge by 13.8% of its half-width
- `row_pitch` = 51.0px, so 5.6px is an 11.0% row shift

These are not small subpixel corrections. The NCC stage is applying
large systematic shifts that move measurement windows away from the
lattice-derived centers.

### Sobel independence test

NCC+Sobel and NCC-only produce byte-identical CSVs. The Sobel stage
has zero effect on the output. This happens because Sobel requires
two strong edge peaks separated by 70-130% of expected slot width.
After NCC has already shifted positions, the Sobel search ROI
is centered on a shifted location where the bracket edges may be
asymmetric or attenuated, causing the separation validation
(deviation >= 0.30) to reject and fall back to the NCC position.

The Sobel stage is effectively dead code on this image.

## Results: 804D5A50

### Answer accuracy

The second key image reveals a catastrophic failure mode for NCC
refinement.

| Mode | Filled | Blank | Expected filled |
| --- | --- | --- | --- |
| lattice | 69 | 31 | 69 |
| ncc+sobel | 3 | 97 | 69 |
| ncc | 3 | 97 | 69 |

NCC refinement causes 66 out of 69 filled answers to be misread
as BLANK. Only 3 questions survive. The lattice-only baseline
reads all 69 correctly.

### Mechanism of failure

NCC shifts were similar in magnitude to the first image:

```
  NCC refinement: 500 slots, 500 accepted (conf>=0.45),
    0 conf-rejected, 0 shift-rejected
  Mean |dx|=4.2px  Mean |dy|=5.3px
```

Again, 100% acceptance rate with no rejections. The NCC confidence
threshold (0.45) is too permissive -- it accepts shifts that move
measurement windows far enough off-center that filled bubbles lose
their fill signal.

The failure mode is: NCC matches the letter glyph template against
the printed (unfilled) bracket, finds a plausible match, and shifts
the position to align with the glyph. But the glyph center is not
the measurement center. The fill zones (green windows between bracket
bars and center exclusion) end up sampling partially outside the
bracket, picking up white background instead of pencil fill.

With enough windows shifted, the adaptive threshold recalculates
using the compressed score distribution and concludes most questions
are blank.

### Sobel independence (confirmed)

NCC+Sobel and NCC-only are again byte-identical. Sobel contributes
nothing on either test image.

## Conclusions

### Finding 1: Sobel x-edge refinement is dead code

On both test images, the Sobel stage produces zero effect. Its
edge-pair separation validation (30% deviation threshold) rejects
every candidate and falls back to the NCC/lattice position. The
stage adds computation time with no benefit.

**Recommendation**: Remove the Sobel stage entirely or disable it
by default. It can be preserved behind a flag for future
experimentation if bracket edge detection is revisited with
different parameters.

### Finding 2: NCC refinement degrades scoring reliability

On 43F257A7, NCC reduces the filled/blank separation gap from
0.166 to 0.052 -- a 3.2x reduction in safety margin. Every filled
question scores lower with NCC. The system still produces correct
answers, but is operating with much less margin.

On 804D5A50, NCC causes 66/69 filled answers to be misread as
blank. This is a total scoring failure caused entirely by the
refinement stage.

**Recommendation**: Switch the default `refine_mode` to `"lattice"`.
The lattice positions derived from timing marks are already
accurate enough for correct scoring. NCC refinement should be
disabled by default and used only as an experimental option.

### Finding 3: NCC confidence threshold is too permissive

The 0.45 confidence threshold accepts 100% of slots on both images,
including shifts of 4-6 pixels that demonstrably harm scoring.
The template matching is not discriminating between good and bad
matches -- it treats all matches as equally trustworthy.

If NCC refinement is retained for future use, the confidence
threshold needs to be raised substantially (perhaps 0.7+) and
the maximum shift limit reduced. Alternatively, NCC could be
used only for y-axis refinement where its shifts are less
damaging to the horizontal measurement windows.

### Finding 4: lattice-only is the best current mode

Pure lattice geometry from timing marks produces:
- Correct answers on both test images
- Highest confidence scores for filled bubbles
- Widest separation between filled and blank populations
- No dependency on template files or refinement parameters

The lattice is already doing the hard work of establishing accurate
bubble positions. Adding refinement stages on top introduces risk
without measurable benefit at current image quality levels.

## Experiment infrastructure

The experiment mode is available for future A/B testing via:

```bash
# lattice-only (recommended default)
python run_pipeline.py -k key.jpg -i scans/ -o output/ -d -r lattice

# NCC + Sobel (legacy behavior)
python run_pipeline.py -k key.jpg -i scans/ -o output/ -d -r ncc+sobel

# NCC only (no Sobel)
python run_pipeline.py -k key.jpg -i scans/ -o output/ -d -r ncc
```

NCC diagnostic counters print automatically when NCC is active:

```
  NCC refinement: 500 slots, 480 accepted (conf>=0.45),
    12 conf-rejected, 8 shift-rejected
  Mean |dx|=0.4px  Mean |dy|=0.3px
```

Debug overlays (`_scored.png`, `_debug.png`) include confidence
tier dots when refinement confidence propagation is active.

## Files modified

- [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) -- `refine_mode`
  parameter, `skip_sobel` parameter, NCC diagnostic counters,
  confidence propagation fix
- [run_pipeline.py](../run_pipeline.py) -- `-r`/`--refine-mode` CLI argument
- [docs/CHANGELOG.md](../docs/CHANGELOG.md) -- documented changes and findings
