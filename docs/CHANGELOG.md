# Changelog

## 2026-03-05

### Additions and New Features

- Added `_default_bounds()` helper in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to centralize float-to-int conversion for bubble edge positions; used by refinement, validation, scoring, and drawing functions
- Added `_check_row_linearity()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) with median-based inlier/outlier detection; identifies choices where Sobel-y locked onto the wrong edge pair by checking if their y-centers deviate from the row median, then fits a line through inliers to predict corrected positions
- Added `_check_column_alignment()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to verify x-center consistency within each choice column; flags positions that deviate from the column median
- Added `_check_row_brightness()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to detect rows where all 5 choices measured as white (edge_mean > 220), indicating the measurement zone landed on the white gap between rows instead of on actual bubbles
- Added regression test `test_all_students_detect_at_least_71` in [tests/test_bubble_reader.py](../tests/test_bubble_reader.py) verifying all student images (including phone photos) detect >= 71 answers
- Added `TestRowLinearity` tests: aligned row produces no outliers, single outlier detected, two minority outliers detected while majority preserved
- Added `TestRowBrightness` tests: normal mixed-brightness row passes, all-white row is flagged

### Behavior or Interface Changes

- Bubble geometry values in [omr_utils/template_loader.py](../omr_utils/template_loader.py) `get_bubble_geometry_px()` now return float values for sub-pixel precision in validation math; consumers use `_default_bounds()` for integer conversion at array-slicing boundaries
- Updated `half_height` from 0.00273 to 0.00250 in [config/dl1200_template.yaml](../config/dl1200_template.yaml) for closer match to physical ~6:1 bubble aspect ratio (5.5px at canonical 2200h)
- Updated `measurement_inset_v` from 0.00136 to 0.00091 (3px to 2px at canonical) for wider 8px measurement zone (was 6px)
- Tightened `_validate_bubble_rect()` height deviation threshold from 50% to 40% and added explicit aspect ratio check (5.0-6.5 range)
- `read_answers()` now applies three post-refinement validation passes: row linearity correction, edge mean measurement, and brightness sanity check with automatic re-measurement for all-white rows

### Fixes and Maintenance

- Fixed phone photo (8B5D0C61) detecting only 69/71: Q44 and Q49 measurement zones were landing in the white gap between adjacent rows because Sobel-y found bracket edges from neighboring rows; row linearity check identifies these misplaced detections and corrects them using the fitted line from correctly-detected choices; now all 4 images detect 71/71

### Decisions and Failures

- Chose median-based inlier identification over pure line-fit for row linearity: fitting a line to all 5 points fails when 2 of 5 are wrong (the fitted line gets pulled to an intermediate position, flagging all 5 as outliers); median is robust to minority outliers and correctly identifies the 2-3 bad choices
- Float geometry with `_default_bounds()` helper chosen over scattered `int()` casts or int-only geometry; `int(cy - 5.5) = int(44.5) = 44` gives 11px total height which couldn't be achieved with int half_height (either 5 or 6, giving 10 or 12px)

### Developer Tests and Notes

- 269 tests pass across all test files (was 263)
- Detection counts: answer key 71/71, phone photo 71/71 (was 69), both flatbed scans 71/71
- All pyflakes clean

---

### Additions and New Features

- Added `_validate_bubble_rect()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) for post-refinement area and aspect ratio validation; rejects detected rectangles with width, height, or area deviating beyond acceptable ranges and falls back to template defaults
- Added `TestValidateBubbleRect` tests in [tests/test_bubble_reader.py](../tests/test_bubble_reader.py) for correct-size pass-through, half-width rejection, and half-height rejection

### Behavior or Interface Changes

- Tightened x-edge separation validation threshold in `_refine_bubble_edges_x()` from 60% to 30% deviation; internal letter strokes (e.g. vertical bars in 'E' and 'C') were producing half-width rectangles on phone photos
- `read_answers()` now calls `_validate_bubble_rect()` after both x and y edge refinement to catch any remaining area/aspect anomalies

### Fixes and Maintenance

- Fixed 53 half-width bubble detections on phone photo (8B5D0C61): E and C choices were detected at 58-68% of expected width because Sobel-x found letter strokes instead of bracket arms; now fall back to template defaults

### Decisions and Failures

- Chose 30% x-deviation threshold based on analysis: bad detections clustered at 35-44px (58-73% of 60px expected), good detections at 55-64px (92-107%); 30% cleanly separates the two populations
- Y-edge threshold kept at 60% because the smaller expected separation (12px) and lack of internal horizontal features means false matches are only from column headers (which have much larger separation)

### Developer Tests and Notes

- 263 tests pass across all test files
- Detection counts unchanged: answer key 71/71, phone photo 69/71, both flatbed scans 71/71
- Zero narrow bubble detections on all 4 images after fix (was 53 on phone photo)

---

### Additions and New Features

- Added `bubble_geometry` section to [config/dl1200_template.yaml](../config/dl1200_template.yaml) with normalized dimensions for all bubble measurement parameters
- Added `get_bubble_geometry_px()` to [omr_utils/template_loader.py](../omr_utils/template_loader.py) to convert normalized bubble geometry to pixel values for any image size, with fallback defaults
- Added `_refine_bubble_edges_y()` and `_refine_bubble_edges_x()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) that return detected edge positions `(center, top, bot)` and `(center, left, right)` instead of just a refined center
- Added edge separation validation: rejects Sobel edge pairs whose separation deviates more than 60% from expected bubble height, preventing column-header edges from being mistaken for bubble bracket edges at Q1 and Q51
- Added neighbor-based y-correction in `read_answers()`: when edge detection fails for a bubble (e.g., at column-top positions), the median y-shift from intra-row refined choices and nearby successfully-refined questions is applied
- Added `min_spread_floor` parameter (default 15.0) and gap significance check to `_find_adaptive_threshold()` to prevent low-contrast images from producing thresholds below the noise level
- Added `_default_geom()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) for backward-compatible default geometry when no template is provided
- Added regression tests: `test_key_detects_at_least_71`, `test_key_q1_detected`, edge separation validation test, and adaptive threshold floor/unimodal tests in [tests/test_bubble_reader.py](../tests/test_bubble_reader.py)
- Results from `read_answers()` now include an `edges` key mapping each choice to its detected `(top_y, bot_y, left_x, right_x)` edge positions

### Behavior or Interface Changes

- Replaced module-level pixel constants (`BUBBLE_HALF_WIDTH`, `BUBBLE_HALF_HEIGHT`, etc.) with dynamic `geom` dict computed from template YAML; all measurement functions now accept detected edge positions and geometry as parameters
- `_compute_edge_mean()` and `_compute_bracket_edge_mean()` now compute measurement zones from actual detected edges (`top_y`, `bot_y`, `left_x`, `right_x`) rather than hardcoded offsets from center
- `score_bubble_fast()` now accepts an optional `geom` dict parameter (default `None` for backward compatibility)
- `draw_answer_debug()` now draws overlays using detected edge positions from results, accurately reflecting the measured zones per bubble
- [omr_utils/student_id_reader.py](../omr_utils/student_id_reader.py) now passes geometry to `score_bubble_fast()`

### Fixes and Maintenance

- Fixed answer key detecting only 66/71 answers: Q1, Q2, Q51, Q52, Q53 are now correctly detected using neighbor-based y-correction and edge separation validation
- Fixed phone photo (8B5D0C61) detecting 68/71: improved to 69/71 (Q44 and Q49 remain marginal due to accumulated drift at column bottom)
- Answer key scores now use correct /71 denominator instead of /66

### Decisions and Failures

- Chose half_height=6px (matching physical bracket height) over the planned 10px because the enlarged 10px measurement zone diluted fill signal when edge detection fell back to defaults, causing most questions to be marked blank (spreads of 30-130 instead of 190+)
- Column-top question detection (Q1, Q51) required a two-part solution: (1) edge separation validation to reject column header edges, AND (2) neighbor-based y-correction to fix fallback positions that were ~8px too low

### Developer Tests and Notes

- 260 tests pass across all test files
- Answer key: 71/71 detected (was 66/71), phone photo: 69/71 (was 68/71), both flatbed scans: 71/71 (unchanged)
- Pipeline verified with debug overlays; all 4 scantron images process correctly

---

### Additions and New Features

- Created full OMR pipeline for Apperson DataLink 1200 bubble sheets
- Added `tests/conftest.py` using `git_file_utils.get_repo_root()` so `pytest tests/` works without `source source_me.sh`
- Added [config/dl1200_template.yaml](../config/dl1200_template.yaml) with calibrated form geometry using normalized coordinates (0.0-1.0)
- Added [omr_utils/template_loader.py](../omr_utils/template_loader.py) for YAML template loading and coordinate computation
- Added [omr_utils/image_registration.py](../omr_utils/image_registration.py) for page detection, perspective warp, and orientation correction
- Added [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) with grayscale intensity-based bubble scoring and gap-based BLANK/MULTIPLE detection
- Added [omr_utils/student_id_reader.py](../omr_utils/student_id_reader.py) for 9-digit student ID grid reading
- Added [omr_utils/csv_writer.py](../omr_utils/csv_writer.py) for answers CSV output with per-question confidence scores
- Added [register_scan.py](../register_scan.py) CLI for image registration
- Added [extract_answers.py](../extract_answers.py) CLI for answer extraction (auto-registers raw images)
- Added [grade_answers.py](../grade_answers.py) CLI for grading student answers against a key
- Added [run_pipeline.py](../run_pipeline.py) batch CLI for end-to-end processing of image directories
- Added [pyproject.toml](../pyproject.toml) with project metadata and dependencies
- Added [VERSION](../VERSION) file (26.03)
- Added test suite: template loader, image registration, bubble reader, grading, and pipeline smoke tests
- Added documentation: CODE_ARCHITECTURE, FILE_STRUCTURE, INSTALL, USAGE, INPUT_FORMATS, OUTPUT_FORMATS, TROUBLESHOOTING
- Updated [README.md](../README.md) with project purpose, quick start, and documentation links
- Added [omr_utils/xlsx_writer.py](../omr_utils/xlsx_writer.py) for consolidated XLSX scoring summary with Summary, Detailed Grades, Student Answers, and Question Analysis tabs
- Added bracket-edge dark reference measurement in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) with `_compute_bracket_edge_mean()` for per-bubble alignment and scoring baseline
- Added `openpyxl` to [pip_requirements.txt](../pip_requirements.txt)
- Added `_refine_row_center_y()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) for two-stage bubble alignment: coarse YAML template positioning followed by local Sobel-y edge detection to snap each bubble's y-coordinate to the actual printed bracket arms
- Per-choice y-refinement in `read_answers()` handles both cumulative template drift (~11px by row 49) and per-row rotation from scanning errors
- Results from `read_answers()` now include a `positions` key mapping each choice to its refined (px, py) pixel coordinates; `draw_answer_debug()` uses these refined positions

### Behavior or Interface Changes

- Changed `MEASUREMENT_INSET` from 0 to 2 in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to exclude dark bracket-edge pixels from fill measurement; based on SVG proportions analysis where bracket edge = 2px of the 12px bubble height
- Added `MEASUREMENT_INSET_H = 2` for horizontal inset in `_compute_edge_mean()` to exclude dark printed vertical bracket borders from measurement strips
- Rewrote `draw_answer_debug()` to use semi-transparent filled overlays via `cv2.addWeighted()` (alpha=0.3) instead of outline-only rectangles; teal filled measurement strips are now distinctly visible and accurately reflect the inset measurement zone
- Removed red bracket-edge fills from debug overlay; teal measurement fills, orange center exclusion outline, and status-colored outer border remain

### Behavior or Interface Changes

- Simplified `draw_answer_debug()` overlay from six zone types (teal strips, four red corner boxes, orange center) to three clean elements per bubble: color-coded outer rectangle, two teal measurement strips, and orange center letter box
- Replaced background-relative bubble scoring with self-referencing scoring in `omr_utils/bubble_reader.py`; each question's lightest choice serves as the empty baseline, eliminating dependency on inter-row background strips that failed on phone photos with uneven lighting
- Added adaptive blank detection: the spread (max - min edge mean) across all 100 questions is sorted and the largest gap between consecutive values separates filled from blank populations automatically per image
- Removed `blank_gap` parameter from `read_answers()` (no longer needed; threshold is computed adaptively)
- Removed module-level `BG_GAP` and `BG_HEIGHT` constants (background strips no longer used in `read_answers`)
- `score_bubble_fast()` now uses bracket-edge dark reference instead of background strips above/below the bubble
- `_compute_edge_mean()` measurement zone kept at full bubble height (`MEASUREMENT_INSET=0`); narrowing the zone broke adaptive thresholding on phone photos by reducing spreads below the gap detection threshold
- [run_pipeline.py](../run_pipeline.py) now outputs `scoring_summary.xlsx` in the output directory after grading all students

### Fixes and Maintenance

- Fixed `tests/test_pipeline_smoke.py` to use correct answer key image (`43F257A7` instead of `8B5D0C61`)
- Removed hardcoded student scantron filenames from test files for FERPA compliance
- Student images now discovered dynamically from `scantrons/` directory in tests

### Decisions and Failures

- Chose grayscale intensity scoring over binary thresholding because adaptive threshold could not distinguish filled bubbles from printed bubble outlines
- Right column bubble x-positions were initially offset by one choice position; corrected after debug overlay analysis
- Student ID extraction produces incorrect results on some images; grid coordinates need further calibration
- Fixed 8B5D phone photo missing 10 answers (Q32, Q42-50): background reference strips overlapped adjacent rows in lower-left area due to slight registration drift; self-referencing scoring eliminated this failure mode
- Fixed 43F key false positives from DataLink machine-printed marks: adaptive threshold correctly separates blank rows (spread <35px) from filled rows (spread >58px) regardless of machine marks biasing the D column

### Developer Tests and Notes

- 60 tests pass across 5 test files (template loader, image registration, bubble reader, grading, pipeline smoke)
- Tests requiring scantron images skip automatically when `scantrons/` is absent
- Pipeline verified on 4 real scantron images: 1 answer key + 3 student sheets (2 flatbed scans, 1 phone photo)
