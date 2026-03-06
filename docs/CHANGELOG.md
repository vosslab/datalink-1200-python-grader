# Changelog

## 2026-03-06

### Additions and New Features

- Added `extract_roi_from_bounds()` to [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py). Bounds-based ROI extraction primitive that accepts explicit lattice bounds from `SlotMap.roi_bounds()`, adds proportional padding, clips to image edges. Replaces center+geom symmetric cropping.
- Created [omr_utils/timing_marks_left.py](../omr_utils/timing_marks_left.py) with 6 left-side timing mark detection functions extracted from `timing_mark_anchors.py`.
- Created [omr_utils/timing_marks_top.py](../omr_utils/timing_marks_top.py) with 6 top-side timing mark detection functions extracted from `timing_mark_anchors.py`.

### Behavior or Interface Changes

- `_extract_rois_from_scan()` in [tools/build_bubble_templates.py](../tools/build_bubble_templates.py) no longer calls `read_answers()`. Extraction iterates all 100 questions x 5 choices directly via `SlotMap.roi_bounds()`, removing the empty-only sampling policy. Expected yield increases from ~37 ROIs/scan to ~500 ROIs/scan.
- [tools/build_bubble_templates.py](../tools/build_bubble_templates.py) ROI extraction now uses `SlotMap.roi_bounds()` + `extract_roi_from_bounds()` instead of center+geom symmetric `extract_roi_1x()`. ROIs now respect lattice cell boundaries.
- [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py) split into three files: shared utilities remain in the main module, left-specific detection moved to `timing_marks_left.py`, top-specific detection moved to `timing_marks_top.py`. Pure code motion, no behavioral changes.

### Removals and Deprecations

- Removed `import omr_utils.bubble_reader` from [tools/build_bubble_templates.py](../tools/build_bubble_templates.py). Template extraction no longer depends on the answer-reading pipeline.
- Deleted `extract_roi_1x()` from [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py). Replaced by `extract_roi_from_bounds()` which uses lattice bounds instead of center+geom symmetric cropping.
- Deleted `extract_bubble_patch()` from [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py). Dead code, only called by removed `extract_letter_templates()`.
- Deleted `extract_letter_templates()` from [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py). Dead code with zero external callers.

### Previous Additions and New Features

- Created [omr_utils/slot_map.py](../omr_utils/slot_map.py) with `SlotMap` class as the single geometry authority for bubble slot positions. Builds pixel coordinates directly from timing mark anchors: row y from `left_question_marks`, column x from `top_fp_x0 + choice_columns * fine_step`. Provides `center()`, `row_center()`, `choice_center()`, `roi_bounds()`, and `geom()` methods.
- Added `draw_lattice_crosshairs()` to [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py). Draws small crosshairs at every `SlotMap.center()` position for independent geometry verification. Saved as `{base_name}_lattice.png` in debug mode.

### Behavior or Interface Changes

- `read_answers()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) now accepts a `slot_map` parameter (SlotMap instance). When provided, skips internal timing mark detection and uses the SlotMap directly. All geometry flows through SlotMap; no YAML coordinates are consulted for bubble placement.
- `_stage_localize_rows()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) simplified to pure lattice positions from SlotMap. No Sobel y-refinement, no neighbor correction, no linearity check, no affine fit.
- `_stage_measure_rows()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) simplified: removed bracket confidence fallback to affine position, column-lock correction pass, and all-white brightness sanity fallback. Uses SlotMap for lattice bounds.
- [run_pipeline.py](../run_pipeline.py) now constructs a `SlotMap` from `estimate_anchor_transform()` output and passes it to `read_answers()`. Single geometry authority for the entire pipeline.
- [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py): removed all `get_bubble_coords()`/`to_pixels()` fallback paths. Debug tools now visualize only the `positions` dict from results, never recompute coordinates from YAML. Removed `_compute_refinement_shift_data()` and shift vector drawing.
- [omr_utils/student_id_reader.py](../omr_utils/student_id_reader.py) now uses `SlotMap.geom()` instead of removed `anchor_geom()` for bubble geometry.

### Removals and Deprecations

- Removed from [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py): `_apply_anchor_transform()`, `_sanitize_anchor_transform()`, `lattice_slot_centers()`, `lattice_row_centers()`, `lattice_choice_centers()`, `lattice_roi_bounds()`, `_lattice_bounds_at_center()`, `anchor_geom()`, `_fit_affine_from_confident_detections()`. All absorbed into `SlotMap` class or removed as no longer needed.
- Removed from [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py): `_check_row_linearity()`, `_check_column_alignment()`, `_check_row_brightness()`, `_select_rect_by_bracket_signal()`. These correction passes are no longer needed with pure lattice geometry.
- Removed K-constants from [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py); moved to [omr_utils/slot_map.py](../omr_utils/slot_map.py) as the sole consumer.
- Removed `draw_lattice_centers_debug()` from [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py), replaced by `draw_lattice_crosshairs()` which uses SlotMap directly.

### Decisions and Failures

- Previous extraction only harvested EMPTY bubbles from `read_answers()` output, applying five cascading filters (MULTIPLE skip, empty-only, fill < 0.12, position check, quality > 10.0) that reduced 500 potential slots to ~37 per scan. Template extraction should sample ALL slots because the printed bracket shape is independent of fill state.
- Architecture decision: SlotMap is the single source of truth for all bubble geometry. No YAML coordinates, no affine correction, no Sobel y-refinement for initial placement. Local x-edge refinement within the lattice ROI is still performed for precise measurement zones.
- The affine fit from confident Sobel detections (`_fit_affine_from_confident_detections()`) was the core contamination source that caused 0/99 correct results. It overwrote all positions with YAML-derived predictions, defeating the lattice-based placement.

### Developer Tests and Notes

- Updated tests in [tests/test_bubble_template_extractor.py](../tests/test_bubble_template_extractor.py): replaced `anchor_geom()` tests with `SlotMap.geom()` equivalents. Added `_make_mock_transform()` and `_make_mock_template()` helpers.
- Updated [tools/build_bubble_templates.py](../tools/build_bubble_templates.py), [tools/diag_roi_centers.py](../tools/diag_roi_centers.py), [tools/diag_roi_size.py](../tools/diag_roi_size.py) to use SlotMap instead of removed functions.

### Previous Additions and New Features

- Added `lattice_slot_centers()`, `lattice_row_centers()`, and `lattice_choice_centers()` to [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py). These functions return slot centers derived entirely from timing-mark lattice data (top footprint x0 + column index * fine step for x, left question marks center_y for y). No YAML `choice_x` normalized coordinates are consulted.
- Added `lattice_roi_bounds()` to [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py). Computes asymmetric ROI boundaries from midpoints between neighboring lattice centers. Edge choices (A/E) and edge rows (first/last) extrapolate from the nearest interior gap. Replaces symmetric `_default_bounds()` construction.
- Added `_lattice_bounds_at_center()` helper to [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py). Preserves asymmetric lattice ROI sizing while allowing refined (shifted) center positions.
- Added `draw_lattice_centers_debug()` to [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py). Draws green dots at every lattice-derived slot center with Q1/Q51 labels for visual verification.
- Added base template bootstrap in [omr_utils/template_matcher.py](../omr_utils/template_matcher.py). `try_load_bubble_templates()` now falls back to `artifacts/base_letter_template.png` for any letter missing a per-letter auto-built template. Per-letter templates in `config/bubble_templates/` take priority when present.

### Behavior or Interface Changes

- `_stage_localize_rows()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) now uses `lattice_slot_centers()` as sole source of slot placement. Returns `(raw_data, lattice_ctx)` tuple instead of bare `raw_data` list.
- Bubble ROI x-centers in `_stage_localize_rows()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) are now derived from the timing-mark lattice (`fp_x0 + col_idx * fine_step`) instead of YAML normalized coordinates. Integer `choice_columns` in the YAML define which column in the 53-column grid each choice maps to, eliminating fractional placements that landed between grid lines.
- All ROI bound computations throughout the bubble reading pipeline now use lattice-derived asymmetric bounds instead of symmetric K-constant half_width/half_height construction.
- `_refine_bubble_edges_y()` and `_refine_bubble_edges_x()` now accept optional `default_top`/`default_bot` and `default_left`/`default_right` parameters for caller-supplied fallback bounds.
- `_validate_bubble_rect()` now accepts optional `fallback_bounds` parameter.
- `_select_rect_by_bracket_signal()` now accepts optional `lattice_bounds` parameter.
- [config/dl1200_template.yaml](../config/dl1200_template.yaml): removed `choice_x` normalized coordinate mappings from both left and right columns. Lattice-based placement uses `choice_columns` integer indices exclusively. Template loader derives `choice_x` from `choice_columns` when needed for backward compatibility.
- `estimate_anchor_transform()` in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py) now stores `top_fp_x0` (lattice origin) and `top_col_ratio` (integer footprint-to-fine ratio) in the transform dict for downstream lattice-based center derivation.
- Added `choice_columns` mappings to left and right column sections in [config/dl1200_template.yaml](../config/dl1200_template.yaml) - integer column indices (A=5/24, B=8/27, C=10/30, D=13/33, E=16/36) in the 53-column top timing mark grid.

### Removals and Deprecations

- Removed `_default_bounds()` from [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) and [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py). All ROI placement now uses `lattice_roi_bounds()` or `_lattice_bounds_at_center()`. Debug drawing uses `_geom_bounds()` for display-only fallbacks.
- Removed `compute_choice_half_widths()`, `compute_row_half_heights()`, and `_compute_column_row_half_heights()` from [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py). The timing-mark grid is regular so symmetric `col_pitch/2` and `row_pitch/2` half-dimensions suffice; Voronoi-style asymmetric boundaries were over-engineered.
- Simplified `extract_roi_1x()` in [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py): removed `half_w_left`, `half_w_right`, `half_h_top`, `half_h_bottom` parameters; now uses symmetric half-widths from geom dict.
- Simplified ROI extraction in [tools/build_bubble_templates.py](../tools/build_bubble_templates.py): removed Voronoi half-width/half-height computation, now calls `extract_roi_1x()` with just geom.

### Fixes and Maintenance

- Fixed bubble template ROI capturing two slots instead of one: `top_col_spacing` in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py) was storing the coarse footprint mark spacing (~104 px) instead of the fine template-column step (~35 px). The footprint detector correctly identifies coarse marks, but the fine-grid step is what downstream bubble geometry needs. Now derives `fine_col_step = fp_spacing / round(col_ratio)` before storing.
- Removed pixel fallback branch from `anchor_geom()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py). Now raises `ValueError` when anchor-derived spacing is unavailable instead of silently using hardcoded pixel defaults. All runtime geometry is expressed as dimensionless fractions of anchor-derived spacing.
- Updated `read_student_id()` and `read_student_id_detailed()` in [omr_utils/student_id_reader.py](../omr_utils/student_id_reader.py) to accept a `transform` parameter instead of calling `anchor_geom({})` with empty dict.
- Updated [run_pipeline.py](../run_pipeline.py) to always compute timing mark transform (not just in debug mode), so student ID reader has valid anchor-derived geometry.
- Fixed float-to-int conversion error in `draw_answer_debug()` in [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py) for template shift line/circle drawing.
- Updated [tools/build_bubble_templates.py](../tools/build_bubble_templates.py) to catch `ValueError` from `anchor_geom()` and skip scans with invalid timing marks instead of crashing.

### Decisions and Failures

- Root cause of two-slot ROIs: the footprint model's `_approx_gcd_spacing()` correctly finds the coarse mark spacing, but the coarse marks in Row-1 are spaced at every Nth fine-grid column. The fine-grid marks are not present in Row-1 data, so no GCD trick can recover the fine step. The fix is semantic: store the fine template-column step (derived from `fp_spacing / round(col_ratio)`) rather than the raw footprint spacing.

### Additions and New Features

- Added offline bubble template construction pipeline in [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py): `extract_roi_1x()` for native-resolution ROI extraction, `_find_medoid_roi()` for selecting most representative ROI, `_align_roi_to_reference()` for translation-only NCC alignment, `_apply_symmetry_augmentation()` for letter-specific mirroring (A=LR, B-E=TB), `_build_letter_template()` for aligned averaging with outlier rejection, `_generate_template_mask()` for bracket-emphasis mask derivation
- New `load_templates_and_masks()` and `save_templates_and_masks()` in [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py) for loading/saving template + mask file pairs (`{letter}.png` and `{letter}_mask.png`)
- New `_save_qc_montage()` in [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py) generates multi-panel QC images showing original ROIs with kept/rejected borders, plus final template and mask
- Added masked NCC matching in [omr_utils/template_matcher.py](../omr_utils/template_matcher.py): `match_bubble_masked()` applies bracket-emphasis mask to template before correlation, uses `TM_CCORR_NORMED` with mask parameter, returns 5-tuple with subpixel-refined position and shift values
- New `_subpixel_peak()` in [omr_utils/template_matcher.py](../omr_utils/template_matcher.py) fits quadratic to 3x3 NCC neighborhood for fractional peak estimation, clamped to +/-0.5px
- Created [tools/build_bubble_templates.py](../tools/build_bubble_templates.py) offline side script: two-pass pipeline (ROI extraction + template construction) from scanned images, accepts `--input-dir`, `--output-dir`, `--template`, and `--dry-run` flags
- Added confidence tier visualization in [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py): small colored dots at bubble corners indicate refinement confidence (green >= 0.6, yellow 0.3-0.6, red < 0.3)
- Created [tests/test_template_matcher.py](../tests/test_template_matcher.py) with 8 tests covering subpixel peak refinement, masked NCC matching, out-of-bounds handling, mask parameter threading, and return type verification
- Created [tests/test_bubble_template_extractor.py](../tests/test_bubble_template_extractor.py) with 23 tests covering ROI extraction, medoid selection, alignment, symmetry augmentation (LR and TB), template building, mask generation, patch quality scoring, template/mask loading, and anchor geometry derivation
- New `anchor_geom(transform)` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) derives all bubble slot geometry from anchor-measured timing spacing using dimensionless fractions (`_K_HALF_HEIGHT`, `_K_HALF_WIDTH`, etc.); replaces fixed pixel assumptions with scale-invariant ratios of row_pitch and col_pitch
- `CANONICAL_TEMPLATE_WIDTH` and `CANONICAL_TEMPLATE_HEIGHT` constants in [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py) define the fixed high-resolution grid (480x88) for offline templates; runtime scaling always goes DOWN from this canonical size
- `estimate_anchor_transform()` in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py) now stores `top_col_spacing` in the transform dict, exposing the top footprint column spacing for downstream geometry derivation

### Fixes and Maintenance

- Fixed `_sanitize_anchor_transform()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) stripping `left_s_q` and `top_col_spacing` keys from the transform dict, which caused `anchor_geom()` to always fall back to fixed defaults. These timing-mark spacing keys are now passed through.
- Fixed float-to-int slice errors in `_compute_bracket_edge_mean()` and `_compute_dual_zone_means()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py); anchor-derived geometry produces float values that need explicit `int()` casts before numpy array slicing.

### Behavior or Interface Changes

- `read_answers()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) now computes geometry via `anchor_geom(transform)` instead of `default_geom()`; bubble dimensions scale with timing-mark spacing rather than using hardcoded pixel values. Falls back to fixed defaults when anchor spacing is unavailable.
- Removed `default_geom()` from [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py); its fixed-pixel fallback values are now inlined in `anchor_geom()`. All callers (`score_bubble_fast`, `student_id_reader`, `debug_drawing`, `bubble_template_extractor`) updated to use `anchor_geom()`.
- `draw_answer_debug()`, `draw_scored_overlay()`, and `draw_combined_debug()` in [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py) now require a `geom` parameter instead of hardcoding geometry.
- `extract_letter_templates()` in [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py) now requires a `geom` parameter.
- [tools/build_bubble_templates.py](../tools/build_bubble_templates.py) now computes anchor geometry from timing marks for each scan, so ROI extraction uses the correct per-scan bubble dimensions.
- `_build_letter_template()` in [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py) now upscales all aligned ROIs to canonical high resolution (480x88) before averaging, so master templates are always higher-res than any individual source scan
- `scale_template_to_bubble()` docstring updated to reflect canonical-template-to-local-slot scaling design; target dimensions now come from anchor-derived geometry when available
- `try_load_bubble_templates()` in [omr_utils/template_matcher.py](../omr_utils/template_matcher.py) now returns `(templates_dict, masks_dict)` tuple instead of a plain dict; callers must destructure the return value
- `refine_row_by_template()` in [omr_utils/template_matcher.py](../omr_utils/template_matcher.py) accepts optional `masks` parameter; when provided, uses masked NCC (`match_bubble_masked()`) instead of unmasked `match_bubble_local()`
- `_stage_template_refine()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) accepts optional `bubble_masks` parameter and stores `refinement_confidence` per choice in raw_data
- `read_answers()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) accepts optional `bubble_masks` parameter; auto-loads masks alongside templates when not provided

- Replaced legacy single-linear left-side timing detection with piecewise 3-segment structural fitter (2 top + 10 ID + 50 question = 62 marks) in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py)
- New `_extract_left_candidates()` extracts dash-like candidates from the left strip using Otsu + morphology with relative area thresholds
- New `_build_left_vertical_family()` filters candidates to a single dominant vertical column by x-center consistency and height similarity
- New `_match_predictions_to_marks_y()` performs 1:1 greedy matching of predicted y-positions to observed marks
- New `_fit_left_footprint()` implements blob-find-and-fit 3-segment fitting: repairs missing marks, splits at the largest gap (between ID and questions), generates per-segment predictions using median spacing, and scores against observed marks
- New `_repair_gaps()` interpolates missing marks where gap exceeds 1.4x median spacing
- New `_score_left_footprint()` applies hard acceptance gates (match count, max residual, spacing CV) then continuous scoring (match fraction, residual quality, spacing consistency)
- Module constants `N_TOP=2`, `N_ID=10`, `N_Q=50`, `N_TOTAL=62` define the DataLink 1200 left-side structure
- Guide lines in `draw_timing_mark_debug()` now derive exclusively from the fitted 50-question segment, using direct 0-indexed access (Q5 = index 4, Q50 = index 49) instead of the old bottom-anchored global formula
- Left marks in debug overlay now colored by segment: red (top 2), green (ID 10), cyan (question 50), with gap-A and gap-B boundary labels
- Transform dict now includes structural keys: `left_top_marks`, `left_id_marks`, `left_question_marks`, `left_s_id`, `left_s_q`, `left_gap_a`, `left_gap_b`, `left_raw_candidates`
- Added 9 new tests in `TestLeftFootprint` class covering candidate extraction, vertical family filtering, y-axis matching, synthetic 62-mark fitting, full pipeline integration, ordering, edge cases, and score rejection
- Added top timing footprint detection with Row-2 verification in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py): new `_approx_gcd_spacing()`, `_fit_row1_model()`, `_predict_row2_right_thins()`, `_match_predictions_to_marks()`, `_score_footprint_hypothesis()`, and `_detect_top_footprint()` functions implement the 10-step footprint detection algorithm
- `_fit_row1_model()` infers base column spacing from any 3 seed blobs using approximate GCD, then predicts column positions across the full strip width
- `_predict_row2_right_thins()` predicts the two right-side Row-2 thin marks (gap-thin-gap-thin pattern) relative to the matched Row-1 footprint extent, not the full strip width
- `_score_footprint_hypothesis()` combines 0.65 * Row-1 coverage + 0.35 * Row-2 thin mark support, with gap irregularity penalty and count bonus, capped at 1.0
- Footprint-based x-axis transform: maps detected marks to their correct template column indices using the fitted spacing model, enabling polyfit across the 53-column grid; achieves `top_confidence=1.000` on all 9 test images (previously 0.000)
- Enhanced cluster diagnostics: prints median component size (width x height) and area per row cluster for easier visual debugging
- Added horizontal dashed guide lines in `draw_timing_mark_debug()` for every-fifth question row pair (Q5/Q55 through Q50/Q100), anchored from the bottom (last left mark = Q50) to avoid cumulative offset errors
- Improved candidate debug overlay colors with 6 distinct cluster colors and row summary labels at left edge

### Behavior or Interface Changes

- Left-side detection in `estimate_anchor_transform()` no longer computes `y_scale`/`y_offset` from a single linear fit; it populates structural segment data instead (y-axis transform was already disabled downstream)
- Removed legacy `_estimate_axis_transform()` call for left marks; the structural fitter is now the sole left-side source of truth
- Old global-index guide line formula `mark_idx = last_idx - (50 - q_num)` replaced with direct question-segment indexing `mark_idx = q_num - 1`
- Failed left fit now produces `left_confidence = 0.0` and no guide lines instead of silently wrong guide lines
- `test_recovers_left_and_top_axis_transform` renamed to `test_recovers_top_axis_transform` since left side no longer produces linear transform
- Consolidated debug output from 4 images per scan (`_timing_candidates.png`, `_timing_final.png`, `_registered.png`, `_answers.png`) to 2 images: `_scored.png` (bubble status with confidence) and `_debug.png` (all layers combined: timing candidates, final marks, guide lines, and bubble overlays)
- Added `draw_scored_overlay()` and `draw_combined_debug()` to [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py) for the consolidated output
- Added Row-2 thin mark boxes (columns 10 and 12) to `draw_timing_mark_debug()` debug overlay, labeled "R2" in cyan-blue
- Reduced top detection strip from 10% to 6% of image height; reduces noise candidates while retaining all footprint rows
- Made strip region overlays more transparent (alpha 0.07 from 0.15) with separate muted fill and bright outline colors so underlying marks remain visible
- Darkened horizontal guide line color; labels now show left-column Q# at left edge and right-column Q# at far right edge
- `_detect_top_primary_row()` now uses footprint detection with Row-2 verification when available (score > 0.20), falling back to cluster-score selection when the footprint fails
- Footprint detection iterates only clusters with `_score_timing_row() >= 0.40` to avoid wasting time on noise clusters
- Row-1 score uses observed-coverage metric (matched/observed) instead of prediction-coverage (matched/predicted), preventing false selection of sparse wrong-row clusters
- Guide line question mapping anchored from bottom (last mark = Q50) instead of top (mark index 10 = Q1) for more stable alignment

### Decisions and Failures

- Added row-pattern timing anchor detection for top strip in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py): new `_extract_components()`, `_cluster_components_into_rows()`, `_score_timing_row()`, `_dedupe_row_components()`, and `_detect_top_primary_row()` functions replace the old fixed-threshold approach
- `_score_timing_row()` evaluates 7 weighted factors (component count, size consistency, fill consistency, fill magnitude, y-alignment, x-spacing regularity, aspect ratio consistency) to identify the primary timing row without forcing a specific mark count
- `_extract_components()` uses only relative thresholds (`area < strip_area * 0.0005`) instead of fixed pixel thresholds, making detection DPI-independent
- Added `_row_projection_bands()` to compute row-wise dark-pixel sums for localizing timing footprint bands; used as a structural cue (small scoring bonus) for row selection, not as the primary detector
- Debug overlay `draw_timing_candidates_debug()` now shows row cluster membership with different colors per cluster and cluster index labels (R0, R1, etc.)
- Debug overlay `draw_timing_mark_debug()` uses M1..Mn labels instead of T1..T7 since mark count is no longer forced

### Behavior or Interface Changes

- Changed top strip geometry from `[start_x..end_x, 0..5%h]` to `[0..w, 0..10%h]` (full width, top 10% height) for row-pattern detection; the row scorer now handles filtering instead of pre-cropping
- Changed left strip geometry from centered `4%` band around `left_edge.x` to `[0..8%w]` (left 8% width, full height)
- Adjusted `_estimate_axis_transform()` count adequacy threshold to scale with expected_count (`min(25, count*0.8)`) so small mark sets (7 top blocks) can reach full confidence

### Removals and Deprecations

- Removed `_detect_top_timing_blocks()`: replaced by `_detect_top_primary_row()` which uses row-pattern scoring instead of fixed area/fill thresholds and forced count of 7
- Removed `_find_best_y_cluster()`: replaced by `_cluster_components_into_rows()` which uses median-height-based gap splitting instead of sliding window
- Removed `_validate_final_blocks()`: replaced by row scoring in `_score_timing_row()`
- Removed `_dedupe_by_x_center()`: replaced by `_dedupe_row_components()`
- Removed `_dedupe_sorted()`, `_dedupe_sorted_marks()`, `_detect_marks_in_strip()`, and `_detect_centers_in_strip()`: dead code chain superseded by `_extract_components()` for top marks and `_extract_left_candidates()` for left marks

### Behavior or Interface Changes

- Changed timing mark detection in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py) to use local Otsu thresholding per strip instead of global Otsu on the full image; phone photos with uneven lighting now threshold correctly in each strip region
- Added morphological cleanup (open + close) before contour detection in `_detect_marks_in_strip()`: open removes small noise specks, close connects broken mark fragments
- Widened timing mark search strips from 3% to 4% of image dimension for better mark coverage without excess noise
- Changed `_detect_marks_in_strip()` to accept grayscale strips instead of pre-binarized strips; thresholding now happens locally inside the function
- Relaxed contour filters: minimum area lowered from 10 to 5 (helps low-res phone photos); left mark aspect ratio relaxed from w/h >= 1.5 to >= 1.2; top mark filter relaxed from h/w <= 3.0 to <= 4.0
- Replaced confidence metric in `_estimate_axis_transform()`: old metric used raw count ratio (unique/expected) which penalized forms with physical gaps in timing marks; new metric combines count adequacy (25 marks is sufficient) with span coverage (marks must span the expected range), giving a more accurate assessment of transform reliability
- Relaxed RMSE penalty threshold from 8.0 to 12.0 px: phone photos inherently have more geometric distortion (8.5px RMSE on a 2213px image is only 0.38% error), and the old threshold unfairly penalized otherwise reliable transforms

### Decisions and Failures

- M2 anchor experiment results: tested 3 approaches (A: widen+relax, B: adaptive threshold, C: morph cleanup) on all 4 test images; none passed alone. Approach A had best top coverage but left axis failed due to RMSE penalty. Approach B (adaptive threshold) helped KEY/8B5D but not phone photos. Approach C (morph only) was mixed. Winner: hybrid of A+C (local Otsu + morph cleanup + relaxed filters) combined with revised confidence metric
- Discovered indices 1, 38, 39 are consistently missing across ALL test images, indicating physical gaps in the DataLink 1200 form timing marks (not detection failures)
- Top timing mark row does not extend to x=0.96: right ~20% has Apperson branding with no marks. The expected_count=53 overcounts the actual physical marks (~50)
- Key image top marks harder to detect than student photos due to text ("VERIFY", "RESCORE") creating noise contours in the strip region

### Fixes and Maintenance

- Disabled y-axis timing mark transform pass-through in `_sanitize_anchor_transform()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py): the y-transform from left timing marks (scale=0.992, offset=+10px) conflicted with the affine fit from Sobel-detected edges, causing double-correction that pushed positions off bubbles and produced 13-21 all-white rows on some images; y_scale and y_offset now stay at identity, and the affine fit handles y-positioning alone
- Replaced linear affine fit with RANSAC + quadratic polynomial in `_fit_affine_from_confident_detections()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py): uses `cv2.estimateAffine2D(RANSAC)` to identify inliers (97% rate, 467/482), then fits `[tx, ty, ty^2, 1]` polynomial via lstsq on inliers only; the quadratic y-term captures barrel distortion that caused +/-2px sinusoidal y-residuals in the linear affine, fixing Q95-Q100 overlay drift

### Additions and New Features

- Added `_run_smoke_tests.sh` script to run `extract_answers.py` on all test images with debug overlays for quick visual verification
- Added affine fit diagnostic output: prints inlier count, median residual, and max residual after RANSAC affine fit

### Decisions and Failures

- Experiment 1 (disable y-transform) eliminated double-correction from timing marks conflicting with affine fit
- Experiment 2 (per-band y-residual diagnostics) revealed +/-2px sinusoidal pattern in linear affine residuals: classic barrel distortion that a linear model cannot capture
- Experiment 4 (RANSAC) used for outlier rejection; combined with quadratic polynomial fit to handle non-linear distortion
- Experiments 3 and 5 were not needed; Experiment 5 was already implemented as `_check_row_linearity()`
- Root cause: two problems stacked: (1) timing mark y-transform double-correcting on top of affine, (2) linear affine unable to model barrel distortion in y-axis

## 2026-03-05

### Additions and New Features

- Added aspect ratio filtering in `_detect_marks_in_strip()` in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py): left dashes require w/h >= 1.5, top boxes reject h/w > 3.0; prevents noise contours from wider search strips
- Added hard error check in `_stage_measure_rows()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py): raises `RuntimeError` when more than 25 rows have all-white measurement zones (all edge_means > 220), indicating catastrophic anchor failure
- Added anchor confidence diagnostic print in `read_answers()`: logs left/top confidence and mark counts after `estimate_anchor_transform()` for runtime visibility
- Added `_fit_affine_from_confident_detections()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to fit a full 2D affine transform from Sobel-confident bubble detections using `numpy.linalg.lstsq()`; extrapolates accurate positions to all 500 bubbles, fixing cumulative y-drift at Q86-Q100
- Added per-bubble bracket confidence check in `_stage_measure_rows()`: when bracket edge mean exceeds 200 (white paper), falls back to the affine-predicted position and re-measures
- Added timing mark debug overlay in [run_pipeline.py](../run_pipeline.py) and [extract_answers.py](../extract_answers.py): debug PNGs now show timing mark bounding boxes via `draw_timing_mark_debug()`
- Stored `template_px`/`template_py` (pre-transform positions) and `affine_px`/`affine_cy` (affine-predicted positions) in raw_data for fallback and debugging

### Behavior or Interface Changes

- Widened left timing mark search strip from 1% to 3% of image width (~34px to ~102px at 1700px) in [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py); fixes root cause of only 4/60 marks detected on real scans
- Widened top timing mark search strip from 1% to 3% of image height by the same factor
- Relaxed `min_marks` threshold in `_estimate_axis_transform()` from `max(8, count//4)` to `max(4, count//8)` and unique index gate from `max(4, ...)` to `max(3, ...)`; accepts transforms from fewer marks on difficult scans
- Passed `transform` to `_stage_measure_rows()` in `read_answers()` so column-lock correction can use top-anchor confidence
- Changed `_sanitize_anchor_transform()` to pass through timing mark scale+offset fully at confidence >= 0.50 instead of blending at 40% with 3% clamp; eliminates the root cause of cumulative y-drift at the bottom of each column
- Removed hardcoded pixel caps on `refine_max_shift`: `_refine_bubble_edges_y()` (was 8px), `_refine_bubble_edges_x()` (was 6px), neighbor correction (was 8px), and `_stage_template_refine()` (was 4px); all now use the geometry value directly
- Increased `refine_max_shift` default from 8.0 to 15.0 in `_default_geom()` and from `8/11` to `15/11` ratio in `template_loader.py`
- `_stage_template_refine()` now applies NCC y-corrections in addition to x-corrections, with bounds recomputed via `_default_bounds()`
- Relaxed `_validate_bubble_rect()` thresholds: width deviation 0.30->0.40, height deviation 0.40->0.50, aspect ratio 5.0-6.5->4.0-7.5
- Pipeline flow now includes affine fit stage between `_stage_localize_rows()` and `_stage_template_refine()`

### Fixes and Maintenance

- Fixed timing mark detection finding only 4/60 left marks due to 34px-wide search strip; widened to 102px, now detects all 60 left timing dashes with 0.983 confidence
- Fixed severe cumulative y-drift where detection rectangles were a full row above actual bubbles at Q94-Q100; root cause was throttled anchor transform plus hardcoded 8px refinement caps that could not accommodate 20-40px accumulated drift

### Decisions and Failures

- Two-stage positioning design: global affine from confident first-pass detections handles large-scale geometry, local Sobel/NCC refinement handles small corrections; this separates macro vs micro positioning concerns
- Confidence threshold lowered from 0.75 to 0.50 for anchor transform pass-through; the affine fit from 150+ well-detected points provides the safety net that the blend/clamp was intended to provide

- Added [docs/BUBBLE_REFACTOR_EXECUTION_PLAN.md](BUBBLE_REFACTOR_EXECUTION_PLAN.md), a manager-grade execution plan for a modular bubble-reader refactor with milestone/workstream/work-package structure, dependency IDs, measurable gates, patch cadence, and rollout controls
- Expanded the new plan to include a complete `dl1200_template.yaml` concept reboot (template v2 schema + migrator), anchor-first relative-coordinate mapping using left dark dashes and top dark boxes, distortion-stress gates, and explicit retention of dual measurement zones after localization
- Added [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py) with timing-mark detection for left-edge dashes and top-edge boxes, plus axis transform estimation for anchor-relative coordinates
- Added stage helpers `_stage_localize_rows()`, `_stage_measure_rows()`, and `_stage_decide_answers()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) so `read_answers()` orchestrates modular stages instead of embedding the full pipeline in one function
- Added `_compute_dual_zone_means()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to preserve explicit left/right measurement zones and keep `_compute_edge_mean()` as a dual-zone aggregate
- Added [tests/test_timing_mark_anchors.py](../tests/test_timing_mark_anchors.py) with synthetic coverage for identity fallback, axis transform recovery, and low-mark-count fallback behavior
- Added `_default_bounds()` helper in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to centralize float-to-int conversion for bubble edge positions; used by refinement, validation, scoring, and drawing functions
- Added `_check_row_linearity()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) with median-based inlier/outlier detection; identifies choices where Sobel-y locked onto the wrong edge pair by checking if their y-centers deviate from the row median, then fits a line through inliers to predict corrected positions
- Added `_check_column_alignment()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to verify x-center consistency within each choice column; flags positions that deviate from the column median
- Added `_check_row_brightness()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to detect rows where all 5 choices measured as white (edge_mean > 220), indicating the measurement zone landed on the white gap between rows instead of on actual bubbles
- Added regression test `test_all_students_detect_at_least_71` in [tests/test_bubble_reader.py](../tests/test_bubble_reader.py) verifying all student images (including phone photos) detect >= 71 answers
- Added `TestRowLinearity` tests: aligned row produces no outliers, single outlier detected, two minority outliers detected while majority preserved
- Added `TestRowBrightness` tests: normal mixed-brightness row passes, all-white row is flagged
- Added [omr_utils/bubble_template_extractor.py](../omr_utils/bubble_template_extractor.py) with cryoEM-inspired class averaging: extracts many instances of each printed bubble letter (A-E) from empty bubbles, quality-scores and median-stacks them to produce clean 5X oversize reference templates (300x55 pixels)
- Added [omr_utils/template_matcher.py](../omr_utils/template_matcher.py) with local NCC (normalized cross-correlation) bubble refinement using `cv2.matchTemplate(TM_CCOEFF_NORMED)` within a 15px search window around approximate positions
- Added [omr_utils/debug_drawing.py](../omr_utils/debug_drawing.py), extracted from [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) to reduce its size from 1163 to ~960 lines; contains `draw_answer_debug()` and `_compute_refinement_shift_data()`
- Added [extract_bubble_templates.py](../extract_bubble_templates.py) CLI script for extracting pixel templates from registered scantron images
- Added `config/bubble_templates/` directory with 5 grayscale PNG reference templates (A.png through E.png) extracted from the answer key scan
- Added `_stage_template_refine()` in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) as an optional NCC refinement pass between localization and measurement stages
- Added `mark_index_to_normalized()` and `normalized_to_mark_index()` to [omr_utils/timing_mark_anchors.py](../omr_utils/timing_mark_anchors.py) for converting between fractional timing mark indices and normalized coordinates
- Added `_ensure_mark_indices()`, `_ensure_coordinates()`, and `migrate_template_to_v3()` to [omr_utils/template_loader.py](../omr_utils/template_loader.py) for v3 template format with timing mark index notation
- Added [tests/test_bubble_template_extractor.py](../tests/test_bubble_template_extractor.py) with 10 tests for template extraction, quality scoring, save/load round-trip, and scaling
- Added [tests/test_template_matcher.py](../tests/test_template_matcher.py) with 6 tests for local NCC matching, row refinement, template loading, and integration comparison with Sobel
- Added `TestV3Migration` tests (5 tests) in [tests/test_template_loader.py](../tests/test_template_loader.py) for v3 YAML loading, round-trip precision, and backward compatibility
- Added `TestMarkIndexConversion` tests (4 tests) in [tests/test_timing_mark_anchors.py](../tests/test_timing_mark_anchors.py) for mark index conversion, round-trip, and fractional interpolation

### Behavior or Interface Changes

- `read_answers()` now computes an anchor transform from timing marks and applies a conservative safety filter (`_sanitize_anchor_transform()`) so only high-confidence near-identity corrections are used at runtime
- `read_answers()` behavior is unchanged at the output interface level, but internally now executes modular localization, measurement, and decision stages
- Bubble geometry values in [omr_utils/template_loader.py](../omr_utils/template_loader.py) `get_bubble_geometry_px()` now return float values for sub-pixel precision in validation math; consumers use `_default_bounds()` for integer conversion at array-slicing boundaries
- Updated `half_height` from 0.00273 to 0.00250 in [config/dl1200_template.yaml](../config/dl1200_template.yaml) for closer match to physical ~6:1 bubble aspect ratio (5.5px at canonical 2200h)
- Updated `measurement_inset_v` from 0.00136 to 0.00091 (3px to 2px at canonical) for wider 8px measurement zone (was 6px)
- Tightened `_validate_bubble_rect()` height deviation threshold from 50% to 40% and added explicit aspect ratio check (5.0-6.5 range)
- `read_answers()` now applies three post-refinement validation passes: row linearity correction, edge mean measurement, and brightness sanity check with automatic re-measurement for all-white rows
- `read_answers()` now accepts optional `bubble_templates` parameter and runs NCC template matching refinement when pixel templates are available; auto-loads from `config/bubble_templates/` if not provided
- [config/dl1200_template.yaml](../config/dl1200_template.yaml) upgraded from v2 (hardcoded normalized coordinates) to v3 (fractional timing mark indices); bubble positions are now expressed in the timing mark coordinate frame rather than as pixel-derived coordinates
- `load_template()` now normalizes all templates to v3, computing both mark indices and normalized coordinates in memory for backward compatibility; accepts v1, v2, or v3 YAML inputs
- Template matching is conservative: only applies x-corrections with confidence > 0.45 and shift <= 4px, preventing regressions on phone photos where templates extracted from flatbed scans may not match well

### Fixes and Maintenance

- Fixed phone photo (8B5D0C61) detecting only 69/71: Q44 and Q49 measurement zones were landing in the white gap between adjacent rows because Sobel-y found bracket edges from neighboring rows; row linearity check identifies these misplaced detections and corrects them using the fitted line from correctly-detected choices; now all 4 images detect 71/71

### Decisions and Failures

- Chose median-based inlier identification over pure line-fit for row linearity: fitting a line to all 5 points fails when 2 of 5 are wrong (the fitted line gets pulled to an intermediate position, flagging all 5 as outliers); median is robust to minority outliers and correctly identifies the 2-3 bad choices
- Float geometry with `_default_bounds()` helper chosen over scattered `int()` casts or int-only geometry; `int(cy - 5.5) = int(44.5) = 44` gives 11px total height which couldn't be achieved with int half_height (either 5 or 6, giving 10 or 12px)
- V3 template format stores bubble positions as fractional timing mark indices (e.g., Q1 at left mark 10.4562, choice A at top mark 4.5896) instead of hardcoded normalized coordinates; this structural representation is invariant across scans of the same form model
- 5X oversize template storage (300x55 pixels for a canonical 60x11 bubble) preserves sub-pixel averaging detail from cryoEM-style class averaging; templates are scaled down to actual bubble dimensions at runtime
- Mark index round-trip precision is < 0.0001 in normalized coordinates (sub-pixel accuracy at any practical image resolution); 4 decimal places in the YAML provide sufficient precision

### Developer Tests and Notes

- 327 tests pass across all test files (was 269)
- Detection counts: answer key 71/71, phone photo 71/71, both flatbed scans 71/71
- All pyflakes clean (36 files checked)
- V3 round-trip verification: Q1A original (0.1212, 0.2164) -> round-trip (0.121201, 0.216400), error < 1e-6

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
