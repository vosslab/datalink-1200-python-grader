"""Read filled bubbles from a registered scantron image."""

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.template_loader
import omr_utils.template_matcher
import omr_utils.timing_mark_anchors


#============================================
def _default_bounds(cx: int, cy: int, geom: dict) -> tuple:
	"""Compute integer pixel bounds from float geometry values.

	Centralizes the float-to-int conversion for bubble edge positions.
	Used by refinement, validation, scoring, and drawing functions.

	Args:
		cx: bubble center x in pixels
		cy: bubble center y in pixels
		geom: pixel geometry dict (may contain float values)

	Returns:
		tuple of (top_y, bot_y, left_x, right_x) as integers
	"""
	top_y = int(cy - geom["half_height"])
	bot_y = int(cy + geom["half_height"])
	left_x = int(cx - geom["half_width"])
	right_x = int(cx + geom["half_width"])
	return (top_y, bot_y, left_x, right_x)


#============================================
def _apply_anchor_transform(px: int, py: int, transform: dict,
	width: int, height: int) -> tuple:
	"""Apply anchor-derived x/y scale+offset to a pixel coordinate."""
	tx = int(round(px * transform["x_scale"] + transform["x_offset"]))
	ty = int(round(py * transform["y_scale"] + transform["y_offset"]))
	tx = max(0, min(width - 1, tx))
	ty = max(0, min(height - 1, ty))
	return (tx, ty)


#============================================
def _sanitize_anchor_transform(transform: dict) -> dict:
	"""Pass through anchor corrections when confidence is adequate.

	When timing mark detection has sufficient confidence (>= 0.50),
	the scale and offset are applied fully. The previous blended/clamped
	approach caused cumulative y-drift at the bottom of each column
	because the correction was throttled to 40% and clamped to 3%.
	"""
	safe = {
		"x_scale": 1.0,
		"x_offset": 0.0,
		"y_scale": 1.0,
		"y_offset": 0.0,
		"top_confidence": transform.get("top_confidence", 0.0),
		"left_confidence": transform.get("left_confidence", 0.0),
	}
	top_conf = float(transform.get("top_confidence", 0.0))
	# y-axis: left timing marks provide visual confirmation only.
	# The affine fit from Sobel-detected edges handles y-positioning;
	# applying the timing-mark y-transform on top of that causes
	# double-correction and pushes positions off bubbles.
	# y_scale and y_offset stay at identity (1.0, 0.0).
	# x-axis uses top timing marks; pass through fully when confident
	if top_conf >= 0.50:
		safe["x_scale"] = float(transform.get("x_scale", 1.0))
		safe["x_offset"] = float(transform.get("x_offset", 0.0))
	return safe


#============================================
def _refine_bubble_edges_y(gray: numpy.ndarray, cx: int, cy: int,
	geom: dict) -> tuple:
	"""Refine vertical bubble position by detecting top and bottom bracket edges.

	Extracts a vertical strip ROI around the template position and uses
	Sobel-y to find the top and bottom bracket arms (strong horizontal
	edges). Returns the refined center plus detected edge positions.

	Includes edge separation validation: rejects edge pairs whose
	separation deviates more than 40% from expected bubble height.
	This prevents column-header edges from being mistaken for bubble
	edges at positions like Q1 and Q51.

	Args:
		gray: grayscale image (0=black, 255=white)
		cx: bubble center x in pixels (template estimate)
		cy: bubble center y in pixels (template estimate)
		geom: pixel geometry dict from default_geom()

	Returns:
		tuple of (refined_cy, top_y, bot_y) as integers;
		falls back to (cy, cy - half_height, cy + half_height)
		if edges are not clearly found
	"""
	h, w = gray.shape
	hh = geom["half_height"]
	hw = geom["half_width"]
	pad = geom["refine_pad_v"]
	# max shift from geometry; no hardcoded cap
	max_shift = float(geom["refine_max_shift"])
	# default edge positions via centralized float-to-int helper
	default_top, default_bot, _, _ = _default_bounds(cx, cy, geom)
	# cap search extent so ROI does not overlap adjacent questions
	search_extent = min(hh + pad, pad * 2)
	y1 = max(0, int(cy - search_extent))
	y2 = min(h, int(cy + search_extent))
	# horizontal ROI: full bubble width
	x1 = max(0, int(cx - hw))
	x2 = min(w, int(cx + hw))
	# guard against zero-size ROI
	if y2 <= y1 or x2 <= x1:
		return (cy, default_top, default_bot)
	roi = gray[y1:y2, x1:x2]
	# apply Sobel-y to detect horizontal edges (top/bottom bracket arms)
	sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
	# collapse horizontally: mean absolute gradient per row
	row_energy = numpy.mean(numpy.abs(sobel_y), axis=1)
	# find the two strongest energy peaks (top and bottom bracket arms)
	# use small min_separation to avoid suppressing the real second peak;
	# physical brackets are ~12px apart, peak suppression just avoids double-count
	min_separation = int(max(4, hh // 3))
	min_edge_strength = 15.0
	# find the strongest peak
	peak1_idx = int(numpy.argmax(row_energy))
	peak1_val = float(row_energy[peak1_idx])
	if peak1_val < min_edge_strength:
		return (cy, default_top, default_bot)
	# zero out rows near peak1 and find second peak
	suppressed = row_energy.copy()
	sup_lo = max(0, peak1_idx - min_separation)
	sup_hi = min(len(suppressed), peak1_idx + min_separation + 1)
	suppressed[sup_lo:sup_hi] = 0.0
	peak2_idx = int(numpy.argmax(suppressed))
	peak2_val = float(suppressed[peak2_idx])
	if peak2_val < min_edge_strength:
		return (cy, default_top, default_bot)
	# top and bottom bracket arms (order by position)
	top_idx = min(peak1_idx, peak2_idx)
	bot_idx = max(peak1_idx, peak2_idx)
	# edge separation validation: reject pairs far outside expected range
	# for measurement padding, so accept separations from 40% to 150% of expected
	expected_separation = hh * 2
	actual_separation = bot_idx - top_idx
	if expected_separation > 0:
		deviation = abs(actual_separation - expected_separation) / expected_separation
		if deviation >= 0.6:
			# likely column header or adjacent row edges, not bubble
			return (cy, default_top, default_bot)
	# convert from ROI coordinates back to image coordinates
	refined_top = y1 + top_idx
	refined_bot = y1 + bot_idx
	refined_cy = (refined_top + refined_bot) // 2
	# sanity check: reject large center shifts
	if abs(refined_cy - cy) > max_shift:
		return (cy, default_top, default_bot)
	return (refined_cy, refined_top, refined_bot)


#============================================
def _refine_bubble_edges_x(gray: numpy.ndarray, cx: int, cy: int,
	top_y: int, bot_y: int, geom: dict) -> tuple:
	"""Refine horizontal bubble position by detecting left and right bracket edges.

	Uses detected top_y/bot_y for a precise vertical ROI instead of
	relying on the coarse template half_height. Includes separation
	validation to reject false edge pairs.

	Args:
		gray: grayscale image (0=black, 255=white)
		cx: bubble center x in pixels (template estimate)
		cy: bubble center y in pixels (already y-refined)
		top_y: detected top edge y position
		bot_y: detected bottom edge y position
		geom: pixel geometry dict from default_geom()

	Returns:
		tuple of (refined_cx, left_x, right_x) as integers;
		falls back to (cx, cx - half_width, cx + half_width)
		if edges are not clearly found
	"""
	h, w = gray.shape
	hw = geom["half_width"]
	hpad = geom["refine_pad_h"]
	# max shift from geometry; no hardcoded cap
	max_shift = float(geom["refine_max_shift"])
	# default edge positions via centralized float-to-int helper
	_, _, default_left, default_right = _default_bounds(cx, cy, geom)
	# horizontal ROI: bubble width + padding on each side
	x1 = max(0, int(cx - hw - hpad))
	x2 = min(w, int(cx + hw + hpad))
	# vertical ROI: use detected top/bot edges for precise bounds
	ry1 = max(0, int(top_y))
	ry2 = min(h, int(bot_y))
	# guard against zero-size ROI
	if ry2 <= ry1 or x2 <= x1:
		return (cx, default_left, default_right)
	roi = gray[ry1:ry2, x1:x2]
	# apply Sobel-x to detect vertical edges (left/right bracket arms)
	sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
	# collapse vertically: mean absolute gradient per column
	col_energy = numpy.mean(numpy.abs(sobel_x), axis=0)
	# find the two strongest energy peaks (left and right bracket arms)
	# small suppression to avoid double-counting the same edge
	min_separation = int(max(4, hw // 3))
	min_edge_strength = 15.0
	# find the strongest peak
	peak1_idx = int(numpy.argmax(col_energy))
	peak1_val = float(col_energy[peak1_idx])
	if peak1_val < min_edge_strength:
		return (cx, default_left, default_right)
	# zero out columns near peak1 and find second peak
	suppressed = col_energy.copy()
	sup_lo = max(0, peak1_idx - min_separation)
	sup_hi = min(len(suppressed), peak1_idx + min_separation + 1)
	suppressed[sup_lo:sup_hi] = 0.0
	peak2_idx = int(numpy.argmax(suppressed))
	peak2_val = float(suppressed[peak2_idx])
	if peak2_val < min_edge_strength:
		return (cx, default_left, default_right)
	# left and right bracket arms (order by position)
	left_idx = min(peak1_idx, peak2_idx)
	right_idx = max(peak1_idx, peak2_idx)
	# edge separation validation: reject pairs far outside expected range
	# x-edges use a tight 30% threshold because internal letter strokes
	# (e.g. the vertical bars in 'E' and 'C') produce strong Sobel peaks
	# that create half-width rectangles on lower-quality images
	expected_separation = hw * 2
	actual_separation = right_idx - left_idx
	if expected_separation > 0:
		deviation = abs(actual_separation - expected_separation) / expected_separation
		if deviation >= 0.3:
			return (cx, default_left, default_right)
	# convert from ROI coordinates back to image coordinates
	refined_left = x1 + left_idx
	refined_right = x1 + right_idx
	refined_cx = (refined_left + refined_right) // 2
	# sanity check: reject large center shifts
	if abs(refined_cx - cx) > max_shift:
		return (cx, default_left, default_right)
	return (refined_cx, refined_left, refined_right)


#============================================
def score_bubble_fast(gray: numpy.ndarray, cx: int, cy: int,
	radius: int, geom: dict = None) -> float:
	"""Score a single bubble using bracket-edge dark reference.

	For standalone use (single bubble). The primary scoring path is
	read_answers() which uses self-referencing scoring across the row.

	Uses the dark printed bracket edges (top/bottom borders) as a
	reference. On unfilled bubbles, the measurement zone is bright
	while bracket edges are dark, producing a high contrast. On filled
	bubbles, pencil marks cover the bracket edges too, so both zones
	are equally dark -- this is detected and returns a high score.

	Args:
		gray: grayscale image (0=black, 255=white)
		cx: bubble center x in pixels
		cy: bubble center y in pixels
		radius: bubble radius in pixels (used for bounds checking only)
		geom: optional pixel geometry dict; uses defaults if None

	Returns:
		fill score (higher = more likely filled, range ~0.0 to 1.0).
		Returns -1.0 for out-of-bounds coordinates.
	"""
	h, w = gray.shape
	# bounds check
	if cx < 0 or cy < 0 or cx >= w or cy >= h:
		return -1.0
	# use default geometry if none provided
	if geom is None:
		geom = default_geom()
	# compute default edge positions from center
	top_y, bot_y, left_x, right_x = _default_bounds(cx, cy, geom)
	# bracket edges provide a dark reference (printed border)
	bracket_mean = _compute_bracket_edge_mean(
		gray, cx, cy, top_y, bot_y, left_x, right_x, geom)
	if bracket_mean < 0:
		return -1.0
	# measurement zone fill level
	measurement_mean = _compute_edge_mean(
		gray, cx, cy, top_y, bot_y, left_x, right_x, geom)
	if measurement_mean < 0:
		return -1.0
	# bracket edges should be dark (printed ink). if both zones are
	# bright (no bracket edges found), the bubble position may be
	# misaligned or this is a blank area -- return low score
	if bracket_mean > 200.0 and measurement_mean > 200.0:
		score = 0.0
		return score
	# when bracket edges are very dark (near zero), use 255 as reference
	if bracket_mean <= 1.0:
		score = (255.0 - measurement_mean) / 255.0
		return score
	# on filled bubbles, pencil covers bracket edges too, so
	# measurement_mean <= bracket_mean indicates a filled bubble
	if measurement_mean <= bracket_mean:
		score = 1.0
		return score
	# unfilled: measurement is brighter than bracket edges
	# higher bracket-to-measurement contrast = less filled
	score = 1.0 - (measurement_mean - bracket_mean) / (255.0 - bracket_mean)
	# clamp to valid range
	score = max(0.0, score)
	return score


#============================================
def default_geom() -> dict:
	"""
	Provides backward-compatible defaults when no template geometry
	is available (e.g. for score_bubble_fast standalone calls).

	Returns:
	"""
	geom = {
		"half_width": 30.0,
		"half_height": 5.5,
		"center_exclusion": 11.0,
		"bracket_edge_height": 2.0,
		"measurement_inset_v": 2.0,
		"measurement_inset_h": 3.0,
		"refine_max_shift": 15.0,
		"refine_pad_v": 8.0,
		"refine_pad_h": 8.0,
	}
	return geom


#============================================
def _validate_bubble_rect(top_y: int, bot_y: int, left_x: int, right_x: int,
	cx: int, cy: int, geom: dict) -> tuple:
	"""Validate detected bubble rectangle against expected area and aspect ratio.

	If the detected rectangle deviates too far from expected dimensions,
	resets to template-based defaults. Prevents half-width or half-height
	detections from passing through to measurement.

	Args:
		top_y: detected top edge y
		bot_y: detected bottom edge y
		left_x: detected left edge x
		right_x: detected right edge x
		cx: template center x
		cy: refined center y
		geom: pixel geometry dict

	Returns:
		validated (top_y, bot_y, left_x, right_x, cx_out) tuple;
		cx_out may differ from cx if edges were kept
	"""
	hw = geom["half_width"]
	hh = geom["half_height"]
	expected_w = hw * 2
	expected_h = hh * 2
	det_w = right_x - left_x
	det_h = bot_y - top_y
	# check width and height deviations independently (relaxed for affine)
	w_ok = (expected_w > 0
		and abs(det_w - expected_w) / expected_w < 0.40)
	h_ok = (expected_h > 0
		and abs(det_h - expected_h) / expected_h < 0.50)
	# explicit aspect ratio check: physical bubbles are ~5.5:1
	det_ar = det_w / det_h if det_h > 0 else 0.0
	ar_ok = (det_h > 0 and 4.0 <= det_ar <= 7.5)
	# check area with resolution-scaled hard bounds
	expected_area = expected_w * expected_h
	det_area = det_w * det_h
	area_ok = (expected_area > 0
		and abs(det_area - expected_area) / expected_area < 0.50)
	if w_ok and h_ok and ar_ok and area_ok:
		# edges are reasonable, keep them
		cx_out = (left_x + right_x) // 2
		return (top_y, bot_y, left_x, right_x, cx_out)
	# fall back to template-based defaults, keeping refined cy
	default_top, default_bot, default_left, default_right = (
		_default_bounds(cx, cy, geom))
	return (default_top, default_bot, default_left, default_right, cx)


#============================================
def _compute_bracket_edge_mean(gray: numpy.ndarray, cx: int, cy: int,
	top_y: int, bot_y: int, left_x: int, right_x: int,
	geom: dict) -> float:
	"""Compute mean brightness of the top and bottom bracket edge strips.

	Measures the dark printed bracket borders at the top and bottom of
	the bubble box, using the left and right side strips (excluding
	the center letter zone). Uses detected edge positions for precise
	measurement zones.

	Args:
		gray: grayscale image (0=black, 255=white)
		cx: bubble center x in pixels
		cy: bubble center y in pixels
		top_y: detected top edge y position
		bot_y: detected bottom edge y position
		left_x: detected left edge x position
		right_x: detected right edge x position
		geom: pixel geometry dict

	Returns:
		average brightness of bracket edge strips (0-255 scale),
		or -1.0 if out of bounds
	"""
	h, w = gray.shape
	if cx < 0 or cy < 0 or cx >= w or cy >= h:
		return -1.0
	# int-cast geom values for array slicing
	ce = int(geom["center_exclusion"])
	beh = int(geom["bracket_edge_height"])
	# horizontal bounds: left and right strips excluding center letter
	lx1 = max(0, left_x)
	lx2 = max(0, cx - ce)
	rx1 = min(w, cx + ce)
	rx2 = min(w, right_x)
	# top bracket edge strip (from detected top edge)
	top_y1 = max(0, top_y)
	top_y2 = max(0, top_y + beh)
	# bottom bracket edge strip (from detected bottom edge)
	bot_y1 = min(h, bot_y - beh)
	bot_y2 = min(h, bot_y)
	# extract four patches: top-left, top-right, bottom-left, bottom-right
	tl = gray[top_y1:top_y2, lx1:lx2]
	tr = gray[top_y1:top_y2, rx1:rx2]
	bl = gray[bot_y1:bot_y2, lx1:lx2]
	br = gray[bot_y1:bot_y2, rx1:rx2]
	# combine all bracket edge pixels
	all_pixels = numpy.concatenate([
		tl.ravel(), tr.ravel(), bl.ravel(), br.ravel()
	])
	if all_pixels.size == 0:
		return -1.0
	result = float(numpy.mean(all_pixels))
	return result


#============================================
def _compute_edge_mean(gray: numpy.ndarray, cx: int, cy: int,
	top_y: int, bot_y: int, left_x: int, right_x: int,
	geom: dict) -> float:
	"""Compute mean brightness from dual left/right measurement zones."""
	left_mean, right_mean = _compute_dual_zone_means(
		gray, cx, cy, top_y, bot_y, left_x, right_x, geom)
	if left_mean < 0 or right_mean < 0:
		return -1.0
	edge_mean = (left_mean + right_mean) / 2.0
	return edge_mean


#============================================
def _compute_dual_zone_means(gray: numpy.ndarray, cx: int, cy: int,
	top_y: int, bot_y: int, left_x: int, right_x: int,
	geom: dict) -> tuple:
	"""Compute left and right measurement-zone means separately.

	This function preserves the dual-zone model explicitly so the
	decision stage can continue to rely on independent left/right
	fill measurements after localization.

	Returns:
		tuple of (left_mean, right_mean), or (-1.0, -1.0) if invalid
	"""
	h, w = gray.shape
	if cx < 0 or cy < 0 or cx >= w or cy >= h:
		return (-1.0, -1.0)
	# int-cast geom values for array slicing
	ce = int(geom["center_exclusion"])
	mi_v = int(geom["measurement_inset_v"])
	mi_h = int(geom["measurement_inset_h"])
	# left edge strip: inset from detected left edge to center exclusion
	lx1 = max(0, left_x + mi_h)
	lx2 = max(0, cx - ce)
	# right edge strip: from center exclusion to inset from detected right edge
	rx1 = min(w, cx + ce)
	rx2 = min(w, right_x - mi_h)
	# vertical bounds: inset from detected top/bottom edges
	y1 = max(0, top_y + mi_v)
	y2 = min(h, bot_y - mi_v)
	left_strip = gray[y1:y2, lx1:lx2]
	right_strip = gray[y1:y2, rx1:rx2]
	if left_strip.size == 0 or right_strip.size == 0:
		return (-1.0, -1.0)
	left_mean = float(numpy.mean(left_strip))
	right_mean = float(numpy.mean(right_strip))
	return (left_mean, right_mean)


#============================================
def _find_adaptive_threshold(spreads: list,
	min_spread_floor: float = 15.0) -> float:
	"""Find the blank/filled threshold using the largest gap in sorted spreads.

	Each question produces a spread value (max edge_mean - min edge_mean
	across its 5 choices). Filled questions have large spreads (one choice
	is much darker), blank questions have small spreads (all similar).
	The natural gap between these two populations gives the threshold.

	A minimum floor is enforced so that low-contrast images do not
	produce a threshold below the noise level. If the largest gap is
	not significantly larger than the average gap (less than 2x), the
	data is likely unimodal and the floor is used instead.

	Args:
		spreads: list of (question_number, spread_value) tuples
		min_spread_floor: minimum threshold value in pixel units

	Returns:
		adaptive threshold in pixels; questions above this are filled
	"""
	sorted_vals = sorted(s for _, s in spreads)
	max_gap = 0.0
	max_gap_idx = 0
	total_gap = 0.0
	num_gaps = len(sorted_vals) - 1
	for i in range(num_gaps):
		gap = sorted_vals[i + 1] - sorted_vals[i]
		total_gap += gap
		if gap > max_gap:
			max_gap = gap
			max_gap_idx = i
	# threshold is the midpoint of the largest gap
	threshold = (sorted_vals[max_gap_idx] + sorted_vals[max_gap_idx + 1]) / 2.0
	# gap significance check: if largest gap < 2x average, data is unimodal
	if num_gaps > 0:
		avg_gap = total_gap / num_gaps
		if max_gap < 2.0 * avg_gap:
			threshold = min_spread_floor
	# enforce minimum floor
	threshold = max(threshold, min_spread_floor)
	return threshold


#============================================
def _check_row_linearity(q_choices: dict, choices: list,
	max_deviation: int = 4) -> list:
	"""Check that refined y-centers in a row follow a line.

	Uses a two-step approach: first identifies outliers via median
	deviation (robust to minority bad detections), then fits a line
	through inliers only to predict corrected positions for outliers.

	Args:
		q_choices: dict mapping choice letter to detection data
		choices: ordered list of choice letters (e.g. [A,B,C,D,E])
		max_deviation: max pixel distance from median to be inlier

	Returns:
		list of (choice, predicted_cy) tuples for outlier choices
	"""
	xs = []
	ys = []
	for choice in choices:
		xs.append(q_choices[choice]["px"])
		ys.append(q_choices[choice]["refined_cy"])
	xs = numpy.array(xs, dtype=float)
	ys = numpy.array(ys, dtype=float)
	# step 1: identify inliers using median y (robust to minority outliers)
	median_y = numpy.median(ys)
	deviations = numpy.abs(ys - median_y)
	inlier_mask = deviations <= max_deviation
	num_inliers = int(numpy.sum(inlier_mask))
	# need at least 3 inliers to fit a reliable line
	if num_inliers < 3:
		return []
	# step 2: fit line through inliers only
	inlier_xs = xs[inlier_mask]
	inlier_ys = ys[inlier_mask]
	coeffs = numpy.polyfit(inlier_xs, inlier_ys, 1)
	# predict y for all positions using the inlier line
	fitted = numpy.polyval(coeffs, xs)
	outliers = []
	for i, choice in enumerate(choices):
		if not inlier_mask[i]:
			predicted_cy = int(fitted[i])
			outliers.append((choice, predicted_cy))
	return outliers


#============================================
def _check_column_alignment(raw_data: list, choices: list,
	col_start: int, col_end: int, max_x_deviation: int = 5) -> dict:
	"""Check x-center consistency within each choice column.

	For each choice letter (A-E), collects the refined x-centers
	across all questions in a column and computes the median. Flags
	any question where a choice's x deviates beyond the threshold.

	Args:
		raw_data: list of q_choices dicts (from first pass)
		choices: ordered list of choice letters
		col_start: column start index (0-based, inclusive)
		col_end: column end index (0-based, exclusive)
		max_x_deviation: max pixel distance from column median

	Returns:
		dict mapping (q_idx, choice) to median_x for flagged positions
	"""
	flagged = {}
	for choice in choices:
		# collect x-centers for this choice letter across the column
		x_vals = []
		for q_idx in range(col_start, col_end):
			x_vals.append(raw_data[q_idx][choice]["px"])
		median_x = int(numpy.median(x_vals))
		# flag deviations from column median
		for q_idx in range(col_start, col_end):
			actual_x = raw_data[q_idx][choice].get("refined_cx", raw_data[q_idx][choice]["px"])
			if abs(actual_x - median_x) > max_x_deviation:
				flagged[(q_idx, choice)] = median_x
	return flagged


#============================================
def _check_row_brightness(edge_means: dict, choices: list,
	white_threshold: float = 220.0) -> bool:
	"""Check if all choices in a row measured as white (misplaced).

	If every choice in the row has edge_mean above the threshold,
	the measurement zone likely landed on white paper between rows
	instead of on the actual bubble.

	Args:
		edge_means: dict mapping choice letter to edge_mean value
		choices: ordered list of choice letters
		white_threshold: mean brightness above which a zone is white

	Returns:
		True if all choices are white (row is misplaced), False otherwise
	"""
	for choice in choices:
		if edge_means[choice] < white_threshold:
			return False
	return True


#============================================
def _select_rect_by_bracket_signal(gray: numpy.ndarray, px: int,
	refined_cy: int, refined_cx: int, top_y: int, bot_y: int,
	left_x: int, right_x: int, geom: dict) -> tuple:
	"""Choose between refined vs template-centered rect by bracket darkness.

	Empty bubbles should align closely to printed bracket borders.
	If the template-centered rectangle has materially darker bracket
	edges than the refined one, prefer template-centered geometry.
	"""
	refined_bracket = _compute_bracket_edge_mean(
		gray, refined_cx, refined_cy, top_y, bot_y, left_x, right_x, geom)
	def_top, def_bot, def_left, def_right = _default_bounds(px, refined_cy, geom)
	default_bracket = _compute_bracket_edge_mean(
		gray, px, refined_cy, def_top, def_bot, def_left, def_right, geom)
	if default_bracket < 0 or refined_bracket < 0:
		return (refined_cx, top_y, bot_y, left_x, right_x)
	# lower mean is darker and generally better aligned to printed edges
	if default_bracket + 6.0 < refined_bracket:
		return (px, def_top, def_bot, def_left, def_right)
	return (refined_cx, top_y, bot_y, left_x, right_x)


#============================================
def _stage_localize_rows(gray: numpy.ndarray, template: dict,
	geom: dict, transform: dict) -> list:
	"""Localize bubble rows and refine vertical positions."""
	h, w = gray.shape
	answers_config = template["answers"]
	num_q = answers_config["num_questions"]
	choices = answers_config["choices"]
	raw_data = []
	for q_num in range(1, num_q + 1):
		q_choices = {}
		for choice in choices:
			norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			px, py = omr_utils.template_loader.to_pixels(norm_x, norm_y, w, h)
			# store raw template pixel position before anchor transform
			template_px = px
			template_py = py
			px, py = _apply_anchor_transform(px, py, transform, w, h)
			refined_cy, top_y, bot_y = _refine_bubble_edges_y(
				gray, px, py, geom)
			y_refined = (refined_cy != py)
			q_choices[choice] = {
				"px": px, "py": py,
				"template_px": template_px, "template_py": template_py,
				"refined_cy": refined_cy, "y_refined": y_refined,
				"top_y": top_y, "bot_y": bot_y,
			}
		raw_data.append(q_choices)
	# two-part y correction for unresolved detections
	left_range = answers_config["left_column"]["question_range"]
	right_range = answers_config["right_column"]["question_range"]
	for q_idx in range(num_q):
		q_choices = raw_data[q_idx]
		row_shifts = []
		unrefined = []
		max_local_shift = int(round(float(geom["refine_max_shift"])))
		for choice in choices:
			cd = q_choices[choice]
			if cd["y_refined"]:
				row_shifts.append(cd["refined_cy"] - cd["py"])
			else:
				unrefined.append(choice)
		if not unrefined:
			continue
		q_num = q_idx + 1
		if left_range[0] <= q_num <= left_range[1]:
			col_start = left_range[0] - 1
			col_end = left_range[1]
		else:
			col_start = right_range[0] - 1
			col_end = right_range[1]
		neighbor_shifts = []
		for offset in range(1, col_end - col_start + 1):
			for neighbor_idx in [q_idx + offset, q_idx - offset]:
				if neighbor_idx < col_start or neighbor_idx >= col_end:
					continue
				n_choices = raw_data[neighbor_idx]
				for n_choice in choices:
					nd = n_choices[n_choice]
					if nd["y_refined"]:
						neighbor_shifts.append(nd["refined_cy"] - nd["py"])
			if len(neighbor_shifts) >= 5:
				break
		all_shifts = row_shifts + neighbor_shifts
		if not all_shifts:
			continue
		median_shift = int(numpy.median(all_shifts))
		median_shift = max(-max_local_shift, min(max_local_shift, median_shift))
		for choice in unrefined:
			cd = q_choices[choice]
			corrected_cy = cd["py"] + median_shift
			cd["refined_cy"] = corrected_cy
			top_y, bot_y, _, _ = _default_bounds(cd["px"], corrected_cy, geom)
			cd["top_y"] = top_y
			cd["bot_y"] = bot_y
	# row-linearity pass to fix y outliers
	for q_idx in range(num_q):
		q_choices = raw_data[q_idx]
		outliers = _check_row_linearity(q_choices, choices)
		for choice, predicted_cy in outliers:
			cd = q_choices[choice]
			max_local_shift = int(round(float(geom["refine_max_shift"])))
			dy = predicted_cy - cd["py"]
			if dy > max_local_shift:
				predicted_cy = cd["py"] + max_local_shift
			elif dy < -max_local_shift:
				predicted_cy = cd["py"] - max_local_shift
			cd["refined_cy"] = predicted_cy
			top_y, bot_y, _, _ = _default_bounds(cd["px"], predicted_cy, geom)
			cd["top_y"] = top_y
			cd["bot_y"] = bot_y
	return raw_data


#============================================
def _fit_affine_from_confident_detections(raw_data: list, choices: list,
	geom: dict) -> None:
	"""Fit a 2D affine transform from confidently-detected bubble positions.

	Collects bubbles where Sobel y-refinement succeeded (y_refined=True),
	uses their template positions (pre-transform) and detected positions
	(post-Sobel) to fit a full affine transform via least-squares. This
	handles rotation, skew, and non-uniform scaling that timing marks
	alone cannot capture. Modifies raw_data in place.

	Requires at least 10 confident points from different rows to fit.
	If insufficient points, the function returns without changes.

	Args:
		raw_data: list of per-question choice dicts from _stage_localize_rows
		choices: ordered list of choice letters
		geom: pixel geometry dict
	"""
	# collect confident detections
	src_points = []
	dst_points = []
	for q_choices in raw_data:
		for choice in choices:
			cd = q_choices[choice]
			if not cd.get("y_refined", False):
				continue
			# source: raw template pixel coords (pre-transform)
			src_points.append([cd["template_px"], cd["template_py"], 1.0])
			# destination: detected position (anchor-transformed px, Sobel-refined cy)
			dst_points.append([cd["px"], cd["refined_cy"]])
	# need sufficient points for a reliable affine fit
	if len(src_points) < 10:
		return
	src = numpy.array(src_points, dtype=float)
	dst = numpy.array(dst_points, dtype=float)
	# Step 1: RANSAC affine to identify inliers vs Sobel outliers
	src_2d = src[:, :2].astype(numpy.float64)
	dst_2d = dst.astype(numpy.float64)
	_, inlier_mask = cv2.estimateAffine2D(
		src_2d.reshape(-1, 1, 2), dst_2d.reshape(-1, 1, 2),
		method=cv2.RANSAC, ransacReprojThreshold=5.0)
	if inlier_mask is None:
		return
	# flatten inlier mask and filter to inliers only
	inlier_mask = inlier_mask.ravel().astype(bool)
	num_inliers = int(numpy.sum(inlier_mask))
	src_in = src[inlier_mask]
	dst_in = dst[inlier_mask]
	# Step 2: fit polynomial with quadratic y-term on inliers only.
	# Linear affine leaves +/-2px sinusoidal y-residuals from barrel
	# distortion; adding ty^2 captures the curvature.
	# Design matrix: [tx, ty, ty^2, 1] for both x and y outputs
	design = numpy.column_stack([
		src_in[:, 0],
		src_in[:, 1],
		src_in[:, 1] ** 2,
		numpy.ones(len(src_in)),
	])
	# fit x and y polynomial coefficients via least-squares
	poly_x, _, _, _ = numpy.linalg.lstsq(design, dst_in[:, 0], rcond=None)
	poly_y, _, _, _ = numpy.linalg.lstsq(design, dst_in[:, 1], rcond=None)
	# compute residuals for diagnostics
	design_all = numpy.column_stack([
		src[:, 0], src[:, 1], src[:, 1] ** 2, numpy.ones(len(src)),
	])
	predicted = numpy.column_stack([
		design_all @ poly_x, design_all @ poly_y,
	])
	residuals = numpy.sqrt(numpy.sum((dst - predicted) ** 2, axis=1))
	print(f"  poly fit: {num_inliers}/{len(src_points)} inliers,"
		f" median residual={numpy.median(residuals):.2f}px,"
		f" max residual={numpy.max(residuals):.2f}px")
	# apply polynomial to ALL bubble positions
	for q_choices in raw_data:
		for choice in choices:
			cd = q_choices[choice]
			tx = cd["template_px"]
			ty = cd["template_py"]
			# compute poly-predicted position: [tx, ty, ty^2, 1] @ coeffs
			new_px = int(round(
				poly_x[0] * tx + poly_x[1] * ty + poly_x[2] * ty * ty + poly_x[3]))
			new_cy = int(round(
				poly_y[0] * tx + poly_y[1] * ty + poly_y[2] * ty * ty + poly_y[3]))
			cd["px"] = new_px
			cd["refined_cy"] = new_cy
			# store affine-predicted position for fallback use
			cd["affine_px"] = new_px
			cd["affine_cy"] = new_cy
			# recompute bounds from new center
			top_y, bot_y, _, _ = _default_bounds(new_px, new_cy, geom)
			cd["top_y"] = top_y
			cd["bot_y"] = bot_y


#============================================
def _stage_measure_rows(gray: numpy.ndarray, raw_data: list,
	template: dict, geom: dict, transform: dict = None) -> tuple:
	"""Refine horizontal edges and compute per-choice measurements."""
	num_q = template["answers"]["num_questions"]
	choices = template["answers"]["choices"]
	all_edge_means = []
	all_positions = []
	all_edges = []
	for q_idx in range(num_q):
		edge_means = {}
		positions = {}
		edges = {}
		q_choices = raw_data[q_idx]
		for choice in choices:
			cd = q_choices[choice]
			px = cd["px"]
			refined_cy = cd["refined_cy"]
			top_y = cd["top_y"]
			bot_y = cd["bot_y"]
			refined_cx, left_x, right_x = _refine_bubble_edges_x(
				gray, px, refined_cy, top_y, bot_y, geom)
			q_choices[choice]["refined_cx"] = refined_cx
			top_y, bot_y, left_x, right_x, refined_cx = (
				_validate_bubble_rect(
					top_y, bot_y, left_x, right_x,
					px, refined_cy, geom))
			refined_cx, top_y, bot_y, left_x, right_x = (
				_select_rect_by_bracket_signal(
					gray, px, refined_cy, refined_cx,
					top_y, bot_y, left_x, right_x, geom))
			q_choices[choice]["refined_cx"] = refined_cx
			edge_means[choice] = _compute_edge_mean(
				gray, refined_cx, refined_cy,
				top_y, bot_y, left_x, right_x, geom)
			positions[choice] = (refined_cx, refined_cy)
			edges[choice] = (top_y, bot_y, left_x, right_x)
		# per-bubble bracket confidence: if bracket edges are white (> 200),
		# the measurement zone landed on white paper; fall back to affine position
		for choice in choices:
			cd = q_choices[choice]
			bracket_mean = _compute_bracket_edge_mean(
				gray, positions[choice][0], positions[choice][1],
				edges[choice][0], edges[choice][1],
				edges[choice][2], edges[choice][3], geom)
			cd["bracket_confidence"] = bracket_mean
			if bracket_mean < 0 or bracket_mean <= 200.0:
				continue
			# bracket edges are bright: bubble is mispositioned
			# fall back to affine-predicted position if available
			affine_px = cd.get("affine_px")
			affine_cy = cd.get("affine_cy")
			if affine_px is None or affine_cy is None:
				continue
			# recompute bounds from affine position
			fb_top, fb_bot, fb_left, fb_right = _default_bounds(
				affine_px, affine_cy, geom)
			# try x refinement at the affine position
			fb_cx, fb_left, fb_right = _refine_bubble_edges_x(
				gray, affine_px, affine_cy, fb_top, fb_bot, geom)
			fb_top, fb_bot, fb_left, fb_right, fb_cx = _validate_bubble_rect(
				fb_top, fb_bot, fb_left, fb_right,
				affine_px, affine_cy, geom)
			# re-measure at corrected position
			new_edge_mean = _compute_edge_mean(
				gray, fb_cx, affine_cy,
				fb_top, fb_bot, fb_left, fb_right, geom)
			if new_edge_mean < 0:
				continue
			edge_means[choice] = new_edge_mean
			positions[choice] = (fb_cx, affine_cy)
			edges[choice] = (fb_top, fb_bot, fb_left, fb_right)
			cd["refined_cx"] = fb_cx
			cd["refined_cy"] = affine_cy
		all_edge_means.append(edge_means)
		all_positions.append(positions)
		all_edges.append(edges)
	# column-lock correction is only safe when top-anchor confidence is
	# strong; otherwise perspective residuals on phone photos can make
	# per-column x medians misleading.
	top_conf = 0.0
	if transform is not None:
		top_conf = float(transform.get("top_confidence", 0.0))
	if top_conf >= 0.75:
		left_range = template["answers"]["left_column"]["question_range"]
		right_range = template["answers"]["right_column"]["question_range"]
		flagged = {}
		flagged.update(_check_column_alignment(
			raw_data, choices, left_range[0] - 1, left_range[1], max_x_deviation=3))
		flagged.update(_check_column_alignment(
			raw_data, choices, right_range[0] - 1, right_range[1], max_x_deviation=3))
		for (q_idx, choice), median_x in flagged.items():
			cd = raw_data[q_idx][choice]
			refined_cy = cd["refined_cy"]
			top_y, bot_y, left_x, right_x = _default_bounds(
				median_x, refined_cy, geom)
			edge_mean = _compute_edge_mean(
				gray, median_x, refined_cy,
				top_y, bot_y, left_x, right_x, geom)
			if edge_mean < 0:
				continue
			all_edge_means[q_idx][choice] = edge_mean
			all_positions[q_idx][choice] = (median_x, refined_cy)
			all_edges[q_idx][choice] = (top_y, bot_y, left_x, right_x)
			cd["refined_cx"] = median_x
	# brightness sanity pass: if a row is all-white, fall back to template y
	for q_idx in range(num_q):
		if not _check_row_brightness(all_edge_means[q_idx], choices):
			continue
		q_choices = raw_data[q_idx]
		for choice in choices:
			cd = q_choices[choice]
			px = cd["px"]
			py = cd["py"]
			top_y, bot_y, _, _ = _default_bounds(px, py, geom)
			refined_cx, left_x, right_x = _refine_bubble_edges_x(
				gray, px, py, top_y, bot_y, geom)
			top_y, bot_y, left_x, right_x, refined_cx = (
				_validate_bubble_rect(
					top_y, bot_y, left_x, right_x,
					px, py, geom))
			all_edge_means[q_idx][choice] = _compute_edge_mean(
				gray, refined_cx, py,
				top_y, bot_y, left_x, right_x, geom)
			all_positions[q_idx][choice] = (refined_cx, py)
			all_edges[q_idx][choice] = (top_y, bot_y, left_x, right_x)
	# hard error if too many rows have all-white measurement zones
	bad_row_count = 0
	for q_idx in range(num_q):
		if _check_row_brightness(all_edge_means[q_idx], choices):
			bad_row_count += 1
	if bad_row_count > 75:
		left_conf = 0.0
		top_conf_val = 0.0
		if transform is not None:
			left_conf = float(transform.get("left_confidence", 0.0))
			top_conf_val = float(transform.get("top_confidence", 0.0))
		raise RuntimeError(
			f"Bubble detection failed: {bad_row_count} of {num_q} rows "
			f"have no dark pixels (all edge_means > 220). "
			f"Anchor confidence: left={left_conf:.3f}, top={top_conf_val:.3f}")
	return (all_edge_means, all_positions, all_edges)


#============================================
def _stage_decide_answers(all_edge_means: list, all_positions: list,
	all_edges: list, choices: list, multi_gap: float) -> list:
	"""Convert measurements into answer labels, scores, and flags."""
	spreads = []
	for q_idx, edge_means in enumerate(all_edge_means):
		vals = list(edge_means.values())
		spread = max(vals) - min(vals)
		spreads.append((q_idx + 1, spread))
	blank_threshold = _find_adaptive_threshold(spreads)
	results = []
	for q_idx, edge_means in enumerate(all_edge_means):
		q_num = q_idx + 1
		vals = list(edge_means.values())
		max_edge = max(vals)
		spread = max_edge - min(vals)
		scores = {}
		for choice in choices:
			scores[choice] = (max_edge - edge_means[choice]) / 255.0
		ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
		best_choice = ranked[0][0]
		best_score = ranked[0][1]
		second_score = ranked[1][1]
		gap_from_second = best_score - second_score
		flags = ""
		if spread < blank_threshold:
			answer = ""
			flags = "BLANK"
		elif gap_from_second < multi_gap:
			answer = best_choice
			other = ranked[1][0]
			flags = f"MULTIPLE({other})"
		else:
			answer = best_choice
		entry = {
			"question": q_num,
			"answer": answer,
			"scores": scores,
			"flags": flags,
			"positions": all_positions[q_idx],
			"edges": all_edges[q_idx],
		}
		results.append(entry)
	return results


#============================================
def _stage_template_refine(gray: numpy.ndarray, raw_data: list,
	template: dict, geom: dict, bubble_templates: dict) -> None:
	"""Optional template-matching refinement pass on localized rows.

	Uses NCC to refine bubble x and y positions by matching pixel
	templates of each letter against the image. Only updates positions
	where the NCC confidence is high and the shift is within the
	geometry limit. Modifies raw_data in place.

	Args:
		gray: grayscale image
		raw_data: list of per-question choice dicts from _stage_localize_rows
		template: loaded template dictionary
		geom: pixel geometry dict
		bubble_templates: dict mapping letter to 5X oversize template array
	"""
	choices = template["answers"]["choices"]
	# require high confidence for NCC-driven position updates
	high_confidence = 0.45
	for q_choices in raw_data:
		# build position dict for this row
		row_positions = {}
		for choice in choices:
			cd = q_choices[choice]
			row_positions[choice] = (cd["px"], cd["refined_cy"])
		# run NCC refinement
		refined = omr_utils.template_matcher.refine_row_by_template(
			gray, bubble_templates, row_positions, geom, choices)
		# update x and y positions when confidence is high and shift is small
		max_shift = int(geom["refine_max_shift"])
		for choice in choices:
			if choice not in refined:
				continue
			rcx, rcy, conf = refined[choice]
			if conf < high_confidence:
				continue
			cd = q_choices[choice]
			# apply x correction
			dx = abs(rcx - cd["px"])
			if dx <= max_shift:
				cd["px"] = rcx
			# apply y correction and recompute bounds
			dy = abs(rcy - cd["refined_cy"])
			if dy <= max_shift:
				cd["refined_cy"] = rcy
				top_y, bot_y, _, _ = _default_bounds(cd["px"], rcy, geom)
				cd["top_y"] = top_y
				cd["bot_y"] = bot_y


#============================================
def read_answers(image: numpy.ndarray, template: dict,
	multi_gap: float = 0.03, bubble_templates: dict = None) -> list:
	"""Read all 100 answers from a registered scantron image.

	Uses self-referencing scoring: for each question, the lightest
	(emptiest) choice in the row is used as the baseline. This avoids
	dependency on local background strips, which can be unreliable
	for phone photos with uneven lighting or machine-printed marks.

	Blank detection uses adaptive thresholding: the spread (max - min
	edge mean) across all 100 questions is analyzed to find the natural
	gap between filled and blank populations.

	Args:
		template: loaded template dictionary
		multi_gap: min spread gap between top two scores for MULTIPLE flag
		bubble_templates: optional dict of pixel templates for NCC refinement;
			if None, attempts to load from config/bubble_templates/

	Returns:
		list of dicts with keys: question, answer, scores, flags,
		positions, edges. answer is a choice letter or empty string
		if blank, scores is a dict of choice->score, flags is a string,
		edges maps choice to (top_y, bot_y, left_x, right_x)
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# light blur to reduce noise while preserving fill signal
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	h, w = image.shape[:2]
	choices = template["answers"]["choices"]
	geom = default_geom()
	raw_transform = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	print(f"  anchor confidence: left={raw_transform.get('left_confidence', 0):.3f}"
		f" top={raw_transform.get('top_confidence', 0):.3f}"
		f" marks: left={len(raw_transform.get('left_marks', []))}"
		f" top={len(raw_transform.get('top_marks', []))}")
	transform = _sanitize_anchor_transform(raw_transform)
	raw_data = _stage_localize_rows(gray, template, geom, transform)
	# fit 2D affine from well-detected Sobel points to fix cumulative drift
	_fit_affine_from_confident_detections(raw_data, choices, geom)
	# optional NCC template matching refinement
	if bubble_templates is None:
		bubble_templates = omr_utils.template_matcher.try_load_bubble_templates()
	if bubble_templates:
		_stage_template_refine(gray, raw_data, template, geom, bubble_templates)
	all_edge_means, all_positions, all_edges = _stage_measure_rows(
		gray, raw_data, template, geom, transform)
	results = _stage_decide_answers(
		all_edge_means, all_positions, all_edges, choices, multi_gap)
	return results


