"""Read filled bubbles from a registered scantron image."""

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.template_loader


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
		geom: pixel geometry dict from get_bubble_geometry_px()

	Returns:
		tuple of (refined_cy, top_y, bot_y) as integers;
		falls back to (cy, cy - half_height, cy + half_height)
		if edges are not clearly found
	"""
	h, w = gray.shape
	hh = geom["half_height"]
	hw = geom["half_width"]
	pad = geom["refine_pad_v"]
	max_shift = geom["refine_max_shift"]
	# default edge positions via centralized float-to-int helper
	default_top, default_bot, _, _ = _default_bounds(cx, cy, geom)
	# cap search extent so ROI does not overlap adjacent questions
	# question spacing is ~31px at canonical; ROI half-size must stay under ~15px
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
	# physical brackets are ~12px at canonical; half_height may be larger
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
		geom: pixel geometry dict from get_bubble_geometry_px()

	Returns:
		tuple of (refined_cx, left_x, right_x) as integers;
		falls back to (cx, cx - half_width, cx + half_width)
		if edges are not clearly found
	"""
	h, w = gray.shape
	hw = geom["half_width"]
	hpad = geom["refine_pad_h"]
	max_shift = geom["refine_max_shift"]
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
		geom = _default_geom()
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
def _default_geom() -> dict:
	"""Return default bubble geometry in pixels for canonical 1700x2200.

	Provides backward-compatible defaults when no template geometry
	is available (e.g. for score_bubble_fast standalone calls).

	Returns:
		dict with pixel geometry values at canonical resolution
	"""
	geom = {
		"half_width": 30.0,
		"half_height": 5.5,
		"center_exclusion": 11.0,
		"bracket_edge_height": 2.0,
		"measurement_inset_v": 2.0,
		"measurement_inset_h": 3.0,
		"refine_max_shift": 8.0,
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
	# check width and height deviations independently
	w_ok = (expected_w > 0
		and abs(det_w - expected_w) / expected_w < 0.30)
	h_ok = (expected_h > 0
		and abs(det_h - expected_h) / expected_h < 0.40)
	# explicit aspect ratio check: physical bubbles are ~5.5:1
	det_ar = det_w / det_h if det_h > 0 else 0.0
	ar_ok = (det_h > 0 and 5.0 <= det_ar <= 6.5)
	# check area with resolution-scaled hard bounds
	# at canonical 1700x2200, expected area is ~660px (60x11)
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
	"""Compute mean brightness of the left and right edge strips.

	Measures the rectangular strips on either side of the center
	exclusion zone (where the printed letter sits). Uses detected
	edge positions to compute measurement zones from actual bubble
	edges rather than hardcoded offsets.

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
		average brightness of left+right edge strips (0-255 scale),
		or -1.0 if out of bounds
	"""
	h, w = gray.shape
	if cx < 0 or cy < 0 or cx >= w or cy >= h:
		return -1.0
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
		return -1.0
	# mean of both edge strips (lower = darker = more filled)
	edge_mean = (float(numpy.mean(left_strip))
		+ float(numpy.mean(right_strip))) / 2.0
	return edge_mean


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
def read_answers(image: numpy.ndarray, template: dict,
	multi_gap: float = 0.03) -> list:
	"""Read all 100 answers from a registered scantron image.

	Uses self-referencing scoring: for each question, the lightest
	(emptiest) choice in the row is used as the baseline. This avoids
	dependency on local background strips, which can be unreliable
	for phone photos with uneven lighting or machine-printed marks.

	Blank detection uses adaptive thresholding: the spread (max - min
	edge mean) across all 100 questions is analyzed to find the natural
	gap between filled and blank populations.

	Args:
		image: BGR registered image (perspective-corrected, canonical size)
		template: loaded template dictionary
		multi_gap: min spread gap between top two scores for MULTIPLE flag

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
	answers_config = template["answers"]
	num_q = answers_config["num_questions"]
	choices = answers_config["choices"]
	# get bubble geometry scaled to this image
	geom = omr_utils.template_loader.get_bubble_geometry_px(template, w, h)
	# first pass: attempt edge detection on all questions
	# track which questions successfully refined vs fell back to template
	raw_data = []
	for q_num in range(1, num_q + 1):
		q_choices = {}
		for choice in choices:
			norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
				template, q_num, choice)
			px, py = omr_utils.template_loader.to_pixels(norm_x, norm_y, w, h)
			# attempt edge detection; record whether refinement succeeded
			refined_cy, top_y, bot_y = _refine_bubble_edges_y(
				gray, px, py, geom)
			# refinement succeeded if cy changed from template
			y_refined = (refined_cy != py)
			q_choices[choice] = {
				"px": px, "py": py,
				"refined_cy": refined_cy, "y_refined": y_refined,
				"top_y": top_y, "bot_y": bot_y,
			}
		raw_data.append(q_choices)
	# correct unrefined choices using two strategies:
	# 1) intra-row: if some choices in the same row refined, use their median shift
	# 2) neighbor: if NO choices in the row refined, use nearby rows' shifts
	left_range = answers_config["left_column"]["question_range"]
	right_range = answers_config["right_column"]["question_range"]
	for q_idx in range(num_q):
		q_choices = raw_data[q_idx]
		# collect shifts from refined choices in this row
		row_shifts = []
		unrefined = []
		for choice in choices:
			cd = q_choices[choice]
			if cd["y_refined"]:
				row_shifts.append(cd["refined_cy"] - cd["py"])
			else:
				unrefined.append(choice)
		if not unrefined:
			# all choices refined, nothing to correct
			continue
		# collect neighbor shifts to supplement sparse intra-row data
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
				for nc in choices:
					nd = n_choices[nc]
					if nd["y_refined"]:
						neighbor_shifts.append(nd["refined_cy"] - nd["py"])
			if len(neighbor_shifts) >= 5:
				break
		# combine intra-row and neighbor shifts for a robust median
		all_shifts = row_shifts + neighbor_shifts
		if not all_shifts:
			continue
		median_shift = int(numpy.median(all_shifts))
		# apply correction to unrefined choices
		for choice in unrefined:
			cd = q_choices[choice]
			corrected_cy = cd["py"] + median_shift
			cd["refined_cy"] = corrected_cy
			# use _default_bounds for float-safe int conversion
			top_y, bot_y, _, _ = _default_bounds(cd["px"], corrected_cy, geom)
			cd["top_y"] = top_y
			cd["bot_y"] = bot_y
	# row linearity check: fix outlier y-centers using line fit
	for q_idx in range(num_q):
		q_choices = raw_data[q_idx]
		outliers = _check_row_linearity(q_choices, choices)
		for choice, predicted_cy in outliers:
			cd = q_choices[choice]
			cd["refined_cy"] = predicted_cy
			top_y, bot_y, _, _ = _default_bounds(cd["px"], predicted_cy, geom)
			cd["top_y"] = top_y
			cd["bot_y"] = bot_y
	# second pass: refine x and compute edge means using corrected positions
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
			# refine x per choice using y edges for precise ROI
			refined_cx, left_x, right_x = _refine_bubble_edges_x(
				gray, px, refined_cy, top_y, bot_y, geom)
			# validate area and aspect ratio of the combined rectangle;
			# resets to template defaults if detection is badly shaped
			top_y, bot_y, left_x, right_x, refined_cx = (
				_validate_bubble_rect(
					top_y, bot_y, left_x, right_x,
					px, refined_cy, geom))
			# compute edge mean using validated edge positions
			edge_means[choice] = _compute_edge_mean(
				gray, refined_cx, refined_cy,
				top_y, bot_y, left_x, right_x, geom)
			positions[choice] = (refined_cx, refined_cy)
			edges[choice] = (top_y, bot_y, left_x, right_x)
		all_edge_means.append(edge_means)
		all_positions.append(positions)
		all_edges.append(edges)
	# brightness sanity check: detect rows where measurement landed on white gap
	# if all 5 choices are white, fall back to template y and re-measure
	for q_idx in range(num_q):
		if not _check_row_brightness(all_edge_means[q_idx], choices):
			continue
		# this row is all-white: re-measure using template y positions
		q_choices = raw_data[q_idx]
		for choice in choices:
			cd = q_choices[choice]
			px = cd["px"]
			py = cd["py"]
			# reset to template y and recompute bounds
			top_y, bot_y, _, _ = _default_bounds(px, py, geom)
			# re-run x refinement with template y
			refined_cx, left_x, right_x = _refine_bubble_edges_x(
				gray, px, py, top_y, bot_y, geom)
			top_y, bot_y, left_x, right_x, refined_cx = (
				_validate_bubble_rect(
					top_y, bot_y, left_x, right_x,
					px, py, geom))
			# re-measure edge mean
			all_edge_means[q_idx][choice] = _compute_edge_mean(
				gray, refined_cx, py,
				top_y, bot_y, left_x, right_x, geom)
			all_positions[q_idx][choice] = (refined_cx, py)
			all_edges[q_idx][choice] = (top_y, bot_y, left_x, right_x)
	# compute per-question spread for adaptive blank detection
	spreads = []
	for q_idx, edge_means in enumerate(all_edge_means):
		vals = list(edge_means.values())
		spread = max(vals) - min(vals)
		spreads.append((q_idx + 1, spread))
	# find threshold that separates filled from blank questions
	blank_threshold = _find_adaptive_threshold(spreads)
	# second pass: build results using self-referencing scores
	results = []
	for q_idx, edge_means in enumerate(all_edge_means):
		q_num = q_idx + 1
		vals = list(edge_means.values())
		max_edge = max(vals)
		spread = max_edge - min(vals)
		# self-referencing score: how much darker than lightest choice
		scores = {}
		for choice in choices:
			scores[choice] = (max_edge - edge_means[choice]) / 255.0
		# sort choices by score descending
		ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
		best_choice = ranked[0][0]
		best_score = ranked[0][1]
		second_score = ranked[1][1]
		# gap between best and second best (in pixel units for multi check)
		gap_from_second = best_score - second_score
		flags = ""
		if spread < blank_threshold:
			# no bubble stands out enough: blank question
			answer = ""
			flags = "BLANK"
		elif gap_from_second < multi_gap:
			# two bubbles have similar high scores: likely multiple marks
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
def draw_answer_debug(image: numpy.ndarray, template: dict,
	results: list) -> numpy.ndarray:
	"""Draw color-coded rectangular bubble overlay on a registered image.

	Uses semi-transparent filled rectangles so every detection zone is
	visible even when zones share edges. Uses detected edge positions
	from read_answers results for accurate overlay alignment.

	- Teal filled strips = measurement zones (left and right of center)
	- Orange outline = center exclusion zone (printed letter area)
	- Status color outline = outer bubble border (green/red/yellow/gray)

	Alpha blending (~0.3) keeps the underlying scantron image visible.

	Args:
		image: BGR registered image
		template: loaded template dictionary
		results: list of answer dicts from read_answers

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	# overlay layer for alpha-blended filled regions
	overlay = debug.copy()
	h, w = debug.shape[:2]
	choices = template["answers"]["choices"]
	# get geometry for fallback and center exclusion
	geom = omr_utils.template_loader.get_bubble_geometry_px(template, w, h)
	# int-cast geom values for cv2.rectangle drawing
	ce = int(geom["center_exclusion"])
	mi_v = int(geom["measurement_inset_v"])
	mi_h = int(geom["measurement_inset_h"])
	# define zone colors (BGR)
	teal = (200, 128, 0)
	orange = (0, 165, 255)
	alpha = 0.3
	for entry in results:
		q_num = entry["question"]
		positions = entry.get("positions", {})
		edges = entry.get("edges", {})
		for choice in choices:
			# use refined positions from read_answers if available
			if choice in positions:
				px, py = positions[choice]
			else:
				# fallback: recompute from template
				norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
					template, q_num, choice)
				px, py = omr_utils.template_loader.to_pixels(
					norm_x, norm_y, w, h)
			# get detected edges for this bubble
			if choice in edges:
				top_y, bot_y, left_x, right_x = edges[choice]
			else:
				# fallback to geometry-based defaults
				top_y, bot_y, left_x, right_x = _default_bounds(px, py, geom)
			# -- layer 1: teal filled measurement strips (alpha blended) --
			# matches _compute_edge_mean: inset from detected edges
			# left measurement strip
			cv2.rectangle(overlay,
				(left_x + mi_h, top_y + mi_v),
				(px - ce, bot_y - mi_v),
				teal, -1)
			# right measurement strip
			cv2.rectangle(overlay,
				(px + ce, top_y + mi_v),
				(right_x - mi_h, bot_y - mi_v),
				teal, -1)
	# blend the filled overlay onto the debug image
	cv2.addWeighted(overlay, alpha, debug, 1.0 - alpha, 0, debug)
	# draw outlines on top of the blended image (no alpha needed)
	for entry in results:
		q_num = entry["question"]
		answer = entry["answer"]
		flags = entry["flags"]
		scores = entry["scores"]
		positions = entry.get("positions", {})
		edges = entry.get("edges", {})
		for choice in choices:
			if choice in positions:
				px, py = positions[choice]
			else:
				norm_x, norm_y = omr_utils.template_loader.get_bubble_coords(
					template, q_num, choice)
				px, py = omr_utils.template_loader.to_pixels(
					norm_x, norm_y, w, h)
			# get detected edges for this bubble
			if choice in edges:
				top_y, bot_y, left_x, right_x = edges[choice]
			else:
				top_y, bot_y, left_x, right_x = _default_bounds(px, py, geom)
			# -- layer 3: orange center exclusion outline --
			cv2.rectangle(debug,
				(px - ce, top_y),
				(px + ce, bot_y),
				orange, 1)
			# -- layer 4: status-colored outer bubble outline --
			if choice == answer and "MULTIPLE" not in flags:
				# selected answer: green
				status_color = (0, 200, 0)
				thickness = 2
			elif choice == answer and "MULTIPLE" in flags:
				# selected but multiple: yellow
				status_color = (0, 255, 255)
				thickness = 2
			elif flags == "BLANK":
				# all blank: gray
				status_color = (128, 128, 128)
				thickness = 1
			else:
				# not selected: red
				status_color = (0, 0, 200)
				thickness = 1
			cv2.rectangle(debug,
				(left_x, top_y),
				(right_x, bot_y),
				status_color, thickness)
			# draw score text for high scores
			if scores[choice] > 0.10:
				score_text = f"{scores[choice]:.2f}"
				cv2.putText(debug, score_text,
					(px - 12, top_y - 2),
					cv2.FONT_HERSHEY_SIMPLEX, 0.25,
					status_color, 1)
	return debug
