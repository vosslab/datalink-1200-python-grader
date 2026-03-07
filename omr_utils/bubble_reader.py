"""Read filled bubbles from a registered scantron image."""

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.slot_map
import omr_utils.template_matcher



#============================================
def _refine_bubble_edges_x(gray: numpy.ndarray, cx: int, cy: int,
	top_y: int, bot_y: int, measure_cfg: dict,
	default_left: int, default_right: int) -> tuple:
	"""Refine horizontal bubble position by detecting left and right bracket edges.

	Uses detected top_y/bot_y for a precise vertical ROI instead of
	relying on coarse estimates. Includes separation validation to
	reject false edge pairs.

	Args:
		gray: grayscale image (0=black, 255=white)
		cx: bubble center x in pixels (template estimate)
		cy: bubble center y in pixels (already y-refined)
		top_y: detected top edge y position
		bot_y: detected bottom edge y position
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		default_left: fallback left x from SlotMap lattice bounds
		default_right: fallback right x from SlotMap lattice bounds

	Returns:
		tuple of (refined_cx, left_x, right_x) as integers;
		falls back to defaults if edges are not clearly found
	"""
	h, w = gray.shape
	# derive search half-width from lattice bounds
	slot_half_w = (default_right - default_left) / 2.0
	hpad = measure_cfg["refine_pad_h"]
	# max shift from geometry; no hardcoded cap
	max_shift = float(measure_cfg["refine_max_shift"])
	# horizontal ROI: slot width + padding on each side
	x1 = max(0, int(cx - slot_half_w - hpad))
	x2 = min(w, int(cx + slot_half_w + hpad))
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
	min_separation = int(max(4, slot_half_w // 3))
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
	expected_separation = slot_half_w * 2
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
def _validate_bubble_rect(top_y: int, bot_y: int, left_x: int, right_x: int,
	cx: int, cy: int, measure_cfg: dict,
	fallback_bounds: tuple) -> tuple:
	"""Validate detected bubble rectangle against expected area and aspect ratio.

	If the detected rectangle deviates too far from expected dimensions,
	resets to lattice fallback bounds. Prevents partial detections from
	passing through to measurement.

	Args:
		top_y: detected top edge y
		bot_y: detected bottom edge y
		left_x: detected left edge x
		right_x: detected right edge x
		cx: template center x
		cy: refined center y
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		fallback_bounds: (top_y, bot_y, left_x, right_x) from
			SlotMap.roi_bounds(); always used as lattice fallback

	Returns:
		validated (top_y, bot_y, left_x, right_x, cx_out) tuple;
		cx_out may differ from cx if edges were kept
	"""
	# derive expected dimensions from lattice fallback bounds
	fb_top, fb_bot, fb_left, fb_right = fallback_bounds
	expected_w = fb_right - fb_left
	expected_h = fb_bot - fb_top
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
	# fall back to lattice bounds
	return (fb_top, fb_bot, fb_left, fb_right, cx)


#============================================
def _compute_edge_mean(gray: numpy.ndarray, cx: int, cy: int,
	top_y: int, bot_y: int, left_x: int, right_x: int,
	measure_cfg: dict) -> float:
	"""Compute mean brightness from dual left/right measurement zones."""
	left_mean, right_mean = _compute_dual_zone_means(
		gray, cx, cy, top_y, bot_y, left_x, right_x, measure_cfg)
	if left_mean < 0 or right_mean < 0:
		return -1.0
	edge_mean = (left_mean + right_mean) / 2.0
	return edge_mean


#============================================
def _compute_dual_zone_means(gray: numpy.ndarray, cx: int, cy: int,
	top_y: int, bot_y: int, left_x: int, right_x: int,
	measure_cfg: dict) -> tuple:
	"""Compute left and right fill-zone means separately.

	Three-zone model: fill zones are the green interior windows
	between bracket bars and center letter. Vertical inset is large
	(starts below bracket horizontal bars), horizontal inset skips
	past bracket arm width.

	Returns:
		tuple of (left_mean, right_mean), or (-1.0, -1.0) if invalid
	"""
	h, w = gray.shape
	if cx < 0 or cy < 0 or cx >= w or cy >= h:
		return (-1.0, -1.0)
	ce = measure_cfg["center_exclusion"]
	fi_v = measure_cfg["fill_inset_v"]
	bi = measure_cfg["bracket_inner_half"]
	# left fill window: bracket inner face to center exclusion
	lx1 = max(0, cx - bi)
	lx2 = max(0, cx - ce)
	# right fill window: center exclusion to bracket inner face
	rx1 = min(w, cx + ce)
	rx2 = min(w, cx + bi)
	# vertical bounds: large inset below/above bracket bars
	y1 = max(0, top_y + fi_v)
	y2 = min(h, bot_y - fi_v)
	# int-cast at array slicing boundary
	left_strip = gray[int(y1):int(y2), int(lx1):int(lx2)]
	right_strip = gray[int(y1):int(y2), int(rx1):int(rx2)]
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
def _stage_localize_rows(gray: numpy.ndarray, template: dict,
	measure_cfg: dict, slot_map: "omr_utils.slot_map.SlotMap") -> list:
	"""Localize bubble rows using pure lattice positions from SlotMap.

	No Sobel y-refinement. No neighbor correction. No linearity check.
	All positions come directly from the SlotMap lattice.

	Args:
		gray: grayscale image
		template: loaded template dictionary
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		slot_map: SlotMap instance (single geometry authority)

	Returns:
		list of per-question choice dicts with lattice positions
	"""
	h, w = gray.shape
	num_q = template["answers"]["num_questions"]
	choices = template["answers"]["choices"]
	raw_data = []
	for q_num in range(1, num_q + 1):
		q_choices = {}
		for choice in choices:
			cx, cy = slot_map.center(q_num, choice)
			# clamp to image bounds
			px = max(0, min(w - 1, cx))
			py = max(0, min(h - 1, cy))
			# get lattice ROI bounds
			top_y, bot_y, left_x, right_x = slot_map.roi_bounds(
				q_num, choice)
			q_choices[choice] = {
				"px": px, "py": py,
				"refined_cy": py, "refined_cx": px,
				"top_y": top_y, "bot_y": bot_y,
				"q_num": q_num,
			}
		raw_data.append(q_choices)
	return raw_data


#============================================
def _stage_measure_rows(gray: numpy.ndarray, raw_data: list,
	template: dict, measure_cfg: dict,
	slot_map: "omr_utils.slot_map.SlotMap") -> tuple:
	"""Refine horizontal edges and compute per-choice measurements.

	Uses lattice ROI bounds from SlotMap. Local x-edge refinement
	within the lattice ROI is still performed for precise measurement.

	Args:
		gray: grayscale image
		raw_data: list of per-question choice dicts
		template: loaded template dictionary
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		slot_map: SlotMap instance for lattice bounds

	Returns:
		tuple of (all_edge_means, all_positions, all_edges)
	"""
	num_q = template["answers"]["num_questions"]
	choices = template["answers"]["choices"]
	all_edge_means = []
	all_positions = []
	all_edges = []
	for q_idx in range(num_q):
		q_num = q_idx + 1
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
			# get lattice bounds for fallback
			lat_bounds = slot_map.roi_bounds(q_num, choice)
			_, _, def_left, def_right = lat_bounds
			# refine horizontal edges within the lattice ROI
			refined_cx, left_x, right_x = _refine_bubble_edges_x(
				gray, px, refined_cy, top_y, bot_y, measure_cfg,
				default_left=def_left, default_right=def_right)
			q_choices[choice]["refined_cx"] = refined_cx
			# validate the detected rectangle
			top_y, bot_y, left_x, right_x, refined_cx = (
				_validate_bubble_rect(
					top_y, bot_y, left_x, right_x,
					px, refined_cy, measure_cfg,
					fallback_bounds=lat_bounds))
			# measure brightness in the measurement zones
			edge_means[choice] = _compute_edge_mean(
				gray, refined_cx, refined_cy,
				top_y, bot_y, left_x, right_x, measure_cfg)
			positions[choice] = (refined_cx, refined_cy)
			edges[choice] = (top_y, bot_y, left_x, right_x)
		all_edge_means.append(edge_means)
		all_positions.append(positions)
		all_edges.append(edges)
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
	template: dict, measure_cfg: dict, bubble_templates: dict,
	slot_map: "omr_utils.slot_map.SlotMap",
	bubble_masks: dict = None) -> None:
	"""Optional template-matching refinement pass on localized rows.

	Uses NCC to refine bubble x and y positions by matching pixel
	templates of each letter against the image. When masks are
	available, uses masked NCC for bracket-focused matching with
	subpixel refinement. Only updates positions where the NCC
	confidence is high and the shift is within the geometry limit.
	Modifies raw_data in place.

	Args:
		gray: grayscale image
		raw_data: list of per-question choice dicts from _stage_localize_rows
		template: loaded template dictionary
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		bubble_templates: dict mapping letter to 5X oversize template array
		bubble_masks: optional dict mapping letter to mask array
		slot_map: SlotMap instance for lattice bounds (required)
	"""
	choices = template["answers"]["choices"]
	# require high confidence for NCC-driven position updates
	high_confidence = 0.45
	for q_choices in raw_data:
		# build position dict and slot dimensions for this row
		row_positions = {}
		slot_dims = {}
		for choice in choices:
			cd = q_choices[choice]
			row_positions[choice] = (cd["px"], cd["refined_cy"])
			# compute slot dimensions from lattice bounds
			q_num = cd["q_num"]
			lat_top, lat_bot, lat_left, lat_right = (
				slot_map.roi_bounds(q_num, choice))
			slot_dims[choice] = (
				lat_right - lat_left, lat_bot - lat_top)
		# run NCC refinement (masked if masks available)
		refined = omr_utils.template_matcher.refine_row_by_template(
			gray, bubble_templates, row_positions, choices,
			masks=bubble_masks, slot_dims=slot_dims)
		# update x and y positions when confidence is high and shift is small
		max_shift = measure_cfg["refine_max_shift"]
		for choice in choices:
			if choice not in refined:
				continue
			rcx, rcy, conf = refined[choice]
			if conf < high_confidence:
				continue
			cd = q_choices[choice]
			# store refinement confidence for downstream use
			cd["refinement_confidence"] = conf
			# apply x correction
			dx = abs(rcx - cd["px"])
			if dx <= max_shift:
				cd["px"] = rcx
			# apply y correction and recompute bounds from lattice
			dy = abs(rcy - cd["refined_cy"])
			if dy <= max_shift:
				cd["refined_cy"] = rcy
				q_num = cd["q_num"]
				lat_top, lat_bot, _, _ = slot_map.roi_bounds(
					q_num, choice)
				# shift lattice bounds by the y offset
				lat_cy = slot_map.row_center(q_num)
				y_offset = rcy - lat_cy
				cd["top_y"] = lat_top + y_offset
				cd["bot_y"] = lat_bot + y_offset


#============================================
def read_answers(image: numpy.ndarray, template: dict,
	slot_map: "omr_utils.slot_map.SlotMap",
	multi_gap: float = 0.03, bubble_templates: dict = None,
	bubble_masks: dict = None) -> list:
	"""Read all 100 answers from a registered scantron image.

	Uses self-referencing scoring: for each question, the lightest
	(emptiest) choice in the row is used as the baseline. This avoids
	dependency on local background strips, which can be unreliable
	for phone photos with uneven lighting or machine-printed marks.

	Blank detection uses adaptive thresholding: the spread (max - min
	edge mean) across all 100 questions is analyzed to find the natural
	gap between filled and blank populations.

	Args:
		image: BGR registered image
		template: loaded template dictionary
		slot_map: SlotMap instance (single geometry authority, required)
		multi_gap: min spread gap between top two scores for MULTIPLE flag
		bubble_templates: optional dict of pixel templates for NCC refinement;
			if None, attempts to load from config/bubble_templates/
		bubble_masks: optional dict of mask arrays for masked NCC;
			if None, attempts to load alongside templates

	Returns:
		list of dicts with keys: question, answer, scores, flags,
		positions, edges. answer is a choice letter or empty string
		if blank, scores is a dict of choice->score, flags is a string,
		edges maps choice to (top_y, bot_y, left_x, right_x)
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# light blur to reduce noise while preserving fill signal
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	choices = template["answers"]["choices"]
	measure_cfg = slot_map.measure_cfg()
	row_pitch = measure_cfg["row_pitch"]
	col_pitch = measure_cfg["col_pitch"]
	print(f"  slot_map geometry: row_pitch={row_pitch:.1f}px"
		f" col_pitch={col_pitch:.1f}px")
	# print K-constant pixel products for measurement zone verification
	print(f"  K pixel products:"
		f" center_excl={measure_cfg['center_exclusion']:.1f}px"
		f" bracket_inner_half={measure_cfg['bracket_inner_half']:.1f}px"
		f" refine_pad_h={measure_cfg['refine_pad_h']:.1f}px")
	# print lattice diagnostic for stride verification
	slot_map.print_lattice_diagnostic()
	# print ROI diagnostics for representative slots
	slot_map.print_roi_diagnostic()
	# localize rows using pure lattice positions
	raw_data = _stage_localize_rows(gray, template, measure_cfg, slot_map)
	# optional NCC template matching refinement
	if bubble_templates is None:
		loaded_templates, loaded_masks = (
			omr_utils.template_matcher.try_load_bubble_templates())
		bubble_templates = loaded_templates
		# use loaded masks if caller did not provide any
		if bubble_masks is None:
			bubble_masks = loaded_masks
	if bubble_templates:
		_stage_template_refine(gray, raw_data, template, measure_cfg,
			bubble_templates, slot_map,
			bubble_masks=bubble_masks)
	# measure brightness and decide answers
	all_edge_means, all_positions, all_edges = _stage_measure_rows(
		gray, raw_data, template, measure_cfg, slot_map)
	results = _stage_decide_answers(
		all_edge_means, all_positions, all_edges, choices, multi_gap)
	return results
