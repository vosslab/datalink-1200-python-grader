"""Read filled bubbles from a registered scantron image."""

# Standard Library
import csv

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.slot_map
import omr_utils.template_matcher


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
	"""Compute per-choice measurements using lattice ROI bounds.

	Uses lattice ROI bounds from SlotMap directly for all
	horizontal positioning.

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
			# use lattice bounds directly
			refined_cx = px
			left_x = def_left
			right_x = def_right
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
	ncc_diag_path: str = None) -> list:
	"""Optional template-matching refinement pass on localized rows.

	Uses unmasked NCC (TM_CCOEFF_NORMED) to refine bubble x and y
	positions by matching pixel templates of each letter against the
	image. Only updates positions where the NCC confidence is high
	and the shift is within the geometry limit. Modifies raw_data
	in place.

	Args:
		gray: grayscale image
		raw_data: list of per-question choice dicts from _stage_localize_rows
		template: loaded template dictionary
		measure_cfg: measurement constants from SlotMap.measure_cfg()
		bubble_templates: dict mapping letter to 5X oversize template array
		slot_map: SlotMap instance for lattice bounds (required)
		ncc_diag_path: optional path to write diagnostic CSV; when
			None, summary is still printed but CSV is not written

	Returns:
		list of per-slot diagnostic dicts (always returned for overlay)
	"""
	choices = template["answers"]["choices"]
	# require high confidence for NCC-driven position updates
	high_confidence = 0.45
	# diagnostic counters for NCC refinement summary
	total_slots = 0
	accepted_count = 0
	conf_rejected = 0
	shift_rejected = 0
	dx_sum = 0.0
	dy_sum = 0.0
	# per-slot diagnostic list
	diag_rows = []
	# collect all dx values for distribution stats
	all_dx = []
	all_dy = []
	all_score_peak = []
	all_score_seed = []
	max_shift = measure_cfg["refine_max_shift"]
	# search radius must cover the full allowed shift range
	search_radius = max(8, int(round(max_shift)) + 1)
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
		# run unmasked NCC refinement
		refined = omr_utils.template_matcher.refine_row_by_template(
			gray, bubble_templates, row_positions, choices,
			search_radius=search_radius,
			slot_dims=slot_dims)
		# update x and y positions when confidence is high and shift is small
		for choice in choices:
			total_slots += 1
			if choice not in refined:
				continue
			rcx, rcy, conf, score_seed = refined[choice]
			cd = q_choices[choice]
			# save seed position before any modification
			seed_x = cd["px"]
			seed_y = cd["refined_cy"]
			# compute signed dx/dy from seed to NCC peak
			dx_ncc = rcx - seed_x
			dy_ncc = rcy - seed_y
			# determine acceptance and reason
			accepted = False
			reason = "none"
			if conf < high_confidence:
				conf_rejected += 1
				reason = "conf_low"
			else:
				# store refinement confidence for downstream use
				cd["refinement_confidence"] = conf
				accepted = True
				reason = "accepted"
				# apply x correction
				slot_dx = abs(dx_ncc)
				if slot_dx <= max_shift:
					cd["px"] = rcx
					dx_sum += slot_dx
				else:
					shift_rejected += 1
					reason = "shift_too_large"
				# apply y correction and recompute bounds from lattice
				slot_dy = abs(dy_ncc)
				if slot_dy <= max_shift:
					cd["refined_cy"] = rcy
					dy_sum += slot_dy
					q_num = cd["q_num"]
					lat_top, lat_bot, _, _ = slot_map.roi_bounds(
						q_num, choice)
					# shift lattice bounds by the y offset
					lat_cy = slot_map.row_center(q_num)
					y_offset = rcy - lat_cy
					cd["top_y"] = lat_top + y_offset
					cd["bot_y"] = lat_bot + y_offset
				else:
					shift_rejected += 1
					reason = "shift_too_large"
				accepted_count += 1
			# save diagnostic positions for overlay drawing
			cd["seed_cx"] = seed_x
			cd["seed_cy"] = seed_y
			cd["ncc_cx"] = rcx
			cd["ncc_cy"] = rcy
			# build diagnostic record
			diag_rows.append({
				"q_num": cd["q_num"],
				"choice": choice,
				"seed_x": seed_x,
				"ncc_x": rcx,
				"dx": dx_ncc,
				"dy": dy_ncc,
				"score_peak": conf,
				"score_seed": score_seed,
				"score_delta": conf - score_seed,
				"accepted": accepted,
				"reason": reason,
			})
			# collect values for distribution stats
			all_dx.append(dx_ncc)
			all_dy.append(dy_ncc)
			all_score_peak.append(conf)
			all_score_seed.append(score_seed)
	# print compact NCC refinement summary
	mean_dx = dx_sum / max(accepted_count, 1)
	mean_dy = dy_sum / max(accepted_count, 1)
	print(f"  NCC refinement: {total_slots} slots,"
		f" {accepted_count} accepted (conf>={high_confidence}),"
		f" {conf_rejected} conf-rejected,"
		f" {shift_rejected} shift-rejected")
	print(f"  Mean |dx|={mean_dx:.1f}px  Mean |dy|={mean_dy:.1f}px")
	# extended distribution stats (always printed when NCC runs)
	if all_dx:
		dx_arr = numpy.array(all_dx)
		peak_arr = numpy.array(all_score_peak)
		seed_arr = numpy.array(all_score_seed)
		delta_arr = peak_arr - seed_arr
		# dx distribution
		print(f"  NCC dx distribution:"
			f" mean={numpy.mean(dx_arr):.1f}"
			f" median={numpy.median(dx_arr):.1f}"
			f" max={numpy.max(numpy.abs(dx_arr)):.1f}"
			f" std={numpy.std(dx_arr):.1f}")
		# score statistics
		print(f"  NCC scores:"
			f" mean_peak={numpy.mean(peak_arr):.3f}"
			f" median_peak={numpy.median(peak_arr):.3f}"
			f" mean_seed={numpy.mean(seed_arr):.3f}"
			f" mean_delta={numpy.mean(delta_arr):.4f}")
		# near-boundary count (peaks near search_radius limit)
		near_boundary = int(numpy.sum(
			numpy.abs(dx_arr) >= search_radius - 1))
		print(f"  Peak offset from center:"
			f" near-boundary count: {near_boundary}/{len(dx_arr)}")
		# config values for interpretation
		row_pitch = measure_cfg["row_pitch"]
		print(f"  Config: row_pitch={row_pitch:.1f}"
			f" search_radius={search_radius}"
			f" refine_max_shift={max_shift:.1f}")
	# write diagnostic CSV if path provided
	if ncc_diag_path is not None and diag_rows:
		_write_ncc_diag_csv(ncc_diag_path, diag_rows)
	return diag_rows


#============================================
def _write_ncc_diag_csv(csv_path: str, diag_rows: list) -> None:
	"""Write per-slot NCC diagnostic records to a CSV file.

	Args:
		csv_path: output CSV file path
		diag_rows: list of diagnostic dicts from _stage_template_refine
	"""
	fieldnames = [
		"q_num", "choice", "seed_x", "ncc_x", "dx", "dy",
		"score_peak", "score_seed", "score_delta",
		"accepted", "reason",
	]
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in diag_rows:
			# format floats for readability
			out = dict(row)
			for key in ("seed_x", "ncc_x", "dx", "dy"):
				out[key] = f"{out[key]:.2f}"
			for key in ("score_peak", "score_seed", "score_delta"):
				out[key] = f"{out[key]:.4f}"
			writer.writerow(out)
	print(f"  NCC diagnostics CSV: {csv_path}")


#============================================
def read_answers(image: numpy.ndarray, template: dict,
	slot_map: "omr_utils.slot_map.SlotMap",
	multi_gap: float = 0.03, bubble_templates: dict = None,
	refine_mode: str = "ncc",
	ncc_diag_path: str = None) -> tuple:
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
		refine_mode: refinement mode:
			"ncc" = NCC template refinement (default),
			"lattice" = pure geometry baseline (no refinement)
		ncc_diag_path: optional path to write NCC diagnostic CSV;
			when None, diagnostics are still printed but not saved

	Returns:
		tuple of (results, ncc_diag) where results is a list of
		dicts with keys: question, answer, scores, flags,
		positions, edges, refinement_confidence; and ncc_diag
		is a list of per-slot diagnostic dicts (or empty list if
		NCC was not run)
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
	# print active refinement mode
	print(f"  refine_mode: {refine_mode}")
	# localize rows using pure lattice positions
	raw_data = _stage_localize_rows(gray, template, measure_cfg, slot_map)
	# optional NCC template matching refinement
	ncc_diag = []
	if refine_mode == "ncc":
		if bubble_templates is None:
			bubble_templates = (
				omr_utils.template_matcher.try_load_bubble_templates())
		if bubble_templates:
			ncc_diag = _stage_template_refine(
				gray, raw_data, template, measure_cfg,
				bubble_templates, slot_map,
				ncc_diag_path=ncc_diag_path)
	# measure brightness and decide answers
	all_edge_means, all_positions, all_edges = _stage_measure_rows(
		gray, raw_data, template, measure_cfg, slot_map)
	results = _stage_decide_answers(
		all_edge_means, all_positions, all_edges, choices, multi_gap)
	# propagate NCC refinement confidence to results for debug overlay
	for q_idx, q_choices in enumerate(raw_data):
		conf_dict = {}
		ncc_positions = {}
		for choice, cd in q_choices.items():
			conf_dict[choice] = cd.get("refinement_confidence", -1.0)
			# propagate NCC diagnostic positions for overlay
			if "seed_cx" in cd:
				ncc_positions[choice] = {
					"seed_cx": cd["seed_cx"],
					"seed_cy": cd["seed_cy"],
					"ncc_cx": cd["ncc_cx"],
					"ncc_cy": cd["ncc_cy"],
				}
		results[q_idx]["refinement_confidence"] = conf_dict
		results[q_idx]["ncc_positions"] = ncc_positions
	return (results, ncc_diag)
