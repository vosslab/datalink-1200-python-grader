"""Left-side timing mark detection for the DataLink 1200 scantron."""

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.timing_mark_anchors


# left-side structural footprint constants: 2 top + 10 ID + 50 question = 62
N_TOP = 2
N_ID = 10
N_Q = 50
N_TOTAL = N_TOP + N_ID + N_Q  # 62


#============================================
def _extract_left_candidates(gray_strip: numpy.ndarray) -> list:
	"""Extract dash-like candidates from the left timing strip.

	Uses Otsu threshold with morphological cleanup, then filters
	connected components by size and aspect ratio.

	Args:
		gray_strip: grayscale image of the left strip region

	Returns:
		list of candidate dicts sorted by center_y, each with keys:
		center_x, center_y, width, height, area, aspect_ratio,
		fill_ratio, bbox
	"""
	strip_h, strip_w = gray_strip.shape[:2]
	strip_area = strip_h * strip_w
	# Otsu threshold on the strip
	_, binary_inv = cv2.threshold(
		gray_strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	# morphological open removes small noise specks
	kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_open)
	# morphological close connects broken dash fragments
	kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel_close)
	contours, _ = cv2.findContours(
		binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# relative minimum area threshold
	min_area = strip_area * 0.0003
	candidates = []
	for contour in contours:
		x, y, bw, bh = cv2.boundingRect(contour)
		bbox_area = bw * bh
		if bbox_area < min_area:
			continue
		# fill ratio: actual contour pixels / bounding box area
		contour_area = cv2.contourArea(contour)
		fill_ratio = contour_area / float(bbox_area)
		# reject very low fill (noise, partial edges)
		if fill_ratio < 0.15:
			continue
		# aspect ratio: width / height
		aspect_ratio = float(bw) / max(1.0, float(bh))
		# reject extremely tall narrow shapes (vertical lines, text)
		if aspect_ratio < 0.3:
			continue
		# reject extremely large blobs (merged regions)
		if bbox_area > strip_area * 0.05:
			continue
		candidates.append({
			"center_x": float(x + bw / 2.0),
			"center_y": float(y + bh / 2.0),
			"width": bw,
			"height": bh,
			"area": bbox_area,
			"aspect_ratio": aspect_ratio,
			"fill_ratio": fill_ratio,
			"bbox": (x, y, bw, bh),
		})
	# sort by center_y
	candidates.sort(key=lambda c: c["center_y"])
	return candidates


#============================================
def _build_left_vertical_family(candidates: list) -> list:
	"""Filter candidates to a single dominant vertical family.

	Selects the dominant x-column of dash-like marks by finding the
	most common x-center cluster, then filters by width/height
	consistency.

	Args:
		candidates: list of candidate dicts sorted by center_y

	Returns:
		filtered list of dash-like candidates forming a vertical column
	"""
	if len(candidates) < 3:
		return candidates
	# find median x-center as the dominant column
	x_centers = [c["center_x"] for c in candidates]
	med_x = float(numpy.median(x_centers))
	# compute median width for tolerance
	widths = [c["width"] for c in candidates]
	med_w = float(numpy.median(widths))
	# filter to candidates near the median x-center
	x_tol = max(med_w * 2.0, 10.0)
	family = [c for c in candidates if abs(c["center_x"] - med_x) < x_tol]
	if len(family) < 3:
		return family
	# filter by height consistency: reject extreme outliers
	heights = [c["height"] for c in family]
	med_h = float(numpy.median(heights))
	# keep candidates within 3x of median height
	family = [c for c in family
		if 0.33 * med_h <= c["height"] <= 3.0 * med_h]
	# deduplicate near-overlapping marks by center_y
	if len(family) >= 2:
		family.sort(key=lambda c: c["center_y"])
		deduped = [family[0]]
		for c in family[1:]:
			prev = deduped[-1]
			# merge if within half the median height
			if abs(c["center_y"] - prev["center_y"]) < med_h * 0.5:
				# keep the one with higher fill_ratio
				if c.get("fill_ratio", 0) > prev.get("fill_ratio", 0):
					deduped[-1] = c
			else:
				deduped.append(c)
		family = deduped
	return family


#============================================
def _match_predictions_to_marks_y(predictions: list, marks: list,
	match_tol: float) -> list:
	"""Match predicted y-positions to observed mark candidates.

	For each predicted position, find the closest unmatched mark
	within the tolerance distance. Returns matched pairs.

	Args:
		predictions: list of predicted y-positions
		marks: list of candidate dicts with center_y
		match_tol: maximum distance for a valid match

	Returns:
		list of (predicted_y, candidate_dict) pairs
	"""
	matches = []
	used_indices = set()
	for pred_y in predictions:
		best_dist = match_tol
		best_idx = -1
		for i, mark in enumerate(marks):
			if i in used_indices:
				continue
			dist = abs(mark["center_y"] - pred_y)
			if dist < best_dist:
				best_dist = dist
				best_idx = i
		if best_idx >= 0:
			matches.append((pred_y, marks[best_idx]))
			used_indices.add(best_idx)
	return matches


#============================================
def _score_left_footprint(predictions: list, family: list,
	s_id: float, s_q: float) -> dict:
	"""Score a left-footprint hypothesis against observed candidates.

	Applies hard acceptance gates first, then continuous scoring.

	Args:
		predictions: list of 62 predicted y-positions (2 + 10 + 50)
		family: list of observed candidate dicts
		s_id: spacing within the ID segment
		s_q: spacing within the question segment

	Returns:
		dict with score (0.0 if gates fail), n_matched, residuals
	"""
	if len(predictions) != N_TOTAL:
		return {"score": 0.0, "n_matched": 0}
	# match tolerance: half of the question spacing
	match_tol = s_q * 0.5
	matches = _match_predictions_to_marks_y(predictions, family, match_tol)
	n_matched = len(matches)
	# hard gate: need at least 60% of 62 marks matched
	if n_matched < N_TOTAL * 0.60:
		return {"score": 0.0, "n_matched": n_matched}
	# compute residuals for the question segment (last 50 predictions)
	q_predictions = predictions[N_TOP + N_ID:]
	q_matches = _match_predictions_to_marks_y(
		q_predictions, family, match_tol)
	if len(q_matches) < N_Q * 0.60:
		return {"score": 0.0, "n_matched": n_matched}
	# question segment residuals
	q_residuals = [abs(pred - m["center_y"]) for pred, m in q_matches]
	max_q_residual = max(q_residuals)
	mean_q_residual = float(numpy.mean(q_residuals))
	# hard gate: max residual in question segment
	if max_q_residual > s_q * 0.45:
		return {"score": 0.0, "n_matched": n_matched}
	# question segment spacing consistency check
	q_matched_ys = sorted([m["center_y"] for _, m in q_matches])
	if len(q_matched_ys) >= 3:
		q_gaps = []
		for i in range(1, len(q_matched_ys)):
			q_gaps.append(q_matched_ys[i] - q_matched_ys[i - 1])
		q_cv = omr_utils.timing_mark_anchors._coeff_of_variation(q_gaps)
		# hard gate: spacing CV in question segment
		if q_cv > 0.40:
			return {"score": 0.0, "n_matched": n_matched}
	else:
		q_cv = 1.0
	# continuous scoring
	# match fraction (out of 62)
	match_score = n_matched / float(N_TOTAL)
	# residual quality (lower is better)
	residual_score = max(0.0, 1.0 - mean_q_residual / max(1.0, s_q * 0.3))
	# spacing consistency
	consistency_score = max(0.0, 1.0 - q_cv * 2.0)
	# combined score
	score = (0.50 * match_score
		+ 0.30 * residual_score
		+ 0.20 * consistency_score)
	return {
		"score": score,
		"n_matched": n_matched,
		"mean_q_residual": mean_q_residual,
		"max_q_residual": max_q_residual,
		"q_cv": q_cv,
	}


#============================================
def _repair_gaps(cy_vals: list, med_spacing: float) -> list:
	"""Fill missing marks by interpolating oversized gaps.

	Walks the sorted y-centers and checks each adjacent gap against
	the median spacing. If a gap is roughly N * median (N >= 2), it
	inserts N-1 interpolated positions to fill the hole.

	Args:
		cy_vals: sorted list of center_y values
		med_spacing: median within-segment spacing

	Returns:
		new sorted list of y-values with missing marks filled in
	"""
	if len(cy_vals) < 2 or med_spacing < 1.0:
		return list(cy_vals)
	repaired = [cy_vals[0]]
	for i in range(1, len(cy_vals)):
		gap = cy_vals[i] - cy_vals[i - 1]
		# how many spacings fit in this gap?
		n_steps = gap / med_spacing
		n_missing = int(round(n_steps)) - 1
		if n_missing >= 1 and n_steps > 1.4:
			# interpolate the missing marks
			actual_step = gap / (n_missing + 1)
			for j in range(1, n_missing + 1):
				repaired.append(cy_vals[i - 1] + j * actual_step)
		repaired.append(cy_vals[i])
	repaired.sort()
	return repaired


#============================================
def _fit_left_footprint(family: list) -> dict:
	"""Fit a 3-segment structural footprint (2+10+50) to left candidates.

	Simple blob-find-and-fit approach:
	1. Sort marks by y, compute median spacing
	2. Repair gaps (fill missing marks where gap > 1.4x median)
	3. Split at the largest remaining gap into upper and lower groups
	4. Upper group: first 2 = top marks, rest = ID marks
	5. Generate exactly 2/10/50 predictions using median spacing per segment
	6. Match predictions back to observed candidates

	Args:
		family: list of candidate dicts sorted by center_y

	Returns:
		dict with fitted parameters and marks, or None if no valid fit.
		Keys on success: top_marks, id_marks, question_marks,
		s_id, s_q, gap_a, gap_b, score, n_matched
	"""
	if len(family) < 20:
		print(f"  Left footprint: too few candidates ({len(family)})")
		return None
	# step 1: sort and compute global median spacing
	sorted_fam = sorted(family, key=lambda c: c["center_y"])
	cy_vals = [c["center_y"] for c in sorted_fam]
	n = len(cy_vals)
	gaps = [cy_vals[i + 1] - cy_vals[i] for i in range(n - 1)]
	med_gap = float(numpy.median(gaps))
	print(f"  Left footprint: {n} family marks, "
		f"median_gap={med_gap:.1f}px")
	# step 2: find structural boundaries (gaps > 1.8x median)
	big_gap_threshold = med_gap * 1.8
	big_gap_indices = sorted(
		[i for i in range(len(gaps)) if gaps[i] > big_gap_threshold])
	print(f"  Left footprint: {len(big_gap_indices)} big gaps "
		f"(threshold={big_gap_threshold:.1f}px)")
	# partition into segments based on number of big gaps found
	if len(big_gap_indices) >= 2:
		# two or more big gaps: split into 3 groups at first two boundaries
		ba = big_gap_indices[0]
		bb = big_gap_indices[1]
		seg_a_ys = cy_vals[:ba + 1]
		seg_b_ys = cy_vals[ba + 1:bb + 1]
		seg_c_ys = cy_vals[bb + 1:]
	elif len(big_gap_indices) == 1:
		# one big gap: split upper (top+ID) from questions at the gap
		boundary = big_gap_indices[0]
		upper_ys = cy_vals[:boundary + 1]
		seg_c_ys = cy_vals[boundary + 1:]
		# first 2 of upper = top, rest = ID
		seg_a_ys = upper_ys[:N_TOP]
		seg_b_ys = upper_ys[N_TOP:]
	else:
		# no big gaps: use count-based split (first 12 = upper, rest = lower)
		seg_a_ys = cy_vals[:N_TOP]
		seg_b_ys = cy_vals[N_TOP:N_TOP + N_ID]
		seg_c_ys = cy_vals[N_TOP + N_ID:]
	# repair gaps within each segment separately
	if len(seg_b_ys) >= 2:
		b_med = float(numpy.median([seg_b_ys[i + 1] - seg_b_ys[i]
			for i in range(len(seg_b_ys) - 1)]))
		seg_b_ys = _repair_gaps(seg_b_ys, b_med)
	if len(seg_c_ys) >= 2:
		c_med = float(numpy.median([seg_c_ys[i + 1] - seg_c_ys[i]
			for i in range(len(seg_c_ys) - 1)]))
		seg_c_ys = _repair_gaps(seg_c_ys, c_med)
	# validate minimum sizes
	if len(seg_c_ys) < 15:
		print("  Left footprint: question segment too small "
			f"({len(seg_c_ys)})")
		return None
	if len(seg_a_ys) < 1:
		print("  Left footprint: top segment empty")
		return None
	# compute per-segment spacings from observed marks
	if len(seg_b_ys) >= 2:
		b_gaps = [seg_b_ys[i + 1] - seg_b_ys[i]
			for i in range(len(seg_b_ys) - 1)]
		s_id = float(numpy.median(b_gaps))
	else:
		s_id = med_gap
	if len(seg_c_ys) >= 2:
		c_gaps = [seg_c_ys[i + 1] - seg_c_ys[i]
			for i in range(len(seg_c_ys) - 1)]
		s_q = float(numpy.median(c_gaps))
	else:
		s_q = med_gap
	# step 5: generate exactly N_TOP/N_ID/N_Q predictions per segment
	# segment A: 2 marks anchored at the first observed top mark
	pred_a = [seg_a_ys[0] + i * s_id for i in range(N_TOP)]
	# use observed marks if we have exactly 2
	if len(seg_a_ys) == N_TOP:
		pred_a = list(seg_a_ys)
	# segment B: 10 marks anchored at the first observed ID mark
	if seg_b_ys:
		pred_b = [seg_b_ys[0] + i * s_id for i in range(N_ID)]
	else:
		# no ID marks found; predict from top segment end
		id_start = seg_a_ys[-1] + s_id * 2.0
		pred_b = [id_start + i * s_id for i in range(N_ID)]
	# segment C: 50 marks anchored at the first observed question mark
	pred_c = [seg_c_ys[0] + i * s_q for i in range(N_Q)]
	print(f"  Left footprint: raw partition "
		f"{len(seg_a_ys)}/{len(seg_b_ys)}/{len(seg_c_ys)} "
		f"(target {N_TOP}/{N_ID}/{N_Q})")
	# step 6: build full prediction list and score
	predictions = pred_a + pred_b + pred_c
	result = _score_left_footprint(
		predictions, sorted_fam, s_id, s_q)
	if result["score"] <= 0.0:
		print(f"  Left footprint: scoring failed "
			f"(matched={result['n_matched']}/{N_TOTAL})")
		return None
	# match predictions back to real candidates for mark dicts
	match_tol = s_q * 0.5
	all_matches = _match_predictions_to_marks_y(
		predictions, sorted_fam, match_tol)
	# build a lookup from predicted y -> matched candidate
	match_map = {}
	for pred, cand in all_matches:
		match_map[pred] = cand
	# build mark dicts for each segment
	top_marks = []
	id_marks = []
	question_marks = []
	for i, pred_y in enumerate(predictions):
		matched_cand = match_map.get(pred_y)
		if matched_cand is not None:
			mark = dict(matched_cand)
		else:
			# synthetic mark at predicted position
			mark = {
				"center_x": 0.0,
				"center_y": pred_y,
				"width": 0,
				"height": 0,
				"area": 0,
				"bbox": (0, int(pred_y), 0, 0),
			}
		if i < N_TOP:
			top_marks.append(mark)
		elif i < N_TOP + N_ID:
			id_marks.append(mark)
		else:
			question_marks.append(mark)
	# compute inter-segment gaps
	s_top = pred_a[-1] - pred_a[0] if len(pred_a) >= 2 else s_id
	gap_a = pred_b[0] - pred_a[-1]
	gap_b = pred_c[0] - pred_b[-1]
	best_result = {
		"s_top": s_top,
		"s_id": s_id,
		"s_q": s_q,
		"y_top": pred_a[0],
		"y_id_start": pred_b[0],
		"y_q_start": pred_c[0],
		"gap_a": gap_a,
		"gap_b": gap_b,
		"score": result["score"],
		"n_matched": result["n_matched"],
		"top_marks": top_marks,
		"id_marks": id_marks,
		"question_marks": question_marks,
	}
	# diagnostics
	print(f"  Left footprint: score={result['score']:.3f} "
		f"matched={result['n_matched']}/{N_TOTAL}")
	print(f"  Left footprint: s_id={s_id:.1f}px "
		f"s_q={s_q:.1f}px "
		f"gap_a={gap_a:.1f}px "
		f"gap_b={gap_b:.1f}px")
	if "mean_q_residual" in result:
		print(f"  Left footprint: mean_q_residual="
			f"{result['mean_q_residual']:.1f}px "
			f"max_q_residual={result['max_q_residual']:.1f}px "
			f"q_cv={result['q_cv']:.3f}")
	return best_result
