"""Detect timing-mark anchors and build a relative coordinate transform."""

# PIP3 modules
import cv2
import numpy


#============================================
def _default_transform() -> dict:
	"""Return identity transform with zero confidence."""
	return {
		"x_scale": 1.0,
		"x_offset": 0.0,
		"y_scale": 1.0,
		"y_offset": 0.0,
		"top_confidence": 0.0,
		"left_confidence": 0.0,
	}


#============================================
def _dedupe_sorted(values: list, min_gap: int = 3) -> list:
	"""Merge near-duplicate centers in a sorted list."""
	if not values:
		return []
	merged = [float(values[0])]
	for value in values[1:]:
		if abs(value - merged[-1]) <= min_gap:
			merged[-1] = (merged[-1] + value) / 2.0
		else:
			merged.append(float(value))
	return merged


#============================================
def _dedupe_sorted_marks(marks: list, axis: str, min_gap: int = 3) -> list:
	"""Merge near-duplicate marks in a sorted list, preserving bboxes.

	When two marks are within min_gap pixels along the axis, they are
	merged by averaging their centers and taking the union of bboxes.

	Args:
		marks: list of dicts with 'center' and 'bbox' keys, sorted by center
		axis: 'x' or 'y' indicating which axis centers are measured along
		min_gap: minimum distance between distinct marks

	Returns:
		deduplicated list of mark dicts
	"""
	if not marks:
		return []
	merged = [marks[0].copy()]
	for mark in marks[1:]:
		prev = merged[-1]
		if abs(mark["center"] - prev["center"]) <= min_gap:
			# average the centers
			prev["center"] = (prev["center"] + mark["center"]) / 2.0
			# union of bounding boxes
			px, py, pw, ph = prev["bbox"]
			mx, my, mw, mh = mark["bbox"]
			ux = min(px, mx)
			uy = min(py, my)
			ux2 = max(px + pw, mx + mw)
			uy2 = max(py + ph, my + mh)
			prev["bbox"] = (ux, uy, ux2 - ux, uy2 - uy)
		else:
			merged.append(mark.copy())
	return merged


#============================================
def _detect_marks_in_strip(gray_strip: numpy.ndarray, axis: str) -> list:
	"""Detect contour centers and bounding boxes in a grayscale strip.

	Uses local Otsu thresholding on the strip (not the full image) for
	better contrast adaptation to phone photos with uneven lighting.
	Applies morphological cleanup to reduce noise and connect broken marks.

	Args:
		gray_strip: grayscale image of the strip region
		axis: 'x' for horizontal centers, 'y' for vertical centers

	Returns:
		list of dicts with 'center' (float) and 'bbox' (x, y, w, h)
		in strip-local coordinates, sorted and deduplicated by center
	"""
	# local Otsu threshold on the strip for better contrast handling
	_, binary_inv = cv2.threshold(
		gray_strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	# morphological open removes small noise specks
	kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_open)
	# morphological close connects broken mark fragments
	if axis == "x":
		# top marks: close horizontally
		kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
	else:
		# left marks: small close
		kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	binary_inv = cv2.morphologyEx(
		binary_inv, cv2.MORPH_CLOSE, kernel_close)
	contours, _ = cv2.findContours(
		binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	marks = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		area = w * h
		# lowered from 10 to 5 for small marks at low resolution
		if area < 5:
			continue
		# relaxed aspect ratio filters
		if axis == "y":
			# left dashes are wider than tall
			if h > 0 and w / h < 1.2:
				continue
		else:
			# top boxes: reject very tall narrow contours (text noise)
			if w > 0 and h / w > 4.0:
				continue
		if axis == "x":
			center = float(x + w / 2.0)
		else:
			center = float(y + h / 2.0)
		marks.append({"center": center, "bbox": (x, y, w, h)})
	# sort by center position
	marks.sort(key=lambda m: m["center"])
	return _dedupe_sorted_marks(marks, axis)


#============================================
def _extract_components(gray_strip: numpy.ndarray) -> list:
	"""Extract connected components from a grayscale strip.

	Uses Otsu threshold with morphological open (3x3) and close (5x3)
	to clean up noise and connect broken fragments. Computes per-component
	statistics for downstream scoring.

	Args:
		gray_strip: grayscale image of the strip region

	Returns:
		list of component dicts with keys: center_x, center_y,
		width, height, area, aspect_ratio, fill_ratio, bbox
	"""
	strip_h, strip_w = gray_strip.shape[:2]
	strip_area = strip_h * strip_w
	# Otsu threshold on the strip
	_, binary_inv = cv2.threshold(
		gray_strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	# morphological open removes small noise specks
	kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_open)
	# morphological close connects broken block fragments
	kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
	binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel_close)
	contours, _ = cv2.findContours(
		binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# relative minimum area threshold
	min_area = strip_area * 0.0005
	components = []
	for contour in contours:
		x, y, bw, bh = cv2.boundingRect(contour)
		bbox_area = bw * bh
		if bbox_area < min_area:
			continue
		# fill ratio: actual contour pixels / bounding box area
		contour_area = cv2.contourArea(contour)
		fill_ratio = contour_area / float(bbox_area)
		# aspect ratio: width / height
		aspect_ratio = float(bw) / max(1.0, float(bh))
		components.append({
			"center_x": float(x + bw / 2.0),
			"center_y": float(y + bh / 2.0),
			"width": bw,
			"height": bh,
			"area": bbox_area,
			"aspect_ratio": aspect_ratio,
			"fill_ratio": fill_ratio,
			"bbox": (x, y, bw, bh),
		})
	return components


#============================================
def _cluster_components_into_rows(components: list) -> list:
	"""Group components into horizontal rows by y-center proximity.

	Sorts components by center_y and splits into clusters when the
	y-gap between consecutive components exceeds k * median_height.

	Args:
		components: list of component dicts with center_y and height

	Returns:
		list of row clusters, each a list of component dicts
	"""
	if not components:
		return []
	# sort by y-center
	sorted_comps = sorted(components, key=lambda c: c["center_y"])
	if len(sorted_comps) == 1:
		return [sorted_comps]
	# compute median component height for gap threshold
	heights = [c["height"] for c in sorted_comps]
	med_height = float(numpy.median(heights))
	# gap threshold: 0.75 * median height
	gap_threshold = max(3.0, 0.75 * med_height)
	# build clusters by splitting on large y-gaps
	clusters = []
	current_cluster = [sorted_comps[0]]
	for comp in sorted_comps[1:]:
		y_gap = comp["center_y"] - current_cluster[-1]["center_y"]
		if y_gap > gap_threshold:
			clusters.append(current_cluster)
			current_cluster = [comp]
		else:
			current_cluster.append(comp)
	clusters.append(current_cluster)
	return clusters


#============================================
def _coeff_of_variation(values: list) -> float:
	"""Compute coefficient of variation (std / mean) for a list of values.

	Args:
		values: list of numeric values

	Returns:
		CV value, or 1.0 if mean is near zero
	"""
	if len(values) < 2:
		return 0.0
	arr = numpy.array(values, dtype=float)
	mean_val = numpy.mean(arr)
	if abs(mean_val) < 1e-6:
		return 1.0
	cv = float(numpy.std(arr) / abs(mean_val))
	return cv


#============================================
def _score_timing_row(cluster: list) -> float:
	"""Score a candidate row cluster as a primary timing row.

	Evaluates multiple factors normalized to 0-1 range:
	- Component count (4-12 ideal)
	- Size consistency (CV of areas)
	- Fill consistency (CV of fill ratios)
	- Fill magnitude (median fill ratio)
	- Y-alignment tightness
	- X-spacing regularity (CV of gaps)
	- Aspect ratio consistency (CV)

	Args:
		cluster: list of component dicts in a single row

	Returns:
		weighted score, higher = more likely primary timing row
	"""
	n = len(cluster)
	if n < 2:
		return 0.0
	# factor 1: component count (4-12 ideal, weight 0.10)
	if 4 <= n <= 12:
		count_score = 1.0
	elif n == 3:
		count_score = 0.5
	elif n > 12:
		# penalize gradually for too many components
		count_score = max(0.0, 1.0 - (n - 12) * 0.1)
	else:
		count_score = 0.2
	# factor 2: size consistency (CV of areas, weight 0.20)
	areas = [c["area"] for c in cluster]
	size_cv = _coeff_of_variation(areas)
	# lower CV = more consistent = higher score
	size_score = max(0.0, 1.0 - size_cv)
	# factor 3: fill consistency (CV of fill ratios, weight 0.10)
	fills = [c["fill_ratio"] for c in cluster]
	fill_cv = _coeff_of_variation(fills)
	fill_consistency_score = max(0.0, 1.0 - fill_cv * 2.0)
	# factor 4: fill magnitude (median fill, weight 0.15)
	med_fill = float(numpy.median(fills))
	fill_mag_score = min(1.0, med_fill / 0.8)
	# factor 5: y-alignment tightness (weight 0.15)
	y_vals = [c["center_y"] for c in cluster]
	y_range = max(y_vals) - min(y_vals)
	med_height = float(numpy.median([c["height"] for c in cluster]))
	# perfect alignment: y_range < 0.5 * median height
	if med_height > 0:
		y_tightness = max(0.0, 1.0 - y_range / max(1.0, med_height))
	else:
		y_tightness = 0.0
	# factor 6: x-spacing regularity (CV of gaps, weight 0.15)
	sorted_by_x = sorted(cluster, key=lambda c: c["center_x"])
	x_gaps = []
	for i in range(1, len(sorted_by_x)):
		gap = sorted_by_x[i]["center_x"] - sorted_by_x[i - 1]["center_x"]
		x_gaps.append(gap)
	if x_gaps:
		gap_cv = _coeff_of_variation(x_gaps)
		spacing_score = max(0.0, 1.0 - gap_cv)
	else:
		spacing_score = 0.0
	# factor 7: aspect ratio consistency (CV, weight 0.15)
	aspects = [c["aspect_ratio"] for c in cluster]
	aspect_cv = _coeff_of_variation(aspects)
	aspect_score = max(0.0, 1.0 - aspect_cv)
	# weighted sum
	score = (
		0.10 * count_score
		+ 0.20 * size_score
		+ 0.10 * fill_consistency_score
		+ 0.15 * fill_mag_score
		+ 0.15 * y_tightness
		+ 0.15 * spacing_score
		+ 0.15 * aspect_score
	)
	return score


#============================================
def _dedupe_row_components(components: list,
	strip_width: int) -> list:
	"""Merge components whose x-centers are too close together.

	Sorts by center_x and merges components within 2% of strip width.
	Keeps the component with higher fill_ratio when merging.

	Args:
		components: list of component dicts
		strip_width: width of the search strip in pixels

	Returns:
		deduplicated list sorted by center_x
	"""
	if not components:
		return []
	sorted_comps = sorted(components, key=lambda c: c["center_x"])
	min_gap = strip_width * 0.02
	merged = [sorted_comps[0]]
	for comp in sorted_comps[1:]:
		prev = merged[-1]
		if abs(comp["center_x"] - prev["center_x"]) < min_gap:
			# keep the one with higher fill ratio
			if comp["fill_ratio"] > prev["fill_ratio"]:
				merged[-1] = comp
		else:
			merged.append(comp)
	return merged


#============================================
def _row_projection_bands(binary_strip: numpy.ndarray,
	min_band_fraction: float = 0.10) -> list:
	"""Find dominant horizontal bands from row-wise dark pixel projection.

	Sums dark pixels per row, then identifies contiguous runs where the
	row sum exceeds a threshold (fraction of max row sum). Returns band
	y-ranges that likely contain timing footprint rows.

	Args:
		binary_strip: binary (inverted) strip image (255 = dark)
		min_band_fraction: minimum fraction of peak row sum to consider
			a row as part of a band

	Returns:
		list of (y_start, y_end) tuples for each detected band,
		sorted by y_start
	"""
	# sum dark pixels per row (row projection profile)
	row_sums = numpy.sum(binary_strip > 0, axis=1).astype(float)
	if row_sums.max() < 1:
		return []
	# threshold: rows with dark pixel count above fraction of peak
	threshold = row_sums.max() * min_band_fraction
	above = row_sums >= threshold
	# find contiguous bands
	bands = []
	in_band = False
	band_start = 0
	for y_idx in range(len(above)):
		if above[y_idx] and not in_band:
			band_start = y_idx
			in_band = True
		elif not above[y_idx] and in_band:
			bands.append((band_start, y_idx))
			in_band = False
	if in_band:
		bands.append((band_start, len(above)))
	return bands


#============================================
def _approx_gcd_spacing(gap1: float, gap2: float,
	tol: float = 0.12) -> float:
	"""Find the largest spacing that approximately divides both gaps.

	Tries small integer multipliers (1-6) for gap1 to derive a candidate
	spacing, then checks if gap2 is also a near-integer multiple.

	Args:
		gap1: first gap distance
		gap2: second gap distance
		tol: fractional tolerance for integer-ness check

	Returns:
		best spacing value, or 0.0 if no valid spacing found
	"""
	best_spacing = 0.0
	for n1 in range(1, 7):
		candidate = gap1 / n1
		if candidate < 1.0:
			continue
		# check if gap2 is a near-integer multiple of candidate
		n2_float = gap2 / candidate
		n2_round = round(n2_float)
		if n2_round < 1:
			continue
		# fractional error relative to the integer
		frac_err = abs(n2_float - n2_round) / max(1.0, n2_round)
		if frac_err > tol:
			continue
		# prefer the largest spacing (most parsimonious model)
		if candidate > best_spacing:
			best_spacing = candidate
	return best_spacing


#============================================
def _fit_row1_model(seed_comps: list, strip_width: int) -> dict:
	"""Fit a linear column-spacing model from 3 seed components.

	Infers a base spacing from the pairwise gaps of the seed triplet,
	then predicts column positions extending from the left edge to the
	right edge of the strip.

	Args:
		seed_comps: list of 3 component dicts sorted by center_x
		strip_width: width of the search strip in pixels

	Returns:
		model dict with keys: x0, spacing, predictions, seed_positions
		or None if no valid spacing found
	"""
	x_vals = sorted([c["center_x"] for c in seed_comps])
	g1 = x_vals[1] - x_vals[0]
	g2 = x_vals[2] - x_vals[1]
	# find base spacing from the two gaps
	spacing = _approx_gcd_spacing(g1, g2)
	# minimum spacing: 2% of strip width
	min_spacing = strip_width * 0.02
	if spacing < min_spacing:
		return None
	# compute origin (x0) by snapping first seed to nearest column
	k0 = round(x_vals[0] / spacing)
	x0 = x_vals[0] - k0 * spacing
	# generate predicted positions across the strip
	predictions = []
	col_idx = 0
	while True:
		x_pred = x0 + col_idx * spacing
		if x_pred > strip_width:
			break
		if x_pred >= 0:
			predictions.append(float(x_pred))
		col_idx += 1
	if len(predictions) < 3:
		return None
	model = {
		"x0": float(x0),
		"spacing": float(spacing),
		"predictions": predictions,
		"seed_positions": x_vals,
	}
	return model


#============================================
def _predict_row2_right_thins(model: dict, row1_matches: list) -> list:
	"""Predict x-positions for the two right-side Row-2 thin marks.

	On the DataLink 1200 form, the Row-2 companion row has two fixed
	thin marks on the right side in a gap-thin-gap-thin pattern.
	These fall at the second-to-last and last columns of the actual
	Row-1 footprint (not the full strip width).

	Args:
		model: fitted Row-1 model dict with x0, spacing, predictions
		row1_matches: list of (predicted_x, component) matched pairs

	Returns:
		list of predicted x-positions for the two thin marks
	"""
	x0 = model["x0"]
	spacing = model["spacing"]
	if not row1_matches:
		return []
	# find the rightmost matched Row-1 column index
	rightmost_x = max(m[1]["center_x"] for m in row1_matches)
	last_matched_col = round((rightmost_x - x0) / spacing)
	# thin marks at the second-to-last and last columns of the footprint
	# gap-thin-gap-thin pattern: cols (last-2) and last have thin marks
	thin_cols = [last_matched_col - 2, last_matched_col]
	thin_positions = []
	for col in thin_cols:
		x_pred = x0 + col * spacing
		if x_pred >= 0:
			thin_positions.append(float(x_pred))
	return thin_positions


#============================================
def _match_predictions_to_marks(predictions: list, marks: list,
	match_tol: float) -> tuple:
	"""Match predicted positions to observed mark components.

	For each predicted position, find the closest unmatched mark
	within the tolerance distance. Returns matched pairs and score.

	Args:
		predictions: list of predicted x-positions
		marks: list of component dicts with center_x
		match_tol: maximum distance for a valid match

	Returns:
		tuple of (matches, score) where matches is a list of
		(predicted_x, component) pairs and score is the fraction
		of predictions that matched
	"""
	matches = []
	used_indices = set()
	for pred_x in predictions:
		best_dist = match_tol
		best_idx = -1
		for i, mark in enumerate(marks):
			if i in used_indices:
				continue
			dist = abs(mark["center_x"] - pred_x)
			if dist < best_dist:
				best_dist = dist
				best_idx = i
		if best_idx >= 0:
			matches.append((pred_x, marks[best_idx]))
			used_indices.add(best_idx)
	n_pred = len(predictions)
	score = len(matches) / max(1, n_pred)
	return (matches, score)


#============================================
def _score_footprint_hypothesis(model: dict, row1_comps: list,
	row2_comps: list, strip_width: int) -> dict:
	"""Score a footprint hypothesis combining Row-1 and Row-2 evidence.

	Scoring formula: 0.65 * row1_support + 0.35 * row2_support
	minus penalties for gap irregularity.

	Args:
		model: fitted Row-1 model dict
		row1_comps: components in the Row-1 cluster
		row2_comps: components in the Row-2 candidate cluster
		strip_width: width of the search strip

	Returns:
		dict with score, row1_matches, row2_matches, model
	"""
	spacing = model["spacing"]
	# match tolerance: fraction of one spacing unit
	match_tol = spacing * 0.35
	# score Row 1: match predicted positions to observed marks
	row1_sorted = sorted(row1_comps, key=lambda c: c["center_x"])
	row1_matches, _ = _match_predictions_to_marks(
		model["predictions"], row1_sorted, match_tol)
	# row1_score: fraction of OBSERVED marks explained by the model
	# (not fraction of predictions matched -- that penalizes correct models
	# that predict positions at empty columns)
	n_observed = len(row1_comps)
	n_matched = len(row1_matches)
	row1_score = n_matched / max(1, n_observed)
	if row1_score <= 0:
		return {"score": 0.0, "row1_matches": [], "row2_matches": [],
			"model": model}
	# score Row 2: match predicted thin mark positions
	thin_preds = _predict_row2_right_thins(model, row1_matches)
	row2_matches = []
	row2_score = 0.0
	if thin_preds and row2_comps:
		row2_sorted = sorted(row2_comps, key=lambda c: c["center_x"])
		# filter Row 2 to thin marks (shorter height than Row 1 median)
		row1_heights = [c["height"] for c in row1_comps]
		row1_med_h = float(numpy.median(row1_heights))
		# thin marks should be noticeably shorter than Row 1 blocks
		thin_candidates = [c for c in row2_sorted
			if c["height"] < row1_med_h * 0.85]
		if thin_candidates:
			row2_matches, row2_score = _match_predictions_to_marks(
				thin_preds, thin_candidates, match_tol)
	# penalty for gap irregularity in matched Row-1 positions
	gap_penalty = 0.0
	if len(row1_matches) >= 3:
		matched_xs = sorted([m[1]["center_x"] for m in row1_matches])
		gaps = []
		for i in range(1, len(matched_xs)):
			gap = matched_xs[i] - matched_xs[i - 1]
			# normalize gap to spacing units
			gap_units = gap / spacing
			# distance from nearest integer
			gap_frac = abs(gap_units - round(gap_units))
			gaps.append(gap_frac)
		# average fractional error as penalty
		avg_frac_err = float(numpy.mean(gaps))
		gap_penalty = avg_frac_err * 0.3
	# bonus for absolute match count: prefer models that explain more marks
	count_bonus = min(0.10, n_matched * 0.015)
	# combined score, capped at 1.0
	combined = (0.65 * row1_score + 0.35 * row2_score
		- gap_penalty + count_bonus)
	combined = max(0.0, min(1.0, combined))
	result = {
		"score": combined,
		"row1_matches": row1_matches,
		"row2_matches": row2_matches,
		"model": model,
		"row1_score": row1_score,
		"row2_score": row2_score,
		"gap_penalty": gap_penalty,
		"count_bonus": count_bonus,
	}
	return result


#============================================
def _detect_top_footprint(all_candidates: list,
	strip_width: int) -> dict:
	"""Detect the top timing footprint using model fitting and Row-2 verification.

	Iterates row clusters, enumerates triplets from prominent blobs,
	fits a Row-1 spacing model, predicts Row-2 thin marks, and scores
	the full footprint. Returns the best hypothesis or None.

	Args:
		all_candidates: list of all component dicts in the top strip
		strip_width: width of the search strip in pixels

	Returns:
		best hypothesis dict with score, model, matches, and the
		winning row1 and row2 clusters. None if no valid footprint.
	"""
	import itertools
	# cluster all candidates into rows
	clusters = _cluster_components_into_rows(all_candidates)
	if len(clusters) < 1:
		return None
	best = None
	best_score = -1.0
	for row1_idx, row1_cluster in enumerate(clusters):
		# skip clusters with too few marks
		if len(row1_cluster) < 3:
			continue
		# skip clusters that don't look like timing rows
		cluster_score = _score_timing_row(row1_cluster)
		if cluster_score < 0.40:
			continue
		# sort by x for scoring
		row1_sorted = sorted(row1_cluster, key=lambda c: c["center_x"])
		# choose prominent blobs: top N by area
		by_area = sorted(row1_cluster, key=lambda c: c["area"],
			reverse=True)
		n_seeds = min(6, len(by_area))
		seed_pool = by_area[:n_seeds]
		# sort seed pool by x for consistent triplet generation
		seed_pool = sorted(seed_pool, key=lambda c: c["center_x"])
		# find Row-2 candidate: next cluster below Row-1
		row1_med_y = float(numpy.median(
			[c["center_y"] for c in row1_cluster]))
		row2_cluster = []
		for other_idx, other_cluster in enumerate(clusters):
			if other_idx == row1_idx:
				continue
			other_med_y = float(numpy.median(
				[c["center_y"] for c in other_cluster]))
			# Row 2 should be below Row 1
			row1_med_h = float(numpy.median(
				[c["height"] for c in row1_cluster]))
			y_offset = other_med_y - row1_med_y
			# within reasonable vertical offset range
			if row1_med_h < y_offset < row1_med_h * 5:
				row2_cluster = other_cluster
				break
		# enumerate triplets from seed pool
		if len(seed_pool) < 3:
			continue
		for triplet in itertools.combinations(seed_pool, 3):
			triplet_sorted = sorted(
				triplet, key=lambda c: c["center_x"])
			model = _fit_row1_model(triplet_sorted, strip_width)
			if model is None:
				continue
			hypothesis = _score_footprint_hypothesis(
				model, row1_sorted, row2_cluster, strip_width)
			if hypothesis["score"] > best_score:
				best_score = hypothesis["score"]
				best = hypothesis
				best["row1_cluster"] = row1_sorted
				best["row2_cluster"] = row2_cluster
				best["row1_idx"] = row1_idx
	if best is not None:
		n_preds = len(best["model"]["predictions"])
		n_r1 = len(best["row1_matches"])
		n_r2 = len(best["row2_matches"])
		print(f"  Footprint: score={best['score']:.3f} "
			f"(R1={best['row1_score']:.2f} "
			f"R2={best['row2_score']:.2f} "
			f"pen={best['gap_penalty']:.3f}) "
			f"spacing={best['model']['spacing']:.1f}px "
			f"matches={n_r1}/{n_preds}+{n_r2}/2")
	return best


#============================================
def _detect_top_primary_row(gray_strip: numpy.ndarray) -> tuple:
	"""Detect the primary timing row in the top strip using row-pattern scoring.

	Pipeline:
	1. Threshold and compute row projection profile to find dark bands
	2. Extract all connected components
	3. Cluster components into horizontal rows by y-center
	4. Score each row cluster for timing-row characteristics,
	   with a bonus for rows inside projection bands
	5. Select the highest-scoring row as primary
	6. Deduplicate near-x components in the winning row

	Uses projection profiles as guidance: row-wise dark-pixel sums
	localize the timing footprint bands, which bias scoring toward
	rows that lie within those bands. The projections are structural
	cues, not the final detector.

	Does not force a specific mark count. Prints diagnostics showing
	candidate count, cluster scores, and final mark positions.

	Args:
		gray_strip: grayscale image of the top strip region

	Returns:
		tuple of (primary_row_marks, all_candidates) where:
		- primary_row_marks: list of dicts with 'center' (x-position)
		  and 'bbox' keys, sorted by x
		- all_candidates: full component list for debug overlay
	"""
	strip_h, strip_w = gray_strip.shape[:2]
	# step 1: threshold and compute row projection to find dark bands
	_, binary_inv = cv2.threshold(
		gray_strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	proj_bands = _row_projection_bands(binary_inv)
	if proj_bands:
		band_strs = [f"[{b[0]}-{b[1]}]" for b in proj_bands]
		print(f"  Projection bands: {', '.join(band_strs)}")
	# step 2: extract all connected components
	all_candidates = _extract_components(gray_strip)
	# step 3: cluster into horizontal rows
	clusters = _cluster_components_into_rows(all_candidates)
	n_candidates = len(all_candidates)
	n_clusters = len(clusters)
	print(f"  Top strip: {n_candidates} candidates, "
		f"{n_clusters} row clusters")
	if n_clusters == 0:
		print("  WARNING: no row clusters found")
		return ([], all_candidates)
	# step 4: score each cluster with projection band bonus
	best_score = -1.0
	best_idx = 0
	for i, cluster in enumerate(clusters):
		score = _score_timing_row(cluster)
		med_y = float(numpy.median([c["center_y"] for c in cluster]))
		med_area = float(numpy.median([c["area"] for c in cluster]))
		med_w = float(numpy.median([c["width"] for c in cluster]))
		med_h = float(numpy.median([c["height"] for c in cluster]))
		# bonus if row median_y falls inside a projection band
		in_band = any(b[0] <= med_y <= b[1] for b in proj_bands)
		if in_band:
			# small bonus (up to 0.10) for being inside a dark band
			score = min(1.0, score + 0.05)
		print(f"  Row {i}: n={len(cluster)} score={score:.2f} "
			f"median_y={med_y:.0f} size={med_w:.0f}x{med_h:.0f} "
			f"area={med_area:.0f}"
			f"{' (in band)' if in_band else ''}")
		if score > best_score:
			best_score = score
			best_idx = i
	# step 4: select winning cluster (fallback)
	winning_cluster = clusters[best_idx]
	print(f"  Primary row: {len(winning_cluster)} components, "
		f"score={best_score:.2f}")
	# step 5: try footprint detection with Row-2 verification
	footprint = _detect_top_footprint(all_candidates, strip_w)
	if footprint is not None and footprint["score"] > 0.20:
		# use footprint-matched marks instead of raw cluster
		matched_comps = [m[1] for m in footprint["row1_matches"]]
		deduped = _dedupe_row_components(matched_comps, strip_w)
		print(f"  Using footprint: {len(deduped)} marks "
			f"(from {len(matched_comps)} matches)")
	else:
		# fallback to cluster-based selection
		if footprint is None:
			print("  Footprint: none found, using cluster fallback")
		else:
			print(f"  Footprint: score too low ({footprint['score']:.3f})"
				", using cluster fallback")
		deduped = _dedupe_row_components(winning_cluster, strip_w)
	if len(deduped) < len(winning_cluster):
		print(f"  After x-dedup: {len(deduped)} components")
	# build result with 'center' key for x-position (backward compat)
	x_positions = [int(round(c["center_x"])) for c in deduped]
	print(f"  Final x-positions: {x_positions}")
	result = []
	for comp in deduped:
		result.append({
			"center": comp["center_x"],
			"bbox": comp["bbox"],
		})
	return (result, all_candidates)


#============================================
def _detect_centers_in_strip(gray_strip: numpy.ndarray, axis: str) -> list:
	"""Detect contour centers along one axis inside a grayscale strip.

	Thin wrapper around _detect_marks_in_strip for backward compatibility.

	Args:
		gray_strip: grayscale strip image
		axis: 'x' or 'y'

	Returns:
		list of center positions (floats)
	"""
	marks = _detect_marks_in_strip(gray_strip, axis)
	return [m["center"] for m in marks]


#============================================
def _estimate_axis_transform(observed: list, expected_start: float,
	expected_end: float, expected_count: int) -> tuple:
	"""Estimate (scale, offset, confidence) for one axis from timing marks."""
	min_marks = max(4, expected_count // 8)
	if len(observed) < min_marks:
		return (1.0, 0.0, 0.0)
	exp_span = float(expected_end - expected_start)
	if exp_span <= 1.0 or expected_count < 2:
		return (1.0, 0.0, 0.0)
	exp_step = exp_span / float(expected_count - 1)
	if exp_step <= 0.0:
		return (1.0, 0.0, 0.0)
	obs_arr = numpy.array(observed, dtype=float)
	# Map observed marks to nearest expected index, then fit against
	# those expected positions. This remains stable when edge marks are
	# missing or merged, which previously biased endpoint-based fitting.
	approx_idx = numpy.rint((obs_arr - expected_start) / exp_step).astype(int)
	approx_idx = numpy.clip(approx_idx, 0, expected_count - 1)
	unique_idx = numpy.unique(approx_idx)
	if len(unique_idx) < max(3, min_marks // 2):
		return (1.0, 0.0, 0.0)
	exp_pts = []
	obs_pts = []
	for idx in unique_idx:
		mask = (approx_idx == idx)
		exp_pts.append(expected_start + float(idx) * exp_step)
		obs_pts.append(float(numpy.mean(obs_arr[mask])))
	exp_pts = numpy.array(exp_pts, dtype=float)
	obs_pts = numpy.array(obs_pts, dtype=float)
	if len(exp_pts) >= 2:
		scale, offset = numpy.polyfit(exp_pts, obs_pts, 1)
	else:
		scale = 1.0
		offset = float(obs_pts[0] - exp_pts[0])
	fit = scale * exp_pts + offset
	rmse = float(numpy.sqrt(numpy.mean((obs_pts - fit) ** 2)))
	# confidence from two factors:
	# 1. count adequacy: scale threshold to expected count so small mark
	#    sets (e.g., 7 top blocks) can still reach full confidence
	adequacy_threshold = min(25.0, expected_count * 0.8)
	count_adequate = min(1.0, len(unique_idx) / adequacy_threshold)
	# 2. span coverage: do detected marks cover the expected range?
	obs_span = float(obs_pts[-1] - obs_pts[0]) if len(obs_pts) >= 2 else 0.0
	span_ratio = obs_span / max(1.0, exp_span)
	confidence = min(count_adequate, span_ratio)
	# penalize unrealistic scales
	if scale < 0.93 or scale > 1.07:
		confidence *= 0.25
	elif scale < 0.97 or scale > 1.03:
		confidence *= 0.60
	# relaxed RMSE penalty: phone photos have more geometric distortion
	if rmse > 12.0:
		confidence *= 0.50
	if rmse > 20.0:
		confidence *= 0.50
	return (float(scale), float(offset), float(confidence))


#============================================
def estimate_anchor_transform(gray: numpy.ndarray, template: dict) -> dict:
	"""Estimate anchor-relative x/y transform from top and left timing marks.

	Args:
		gray: registered grayscale image
		template: loaded template dictionary

	Returns:
		dict with x/y scale+offset, confidence scores, and detected
		mark positions. Keys include:
		- x_scale, x_offset, y_scale, y_offset: transform parameters
		- top_confidence, left_confidence: detection quality scores
		- left_marks: list of dicts with center_y and bbox in image coords
		- top_marks: list of dicts with center_x and bbox in image coords
		- left_strip_region: (x1, y1, x2, y2) of left search strip
		- top_strip_region: (x1, y1, x2, y2) of top search strip
	"""
	h, w = gray.shape
	transform = _default_transform()
	# initialize empty mark lists and strip regions
	transform["left_marks"] = []
	transform["top_marks"] = []
	transform["top_raw_candidates"] = []
	transform["top_row2_marks"] = []
	transform["left_strip_region"] = (0, 0, 0, 0)
	transform["top_strip_region"] = (0, 0, 0, 0)
	timing = template.get("timing_marks", {})
	left_edge = timing.get("left_edge", {})
	top_edge = timing.get("top_edge", {})
	# detect left timing marks (y centers in image coordinates)
	if left_edge:
		y1 = int(round(left_edge.get("start_y", 0.067) * h))
		y2 = int(round(left_edge.get("end_y", 0.91) * h))
		# left 10% of image width for timing dash search
		x1 = 0
		x2 = int(round(w * 0.10))
		y1 = max(0, y1)
		y2 = min(h, y2)
		transform["left_strip_region"] = (x1, y1, x2, y2)
		if y2 > y1 and x2 > x1:
			# pass grayscale strip for local thresholding
			left_strip_gray = gray[y1:y2, x1:x2]
			left_marks_local = _detect_marks_in_strip(
				left_strip_gray, axis="y")
			# convert strip-local coords to image coords
			left_marks_image = []
			for mark in left_marks_local:
				bx, by, bw, bh = mark["bbox"]
				left_marks_image.append({
					"center_y": mark["center"] + y1,
					"bbox": (bx + x1, by + y1, bw, bh),
				})
			transform["left_marks"] = left_marks_image
			# extract centers for axis transform fitting
			left_centers = [m["center_y"] for m in left_marks_image]
			exp_start = float(left_edge.get("start_y", 0.067) * h)
			exp_end = float(left_edge.get("end_y", 0.91) * h)
			exp_count = int(left_edge.get("expected_count", 60))
			y_scale, y_offset, left_conf = _estimate_axis_transform(
				left_centers, exp_start, exp_end, exp_count)
			if left_conf >= 0.35:
				transform["y_scale"] = y_scale
				transform["y_offset"] = y_offset
				transform["left_confidence"] = left_conf
	# detect top timing marks (x centers in image coordinates)
	if top_edge:
		# full width, top 6% height for row-pattern detection
		x1 = 0
		x2 = w
		y1 = 0
		y2 = int(round(h * 0.06))
		transform["top_strip_region"] = (x1, y1, x2, y2)
		if y2 > y1 and x2 > x1:
			# pass grayscale strip for row-pattern primary row detection
			top_strip_gray = gray[y1:y2, x1:x2]
			top_marks_local, raw_candidates = _detect_top_primary_row(
				top_strip_gray)
			# store raw candidates in image coords for debug overlay
			raw_candidates_image = []
			for cand in raw_candidates:
				bx, by, bw, bh = cand["bbox"]
				raw_candidates_image.append({
					"center_x": cand["center_x"] + x1,
					"center_y": cand["center_y"] + y1,
					"bbox": (bx + x1, by + y1, bw, bh),
					"area": cand.get("area", bw * bh),
				})
			transform["top_raw_candidates"] = raw_candidates_image
			# convert strip-local coords to image coords
			top_marks_image = []
			for mark in top_marks_local:
				bx, by, bw, bh = mark["bbox"]
				top_marks_image.append({
					"center_x": mark["center"] + x1,
					"bbox": (bx + x1, by + y1, bw, bh),
				})
			transform["top_marks"] = top_marks_image
			# extract centers for axis transform fitting
			top_centers = [m["center_x"] for m in top_marks_image]
			exp_start = float(top_edge.get("start_x", 0.04) * w)
			exp_end = float(top_edge.get("end_x", 0.96) * w)
			exp_count = int(top_edge.get("expected_count", 53))
			# use footprint model spacing to map marks to column indices
			# the footprint spacing is N times the template column step
			exp_step = (exp_end - exp_start) / max(1, exp_count - 1)
			# try footprint-based transform first
			top_conf = 0.0
			fp = _detect_top_footprint(raw_candidates, x2 - x1)
			if fp is not None and fp["score"] > 0.20:
				# store Row-2 matches in image coords for debug overlay
				r2_marks_image = []
				for _pred, comp in fp.get("row2_matches", []):
					bx, by, bw, bh = comp["bbox"]
					r2_marks_image.append({
						"center_x": comp["center_x"],
						"center_y": comp["center_y"],
						"bbox": (bx, by, bw, bh),
					})
				transform["top_row2_marks"] = r2_marks_image
				fp_spacing = fp["model"]["spacing"]
				fp_x0 = fp["model"]["x0"]
				# map each mark to its column in the 53-column grid
				# using the footprint model's spacing/origin
				col_ratio = fp_spacing / max(1.0, exp_step)
				if col_ratio >= 1.0:
					mapped_centers = []
					for cx in top_centers:
						# footprint column index (float)
						fp_col_f = (cx - fp_x0) / fp_spacing
						# template column index (float ratio)
						tmpl_col = round(fp_col_f * col_ratio)
						# clamp to valid range
						tmpl_col = max(0, min(exp_count - 1, tmpl_col))
						mapped_centers.append(
							exp_start + tmpl_col * exp_step)
					if len(mapped_centers) >= 3:
						# compute transform using mapped expected
						# positions vs observed positions
						exp_arr = numpy.array(mapped_centers)
						obs_arr = numpy.array(top_centers)
						x_scale, x_offset = numpy.polyfit(
							exp_arr, obs_arr, 1)
						fit_vals = x_scale * exp_arr + x_offset
						rmse = float(numpy.sqrt(
							numpy.mean((obs_arr - fit_vals) ** 2)))
						# confidence from mark count and fit quality
						n_marks = len(top_centers)
						top_conf = min(1.0, n_marks / 6.0)
						# penalize unrealistic scales
						if x_scale < 0.93 or x_scale > 1.07:
							top_conf *= 0.25
						elif x_scale < 0.97 or x_scale > 1.03:
							top_conf *= 0.60
						if rmse > 12.0:
							top_conf *= 0.50
						if top_conf >= 0.35:
							transform["x_scale"] = float(x_scale)
							transform["x_offset"] = float(x_offset)
							transform["top_confidence"] = float(top_conf)
			# fallback: try raw mark positions with 7-mark model
			if top_conf < 0.35:
				x_scale, x_offset, top_conf = _estimate_axis_transform(
					top_centers, exp_start, exp_end, 7)
				if top_conf >= 0.35:
					transform["x_scale"] = x_scale
					transform["x_offset"] = x_offset
					transform["top_confidence"] = top_conf
	return transform


#============================================
def mark_index_to_normalized(mark_index: float, edge_start: float,
	edge_end: float, edge_count: int) -> float:
	"""Convert a fractional timing mark index to a normalized coordinate.

	A fractional index of 10.46 means the position is 46% of the way
	between mark 10 and mark 11. This converts such indices to the
	normalized (0.0-1.0) coordinate system used by the template.

	Args:
		mark_index: fractional mark index (e.g., 10.46)
		edge_start: normalized position of first mark on the edge
		edge_end: normalized position of last mark on the edge
		edge_count: total number of expected marks on the edge

	Returns:
		normalized coordinate (0.0 to 1.0)
	"""
	# spacing between adjacent marks in normalized coordinates
	step = (edge_end - edge_start) / max(1, edge_count - 1)
	norm = edge_start + mark_index * step
	return norm


#============================================
def normalized_to_mark_index(norm_coord: float, edge_start: float,
	edge_end: float, edge_count: int) -> float:
	"""Convert a normalized coordinate to a fractional timing mark index.

	Inverse of mark_index_to_normalized. Used to compute mark indices
	from existing hardcoded coordinates during template migration.

	Args:
		norm_coord: normalized coordinate (0.0 to 1.0)
		edge_start: normalized position of first mark on the edge
		edge_end: normalized position of last mark on the edge
		edge_count: total number of expected marks on the edge

	Returns:
		fractional timing mark index
	"""
	step = (edge_end - edge_start) / max(1, edge_count - 1)
	mark_index = (norm_coord - edge_start) / step
	return mark_index


#============================================
def draw_timing_candidates_debug(image: numpy.ndarray,
	transform: dict) -> numpy.ndarray:
	"""Draw all raw contour candidates in the timing mark search strips.

	Shows every detected contour before filtering, with search strip
	regions highlighted. Useful for diagnosing why the detector accepts
	or rejects specific contours.

	Args:
		image: BGR image to annotate (will be copied)
		transform: dict from estimate_anchor_transform with raw candidates

	Returns:
		annotated copy of the image showing all candidates
	"""
	debug = image.copy()
	overlay = debug.copy()
	h, w = debug.shape[:2]
	# colors (BGR): fill colors are muted, outline colors are brighter
	gray_color = (160, 160, 160)
	strip_fill_left = (180, 180, 60)
	strip_fill_top = (180, 60, 180)
	strip_outline_left = (255, 255, 0)
	strip_outline_top = (255, 0, 255)
	strip_alpha = 0.07
	# draw semi-transparent search strip regions with separate outline
	lx1, ly1, lx2, ly2 = transform.get("left_strip_region", (0, 0, 0, 0))
	if lx2 > lx1 and ly2 > ly1:
		cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), strip_fill_left, -1)
	tx1, ty1, tx2, ty2 = transform.get("top_strip_region", (0, 0, 0, 0))
	if tx2 > tx1 and ty2 > ty1:
		cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), strip_fill_top, -1)
	cv2.addWeighted(overlay, strip_alpha, debug, 1.0 - strip_alpha, 0, debug)
	# draw bright outlines for strip boundaries (on top of blended fill)
	if lx2 > lx1 and ly2 > ly1:
		cv2.rectangle(debug, (lx1, ly1), (lx2, ly2), strip_outline_left, 1)
	if tx2 > tx1 and ty2 > ty1:
		cv2.rectangle(debug, (tx1, ty1), (tx2, ty2), strip_outline_top, 1)
	# cluster raw candidates to show row membership with different colors
	raw_candidates = transform.get("top_raw_candidates", [])
	# assign cluster colors by re-clustering the raw candidates
	cluster_colors = [
		(160, 160, 160),  # gray for cluster 0
		(0, 200, 255),    # yellow for cluster 1 (primary row)
		(60, 180, 120),   # green for cluster 2 (companion row)
		(120, 60, 180),   # purple for cluster 3
		(180, 120, 60),   # brown-blue for cluster 4
		(60, 120, 220),   # orange for cluster 5
	]
	if raw_candidates:
		# enrich raw candidates with height key for clustering
		enriched = []
		for cand in raw_candidates:
			enriched_cand = dict(cand)
			enriched_cand["height"] = cand["bbox"][3]
			enriched.append(enriched_cand)
		# re-cluster for color assignment
		cluster_comps = _cluster_components_into_rows(enriched)
		# build lookup: component center -> cluster index
		cluster_map = {}
		for cidx, cluster in enumerate(cluster_comps):
			for comp in cluster:
				key = (round(comp["center_x"], 1), round(comp["center_y"], 1))
				cluster_map[key] = cidx
		# draw cluster summary labels at left edge of image
		for cidx, cluster in enumerate(cluster_comps):
			color = cluster_colors[cidx % len(cluster_colors)]
			med_y = int(numpy.median([c["center_y"] for c in cluster]))
			label = f"Row{cidx}: n={len(cluster)}"
			cv2.putText(debug, label, (5, med_y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
		for cand in raw_candidates:
			bx, by, bw, bh = cand["bbox"]
			# look up cluster index for color
			key = (round(cand["center_x"], 1), round(cand["center_y"], 1))
			cidx = cluster_map.get(key, 0)
			color = cluster_colors[cidx % len(cluster_colors)]
			cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), color, 1)
			# label with cluster index and dimensions
			cv2.putText(debug, f"R{cidx}:{bw}x{bh}",
				(bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX,
				0.25, color, 1)
	# draw all left marks in gray
	left_marks = transform.get("left_marks", [])
	for mark in left_marks:
		bx, by, bw, bh = mark["bbox"]
		cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), gray_color, 1)
	return debug


#============================================
def draw_timing_mark_debug(image: numpy.ndarray,
	transform: dict) -> numpy.ndarray:
	"""Draw final timing mark overlay with guide lines across the page.

	Shows only the selected marks (not raw candidates). Left marks get
	cyan bounding boxes with index labels. Top marks get magenta
	bounding boxes labeled T1-T7 with vertical guide lines extending
	down the full page height.

	Args:
		image: BGR image to annotate (will be copied)
		transform: dict from estimate_anchor_transform with mark positions

	Returns:
		annotated copy of the image showing only final selected marks
	"""
	debug = image.copy()
	overlay = debug.copy()
	h, w = debug.shape[:2]
	# colors (BGR): fill colors are muted, outline colors are brighter
	cyan = (255, 255, 0)
	magenta = (255, 0, 255)
	strip_fill_left = (180, 180, 60)
	strip_fill_top = (180, 60, 180)
	strip_outline_left = (255, 255, 0)
	strip_outline_top = (255, 0, 255)
	strip_alpha = 0.07
	# draw semi-transparent search strip regions with separate outline
	lx1, ly1, lx2, ly2 = transform.get("left_strip_region", (0, 0, 0, 0))
	if lx2 > lx1 and ly2 > ly1:
		cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), strip_fill_left, -1)
	tx1, ty1, tx2, ty2 = transform.get("top_strip_region", (0, 0, 0, 0))
	if tx2 > tx1 and ty2 > ty1:
		cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), strip_fill_top, -1)
	cv2.addWeighted(overlay, strip_alpha, debug, 1.0 - strip_alpha, 0, debug)
	# draw bright outlines for strip boundaries (on top of blended fill)
	if lx2 > lx1 and ly2 > ly1:
		cv2.rectangle(debug, (lx1, ly1), (lx2, ly2), strip_outline_left, 1)
	if tx2 > tx1 and ty2 > ty1:
		cv2.rectangle(debug, (tx1, ty1), (tx2, ty2), strip_outline_top, 1)
	# draw left timing mark bounding boxes (cyan) with index labels
	left_marks = transform.get("left_marks", [])
	for idx, mark in enumerate(left_marks):
		bx, by, bw, bh = mark["bbox"]
		cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), cyan, 1)
		# index label to the right of the mark
		label_x = bx + bw + 2
		label_y = by + bh
		cv2.putText(debug, str(idx), (label_x, label_y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.25, cyan, 1)
	# draw final top primary row marks (magenta) with vertical guide lines
	top_marks = transform.get("top_marks", [])
	for idx, mark in enumerate(top_marks):
		bx, by, bw, bh = mark["bbox"]
		# magenta bounding box around each block
		cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), magenta, 2)
		# label M1..Mn (count not forced) below the block
		label_text = f"M{idx + 1}"
		label_x = bx
		label_y = by + bh + 12
		cv2.putText(debug, label_text, (label_x, label_y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, magenta, 1)
		# extend vertical guide line from block center-x down full page
		cx = int(round(mark["center_x"]))
		cv2.line(debug, (cx, 0), (cx, h), magenta, 1)
	# draw Row-2 thin mark boxes (columns 10 and 12 of the footprint)
	row2_color = (255, 128, 0)  # bright cyan-blue in BGR
	row2_marks = transform.get("top_row2_marks", [])
	for mark in row2_marks:
		bx, by, bw, bh = mark["bbox"]
		cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), row2_color, 2)
		# label as R2 (Row 2 verification mark)
		cv2.putText(debug, "R2",
			(bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX,
			0.30, row2_color, 1)
	# draw horizontal guide lines for every-fifth question row pair
	# anchor from bottom: the last left mark corresponds to Q50.
	# work backwards to find the mark index for each question.
	guide_color = (0, 80, 160)  # dark orange-brown in BGR
	n_left = len(left_marks)
	# dash parameters: segment length and gap length in pixels
	dash_len = 12
	gap_len = 8
	if n_left >= 10:
		# last mark index is Q50; Q_n maps to mark (last - (50 - n))
		last_idx = n_left - 1
		for q_num in range(5, 51, 5):
			# mark index for question q_num, anchored from bottom
			mark_idx = last_idx - (50 - q_num)
			if mark_idx < 0 or mark_idx >= n_left:
				continue
			# get y-position from the detected left mark
			cy = int(round(left_marks[mark_idx]["center_y"]))
			# draw dashed horizontal line across full page width
			x = 0
			while x < w:
				x_end = min(x + dash_len, w)
				cv2.line(debug, (x, cy), (x_end, cy),
					guide_color, 1)
				x += dash_len + gap_len
			# label with left-column Q# near left edge
			paired_q = q_num + 50
			label_left = f"Q{q_num}"
			cv2.putText(debug, label_left, (5, cy - 4),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, guide_color, 1)
			# label with right-column Q# near right edge
			label_right = f"Q{paired_q}"
			label_x = w - 55
			cv2.putText(debug, label_right, (label_x, cy - 4),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, guide_color, 1)
	return debug
