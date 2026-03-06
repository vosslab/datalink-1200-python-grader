"""Detect timing-mark anchors and build a relative coordinate transform."""

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.timing_marks_left
import omr_utils.timing_marks_top


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
# shared helper used by timing_marks_left and timing_marks_top
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
	transform["top_col_spacing"] = 0.0
	transform["top_fp_x0"] = 0.0
	transform["top_col_ratio"] = 0
	transform["left_strip_region"] = (0, 0, 0, 0)
	transform["top_strip_region"] = (0, 0, 0, 0)
	timing = template.get("timing_marks", {})
	left_edge = timing.get("left_edge", {})
	top_edge = timing.get("top_edge", {})
	# initialize left-side structural footprint keys
	transform["left_top_marks"] = []
	transform["left_id_marks"] = []
	transform["left_question_marks"] = []
	transform["left_s_id"] = 0.0
	transform["left_s_q"] = 0.0
	transform["left_gap_a"] = 0.0
	transform["left_gap_b"] = 0.0
	transform["left_raw_candidates"] = []
	# detect left timing marks using 3-segment structural fitting
	if left_edge:
		y1 = int(round(left_edge.get("start_y", 0.067) * h))
		y2 = int(round(left_edge.get("end_y", 0.91) * h))
		# left 8% of image width for timing dash search
		x1 = 0
		x2 = int(round(w * 0.08))
		y1 = max(0, y1)
		y2 = min(h, y2)
		transform["left_strip_region"] = (x1, y1, x2, y2)
		if y2 > y1 and x2 > x1:
			# extract candidates using structural approach
			left_strip_gray = gray[y1:y2, x1:x2]
			raw_candidates = omr_utils.timing_marks_left._extract_left_candidates(left_strip_gray)
			# convert strip-local coords to image coords
			raw_candidates_image = []
			for cand in raw_candidates:
				img_cand = dict(cand)
				bx, by, bw, bh = cand["bbox"]
				img_cand["center_y"] = cand["center_y"] + y1
				img_cand["center_x"] = cand["center_x"] + x1
				img_cand["bbox"] = (bx + x1, by + y1, bw, bh)
				raw_candidates_image.append(img_cand)
			transform["left_raw_candidates"] = raw_candidates_image
			# build vertical family from image-coordinate candidates
			family = omr_utils.timing_marks_left._build_left_vertical_family(raw_candidates_image)
			print(f"  Left strip: {len(raw_candidates_image)} raw "
				f"candidates, {len(family)} in vertical family")
			# attempt 3-segment structural fit
			footprint = omr_utils.timing_marks_left._fit_left_footprint(family)
			if footprint is not None and footprint["score"] > 0.0:
				# populate structured footprint results
				# convert segment marks to image-coord dicts with center_y and bbox
				all_fitted = []
				for seg_marks in [footprint["top_marks"],
					footprint["id_marks"], footprint["question_marks"]]:
					for mark in seg_marks:
						all_fitted.append({
							"center_y": mark["center_y"],
							"bbox": mark["bbox"],
						})
				transform["left_marks"] = all_fitted
				transform["left_top_marks"] = footprint["top_marks"]
				transform["left_id_marks"] = footprint["id_marks"]
				transform["left_question_marks"] = footprint["question_marks"]
				transform["left_s_id"] = footprint["s_id"]
				transform["left_s_q"] = footprint["s_q"]
				transform["left_gap_a"] = footprint["gap_a"]
				transform["left_gap_b"] = footprint["gap_b"]
				transform["left_confidence"] = footprint["score"]
			else:
				# fallback: store raw marks with zero confidence
				left_marks_image = []
				for cand in raw_candidates_image:
					left_marks_image.append({
						"center_y": cand["center_y"],
						"bbox": cand["bbox"],
					})
				transform["left_marks"] = left_marks_image
				transform["left_confidence"] = 0.0
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
			top_marks_local, raw_candidates = omr_utils.timing_marks_top._detect_top_primary_row(
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
			fp = omr_utils.timing_marks_top._detect_top_footprint(raw_candidates, x2 - x1)
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
				# derive the fine template-column step from the
				# measured footprint spacing; fp_spacing is the coarse
				# mark spacing (every Nth column), so divide by the
				# rounded integer ratio to recover the fine grid step
				fine_col_step = fp_spacing / max(1.0, round(col_ratio))
				# store fine column spacing for downstream bubble geometry
				transform["top_col_spacing"] = float(fine_col_step)
				# store lattice origin and integer ratio for lattice-based centers
				transform["top_fp_x0"] = float(fp_x0)
				transform["top_col_ratio"] = int(round(col_ratio))
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
	# draw left timing marks colored by segment
	# top 2: red, ID 10: green, question 50: cyan
	left_top_color = (0, 0, 255)    # red in BGR
	left_id_color = (0, 200, 0)     # green in BGR
	left_q_color = (255, 255, 0)    # cyan in BGR
	# draw top 2 marks (red)
	left_top_marks = transform.get("left_top_marks", [])
	for idx, mark in enumerate(left_top_marks):
		bx, by, bw, bh = mark["bbox"]
		if bw > 0 and bh > 0:
			cv2.rectangle(debug, (bx, by), (bx + bw, by + bh),
				left_top_color, 1)
			label_x = bx + bw + 2
			cv2.putText(debug, f"T{idx}", (label_x, by + bh),
				cv2.FONT_HERSHEY_SIMPLEX, 0.25, left_top_color, 1)
	# draw ID 10 marks (green)
	left_id_marks = transform.get("left_id_marks", [])
	for idx, mark in enumerate(left_id_marks):
		bx, by, bw, bh = mark["bbox"]
		if bw > 0 and bh > 0:
			cv2.rectangle(debug, (bx, by), (bx + bw, by + bh),
				left_id_color, 1)
			label_x = bx + bw + 2
			cv2.putText(debug, f"ID{idx}", (label_x, by + bh),
				cv2.FONT_HERSHEY_SIMPLEX, 0.25, left_id_color, 1)
	# draw question 50 marks (cyan)
	question_marks = transform.get("left_question_marks", [])
	for idx, mark in enumerate(question_marks):
		bx, by, bw, bh = mark["bbox"]
		if bw > 0 and bh > 0:
			cv2.rectangle(debug, (bx, by), (bx + bw, by + bh),
				left_q_color, 1)
			label_x = bx + bw + 2
			cv2.putText(debug, str(idx), (label_x, by + bh),
				cv2.FONT_HERSHEY_SIMPLEX, 0.25, left_q_color, 1)
	# draw segment boundary labels
	if left_top_marks and left_id_marks:
		mid_y = int((left_top_marks[-1]["center_y"]
			+ left_id_marks[0]["center_y"]) / 2)
		cv2.putText(debug, "gap-A", (5, mid_y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.25, (128, 128, 128), 1)
	if left_id_marks and question_marks:
		mid_y = int((left_id_marks[-1]["center_y"]
			+ question_marks[0]["center_y"]) / 2)
		cv2.putText(debug, "gap-B", (5, mid_y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.25, (128, 128, 128), 1)
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
	# draw horizontal guide lines from fitted question marks only
	# guide lines come exclusively from the 50-question segment
	guide_color = (0, 0, 255)  # bright red in BGR for high contrast
	guide_label_color = (0, 0, 200)  # slightly darker red for labels
	dash_len = 20
	gap_len = 10
	guide_thickness = 2
	if len(question_marks) == omr_utils.timing_marks_left.N_Q:
		for q_num in range(5, 51, 5):
			# 0-indexed: Q5 is index 4, Q50 is index 49
			mark_idx = q_num - 1
			cy = int(round(question_marks[mark_idx]["center_y"]))
			# draw dashed horizontal line across full page width
			x = 0
			while x < w:
				x_end = min(x + dash_len, w)
				cv2.line(debug, (x, cy), (x_end, cy),
					guide_color, guide_thickness)
				x += dash_len + gap_len
			# label with left-column Q# near left edge
			label_left = f"Q{q_num}"
			cv2.putText(debug, label_left, (5, cy - 6),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, guide_label_color, 2)
			# label with right-column Q# near right edge
			paired_q = q_num + 50
			label_right = f"Q{paired_q}"
			label_x = w - 65
			cv2.putText(debug, label_right, (label_x, cy - 6),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, guide_label_color, 2)
	return debug
