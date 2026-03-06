"""Top-edge timing mark detection for the DataLink 1200 scantron."""

# Standard Library
import itertools

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.timing_mark_anchors


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
	# shared helper used by timing_marks_top
	row1_matches, _ = omr_utils.timing_mark_anchors._match_predictions_to_marks(
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
			# shared helper used by timing_marks_top
			row2_matches, row2_score = omr_utils.timing_mark_anchors._match_predictions_to_marks(
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
	# cluster all candidates into rows
	clusters = omr_utils.timing_mark_anchors._cluster_components_into_rows(
		all_candidates)
	if len(clusters) < 1:
		return None
	best = None
	best_score = -1.0
	for row1_idx, row1_cluster in enumerate(clusters):
		# skip clusters with too few marks
		if len(row1_cluster) < 3:
			continue
		# skip clusters that don't look like timing rows
		cluster_score = omr_utils.timing_mark_anchors._score_timing_row(
			row1_cluster)
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
	proj_bands = omr_utils.timing_mark_anchors._row_projection_bands(
		binary_inv)
	if proj_bands:
		band_strs = [f"[{b[0]}-{b[1]}]" for b in proj_bands]
		print(f"  Projection bands: {', '.join(band_strs)}")
	# step 2: extract all connected components
	all_candidates = omr_utils.timing_mark_anchors._extract_components(
		gray_strip)
	# step 3: cluster into horizontal rows
	clusters = omr_utils.timing_mark_anchors._cluster_components_into_rows(
		all_candidates)
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
		score = omr_utils.timing_mark_anchors._score_timing_row(cluster)
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
		deduped = omr_utils.timing_mark_anchors._dedupe_row_components(
			matched_comps, strip_w)
		print(f"  Using footprint: {len(deduped)} marks "
			f"(from {len(matched_comps)} matches)")
	else:
		# fallback to cluster-based selection
		if footprint is None:
			print("  Footprint: none found, using cluster fallback")
		else:
			print(f"  Footprint: score too low ({footprint['score']:.3f})"
				", using cluster fallback")
		deduped = omr_utils.timing_mark_anchors._dedupe_row_components(
			winning_cluster, strip_w)
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
