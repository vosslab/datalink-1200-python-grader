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
def _detect_marks_in_strip(binary_inv: numpy.ndarray, axis: str) -> list:
	"""Detect contour centers and bounding boxes in a binary strip image.

	Args:
		binary_inv: inverted binary image of the strip region
		axis: 'x' for horizontal centers, 'y' for vertical centers

	Returns:
		list of dicts with 'center' (float) and 'bbox' (x, y, w, h)
		in strip-local coordinates, sorted and deduplicated by center
	"""
	contours, _ = cv2.findContours(
		binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	marks = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		area = w * h
		if area < 10:
			continue
		# filter by aspect ratio to reject noise from wider search strips
		if axis == "y":
			# left dashes are wider than tall
			if h > 0 and w / h < 1.5:
				continue
		else:
			# top boxes are roughly square or wider than tall
			if w > 0 and h / w > 3.0:
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
def _detect_centers_in_strip(binary_inv: numpy.ndarray, axis: str) -> list:
	"""Detect contour centers along one axis inside a binary strip image.

	Thin wrapper around _detect_marks_in_strip for backward compatibility.
	"""
	marks = _detect_marks_in_strip(binary_inv, axis)
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
	coverage = len(unique_idx) / float(max(1, expected_count))
	confidence = min(1.0, coverage)
	# Penalize unrealistic scales and poor fit residuals.
	if scale < 0.93 or scale > 1.07:
		confidence *= 0.25
	elif scale < 0.97 or scale > 1.03:
		confidence *= 0.60
	if rmse > 8.0:
		confidence *= 0.50
	if rmse > 15.0:
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
	transform["left_strip_region"] = (0, 0, 0, 0)
	transform["top_strip_region"] = (0, 0, 0, 0)
	timing = template.get("timing_marks", {})
	left_edge = timing.get("left_edge", {})
	top_edge = timing.get("top_edge", {})
	# Otsu-threshold once and reuse for both strips
	_, binary_inv = cv2.threshold(
		gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	# detect left timing marks (y centers in image coordinates)
	if left_edge:
		left_x = int(round(left_edge.get("x", 0.018) * w))
		y1 = int(round(left_edge.get("start_y", 0.067) * h))
		y2 = int(round(left_edge.get("end_y", 0.91) * h))
		strip_half_w = max(10, int(round(w * 0.03)))
		x1 = max(0, left_x - strip_half_w)
		x2 = min(w, left_x + strip_half_w)
		y1 = max(0, y1)
		y2 = min(h, y2)
		transform["left_strip_region"] = (x1, y1, x2, y2)
		if y2 > y1 and x2 > x1:
			left_strip = binary_inv[y1:y2, x1:x2]
			left_marks_local = _detect_marks_in_strip(left_strip, axis="y")
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
		top_y = int(round(top_edge.get("y", 0.012) * h))
		x1 = int(round(top_edge.get("start_x", 0.04) * w))
		x2 = int(round(top_edge.get("end_x", 0.96) * w))
		strip_half_h = max(10, int(round(h * 0.03)))
		y1 = max(0, top_y - strip_half_h)
		y2 = min(h, top_y + strip_half_h)
		x1 = max(0, x1)
		x2 = min(w, x2)
		transform["top_strip_region"] = (x1, y1, x2, y2)
		if y2 > y1 and x2 > x1:
			top_strip = binary_inv[y1:y2, x1:x2]
			top_marks_local = _detect_marks_in_strip(top_strip, axis="x")
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
			x_scale, x_offset, top_conf = _estimate_axis_transform(
				top_centers, exp_start, exp_end, exp_count)
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
def draw_timing_mark_debug(image: numpy.ndarray,
	transform: dict) -> numpy.ndarray:
	"""Draw timing mark bounding boxes and search strips on a debug image.

	Draws rectangles around each detected timing mark and semi-transparent
	overlays showing the search strip regions. Intended to be called
	independently from draw_answer_debug for modular debugging.

	Args:
		image: BGR image to annotate (will be copied)
		transform: dict from estimate_anchor_transform with mark positions

	Returns:
		annotated copy of the image
	"""
	debug = image.copy()
	overlay = debug.copy()
	h, w = debug.shape[:2]
	# colors (BGR)
	cyan = (255, 255, 0)
	magenta = (255, 0, 255)
	strip_color_left = (200, 200, 0)
	strip_color_top = (200, 0, 200)
	strip_alpha = 0.15
	# draw semi-transparent search strip regions
	lx1, ly1, lx2, ly2 = transform.get("left_strip_region", (0, 0, 0, 0))
	if lx2 > lx1 and ly2 > ly1:
		cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), strip_color_left, -1)
	tx1, ty1, tx2, ty2 = transform.get("top_strip_region", (0, 0, 0, 0))
	if tx2 > tx1 and ty2 > ty1:
		cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), strip_color_top, -1)
	# blend strip overlays onto debug image
	cv2.addWeighted(overlay, strip_alpha, debug, 1.0 - strip_alpha, 0, debug)
	# draw left timing mark bounding boxes (cyan)
	left_marks = transform.get("left_marks", [])
	for idx, mark in enumerate(left_marks):
		bx, by, bw, bh = mark["bbox"]
		cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), cyan, 1)
		# draw index label to the right of the mark
		label_x = bx + bw + 2
		label_y = by + bh
		cv2.putText(debug, str(idx), (label_x, label_y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.25, cyan, 1)
	# draw top timing mark bounding boxes (magenta)
	top_marks = transform.get("top_marks", [])
	for idx, mark in enumerate(top_marks):
		bx, by, bw, bh = mark["bbox"]
		cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), magenta, 1)
		# draw index label below the mark
		label_x = bx
		label_y = by + bh + 10
		cv2.putText(debug, str(idx), (label_x, label_y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.25, magenta, 1)
	return debug
