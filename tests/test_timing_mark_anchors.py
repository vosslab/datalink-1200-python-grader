"""Tests for omr_utils.timing_mark_anchors."""

# PIP3 modules
import numpy

# local repo modules
import omr_utils.timing_mark_anchors


#============================================
def _make_template(left_count: int = 30, top_count: int = 28) -> dict:
	"""Create a minimal template with timing-mark metadata."""
	return {
		"timing_marks": {
			"left_edge": {
				"x": 0.05,
				"start_y": 0.10,
				"end_y": 0.90,
				"expected_count": left_count,
			},
			"top_edge": {
				"y": 0.06,
				"start_x": 0.08,
				"end_x": 0.92,
				"expected_count": top_count,
			},
		}
	}


#============================================
def test_identity_transform_on_blank_image() -> None:
	"""Blank image has no timing marks, so transform stays identity."""
	gray = numpy.full((300, 400), 255, dtype=numpy.uint8)
	template = _make_template()
	result = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	assert result["x_scale"] == 1.0
	assert result["y_scale"] == 1.0
	assert result["x_offset"] == 0.0
	assert result["y_offset"] == 0.0
	assert result["left_confidence"] == 0.0
	assert result["top_confidence"] == 0.0


#============================================
def test_recovers_left_and_top_axis_transform() -> None:
	"""Detected timing marks recover scale and offset on both axes.

	Uses a larger synthetic image (1000x800) so top blocks have
	realistic relative size for row-pattern detection.
	"""
	h = 800
	w = 1000
	gray = numpy.full((h, w), 255, dtype=numpy.uint8)
	template = _make_template(left_count=34, top_count=7)
	left = template["timing_marks"]["left_edge"]
	top = template["timing_marks"]["top_edge"]
	# synthetic residual distortion to recover
	y_scale = 1.04
	y_offset = 5.0
	x_scale = 0.97
	x_offset = -4.0
	# draw left dashes (wider than tall, in left 10% of image)
	left_x = int(round(left["x"] * w))
	exp_left = numpy.linspace(
		left["start_y"] * h, left["end_y"] * h, left["expected_count"])
	for exp_y in exp_left:
		y = int(round(exp_y * y_scale + y_offset))
		y1 = max(0, y - 2)
		y2 = min(h, y + 3)
		x1 = max(0, left_x - 8)
		x2 = min(w, left_x + 8)
		gray[y1:y2, x1:x2] = 0
	# draw 7 large top blocks in top 10% strip
	# each block ~2% of width, ~30% of strip height
	strip_h = int(h * 0.10)
	block_w = int(w * 0.02)
	block_h = int(strip_h * 0.30)
	exp_top = numpy.linspace(
		top["start_x"] * w, top["end_x"] * w, 7)
	# blocks centered at ~40% of strip height
	block_cy = int(strip_h * 0.40)
	for exp_x in exp_top:
		x = int(round(exp_x * x_scale + x_offset))
		x1 = max(0, x - block_w // 2)
		x2 = min(w, x + block_w // 2)
		y1 = max(0, block_cy - block_h // 2)
		y2 = min(strip_h, block_cy + block_h // 2)
		gray[y1:y2, x1:x2] = 0
	result = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	assert result["left_confidence"] > 0.35
	assert result["top_confidence"] > 0.35
	assert abs(result["y_scale"] - y_scale) < 0.10
	assert abs(result["x_scale"] - x_scale) < 0.10
	assert abs(result["y_offset"] - y_offset) < 12.0
	assert abs(result["x_offset"] - x_offset) < 16.0


#============================================
def test_low_mark_count_keeps_identity_on_axis() -> None:
	"""Too few marks should not override an axis transform."""
	h = 300
	w = 400
	gray = numpy.full((h, w), 255, dtype=numpy.uint8)
	template = _make_template(left_count=40, top_count=40)
	left = template["timing_marks"]["left_edge"]
	left_x = int(round(left["x"] * w))
	# add only three left marks (below minimum confidence threshold)
	for y in [40, 120, 220]:
		gray[y - 1:y + 2, left_x - 4:left_x + 4] = 0
	result = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	assert result["y_scale"] == 1.0
	assert result["y_offset"] == 0.0
	assert result["left_confidence"] == 0.0


#============================================
def test_axis_fit_is_stable_when_edge_marks_are_missing() -> None:
	"""Missing first/last marks should not collapse scale estimates."""
	expected_count = 60
	expected_start = 100.0
	expected_end = 1900.0
	expected = numpy.linspace(expected_start, expected_end, expected_count)
	# Simulate realistic detection: first/last marks missing, small jitter.
	observed = []
	for i, pos in enumerate(expected[1:-1]):
		jitter = 0.2 if i % 2 == 0 else -0.2
		observed.append(float(pos + 4.0 + jitter))
	scale, offset, confidence = omr_utils.timing_mark_anchors._estimate_axis_transform(
		observed, expected_start, expected_end, expected_count)
	assert 0.99 <= scale <= 1.01
	assert 2.0 <= offset <= 6.0
	assert confidence > 0.75


#============================================
def test_blank_image_returns_empty_marks() -> None:
	"""Blank image should return empty mark lists and zero-area strip regions."""
	gray = numpy.full((300, 400), 255, dtype=numpy.uint8)
	template = _make_template()
	result = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	assert result["left_marks"] == []
	assert result["top_marks"] == []
	# strip regions should still be defined (non-zero search areas)
	lx1, ly1, lx2, ly2 = result["left_strip_region"]
	assert lx2 > lx1
	assert ly2 > ly1


#============================================
def test_mark_positions_returned_with_bboxes() -> None:
	"""Transform result includes mark positions and valid bounding boxes.

	Uses larger synthetic image with 7 large blocks in top 10% strip
	to match row-pattern detection expectations.
	"""
	h = 800
	w = 1000
	gray = numpy.full((h, w), 255, dtype=numpy.uint8)
	template = _make_template(left_count=34, top_count=7)
	left = template["timing_marks"]["left_edge"]
	top = template["timing_marks"]["top_edge"]
	# draw left dashes at expected positions
	left_x = int(round(left["x"] * w))
	exp_left = numpy.linspace(
		left["start_y"] * h, left["end_y"] * h, left["expected_count"])
	for exp_y in exp_left:
		y = int(round(exp_y))
		y1 = max(0, y - 2)
		y2 = min(h, y + 3)
		x1 = max(0, left_x - 8)
		x2 = min(w, left_x + 8)
		gray[y1:y2, x1:x2] = 0
	# draw 7 large top blocks in top 10% strip
	strip_h = int(h * 0.10)
	block_w = int(w * 0.02)
	block_h = int(strip_h * 0.30)
	exp_top = numpy.linspace(
		top["start_x"] * w, top["end_x"] * w, 7)
	block_cy = int(strip_h * 0.40)
	for exp_x in exp_top:
		x = int(round(exp_x))
		x1 = max(0, x - block_w // 2)
		x2 = min(w, x + block_w // 2)
		y1 = max(0, block_cy - block_h // 2)
		y2 = min(strip_h, block_cy + block_h // 2)
		gray[y1:y2, x1:x2] = 0
	result = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	# verify left marks returned
	left_marks = result["left_marks"]
	assert len(left_marks) >= 20
	for mark in left_marks:
		assert "center_y" in mark
		assert "bbox" in mark
		bx, by, bw, bh = mark["bbox"]
		# bounding box must have positive dimensions
		assert bw > 0
		assert bh > 0
		# center must be within image bounds
		assert 0 <= mark["center_y"] < h
	# verify top marks returned (at least 5 of 7 expected)
	top_marks = result["top_marks"]
	assert len(top_marks) >= 5
	for mark in top_marks:
		assert "center_x" in mark
		assert "bbox" in mark
		bx, by, bw, bh = mark["bbox"]
		assert bw > 0
		assert bh > 0
		assert 0 <= mark["center_x"] < w
	# verify strip regions are valid
	lx1, ly1, lx2, ly2 = result["left_strip_region"]
	assert lx2 > lx1
	assert ly2 > ly1
	tx1, ty1, tx2, ty2 = result["top_strip_region"]
	assert tx2 > tx1
	assert ty2 > ty1


#============================================
def test_mark_bboxes_in_image_coordinates() -> None:
	"""Bounding boxes should be in image coordinates, not strip-local."""
	h = 400
	w = 500
	gray = numpy.full((h, w), 255, dtype=numpy.uint8)
	template = _make_template(left_count=20, top_count=20)
	left = template["timing_marks"]["left_edge"]
	left_x = int(round(left["x"] * w))
	# draw a single visible left dash near y=200
	target_y = 200
	gray[target_y - 2:target_y + 3, left_x - 5:left_x + 5] = 0
	# draw enough marks to pass the minimum threshold
	exp_left = numpy.linspace(
		left["start_y"] * h, left["end_y"] * h, 20)
	for exp_y in exp_left:
		y = int(round(exp_y))
		y1 = max(0, y - 1)
		y2 = min(h, y + 2)
		x1 = max(0, left_x - 4)
		x2 = min(w, left_x + 4)
		gray[y1:y2, x1:x2] = 0
	result = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	left_marks = result["left_marks"]
	assert len(left_marks) > 0
	# all bbox y-coordinates should be in image space (not strip-local)
	strip_y1 = result["left_strip_region"][1]
	for mark in left_marks:
		_, by, _, _ = mark["bbox"]
		# bbox y should be >= strip_y1 (offset from top of image)
		assert by >= strip_y1, (
			f"bbox y={by} should be >= strip_y1={strip_y1}")


#============================================
class TestMarkIndexConversion:
	"""Tests for mark_index_to_normalized and normalized_to_mark_index."""

	def test_integer_index_matches_linspace(self) -> None:
		"""Integer mark index matches numpy.linspace result."""
		start = 0.067
		end = 0.91
		count = 60
		# mark index 0 should be at start
		result = omr_utils.timing_mark_anchors.mark_index_to_normalized(
			0.0, start, end, count)
		assert abs(result - start) < 1e-10
		# mark index count-1 should be at end
		result = omr_utils.timing_mark_anchors.mark_index_to_normalized(
			float(count - 1), start, end, count)
		assert abs(result - end) < 1e-10
		# mark index 30 should be at midpoint of 60-mark span
		mid_idx = (count - 1) / 2.0
		result = omr_utils.timing_mark_anchors.mark_index_to_normalized(
			mid_idx, start, end, count)
		expected_mid = (start + end) / 2.0
		assert abs(result - expected_mid) < 1e-10

	def test_roundtrip_preserves_value(self) -> None:
		"""Converting norm to index and back preserves the value."""
		start = 0.04
		end = 0.96
		count = 53
		original = 0.2164
		idx = omr_utils.timing_mark_anchors.normalized_to_mark_index(
			original, start, end, count)
		recovered = omr_utils.timing_mark_anchors.mark_index_to_normalized(
			idx, start, end, count)
		assert abs(recovered - original) < 1e-10

	def test_fractional_index(self) -> None:
		"""Fractional index interpolates correctly between marks."""
		start = 0.0
		end = 1.0
		count = 11
		# step = 0.1, index 5.5 should be at 0.55
		result = omr_utils.timing_mark_anchors.mark_index_to_normalized(
			5.5, start, end, count)
		assert abs(result - 0.55) < 1e-10

	def test_normalized_to_mark_index_inverse(self) -> None:
		"""normalized_to_mark_index is the exact inverse of mark_index_to_normalized."""
		start = 0.067
		end = 0.91
		count = 60
		# test a range of mark indices
		for idx in [0.0, 5.5, 10.456, 29.5, 59.0]:
			norm = omr_utils.timing_mark_anchors.mark_index_to_normalized(
				idx, start, end, count)
			back = omr_utils.timing_mark_anchors.normalized_to_mark_index(
				norm, start, end, count)
			assert abs(back - idx) < 1e-10
