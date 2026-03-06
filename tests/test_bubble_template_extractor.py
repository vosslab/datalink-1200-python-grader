"""Tests for omr_utils.bubble_template_extractor alignment and symmetry,
and omr_utils.slot_map.SlotMap geometry derivation."""

# PIP3 modules
import numpy

# local repo modules
import omr_utils.bubble_reader
import omr_utils.bubble_template_extractor
import omr_utils.slot_map


#============================================
def _make_bracket_roi(width: int = 60, height: int = 20,
	bracket_dark: int = 40, interior: int = 200) -> numpy.ndarray:
	"""Create a synthetic bracket-pair ROI.

	Args:
		width: ROI width
		height: ROI height
		bracket_dark: darkness of bracket edges
		interior: brightness of interior

	Returns:
		grayscale numpy array
	"""
	roi = numpy.full((height, width), interior, dtype=numpy.uint8)
	# top bracket edge
	roi[0:2, :] = bracket_dark
	# bottom bracket edge
	roi[-2:, :] = bracket_dark
	return roi


#============================================
def test_extract_roi_1x_returns_native_size() -> None:
	"""extract_roi_1x returns a crop at native resolution."""
	gray = numpy.full((200, 400), 180, dtype=numpy.uint8)
	geom = {"half_width": 30.0, "half_height": 5.5}
	roi = omr_utils.bubble_template_extractor.extract_roi_1x(
		gray, 200, 100, geom)
	assert roi is not None
	# should be approximately (2*(5+pad_y)) x (2*(30+pad_x))
	assert roi.shape[0] > 10
	assert roi.shape[1] > 60


#============================================
def test_extract_roi_1x_out_of_bounds() -> None:
	"""extract_roi_1x returns None when ROI extends beyond image."""
	gray = numpy.full((50, 50), 180, dtype=numpy.uint8)
	geom = {"half_width": 30.0, "half_height": 5.5}
	roi = omr_utils.bubble_template_extractor.extract_roi_1x(
		gray, 5, 5, geom)
	assert roi is None


#============================================
def test_find_medoid_roi_single() -> None:
	"""Medoid of a single ROI is index 0."""
	rois = [_make_bracket_roi()]
	idx = omr_utils.bubble_template_extractor._find_medoid_roi(rois)
	assert idx == 0


#============================================
def test_find_medoid_roi_identical() -> None:
	"""Medoid of identical ROIs returns a valid index."""
	roi = _make_bracket_roi()
	rois = [roi.copy() for _ in range(5)]
	idx = omr_utils.bubble_template_extractor._find_medoid_roi(rois)
	assert 0 <= idx < 5


#============================================
def test_find_medoid_roi_selects_best_match() -> None:
	"""Medoid should be the ROI most similar to others."""
	# create 4 similar ROIs and 1 outlier
	normal = _make_bracket_roi(interior=200)
	outlier = _make_bracket_roi(interior=50)
	rois = [normal.copy(), normal.copy(), normal.copy(),
		normal.copy(), outlier]
	idx = omr_utils.bubble_template_extractor._find_medoid_roi(rois)
	# medoid should not be the outlier (index 4)
	assert idx != 4


#============================================
def test_align_roi_to_reference_no_shift() -> None:
	"""Aligning ROI to a center crop of itself should give near-zero shift."""
	# create a more distinctive pattern with gradient for unique features
	roi = numpy.full((40, 100), 200, dtype=numpy.uint8)
	# dark bracket edges
	roi[5:7, 15:85] = 40
	roi[33:35, 15:85] = 40
	# add a unique dark spot to break symmetry
	roi[18:22, 45:55] = 60
	# reference: center crop of the ROI
	ref = roi[5:35, 15:85]
	aligned, dx, dy, score = (
		omr_utils.bubble_template_extractor._align_roi_to_reference(
			roi, ref))
	assert score > 0.3
	assert abs(dx) <= 3
	assert abs(dy) <= 3


#============================================
def test_align_roi_to_reference_detects_shift() -> None:
	"""Aligning a shifted ROI should detect the translation offset."""
	# create a large canvas with a bracket pattern
	canvas = numpy.full((40, 100), 200, dtype=numpy.uint8)
	canvas[5:7, 10:90] = 40
	canvas[33:35, 10:90] = 40
	# reference: center crop
	ref = canvas[5:35, 20:80]
	# shifted ROI: same pattern but shifted right by 5px
	shifted_canvas = numpy.full((40, 100), 200, dtype=numpy.uint8)
	shifted_canvas[5:7, 15:95] = 40
	shifted_canvas[33:35, 15:95] = 40
	aligned, dx, dy, score = (
		omr_utils.bubble_template_extractor._align_roi_to_reference(
			shifted_canvas, ref))
	assert score > 0.3


#============================================
def test_symmetry_augmentation_lr() -> None:
	"""Left-right symmetry doubles count and flips correctly."""
	roi = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.uint8)
	augmented = omr_utils.bubble_template_extractor._apply_symmetry_augmentation(
		[roi], "A")
	assert len(augmented) == 2
	# second should be left-right flipped
	expected_flip = numpy.fliplr(roi)
	numpy.testing.assert_array_equal(augmented[1], expected_flip)


#============================================
def test_symmetry_augmentation_tb() -> None:
	"""Top-bottom symmetry (B-E) doubles count and flips correctly."""
	roi = numpy.array([[1, 2], [3, 4], [5, 6]], dtype=numpy.uint8)
	augmented = omr_utils.bubble_template_extractor._apply_symmetry_augmentation(
		[roi], "B")
	assert len(augmented) == 2
	expected_flip = numpy.flipud(roi)
	numpy.testing.assert_array_equal(augmented[1], expected_flip)


#============================================
def test_symmetry_augmentation_preserves_originals() -> None:
	"""Symmetry augmentation keeps original ROIs unchanged."""
	roi1 = _make_bracket_roi()
	roi2 = _make_bracket_roi(interior=180)
	originals = [roi1.copy(), roi2.copy()]
	augmented = omr_utils.bubble_template_extractor._apply_symmetry_augmentation(
		originals, "C")
	assert len(augmented) == 4
	# first two should be unchanged
	numpy.testing.assert_array_equal(augmented[0], roi1)
	numpy.testing.assert_array_equal(augmented[1], roi2)


#============================================
def test_build_letter_template_produces_template() -> None:
	"""_build_letter_template returns a valid template at canonical resolution."""
	# create 10 similar bracket ROIs
	rois = [_make_bracket_roi(width=60, height=20) for _ in range(10)]
	template, mask, table = (
		omr_utils.bubble_template_extractor._build_letter_template(
			rois, "A", reject_threshold=0.3))
	assert template is not None
	assert mask is not None
	assert template.dtype == numpy.uint8
	assert mask.dtype == numpy.uint8
	assert template.shape == mask.shape
	assert len(table) > 0
	# template should be at canonical high resolution
	canon_w = omr_utils.bubble_template_extractor.CANONICAL_TEMPLATE_WIDTH
	canon_h = omr_utils.bubble_template_extractor.CANONICAL_TEMPLATE_HEIGHT
	assert template.shape == (canon_h, canon_w)


#============================================
def test_build_letter_template_too_few_rois() -> None:
	"""_build_letter_template returns None with fewer than 3 ROIs."""
	rois = [_make_bracket_roi(), _make_bracket_roi()]
	template, mask, table = (
		omr_utils.bubble_template_extractor._build_letter_template(
			rois, "B"))
	assert template is None
	assert mask is None


#============================================
def test_generate_template_mask_shape() -> None:
	"""_generate_template_mask returns same shape as input."""
	template = _make_bracket_roi(width=40, height=15)
	mask = omr_utils.bubble_template_extractor._generate_template_mask(
		template)
	assert mask.shape == template.shape
	assert mask.dtype == numpy.uint8


#============================================
def test_generate_template_mask_dark_regions_high() -> None:
	"""Mask should have high values where template has dark bracket edges."""
	template = _make_bracket_roi(width=40, height=15,
		bracket_dark=30, interior=220)
	mask = omr_utils.bubble_template_extractor._generate_template_mask(
		template)
	# top edge region should have high mask values
	top_mask_mean = float(numpy.mean(mask[0:2, :]))
	# interior should have lower (or zero) mask values
	mid_mask_mean = float(numpy.mean(mask[6:9, :]))
	assert top_mask_mean > mid_mask_mean


#============================================
def test_score_patch_quality_good_bubble() -> None:
	"""A clean bracket patch should score positive."""
	patch = _make_bracket_roi(bracket_dark=40, interior=210)
	score = omr_utils.bubble_template_extractor._score_patch_quality(patch)
	assert score > 0


#============================================
def test_score_patch_quality_uniform() -> None:
	"""A uniform patch should score near zero."""
	patch = numpy.full((20, 60), 150, dtype=numpy.uint8)
	score = omr_utils.bubble_template_extractor._score_patch_quality(patch)
	assert abs(score) < 5.0


#============================================
def test_load_templates_and_masks_empty_dir(tmp_path) -> None:
	"""Loading from empty directory returns empty dicts."""
	templates, masks = (
		omr_utils.bubble_template_extractor.load_templates_and_masks(
			str(tmp_path)))
	assert templates == {}
	assert masks == {}


#============================================
def test_load_templates_and_masks_nonexistent(tmp_path) -> None:
	"""Loading from nonexistent directory returns empty dicts."""
	nonexistent = str(tmp_path / "nonexistent_dir_12345")
	templates, masks = (
		omr_utils.bubble_template_extractor.load_templates_and_masks(
			nonexistent))
	assert templates == {}
	assert masks == {}


#============================================
def _make_mock_transform(row_pitch: float = 46.8,
	col_spacing: float = 45.1) -> dict:
	"""Build a minimal transform dict for SlotMap construction.

	Args:
		row_pitch: left_s_q spacing
		col_spacing: top_col_spacing fine step

	Returns:
		transform dict suitable for SlotMap()
	"""
	# build 50 question marks with uniform spacing starting at y=200
	question_marks = []
	for i in range(50):
		cy = 200.0 + i * row_pitch
		question_marks.append({"center_y": cy})
	transform = {
		"top_fp_x0": 100.0,
		"top_col_spacing": col_spacing,
		"left_s_q": row_pitch,
		"left_question_marks": question_marks,
	}
	return transform


#============================================
def _make_mock_template() -> dict:
	"""Build a minimal template dict for SlotMap construction."""
	template = {
		"answers": {
			"num_questions": 100,
			"choices": ["A", "B", "C", "D", "E"],
			"left_column": {
				"question_range": [1, 50],
				"choice_columns": {"A": 5, "B": 8, "C": 10, "D": 13, "E": 16},
			},
			"right_column": {
				"question_range": [51, 100],
				"choice_columns": {"A": 24, "B": 27, "C": 30, "D": 33, "E": 36},
			},
		},
	}
	return template


#============================================
def test_slot_map_geom_with_valid_transform() -> None:
	"""SlotMap.geom() computes geometry from anchor spacing values."""
	transform = _make_mock_transform(46.8, 45.1)
	template = _make_mock_template()
	sm = omr_utils.slot_map.SlotMap(transform, template)
	geom = sm.geom()
	# should derive values from spacing, not use fixed defaults
	assert geom["row_pitch"] == 46.8
	assert geom["col_pitch"] == 45.1
	# half_height should be a fraction of row_pitch
	assert 4.0 < geom["half_height"] < 8.0
	# half_width should be a fraction of col_pitch
	assert 25.0 < geom["half_width"] < 35.0


#============================================
def test_slot_map_raises_on_zero_spacing() -> None:
	"""SlotMap raises ValueError when spacing is zero."""
	transform = _make_mock_transform(0.0, 0.0)
	# override fp_x0 to nonzero so only spacing check fires
	transform["top_fp_x0"] = 100.0
	template = _make_mock_template()
	try:
		omr_utils.slot_map.SlotMap(transform, template)
		assert False, "expected ValueError"
	except ValueError:
		pass


#============================================
def test_slot_map_raises_on_missing_marks() -> None:
	"""SlotMap raises ValueError when question marks are missing."""
	transform = {
		"top_fp_x0": 100.0,
		"top_col_spacing": 45.1,
		"left_s_q": 46.8,
		"left_question_marks": [],
	}
	template = _make_mock_template()
	try:
		omr_utils.slot_map.SlotMap(transform, template)
		assert False, "expected ValueError"
	except ValueError:
		pass


#============================================
def test_slot_map_geom_scales_with_resolution() -> None:
	"""Doubling the spacing should approximately double the geometry."""
	transform_1x = _make_mock_transform(46.8, 45.1)
	transform_2x = _make_mock_transform(93.6, 90.2)
	template = _make_mock_template()
	sm_1x = omr_utils.slot_map.SlotMap(transform_1x, template)
	sm_2x = omr_utils.slot_map.SlotMap(transform_2x, template)
	geom_1x = sm_1x.geom()
	geom_2x = sm_2x.geom()
	# half_height should scale proportionally
	ratio_h = geom_2x["half_height"] / geom_1x["half_height"]
	assert 1.95 < ratio_h < 2.05
	# half_width should scale proportionally
	ratio_w = geom_2x["half_width"] / geom_1x["half_width"]
	assert 1.95 < ratio_w < 2.05


#============================================
def test_slot_map_geom_valid_includes_all_keys() -> None:
	"""SlotMap.geom() with valid spacing should include all expected keys."""
	transform = _make_mock_transform(46.8, 45.1)
	template = _make_mock_template()
	sm = omr_utils.slot_map.SlotMap(transform, template)
	geom = sm.geom()
	expected_keys = [
		"half_width", "half_height", "center_exclusion",
		"bracket_edge_height", "measurement_inset_v",
		"measurement_inset_h", "refine_max_shift",
		"refine_pad_v", "refine_pad_h", "row_pitch", "col_pitch",
	]
	for key in expected_keys:
		assert key in geom
