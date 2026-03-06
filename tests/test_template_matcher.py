"""Tests for omr_utils.template_matcher masked NCC and subpixel refinement."""

# PIP3 modules
import numpy

# local repo modules
import omr_utils.template_matcher


#============================================
def _make_synthetic_bubble(size: int = 64, center_offset: tuple = (0, 0),
	brightness: int = 200, bracket_dark: int = 50) -> numpy.ndarray:
	"""Create a synthetic bubble image with dark bracket edges.

	Args:
		size: image height and width
		center_offset: (dx, dy) shift of the bubble center from image center
		brightness: interior brightness value
		bracket_dark: bracket edge darkness value

	Returns:
		grayscale numpy array
	"""
	img = numpy.full((size, size), brightness, dtype=numpy.uint8)
	cy = size // 2 + center_offset[1]
	cx = size // 2 + center_offset[0]
	# draw dark bracket edges (top and bottom strips)
	half_w = 20
	half_h = 5
	# top bracket
	img[cy - half_h:cy - half_h + 2, cx - half_w:cx + half_w] = bracket_dark
	# bottom bracket
	img[cy + half_h - 2:cy + half_h, cx - half_w:cx + half_w] = bracket_dark
	return img


#============================================
def _make_template_and_mask(tw: int = 42, th: int = 12) -> tuple:
	"""Create a small synthetic template and mask.

	Returns:
		tuple of (template, mask) as numpy arrays
	"""
	# template: dark edges, bright interior
	template = numpy.full((th, tw), 200, dtype=numpy.uint8)
	template[0:2, :] = 50
	template[-2:, :] = 50
	# mask: high weight on bracket edges
	mask = numpy.zeros((th, tw), dtype=numpy.uint8)
	mask[0:2, :] = 255
	mask[-2:, :] = 255
	return (template, mask)


#============================================
def test_subpixel_peak_at_center() -> None:
	"""Subpixel peak at center of a symmetric result map returns center."""
	# 5x5 result map with peak at (2, 2)
	result = numpy.zeros((5, 5), dtype=numpy.float32)
	result[2, 2] = 1.0
	result[2, 1] = 0.5
	result[2, 3] = 0.5
	result[1, 2] = 0.5
	result[3, 2] = 0.5
	sub_x, sub_y = omr_utils.template_matcher._subpixel_peak(result, 2, 2)
	# symmetric neighborhood: subpixel offset should be ~0
	assert abs(sub_x - 2.0) < 0.01
	assert abs(sub_y - 2.0) < 0.01


#============================================
def test_subpixel_peak_offset_right() -> None:
	"""Subpixel peak should shift right when right neighbor is brighter."""
	result = numpy.zeros((5, 5), dtype=numpy.float32)
	result[2, 2] = 1.0
	result[2, 1] = 0.3
	result[2, 3] = 0.7
	result[1, 2] = 0.5
	result[3, 2] = 0.5
	sub_x, sub_y = omr_utils.template_matcher._subpixel_peak(result, 2, 2)
	# peak should be shifted right (sub_x > 2.0)
	assert sub_x > 2.0


#============================================
def test_subpixel_peak_edge_case() -> None:
	"""Subpixel peak at edge of result map falls back to integer position."""
	result = numpy.zeros((3, 3), dtype=numpy.float32)
	result[0, 0] = 1.0
	sub_x, sub_y = omr_utils.template_matcher._subpixel_peak(result, 0, 0)
	# at edge, cannot fit quadratic: should return integer position
	assert sub_x == 0.0
	assert sub_y == 0.0


#============================================
def test_match_bubble_local_on_synthetic() -> None:
	"""Local NCC match runs and returns a valid 3-tuple on synthetic data."""
	# create a larger image with a bubble offset from center
	img = _make_synthetic_bubble(size=100, center_offset=(3, -2))
	template, _ = _make_template_and_mask()
	# approximate center at image center
	rcx, rcy, conf = omr_utils.template_matcher.match_bubble_local(
		img, template, 50, 50, search_radius=10)
	# should find a match with some confidence
	assert conf > 0.0
	# refined position should be within search radius of approximate
	assert abs(rcx - 50) <= 12
	assert abs(rcy - 50) <= 12


#============================================
def test_match_bubble_masked_returns_five_values() -> None:
	"""Masked match returns a 5-tuple (cx, cy, confidence, dx, dy)."""
	img = _make_synthetic_bubble(size=100)
	template, mask = _make_template_and_mask()
	result = omr_utils.template_matcher.match_bubble_masked(
		img, template, mask, 50, 50, search_radius=10)
	assert len(result) == 5
	# unpack to verify types
	rcx, rcy, conf, dx, dy = result
	assert isinstance(conf, float)


#============================================
def test_match_bubble_masked_out_of_bounds() -> None:
	"""Masked match at image edge returns default with zero confidence."""
	img = numpy.full((20, 20), 200, dtype=numpy.uint8)
	template, mask = _make_template_and_mask()
	rcx, rcy, conf, dx, dy = omr_utils.template_matcher.match_bubble_masked(
		img, template, mask, 5, 5, search_radius=10)
	assert conf == 0.0
	assert rcx == 5
	assert rcy == 5


#============================================
def test_refine_row_by_template_with_masks() -> None:
	"""refine_row_by_template accepts and uses masks parameter."""
	# create synthetic image and templates/masks
	img = _make_synthetic_bubble(size=200)
	template_img, mask_img = _make_template_and_mask()
	# build 5X oversize template (scale_template_to_bubble expects this)
	templates = {"A": template_img}
	masks = {"A": mask_img}
	geom = {"half_width": 20.0, "half_height": 5.0}
	row_positions = {"A": (100, 100)}
	choices = ["A"]
	# should not raise when masks are provided
	refined = omr_utils.template_matcher.refine_row_by_template(
		img, templates, row_positions, geom, choices,
		search_radius=10, masks=masks)
	assert "A" in refined
	# result is a 3-tuple (cx, cy, confidence)
	assert len(refined["A"]) == 3


#============================================
def test_try_load_bubble_templates_returns_tuple() -> None:
	"""try_load_bubble_templates returns a (templates, masks) tuple."""
	result = omr_utils.template_matcher.try_load_bubble_templates()
	assert isinstance(result, tuple)
	assert len(result) == 2
	templates, masks = result
	assert isinstance(templates, dict)
	assert isinstance(masks, dict)
