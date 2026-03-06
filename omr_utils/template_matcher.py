"""Local NCC-based bubble refinement using pixel templates.

Uses normalized cross-correlation (cv2.matchTemplate with TM_CCOEFF_NORMED)
to refine approximate bubble positions within a small search window. This
replaces reliance on hardcoded YAML coordinates with a template-first
localization approach.
"""

# Standard Library
import os

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.template_loader
import omr_utils.bubble_template_extractor


#============================================
def match_bubble_local(gray: numpy.ndarray, template_img: numpy.ndarray,
	approx_cx: int, approx_cy: int,
	search_radius: int = DEFAULT_SEARCH_RADIUS) -> tuple:
	"""Find exact bubble center using local NCC template matching.

	Extracts a search region around the approximate position, scales
	the reference template to match, and runs normalized cross-correlation
	to find the peak.

	Args:
		gray: grayscale image
		template_img: scaled template image (same pixel size as expected
			bubble in the image)
		approx_cx: approximate bubble center x from first pass
		approx_cy: approximate bubble center y from first pass
		search_radius: pixels to search beyond the template extent

	Returns:
		tuple of (refined_cx, refined_cy, confidence) where confidence
		is the NCC peak value (0.0 to 1.0). Returns the original
		position with confidence=0 if matching fails.
	"""
	h, w = gray.shape
	th, tw = template_img.shape
	# compute search region (template size + search radius on each side)
	half_tw = tw // 2
	half_th = th // 2
	roi_x1 = approx_cx - half_tw - search_radius
	roi_y1 = approx_cy - half_th - search_radius
	roi_x2 = approx_cx + half_tw + search_radius
	roi_y2 = approx_cy + half_th + search_radius
	# bounds check: ROI must fit within image
	if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > w or roi_y2 > h:
		return (approx_cx, approx_cy, 0.0)
	# ROI must be larger than template in both dimensions
	roi_w = roi_x2 - roi_x1
	roi_h = roi_y2 - roi_y1
	if roi_w <= tw or roi_h <= th:
		return (approx_cx, approx_cy, 0.0)
	# extract search region
	roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
	# run normalized cross-correlation
	result = cv2.matchTemplate(roi, template_img, cv2.TM_CCOEFF_NORMED)
	# find peak location
	_, max_val, _, max_loc = cv2.minMaxLoc(result)
	confidence = float(max_val)
	# convert peak location back to image coordinates
	# max_loc is (x, y) of the top-left corner of the best match
	# bubble center = match top-left + template half dimensions
	refined_cx = roi_x1 + max_loc[0] + half_tw
	refined_cy = roi_y1 + max_loc[1] + half_th
	return (refined_cx, refined_cy, confidence)


#============================================
def refine_row_by_template(gray: numpy.ndarray, templates: dict,
	row_positions: dict, geom: dict, choices: list,
	search_radius: int = DEFAULT_SEARCH_RADIUS) -> dict:
	"""Refine all 5 choice positions in a question row using templates.

	For each choice, uses the corresponding letter's pixel template to
	refine the position via NCC. Only updates positions where the match
	confidence exceeds the minimum threshold.

	Args:
		gray: grayscale image
		templates: dict mapping letter to 5X oversize template array
		row_positions: dict mapping choice letter to (cx, cy) approx position
		geom: pixel geometry dict for scaling templates
		choices: list of choice letters ["A", "B", "C", "D", "E"]
		search_radius: search radius in pixels

	Returns:
		dict mapping choice letter to (refined_cx, refined_cy, confidence).
		Choices without templates or below confidence threshold keep
		their original positions with confidence=0.
	"""
	refined = {}
	for choice in choices:
		if choice not in row_positions:
			continue
		cx, cy = row_positions[choice]
		# get the letter's template
		template_img = templates.get(choice)
		if template_img is None:
			refined[choice] = (cx, cy, 0.0)
			continue
		# scale 5X template to actual bubble size
		scaled = omr_utils.bubble_template_extractor.scale_template_to_bubble(
			template_img, geom)
		# run local NCC matching
		rcx, rcy, conf = match_bubble_local(
			gray, scaled, cx, cy, search_radius)
		if conf >= MIN_MATCH_CONFIDENCE:
			refined[choice] = (rcx, rcy, conf)
		else:
			# keep original position when confidence is low
			refined[choice] = (cx, cy, 0.0)
	return refined


#============================================
def try_load_bubble_templates() -> dict:
	"""Attempt to load bubble templates from the default config path.

	Returns:
		dict mapping letter to numpy array, or empty dict if templates
		are not available
	"""
	# find the repo root by looking for config/ relative to this module
	module_dir = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.dirname(module_dir)
	template_dir = os.path.join(repo_root, "config", "bubble_templates")
	return omr_utils.bubble_template_extractor.load_templates(template_dir)
