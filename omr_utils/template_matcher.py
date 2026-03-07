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
import omr_utils.bubble_template_extractor


#============================================
def match_bubble_local(gray: numpy.ndarray, template_img: numpy.ndarray,
	approx_cx: int, approx_cy: int,
	search_radius: int = 8) -> tuple:
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
	# extract search region and normalize contrast to match template
	roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
	roi = omr_utils.bubble_template_extractor.normalize_roi_percentile(roi)
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
def _subpixel_peak(result_map: numpy.ndarray,
	peak_x: int, peak_y: int) -> tuple:
	"""Refine NCC peak location to subpixel precision via quadratic fit.

	Fits a 1D quadratic to the 3-point neighborhood along each axis
	independently and solves for the fractional peak offset.

	Args:
		result_map: 2D NCC result array from cv2.matchTemplate
		peak_x: integer x of the peak
		peak_y: integer y of the peak

	Returns:
		tuple of (sub_x, sub_y) as floats with subpixel precision.
		Falls back to integer position if the neighborhood is invalid.
	"""
	rh, rw = result_map.shape
	sub_x = float(peak_x)
	sub_y = float(peak_y)
	# x-axis quadratic refinement
	if 1 <= peak_x < rw - 1:
		left = float(result_map[peak_y, peak_x - 1])
		center = float(result_map[peak_y, peak_x])
		right = float(result_map[peak_y, peak_x + 1])
		denom = 2.0 * (2.0 * center - left - right)
		if abs(denom) > 1e-8:
			dx = (right - left) / denom
			# clamp to [-0.5, 0.5] to avoid extrapolation
			dx = max(-0.5, min(0.5, dx))
			sub_x = peak_x + dx
	# y-axis quadratic refinement
	if 1 <= peak_y < rh - 1:
		top = float(result_map[peak_y - 1, peak_x])
		center = float(result_map[peak_y, peak_x])
		bottom = float(result_map[peak_y + 1, peak_x])
		denom = 2.0 * (2.0 * center - top - bottom)
		if abs(denom) > 1e-8:
			dy = (bottom - top) / denom
			dy = max(-0.5, min(0.5, dy))
			sub_y = peak_y + dy
	return (sub_x, sub_y)


#============================================
def match_bubble_masked(gray: numpy.ndarray, template_img: numpy.ndarray,
	mask_img: numpy.ndarray, approx_cx: int, approx_cy: int,
	search_radius: int = 8) -> tuple:
	"""Find exact bubble center using masked NCC template matching.

	Multiplies template and ROI regions by the mask before correlation,
	so only bracket structure contributes to the match score. Includes
	subpixel peak refinement.

	Args:
		gray: grayscale image
		template_img: scaled template (same pixel size as bubble)
		mask_img: mask array (same size as template, high values =
			important regions)
		approx_cx: approximate bubble center x
		approx_cy: approximate bubble center y
		search_radius: pixels to search beyond template extent

	Returns:
		tuple of (refined_cx, refined_cy, confidence, dx, dy) where
		dx/dy are the subpixel shift from approximate position.
		Returns (approx_cx, approx_cy, 0.0, 0, 0) if matching fails.
	"""
	h, w = gray.shape
	th, tw = template_img.shape
	# compute search region
	half_tw = tw // 2
	half_th = th // 2
	roi_x1 = approx_cx - half_tw - search_radius
	roi_y1 = approx_cy - half_th - search_radius
	roi_x2 = approx_cx + half_tw + search_radius
	roi_y2 = approx_cy + half_th + search_radius
	# bounds check
	if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > w or roi_y2 > h:
		return (approx_cx, approx_cy, 0.0, 0, 0)
	roi_w = roi_x2 - roi_x1
	roi_h = roi_y2 - roi_y1
	if roi_w <= tw or roi_h <= th:
		return (approx_cx, approx_cy, 0.0, 0, 0)
	# extract search region and normalize contrast to match template
	roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
	roi = omr_utils.bubble_template_extractor.normalize_roi_percentile(roi)
	# ensure mask matches template dimensions
	if mask_img.shape != template_img.shape:
		mask_img = cv2.resize(mask_img, (tw, th),
			interpolation=cv2.INTER_AREA)
	# apply mask to template (zero out unimportant regions)
	masked_template = cv2.multiply(
		template_img.astype(numpy.float32),
		(mask_img.astype(numpy.float32) / 255.0))
	masked_template = masked_template.astype(numpy.uint8)
	# run NCC with the masked template
	# use TM_CCORR_NORMED which is compatible with masked correlation
	result = cv2.matchTemplate(roi, masked_template,
		cv2.TM_CCORR_NORMED, mask=mask_img)
	# find integer peak
	_, max_val, _, max_loc = cv2.minMaxLoc(result)
	confidence = float(max_val)
	# subpixel refinement
	sub_x, sub_y = _subpixel_peak(result, max_loc[0], max_loc[1])
	# convert to image coordinates
	refined_cx = roi_x1 + sub_x + half_tw
	refined_cy = roi_y1 + sub_y + half_th
	# compute shift from approximate position
	dx = refined_cx - approx_cx
	dy = refined_cy - approx_cy
	return (refined_cx, refined_cy, confidence, dx, dy)


#============================================
def refine_row_by_template(gray: numpy.ndarray, templates: dict,
	row_positions: dict, choices: list,
	search_radius: int = 8, masks: dict = None,
	slot_dims: dict = None) -> dict:
	"""Refine all 5 choice positions in a question row using templates.

	For each choice, uses the corresponding letter's pixel template to
	refine the position via NCC. When masks are provided, uses masked
	NCC for bracket-focused matching with subpixel refinement. Only
	updates positions where the match confidence exceeds the threshold.

	Args:
		gray: grayscale image
		templates: dict mapping letter to 5X oversize template array
		row_positions: dict mapping choice letter to (cx, cy) approx position
		choices: list of choice letters ["A", "B", "C", "D", "E"]
		search_radius: search radius in pixels
		masks: optional dict mapping letter to mask array
		slot_dims: dict mapping choice to (width, height) from
			SlotMap.roi_bounds(); required for template scaling

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
		# get slot dimensions from lattice bounds
		if slot_dims is not None and choice in slot_dims:
			sw, sh = slot_dims[choice]
		else:
			raise ValueError(
				f"slot_dims missing for choice '{choice}': "
				"SlotMap.roi_bounds() must provide dimensions"
			)
		# scale canonical template to actual slot size
		scaled = omr_utils.bubble_template_extractor.scale_template_to_bubble(
			template_img, sw, sh)
		# use masked matching if mask is available
		mask_img = None
		if masks is not None:
			mask_img = masks.get(choice)
		if mask_img is not None:
			# scale mask to match template size
			scaled_mask = cv2.resize(mask_img,
				(scaled.shape[1], scaled.shape[0]),
				interpolation=cv2.INTER_AREA)
			rcx, rcy, conf, _, _ = match_bubble_masked(
				gray, scaled, scaled_mask, cx, cy, search_radius)
		else:
			# fall back to unmasked NCC
			rcx, rcy, conf = match_bubble_local(
				gray, scaled, cx, cy, search_radius)
		if conf >= 0.30:
			refined[choice] = (rcx, rcy, conf)
		else:
			# keep original position when confidence is low
			refined[choice] = (cx, cy, 0.0)
	return refined


#============================================
def _load_base_template(repo_root: str) -> tuple:
	"""Load the hand-made base template as bootstrap for all letters.

	The base template in artifacts/base_letter_template.png represents
	a single correct slot structure (left bracket + letter + right bracket).
	It is used as a fallback when per-letter auto-built templates are
	not available.

	Args:
		repo_root: repository root directory

	Returns:
		tuple of (template_img, mask_img) as numpy arrays, or
		(None, None) if the base template is not found
	"""
	base_path = os.path.join(repo_root, "artifacts",
		"base_letter_template.png")
	if not os.path.isfile(base_path):
		return (None, None)
	base_img = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
	if base_img is None:
		return (None, None)
	# generate mask from the base template
	base_mask = omr_utils.bubble_template_extractor._generate_template_mask(
		base_img)
	return (base_img, base_mask)


#============================================
def try_load_bubble_templates() -> tuple:
	"""Attempt to load bubble templates and masks from the default config path.

	Per-letter auto-built templates in config/bubble_templates/ take
	priority. For any missing letter, falls back to the hand-made base
	template in artifacts/base_letter_template.png as bootstrap.

	Returns:
		tuple of (templates_dict, masks_dict) where each maps letter to
		numpy array. Returns (empty_dict, empty_dict) if not available.
	"""
	# find the repo root by looking for config/ relative to this module
	module_dir = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.dirname(module_dir)
	template_dir = os.path.join(repo_root, "config", "bubble_templates")
	templates, masks = (
		omr_utils.bubble_template_extractor.load_templates_and_masks(
			template_dir))
	# fill missing letters with base template as bootstrap
	all_letters = ["A", "B", "C", "D", "E"]
	missing = [c for c in all_letters if c not in templates]
	if missing:
		base_img, base_mask = _load_base_template(repo_root)
		if base_img is not None:
			for letter in missing:
				templates[letter] = base_img
				if base_mask is not None:
					masks[letter] = base_mask
	return (templates, masks)
