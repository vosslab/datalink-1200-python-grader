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
		tuple of (refined_cx, refined_cy, confidence, dx, dy,
		score_at_seed) where confidence is the NCC peak value
		(0.0 to 1.0), dx/dy are the shift from approximate
		position, and score_at_seed is the NCC score at the
		zero-shift baseline. Returns the original position with
		confidence=0 if matching fails.
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
		return (approx_cx, approx_cy, 0.0, 0, 0, 0.0)
	# ROI must be larger than template in both dimensions
	roi_w = roi_x2 - roi_x1
	roi_h = roi_y2 - roi_y1
	if roi_w <= tw or roi_h <= th:
		return (approx_cx, approx_cy, 0.0, 0, 0, 0.0)
	# extract search region and normalize contrast to match template
	roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
	roi = omr_utils.bubble_template_extractor.normalize_roi_percentile(roi)
	# run normalized cross-correlation
	result = cv2.matchTemplate(roi, template_img, cv2.TM_CCOEFF_NORMED)
	# extract NCC score at seed position (zero-shift baseline)
	# result map center = search_radius corresponds to zero shift
	rh, rw = result.shape[:2]
	seed_rx = search_radius
	seed_ry = search_radius
	if 0 <= seed_rx < rw and 0 <= seed_ry < rh:
		score_at_seed = float(result[seed_ry, seed_rx])
	else:
		score_at_seed = 0.0
	# find peak location
	_, max_val, _, max_loc = cv2.minMaxLoc(result)
	confidence = float(max_val)
	# convert peak location back to image coordinates
	# max_loc is (x, y) of the top-left corner of the best match
	# bubble center = match top-left + template half dimensions
	refined_cx = roi_x1 + max_loc[0] + half_tw
	refined_cy = roi_y1 + max_loc[1] + half_th
	# compute shift from approximate position
	dx = refined_cx - approx_cx
	dy = refined_cy - approx_cy
	return (refined_cx, refined_cy, confidence, dx, dy, score_at_seed)


#============================================
def refine_row_by_template(gray: numpy.ndarray, templates: dict,
	row_positions: dict, choices: list,
	search_radius: int = 8,
	slot_dims: dict = None) -> dict:
	"""Refine all 5 choice positions in a question row using templates.

	For each choice, uses the corresponding letter's pixel template to
	refine the position via unmasked NCC (TM_CCOEFF_NORMED). Only
	updates positions where the match confidence exceeds the threshold.

	Args:
		gray: grayscale image
		templates: dict mapping letter to 5X oversize template array
		row_positions: dict mapping choice letter to (cx, cy) approx position
		choices: list of choice letters ["A", "B", "C", "D", "E"]
		search_radius: search radius in pixels
		slot_dims: dict mapping choice to (width, height) from
			SlotMap.roi_bounds(); required for template scaling

	Returns:
		dict mapping choice letter to (refined_cx, refined_cy,
		confidence, score_at_seed). Choices without templates or
		below confidence threshold keep their original positions
		with confidence=0.
	"""
	refined = {}
	for choice in choices:
		if choice not in row_positions:
			continue
		cx, cy = row_positions[choice]
		# get the letter's template
		template_img = templates.get(choice)
		if template_img is None:
			refined[choice] = (cx, cy, 0.0, 0.0)
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
		# unmasked NCC matching
		rcx, rcy, conf, _, _, score_seed = match_bubble_local(
			gray, scaled, cx, cy, search_radius)
		if conf >= 0.30:
			refined[choice] = (rcx, rcy, conf, score_seed)
		else:
			# keep original position when confidence is low
			refined[choice] = (cx, cy, 0.0, 0.0)
	return refined


#============================================
def _load_base_template(repo_root: str) -> numpy.ndarray:
	"""Load the hand-made base template as bootstrap for all letters.

	The base template in artifacts/base_letter_template.png represents
	a single correct slot structure (left bracket + letter + right bracket).
	It is used as a fallback when per-letter auto-built templates are
	not available.

	Args:
		repo_root: repository root directory

	Returns:
		template numpy array, or None if the base template is not found
	"""
	base_path = os.path.join(repo_root, "artifacts",
		"base_letter_template.png")
	if not os.path.isfile(base_path):
		return None
	base_img = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
	return base_img


#============================================
def try_load_bubble_templates() -> dict:
	"""Attempt to load bubble templates from the default config path.

	Per-letter auto-built templates in config/bubble_templates/ take
	priority. For any missing letter, falls back to the hand-made base
	template in artifacts/base_letter_template.png as bootstrap.

	Returns:
		dict mapping letter to numpy array. Returns empty dict if
		not available.
	"""
	# find the repo root by looking for config/ relative to this module
	module_dir = os.path.dirname(os.path.abspath(__file__))
	repo_root = os.path.dirname(module_dir)
	template_dir = os.path.join(repo_root, "config", "bubble_templates")
	templates = omr_utils.bubble_template_extractor.load_templates(
		template_dir)
	# fill missing letters with base template as bootstrap
	all_letters = ["A", "B", "C", "D", "E"]
	missing = [c for c in all_letters if c not in templates]
	if missing:
		base_img = _load_base_template(repo_root)
		if base_img is not None:
			for letter in missing:
				templates[letter] = base_img
	return templates
