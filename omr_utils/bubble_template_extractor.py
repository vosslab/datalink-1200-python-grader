"""Extract pixel templates of printed bubble letters from registered scans.

Follows a cryoEM-inspired class averaging approach: extract many instances
of each letter (A-E), align them, and average to produce a clean reference
template with improved SNR. Templates are stored at 5X oversize resolution
to preserve sub-pixel detail from the averaging process.
"""

# Standard Library
import os

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.template_loader


#============================================
def extract_bubble_patch(gray: numpy.ndarray, cx: int, cy: int,
	geom: dict) -> numpy.ndarray:
	"""Crop a single bubble region from a grayscale image.

	Extracts the pixel region around a bubble center using geometry
	dimensions, adds small padding, and resizes to 5X oversize for
	sub-pixel precision in the averaged template.

	Args:
		gray: grayscale registered image
		cx: bubble center x in pixels
		cy: bubble center y in pixels
		geom: pixel geometry dict with half_width and half_height

	Returns:
		numpy array of shape (TEMPLATE_HEIGHT, TEMPLATE_WIDTH), uint8,
		or None if the region is out of bounds
	"""
	h, w = gray.shape
	half_w = int(geom["half_width"])
	half_h = int(geom["half_height"])
	# compute extraction region with padding
	x1 = cx - half_w - EXTRACT_PAD_X
	x2 = cx + half_w + EXTRACT_PAD_X
	y1 = cy - half_h - EXTRACT_PAD_Y
	y2 = cy + half_h + EXTRACT_PAD_Y
	# bounds check
	if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
		return None
	# extract and resize to 5X oversize
	patch = gray[y1:y2, x1:x2]
	resized = cv2.resize(patch, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
		interpolation=cv2.INTER_CUBIC)
	return resized


#============================================
def _score_patch_quality(patch: numpy.ndarray) -> float:
	"""Score how well a patch represents a clean empty bubble.

	A good empty bubble patch has dark bracket edges (top/bottom rows)
	and a bright interior with the printed letter. Computes a contrast
	metric: lower bracket edge brightness relative to interior brightness.

	Args:
		patch: grayscale template-sized patch

	Returns:
		quality score (higher = better contrast, cleaner bubble)
	"""
	ph, pw = patch.shape
	# bracket edge region: top and bottom 15% of patch height
	edge_height = max(1, int(ph * 0.15))
	# top and bottom edge strips
	top_strip = patch[:edge_height, :]
	bot_strip = patch[-edge_height:, :]
	edge_mean = float(numpy.mean(numpy.concatenate([
		top_strip.ravel(), bot_strip.ravel()])))
	# interior region: middle 50% of height
	interior_y1 = int(ph * 0.25)
	interior_y2 = int(ph * 0.75)
	interior = patch[interior_y1:interior_y2, :]
	interior_mean = float(numpy.mean(interior))
	# contrast: bright interior minus dark edges
	contrast = interior_mean - edge_mean
	return contrast


#============================================
def extract_letter_templates(gray: numpy.ndarray, template: dict,
	results: list, empty_score_max: float = 0.12,
	min_samples: int = 10) -> dict:
	"""Extract averaged pixel templates for each bubble letter (A-E).

	Collects patches from empty bubbles (low fill score) across all
	questions, groups by letter, filters for quality, and computes a
	median stack for each letter to produce clean reference templates.

	Args:
		template: loaded template dictionary
		results: list of answer dicts from read_answers()
		empty_score_max: maximum fill score to consider a bubble empty
		min_samples: minimum patches required per letter for averaging

	Returns:
		dict mapping letter string to numpy array template, e.g.
		{"A": array, "B": array, ...}. Letters with insufficient
		samples are omitted.
	"""
	h, w = gray.shape
	choices = template["answers"]["choices"]
	geom = omr_utils.template_loader.get_bubble_geometry_px(template, w, h)
	# collect patches grouped by letter
	patches_by_letter = {choice: [] for choice in choices}
	for entry in results:
		answer = entry.get("answer", "")
		flags = entry.get("flags", "")
		scores = entry.get("scores", {})
		positions = entry.get("positions", {})
		# skip rows flagged as MULTIPLE (unreliable)
		if "MULTIPLE" in flags:
			continue
		for choice in choices:
			# only use empty bubbles: BLANK rows or non-selected choices
			is_empty = False
			if flags == "BLANK":
				is_empty = True
			elif answer and choice != answer:
				is_empty = True
			if not is_empty:
				continue
			# check fill score is below threshold
			score = float(scores.get(choice, 1.0))
			if score > empty_score_max:
				continue
			# get refined position
			if choice not in positions:
				continue
			px, py = positions[choice]
			# extract patch
			patch = extract_bubble_patch(gray, px, py, geom)
			if patch is not None:
				patches_by_letter[choice].append(patch)
	# build averaged templates via median stacking
	templates = {}
	for choice in choices:
		patches = patches_by_letter[choice]
		if len(patches) < min_samples:
			continue
		# score patch quality and sort descending
		scored = [(p, _score_patch_quality(p)) for p in patches]
		scored.sort(key=lambda item: item[1], reverse=True)
		# take top 75% by quality to reject outliers
		keep_count = max(min_samples, int(len(scored) * 0.75))
		best_patches = [p for p, _ in scored[:keep_count]]
		# median stack for robust averaging (rejects outlier pixels)
		stack = numpy.stack(best_patches, axis=0)
		median_template = numpy.median(stack, axis=0).astype(numpy.uint8)
		templates[choice] = median_template
	return templates


#============================================
def save_templates(templates: dict, output_dir: str) -> list:
	"""Save pixel templates as grayscale PNG files.

	Args:
		templates: dict mapping letter to numpy array
		output_dir: directory to save template PNGs

	Returns:
		list of saved file paths
	"""
	os.makedirs(output_dir, exist_ok=True)
	saved_paths = []
	for letter, template_img in sorted(templates.items()):
		filename = f"{letter}.png"
		filepath = os.path.join(output_dir, filename)
		cv2.imwrite(filepath, template_img)
		saved_paths.append(filepath)
	return saved_paths


#============================================
def load_templates(template_dir: str) -> dict:
	"""Load pixel templates from a directory of PNG files.

	Expects files named A.png, B.png, C.png, D.png, E.png.

	Args:
		template_dir: directory containing template PNG files

	Returns:
		dict mapping letter to numpy array, or empty dict if
		directory does not exist or contains no valid templates
	"""
	if not os.path.isdir(template_dir):
		return {}
	templates = {}
	for letter in ["A", "B", "C", "D", "E"]:
		filepath = os.path.join(template_dir, f"{letter}.png")
		if os.path.isfile(filepath):
			img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
			if img is not None:
				templates[letter] = img
	return templates


#============================================
def scale_template_to_bubble(template_img: numpy.ndarray,
	geom: dict) -> numpy.ndarray:
	"""Scale a 5X oversize template down to match actual bubble size.

	Uses the runtime geometry to determine the target bubble dimensions
	at the current image resolution.

	Args:
		template_img: 5X oversize template array
		geom: pixel geometry dict with half_width and half_height

	Returns:
		scaled template array matching the expected bubble dimensions
	"""
	# target size includes padding used during extraction
	target_w = int(geom["half_width"]) * 2 + EXTRACT_PAD_X * 2
	target_h = int(geom["half_height"]) * 2 + EXTRACT_PAD_Y * 2
	# ensure minimum size
	target_w = max(target_w, 5)
	target_h = max(target_h, 3)
	scaled = cv2.resize(template_img, (target_w, target_h),
		interpolation=cv2.INTER_AREA)
	return scaled
