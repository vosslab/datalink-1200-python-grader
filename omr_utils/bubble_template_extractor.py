"""Extract pixel templates of printed bubble letters from registered scans.

Follows a cryoEM-inspired class averaging approach: extract many instances
of each letter (A-E), align them, and average to produce a clean reference
template with improved SNR. Templates are stored at canonical high resolution
to preserve sub-pixel detail from the averaging process.

Offline template construction pipeline:
1. extract_roi_from_bounds() - crop at native resolution using lattice bounds
2. _find_medoid_roi() - pick the most representative ROI as reference
3. _align_roi_to_reference() - translation-only NCC alignment
4. _apply_symmetry_augmentation() - mirror by letter symmetry
5. _build_letter_template() - aligned average with outlier rejection
6. _generate_template_mask() - derive bracket-emphasis mask
"""

# Standard Library
import os
import time

# PIP3 modules
import cv2
import numpy
import scipy.stats

# local repo modules

# Canonical template resolution: all offline-built templates are stored
# at this fixed size so runtime scaling always goes DOWN.
# Chosen to be ~8x the typical runtime bubble size (target ~60x11 px),
# keeping detail without being absurdly oversized.
CANONICAL_TEMPLATE_WIDTH = 480
CANONICAL_TEMPLATE_HEIGHT = 88


#============================================
def extract_roi_from_bounds(gray: numpy.ndarray, left_x: int,
	top_y: int, right_x: int, bot_y: int,
	pad_factor: float = 0.4) -> numpy.ndarray:
	"""Crop bubble ROI using explicit lattice bounds with padding.

	Accepts explicit pixel bounds only. Adds proportional padding for
	alignment search room, clips to image edges, returns None if the
	resulting region is too small.

	Args:
		gray: grayscale image
		left_x: left bound from SlotMap.roi_bounds()
		top_y: top bound from SlotMap.roi_bounds()
		right_x: right bound from SlotMap.roi_bounds()
		bot_y: bottom bound from SlotMap.roi_bounds()
		pad_factor: fractional padding beyond bounds (0.4 = 40%)

	Returns:
		numpy array (grayscale crop) or None if too small
	"""
	img_h, img_w = gray.shape
	bounds_w = right_x - left_x
	bounds_h = bot_y - top_y
	# add proportional padding for alignment search room
	pad_x = max(4.0, bounds_w * pad_factor)
	pad_y = max(2.0, bounds_h * pad_factor)
	# compute padded region, clipped to image edges
	x1 = max(0, int(left_x - pad_x))
	y1 = max(0, int(top_y - pad_y))
	x2 = min(img_w, int(right_x + pad_x))
	y2 = min(img_h, int(bot_y + pad_y))
	# reject if resulting region is too small
	if (x2 - x1) < 5 or (y2 - y1) < 3:
		return None
	roi = gray[y1:y2, x1:x2].copy()
	return roi


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
def scale_template_to_bubble(template_img: numpy.ndarray,
	slot_width: int, slot_height: int) -> numpy.ndarray:
	"""Scale a canonical high-res template down to local bubble size.

	The template lives in a canonical high-resolution coordinate frame.
	Runtime slot dimensions come from SlotMap.roi_bounds() lattice
	geometry. The template is always scaled DOWN to match the actual
	slot size with small padding for alignment room.

	Args:
		template_img: canonical high-resolution template array
		slot_width: slot width in pixels from roi_bounds (right_x - left_x)
		slot_height: slot height in pixels from roi_bounds (bot_y - top_y)

	Returns:
		scaled template array matching the expected bubble dimensions
	"""
	# add small proportional padding for alignment search room
	pad_x = max(2, int(slot_width * 0.1))
	pad_y = max(1, int(slot_height * 0.2))
	target_w = slot_width + pad_x * 2
	target_h = slot_height + pad_y * 2
	# ensure minimum size; int-cast for cv2.resize
	target_w = int(max(target_w, 5))
	target_h = int(max(target_h, 3))
	scaled = cv2.resize(template_img, (target_w, target_h),
		interpolation=cv2.INTER_AREA)
	return scaled


#============================================
def _find_medoid_roi(rois: list) -> int:
	"""Select the ROI with highest average NCC to all others as reference.

	The medoid is the most representative sample: the one that best
	matches the rest of the collection.

	Args:
		rois: list of grayscale numpy arrays (all same shape)

	Returns:
		index of the medoid ROI in the input list
	"""
	n = len(rois)
	if n <= 1:
		return 0
	# resize all ROIs to the same shape (smallest common dimensions)
	min_h = min(r.shape[0] for r in rois)
	min_w = min(r.shape[1] for r in rois)
	resized = []
	for r in rois:
		if r.shape[0] != min_h or r.shape[1] != min_w:
			resized.append(cv2.resize(r, (min_w, min_h),
				interpolation=cv2.INTER_AREA))
		else:
			resized.append(r)
	# compute pairwise NCC scores
	avg_scores = numpy.zeros(n, dtype=numpy.float64)
	for i in range(n):
		# print progress every 200 candidates
		if i > 0 and i % 200 == 0:
			print(f"      medoid: {i}/{n} candidates scored")
		total = 0.0
		count = 0
		for j in range(n):
			if i == j:
				continue
			# direct pixel correlation (no template matching needed,
			# since they are the same size)
			ri = resized[i].astype(numpy.float64)
			rj = resized[j].astype(numpy.float64)
			ri_norm = ri - ri.mean()
			rj_norm = rj - rj.mean()
			denom = numpy.sqrt(numpy.sum(ri_norm ** 2) * numpy.sum(rj_norm ** 2))
			if denom < 1e-6:
				continue
			ncc = float(numpy.sum(ri_norm * rj_norm) / denom)
			total += ncc
			count += 1
		if count > 0:
			avg_scores[i] = total / count
	medoid_idx = int(numpy.argmax(avg_scores))
	return medoid_idx


#============================================
def _align_roi_to_reference(roi: numpy.ndarray,
	reference: numpy.ndarray) -> tuple:
	"""Align an ROI to a reference via translation-only NCC.

	Uses cv2.matchTemplate to find the best translation offset. The
	reference must be smaller than the ROI (or they must match in size).

	Args:
		roi: grayscale ROI to align
		reference: grayscale reference template (must be <= roi in size)

	Returns:
		tuple of (aligned_roi, dx, dy, score) where aligned_roi is
		cropped to match reference dimensions, dx/dy are the shift,
		and score is the NCC peak value. Returns (roi_center_crop,
		0, 0, 0.0) if matching fails.
	"""
	rh, rw = roi.shape
	th, tw = reference.shape
	# if reference is larger, crop it to fit
	if th > rh or tw > rw:
		# crop reference center to match roi
		crop_h = min(th, rh)
		crop_w = min(tw, rw)
		cy = th // 2
		cx = tw // 2
		ref_crop = reference[cy - crop_h // 2:cy + crop_h // 2,
			cx - crop_w // 2:cx + crop_w // 2]
		aligned = cv2.resize(roi, (ref_crop.shape[1], ref_crop.shape[0]),
			interpolation=cv2.INTER_AREA)
		return (aligned, 0, 0, 0.0)
	# run NCC template matching
	result = cv2.matchTemplate(roi, reference, cv2.TM_CCOEFF_NORMED)
	_, max_val, _, max_loc = cv2.minMaxLoc(result)
	score = float(max_val)
	# max_loc is top-left of best match position
	best_x, best_y = max_loc
	# center offset relative to "centered" position
	center_x = (rw - tw) // 2
	center_y = (rh - th) // 2
	dx = best_x - center_x
	dy = best_y - center_y
	# crop aligned region from roi at the matched position
	aligned = roi[best_y:best_y + th, best_x:best_x + tw].copy()
	return (aligned, dx, dy, score)


# symmetry axis rules per letter
_SYMMETRY_AXIS = {
	"A": "lr",  # left-right mirror
	"B": "tb",  # top-bottom mirror
	"C": "tb",
	"D": "tb",
	"E": "tb",
}


#============================================
def _apply_symmetry_augmentation(rois: list, letter: str) -> list:
	"""Double the ROI count by appending mirrored copies.

	Each letter has a known symmetry axis:
	- A: left-right flip (approximate bilateral symmetry)
	- B/C/D/E: top-bottom flip (bracket pair is vertically symmetric)

	Args:
		rois: list of grayscale numpy arrays
		letter: single letter string (A-E)

	Returns:
		list of rois + their mirrored copies (2x length)
	"""
	axis = _SYMMETRY_AXIS.get(letter, "tb")
	augmented = list(rois)
	for roi in rois:
		if axis == "lr":
			mirrored = numpy.fliplr(roi)
		else:
			mirrored = numpy.flipud(roi)
		augmented.append(mirrored)
	return augmented


#============================================
def _build_letter_template(rois: list, letter: str,
	reject_threshold: float = 0.5) -> tuple:
	"""Build a sharp per-letter template from aligned ROIs.

	Pipeline: symmetry augmentation -> medoid selection -> alignment
	-> outlier rejection -> trimmed mean.

	Args:
		rois: list of native-resolution grayscale ROI arrays
		letter: bubble letter (A-E)
		reject_threshold: minimum NCC score to keep an aligned ROI

	Returns:
		tuple of (template, mask, alignment_table) where:
		- template: uint8 grayscale array
		- mask: uint8 array emphasizing bracket structure
		- alignment_table: list of dicts with dx, dy, score, kept
		Returns (None, None, []) if fewer than 3 ROIs survive
	"""
	if len(rois) < 3:
		return (None, None, [])
	# apply symmetry augmentation to double sample count
	augmented = _apply_symmetry_augmentation(rois, letter)
	# find the medoid as alignment reference
	n_aug = len(augmented)
	print(f"    finding medoid in {n_aug} augmented ROIs...")
	t0 = time.time()
	medoid_idx = _find_medoid_roi(augmented)
	medoid_elapsed = time.time() - t0
	print(f"    medoid found (idx {medoid_idx}, {medoid_elapsed:.1f}s)")
	reference = augmented[medoid_idx]
	# align all ROIs to reference
	print(f"    aligning {n_aug} ROIs to reference...")
	t1 = time.time()
	aligned_rois = []
	alignment_table = []
	for i, roi in enumerate(augmented):
		aligned, dx, dy, score = _align_roi_to_reference(roi, reference)
		entry = {"index": i, "dx": dx, "dy": dy, "score": score,
			"kept": score >= reject_threshold}
		alignment_table.append(entry)
		if score >= reject_threshold:
			aligned_rois.append(aligned)
	align_elapsed = time.time() - t1
	print(f"    alignment done ({align_elapsed:.1f}s)")
	# need at least 3 aligned ROIs for a good template
	if len(aligned_rois) < 3:
		return (None, None, alignment_table)
	# upscale all aligned ROIs to canonical high resolution before averaging;
	# this ensures the master template is always higher-res than any
	# individual source, so runtime scaling always goes DOWN
	canonical_w = CANONICAL_TEMPLATE_WIDTH
	canonical_h = CANONICAL_TEMPLATE_HEIGHT
	uniform = []
	for ar in aligned_rois:
		upscaled = cv2.resize(ar, (canonical_w, canonical_h),
			interpolation=cv2.INTER_CUBIC)
		uniform.append(upscaled.astype(numpy.float32))
	# trimmed mean: reject top and bottom 10% at each pixel
	stack = numpy.stack(uniform, axis=0)
	template_float = scipy.stats.trim_mean(stack, proportiontocut=0.1, axis=0)
	template = numpy.clip(template_float, 0, 255).astype(numpy.uint8)
	# generate mask from the template
	mask = _generate_template_mask(template)
	return (template, mask, alignment_table)


#============================================
def _generate_template_mask(template: numpy.ndarray) -> numpy.ndarray:
	"""Derive a mask emphasizing bracket structure over background.

	Dark regions (bracket edges, printed letter) get high mask weight.
	Bright background regions get low weight. This lets the masked NCC
	matcher focus on structural features rather than uniform paper.

	Args:
		template: uint8 grayscale template

	Returns:
		uint8 mask array (same shape as template), values 0-255
	"""
	# invert: dark bracket features become bright mask values
	inverted = 255 - template
	# apply mild Gaussian blur to smooth mask edges
	blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
	# threshold to suppress low-contrast background
	# use Otsu to find automatic threshold
	_, mask = cv2.threshold(blurred, 0, 255,
		cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	# dilate slightly to include edge neighborhoods
	kernel = numpy.ones((3, 3), numpy.uint8)
	mask = cv2.dilate(mask, kernel, iterations=1)
	return mask


#============================================
def load_templates_and_masks(template_dir: str) -> tuple:
	"""Load pixel templates and their masks from a directory.

	Expects files named A.png, B.png, ..., E.png for templates
	and A_mask.png, B_mask.png, ..., E_mask.png for masks.

	Args:
		template_dir: directory containing template and mask PNG files

	Returns:
		tuple of (templates_dict, masks_dict) where each maps
		letter to numpy array. Empty dicts if directory missing.
	"""
	if not os.path.isdir(template_dir):
		return ({}, {})
	templates = {}
	masks = {}
	for letter in ["A", "B", "C", "D", "E"]:
		# load template
		template_path = os.path.join(template_dir, f"{letter}.png")
		if os.path.isfile(template_path):
			img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
			if img is not None:
				templates[letter] = img
		# load mask
		mask_path = os.path.join(template_dir, f"{letter}_mask.png")
		if os.path.isfile(mask_path):
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			if mask is not None:
				masks[letter] = mask
	return (templates, masks)


#============================================
def save_templates_and_masks(templates: dict, masks: dict,
	output_dir: str) -> list:
	"""Save templates and masks as PNG files.

	Args:
		templates: dict mapping letter to template numpy array
		masks: dict mapping letter to mask numpy array
		output_dir: directory to save files

	Returns:
		list of saved file paths
	"""
	os.makedirs(output_dir, exist_ok=True)
	saved_paths = []
	for letter, template_img in sorted(templates.items()):
		# save template
		filepath = os.path.join(output_dir, f"{letter}.png")
		cv2.imwrite(filepath, template_img)
		saved_paths.append(filepath)
	for letter, mask_img in sorted(masks.items()):
		# save mask
		filepath = os.path.join(output_dir, f"{letter}_mask.png")
		cv2.imwrite(filepath, mask_img)
		saved_paths.append(filepath)
	return saved_paths


#============================================
def _save_qc_montage(rois: list, aligned_rois: list,
	alignment_table: list, letter: str, template: numpy.ndarray,
	mask: numpy.ndarray, output_dir: str) -> str:
	"""Save a QC montage showing alignment results for one letter.

	Creates a multi-panel image:
	- Top row: original ROIs (green border = kept, red = rejected)
	- Middle row: aligned ROIs (kept only)
	- Bottom row: final template and mask

	Args:
		rois: original ROI list (before augmentation)
		aligned_rois: aligned ROI list
		alignment_table: list of alignment dicts with score and kept
		letter: bubble letter
		template: final averaged template
		mask: final mask
		output_dir: directory to save QC images

	Returns:
		path to saved montage file
	"""
	os.makedirs(output_dir, exist_ok=True)
	# determine cell size from template
	cell_h = template.shape[0]
	cell_w = template.shape[1]
	# limit to first 20 ROIs for montage readability
	show_count = min(len(rois), 20)
	cols = min(show_count, 10)
	rows = (show_count + cols - 1) // cols
	# panel 1: original ROIs in a grid
	panel_h = cell_h * rows
	panel_w = cell_w * cols
	# build color montage for border annotations
	panel1 = numpy.full((panel_h, panel_w, 3), 200, dtype=numpy.uint8)
	for i in range(show_count):
		r = i // cols
		c = i % cols
		y1 = r * cell_h
		x1 = c * cell_w
		# resize ROI to cell size
		roi_resized = cv2.resize(rois[i], (cell_w, cell_h),
			interpolation=cv2.INTER_AREA)
		roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
		# border color: green if kept (index i in alignment_table), red if not
		if i < len(alignment_table) and alignment_table[i]["kept"]:
			border_color = (0, 200, 0)
		else:
			border_color = (0, 0, 200)
		cv2.rectangle(roi_bgr, (0, 0),
			(cell_w - 1, cell_h - 1), border_color, 2)
		panel1[y1:y1 + cell_h, x1:x1 + cell_w] = roi_bgr
	# panel 2: template and mask side by side
	tmpl_bgr = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
	mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	panel2 = numpy.concatenate([tmpl_bgr, mask_bgr], axis=1)
	# combine panels vertically
	# resize panel2 to match panel1 width
	if panel2.shape[1] != panel_w:
		scale = panel_w / panel2.shape[1]
		new_h = max(1, int(panel2.shape[0] * scale))
		panel2 = cv2.resize(panel2, (panel_w, new_h),
			interpolation=cv2.INTER_AREA)
	montage = numpy.concatenate([panel1, panel2], axis=0)
	# save
	filepath = os.path.join(output_dir, f"qc_{letter}.png")
	cv2.imwrite(filepath, montage)
	return filepath
