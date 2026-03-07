"""Runtime utilities for bubble template I/O and ROI operations.

Provides the functions used by the grading pipeline at runtime:
- extract_roi_from_bounds() - crop at native resolution using lattice bounds
- normalize_roi_percentile() - percentile-based contrast stretch
- scale_template_to_bubble() - scale canonical template to slot size
- save_templates() / load_templates() - template PNG I/O

Offline template construction (alignment, medoid, symmetry) lives in
omr_utils/template_builder.py.
"""

# Standard Library
import os
import sys

# PIP3 modules
import cv2
import numpy

# ANSI color codes for stderr diagnostics
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


#============================================
def _log_image_saved(filepath: str) -> None:
	"""Print a colored diagnostic to stderr when an image is saved.

	Args:
		filepath: path to the saved image file
	"""
	sys.stderr.write(
		f"  {_GREEN}SAVED{_RESET} {_CYAN}{filepath}{_RESET}\n")


#============================================
def extract_roi_from_bounds(gray: numpy.ndarray, left_x: int,
	top_y: int, right_x: int, bot_y: int) -> numpy.ndarray:
	"""Crop bubble ROI using exact lattice bounds with no padding.

	Accepts explicit pixel bounds only. Crops exactly at the lattice
	midpoint boundaries so neighboring slots tile without overlap or
	gaps. Clips to image edges, returns None if the resulting region
	is too small.

	Args:
		gray: grayscale image
		left_x: left bound from SlotMap.roi_bounds()
		top_y: top bound from SlotMap.roi_bounds()
		right_x: right bound from SlotMap.roi_bounds()
		bot_y: bottom bound from SlotMap.roi_bounds()

	Returns:
		numpy array (grayscale crop) or None if too small
	"""
	img_h, img_w = gray.shape
	# crop exactly at lattice bounds, clipped to image edges
	x1 = max(0, left_x)
	y1 = max(0, top_y)
	x2 = min(img_w, right_x)
	y2 = min(img_h, bot_y)
	# reject if resulting region is too small
	if (x2 - x1) < 5 or (y2 - y1) < 3:
		return None
	roi = gray[y1:y2, x1:x2].copy()
	return roi


#============================================
def normalize_roi_percentile(roi: numpy.ndarray,
	low_pct: int = 1, high_pct: int = 75) -> numpy.ndarray:
	"""Percentile-based contrast stretch for a single ROI.

	Linearly maps the [low_pct, high_pct] intensity range to [0, 255]
	and clips. The asymmetric defaults (1/75) match typical bubble ROI
	content: mostly white paper background with a small dark
	bracket/letter feature.

	Args:
		roi: grayscale uint8 array
		low_pct: percentile that maps to 0 (clips bottom)
		high_pct: percentile that maps to 255 (clips top)

	Returns:
		contrast-stretched uint8 array, same shape as input
	"""
	# compute percentile anchors
	low = numpy.percentile(roi, low_pct)
	high = numpy.percentile(roi, high_pct)
	# guard against flat patches where percentiles collapse
	if high - low < 1:
		return roi
	# linear stretch from [low, high] to [0, 255], clip, cast
	stretched = (roi.astype(numpy.float32) - low) * (255.0 / (high - low))
	result = numpy.clip(stretched, 0, 255).astype(numpy.uint8)
	return result


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
		_log_image_saved(filepath)
		saved_paths.append(filepath)
	return saved_paths


#============================================
def scale_template_to_bubble(template_img: numpy.ndarray,
	slot_width: int, slot_height: int) -> numpy.ndarray:
	"""Scale a canonical high-res template down to exact slot size.

	The template lives in a canonical high-resolution coordinate frame.
	Runtime slot dimensions come from SlotMap.roi_bounds() lattice
	geometry. The template is scaled to exactly match the slot size
	with no padding, so it tiles cleanly with neighboring slots.

	Args:
		template_img: canonical high-resolution template array
		slot_width: slot width in pixels from roi_bounds (right_x - left_x)
		slot_height: slot height in pixels from roi_bounds (bot_y - top_y)

	Returns:
		scaled template array matching the exact slot dimensions
	"""
	# scale to exact slot size with no padding
	target_w = int(max(slot_width, 5))
	target_h = int(max(slot_height, 3))
	scaled = cv2.resize(template_img, (target_w, target_h),
		interpolation=cv2.INTER_AREA)
	return scaled


#============================================
def load_templates(template_dir: str) -> dict:
	"""Load pixel templates from a directory.

	Expects files named A.png, B.png, ..., E.png.

	Args:
		template_dir: directory containing template PNG files

	Returns:
		dict mapping letter to numpy array. Empty dict if
		directory missing.
	"""
	if not os.path.isdir(template_dir):
		return {}
	templates = {}
	for letter in ["A", "B", "C", "D", "E"]:
		template_path = os.path.join(template_dir, f"{letter}.png")
		if os.path.isfile(template_path):
			img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
			if img is not None:
				templates[letter] = img
	return templates
