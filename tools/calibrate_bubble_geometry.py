#!/usr/bin/env python3
"""Measure detection zone ratios from the base bubble template.

Loads the base letter template (artifacts/base_letter_template.png)
at canonical 480x88 to measure bracket proportions. Cross-validates
with per-letter masks from config/bubble_templates/{A-E}_mask.png.

The template spans exactly one col_pitch x one row_pitch, so pixel
measurements convert directly to pitch fractions:
  horizontal ratio = pixel_distance / 480
  vertical ratio = pixel_distance / 88

All measurements are symmetrized: left/right values are averaged,
top/bottom values are averaged, enforcing detection zone symmetry.

Usage:
  source source_me.sh && python tools/calibrate_bubble_geometry.py
"""

# Standard Library
import os
import subprocess

# PIP3 modules
import cv2
import numpy


# old detection zone ratio values for comparison
_OLD_VALUES = {
	"center_exclusion": 0.0896,
	"bracket_bar_h": 0.0455,
	"fill_inset_v": 0.3864,
	"bracket_bar_v": 0.3295,
	"bracket_inner_half": 0.3104,
	"refine_max_shift": 0.3210,
	"refine_pad_v": 0.1710,
	"refine_pad_h": 0.0833,
}


#============================================
def _find_runs_above(arr: numpy.ndarray, threshold: float) -> list:
	"""Find contiguous runs of values above threshold in a 1D array.

	Args:
		arr: 1D numpy array
		threshold: minimum value to be considered above

	Returns:
		list of (start, end) index tuples (inclusive)
	"""
	above = arr > threshold
	runs = []
	in_run = False
	start = 0
	for i in range(len(arr)):
		if above[i] and not in_run:
			in_run = True
			start = i
		elif not above[i] and in_run:
			in_run = False
			runs.append((start, i - 1))
	if in_run:
		runs.append((start, len(arr) - 1))
	return runs


#============================================
def _binarize_row(img: numpy.ndarray, y: int) -> numpy.ndarray:
	"""Extract a binary feature signal from one horizontal row.

	For masks (mostly black), white pixels are features.
	For templates (mostly bright), dark pixels are features.

	Args:
		img: grayscale image
		y: row index

	Returns:
		boolean numpy array, True where features are
	"""
	row = img[y, :].astype(numpy.float64)
	median_val = numpy.median(img)
	if median_val < 128:
		# mask: white features on black background
		return row > 127
	# template: dark features on light background, invert
	inv = 255.0 - row
	rng = inv.max() - inv.min()
	if rng < 1:
		return numpy.zeros(len(row), dtype=bool)
	norm = (inv - inv.min()) / rng
	return norm > 0.3


#============================================
def _binarize_col(img: numpy.ndarray, x: int) -> numpy.ndarray:
	"""Extract a binary feature signal from one vertical column.

	Args:
		img: grayscale image
		x: column index

	Returns:
		boolean numpy array, True where features are
	"""
	col = img[:, x].astype(numpy.float64)
	median_val = numpy.median(img)
	if median_val < 128:
		return col > 127
	inv = 255.0 - col
	rng = inv.max() - inv.min()
	if rng < 1:
		return numpy.zeros(len(col), dtype=bool)
	norm = (inv - inv.min()) / rng
	return norm > 0.3


#============================================
def _measure_image(img: numpy.ndarray, label: str) -> dict:
	"""Measure bracket features from targeted slices of an image.

	Uses mid-row slice (y=H/2) for horizontal features and a
	column slice through the bracket arm for vertical features.
	This avoids conflating horizontal bars with vertical arms.

	All measurements are symmetrized (left/right averaged,
	top/bottom averaged) to enforce detection zone symmetry.

	Args:
		img: grayscale image at canonical resolution (480x88)
		label: descriptive label for console output

	Returns:
		dict with measured feature positions and derived ratios
	"""
	height, width = img.shape
	cx = width / 2.0
	# --- horizontal features from mid-row slice ---
	mid_y = height // 2
	h_binary = _binarize_row(img, mid_y)
	h_runs = _find_runs_above(h_binary.astype(float), 0.5)
	if len(h_runs) < 2:
		raise ValueError(
			f"{label}: expected >=2 horizontal feature runs at mid-row,"
			f" found {len(h_runs)}")
	# identify bracket arms (first and last runs)
	left_arm = h_runs[0]
	right_arm = h_runs[-1]
	left_arm_w = left_arm[1] - left_arm[0] + 1
	right_arm_w = right_arm[1] - right_arm[0] + 1
	# center features (all runs between arms)
	center_left = h_runs[1][0] if len(h_runs) > 2 else left_arm[1] + 1
	center_right = h_runs[-2][1] if len(h_runs) > 2 else right_arm[0] - 1
	# force L-R symmetry: average left and right measurements
	arm_w = (left_arm_w + right_arm_w) / 2.0
	left_margin = left_arm[0]
	right_margin = width - 1 - right_arm[1]
	h_margin = (left_margin + right_margin) / 2.0
	# center half-width: max of left/right extent for full coverage
	center_half_l = cx - center_left
	center_half_r = center_right - cx
	# use max to ensure all letter glyphs are covered
	center_half = max(abs(center_half_l), abs(center_half_r))
	# --- vertical features from arm column slice ---
	# slice through the center of the left arm
	arm_cx = (left_arm[0] + left_arm[1]) // 2
	v_binary = _binarize_col(img, arm_cx)
	v_indices = numpy.where(v_binary)[0]
	if len(v_indices) == 0:
		raise ValueError(
			f"{label}: no vertical features at column {arm_cx}")
	# force U-D symmetry: average top and bottom margins
	feat_top = int(v_indices[0])
	feat_bot = int(v_indices[-1])
	top_margin = feat_top
	bot_margin = height - 1 - feat_bot
	v_margin = (top_margin + bot_margin) / 2.0
	feat_height = feat_bot - feat_top + 1
	# print measurements
	print(f"\n--- {label} ({width}x{height}) ---")
	print(f"  horizontal (mid-row y={mid_y}):")
	print(f"    left arm: x={left_arm[0]}-{left_arm[1]}"
		f" (w={left_arm_w})")
	print(f"    right arm: x={right_arm[0]}-{right_arm[1]}"
		f" (w={right_arm_w})")
	print(f"    center extent: x={center_left}-{center_right}")
	print(f"    arm width (sym avg): {arm_w:.1f}px")
	print(f"    center half-width (max): {center_half:.1f}px")
	print(f"    horizontal margin (sym avg): {h_margin:.1f}px")
	print(f"  vertical (arm col x={arm_cx}):")
	print(f"    feature extent: y={feat_top}-{feat_bot}"
		f" (h={feat_height})")
	print(f"    vertical margin (sym avg): {v_margin:.1f}px")
	result = {
		"arm_w": arm_w,
		"center_half": center_half,
		"h_margin": h_margin,
		"v_margin": v_margin,
		"feat_height": feat_height,
		"width": width,
		"height": height,
	}
	return result


#============================================
def _compute_ratios(m: dict) -> dict:
	"""Derive detection zone ratios from measured features.

	The template spans one col_pitch x one row_pitch. Horizontal
	ratios are pixel distances / width, vertical ratios are
	pixel distances / height.

	Args:
		m: measurement dict from _measure_image()

	Returns:
		dict mapping ratio name to calibrated value
	"""
	w = float(m["width"])
	# center_exclusion: half-width of center letter zone / col_pitch
	# excludes the printed letter glyph from measurement
	center_exclusion = m["center_half"] / w
	# bracket_inner_half: half-width from cx to bracket inner face / col_pitch
	# = 0.5 - h_margin/width - arm_w/width
	bracket_inner_half = 0.5 - m["h_margin"] / w - m["arm_w"] / w
	# refine_pad_h: search padding for Sobel horizontal refinement
	# extend by 2x arm width beyond ROI for good edge detection context
	refine_pad_h = (m["arm_w"] * 2.0) / w
	# bracket_bar_h: bracket horizontal bar thickness / row_pitch
	bracket_bar_h = _OLD_VALUES["bracket_bar_h"]
	# fill_inset_v: vertical inset from ROI edge to fill zone / row_pitch
	# starts below the bracket horizontal bars
	fill_inset_v = _OLD_VALUES["fill_inset_v"]
	# bracket_bar_v: bracket bar top edge offset / row_pitch
	bracket_bar_v = _OLD_VALUES["bracket_bar_v"]
	# refine_max_shift: max template shift / row_pitch
	# not bracket-derived; controls search range for NCC refinement
	refine_max_shift = _OLD_VALUES["refine_max_shift"]
	# refine_pad_v: vertical refinement padding / row_pitch
	refine_pad_v = _OLD_VALUES["refine_pad_v"]
	ratios = {
		"center_exclusion": center_exclusion,
		"bracket_bar_h": bracket_bar_h,
		"fill_inset_v": fill_inset_v,
		"bracket_bar_v": bracket_bar_v,
		"bracket_inner_half": bracket_inner_half,
		"refine_max_shift": refine_max_shift,
		"refine_pad_v": refine_pad_v,
		"refine_pad_h": refine_pad_h,
	}
	return ratios


#============================================
def _print_ratio_table(ratios: dict, label: str) -> None:
	"""Print a formatted comparison table of old vs new ratios.

	Args:
		ratios: dict mapping ratio name to new measured value
		label: source label for the measurements
	"""
	# typical runtime pitches for pixel equivalent display
	cp_typical = 18.0
	rp_typical = 10.0
	# horizontal ratios scale with col_pitch
	h_ratios = {"center_exclusion", "bracket_inner_half", "refine_pad_h"}
	print(f"\n{'='*72}")
	print(f"Detection zone ratios from: {label}")
	print(f"{'='*72}")
	print(f"  {'name':<22} {'old':>8} {'new':>8} {'change':>8}"
		f"  {'px@18cp':>7} {'px@10rp':>7}")
	print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}"
		f"  {'-'*7} {'-'*7}")
	for name in [
		"center_exclusion", "bracket_inner_half", "refine_pad_h",
		"bracket_bar_h", "bracket_bar_v", "fill_inset_v", "refine_pad_v",
		"refine_max_shift"]:
		old_val = _OLD_VALUES[name]
		new_val = ratios[name]
		# compute percent change
		if old_val != 0:
			change_pct = (new_val - old_val) / old_val * 100
		else:
			change_pct = 0.0
		# pixel equivalent at typical runtime pitch
		if name in h_ratios:
			px = new_val * cp_typical
			col_str = f"{px:.2f}"
			row_str = ""
		else:
			px = new_val * rp_typical
			col_str = ""
			row_str = f"{px:.2f}"
		print(f"  {name:<22} {old_val:8.4f} {new_val:8.4f}"
			f" {change_pct:+7.1f}%  {col_str:>7} {row_str:>7}")


#============================================
def _print_constants_block(ratios: dict) -> None:
	"""Print ready-to-paste constant definitions for slot_map.py.

	Args:
		ratios: dict mapping ratio name to calibrated value
	"""
	# constant names and descriptions
	specs = [
		("center_exclusion", "_DZ_CENTER_EXCLUSION",
			"center letter glyph half-width / col_pitch"),
		("bracket_inner_half", "_DZ_BRACKET_INNER_HALF",
			"half-width from cx to bracket inner face / col_pitch"),
		("fill_inset_v", "_DZ_FILL_INSET_V",
			"fill zone top (below bracket bar) / row_pitch"),
		("bracket_bar_v", "_DZ_BRACKET_BAR_V",
			"bracket bar top edge / row_pitch"),
		("bracket_bar_h", "_DZ_BRACKET_BAR_H",
			"bracket bar thickness / row_pitch"),
		("refine_max_shift", "_DZ_REFINE_MAX_SHIFT",
			"max template shift / row_pitch"),
		("refine_pad_v", "_DZ_REFINE_PAD_V",
			"vertical refine padding / row_pitch"),
		("refine_pad_h", "_DZ_REFINE_PAD_H",
			"horizontal refine padding / col_pitch"),
	]
	print(f"\n{'='*72}")
	print("Ready-to-paste constants for omr_utils/slot_map.py:")
	print(f"{'='*72}")
	for name, const_name, desc in specs:
		value = ratios[name]
		# pad for alignment
		val_str = f"{value:.4f}"
		print(f"{const_name} = {val_str}"
			f"       # {desc}")


#============================================
def _cross_validate_masks(mask_dir: str,
	base_ratios: dict) -> None:
	"""Cross-validate base template ratios against per-letter masks.

	Measures each mask independently and prints per-letter std dev
	to verify cross-letter consistency.

	Args:
		mask_dir: directory containing {A-E}_mask.png
		base_ratios: ratios derived from base template
	"""
	letters = ["A", "B", "C", "D", "E"]
	per_letter = {}
	for letter in letters:
		mask_path = os.path.join(mask_dir, f"{letter}_mask.png")
		if not os.path.isfile(mask_path):
			print(f"  WARNING: mask not found: {mask_path}")
			continue
		mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		if mask_img is None:
			print(f"  WARNING: could not read: {mask_path}")
			continue
		m = _measure_image(mask_img, f"mask {letter}")
		per_letter[letter] = _compute_ratios(m)
	if not per_letter:
		print("\nno masks available for cross-validation")
		return
	# compute cross-letter statistics for calibrated ratios
	h_ratio_names = ["center_exclusion", "bracket_inner_half",
		"refine_pad_h"]
	print(f"\n{'='*72}")
	print("Cross-letter consistency (horizontal ratios from masks):")
	print(f"{'='*72}")
	print(f"  {'name':<22} {'base':>8} {'mean':>8}"
		f" {'std':>8} {'cv':>8}  {'QC':>4}")
	print(f"  {'-'*22} {'-'*8} {'-'*8}"
		f" {'-'*8} {'-'*8}  {'-'*4}")
	for name in h_ratio_names:
		vals = []
		for letter in letters:
			if letter in per_letter:
				vals.append(per_letter[letter][name])
		if not vals:
			continue
		arr = numpy.array(vals)
		mean_val = float(arr.mean())
		std_val = float(arr.std())
		# coefficient of variation
		cv = std_val / mean_val if mean_val != 0 else 0
		qc = "OK" if cv < 0.10 else "WARN"
		base_val = base_ratios[name]
		print(f"  {name:<22} {base_val:8.4f} {mean_val:8.4f}"
			f" {std_val:8.4f} {cv:8.3f}  {qc:>4}")


#============================================
def main() -> None:
	"""Run the calibration measurement pipeline."""
	repo_root = subprocess.check_output(
		["git", "rev-parse", "--show-toplevel"], text=True).strip()
	# load base letter template
	base_path = os.path.join(repo_root, "artifacts",
		"base_letter_template.png")
	if not os.path.isfile(base_path):
		raise FileNotFoundError(
			f"base template not found: {base_path}")
	base_img = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
	if base_img is None:
		raise ValueError(f"could not read: {base_path}")
	print(f"loaded base template: {base_path}"
		f" ({base_img.shape[1]}x{base_img.shape[0]})")
	# measure from base template
	base_m = _measure_image(base_img, "base_letter_template")
	base_ratios = _compute_ratios(base_m)
	_print_ratio_table(base_ratios, "base_letter_template")
	# cross-validate with per-letter masks
	mask_dir = os.path.join(repo_root, "config", "bubble_templates")
	_cross_validate_masks(mask_dir, base_ratios)
	# print final constants
	_print_constants_block(base_ratios)
	print("\ndone.")


#============================================
if __name__ == "__main__":
	main()
