#!/usr/bin/env python3
"""Offline side script to build per-letter bubble templates from scanned images.

Two-pass pipeline:
  Pass 1 - ROI extraction: load each scan, run anchor transform and
    SlotMap to extract all bubble ROIs grouped by letter (A-E).
  Pass 2 - Template construction: align ROIs, reject outliers, build
    averaged templates.

Usage:
  source source_me.sh && python tools/build_bubble_templates.py \
    --input-dir scans/ --output-dir output_bubble_templates/
"""

# Standard Library
import os
import json
import time
import glob
import subprocess
import argparse

# PIP3 modules
import cv2
import numpy

# local repo modules
import omr_utils.bubble_template_extractor
import omr_utils.slot_map
import omr_utils.template_builder
import omr_utils.template_loader
import omr_utils.timing_mark_anchors


#============================================
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Build per-letter bubble templates from scanned images"
	)
	parser.add_argument(
		"-i", "--input-dir", dest="input_dir", required=True,
		help="Directory containing scan images (PNG/JPG)"
	)
	parser.add_argument(
		"-o", "--output-dir", dest="output_dir", required=True,
		help="Directory for ROI output and QC images"
	)
	parser.add_argument(
		"-t", "--template", dest="template_file", default=None,
		help="YAML template file (default: config/dl1200_template.yaml)"
	)
	parser.add_argument(
		"-n", "--dry-run", dest="dry_run", action="store_true",
		help="Only extract ROIs, skip template construction"
	)
	parser.set_defaults(dry_run=False)
	args = parser.parse_args()
	return args


#============================================
def _find_scan_images(input_dir: str) -> list:
	"""Find all image files in the input directory.

	Args:
		input_dir: directory to search

	Returns:
		sorted list of image file paths
	"""
	extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
	paths = []
	for ext in extensions:
		paths.extend(glob.glob(os.path.join(input_dir, ext)))
		# also check uppercase
		paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
	paths = sorted(set(paths))
	return paths


#============================================
def _extract_rois_from_scan(image_path: str, template: dict,
	output_dir: str, metadata: list,
	reject_fraction: float = 0.2) -> dict:
	"""Extract bubble ROIs from all slots in a single scan image.

	Iterates all question/choice slots via SlotMap directly, without
	filtering by fill state. The printed bracket shape is independent
	of whether a bubble is filled, so all slots are valid for template
	construction.

	Args:
		image_path: path to the scan image
		template: loaded YAML template dict
		output_dir: base output directory for ROI images
		metadata: list to append metadata entries to

	Returns:
		dict mapping letter to list of ROI arrays
	"""
	image = cv2.imread(image_path)
	if image is None:
		print(f"  WARNING: could not read {image_path}")
		return {}
	scan_id = os.path.splitext(os.path.basename(image_path))[0]
	print(f"  processing scan: {scan_id}")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	# normalize full image contrast (1% black / 25% white stretch)
	gray = omr_utils.bubble_template_extractor.normalize_roi_percentile(gray)
	# derive geometry from timing marks so ROI size matches the scan
	raw_transform = omr_utils.timing_mark_anchors.estimate_anchor_transform(
		gray, template)
	try:
		slot_map = omr_utils.slot_map.SlotMap(raw_transform, template)
		measure_cfg = slot_map.measure_cfg()
	except ValueError as exc:
		print(f"    SKIP: {exc}")
		return {}
	print(f"    anchor geometry: row_pitch={measure_cfg['row_pitch']:.1f}px"
		f" col_pitch={measure_cfg['col_pitch']:.1f}px")
	# iterate all question/choice slots directly via SlotMap
	choices = template["answers"]["choices"]
	num_questions = int(template["answers"]["num_questions"])
	rois_by_letter = {c: [] for c in choices}
	# diagnostic counters
	attempted = 0
	extracted = 0
	skipped_bounds = 0
	# collect raw and normalized ROIs and their metadata entries per letter
	raw_rois_by_letter = {c: [] for c in choices}
	norm_rois_by_letter = {c: [] for c in choices}
	meta_entries_by_letter = {c: [] for c in choices}
	for q_num in range(1, num_questions + 1):
		for choice in choices:
			attempted += 1
			top_y, bot_y, left_x, right_x = slot_map.roi_bounds(q_num, choice)
			roi = omr_utils.bubble_template_extractor.extract_roi_from_bounds(
				gray, left_x, top_y, right_x, bot_y)
			if roi is None:
				skipped_bounds += 1
				continue
			extracted += 1
			# normalize ROI for statistical stages
			norm_roi = omr_utils.bubble_template_extractor.normalize_roi_percentile(roi)
			# save raw ROI to letter subdirectory
			letter_dir = os.path.join(output_dir, choice)
			os.makedirs(letter_dir, exist_ok=True)
			roi_filename = f"{scan_id}_q{q_num:03d}_{choice}.png"
			roi_path = os.path.join(letter_dir, roi_filename)
			cv2.imwrite(roi_path, roi)
			raw_rois_by_letter[choice].append(roi)
			norm_rois_by_letter[choice].append(norm_roi)
			# build metadata entry
			cx, cy = slot_map.center(q_num, choice)
			meta_entries_by_letter[choice].append({
				"scan_id": scan_id,
				"question": q_num,
				"choice": choice,
				"center_x": int(cx),
				"center_y": int(cy),
				"roi_path": roi_path,
				"mean_grayscale": float(numpy.mean(roi)),
				"mean_grayscale_normalized": float(numpy.mean(norm_roi)),
				"rejected_dark": False,
			})
	print(f"    slots: {attempted} attempted, {extracted} extracted, "
		f"{skipped_bounds} bounds-skip")
	# apply per-image per-letter darkness filter using normalized ROIs
	for letter in sorted(raw_rois_by_letter.keys()):
		raw_list = raw_rois_by_letter[letter]
		norm_list = norm_rois_by_letter[letter]
		entries = meta_entries_by_letter[letter]
		if not raw_list:
			continue
		# run darkness filter on normalized ROIs for exposure independence
		norm_kept, norm_rejected, means, cutoff = (
			omr_utils.template_builder._filter_dark_rois(
				norm_list, reject_fraction))
		# compute summary statistics for console output
		means_arr = numpy.array(means)
		min_mean = float(numpy.min(means_arr))
		med_mean = float(numpy.median(means_arr))
		max_mean = float(numpy.max(means_arr))
		n_rejected = len(norm_rejected)
		n_kept = len(norm_kept)
		# console output per image-letter group
		print(f"  image {scan_id} letter {letter}:"
			f"  raw={len(raw_list)}"
			f"  mean intensity: min={min_mean:.1f}"
			f" median={med_mean:.1f} max={max_mean:.1f}")
		print(f"    cutoff({reject_fraction*100:.0f}%)={cutoff:.1f}"
			f"  kept={n_kept} rejected_dark={n_rejected}")
		# identify which indices were rejected so we can sync raw lists
		reject_set = set()
		sorted_indices = sorted(range(len(norm_list)),
			key=lambda k: means[k])
		for idx in sorted_indices[:n_rejected]:
			reject_set.add(idx)
		# mark rejected entries in metadata
		for idx in range(len(entries)):
			entries[idx]["mean_grayscale"] = float(numpy.mean(raw_list[idx]))
			entries[idx]["rejected_dark"] = idx in reject_set
		# extend metadata list
		metadata.extend(entries)
		# build kept lists for raw (QC audit) and normalized (downstream)
		raw_kept = [raw_list[i] for i in range(len(raw_list))
			if i not in reject_set]
		norm_kept_synced = [norm_list[i] for i in range(len(norm_list))
			if i not in reject_set]
		# save QC images with both raw and normalized montages
		omr_utils.template_builder._save_filter_qc(
			raw_list, raw_kept, scan_id, letter, output_dir,
			norm_rois_before=norm_list,
			norm_rois_after=norm_kept_synced)
		# store normalized kept ROIs for downstream template building
		rois_by_letter[letter] = norm_kept_synced
	return rois_by_letter


#============================================
def _load_base_reference(repo_root: str) -> numpy.ndarray:
	"""Load the base letter template for pass-1 alignment anchoring.

	The base template provides a common alignment reference so that
	bracket positions are consistent across all per-letter templates.

	Args:
		repo_root: repository root directory

	Returns:
		grayscale numpy array at 480x88

	Raises:
		FileNotFoundError: if base template file does not exist
		ValueError: if base template is not 480x88 grayscale
	"""
	base_path = os.path.join(repo_root, "artifacts",
		"base_letter_template.png")
	if not os.path.isfile(base_path):
		raise FileNotFoundError(
			f"base reference not found: {base_path}")
	base_img = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
	if base_img is None:
		raise ValueError(f"could not read base reference: {base_path}")
	# validate dimensions match canonical template size
	expected_h = omr_utils.template_builder.CANONICAL_TEMPLATE_HEIGHT
	expected_w = omr_utils.template_builder.CANONICAL_TEMPLATE_WIDTH
	if base_img.shape[0] != expected_h or base_img.shape[1] != expected_w:
		raise ValueError(
			f"base reference must be {expected_w}x{expected_h}, "
			f"got {base_img.shape[1]}x{base_img.shape[0]}")
	print(f"loaded base reference: {base_path}"
		f" ({expected_w}x{expected_h} grayscale)")
	return base_img


#============================================
def _build_templates(all_rois: dict, output_dir: str,
	repo_root: str) -> None:
	"""Build per-letter templates from collected ROIs.

	Args:
		all_rois: dict mapping letter to list of ROI arrays
		output_dir: base output directory
		repo_root: repository root for saving final templates
	"""
	# load base reference for pass-1 alignment anchoring
	base_reference = _load_base_reference(repo_root)
	# target directory for final templates
	template_out = os.path.join(repo_root, "config", "bubble_templates")
	all_templates = {}
	for letter in sorted(all_rois.keys()):
		rois = all_rois[letter]
		print(f"  letter {letter}: {len(rois)} ROIs collected")
		if len(rois) < 6:
			print(f"    SKIP: not enough ROIs for letter {letter}")
			continue
		# build template with two-pass alignment and symmetry
		t_start = time.time()
		template, alignment_table = (
			omr_utils.template_builder._build_letter_template(
				rois, letter, output_dir=output_dir,
				base_reference=base_reference))
		t_elapsed = time.time() - t_start
		if template is None:
			print(f"    SKIP: template construction failed for {letter}")
			continue
		# count kept/rejected
		kept = sum(1 for e in alignment_table if e["kept"])
		rejected = len(alignment_table) - kept
		print(f"    aligned: {kept} kept, {rejected} rejected"
			f" ({t_elapsed:.1f}s)")
		all_templates[letter] = template
	# save final templates
	if all_templates:
		saved = omr_utils.bubble_template_extractor.save_templates(
			all_templates, template_out)
		print(f"\nsaved {len(saved)} template files to {template_out}/")
	else:
		print("\nWARNING: no templates were built")


#============================================
def main() -> None:
	"""Run the two-pass template building pipeline."""
	args = parse_args()
	repo_root = subprocess.check_output(
		["git", "rev-parse", "--show-toplevel"], text=True).strip()
	# validate input directory
	if not os.path.isdir(args.input_dir):
		raise FileNotFoundError(f"input directory not found: {args.input_dir}")
	# find scan images
	image_paths = _find_scan_images(args.input_dir)
	if not image_paths:
		raise FileNotFoundError(
			f"no image files found in {args.input_dir}")
	print(f"found {len(image_paths)} scan images")
	# load YAML template
	if args.template_file is not None:
		template_path = args.template_file
	else:
		# default: config/dl1200_template.yaml relative to repo root
		template_path = os.path.join(repo_root, "config",
			"dl1200_template.yaml")
	template = omr_utils.template_loader.load_template(template_path)
	# Pass 1: extract ROIs from each scan
	print("\n=== Pass 1: ROI extraction ===")
	os.makedirs(args.output_dir, exist_ok=True)
	metadata = []
	all_rois = {}
	for image_path in image_paths:
		scan_rois = _extract_rois_from_scan(
			image_path, template, args.output_dir, metadata)
		for letter, rois in scan_rois.items():
			if letter not in all_rois:
				all_rois[letter] = []
			all_rois[letter].extend(rois)
	# save metadata
	metadata_path = os.path.join(args.output_dir, "metadata.json")
	with open(metadata_path, "w") as fh:
		json.dump(metadata, fh, indent=2)
	print(f"\nsaved metadata for {len(metadata)} ROIs to {metadata_path}")
	# summary
	total_rois = 0
	for letter in sorted(all_rois.keys()):
		count = len(all_rois[letter])
		total_rois += count
		print(f"  {letter}: {count} ROIs")
	print(f"  total: {total_rois} ROIs from {len(image_paths)} scans")
	if args.dry_run:
		print("\ndry run: skipping template construction")
		return
	# Pass 2: build templates
	print("\n=== Pass 2: template construction ===")
	_build_templates(all_rois, args.output_dir, repo_root)
	print("\ndone.")


#============================================
if __name__ == "__main__":
	main()
