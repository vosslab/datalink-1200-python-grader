#!/usr/bin/env python3
"""Refine answer-grid template coordinates from empty-bubble fits."""

# Standard Library
import os
import argparse

# PIP3 modules
import yaml

# local repo modules
import omr_utils.template_loader
import omr_utils.template_refiner


#============================================
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Refine template answer-grid coordinates from empty bubbles"
	)
	parser.add_argument(
		"-i", "--input", dest="input_path", required=True,
		help="Image file or directory to use for refinement"
	)
	parser.add_argument(
		"-o", "--output", dest="output_path", required=True,
		help="Path to write refined template YAML"
	)
	parser.add_argument(
		"-t", "--template", dest="template_path", default=None,
		help="Path to base template YAML (default: config/dl1200_template.yaml)"
	)
	parser.add_argument(
		"-r", "--registered", dest="registered", action="store_true",
		help="Input images are already registered at canonical size"
	)
	parser.add_argument(
		"--empty-score-max", dest="empty_score_max", type=float, default=0.12,
		help="Maximum score to treat a bubble as empty candidate (default: 0.12)"
	)
	parser.add_argument(
		"--min-samples", dest="min_samples", type=int, default=2,
		help="Minimum samples per bubble before applying correction (default: 2)"
	)
	parser.add_argument(
		"--outlier-radius-px", dest="outlier_radius_px", type=float, default=6.0,
		help="Trim offsets farther than this radius from median (default: 6.0)"
	)
	args = parser.parse_args()
	return args


#============================================
def get_default_template_path() -> str:
	"""Return absolute default template path."""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	return os.path.join(script_dir, "config", "dl1200_template.yaml")


#============================================
def main() -> None:
	"""Run template refinement and write the refined YAML file."""
	args = parse_args()
	template_path = args.template_path
	if template_path is None:
		template_path = get_default_template_path()
	template = omr_utils.template_loader.load_template(template_path)
	image_paths = omr_utils.template_refiner.collect_image_paths(args.input_path)
	if not image_paths:
		raise RuntimeError("no images found for refinement")
	refined_template, report = omr_utils.template_refiner.refine_template_from_images(
		template,
		image_paths,
		registered=args.registered,
		empty_score_max=args.empty_score_max,
		min_samples=max(1, int(args.min_samples)),
		outlier_radius_px=max(1.0, float(args.outlier_radius_px)),
	)
	output_dir = os.path.dirname(args.output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(args.output_path, "w") as fh:
		yaml.safe_dump(refined_template, fh, sort_keys=False)
	print(f"images: {report['image_count']}")
	print(f"raw offsets: {report['raw_offset_count']}")
	print(f"refined bubbles: {report['aggregated_bubbles']}")
	print(f"wrote: {args.output_path}")


#============================================
if __name__ == "__main__":
	main()
