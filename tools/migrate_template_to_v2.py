#!/usr/bin/env python3
"""Migrate a DataLink template YAML from v1 fields to v2 shape contract."""

# Standard Library
import argparse
import os

# PIP3 modules
import yaml

# local repo modules
import omr_utils.template_loader


#============================================
def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Migrate a template YAML file to template_version=2"
	)
	parser.add_argument(
		"-i", "--input", dest="input_file", required=True,
		help="Input YAML template path"
	)
	parser.add_argument(
		"-o", "--output", dest="output_file", required=True,
		help="Output YAML template path"
	)
	args = parser.parse_args()
	return args


#============================================
def main() -> None:
	"""Run template migration."""
	args = parse_args()
	if not os.path.isfile(args.input_file):
		raise FileNotFoundError(f"template not found: {args.input_file}")
	with open(args.input_file, "r") as fh:
		template = yaml.safe_load(fh)
	migrated = omr_utils.template_loader.migrate_template_to_v2(template)
	output_dir = os.path.dirname(args.output_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(args.output_file, "w") as fh:
		yaml.safe_dump(migrated, fh, sort_keys=False)
	print(f"wrote migrated template: {args.output_file}")


#============================================
if __name__ == "__main__":
	main()
