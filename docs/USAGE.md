# Usage

Grade Apperson DataLink 1200 bubble sheets from phone photos or flatbed scans. The pipeline registers images, extracts answers, and grades students against a scanned answer key.

All commands use the environment bootstrap: `source source_me.sh && python ...`

## Quick start

Process a directory of scantron images against an answer key:

```bash
source source_me.sh && python run_pipeline.py \
  -i scantrons/ \
  -k scantrons/answer_key.jpg \
  -o data/output/
```

Output appears in `data/output/`: registered images, answer CSVs, and grade CSVs.

## CLI scripts

### run_pipeline.py -- batch pipeline

Process all images in a directory: register, extract, and grade against a key image.

```bash
source source_me.sh && python run_pipeline.py \
  -i scantrons/ \
  -k scantrons/answer_key.jpg \
  -o data/output/ \
  -d
```

| Flag | Description |
| --- | --- |
| `-i`, `--input` | Path to scan image or directory of images (required) |
| `-k`, `--key` | Path to answer key image (required) |
| `-o`, `--output-dir` | Output directory (default: `data/output`) |
| `-t`, `--template` | Path to template YAML (default: `config/dl1200_template.yaml`) |
| `-d`, `--debug` | Save debug overlays for all stages |

### register_scan.py -- image registration

Detect the page boundary in a phone photo or scan, correct perspective, and warp to a canonical rectangle.

```bash
source source_me.sh && python register_scan.py \
  -i scantrons/photo.jpg \
  -o data/output/registered.png
```

| Flag | Description |
| --- | --- |
| `-i`, `--input` | Path to raw scan or phone photo (required) |
| `-o`, `--output` | Path for registered output image (required) |
| `-t`, `--template` | Path to template YAML (default: `config/dl1200_template.yaml`) |
| `-d`, `--debug` | Save debug overlays showing detected contour and bubble grid |

### extract_answers.py -- answer extraction

Extract answers and student ID from a scantron image. Auto-registers raw images unless `-r` is passed.

```bash
source source_me.sh && python extract_answers.py \
  -i scantrons/student.jpg \
  -o data/output/student_answers.csv
```

| Flag | Description |
| --- | --- |
| `-i`, `--input` | Path to image (required) |
| `-o`, `--output` | Path for output CSV (required) |
| `-t`, `--template` | Path to template YAML (default: `config/dl1200_template.yaml`) |
| `-r`, `--registered` | Skip registration (input is already a registered image) |
| `-d`, `--debug` | Save debug overlay with color-coded bubble confidence |

### grade_answers.py -- grading

Compare a student answers CSV to an answer key CSV.

```bash
source source_me.sh && python grade_answers.py \
  -i data/output/student_answers.csv \
  -k data/output/key_answers.csv \
  -o data/output/student_grades.csv
```

| Flag | Description |
| --- | --- |
| `-i`, `--input` | Path to student answers CSV (required) |
| `-k`, `--key` | Path to answer key CSV (required) |
| `-o`, `--output` | Path for graded results CSV (required) |

### migrate_template_to_v2.py -- template migration

Convert a legacy template using `answers.bubble_geometry` into v2 shape-contract
format using `answers.bubble_shape`.

```bash
source source_me.sh && python migrate_template_to_v2.py \
  -i config/dl1200_template.yaml \
  -o config/dl1200_template.v2.yaml
```

| Flag | Description |
| --- | --- |
| `-i`, `--input` | Path to input template YAML (required) |
| `-o`, `--output` | Path for migrated template YAML (required) |

### refine_template_from_empty_fits.py -- template calibration

Learn refined answer-grid coordinates from high-confidence empty bubble fits
across one image or a directory of images.

```bash
source source_me.sh && python refine_template_from_empty_fits.py \
  -i scantrons/ \
  -t config/dl1200_template.yaml \
  -o output/refined_template.yaml
```

| Flag | Description |
| --- | --- |
| `-i`, `--input` | Image file or directory to use for refinement (required) |
| `-o`, `--output` | Path for refined template YAML (required) |
| `-t`, `--template` | Path to base template YAML (default: `config/dl1200_template.yaml`) |
| `-r`, `--registered` | Skip registration (input images already canonical) |
| `--empty-score-max` | Max score treated as empty candidate (default: `0.12`) |
| `--min-samples` | Min samples per bubble before applying correction (default: `2`) |
| `--outlier-radius-px` | Outlier trim radius for per-bubble offsets (default: `6.0`) |

## Examples

Register a single phone photo and extract answers:

```bash
source source_me.sh && python register_scan.py \
  -i scantrons/photo.jpg -o data/output/photo_registered.png -d
source source_me.sh && python extract_answers.py \
  -i data/output/photo_registered.png -o data/output/photo_answers.csv -d
```

Grade one student against the key:

```bash
source source_me.sh && python grade_answers.py \
  -i data/output/photo_answers.csv \
  -k data/output/key_answers.csv \
  -o data/output/photo_grades.csv
```

## Inputs and outputs

- **Inputs**: JPEG or PNG images of Apperson DataLink 1200 forms (phone photos or flatbed scans). See [docs/INPUT_FORMATS.md](INPUT_FORMATS.md) for details.
- **Outputs**: CSV files with extracted answers, confidence scores, and grades. See [docs/OUTPUT_FORMATS.md](OUTPUT_FORMATS.md) for schemas.
- **Debug overlays**: PNG images with color-coded bubble scores, saved with `-d` flag.
- **Output directory**: `data/output/` by default (git-ignored).

## Debug mode

All scripts support `-d`/`--debug` which saves overlay images alongside normal output:

- Contour detection overlay on the raw photo
- Bubble grid overlay on the registered image
- Color-coded answer overlay (green=filled, red=empty, yellow=uncertain)

## Known gaps

- No `--dry-run` flag; all scripts write output files directly.
- No batch summary report across all students (individual grade CSVs only).
