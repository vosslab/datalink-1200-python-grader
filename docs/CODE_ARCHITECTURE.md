# Code architecture

## Overview

Python-based optical mark recognition (OMR) grader for Apperson DataLink 1200 (AccuScan #28040) bubble sheets. Processes phone photos or flatbed scans of 100-question scantron forms, extracts student answers (A-E) and 9-digit student IDs, and grades them against a scanned answer key.

Uses classical image processing (OpenCV thresholding, contour detection, perspective warp) with no ML/AI for bubble scoring.

## Major components

### CLI scripts (repo root)

| Script | Purpose |
| --- | --- |
| [register_scan.py](register_scan.py) | Detect page boundary in a photo, perspective warp to canonical rectangle |
| [extract_answers.py](extract_answers.py) | Load template, score bubbles, extract answers and student ID to CSV |
| [grade_answers.py](grade_answers.py) | Compare student answers CSV to answer key CSV, produce graded results |
| [run_pipeline.py](run_pipeline.py) | Batch pipeline: register, extract, and grade a directory of scantron images |
| [refine_template_from_empty_fits.py](refine_template_from_empty_fits.py) | Fit refined answer-grid coordinates from high-confidence empty bubbles |

### Library package: `omr_utils/`

| Module | Responsibility |
| --- | --- |
| [omr_utils/template_loader.py](omr_utils/template_loader.py) | Load YAML template, migrate v1/v2/v3, compute normalized and pixel bubble coordinates |
| [omr_utils/timing_mark_anchors.py](omr_utils/timing_mark_anchors.py) | Detect left/top timing marks, estimate anchor transform, mark index conversion |
| [omr_utils/image_registration.py](omr_utils/image_registration.py) | Contour detection, corner ordering, perspective warp, orientation detection |
| [omr_utils/bubble_reader.py](omr_utils/bubble_reader.py) | Grayscale intensity bubble scoring, per-question answer extraction |
| [omr_utils/debug_drawing.py](omr_utils/debug_drawing.py) | Debug overlay drawing for answer bubbles (extracted from bubble_reader) |
| [omr_utils/bubble_template_extractor.py](omr_utils/bubble_template_extractor.py) | Extract, average, and store 5X oversize pixel templates for bubble letters A-E |
| [omr_utils/template_matcher.py](omr_utils/template_matcher.py) | Local NCC-based bubble refinement using pixel templates |
| [omr_utils/template_refiner.py](omr_utils/template_refiner.py) | Collect empty-bubble offsets and fit updated answer-grid template params |
| [omr_utils/student_id_reader.py](omr_utils/student_id_reader.py) | 9-digit student ID grid extraction |
| [omr_utils/csv_writer.py](omr_utils/csv_writer.py) | CSV output (answers, confidences, flags) and CSV reading |

### Configuration

| File | Purpose |
| --- | --- |
| [config/dl1200_template.yaml](config/dl1200_template.yaml) | Form geometry: v3 template with timing mark indices for 100 questions and student ID grid |
| `config/bubble_templates/` | 5X oversize grayscale PNG reference templates for bubble letters A-E |

## Data flow

The answer key goes through the same pipeline as student sheets.

```text
Phone photo or flatbed scan (JPEG/PNG)
  |
  v
Image Registration (image_registration.py)
  - Grayscale + blur + adaptive threshold + morphological close
  - Find largest quadrilateral contour (or full-image fallback for flatbed scans)
  - Order corners, perspective warp to canonical rectangle
  - Detect and correct orientation via content density analysis
  - Resize to canonical dimensions (1700x2200)
  |
  v
Template Loading (template_loader.py)
  - Load YAML with timing mark indices (v3) or normalized coords (v2)
  - Migrate to v3: compute both mark indices and coordinates
  - Derive pixel positions for all 500 answer bubbles and 90 ID bubbles
  |
  v
Answer Extraction (bubble_reader.py + student_id_reader.py)
  - Three-stage pipeline: localize rows, measure rows, decide answers
  - Optional NCC template matching refinement (template_matcher.py)
  - Grayscale intensity scoring with dual measurement zones
  - Per-question relative scoring with gap-based BLANK/MULTIPLE detection
  - Student ID: same scoring on 9-digit x 10-value grid
  |
  v
CSV Output (csv_writer.py)
  - Format: student_id, q1..q100, conf1..conf100, flags
  - Confidence = gap between top and second-best choice score
  |
  v
Grading (grade_answers.py)
  - Compare student CSV to answer key CSV
  - Output: raw score, total questions, percentage, per-question results
  - Track low-confidence answers (gap < 0.05)
```

## Key design decisions

- **Timing mark indices**: template v3 stores bubble positions as fractional timing mark indices, expressing the structural relationship between bubbles and the timing mark grid. Positions are invariant across scans of the same form model.
- **Minimal shape contract**: template v2/v3 stores bubble shape as aspect ratio + target area at canonical size; runtime geometry is derived from those values.
- **Grid-based template**: approximately 10 reference measurements define all 500 bubble positions. Only the first bubble position and spacing are needed per column.
- **Anchor-relative correction**: timing marks on the left and top edges provide a conservative anchor transform used during bubble localization when confidence is high.
- **Pixel template matching**: cryoEM-inspired class-averaged pixel templates of printed bubble letters enable NCC-based local refinement for sub-pixel bubble localization.
- **Grayscale scoring over binary**: bubble fill is measured from grayscale intensity, not binary thresholds. This naturally distinguishes filled pencil marks from printed bubble outlines.
- **Ring-based baseline**: each bubble score is relative to its local surrounding ring, compensating for uneven lighting and paper tone.
- **Answer key as scanned sheet**: the key sheet goes through the same extraction pipeline, avoiding manual answer entry.

## Testing and verification

Tests live in [tests/](tests/) and run with pytest:

```bash
source source_me.sh && python -m pytest tests/ -v
```

- **Unit tests**: template loading, bubble scoring on synthetic images, grading logic, CSV round-trip
- **Integration tests**: real scantron image registration and answer extraction (skipped if `scantrons/` directory is absent)
- **Smoke tests**: end-to-end pipeline from raw image to graded results
- **Repo hygiene**: pyflakes lint, indentation, ASCII compliance, import checks

## Extension points

- **New form types**: create a new YAML template in [config/](config/) with the form's bubble geometry. The pipeline reads coordinates from the template at runtime.
- **New scoring algorithms**: replace or extend `score_bubble_fast()` in [omr_utils/bubble_reader.py](omr_utils/bubble_reader.py).
- **Additional output formats**: extend [omr_utils/csv_writer.py](omr_utils/csv_writer.py) or add new writer modules.
- **Student ID formats**: adjust digit count and grid positions in the YAML template.

## Known gaps

- Student ID extraction produces incorrect results on some images; the ID grid coordinates may need further calibration.
- Phone photos with extreme perspective or poor lighting may fail contour detection; the full-image fallback works but skips perspective correction.
