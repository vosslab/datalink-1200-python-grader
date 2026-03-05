# Changelog

## 2026-03-05

### Additions and New Features

- Created full OMR pipeline for Apperson DataLink 1200 bubble sheets
- Added `tests/conftest.py` using `git_file_utils.get_repo_root()` so `pytest tests/` works without `source source_me.sh`
- Added [config/dl1200_template.yaml](../config/dl1200_template.yaml) with calibrated form geometry using normalized coordinates (0.0-1.0)
- Added [omr_utils/template_loader.py](../omr_utils/template_loader.py) for YAML template loading and coordinate computation
- Added [omr_utils/image_registration.py](../omr_utils/image_registration.py) for page detection, perspective warp, and orientation correction
- Added [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) with grayscale intensity-based bubble scoring and gap-based BLANK/MULTIPLE detection
- Added [omr_utils/student_id_reader.py](../omr_utils/student_id_reader.py) for 9-digit student ID grid reading
- Added [omr_utils/csv_writer.py](../omr_utils/csv_writer.py) for answers CSV output with per-question confidence scores
- Added [register_scan.py](../register_scan.py) CLI for image registration
- Added [extract_answers.py](../extract_answers.py) CLI for answer extraction (auto-registers raw images)
- Added [grade_answers.py](../grade_answers.py) CLI for grading student answers against a key
- Added [run_pipeline.py](../run_pipeline.py) batch CLI for end-to-end processing of image directories
- Added [pyproject.toml](../pyproject.toml) with project metadata and dependencies
- Added [VERSION](../VERSION) file (26.03)
- Added test suite: template loader, image registration, bubble reader, grading, and pipeline smoke tests
- Added documentation: CODE_ARCHITECTURE, FILE_STRUCTURE, INSTALL, USAGE, INPUT_FORMATS, OUTPUT_FORMATS, TROUBLESHOOTING
- Updated [README.md](../README.md) with project purpose, quick start, and documentation links

### Behavior or Interface Changes

- Replaced background-relative bubble scoring with self-referencing scoring in `omr_utils/bubble_reader.py`; each question's lightest choice serves as the empty baseline, eliminating dependency on inter-row background strips that failed on phone photos with uneven lighting
- Added adaptive blank detection: the spread (max - min edge mean) across all 100 questions is sorted and the largest gap between consecutive values separates filled from blank populations automatically per image
- Removed `blank_gap` parameter from `read_answers()` (no longer needed; threshold is computed adaptively)
- Removed module-level `BG_GAP` and `BG_HEIGHT` constants (background strips no longer used in `read_answers`)

### Decisions and Failures

- Chose grayscale intensity scoring over binary thresholding because adaptive threshold could not distinguish filled bubbles from printed bubble outlines
- Right column bubble x-positions were initially offset by one choice position; corrected after debug overlay analysis
- Student ID extraction produces incorrect results on some images; grid coordinates need further calibration
- Fixed 8B5D phone photo missing 10 answers (Q32, Q42-50): background reference strips overlapped adjacent rows in lower-left area due to slight registration drift; self-referencing scoring eliminated this failure mode
- Fixed 43F key false positives from DataLink machine-printed marks: adaptive threshold correctly separates blank rows (spread <35px) from filled rows (spread >58px) regardless of machine marks biasing the D column

### Developer Tests and Notes

- 60 tests pass across 5 test files (template loader, image registration, bubble reader, grading, pipeline smoke)
- Tests requiring scantron images skip automatically when `scantrons/` is absent
- Pipeline verified on 4 real scantron images: 1 answer key + 3 student sheets (2 flatbed scans, 1 phone photo)
