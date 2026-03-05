# Troubleshooting

## Image registration fails

### Symptom: "no suitable page contour found"

The pipeline could not detect a quadrilateral boundary for the scantron form.

- Ensure all four corners of the form are visible in the photo
- Improve lighting to increase contrast between the form and background
- Place the form on a dark surface for better edge detection
- For flatbed scans, the pipeline uses a full-image fallback automatically

### Symptom: registered image is rotated or upside down

Orientation detection uses content density analysis. If the form is misordered:

- Run with `-d` to check the debug overlay
- Verify that the form header (student ID area) appears at the top of the registered image
- Flatbed scans with no visible page boundary may not orient correctly if the scan is pre-rotated

## Answer extraction issues

### Symptom: too many BLANK detections

The bubble fill scoring uses a gap-based threshold. If most questions show BLANK:

- The registered image may be misaligned. Run with `-d` and verify the grid overlay matches actual bubble positions.
- The template coordinates may need recalibration for a different form printing batch.

### Symptom: too many MULTIPLE detections

Two or more bubbles score similarly high for the same question.

- Erasure residue can cause ghost marks. The ring-based scoring helps, but heavy erasures may still trigger MULTIPLE.
- Phone photos with uneven lighting can cause false positives. Use more uniform lighting.
- Check the debug overlay to verify bubble alignment.

### Symptom: wrong answers detected

- Run with `-d` to see per-bubble confidence in the debug overlay
- Green circles = detected answer, red = empty, yellow = uncertain
- If bubbles are offset from the overlay circles, the template coordinates need adjustment in [config/dl1200_template.yaml](../config/dl1200_template.yaml)

## Student ID issues

### Symptom: incorrect student ID

Student ID extraction uses the same bubble scoring as answers but on a 9-digit x 10-value grid. The grid coordinates may need calibration adjustments.

- Run `extract_answers.py` with `-d` to check the grid overlay on the student ID area
- Compare the overlay positions to the actual bubble grid

## Grading issues

### Symptom: unexpected blanks in answer key

If the answer key CSV has blank answers for questions that should have answers:

- Run the key image through `extract_answers.py -d` and inspect the debug overlay
- Lightly filled bubbles or pen (instead of pencil) may score below the detection threshold
- Re-fill the key sheet with darker marks and re-scan

### Symptom: low scores for all students

- Verify the answer key CSV has the correct answers for each question
- Check for systematic misalignment: if all answers are shifted by one position, the template Y-coordinate calibration may be off

## Debug mode

All CLI scripts support `-d`/`--debug` to save diagnostic overlays:

```bash
source source_me.sh && python register_scan.py -i photo.jpg -o registered.png -d
source source_me.sh && python extract_answers.py -i registered.png -o answers.csv -d
```

Debug files are saved alongside the output with `_debug`, `_contour_debug`, or `_grid_debug` suffixes.

## Test suite

Run focused tests to diagnose specific issues:

```bash
# all tests
source source_me.sh && python -m pytest tests/ -v

# template coordinate tests only
source source_me.sh && python -m pytest tests/test_template_loader.py -v

# bubble reader tests (requires scantrons/)
source source_me.sh && python -m pytest tests/test_bubble_reader.py -v

# grading logic only (no images needed)
source source_me.sh && python -m pytest tests/test_grade_answers.py -v
```
