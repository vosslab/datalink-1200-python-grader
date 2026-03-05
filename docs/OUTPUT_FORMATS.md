# Output formats

## Answers CSV

Produced by [extract_answers.py](../extract_answers.py) and the batch pipeline. One row per scantron image.

### Header

```text
student_id,q1,q2,...,q100,conf1,conf2,...,conf100,flags
```

### Columns

| Column | Type | Description |
| --- | --- | --- |
| `student_id` | string | 9-digit student ID from the bubble grid |
| `q1` through `q100` | string | Detected answer: A, B, C, D, E, or blank |
| `conf1` through `conf100` | float | Confidence score (gap between top and second-best bubble score) |
| `flags` | string | Space-separated flags, e.g., `q5:BLANK q47:MULTIPLE` |

### Flag values

| Flag | Meaning |
| --- | --- |
| `BLANK` | No bubble filled above threshold for this question |
| `MULTIPLE` | Two or more bubbles filled; highest-scoring choice is reported |

### Example

```csv
student_id,q1,q2,q3,...,conf1,conf2,conf3,...,flags
123456789,B,A,D,...,0.180,0.150,0.200,...,q5:BLANK
```

## Graded CSV

Produced by [grade_answers.py](../grade_answers.py) and the batch pipeline. One row per student.

### Header

```text
student_id,raw_score,total,percentage,q1,q2,...,q100,low_confidence,flags
```

### Columns

| Column | Type | Description |
| --- | --- | --- |
| `student_id` | string | Student ID |
| `raw_score` | integer | Number of correct answers |
| `total` | integer | Number of graded questions (key blanks excluded) |
| `percentage` | float | Score as percentage (0.0-100.0) |
| `q1` through `q100` | string | `1` = correct, `0` = incorrect, blank = not graded (key blank) |
| `low_confidence` | string | Space-separated list of low-confidence questions, e.g., `q12 q45` |
| `flags` | string | Extraction flags from the original answers CSV |

### Example

```csv
student_id,raw_score,total,percentage,q1,q2,...,low_confidence,flags
123456789,47,68,69.1,1,1,0,...,q12 q45,
```

## Registered images

When using `-d`/`--debug` or the batch pipeline, registered images are saved as PNG files in the output directory:

- `{basename}_registered.png` -- perspective-corrected image at canonical dimensions (1700x2200)
- `{basename}_debug.png` -- debug overlay with color-coded bubble scores
- `{basename}_contour_debug.png` -- raw image with detected page boundary
- `{basename}_grid_debug.png` -- registered image with bubble grid overlay
