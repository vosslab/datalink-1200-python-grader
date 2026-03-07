# OMR geometry contract

DataLink-1200 grader geometry rules. All code that reads or generates
answer positions must obey this contract. No component may bypass it.

## 1. No fixed geometry

The system must not assume fixed pixel distances, DPI, column widths,
bubble spacing, page offsets, or image size. All geometry must be
derived dynamically from timing marks.

Forbidden:

```python
x = 234
bubble_width = 18
column_pitch = 30
```

## 2. Only timing marks define geometry

Two timing mark families establish the coordinate axes:

| Family | Axis | Produces |
| --- | --- | --- |
| Left marks | vertical | `y0`, `row_pitch` |
| Top marks | horizontal | `x0`, `col_pitch` |

All bubble positions derive from these. Nothing else.

## 3. Local 15-column lattice

Columns follow the logical answer layout:

| Index | Role | Index | Role |
| --- | --- | --- | --- |
| 0 | Q# left | 7 | Q# right |
| 1 | A | 8 | A |
| 2 | B | 9 | B |
| 3 | C | 10 | C |
| 4 | D | 11 | D |
| 5 | E | 12 | E |
| 6 | gap | 13-14 | margin |

`choice_columns` in the YAML refers only to this local lattice.
No global 53-grid or sparse index scheme is permitted.

## 4. Pixel mapping

Pixel coordinates are computed only by:

```
x = x0 + column_index * col_pitch
y = y0 + row_index * row_pitch
```

No other geometry formula is allowed for answer placement.

## 5. SlotMap is the single authority

`SlotMap` is the only module that maps `(question_number, choice_letter)`
to `(pixel_x, pixel_y)` for answer bubbles and `(digit_index, value)`
to `(pixel_x, pixel_y)` for student ID bubbles. No other module may
compute answer or student ID coordinates. All geometry must pass
through SlotMap.

Student ID geometry uses the same horizontal lattice as answer bubbles.
Vertical positions come from `left_id_marks` (10 marks with `center_y`).
`sid_roi_bounds(digit_idx, value)` is the primary geometry method for
student ID scoring. `sid_center(digit_idx, value)` exists only for
debug overlay and labeling.

## 6. ROI construction

Bubble ROIs derive from lattice midpoints:

```
left_x  = midpoint(column - 1, column)
right_x = midpoint(column, column + 1)
top_y   = midpoint(row - 1, row)
bot_y   = midpoint(row, row + 1)
```

ROI size must never be fixed in pixels.

## 7. DPI independence

The system must work identically for 200, 300, 400, 600 DPI scans
and phone camera images. All pixel distances derive from `row_pitch`
and `col_pitch`. No thresholds may depend on absolute pixel size.

## 8. Debug verification

Every run must be able to generate overlays showing:

- Detected timing marks
- Lattice column lines (0-14)
- Lattice row lines
- Bubble ROI rectangles

If column lines do not align with Q#, A-E, gap, the bug is in
timing marks or SlotMap, not the YAML.

## 9. Prohibited practices

- Hard-coded pixel constants
- Hard-coded bubble widths
- Manual offsets added to bubble centers
- Geometry derived from template images
- Global column index systems not matching the 15-column lattice
- Reinterpreting local lattice indices back into a hidden global grid

## 10. Agent compliance

Any agent modifying the grader must ensure:

- Geometry changes respect this contract
- SlotMap remains the single source of answer coordinates
- No fixed geometry is introduced

Any patch violating this contract must be rejected.
