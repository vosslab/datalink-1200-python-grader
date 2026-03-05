# Input formats

## Supported image formats

The pipeline accepts any image format readable by OpenCV:

| Format | Extensions | Notes |
| --- | --- | --- |
| JPEG | `.jpg`, `.jpeg` | Most common for phone photos |
| PNG | `.png` | Lossless, good for registered images |
| TIFF | `.tif`, `.tiff` | High quality scans |
| BMP | `.bmp` | Uncompressed |

## Supported form

Apperson DataLink 1200 / AccuScan #28040 bubble sheet:

- 100 questions (1-50 left column, 51-100 right column)
- 5 choices per question (A, B, C, D, E)
- 9-digit student ID grid
- Portrait orientation when correctly oriented

## Image sources

### Phone photos

Phone photos are the primary expected input. The pipeline handles:

- Perspective distortion from angled photos
- Variable lighting and color cast
- Background clutter around the form

For best results:

- Fill most of the frame with the scantron form
- Ensure all four corners of the form are visible
- Avoid heavy shadows across the form
- Use adequate lighting

### Flatbed scans

Flatbed scans work well and typically need less correction. The pipeline detects that the scan fills the full image and skips perspective correction.

Recommended scan settings:

- 200-400 DPI (higher is not needed)
- Grayscale or color
- JPEG or PNG output

## Answer key

The answer key is a filled-in scantron form processed through the same pipeline as student sheets. No manual YAML entry is needed. Fill in the correct answers on a blank form, scan or photograph it, and provide it as the `-k`/`--key` argument.

## Batch input

When using [run_pipeline.py](../run_pipeline.py) with a directory path, it collects all image files matching the supported extensions (case-insensitive). The answer key image is automatically excluded from the student list.
