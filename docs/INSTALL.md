# Install

This repo is a collection of CLI scripts run from the repo root. "Installed" means dependencies are available and `source source_me.sh && python run_pipeline.py --help` works.

## Requirements

- Python 3.12+
- pip package manager
- Bash shell (the [source_me.sh](../source_me.sh) bootstrap requires Bash, not Zsh)

## Install steps

Clone the repository:

```bash
git clone <repo-url>
cd datalink-1200-python-grader
```

Install runtime dependencies:

```bash
pip install -r pip_requirements.txt
```

Install development dependencies (pytest, pyflakes, etc.):

```bash
pip install -r pip_requirements-dev.txt
```

### Runtime dependencies

Listed in [pip_requirements.txt](../pip_requirements.txt):

| Package | Purpose |
| --- | --- |
| numpy | Array operations for image data |
| opencv-python | Image processing, contour detection, perspective warp |
| pillow | Image I/O support |
| pyyaml | YAML template loading |
| scipy | Scientific computing utilities |

## Environment bootstrap

All commands use the bootstrap script to configure the Python environment:

```bash
source source_me.sh && python run_pipeline.py --help
```

The bootstrap sets `PYTHONUNBUFFERED=1` and `PYTHONDONTWRITEBYTECODE=1`.

## Verify install

Run the test suite to confirm dependencies and modules load correctly:

```bash
source source_me.sh && python -m pytest tests/test_template_loader.py -q
```

This runs template loading tests that do not require scantron images. To run all tests:

```bash
source source_me.sh && python -m pytest tests/ -v
```

Tests that require scantron images in `scantrons/` skip automatically when the directory is absent.

## Known gaps

- No `setup.py` or pip-installable package; scripts are run directly from the repo root.
- The `scantrons/` directory with test images is git-ignored and must be provided separately.
