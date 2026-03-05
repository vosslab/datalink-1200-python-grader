# File structure

## Top-level layout

```text
datalink-1200-python-grader/
+- register_scan.py          # CLI: raw photo -> registered image
+- extract_answers.py        # CLI: registered image -> answers CSV
+- grade_answers.py          # CLI: student CSV + key CSV -> graded CSV
+- run_pipeline.py           # CLI: batch register + extract + grade
+- omr_utils/                # shared OMR library package
+- config/                   # form geometry templates (YAML)
+- tests/                    # pytest test suite
+- docs/                     # project documentation
+- artifacts/                # reference images and research notes
+- devel/                    # developer scripts
+- scantrons/                # input scantron images (git-ignored)
+- data/                     # pipeline output directory (git-ignored)
+- output_smoke/             # smoke test output (git-ignored)
+- pyproject.toml            # package metadata and dependencies
+- pip_requirements.txt      # runtime Python dependencies
+- pip_requirements-dev.txt  # developer Python dependencies
+- source_me.sh              # environment bootstrap script
+- VERSION                   # version string (synced with pyproject.toml)
+- CLAUDE.md                 # AI agent instructions
+- AGENTS.md                 # coding style and environment rules
+- README.md                 # project overview and quick start
+- LICENSE.MIT               # license file
```

## Key subtrees

### [omr_utils/](omr_utils/) -- OMR library

| File | Purpose |
| --- | --- |
| [omr_utils/__init__.py](omr_utils/__init__.py) | Empty package marker (docstring only) |
| [omr_utils/template_loader.py](omr_utils/template_loader.py) | Load YAML template, compute bubble coordinates |
| [omr_utils/image_registration.py](omr_utils/image_registration.py) | Page detection, perspective warp, orientation fix |
| [omr_utils/bubble_reader.py](omr_utils/bubble_reader.py) | Bubble fill scoring and answer extraction |
| [omr_utils/student_id_reader.py](omr_utils/student_id_reader.py) | Student ID grid reading |
| [omr_utils/csv_writer.py](omr_utils/csv_writer.py) | CSV output and input for answers |

### [config/](config/) -- templates

| File | Purpose |
| --- | --- |
| [config/dl1200_template.yaml](config/dl1200_template.yaml) | Apperson DataLink 1200 form geometry (normalized coordinates) |

### [tests/](tests/) -- test suite

| File | Purpose |
| --- | --- |
| [tests/test_template_loader.py](tests/test_template_loader.py) | Template loading and coordinate computation |
| [tests/test_image_registration.py](tests/test_image_registration.py) | Image registration with real scantron images |
| [tests/test_bubble_reader.py](tests/test_bubble_reader.py) | Bubble scoring and answer extraction |
| [tests/test_grade_answers.py](tests/test_grade_answers.py) | Grading logic and CSV round-trip |
| [tests/test_pipeline_smoke.py](tests/test_pipeline_smoke.py) | End-to-end pipeline smoke tests |
| [tests/test_pyflakes_code_lint.py](tests/test_pyflakes_code_lint.py) | Pyflakes lint gate for all Python files |
| [tests/test_indentation.py](tests/test_indentation.py) | Tab indentation enforcement |
| [tests/test_ascii_compliance.py](tests/test_ascii_compliance.py) | ASCII character compliance |
| [tests/git_file_utils.py](tests/git_file_utils.py) | Shared utility: `get_repo_root()` |
| [tests/conftest.py](tests/conftest.py) | Pytest configuration and shared fixtures |

### [artifacts/](artifacts/) -- reference materials

| File | Purpose |
| --- | --- |
| [artifacts/datalink-1200_image.png](artifacts/datalink-1200_image.png) | Blank DataLink 1200 form reference image |
| [artifacts/deep-research-report.md](artifacts/deep-research-report.md) | Background research on DataLink 1200 form support |

### [docs/](docs/) -- documentation

| File | Purpose |
| --- | --- |
| [docs/CODE_ARCHITECTURE.md](docs/CODE_ARCHITECTURE.md) | System design, components, data flow |
| [docs/FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md) | This file: directory map |
| [docs/INSTALL.md](docs/INSTALL.md) | Setup steps and dependencies |
| [docs/USAGE.md](docs/USAGE.md) | CLI usage for all scripts |
| [docs/INPUT_FORMATS.md](docs/INPUT_FORMATS.md) | Supported image input formats |
| [docs/OUTPUT_FORMATS.md](docs/OUTPUT_FORMATS.md) | CSV output schemas |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Known issues and debugging steps |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Chronological record of changes |
| [docs/AUTHORS.md](docs/AUTHORS.md) | Primary maintainers |
| [docs/PYTHON_STYLE.md](docs/PYTHON_STYLE.md) | Python coding conventions |
| [docs/REPO_STYLE.md](docs/REPO_STYLE.md) | Repository organization rules |
| [docs/MARKDOWN_STYLE.md](docs/MARKDOWN_STYLE.md) | Markdown formatting conventions |

## Generated artifacts

These directories are git-ignored and created at runtime:

| Path | Contents |
| --- | --- |
| `data/output/` | Pipeline output: registered images, answer CSVs, grade CSVs, debug overlays |
| `output_smoke/` | Smoke test output (reused across runs) |
| `scantrons/` | Input scantron images (user-provided, not committed) |

## Where to add new work

- **New form template**: add a YAML file to [config/](config/)
- **New OMR module**: add a `.py` file to [omr_utils/](omr_utils/)
- **New CLI script**: add to repo root following the existing `register_scan.py` pattern
- **New tests**: add `test_*.py` to [tests/](tests/)
- **Documentation**: add `SCREAMING_SNAKE_CASE.md` files to [docs/](docs/)
