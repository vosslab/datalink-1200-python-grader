# datalink-1200-python-grader

Python OMR grader for Apperson DataLink 1200 (AccuScan #28040) bubble sheets. Processes phone photos or flatbed scans of 100-question scantron forms, extracts student answers and IDs, and grades them against a scanned answer key.

## Quick start

```bash
# install dependencies
pip install -r pip_requirements.txt

# batch grade a directory of scantron images
source source_me.sh && python run_pipeline.py \
  -i scantrons/ \
  -k scantrons/answer_key.jpg \
  -o data/output/
```

The answer key is a filled-in scantron form that goes through the same pipeline as student sheets.

## Pipeline

1. **Register** -- detect page boundary, correct perspective, warp to canonical rectangle
2. **Extract** -- score bubbles, read answers (A-E) and 9-digit student ID
3. **Grade** -- compare student answers to key, produce scores with confidence measures

Each step has a standalone CLI script. See [docs/USAGE.md](docs/USAGE.md) for details.

## Documentation

- [docs/INSTALL.md](docs/INSTALL.md) -- setup and dependencies
- [docs/USAGE.md](docs/USAGE.md) -- CLI usage for all scripts
- [docs/CODE_ARCHITECTURE.md](docs/CODE_ARCHITECTURE.md) -- system design and data flow
- [docs/FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md) -- directory map
- [docs/INPUT_FORMATS.md](docs/INPUT_FORMATS.md) -- supported image formats and form details
- [docs/OUTPUT_FORMATS.md](docs/OUTPUT_FORMATS.md) -- CSV output schemas
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) -- known issues and debugging

## Author

Neil Voss, https://bsky.app/profile/neilvosslab.bsky.social

## License

See [LICENSE.MIT](LICENSE.MIT).
