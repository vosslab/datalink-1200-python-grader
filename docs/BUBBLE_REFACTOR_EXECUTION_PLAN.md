# Bubble Refactor Execution Plan

## Title and objective
Build a modular bubble-reading subsystem that (1) uses a minimal geometry contract with fixed bubble aspect ratio and total bubble area, (2) achieves stable visual target alignment, and (3) replaces the current monolithic path with staged components that are testable and releasable in patches.

## Design philosophy
- Parameter minimalism: keep only fixed `aspect_ratio` and `target_area_px` as shape invariants; derive width/height and all dependent bounds from those values.
- Anchor-first alignment: use high-contrast form primitives (left-edge dark dashes and top dark boxes) as primary geometric constraints, not only coarse template centers.
- Relative-coordinate runtime: compute bubble locations in an anchor-defined normalized frame and avoid mandatory global rescaling for scoring logic.
- Preserve dual detection zones: once a bubble is localized, keep two measurement zones (left and right strips) as a required part of scoring.
- Distortion-aware mapping: support mild page warp and perspective residuals using robust anchor-fit plus local correction when needed.
- Stage contracts over ad hoc heuristics: isolate localization, measurement, classification, and visualization as separate components with explicit I/O contracts.
- Visual correctness is a hard gate: overlay alignment quality is a release criterion, not a debug-only artifact.
- Additive migration: run new and legacy pipelines in parallel until parity and accuracy gates pass, then cut over.

## Scope and non-goals
### Scope
- Refactor answer-bubble extraction and overlay generation in `omr_utils/bubble_reader.py` into modular components.
- Reduce geometry control surface so bubble shape is driven by `aspect_ratio` and `target_area_px`.
- Redesign `config/dl1200_template.yaml` concept into a v2 contract that separates stable form anchors from algorithm tuning knobs.
- Reconsider hard canonical `1700x2200` dependence so stages can operate from normalized coordinates on native registered resolutions.
- Add a timing-mark anchor strategy that uses left-side dark dashes and top dark boxes as high-confidence guides for row and column alignment.
- Move bubble lookup to anchor-relative coordinates so localization remains stable across scan sizes and slight warp differences.
- Keep dual measurement zones after localization to preserve current signal behavior.
- Include distortion handling for perspective/curvature residuals after registration.
- Introduce measurable visual and scoring gates that block release if overlay alignment drifts.
- Preserve current CLI entry points and output schemas during migration.

### Non-goals
- Student ID subsystem redesign (only touch for compatibility shims if required).
- Image registration algorithm rewrite.
- New UI or reporting front-end beyond existing debug outputs and test artifacts.

## Current state summary
- The current bubble-reading implementation is concentrated in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py) (957 lines), with `read_answers()` combining localization, correction, measurement, classification, and result packaging in one function.
- Bubble geometry currently exposes many independently tuned controls (half sizes, insets, pads, shifts), which increases tuning surface and drift risk.
- Recent tests report acceptable detection counts, but visual overlays still miss targets in representative cases; this indicates metric coverage is not yet aligned with visual quality expectations.
- Changelog history shows repeated reactive tuning cycles (edge-separation thresholds, insets, half-height changes, row-correction heuristics) to recover specific failures, which is a signal that the template and geometry concept needs reset rather than incremental adjustment.
- Current template assumptions include canonical dimensions (`1700x2200`), which may be masking scale-specific behavior and should be treated as optional reference values rather than hard runtime constraints.
- This repo does not currently contain `refactor_progress.md`, `docs/active_plans/`, or `docs/archive/`; this document serves as the active execution plan baseline.

## Architecture boundaries and ownership
- Component: `geometry_contract` module
  - Owner: Coder 1
  - Responsibility: derive all bubble bounds from `aspect_ratio` and `target_area_px`; validate contract inputs.
- Component: `template_contract_v2` module/schema
  - Owner: Coder 1
  - Responsibility: define a start-over template model with only stable anchors, lattice rules, and bubble shape contract references.
- Component: `timing_mark_anchor_stage` module
  - Owner: Coder 2
  - Responsibility: detect and fit left-edge dash and top-box timing marks to produce robust row/column anchor transforms, normalized coordinate frames, and distortion-aware corrections.
- Component: `bubble_localization_stage` module
  - Owner: Coder 2
  - Responsibility: locate per-choice bubble rectangles from template anchors and image evidence.
- Component: `bubble_measurement_stage` module
  - Owner: Coder 3
  - Responsibility: compute measurement features from localized rectangles while preserving dual-zone (left/right) measurements.
- Component: `bubble_decision_stage` module
  - Owner: Coder 4
  - Responsibility: map features to answer decisions, BLANK/MULTIPLE flags, and confidence.
- Component: `bubble_overlay_stage` module
  - Owner: Coder 5
  - Responsibility: deterministic overlay rendering from stage outputs.
- Component: `bubble_quality_gates` module/tests
  - Owner: Coder 6
  - Responsibility: visual and scoring gate checks, regression harness, release blockers.
- Component: migration adapters in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py)
  - Owner: Coder 7
  - Responsibility: backward-compatible wrappers and cutover controls.
- Component: docs and release tracking
  - Owner: Coder 8
  - Responsibility: plan/progress/changelog/test-note hygiene and release checklist execution.

## Mapping: milestones and workstreams map to components and patches
| Milestone | Workstreams | Components | Patch grouping intent |
| --- | --- | --- | --- |
| M-0 Template Contract Reset | WS-0A, WS-0B, WS-0C, WS-0D, WS-0E | `template_contract_v2`, `template_migrator`, `timing_mark_anchor_stage`, `bubble_quality_gates` | Patch 1-4 reboot template concept and anchor-relative mapping |
| M-A Contract + Baseline | WS-A1, WS-A2, WS-A3, WS-A4 | `geometry_contract`, `bubble_quality_gates` | Patch 5-6 lock geometry contract and baseline gates |
| M-B Modular Extraction | WS-B1, WS-B2, WS-B3, WS-B4 | `bubble_localization_stage`, `bubble_measurement_stage`, `bubble_decision_stage`, `bubble_overlay_stage`, migration adapters | Patch 7-11 split monolith into stage modules with compatibility wrappers |
| M-C Accuracy Recovery | WS-C1, WS-C2, WS-C3, WS-C4 | all stage components + `bubble_quality_gates` | Patch 12-14 close visual, distortion, and dual-zone threshold gaps |
| M-D Cutover + Cleanup | WS-D1, WS-D2, WS-D3, WS-D4 | migration adapters, docs, tests | Patch 15 plus closure docs/tests |

## Milestone plan (ordered, dependency-aware)
### Dependency IDs
- DEP-000: Template v2 concept, schema, and migration translator approved.
- DEP-00A: Timing-mark anchor model validated for row/column alignment and scale normalization.
- DEP-001: Baseline visual/scoring dataset and metric spec approved.
- DEP-002: Geometry contract (`aspect_ratio`, `target_area_px`) approved.
- DEP-003: Stage I/O contracts frozen.
- DEP-004: New staged path passes output parity gate against legacy path.
- DEP-005: New staged path passes visual/scoring accuracy gates.
- DEP-006: Release owner approves cutover and rollback plan.

### Ordered milestones
1. M-0 Template Contract Reset
   - Depends on: none
   - Produces: DEP-000, DEP-00A
2. M-A Contract + Baseline
   - Depends on: none
   - Produces: DEP-001, DEP-002
3. M-B Modular Extraction
   - Depends on: DEP-000 (template reboot), DEP-001 (shared baseline), DEP-002 (geometry contract)
   - Produces: DEP-003, DEP-004
4. M-C Accuracy Recovery
   - Depends on: DEP-003 (stable interfaces), DEP-004 (parity safety)
   - Produces: DEP-005
5. M-D Cutover + Cleanup
   - Depends on: DEP-005 (quality proven)
   - Produces: DEP-006 and closure artifacts

## Workstream breakdown
Capacity targets applied from reference: 4 workstreams per milestone, 6 to 16 work packages per workstream, and at least 16 ready work packages at milestone start.

### M-0 workstreams
#### WS-0A Template Concept Reboot
- Goal: redefine what template data is stable form truth versus algorithm behavior.
- Owner: Coder 1
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: failure patterns from changelog and baseline fixtures
  - Provides: template v2 concept and field taxonomy
- Expected patches: 1 to 2 patches

#### WS-0B Schema and Validator
- Goal: create a strict v2 schema for `dl1200_template.yaml` with validation tooling.
- Owner: Coder 1
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: concept from WS-0A
  - Provides: schema, examples, and validator tests
- Expected patches: 2 patches

#### WS-0C Template Migrator
- Goal: provide deterministic translation from v1 template to v2 format.
- Owner: Coder 7
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: v2 schema from WS-0B
  - Provides: migration script and parity report
- Expected patches: 1 to 2 patches

#### WS-0D Template Gate Harness
- Goal: add gates proving template v2 aligns overlays better than v1 on labeled set.
- Owner: Coder 6
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: baseline labels and migrator outputs
  - Provides: template quality gate and failure diagnostics
- Expected patches: 2 patches

#### WS-0E Timing-Mark Anchor Model
- Goal: use left-side dark dashes and top dark boxes as explicit anchor chains for row/column and scale alignment.
- Owner: Coder 2
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: registered grayscale images and v2 template anchors
  - Provides: per-row and per-column anchor transforms, normalized coordinate frame, distortion residual model, and confidence diagnostics
- Expected patches: 2 patches

### M-A workstreams
#### WS-A1 Baseline Dataset and Annotation
- Goal: lock baseline truth set for visual and scoring evaluation.
- Owner: Coder 6
- Work packages: 6 planned (target range 6-16): WP-A1 to WP-A6
- Interfaces:
  - Needs: access to current scantron fixtures and debug outputs
  - Provides: labeled bubble boxes, expected answers, visual metric definitions
- Expected patches: 2 patches (fixtures/labels, metric spec)

#### WS-A2 Geometry Contract Definition
- Goal: define and approve minimal shape contract and derivation formulas.
- Owner: Coder 1
- Work packages: 6 planned (target range 6-16): WP-A7 to WP-A12
- Interfaces:
  - Needs: baseline failure examples from WS-A1
  - Provides: contract doc, derived-parameter equations, validation rules
- Expected patches: 2 patches (contract module skeleton, validation tests)

#### WS-A3 Stage Contract Scaffolding
- Goal: define stage boundaries and data contracts before code extraction.
- Owner: Coder 2
- Work packages: 6 planned (target range 6-16): WP-A13 to WP-A18
- Interfaces:
  - Needs: geometry contract from WS-A2
  - Provides: localization/measurement/decision/overlay contract schemas
- Expected patches: 1 to 2 patches (contract types + adapter stubs)

#### WS-A4 Gate Harness and Reporting
- Goal: enforce release-blocking gates for visual alignment and scoring.
- Owner: Coder 8
- Work packages: 6 planned (target range 6-16): WP-A19 to WP-A24
- Interfaces:
  - Needs: metrics from WS-A1 and stage contracts from WS-A3
  - Provides: repeatable gate command set and patch reporting template
- Expected patches: 2 patches (gate runner + reporting docs)

### M-B workstreams
#### WS-B1 Localization Stage Extraction
- Goal: isolate bubble localization into a dedicated module with deterministic output.
- Owner: Coder 2
- Work packages: 7 planned (target range 6-16)
- Interfaces:
  - Needs: stage contracts (DEP-003), geometry contract
  - Provides: localized rectangles and confidence metadata
- Expected patches: 2 patches

#### WS-B2 Measurement Stage Extraction
- Goal: isolate brightness/edge feature measurement from localization logic while retaining dual-zone scoring.
- Owner: Coder 3
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: localization outputs from WS-B1
  - Provides: per-choice feature vectors with explicit left-zone and right-zone measurements
- Expected patches: 2 patches

#### WS-B3 Decision Stage Extraction
- Goal: isolate blank/multiple/answer decision policy from measurement collection.
- Owner: Coder 4
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: measurement features from WS-B2
  - Provides: answers, flags, confidence, ranking explanation
- Expected patches: 2 patches

#### WS-B4 Overlay Stage Extraction
- Goal: isolate debug overlay rendering and align it to localized rectangles only.
- Owner: Coder 5
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: localization and decision outputs
  - Provides: deterministic overlay image + visual diagnostics summary
- Expected patches: 2 patches

### M-C workstreams
#### WS-C1 Parameter Reduction and Tuning
- Goal: remove legacy fixed geometry knobs and tune only contract-driven derivations.
- Owner: Coder 1
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: modular stages from M-B
  - Provides: reduced config footprint and derivation calibration notes
- Expected patches: 2 patches

#### WS-C2 Visual Error Correction
- Goal: eliminate overlay miss failures using measurable alignment diagnostics.
- Owner: Coder 5
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: baseline labels and gate harness
  - Provides: reduced center/edge error distributions and pass artifacts
- Expected patches: 2 patches

#### WS-C3 Robustness Hardening
- Goal: stabilize behavior across phone photos, flatbed scans, and borderline marks.
- Owner: Coder 3
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: measurement/decision modules and failure catalog
  - Provides: hardened thresholds, distortion fallback policy, and confidence-based recovery rules
- Expected patches: 2 patches

#### WS-C4 Regression Expansion
- Goal: add test coverage that reflects visual correctness and not only detection counts.
- Owner: Coder 6
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: corrected outputs and expected overlays
  - Provides: regression suite and gate thresholds in CI commands
- Expected patches: 2 patches

### M-D workstreams
#### WS-D1 Cutover Adapter
- Goal: make staged pipeline default while preserving backward compatibility.
- Owner: Coder 7
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: DEP-005 gate pass
  - Provides: default-path switch, compatibility wrapper
- Expected patches: 2 patches

#### WS-D2 Legacy Cleanup
- Goal: remove or quarantine dead monolithic logic after cutover stability window.
- Owner: Coder 2
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: cutover stability report
  - Provides: simplified module surface and reduced maintenance burden
- Expected patches: 1 to 2 patches

#### WS-D3 Release Readiness
- Goal: execute release gates, rollback rehearsal, and operational sign-off.
- Owner: Coder 8
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: final test artifacts and migration notes
  - Provides: release checklist completion and rollback packet
- Expected patches: 1 patch

#### WS-D4 Documentation and Training
- Goal: align architecture/usage/troubleshooting docs with new stage model.
- Owner: Coder 8
- Work packages: 6 planned (target range 6-16)
- Interfaces:
  - Needs: final module names and CLI behavior confirmation
  - Provides: updated docs and handoff notes
- Expected patches: 1 patch

## Per-milestone deliverables and done checks
### M-0 Template Contract Reset
- Depends on: none
- Entry criteria: current v1 template behavior reproducible on fixture set.
- Deliverables:
  - Template v2 concept with stable-anchor-first field model.
  - Schema validator and migration translator from v1 to v2 (DEP-000).
  - Timing-mark anchor model using left dashes and top boxes with confidence scoring (DEP-00A).
  - Relative-coordinate mapping contract from anchors to bubble lattice indices.
  - Distortion-resilient mapping policy (global fit plus local correction fallback) documented and tested.
  - Template quality gate with labeled overlay metrics.
- Done checks:
  - `dl1200_template.yaml` v2 draft has no algorithm threshold knobs.
  - Migration translator reproduces v1 anchor coordinates within tolerance.
  - Left-dash and top-box timing-mark detection recall >= 98% and precision >= 99% on labeled fixture set.
  - Relative-coordinate mapping error <= 1.0 px median and <= 2.0 px P95 on labeled bubble centers.
  - Distortion-stress subset (phone-photo residual perspective cases) meets center error <= 2.5 px P95.
  - Pipeline passes visual gates at multiple registered resolutions without mandatory hard resize to `1700x2200`.
  - Gate report compares v1 versus v2 overlay errors on shared fixtures.
- Exit criteria: DEP-000 and DEP-00A marked complete by design review sign-off.

### M-A Contract + Baseline
- Depends on: none
- Entry criteria: current pipeline runs on baseline fixtures.
- Deliverables:
  - Baseline labeled dataset and metric definitions (DEP-001).
  - Approved geometry contract and derivation formulas (DEP-002).
  - 16 ready work packages with dependencies set to none.
- Done checks:
  - Visual metric spec committed and reviewed.
  - Contract doc includes formulas for width/height derivation from area and aspect ratio.
  - Gate runner command returns deterministic baseline report.
- Exit criteria: DEP-001 and DEP-002 marked complete by owner sign-off.

### M-B Modular Extraction
- Depends on: DEP-000 (template reboot), DEP-001 (shared baseline), DEP-002 (contract)
- Entry criteria: stage contracts drafted and accepted.
- Deliverables:
  - Localization, measurement, decision, and overlay modules created.
  - Dual-zone measurement preserved in staged measurement outputs.
  - Legacy wrapper keeps output schema unchanged.
  - Parity report proving staged vs legacy output equivalence for baseline set.
- Done checks:
  - New modules expose documented contracts.
  - Dual-zone measurements are present for every localized bubble and consumed by decision stage.
  - `read_answers()` reduced to orchestration logic with stage calls, not embedded stage internals.
  - Parity mismatch rate is 0 for question answer and flags on baseline set.
  - Any patch touching more than two components is split before merge.
- Exit criteria: DEP-003 and DEP-004 complete.

### M-C Accuracy Recovery
- Depends on: DEP-003 (stable interfaces), DEP-004 (parity)
- Entry criteria: staged path enabled in test mode.
- Deliverables:
  - Accuracy improvements on full fixture set.
  - Config surface reduced to contract-driven geometry plus explicit thresholds.
  - Expanded regression tests covering visual alignment metrics.
- Done checks:
  - P95 bubble-center error <= 1.5 px on labeled set.
  - P95 edge-box IoU >= 0.92 on labeled set.
  - Question-level answer accuracy >= 99.5% on evaluation set.
  - Zero known "complete visual fail" cases in gate report.
- Exit criteria: DEP-005 complete.

### M-D Cutover + Cleanup
- Depends on: DEP-005 (quality gate pass)
- Entry criteria: release candidate branch created.
- Deliverables:
  - New staged path enabled by default.
  - Legacy path either removed or disabled behind explicit fallback switch.
  - Final docs and changelog closure.
- Done checks:
  - Rollback rehearsal completed with documented command path.
  - Release checklist fully checked with owner initials.
  - Post-cutover smoke run passes on baseline and regression fixtures.
- Exit criteria: DEP-006 complete and plan archived.

## Work package template (required for assignment-ready chunks)
Work package title: `<verb + object>`
- Owner: `<coder>`
- Touch points: `<files/components>`
- Acceptance criteria: `<measurable outcomes>`
- Verification commands: `<bash -lc "source source_me.sh && ...">`
- Dependencies: `<work package IDs or none>`

### Ready-at-start package set (16)
All items below are assignment-ready and start with `Dependencies: none`.

1. WP-T1 Define v2 template field taxonomy
   - Owner: Coder 1
   - Touch points: `config/dl1200_template.yaml`, new schema docs
   - Acceptance criteria: every field labeled as anchor, lattice, shape, or metadata; threshold knobs excluded.
   - Verification commands: `bash -lc "rg -n \"template v2|field taxonomy\" docs/BUBBLE_REFACTOR_EXECUTION_PLAN.md"`
   - Dependencies: none
2. WP-T2 Draft v2 template schema
   - Owner: Coder 1
   - Touch points: schema module and template fixture files
   - Acceptance criteria: schema validates required anchors and choice-lattice definitions.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_template_loader.py -k schema"`
   - Dependencies: none
3. WP-T3 Build template validator tests
   - Owner: Coder 1
   - Touch points: `tests/test_template_loader.py`
   - Acceptance criteria: malformed template files fail with actionable errors.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_template_loader.py -k template"`
   - Dependencies: none
4. WP-T4 Implement v1-to-v2 migrator scaffold
   - Owner: Coder 7
   - Touch points: new migrator module under `omr_utils/`
   - Acceptance criteria: migrator emits a valid v2 file from current v1 template.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_template_loader.py -k migrator"`
   - Dependencies: none
5. WP-T5 Define template parity report format
   - Owner: Coder 7
   - Touch points: tests/util report helper
   - Acceptance criteria: report includes anchor delta, lattice delta, overlay error deltas, and timing-mark fit residuals for both left and top anchors.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_pipeline_smoke.py -k template"`
   - Dependencies: none
6. WP-T6 Add template quality gate thresholds
   - Owner: Coder 6
   - Touch points: gate harness tests
   - Acceptance criteria: v2 template must beat or match v1 on center-error, IoU, timing-mark anchor residuals, relative-frame mapping error, and distortion-stress subset metrics.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_bubble_reader.py -k gate"`
   - Dependencies: none
7. WP-T7 Publish template v2 migration policy
   - Owner: Coder 8
   - Touch points: plan docs + troubleshooting docs
   - Acceptance criteria: rollback and fallback instructions documented before cutover.
   - Verification commands: `bash -lc "rg -n \"template v2|rollback\" docs/BUBBLE_REFACTOR_EXECUTION_PLAN.md"`
   - Dependencies: none
8. WP-T8 Approve template reboot design review
   - Owner: Coder 8
   - Touch points: plan sign-off notes
   - Acceptance criteria: design review captures accepted/rejected fields with rationale.
   - Verification commands: `bash -lc "rg -n \"DEP-000\" docs/BUBBLE_REFACTOR_EXECUTION_PLAN.md"`
   - Dependencies: none
9. WP-A1 Capture baseline overlays
   - Owner: Coder 6
   - Touch points: `scantrons/`, `output_smoke/`, `tests/test_bubble_reader.py`
   - Acceptance criteria: baseline fixtures produce reproducible overlays in committed artifact manifest.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_bubble_reader.py -k detect_at_least_71"`
   - Dependencies: none
10. WP-A2 Annotate bubble target boxes
   - Owner: Coder 6
   - Touch points: new label file under `tests/fixtures/`
   - Acceptance criteria: at least 1,000 bubble boxes labeled with question/choice IDs.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_bubble_reader.py -k overlay"`
   - Dependencies: none
11. WP-A3 Define visual metric calculator
   - Owner: Coder 6
   - Touch points: `tests/` metric utility module
   - Acceptance criteria: center error and IoU metrics computed for each labeled bubble.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_bubble_reader.py -k metric"`
   - Dependencies: none
12. WP-A4 Specify scoring metric contract
   - Owner: Coder 3
   - Touch points: docs plan appendix + tests
   - Acceptance criteria: question accuracy, blank precision, and multiple precision formulas documented and tested.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_grade_answers.py -k confidence"`
   - Dependencies: none
13. WP-A5 Draft geometry derivation formulas
   - Owner: Coder 1
   - Touch points: new `geometry_contract` doc/module
   - Acceptance criteria: formulas derive half-width/half-height from `aspect_ratio` and `target_area_px`.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_template_loader.py -k geometry"`
   - Dependencies: none
14. WP-A6 Add geometry validation tests
   - Owner: Coder 1
   - Touch points: `tests/test_template_loader.py`
   - Acceptance criteria: invalid contract values fail deterministically with clear errors.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_template_loader.py -k contract"`
   - Dependencies: none
15. WP-A7 Define localization stage schema
   - Owner: Coder 2
   - Touch points: new stage contract module
   - Acceptance criteria: schema defines inputs/outputs with typed keys and units.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_bubble_reader.py -k localization"`
   - Dependencies: none
16. WP-A8 Define measurement stage schema
   - Owner: Coder 3
   - Touch points: stage contract module
   - Acceptance criteria: measurement feature payload documented and validated.
   - Verification commands: `bash -lc "source source_me.sh && python -m pytest tests/test_bubble_reader.py -k measurement"`
   - Dependencies: none

## Acceptance criteria and gates
### Unit/verification gate
- Template v2 schema validation and v1-to-v2 migration tests pass.
- Timing-mark anchor extraction tests pass across scan quality variants for left-dash and top-box markers.
- Geometry derivation and validation tests pass for boundary and invalid inputs.
- Stage contract tests pass with deterministic schema validation.
- Modularity checks pass: stage modules are independently unit-tested and imported without `read_answers()` side effects.

### Integration gate
- Migrated template v2 produces anchor/lattice parity with v1 within tolerance.
- Timing-mark anchored alignment (left + top) outperforms center-only template alignment on overlay metrics.
- End-to-end extraction with staged modules matches legacy answers/flags on baseline fixtures (DEP-004).
- Relative-coordinate pipeline produces equivalent answers across at least three registered resolutions per fixture image.
- Distortion-stress fixtures meet visual and scoring gates without per-image manual tuning.

### Regression gate
- Full bubble-reader regression suite passes, including visual metric tests.
- No increase in BLANK/MULTIPLE false positives relative to baseline by more than 0.2 percentage points.
- Dual-zone consistency checks pass (left/right zone divergence stays within defined tolerance bands for blank and filled populations).

### Release gate
- DEP-005 met (visual + scoring thresholds).
- DEP-006 approved with rollback rehearsal evidence attached.

## Test and verification strategy
- Unit checks (module-level):
  - `bash -lc "source source_me.sh && python -m pytest tests/test_template_loader.py -k geometry"`
  - `bash -lc "source source_me.sh && python -m pytest tests/test_bubble_reader.py -k \"linearity or brightness or validate\""`
- Integration checks (pipeline behavior):
  - `bash -lc "source source_me.sh && python -m pytest tests/test_pipeline_smoke.py -k bubble"`
- Regression checks:
  - `bash -lc "source source_me.sh && python -m pytest tests/test_bubble_reader.py"`
  - `bash -lc "source source_me.sh && python -m pytest tests/test_grade_answers.py"`
- Repository gates:
  - `bash -lc "source source_me.sh && python -m pytest tests/test_pyflakes_code_lint.py"`
  - `bash -lc "source source_me.sh && python -m pytest tests/test_ascii_compliance.py"`
- Failure semantics:
  - Any gate failure blocks milestone exit.
  - Release candidate cannot advance with open regression failures.

## Migration and compatibility policy
- Additive-first:
  - Introduce staged modules behind compatibility wrappers in [omr_utils/bubble_reader.py](../omr_utils/bubble_reader.py).
  - Ship template v2 with a translator from current `dl1200_template.yaml` and keep v1 loading support during migration.
  - Keep canonical `1700x2200` resizing as an optional compatibility path, not a required runtime step.
  - Prefer anchor-derived relative coordinates as the default runtime coordinate model for localization and scoring.
  - Preserve dual-zone measurement as a stable contract during and after migration.
  - Keep public `read_answers()` and `draw_answer_debug()` signatures stable during M-A through M-C.
- Backward compatibility promise:
  - Output schema (`question`, `answer`, `scores`, `flags`, `positions`, `edges`) remains unchanged through cutover.
- Deletion criteria:
  - Legacy monolithic branches and template-v1-only paths are removable only after two consecutive regression runs pass post-cutover and rollback rehearsal succeeds.
- Rollback strategy:
  - Retain legacy execution path behind explicit switch until DEP-006 sign-off; rollback is a one-flag change, not a hot patch.

## Risk register and mitigations
| Risk | Impact | Trigger | Owner | Mitigation |
| --- | --- | --- | --- | --- |
| R-001 Contract underspecification | high | repeated tuning knobs reappear | Coder 1 | enforce two-invariant contract tests and review gate |
| R-002 Modular split breaks parity | high | staged path diverges from legacy output | Coder 7 | parity comparator required before merge |
| R-003 Visual metrics too weak | high | overlays pass tests but still look wrong | Coder 6 | include labeled IoU/center-error gates, not only count metrics |
| R-004 Scope creep into student ID/registration | medium | unrelated modules touched in many patches | Coder 8 | enforce scope gate in PR review checklist |
| R-005 Patch size review bottleneck | medium | patches touch >2 components | Coder 8 | split patches per component boundary before review |
| R-006 Data drift in fixture set | medium | new failures appear outside baseline images | Coder 6 | keep rotating regression fixtures and compare distributions |
| R-007 Cutover regression in production usage | high | post-release mismatch reports | Coder 7 | staged rollout plus rollback rehearsal and documented fallback |
| R-008 Distortion model mismatch | high | local warp causes anchor-fit residual spikes | Coder 2 | add distortion-stress fixture class and require residual gate before release |
| R-009 Monolith regression | medium | stage logic drifts back into `read_answers()` | Coder 2 | enforce orchestration-only rule and modularity gate in review |

## Rollout and release checklist
1. Confirm DEP-000, DEP-00A, and DEP-001 through DEP-005 are complete with evidence links.
2. Run full regression and repository lint gates on release candidate.
3. Execute rollback rehearsal and store command transcript.
4. Enable staged path as default and run smoke checks on all fixture classes.
5. Publish release notes and operator troubleshooting deltas.
6. Keep legacy fallback available for one stabilization window.
7. Remove legacy path only after stabilization criteria are met.

## Documentation close-out requirements
- Update [docs/CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) with new stage/component boundaries.
- Update [docs/FILE_STRUCTURE.md](FILE_STRUCTURE.md) with any new modules/tests/fixtures.
- Update [docs/INPUT_FORMATS.md](INPUT_FORMATS.md) with template v2 schema and migration notes.
- Update [docs/USAGE.md](USAGE.md) if CLI flags or debug outputs change.
- Update [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) with new visual-gate failure diagnosis.
- Update [docs/CHANGELOG.md](CHANGELOG.md) using `Patch 1`, `Patch 2`, ... labels for implementation summaries.
- Create and maintain `refactor_progress.md` (new file at repo root) during implementation and archive it at closure.

## Patch plan and reporting format
Required report format:
- `Patch 1: [template_contract_v2] define stable-anchor template schema`
- `Patch 2: [template_migrator] add v1-to-v2 translator and parity checks`
- `Patch 3: [timing_mark_anchor_stage] add left-dash and top-box anchor extraction with fit diagnostics`
- `Patch 4: [timing_mark_anchor_stage] add anchor-relative coordinate frame and lattice mapper`
- `Patch 5: [geometry_contract] define two-invariant bubble geometry contract`
- `Patch 6: [bubble_quality_gates] add baseline visual/scoring gate harness`
- `Patch 7: [bubble_localization_stage] extract localization stage with contract outputs`
- `Patch 8: [bubble_measurement_stage] extract measurement stage with dual-zone feature payload`
- `Patch 9: [bubble_decision_stage] extract decision stage and parity comparator hooks`
- `Patch 10: [bubble_overlay_stage] extract deterministic overlay renderer`
- `Patch 11: [migration_adapter] keep legacy-compatible wrappers in bubble_reader`
- `Patch 12: [geometry_contract] remove redundant fixed geometry parameters`
- `Patch 13: [bubble_quality_gates] enforce IoU, center-error, anchor-residual, relative-frame, and dual-zone thresholds`
- `Patch 14: [tests] expand regression coverage for visual failures and distortion stress`
- `Patch 15: [migration_adapter] switch default path to staged pipeline`
- `Patch N: tests, migration, docs`

Cadence and sizing rules:
- Target 1 to 2 reviewable patches per coder per week.
- Split any patch touching more than two components.

## Open questions and decisions needed
1. Decision needed: choose canonical `target_area_px` reference value and tolerance band by image scale.
   - Owner: Coder 1 with reviewer sign-off.
2. Decision needed: finalize template v2 field boundary between stable anchors and algorithm behavior knobs.
   - Owner: Coder 1 with reviewer sign-off.
3. Decision needed: should canonical `1700x2200` be retained only for debug rendering, or for any runtime logic.
   - Owner: Coder 2 with reviewer sign-off.
4. Decision needed: weighting policy when left-dash and top-box anchors disagree in low-quality scans.
   - Owner: Coder 2 with reviewer sign-off.
5. Decision needed: exact relative-coordinate transform model (affine vs piecewise affine) for anchor-to-lattice mapping.
   - Owner: Coder 2 with reviewer sign-off.
6. Decision needed: distortion fallback trigger (when to switch from global to local correction model).
   - Owner: Coder 2 with reviewer sign-off.
7. Decision needed: define minimum labeled dataset size beyond the initial 1,000 bubble boxes.
   - Owner: Coder 6.
8. Decision needed: keep or remove legacy fallback after stabilization window.
   - Owner: release owner (Coder 8) with stakeholder approval.
9. Decision needed: acceptable confidence-gap behavior for lightly marked but valid bubbles.
   - Owner: Coder 3 with grading-policy reviewer.
10. Decision needed: threshold for classifying a visual failure as release-blocking when scoring still passes.
   - Owner: Coder 6 and release owner.
11. Decision needed: exact dual-zone fusion rule (mean/min/weighted) for final fill score.
   - Owner: Coder 3 with reviewer sign-off.
12. Decision needed: modularity guardrail definition (`read_answers()` orchestration-only rule and max stage complexity limits).
   - Owner: Coder 2 with reviewer sign-off.
