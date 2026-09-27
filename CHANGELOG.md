# Changelog

## 0.4.0

### Breaking changes

- `derivative_points` (`-dm`) now defaults to `4` on the binary path, matching
  the native engine. Pass it explicitly to preserve old results.
- Unknown variant abbreviations throw instead of silently producing an all-zero
  SELECT mask; missing output files for requested variants throw instead of
  returning partial results.
- Combining `request=` with analysis keywords (`file_path`, `channels`,
  `flavors`, or any other keyword) throws instead of ignoring them.
- Removed nonfunctional options: `DDARequest` no longer accepts `ct_pairs` /
  `cd_pairs` (the binary has no pair-list flag), and the write-only `sfreq`
  keyword is gone from `run_st` / `run_ct` / `run_de` / `run_batch`.
- `ChannelStructureSelectionResult.channel_index` removed (use the element
  position in `.results`).
- `collect_results(::Vector{DEResult})` returns a dedicated `DEGroupResult`
  instead of a degenerate `GroupResult`; `GroupResult.variant` removed.
- `StructuredTimepoint` stores `Int64` window bounds and renames its `error`
  field to `value` (stride-1 flavors store a measure, not an error).
- `DDAResult.created_at` is a `DateTime`; `TimeRange` stores `Int64`.
- Native CT/DE row labels join with `-` (was `&`), matching pair labels.
- `write_model_LaTeX` drops its unused `_SSYM` parameter;
  `SUPPORTED_PLATFORMS` constant removed.
- Julia compat floor raised to 1.10; stdlib `[compat]` entries declared.

### Deprecated

- `result.ST`-style property access: use `flavor_matrix(result, "ST")` or
  `result["ST"]`.
- Calling `structure_selection_select()` without `run=`: pass the
  `StructureSelectionRun` explicitly.
- Legacy kwargs, all still functional with warnings: `dm` →
  `derivative_points`, `model_encoding` → `model`, `ct_window_length` →
  `WL_CT`, `ct_window_step` → `WS_CT`, `candidate_delays` → `delays`.

### Added

- Native Julia engine (`run_dda_matrix`) for ST/CT/CD/DE/SY with optional CUDA.
- Bounded native CPU parallelism via `num_cores` and `parallelism=:threads`
  or `:processes`, with temporary process workers cleaned up on completion
  or error. Complete window batches run concurrently; single-window inputs
  parallelize independent regressions. The default is `num_cores=1`.
- Reproducible native parallelism benchmark with serial-output parity checks.
- `flavor_matrix(result, flavor)` as the canonical flavor accessor.
- `OptionalDeps`-backed lazy loading: Plots, DataFrames, and CUDA are only
  loaded when used. Plots is no longer installed automatically.

### Fixed

- Native scratch buffers are task-local, preventing concurrent tasks from
  sharing mutable storage. Separate design and target buffers also prevent
  aliasing for one-term models.
- `Flavors.DE.name` is "Dynamical Ergodicity", matching every other surface
  (was inconsistently "Delay Embedding").
- `generate_monomials` docstring documents its actual `Vector{Vector{Int}}`
  return type; parser docs describe the stride-1 `.value` measure.
- Exported `run_st` / `run_ct` / `run_de` carry full docstrings (previously
  only private implementations were documented).

### Changed

- `StructureSelectionTrial.model` and `StructureSelectionResult.best_model`
  are typed as `Union{AbstractVector{Int}, AbstractMatrix{Int}}` instead of
  `Any`; `result`/`best_result` remain flexible by design.
- `run_dda_matrix` internals split into context/window-batch/unpack/assembly
  stages; numerics are unchanged (parity fixture passes unchanged).

- p-values in `compare_windows` come from HypothesisTests.jl (replacing ~180
  lines of hand-rolled numerics); returned statistics keep their meaning.
- The binary command is built once per run; runner failures propagate the
  original process error context.
- Structure-selection pool locks simplified to mkdir locks with mtime-based
  staleness (default 12 h, tunable).
- Defaults single-sourced in `DDADefaults`; public exports declared only at
  the top-level module; duplicated helpers (window bounds, pair labels,
  strict matrix reader) consolidated.
