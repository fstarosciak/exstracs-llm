# Implementation plan: ExSTraCS in Rust (step 1 of the LLM-seeded ExSTraCS project)

Source of truth for the research project: `../specyfikacja-projektu-exstracs-llm.md` (Polish).
This plan covers the first deliverable agreed with the supervisor: an implementation of the
original ExSTraCS model in Rust, built so that LLM-generated rules can be loaded as a seed
population.

## 1. Goal of this step

Build the `exstracs` crate (library + CLI) that:

1. trains an ExSTraCS classifier on a tabular CSV dataset (discrete and continuous attributes,
   missing values),
2. predicts and reports accuracy, balanced accuracy, macro-F1 and coverage,
3. saves and loads the rule population as human-readable JSON, including rule provenance
   (`covering`, `ga`, `llm`, `random`, `tree`, `expert`) and parent ids,
4. seeds the initial population from a rule file (spec section 5.3), validating every rule and
   counting rejections instead of silently fixing them,
5. supports the fixed-conditions mode (variant V2a): rule conditions stay frozen, only rule
   parameters (accuracy, fitness) are learned,
6. ships a multiplexer data generator to verify the implementation (spec criterion K1).

Out of scope for this step (later milestones): the Python LLM rule generator, JSON Schema file,
rule corruption tool, expert knowledge (EK), attribute tracking/feedback, rule compaction,
PyO3 bindings, golden-vector tests against scikit-ExSTraCS, experiment scripts.

## 2. Repository layout

```
exstracs-llm/
  Cargo.toml              workspace (single member for now)
  README.md               overview, build, usage
  docs/PLAN.md            this file
  examples/               small example inputs (multiplexer optimal rules)
  exstracs/               the crate; written so it can be moved into the acs2-rust workspace
    src/lib.rs            module wiring and public re-exports
    src/error.rs          crate error type
    src/config.rs         hyperparameters (defaults follow scikit-ExSTraCS)
    src/data.rs           schema, CSV loading, type inference, missing values
    src/classifier.rs     ALKR condition, predicates, classifier, provenance
    src/population.rs     population container, match/correct sets
    src/ga.rs             parent selection, crossover, mutation
    src/model.rs          ExSTraCS learning loop, covering, updates, subsumption, deletion, prediction
    src/metrics.rs        accuracy, balanced accuracy, macro-F1, coverage
    src/rules.rs          rule-file format, validation, rejection report
    src/snapshot.rs       population snapshot (JSON) types and file I/O
    src/multiplexer.rs    multiplexer dataset generator
    src/main.rs           `exstracs` CLI
    tests/                integration tests (multiplexer, seeding, determinism)
```

The crate depends only on general-purpose crates, not on `acs2-*` crates, because ExSTraCS uses
its own rule representation. It follows the acs2-rust conventions (edition, lints, seeded
ChaCha RNG, clap CLI) so the module split can be agreed with the library author and the crate
moved into that workspace later.

## 3. Algorithm summary (ExSTraCS 2.0 core, priority P0)

One learning iteration on training instance `(x, y)`:

1. **Match set [M]**: rules whose condition matches `x`. Discrete predicate: equal value.
   Continuous predicate: `low <= v <= high`. A missing value matches any predicate.
2. **Correct set [C]**: rules in [M] whose class is `y`.
3. **Covering** if [C] is empty: new rule for class `y` matching `x`; it specifies a random
   number `1..=RSL` of the non-missing attributes of `x`; continuous attributes get the interval
   `v ± r`, `r = U{25..75}% * range / 2`.
4. **Parameter update** for every rule in [M]: experience, correct count,
   `accuracy = correct / experience`, `fitness = accuracy^nu`, average match-set size (MAM rule
   with `beta`).
5. **Correct-set subsumption** (optional): the most general accurate and experienced rule in [C]
   absorbs the rules it is more general than.
6. **Niche GA** in [C] when `iteration - mean GA timestamp > theta_GA`: tournament (or roulette)
   selection of two parents, uniform crossover with probability `chi`, mutation with probability
   `mu` per attribute (generalise / specify from `x` / shift an interval bound), rule specificity
   limit, GA subsumption, insertion (an identical rule only increases numerosity).
7. **Deletion** while the micro-population exceeds `N`: roulette on the deletion vote
   `avgMatchSetSize * numerosity`, scaled by `meanFitness / fitness` for experienced rules with
   `fitness < delta * meanFitness`.

**Prediction**: each class gets the vote `sum(fitness * numerosity)` over matching rules. Ties go
to the class that is more frequent in training data. An instance matched by no rule is
"uncovered" and gets the majority training class (coverage is reported separately).

**Rule specificity limit (RSL)**: configurable; by default the smallest `i` with
`avgStateCount^i >= nTrainingInstances`, capped at the number of attributes (as in
scikit-ExSTraCS).

**Seeding**: rules from a rule file enter the population before the first iteration with the
same parameters as a fresh covering rule (experience 0, `init_fitness`), configurable starting
numerosity (default 1), their provenance and external id. In the default mode they are subject
to GA and deletion like any other rule. In fixed-conditions mode covering, GA, subsumption and
deletion are disabled.

### Points to verify against scikit-ExSTraCS (milestone M2, golden vectors)

The implementation follows the papers, not the GPL code. These behaviours may differ from the
reference implementation and must be checked with golden vectors before experiments:

- interval bounds inclusive vs exclusive; missing values matching;
- "more general" test for continuous attributes in subsumption;
- deletion vote fitness normalisation (per rule vs per micro-classifier);
- offspring parameter initialisation (accuracy, fitness reduction);
- details of continuous mutation and crossover;
- RSL auto value for continuous attributes;
- prediction tie-breaking and the no-match fallback.

## 4. Work breakdown

Execution uses the cavecrew agents. The main agent fixes all public types and signatures first,
so parallel builders implement against a stable API.

| Step | Work | Owner |
| --- | --- | --- |
| 0 | Map acs2-rust conventions (edition, deps, lints, RNG, CLI) | cavecrew-investigator |
| 1 | Workspace, crate manifest, `.gitignore`, `lib.rs`, `error.rs`, `config.rs`, typed skeletons of every module | main agent |
| 2a | `data.rs` + `multiplexer.rs` bodies and unit tests | cavecrew-builder |
| 2b | `classifier.rs` bodies and unit tests | cavecrew-builder |
| 2c | `ga.rs` + `metrics.rs` bodies and unit tests | cavecrew-builder |
| 2d | `rules.rs` + `snapshot.rs` bodies and unit tests | cavecrew-builder |
| 2e | `main.rs` CLI | cavecrew-builder |
| 2f | `population.rs` + `model.rs` (learning loop) | main agent |
| 3 | Integration tests and example rule file | cavecrew-builder |
| 4 | `cargo build`, `cargo test`, `cargo clippy -D warnings`, multiplexer runs in release | main agent |
| 5 | Review of the whole crate | cavecrew-reviewer |
| 6 | Fix findings, README usage guide | main agent / cavecrew-builder |

Nothing is committed or pushed; the repository is only initialised.

Status (2026-11-13): steps 0–6 done. All acceptance checks in section 5 pass, including the
ignored 11-bit test in release mode. Notes from review and runs:

- With the scikit-ExSTraCS default `nu = 1` the 6-bit multiplexer reaches 97–100% but does not
  stay at 100%; `nu = 5` gives a stable 100%. Check against scikit-ExSTraCS at the same settings.
- `tracked_accuracy` counts the majority-class fallback for uncovered instances, the same way
  the test metrics do.
- For numeric columns treated as discrete, categories are the raw CSV strings, so "1" and "1.0"
  are different values.

## 5. Acceptance checks for this step

- `cargo build`, `cargo test` and `cargo clippy --all-targets -- -D warnings` pass.
- 6-bit multiplexer: test accuracy >= 99% for fixed seeds (integration test).
- 11-bit multiplexer: solved in a release run (ignored test, run on demand).
- Same seed gives an identical population snapshot.
- Seeding the 8 optimal 6-multiplexer rules in fixed-conditions mode gives 100% accuracy;
  invalid rules are rejected with the right reason and counted.
- Snapshot save/load round trip gives identical predictions.

## 6. Roadmap after this step (from the project spec)

1. M1: 20-bit multiplexer at a budget comparable to scikit-ExSTraCS; timing benchmark.
2. M2: golden vectors from scikit-ExSTraCS for matching, updates, subsumption and voting; fix
   differences listed in section 3; real datasets (K2); missing-value parity; P1 components
   (EK, attribute tracking, RSL options).
3. Python LLM rule generator (Ollama, frozen prompts, response cache), JSON Schema for the rule
   format, Python validator consistent with the Rust one (K3).
4. M3: variants V0–V3 (random rules generator, optional rule pre-evaluation pass), lineage
   analysis, experiment scripts.
5. M4: corrupted-rules experiment (T1–T4) on the multiplexer and one real dataset; pilot report.
