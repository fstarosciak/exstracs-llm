# exstracs-llm

Research code for the master's thesis *Evolutionary Refinement of LLM-Generated Rules in the
ExSTraCS Learning Classifier System*.

This repository holds:

- the `exstracs` crate, a Rust implementation of the ExSTraCS supervised learning classifier
  system (library and CLI), which can start from a population of rules loaded from a file;
- `examples/`, small input files;
- `docs/PLAN.md`, the implementation plan and roadmap.

Later steps add the LLM rule generator (Python), the experiment scripts and the reports.

The crate does not depend on `acs2-rust` crates. It follows that workspace's conventions
(edition 2021, `clippy.toml`, seeded ChaCha8 RNG, no code comments), so it can be moved into
`acs2-rust` as a new workspace member.

## Acknowledgements

The project conventions follow [acs2-rust](https://github.com/lyskawson/acs2-rust)
by Aleksander Łyskawa (MIT License, © 2026 Aleksander Łyskawa); `clippy.toml` is copied
from that repository unchanged. No other code from acs2-rust is included.

## Build and test

Requires a stable Rust toolchain.

```sh
cargo build --release
cargo test
cargo test --release -- --ignored        # slower runs (11-bit multiplexer)
cargo clippy --all-targets -- -D warnings
```

The binary is `target/release/exstracs`. The examples below use `exstracs` for it
(`cargo run --release --` also works).

## What the model does

ExSTraCS evolves a population of rules of the form "conditions on some attributes → class".
Each training iteration takes one instance and does the following:

1. It builds the match set (rules whose conditions hold) and the correct set (matching rules
   with the right class).
2. If the correct set is empty, covering creates a new rule from the instance.
3. It updates each matching rule's accuracy, fitness (`accuracy^nu`) and estimated niche size.
4. It optionally runs correct-set subsumption.
5. It runs a genetic algorithm in the correct set: selection, crossover, mutation and GA
   subsumption.
6. While the population is larger than `N`, it deletes rules.

To predict, each matching rule votes for its class with weight `fitness * numerosity`.
Rules use the ALKR representation: a condition stores only the attributes it specifies, as a
value (discrete attributes) or an interval (continuous attributes). A missing value matches any
condition. Details and the points still to check against scikit-ExSTraCS are in
[docs/PLAN.md](docs/PLAN.md).

## CLI

```
exstracs train           train a model (optionally seeded with rules) and report a learning curve
exstracs predict         evaluate a saved population on a labelled CSV file
exstracs validate-rules  validate a seed rule file against a training data schema
exstracs gen-mux         generate a multiplexer benchmark CSV file
```

`exstracs <command> --help` lists all options.

### Data

Input is a CSV file with a header row and a class column (default name `class`, change it with
`--class-column`). The attribute types are inferred from the training file:

- an attribute is **continuous** if every non-missing value is numeric and it has more than
  `--discrete-limit` (default 10) distinct values;
- otherwise it is **discrete**.

To set a type yourself, use `--discrete a,b` or `--continuous c`. To drop columns such as ids,
use `--ignore id`. The cells `""`, `NA`, `NaN` and `?` count as missing. A test file is encoded
with the training schema. A discrete value that does not appear in the training data matches no
rule that specifies that attribute.

### Train

```sh
exstracs gen-mux --address-bits 2 --out runs/mux6.csv          # all 64 rows of the 6-bit multiplexer
exstracs train --train runs/mux6.csv --test runs/mux6.csv \
  --iterations 20000 --population-size 500 --nu 5 --seed 1 --out runs/mux6-v0
```

The run prints a JSON summary to stdout. With `--out DIR` it also writes these files:

| File | Content |
| --- | --- |
| `summary.json` | config, data sizes, RSL, runtime, train/test metrics, population statistics, seeding counts |
| `curve.csv` | one row every `--report-every` iterations: `iteration, tracked_accuracy, test_accuracy, test_balanced_accuracy, test_coverage, macro_size, micro_size, mean_specificity, seeded_alive, seed_descendants` |
| `population.json` | population snapshot: schema, config, every rule with its conditions, class, statistics, origin, parents, seed lineage |
| `rules_report.json` | validation report of the seed rule file (only with `--rules`) |

`tracked_accuracy` is the share of correct predictions the model made on each training instance
just before learning from it, over the iterations since the previous row. The metrics are
accuracy, balanced accuracy, macro-F1 and coverage. Coverage is the share of instances that
match at least one rule; an uncovered instance gets the majority training class.

Hyperparameters default to the scikit-ExSTraCS defaults (`N = 1000`, `nu = 1`, `chi = 0.8`,
`mu = 0.04`, `theta_GA = 25`, `theta_del = 20`, `theta_sub = 20`, `acc_sub = 0.99`, `beta = 0.2`,
`delta = 0.1`, `init_fitness = 0.01`, `fitness_reduction = 0.1`, tournament selection with
`theta_sel = 0.5`, GA subsumption on, correct-set subsumption off). Change them with flags such
as `--population-size` or `--nu`, or pass a JSON file with `--config` (any subset of the
`Config` fields; flags win over the file). On noise-free problems such as the multiplexer use
`--nu 5`. The rule specificity limit (`--rsl`) defaults to the smallest `i` with
`avgStateCount^i >= nTrainingInstances`.

The same `--seed` gives the same population, byte for byte.

### Seed the population with rules

```sh
exstracs validate-rules --train runs/mux6.csv --rules examples/mux6_optimal_rules.json

# V1: seeded rules evolve like any other rule
exstracs train --train runs/mux6.csv --test runs/mux6.csv \
  --rules examples/mux6_optimal_rules.json --nu 5 --iterations 5000 --out runs/mux6-v1

# V2a: seeded conditions stay fixed; covering, GA, subsumption and deletion are off,
# only accuracy and fitness (the voting weights) are learned
exstracs train --train runs/mux6.csv --test runs/mux6.csv \
  --rules examples/mux6_optimal_rules.json --fixed-conditions --iterations 640 --out runs/mux6-v2a
```

Seeded rules start like a new covering rule: experience 0, fitness `init_fitness`, numerosity
`--seed-numerosity` (default 1). They keep their origin (`llm`, `random`, `tree` or `expert`)
and their id from the file. Rules created later by the GA inherit the ids of their seeded
ancestors in `seed_lineage`. The learning curve columns `seeded_alive` and `seed_descendants`
count seeded rules that are still in the population and GA rules that descend from them.

### Rule file format

```json
{
  "dataset": "adult",
  "source": "llm",
  "model": "model name and version",
  "prompt_hash": "sha256:...",
  "rules": [
    {
      "id": "llm-001",
      "conditions": [
        { "attribute": "education-num", "type": "continuous", "low": 13, "high": null },
        { "attribute": "marital-status", "type": "discrete", "value": "Married-civ-spouse" }
      ],
      "class": ">50K",
      "rationale": "optional, not used by the model"
    }
  ]
}
```

- `source` is one of `llm`, `random`, `tree`, `expert`. `dataset`, `model`, `prompt_hash`, `id`
  and `rationale` are optional; a rule without an `id` gets `rule-<index>`.
- A discrete condition has one `value`, which is a string exactly as it appears in the CSV.
- A `null` (or missing) interval bound means no limit; it is replaced by the training minimum or
  maximum.

Every rule is checked. Invalid rules are **rejected and counted, never repaired**. Each rejected
rule gets the first failing reason:

| Reason | Meaning |
| --- | --- |
| `malformed` | the rule does not match the format |
| `unknown_attribute` | the attribute is not in the data |
| `repeated_attribute` | the same attribute appears twice in one rule |
| `type_mismatch` | discrete condition on a continuous attribute or the other way round |
| `unknown_value` | the discrete value does not occur in the training data |
| `non_finite_bound` | an interval bound is not finite |
| `inverted_interval` | `low > high` |
| `outside_data_range` | the interval does not overlap the training range |
| `unknown_class` | the class does not occur in the training data |
| `too_specific` | more conditions than the rule specificity limit |
| `duplicate` | same conditions and class as an earlier accepted rule |

### Predict

```sh
exstracs predict --model runs/mux6-v1/population.json --data runs/mux6.csv --out runs/predictions.csv
```

This prints the metrics as JSON. `--out` writes `row,actual,predicted,covered` with class
names. A snapshot can also be loaded in code (`ExSTraCS::from_snapshot`) and trained further;
that continuation uses a fresh RNG stream, so it is not bit-identical to an uninterrupted run.

## Library use

```rust
use exstracs::{rules, Config, Dataset, ExSTraCS, LoadOptions, RuleFile};
use std::path::Path;

let options = LoadOptions::default();
let train = Dataset::from_csv(Path::new("train.csv"), &options)?;
let test = Dataset::from_csv_with_schema(Path::new("test.csv"), &train.schema, &options)?;

let mut model = ExSTraCS::new(Config { nu: 5, seed: 1, ..Config::default() }, &train)?;
let file = RuleFile::load(Path::new("rules.json"))?;
let validated = rules::validate(&file, model.schema(), model.rule_specificity_limit());
model.seed(&validated.rules, 1);

model.train(&train, 50_000)?;
println!("{:?}", model.evaluate(&test));
model.snapshot().save(Path::new("population.json"))?;
```

## Verification so far

These are release runs on a laptop, using `--nu 5`:

| Problem | Setting | Result |
| --- | --- | --- |
| 6-bit multiplexer | N = 500, 20k iterations, seeds 1–5 | 100% accuracy from 10k iterations onwards |
| 11-bit multiplexer | N = 1000, 100k iterations | 100% from 40k iterations, 0.7 s |
| 20-bit multiplexer | N = 2000, 20k training / 5k test samples, 500k iterations | 100% test accuracy from 100k iterations, 13 s |
| 6-bit multiplexer, optimal rules, `--fixed-conditions` | 640 iterations | 100% from iteration 0, population unchanged |

Comparison with scikit-ExSTraCS (golden vectors, real datasets, timing) is the next milestone;
see [docs/PLAN.md](docs/PLAN.md).

## License

MIT, see [LICENSE](LICENSE).
