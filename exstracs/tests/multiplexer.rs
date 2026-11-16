use exstracs::multiplexer::multiplexer_table;
use exstracs::{Config, Dataset, ExSTraCS, LoadOptions};

fn complete_multiplexer(address_bits: usize) -> Dataset {
    let table = multiplexer_table(address_bits, None, 0).unwrap();
    Dataset::from_table(&table, &LoadOptions::default()).unwrap()
}

fn mux_config(population_size: u64, seed: u64) -> Config {
    Config {
        population_size,
        nu: 5,
        seed,
        ..Config::default()
    }
}

fn trained(data: &Dataset, config: Config, iterations: u64) -> ExSTraCS {
    let mut model = ExSTraCS::new(config, data).unwrap();
    model.train(data, iterations).unwrap();
    model
}

#[test]
fn six_bit_multiplexer_is_solved() {
    let data = complete_multiplexer(2);
    assert_eq!(data.len(), 64);
    for seed in 1..=3 {
        let metrics = trained(&data, mux_config(500, seed), 20_000).evaluate(&data);
        assert!(
            metrics.accuracy >= 0.99,
            "seed {seed}: accuracy {} on the complete 6-multiplexer",
            metrics.accuracy
        );
    }
}

#[test]
#[ignore = "slow in debug builds; run with --release -- --ignored"]
fn eleven_bit_multiplexer_is_solved() {
    let data = complete_multiplexer(3);
    assert_eq!(data.len(), 2048);
    let metrics = trained(&data, mux_config(1000, 1), 100_000).evaluate(&data);
    assert!(
        metrics.accuracy >= 0.99,
        "accuracy {} on the complete 11-multiplexer",
        metrics.accuracy
    );
}

#[test]
fn same_seed_gives_identical_population() {
    let data = complete_multiplexer(2);
    let first = trained(&data, mux_config(500, 7), 3_000).snapshot();
    let second = trained(&data, mux_config(500, 7), 3_000).snapshot();
    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap()
    );
    let other = trained(&data, mux_config(500, 8), 3_000).snapshot();
    assert_ne!(first, other);
    assert_ne!(first.classifiers, other.classifiers);
}

#[test]
fn tracked_training_accuracy_is_high_after_learning() {
    let data = complete_multiplexer(2);
    let mut model = trained(&data, mux_config(500, 1), 20_000);
    let stats = model.train(&data, 2_000).unwrap();
    assert_eq!(stats.iterations, 2_000);
    let tracked = stats.correct as f64 / stats.iterations as f64;
    assert!(
        tracked >= 0.95,
        "tracked accuracy {tracked} ({} of {} correct)",
        stats.correct,
        stats.iterations
    );
}
