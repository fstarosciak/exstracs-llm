use std::collections::BTreeSet;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use exstracs::multiplexer::multiplexer_table;
use exstracs::rules::{validate, RejectReason, Rejection};
use exstracs::{Config, Dataset, ExSTraCS, LoadOptions, Origin, RuleFile, SeedRule, Snapshot};

const OPTIMAL_RULES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../examples/mux6_optimal_rules.json"
);

const INVALID_RULES: &str = r#"{
  "dataset": "multiplexer-6",
  "source": "llm",
  "rules": [
    {"id": "valid", "class": "1", "conditions": [
      {"attribute": "A0", "type": "discrete", "value": "0"},
      {"attribute": "A1", "type": "discrete", "value": "0"},
      {"attribute": "R0", "type": "discrete", "value": "1"}
    ]},
    {"id": "copy", "class": "1", "conditions": [
      {"attribute": "R0", "type": "discrete", "value": "1"},
      {"attribute": "A1", "type": "discrete", "value": "0"},
      {"attribute": "A0", "type": "discrete", "value": "0"}
    ]},
    {"id": "unknown-attribute", "class": "1", "conditions": [
      {"attribute": "R4", "type": "discrete", "value": "1"}
    ]},
    {"id": "unknown-value", "class": "0", "conditions": [
      {"attribute": "A0", "type": "discrete", "value": "2"}
    ]},
    {"id": "type-mismatch", "class": "0", "conditions": [
      {"attribute": "A1", "type": "continuous", "low": 0, "high": 1}
    ]},
    {"id": "unknown-class", "class": "2", "conditions": [
      {"attribute": "R1", "type": "discrete", "value": "1"}
    ]},
    {"id": "too-specific", "class": "1", "conditions": [
      {"attribute": "A0", "type": "discrete", "value": "0"},
      {"attribute": "A1", "type": "discrete", "value": "1"},
      {"attribute": "R1", "type": "discrete", "value": "1"},
      {"attribute": "R2", "type": "discrete", "value": "0"}
    ]},
    {"id": "malformed", "class": "1", "conditions": "A0 = 1 and R2 = 1"}
  ]
}"#;

fn six_multiplexer() -> Dataset {
    let table = multiplexer_table(2, None, 0).unwrap();
    Dataset::from_table(&table, &LoadOptions::default()).unwrap()
}

fn fixed() -> Config {
    Config {
        fixed_conditions: true,
        ..Config::default()
    }
}

fn evolving(seed: u64) -> Config {
    Config {
        population_size: 500,
        nu: 5,
        seed,
        ..Config::default()
    }
}

fn optimal_file() -> RuleFile {
    RuleFile::load(Path::new(OPTIMAL_RULES)).unwrap()
}

fn model_and_rules(config: Config, data: &Dataset) -> (ExSTraCS, Vec<SeedRule>) {
    let model = ExSTraCS::new(config, data).unwrap();
    let validated = validate(
        &optimal_file(),
        model.schema(),
        model.rule_specificity_limit(),
    );
    assert!(
        validated.report.rejected.is_empty(),
        "{:?}",
        validated.report.rejected
    );
    (model, validated.rules)
}

fn corrupted(rules: &[SeedRule]) -> Vec<SeedRule> {
    rules
        .iter()
        .map(|rule| SeedRule {
            class: 1 - rule.class,
            ..rule.clone()
        })
        .collect()
}

#[test]
fn optimal_rules_validate_and_keep_their_ids() {
    let data = six_multiplexer();
    let mut model = ExSTraCS::new(fixed(), &data).unwrap();
    let file = optimal_file();
    let file_ids: Vec<String> = file
        .rules
        .iter()
        .map(|rule| rule["id"].as_str().unwrap().to_string())
        .collect();
    let validated = validate(&file, model.schema(), model.rule_specificity_limit());
    assert_eq!(validated.report.source, Origin::Expert);
    assert_eq!(validated.report.total, 8);
    assert_eq!(validated.report.accepted, 8);
    assert!(validated.report.rejected.is_empty());
    assert!(validated
        .rules
        .iter()
        .all(|rule| rule.origin == Origin::Expert));
    let rule_ids: Vec<String> = validated
        .rules
        .iter()
        .map(|rule| rule.external_id.clone())
        .collect();
    assert_eq!(rule_ids, file_ids);
    assert_eq!(model.seed(&validated.rules, 1), 8);
    assert_eq!(model.population_stats().seeded_alive, 8);
    let classifiers = model.population().classifiers();
    assert_eq!(classifiers.len(), 8);
    let mut seen = BTreeSet::new();
    for classifier in classifiers {
        assert_eq!(classifier.origin, Origin::Expert);
        let id = classifier
            .external_id
            .clone()
            .expect("seeded classifier keeps its external id");
        assert!(file_ids.contains(&id));
        assert_eq!(classifier.seed_lineage, vec![id.clone()]);
        seen.insert(id);
    }
    assert_eq!(seen.len(), 8);
}

#[test]
fn fixed_optimal_rules_are_perfect_before_and_after_training() {
    let data = six_multiplexer();
    let (mut model, rules) = model_and_rules(fixed(), &data);
    assert_eq!(model.seed(&rules, 1), 8);
    let before = model.evaluate(&data);
    assert_eq!(before.accuracy, 1.0);
    assert_eq!(before.coverage, 1.0);
    let stats = model.train(&data, 640).unwrap();
    assert_eq!(stats.correct, 640);
    assert_eq!(stats.covering, 0);
    assert_eq!(stats.ga_runs, 0);
    let expected: Vec<_> = rules
        .iter()
        .map(|rule| (rule.condition.clone(), rule.class))
        .collect();
    let classifiers = model.population().classifiers();
    let actual: Vec<_> = classifiers
        .iter()
        .map(|classifier| (classifier.condition.clone(), classifier.class))
        .collect();
    assert_eq!(actual, expected);
    for classifier in classifiers {
        assert_eq!(classifier.experience, 80);
        assert_eq!(classifier.accuracy, 1.0);
        assert!(!matches!(classifier.origin, Origin::Covering | Origin::Ga));
    }
    let after = model.evaluate(&data);
    assert_eq!(after.accuracy, 1.0);
    assert_eq!(after.coverage, 1.0);
}

#[test]
fn evolving_from_optimal_rules_keeps_the_solution() {
    let data = six_multiplexer();
    let (mut model, rules) = model_and_rules(evolving(1), &data);
    assert_eq!(model.seed(&rules, 1), 8);
    model.train(&data, 5_000).unwrap();
    let metrics = model.evaluate(&data);
    assert!(
        metrics.accuracy >= 0.99,
        "accuracy {} after evolving from optimal rules",
        metrics.accuracy
    );
    let ids: BTreeSet<&str> = rules.iter().map(|rule| rule.external_id.as_str()).collect();
    for classifier in model.population().classifiers() {
        if classifier.origin == Origin::Ga && !classifier.seed_lineage.is_empty() {
            assert!(
                classifier
                    .seed_lineage
                    .iter()
                    .all(|id| ids.contains(id.as_str())),
                "unexpected lineage {:?}",
                classifier.seed_lineage
            );
        }
    }
}

#[test]
fn fixed_corrupted_rules_are_always_wrong() {
    let data = six_multiplexer();
    let (mut model, rules) = model_and_rules(fixed(), &data);
    assert_eq!(model.seed(&corrupted(&rules), 1), 8);
    model.train(&data, 640).unwrap();
    let metrics = model.evaluate(&data);
    assert_eq!(metrics.accuracy, 0.0);
    assert_eq!(metrics.coverage, 1.0);
    assert!(model
        .population()
        .classifiers()
        .iter()
        .all(|classifier| classifier.experience == 80 && classifier.correct_count == 0));
}

#[test]
fn evolution_repairs_corrupted_rules() {
    let data = six_multiplexer();
    let (mut model, rules) = model_and_rules(evolving(1), &data);
    assert_eq!(model.seed(&corrupted(&rules), 1), 8);
    model.train(&data, 20_000).unwrap();
    let metrics = model.evaluate(&data);
    assert!(
        metrics.accuracy >= 0.99,
        "accuracy {} after evolving from corrupted rules",
        metrics.accuracy
    );
}

#[test]
fn seeding_numerosity_and_repeated_seeding_merge() {
    let data = six_multiplexer();
    let (mut model, rules) = model_and_rules(fixed(), &data);
    assert_eq!(model.seed(&rules, 3), 8);
    assert_eq!(model.population().len(), 8);
    assert_eq!(model.population().micro_size(), 24);

    let (mut twice, rules) = model_and_rules(fixed(), &data);
    assert_eq!(twice.seed(&rules, 1), 8);
    assert_eq!(twice.seed(&rules, 1), 0);
    assert_eq!(twice.population().len(), 8);
    assert!(twice
        .population()
        .classifiers()
        .iter()
        .all(|classifier| classifier.numerosity == 2));
    assert_eq!(twice.population().micro_size(), 16);
}

#[test]
fn invalid_rules_are_reported_with_reasons() {
    let data = six_multiplexer();
    let file = RuleFile::from_json_str(INVALID_RULES).unwrap();
    let validated = validate(&file, &data.schema, 3);
    let report = &validated.report;
    assert_eq!(report.source, Origin::Llm);
    assert_eq!(report.total, 8);
    assert_eq!(report.accepted, 1);
    assert_eq!(report.rejected.len(), 7);
    assert_eq!(validated.rules.len(), 1);
    assert_eq!(validated.rules[0].external_id, "valid");
    assert_eq!(validated.rules[0].class, 1);
    assert_eq!(validated.rules[0].condition.len(), 3);
    assert_eq!(validated.rules[0].origin, Origin::Llm);
    let expected = [
        Rejection {
            index: 1,
            id: "copy".to_string(),
            reason: RejectReason::Duplicate {
                of: "valid".to_string(),
            },
        },
        Rejection {
            index: 2,
            id: "unknown-attribute".to_string(),
            reason: RejectReason::UnknownAttribute {
                attribute: "R4".to_string(),
            },
        },
        Rejection {
            index: 3,
            id: "unknown-value".to_string(),
            reason: RejectReason::UnknownValue {
                attribute: "A0".to_string(),
                value: "2".to_string(),
            },
        },
        Rejection {
            index: 4,
            id: "type-mismatch".to_string(),
            reason: RejectReason::TypeMismatch {
                attribute: "A1".to_string(),
            },
        },
        Rejection {
            index: 5,
            id: "unknown-class".to_string(),
            reason: RejectReason::UnknownClass {
                class: "2".to_string(),
            },
        },
        Rejection {
            index: 6,
            id: "too-specific".to_string(),
            reason: RejectReason::TooSpecific {
                specified: 4,
                limit: 3,
            },
        },
    ];
    assert_eq!(report.rejected[..6], expected);
    let malformed = &report.rejected[6];
    assert_eq!(malformed.index, 7);
    assert_eq!(malformed.id, "malformed");
    assert!(matches!(malformed.reason, RejectReason::Malformed { .. }));
}

#[test]
fn snapshot_round_trip_preserves_the_model() {
    let data = six_multiplexer();
    let (mut model, rules) = model_and_rules(evolving(3), &data);
    assert_eq!(model.seed(&rules, 1), 8);
    model.train(&data, 2_000).unwrap();
    let original = model.snapshot();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "exstracs-seeding-snapshot-{}-{nanos}.json",
        std::process::id()
    ));
    original.save(&path).unwrap();
    let loaded = Snapshot::load(&path);
    std::fs::remove_file(&path).unwrap();
    let restored = ExSTraCS::from_snapshot(loaded.unwrap()).unwrap();
    assert_eq!(restored.snapshot(), original);
    for row in &data.rows {
        assert_eq!(restored.predict(row), model.predict(row));
    }
}
