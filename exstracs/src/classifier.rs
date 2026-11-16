use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::data::{Attribute, Schema, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    Covering,
    Ga,
    Llm,
    Random,
    Tree,
    Expert,
}

impl Origin {
    pub fn is_seeded(self) -> bool {
        !matches!(self, Origin::Covering | Origin::Ga)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Predicate {
    Equals(u32),
    Interval { low: f64, high: f64 },
}

impl Predicate {
    pub fn matches(&self, value: Value) -> bool {
        match (*self, value) {
            (_, Value::Missing) => true,
            (Predicate::Equals(expected), Value::Discrete(actual)) => expected == actual,
            (Predicate::Interval { low, high }, Value::Continuous(actual)) => {
                low <= actual && actual <= high
            }
            _ => false,
        }
    }

    pub fn contains(&self, other: &Predicate) -> bool {
        match (*self, *other) {
            (Predicate::Equals(outer), Predicate::Equals(inner)) => outer == inner,
            (
                Predicate::Interval {
                    low: outer_low,
                    high: outer_high,
                },
                Predicate::Interval {
                    low: inner_low,
                    high: inner_high,
                },
            ) => outer_low <= inner_low && outer_high >= inner_high,
            _ => false,
        }
    }

    pub fn cover<R: Rng + ?Sized>(
        attribute: &Attribute,
        value: Value,
        rng: &mut R,
    ) -> Option<Predicate> {
        match value {
            Value::Discrete(actual) if attribute.is_discrete() => Some(Predicate::Equals(actual)),
            Value::Continuous(actual) if !attribute.is_discrete() => {
                let percent = f64::from(rng.gen_range(25..=75u32));
                let radius = percent * 0.01 * attribute.range() / 2.0;
                Some(Predicate::Interval {
                    low: actual - radius,
                    high: actual + radius,
                })
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Condition {
    specs: Vec<(usize, Predicate)>,
}

impl Condition {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_specs(specs: Vec<(usize, Predicate)>) -> Self {
        let mut specs = specs;
        specs.sort_by_key(|(attribute, _)| *attribute);
        for pair in specs.windows(2) {
            assert_ne!(
                pair[0].0, pair[1].0,
                "duplicate attribute {} in condition",
                pair[0].0
            );
        }
        Self { specs }
    }

    pub fn specs(&self) -> &[(usize, Predicate)] {
        &self.specs
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn get(&self, attribute: usize) -> Option<&Predicate> {
        self.search(attribute)
            .ok()
            .map(|index| &self.specs[index].1)
    }

    pub fn set(&mut self, attribute: usize, predicate: Predicate) {
        match self.search(attribute) {
            Ok(index) => self.specs[index].1 = predicate,
            Err(index) => self.specs.insert(index, (attribute, predicate)),
        }
    }

    pub fn remove(&mut self, attribute: usize) -> Option<Predicate> {
        self.search(attribute)
            .ok()
            .map(|index| self.specs.remove(index).1)
    }

    pub fn matches(&self, row: &[Value]) -> bool {
        self.specs
            .iter()
            .all(|(attribute, predicate)| predicate.matches(row[*attribute]))
    }

    pub fn is_more_general(&self, other: &Condition) -> bool {
        self.specs.iter().all(|(attribute, predicate)| {
            other
                .get(*attribute)
                .is_some_and(|inner| predicate.contains(inner))
        }) && self != other
    }

    pub fn cover<R: Rng + ?Sized>(
        row: &[Value],
        schema: &Schema,
        rsl: usize,
        rng: &mut R,
    ) -> Condition {
        assert!(rsl >= 1, "rule specificity limit must be at least 1");
        let candidates: Vec<usize> = row
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != Value::Missing)
            .map(|(attribute, _)| attribute)
            .collect();
        let mut condition = Condition::new();
        if candidates.is_empty() {
            return condition;
        }
        let specificity = rng.gen_range(1..=rsl);
        let amount = specificity.min(candidates.len());
        let mut picked: Vec<usize> = rand::seq::index::sample(rng, candidates.len(), amount)
            .into_iter()
            .map(|index| candidates[index])
            .collect();
        picked.sort_unstable();
        for attribute in picked {
            if let Some(predicate) =
                Predicate::cover(&schema.attributes[attribute], row[attribute], rng)
            {
                condition.set(attribute, predicate);
            }
        }
        condition
    }

    fn search(&self, attribute: usize) -> Result<usize, usize> {
        self.specs
            .binary_search_by_key(&attribute, |(candidate, _)| *candidate)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Classifier {
    pub id: u64,
    pub condition: Condition,
    pub class: u32,
    pub numerosity: u32,
    pub experience: u32,
    pub correct_count: u32,
    pub accuracy: f64,
    pub fitness: f64,
    pub avg_match_set_size: f64,
    pub ga_timestamp: u64,
    pub birth_iteration: u64,
    pub origin: Origin,
    pub external_id: Option<String>,
    pub parents: Vec<u64>,
    pub seed_lineage: Vec<String>,
}

impl Classifier {
    pub fn new(
        id: u64,
        condition: Condition,
        class: u32,
        origin: Origin,
        iteration: u64,
        fitness: f64,
        avg_match_set_size: f64,
    ) -> Self {
        Self {
            id,
            condition,
            class,
            numerosity: 1,
            experience: 0,
            correct_count: 0,
            accuracy: 0.0,
            fitness,
            avg_match_set_size,
            ga_timestamp: iteration,
            birth_iteration: iteration,
            origin,
            external_id: None,
            parents: Vec::new(),
            seed_lineage: Vec::new(),
        }
    }

    pub fn same_rule(&self, other: &Classifier) -> bool {
        self.class == other.class && self.condition == other.condition
    }

    pub fn is_subsumer(&self, theta_sub: u32, acc_sub: f64) -> bool {
        self.experience > theta_sub && self.accuracy > acc_sub
    }

    pub fn subsumes(&self, other: &Classifier, theta_sub: u32, acc_sub: f64) -> bool {
        self.class == other.class
            && self.is_subsumer(theta_sub, acc_sub)
            && self.condition.is_more_general(&other.condition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::AttributeKind;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn discrete(name: &str, values: &[&str]) -> Attribute {
        Attribute {
            name: name.to_string(),
            kind: AttributeKind::Discrete {
                values: values.iter().map(|value| value.to_string()).collect(),
            },
        }
    }

    fn continuous(name: &str, min: f64, max: f64) -> Attribute {
        Attribute {
            name: name.to_string(),
            kind: AttributeKind::Continuous { min, max },
        }
    }

    fn schema() -> Schema {
        Schema {
            attributes: vec![
                discrete("color", &["red", "green", "blue"]),
                continuous("size", 0.0, 10.0),
                discrete("shape", &["round", "square"]),
                continuous("weight", -5.0, 5.0),
            ],
            class_column: "class".to_string(),
            classes: vec!["no".to_string(), "yes".to_string()],
        }
    }

    fn row() -> Vec<Value> {
        vec![
            Value::Discrete(1),
            Value::Continuous(4.0),
            Value::Discrete(0),
            Value::Continuous(-1.5),
        ]
    }

    fn interval(low: f64, high: f64) -> Predicate {
        Predicate::Interval { low, high }
    }

    fn untouched(rng: &mut ChaCha8Rng, reference: &mut ChaCha8Rng) -> bool {
        rng.gen::<u64>() == reference.gen::<u64>()
    }

    fn rule(condition: Condition, class: u32, experience: u32, accuracy: f64) -> Classifier {
        let mut classifier = Classifier::new(0, condition, class, Origin::Covering, 0, 0.5, 1.0);
        classifier.experience = experience;
        classifier.accuracy = accuracy;
        classifier
    }

    #[test]
    fn origin_seeded_flags() {
        assert!(!Origin::Covering.is_seeded());
        assert!(!Origin::Ga.is_seeded());
        assert!(Origin::Llm.is_seeded());
        assert!(Origin::Random.is_seeded());
        assert!(Origin::Tree.is_seeded());
        assert!(Origin::Expert.is_seeded());
    }

    #[test]
    fn equals_matches_same_discrete_value_and_missing() {
        let predicate = Predicate::Equals(2);
        assert!(predicate.matches(Value::Discrete(2)));
        assert!(!predicate.matches(Value::Discrete(1)));
        assert!(!predicate.matches(Value::Continuous(2.0)));
        assert!(predicate.matches(Value::Missing));
    }

    #[test]
    fn interval_bounds_are_inclusive() {
        let predicate = interval(1.0, 3.0);
        assert!(predicate.matches(Value::Continuous(1.0)));
        assert!(predicate.matches(Value::Continuous(2.0)));
        assert!(predicate.matches(Value::Continuous(3.0)));
        assert!(!predicate.matches(Value::Continuous(0.999)));
        assert!(!predicate.matches(Value::Continuous(3.001)));
        assert!(!predicate.matches(Value::Discrete(2)));
        assert!(predicate.matches(Value::Missing));
    }

    #[test]
    fn contains_requires_same_kind_and_enclosure() {
        assert!(Predicate::Equals(1).contains(&Predicate::Equals(1)));
        assert!(!Predicate::Equals(1).contains(&Predicate::Equals(2)));
        let outer = interval(0.0, 10.0);
        assert!(outer.contains(&interval(2.0, 5.0)));
        assert!(outer.contains(&outer));
        assert!(outer.contains(&interval(0.0, 10.0)));
        assert!(!outer.contains(&interval(-1.0, 5.0)));
        assert!(!outer.contains(&interval(2.0, 11.0)));
        assert!(!interval(2.0, 5.0).contains(&outer));
        assert!(!Predicate::Equals(1).contains(&outer));
        assert!(!outer.contains(&Predicate::Equals(1)));
    }

    #[test]
    fn cover_discrete_returns_equals_without_rng() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut reference = rng.clone();
        let attribute = discrete("color", &["red", "green"]);
        assert_eq!(
            Predicate::cover(&attribute, Value::Discrete(1), &mut rng),
            Some(Predicate::Equals(1))
        );
        assert!(untouched(&mut rng, &mut reference));
    }

    #[test]
    fn cover_missing_and_kind_mismatch_return_none() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut reference = rng.clone();
        let nominal = discrete("color", &["red", "green"]);
        let numeric = continuous("size", 0.0, 10.0);
        assert_eq!(Predicate::cover(&nominal, Value::Missing, &mut rng), None);
        assert_eq!(Predicate::cover(&numeric, Value::Missing, &mut rng), None);
        assert_eq!(
            Predicate::cover(&nominal, Value::Continuous(1.0), &mut rng),
            None
        );
        assert_eq!(
            Predicate::cover(&numeric, Value::Discrete(1), &mut rng),
            None
        );
        assert!(untouched(&mut rng, &mut reference));
    }

    #[test]
    fn cover_continuous_uses_exactly_one_draw() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut reference = rng.clone();
        let attribute = continuous("size", 0.0, 10.0);
        let percent = f64::from(reference.gen_range(25..=75u32));
        let radius = percent * 0.01 * 10.0 / 2.0;
        assert_eq!(
            Predicate::cover(&attribute, Value::Continuous(4.0), &mut rng),
            Some(interval(4.0 - radius, 4.0 + radius))
        );
        assert!(untouched(&mut rng, &mut reference));
    }

    #[test]
    fn cover_continuous_radius_within_bounds_and_contains_value() {
        let attribute = continuous("size", 2.0, 12.0);
        let half_range = attribute.range() / 2.0;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for step in 0..500 {
            let value = 2.0 + f64::from(step % 11);
            let Some(Predicate::Interval { low, high }) =
                Predicate::cover(&attribute, Value::Continuous(value), &mut rng)
            else {
                panic!("expected interval");
            };
            let radius = (high - low) / 2.0;
            assert!(radius >= 0.25 * half_range - 1e-9);
            assert!(radius <= 0.75 * half_range + 1e-9);
            assert!(((low + high) / 2.0 - value).abs() < 1e-9);
            assert!(Predicate::Interval { low, high }.matches(Value::Continuous(value)));
        }
    }

    #[test]
    fn from_specs_sorts_by_attribute() {
        let condition = Condition::from_specs(vec![
            (3, interval(0.0, 1.0)),
            (0, Predicate::Equals(1)),
            (1, interval(2.0, 3.0)),
        ]);
        let attributes: Vec<usize> = condition
            .specs()
            .iter()
            .map(|(attribute, _)| *attribute)
            .collect();
        assert_eq!(attributes, vec![0, 1, 3]);
        assert_eq!(condition.len(), 3);
        assert!(!condition.is_empty());
        assert!(Condition::new().is_empty());
    }

    #[test]
    #[should_panic(expected = "duplicate attribute")]
    fn from_specs_rejects_duplicates() {
        Condition::from_specs(vec![(2, Predicate::Equals(0)), (2, Predicate::Equals(1))]);
    }

    #[test]
    fn get_set_remove_keep_order() {
        let mut condition = Condition::new();
        condition.set(5, Predicate::Equals(1));
        condition.set(1, interval(0.0, 1.0));
        condition.set(3, Predicate::Equals(0));
        let attributes: Vec<usize> = condition
            .specs()
            .iter()
            .map(|(attribute, _)| *attribute)
            .collect();
        assert_eq!(attributes, vec![1, 3, 5]);
        assert_eq!(condition.get(3), Some(&Predicate::Equals(0)));
        assert_eq!(condition.get(2), None);
        condition.set(3, Predicate::Equals(2));
        assert_eq!(condition.get(3), Some(&Predicate::Equals(2)));
        assert_eq!(condition.len(), 3);
        assert_eq!(condition.remove(1), Some(interval(0.0, 1.0)));
        assert_eq!(condition.remove(1), None);
        assert_eq!(condition.get(1), None);
        assert_eq!(
            condition.specs(),
            &[(3, Predicate::Equals(2)), (5, Predicate::Equals(1))]
        );
    }

    #[test]
    fn condition_matches_row() {
        let condition =
            Condition::from_specs(vec![(0, Predicate::Equals(1)), (1, interval(3.0, 5.0))]);
        assert!(condition.matches(&row()));
        let mut with_missing = row();
        with_missing[1] = Value::Missing;
        assert!(condition.matches(&with_missing));
        let mut outside = row();
        outside[1] = Value::Continuous(5.5);
        assert!(!condition.matches(&outside));
        let mut other_value = row();
        other_value[0] = Value::Discrete(2);
        assert!(!condition.matches(&other_value));
        assert!(Condition::new().matches(&row()));
    }

    #[test]
    fn more_general_with_fewer_attributes() {
        let general = Condition::from_specs(vec![(0, Predicate::Equals(1))]);
        let specific =
            Condition::from_specs(vec![(0, Predicate::Equals(1)), (2, Predicate::Equals(0))]);
        assert!(general.is_more_general(&specific));
        assert!(!specific.is_more_general(&general));
        assert!(Condition::new().is_more_general(&general));
    }

    #[test]
    fn more_general_with_wider_interval() {
        let wide = Condition::from_specs(vec![(1, interval(0.0, 10.0))]);
        let narrow = Condition::from_specs(vec![(1, interval(2.0, 4.0))]);
        assert!(wide.is_more_general(&narrow));
        assert!(!narrow.is_more_general(&wide));
    }

    #[test]
    fn not_more_general_when_equal() {
        let condition =
            Condition::from_specs(vec![(0, Predicate::Equals(1)), (1, interval(0.0, 1.0))]);
        assert!(!condition.is_more_general(&condition.clone()));
        assert!(!Condition::new().is_more_general(&Condition::new()));
    }

    #[test]
    fn not_more_general_with_different_discrete_value() {
        let first = Condition::from_specs(vec![(0, Predicate::Equals(1))]);
        let second =
            Condition::from_specs(vec![(0, Predicate::Equals(2)), (1, interval(0.0, 1.0))]);
        assert!(!first.is_more_general(&second));
    }

    #[test]
    fn cover_matches_row_and_respects_rsl() {
        let schema = schema();
        let row = row();
        for seed in 0..200 {
            for rsl in 1..=5 {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let condition = Condition::cover(&row, &schema, rsl, &mut rng);
                assert!(condition.matches(&row));
                assert!(!condition.is_empty());
                assert!(condition.len() <= rsl.min(row.len()));
                assert!(condition
                    .specs()
                    .windows(2)
                    .all(|pair| pair[0].0 < pair[1].0));
            }
        }
    }

    #[test]
    fn cover_reaches_every_specificity() {
        let schema = schema();
        let row = row();
        let mut seen = [false; 5];
        for seed in 0..500 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            seen[Condition::cover(&row, &schema, 4, &mut rng).len()] = true;
        }
        assert_eq!(seen, [false, true, true, true, true]);
    }

    #[test]
    fn cover_skips_missing_attributes() {
        let schema = schema();
        let mut row = row();
        row[1] = Value::Missing;
        row[3] = Value::Missing;
        for seed in 0..200 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let condition = Condition::cover(&row, &schema, 4, &mut rng);
            assert!(condition.get(1).is_none());
            assert!(condition.get(3).is_none());
            assert!(!condition.is_empty());
            assert!(condition.len() <= 2);
            assert!(condition.matches(&row));
        }
    }

    #[test]
    fn cover_all_missing_is_empty_without_rng() {
        let schema = schema();
        let row = vec![Value::Missing; 4];
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut reference = rng.clone();
        assert!(Condition::cover(&row, &schema, 3, &mut rng).is_empty());
        assert!(untouched(&mut rng, &mut reference));
    }

    #[test]
    fn cover_is_deterministic_for_seed() {
        let schema = schema();
        let row = row();
        for seed in 0..50 {
            let mut first = ChaCha8Rng::seed_from_u64(seed);
            let mut second = ChaCha8Rng::seed_from_u64(seed);
            assert_eq!(
                Condition::cover(&row, &schema, 3, &mut first),
                Condition::cover(&row, &schema, 3, &mut second)
            );
            assert!(untouched(&mut first, &mut second));
        }
    }

    #[test]
    #[should_panic(expected = "rule specificity limit")]
    fn cover_rejects_zero_rsl() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        Condition::cover(&row(), &schema(), 0, &mut rng);
    }

    #[test]
    fn new_classifier_defaults() {
        let condition = Condition::from_specs(vec![(0, Predicate::Equals(1))]);
        let classifier = Classifier::new(9, condition.clone(), 1, Origin::Llm, 42, 0.3, 7.5);
        assert_eq!(classifier.id, 9);
        assert_eq!(classifier.condition, condition);
        assert_eq!(classifier.class, 1);
        assert_eq!(classifier.numerosity, 1);
        assert_eq!(classifier.experience, 0);
        assert_eq!(classifier.correct_count, 0);
        assert_eq!(classifier.accuracy, 0.0);
        assert_eq!(classifier.fitness, 0.3);
        assert_eq!(classifier.avg_match_set_size, 7.5);
        assert_eq!(classifier.ga_timestamp, 42);
        assert_eq!(classifier.birth_iteration, 42);
        assert_eq!(classifier.origin, Origin::Llm);
        assert_eq!(classifier.external_id, None);
        assert!(classifier.parents.is_empty());
        assert!(classifier.seed_lineage.is_empty());
    }

    #[test]
    fn same_rule_compares_condition_and_class() {
        let condition = Condition::from_specs(vec![(0, Predicate::Equals(1))]);
        let first = Classifier::new(1, condition.clone(), 0, Origin::Covering, 0, 0.1, 1.0);
        let mut second = Classifier::new(2, condition, 0, Origin::Ga, 5, 0.9, 3.0);
        assert!(first.same_rule(&second));
        second.class = 1;
        assert!(!first.same_rule(&second));
        second.class = 0;
        second.condition.set(1, interval(0.0, 1.0));
        assert!(!first.same_rule(&second));
    }

    #[test]
    fn subsumer_thresholds_are_strict() {
        let condition = Condition::new();
        assert!(rule(condition.clone(), 0, 21, 0.995).is_subsumer(20, 0.99));
        assert!(!rule(condition.clone(), 0, 20, 0.995).is_subsumer(20, 0.99));
        assert!(!rule(condition, 0, 21, 0.99).is_subsumer(20, 0.99));
    }

    #[test]
    fn subsumes_cases() {
        let general = Condition::from_specs(vec![(1, interval(0.0, 10.0))]);
        let specific =
            Condition::from_specs(vec![(0, Predicate::Equals(1)), (1, interval(2.0, 4.0))]);
        let subsumer = rule(general.clone(), 1, 50, 1.0);
        let target = rule(specific.clone(), 1, 0, 0.0);
        assert!(subsumer.subsumes(&target, 20, 0.99));
        assert!(!subsumer.subsumes(&rule(specific.clone(), 0, 0, 0.0), 20, 0.99));
        assert!(!rule(general.clone(), 1, 20, 1.0).subsumes(&target, 20, 0.99));
        assert!(!rule(general.clone(), 1, 50, 0.9).subsumes(&target, 20, 0.99));
        assert!(!subsumer.subsumes(&rule(general, 1, 0, 0.0), 20, 0.99));
        assert!(!rule(specific, 1, 50, 1.0).subsumes(&rule(Condition::new(), 1, 0, 0.0), 20, 0.99));
    }
}
