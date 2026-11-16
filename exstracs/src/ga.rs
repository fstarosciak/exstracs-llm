use rand::Rng;

use crate::classifier::{Classifier, Condition, Predicate};
use crate::config::Selection;
use crate::data::{Schema, Value};

pub fn select_parent<R: Rng + ?Sized>(
    classifiers: &[Classifier],
    set: &[usize],
    method: Selection,
    theta_sel: f64,
    rng: &mut R,
) -> usize {
    assert!(!set.is_empty());
    match method {
        Selection::Tournament => tournament(classifiers, set, theta_sel, rng),
        Selection::Roulette => roulette(classifiers, set, rng),
    }
}

fn tournament<R: Rng + ?Sized>(
    classifiers: &[Classifier],
    set: &[usize],
    theta_sel: f64,
    rng: &mut R,
) -> usize {
    let size = ((set.len() as f64 * theta_sel).floor() as usize).clamp(1, set.len());
    let positions = rand::seq::index::sample(rng, set.len(), size);
    let mut winner = set[positions.index(0)];
    for position in positions.iter().skip(1) {
        let candidate = set[position];
        if classifiers[candidate].fitness > classifiers[winner].fitness {
            winner = candidate;
        }
    }
    winner
}

fn roulette<R: Rng + ?Sized>(classifiers: &[Classifier], set: &[usize], rng: &mut R) -> usize {
    let total: f64 = set.iter().map(|&index| classifiers[index].fitness).sum();
    if total <= 0.0 {
        return set[rng.gen_range(0..set.len())];
    }
    let threshold = rng.gen::<f64>() * total;
    let mut cumulative = 0.0;
    for &index in set {
        cumulative += classifiers[index].fitness;
        if cumulative > threshold {
            return index;
        }
    }
    set[set.len() - 1]
}

pub fn crossover<R: Rng + ?Sized>(
    first: &mut Condition,
    second: &mut Condition,
    rng: &mut R,
) -> bool {
    let original_first = first.clone();
    let original_second = second.clone();
    let mut attributes: Vec<usize> = first
        .specs()
        .iter()
        .chain(second.specs())
        .map(|&(attribute, _)| attribute)
        .collect();
    attributes.sort_unstable();
    attributes.dedup();
    for attribute in attributes {
        match (
            first.get(attribute).copied(),
            second.get(attribute).copied(),
        ) {
            (Some(predicate), None) => {
                if rng.gen_bool(0.5) {
                    first.remove(attribute);
                    second.set(attribute, predicate);
                }
            }
            (None, Some(predicate)) => {
                if rng.gen_bool(0.5) {
                    second.remove(attribute);
                    first.set(attribute, predicate);
                }
            }
            (Some(Predicate::Equals(a)), Some(Predicate::Equals(b))) => {
                if rng.gen_bool(0.5) {
                    first.set(attribute, Predicate::Equals(b));
                    second.set(attribute, Predicate::Equals(a));
                }
            }
            (
                Some(Predicate::Interval {
                    low: mut first_low,
                    high: mut first_high,
                }),
                Some(Predicate::Interval {
                    low: mut second_low,
                    high: mut second_high,
                }),
            ) => {
                if rng.gen_bool(0.5) {
                    std::mem::swap(&mut first_low, &mut second_low);
                }
                if rng.gen_bool(0.5) {
                    std::mem::swap(&mut first_high, &mut second_high);
                }
                first.set(attribute, ordered_interval(first_low, first_high));
                second.set(attribute, ordered_interval(second_low, second_high));
            }
            _ => {}
        }
    }
    *first != original_first || *second != original_second
}

fn ordered_interval(low: f64, high: f64) -> Predicate {
    if low > high {
        Predicate::Interval {
            low: high,
            high: low,
        }
    } else {
        Predicate::Interval { low, high }
    }
}

pub fn enforce_specificity_limit<R: Rng + ?Sized>(
    condition: &mut Condition,
    rsl: usize,
    rng: &mut R,
) -> bool {
    let mut removed = false;
    while condition.len() > rsl {
        let position = rng.gen_range(0..condition.len());
        let attribute = condition.specs()[position].0;
        condition.remove(attribute);
        removed = true;
    }
    removed
}

pub fn mutate<R: Rng + ?Sized>(
    condition: &mut Condition,
    row: &[Value],
    schema: &Schema,
    mu: f64,
    rsl: usize,
    rng: &mut R,
) -> bool {
    let original = condition.clone();
    for (attribute, definition) in schema.attributes.iter().enumerate() {
        if !rng.gen_bool(mu) {
            continue;
        }
        match condition.get(attribute).copied() {
            Some(Predicate::Interval { low, high }) => {
                let coin = rng.gen_bool(0.5);
                let range = definition.range();
                if coin && range > 0.0 {
                    let delta = rng.gen::<f64>() * 0.5 * range;
                    let negative = rng.gen_bool(0.5);
                    let move_low = rng.gen_bool(0.5);
                    let shift = if negative { -delta } else { delta };
                    let (new_low, new_high) = if move_low {
                        (low + shift, high)
                    } else {
                        (low, high + shift)
                    };
                    let candidate = Predicate::Interval {
                        low: new_low,
                        high: new_high,
                    };
                    if new_low <= new_high && candidate.matches(row[attribute]) {
                        condition.set(attribute, candidate);
                    }
                } else {
                    condition.remove(attribute);
                }
            }
            Some(Predicate::Equals(_)) => {
                condition.remove(attribute);
            }
            None => {
                if condition.len() < rsl {
                    if let Some(predicate) = Predicate::cover(definition, row[attribute], rng) {
                        condition.set(attribute, predicate);
                    }
                }
            }
        }
    }
    *condition != original
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::classifier::Origin;
    use crate::data::{Attribute, AttributeKind};

    fn classifier(id: u64, fitness: f64) -> Classifier {
        Classifier {
            id,
            condition: Condition::new(),
            class: 0,
            numerosity: 1,
            experience: 0,
            correct_count: 0,
            accuracy: 0.0,
            fitness,
            avg_match_set_size: 1.0,
            ga_timestamp: 0,
            birth_iteration: 0,
            origin: Origin::Covering,
            external_id: None,
            parents: Vec::new(),
            seed_lineage: Vec::new(),
        }
    }

    fn population(fitnesses: &[f64]) -> Vec<Classifier> {
        fitnesses
            .iter()
            .enumerate()
            .map(|(id, &fitness)| classifier(id as u64, fitness))
            .collect()
    }

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
                discrete("a0", &["x", "y", "z"]),
                continuous("a1", 0.0, 10.0),
                discrete("a2", &["p", "q"]),
                continuous("a3", -5.0, 5.0),
                continuous("a4", 0.0, 1.0),
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
            Value::Missing,
        ]
    }

    fn first_parent() -> Condition {
        Condition::from_specs(vec![
            (0, Predicate::Equals(1)),
            (
                1,
                Predicate::Interval {
                    low: 2.0,
                    high: 6.0,
                },
            ),
            (
                3,
                Predicate::Interval {
                    low: -3.0,
                    high: 0.0,
                },
            ),
        ])
    }

    fn second_parent() -> Condition {
        Condition::from_specs(vec![
            (
                1,
                Predicate::Interval {
                    low: 3.5,
                    high: 9.0,
                },
            ),
            (2, Predicate::Equals(0)),
            (
                3,
                Predicate::Interval {
                    low: -2.0,
                    high: 1.0,
                },
            ),
            (
                4,
                Predicate::Interval {
                    low: 0.1,
                    high: 0.2,
                },
            ),
        ])
    }

    fn intervals_ordered(condition: &Condition) -> bool {
        condition
            .specs()
            .iter()
            .all(|(_, predicate)| match predicate {
                Predicate::Interval { low, high } => low <= high,
                Predicate::Equals(_) => true,
            })
    }

    #[test]
    fn tournament_with_full_theta_picks_max_fitness() {
        let classifiers = population(&[0.1, 0.9, 0.5, 0.3, 0.9]);
        for seed in 0..50 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let all = [0, 1, 2, 3];
            assert_eq!(
                select_parent(&classifiers, &all, Selection::Tournament, 1.0, &mut rng),
                1
            );
            let subset = [0, 2, 3];
            assert_eq!(
                select_parent(&classifiers, &subset, Selection::Tournament, 1.0, &mut rng),
                2
            );
        }
    }

    #[test]
    fn tournament_returns_member_of_set() {
        let classifiers = population(&[0.1, 0.9, 0.5, 0.3, 0.7, 0.2]);
        let set = [5, 0, 3];
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for theta in [0.0, 0.2, 0.5, 0.99] {
            for _ in 0..50 {
                let chosen =
                    select_parent(&classifiers, &set, Selection::Tournament, theta, &mut rng);
                assert!(set.contains(&chosen));
            }
        }
    }

    #[test]
    fn roulette_never_picks_zero_fitness_element() {
        let classifiers = population(&[0.0, 0.5, 0.0, 0.25, 0.0]);
        let set = [0, 1, 2, 3, 4];
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut seen = [0usize; 5];
        for _ in 0..2000 {
            seen[select_parent(&classifiers, &set, Selection::Roulette, 0.5, &mut rng)] += 1;
        }
        assert_eq!(seen[0], 0);
        assert_eq!(seen[2], 0);
        assert_eq!(seen[4], 0);
        assert!(seen[1] > seen[3]);
        assert!(seen[3] > 0);
    }

    #[test]
    fn roulette_with_zero_total_picks_uniformly_from_set() {
        let classifiers = population(&[0.0, 0.0, 0.0, 0.0]);
        let set = [1, 3];
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut seen = [0usize; 4];
        for _ in 0..200 {
            seen[select_parent(&classifiers, &set, Selection::Roulette, 0.5, &mut rng)] += 1;
        }
        assert_eq!(seen[0] + seen[2], 0);
        assert!(seen[1] > 0 && seen[3] > 0);
    }

    #[test]
    #[should_panic]
    fn select_parent_rejects_empty_set() {
        let classifiers = population(&[1.0]);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        select_parent(&classifiers, &[], Selection::Tournament, 0.5, &mut rng);
    }

    #[test]
    fn crossover_preserves_matching_row() {
        let row = row();
        let total = first_parent().len() + second_parent().len();
        assert!(first_parent().matches(&row));
        assert!(second_parent().matches(&row));
        let mut any_change = false;
        for seed in 0..200 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut first = first_parent();
            let mut second = second_parent();
            let changed = crossover(&mut first, &mut second, &mut rng);
            assert_eq!(
                changed,
                first != first_parent() || second != second_parent()
            );
            any_change |= changed;
            assert!(first.matches(&row));
            assert!(second.matches(&row));
            assert_eq!(first.len() + second.len(), total);
            assert!(intervals_ordered(&first));
            assert!(intervals_ordered(&second));
        }
        assert!(any_change);
    }

    #[test]
    fn crossover_keeps_intervals_ordered() {
        for seed in 0..100 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut first = Condition::from_specs(vec![(
                1,
                Predicate::Interval {
                    low: 0.0,
                    high: 1.0,
                },
            )]);
            let mut second = Condition::from_specs(vec![(
                1,
                Predicate::Interval {
                    low: 2.0,
                    high: 3.0,
                },
            )]);
            crossover(&mut first, &mut second, &mut rng);
            assert!(intervals_ordered(&first));
            assert!(intervals_ordered(&second));
        }
    }

    #[test]
    fn crossover_with_identical_parents_returns_false() {
        for seed in 0..100 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut first = first_parent();
            let mut second = first_parent();
            assert!(!crossover(&mut first, &mut second, &mut rng));
            assert_eq!(first, first_parent());
            assert_eq!(second, first_parent());
        }
    }

    #[test]
    fn crossover_leaves_kind_mismatch_untouched_without_rng() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut first = Condition::from_specs(vec![(0, Predicate::Equals(1))]);
        let mut second = Condition::from_specs(vec![(
            0,
            Predicate::Interval {
                low: 0.0,
                high: 1.0,
            },
        )]);
        let (first_before, second_before) = (first.clone(), second.clone());
        assert!(!crossover(&mut first, &mut second, &mut rng));
        assert_eq!(first, first_before);
        assert_eq!(second, second_before);
        assert_eq!(rng.gen::<u64>(), ChaCha8Rng::seed_from_u64(9).gen::<u64>());
    }

    #[test]
    fn enforce_specificity_limit_trims_to_rsl() {
        let original = Condition::from_specs(vec![
            (0, Predicate::Equals(1)),
            (
                1,
                Predicate::Interval {
                    low: 2.0,
                    high: 6.0,
                },
            ),
            (2, Predicate::Equals(0)),
            (
                3,
                Predicate::Interval {
                    low: -3.0,
                    high: 0.0,
                },
            ),
            (
                4,
                Predicate::Interval {
                    low: 0.1,
                    high: 0.2,
                },
            ),
        ]);
        for seed in 0..50 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut condition = original.clone();
            assert!(enforce_specificity_limit(&mut condition, 2, &mut rng));
            assert_eq!(condition.len(), 2);
            for (attribute, predicate) in condition.specs() {
                assert_eq!(original.get(*attribute), Some(predicate));
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut condition = original.clone();
        assert!(!enforce_specificity_limit(&mut condition, 5, &mut rng));
        assert!(!enforce_specificity_limit(&mut condition, 9, &mut rng));
        assert_eq!(condition, original);
        assert!(enforce_specificity_limit(&mut condition, 0, &mut rng));
        assert!(condition.is_empty());
    }

    #[test]
    fn mutate_with_zero_rate_is_noop() {
        let schema = schema();
        let row = row();
        for seed in 0..50 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut condition = first_parent();
            assert!(!mutate(&mut condition, &row, &schema, 0.0, 5, &mut rng));
            assert_eq!(condition, first_parent());
            let mut empty = Condition::new();
            assert!(!mutate(&mut empty, &row, &schema, 0.0, 5, &mut rng));
            assert!(empty.is_empty());
        }
    }

    #[test]
    fn mutate_with_full_rate_keeps_matching_and_respects_rsl() {
        let schema = schema();
        let row = row();
        let starts = [
            (Condition::new(), 0),
            (Condition::new(), 1),
            (Condition::new(), 2),
            (first_parent(), 3),
            (first_parent(), 5),
            (second_parent(), 4),
        ];
        for seed in 0..100 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for (start, rsl) in &starts {
                let mut condition = start.clone();
                for _ in 0..10 {
                    let before = condition.clone();
                    let changed = mutate(&mut condition, &row, &schema, 1.0, *rsl, &mut rng);
                    assert_eq!(changed, condition != before);
                    assert!(condition.matches(&row));
                    assert!(condition.len() <= *rsl);
                    assert!(intervals_ordered(&condition));
                }
            }
        }
    }

    #[test]
    fn mutate_with_full_rate_generalises_equals_and_covers_free_slots() {
        let schema = schema();
        let row = row();
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let mut condition =
            Condition::from_specs(vec![(0, Predicate::Equals(1)), (2, Predicate::Equals(0))]);
        assert!(mutate(&mut condition, &row, &schema, 1.0, 2, &mut rng));
        assert_eq!(condition.get(0), None);
        assert_eq!(condition.get(2), None);
        assert!(condition.get(1).is_some());
        assert!(condition.get(3).is_some());
        assert_eq!(condition.get(4), None);
        assert!(condition.matches(&row));
    }

    #[test]
    fn same_seed_gives_same_results() {
        fn run(seed: u64) -> (Vec<usize>, Condition, Condition, Condition, bool) {
            let schema = schema();
            let row = row();
            let classifiers = population(&[0.2, 0.8, 0.4, 0.6, 0.1]);
            let set = [0, 1, 2, 3, 4];
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut picks = Vec::new();
            for _ in 0..20 {
                picks.push(select_parent(
                    &classifiers,
                    &set,
                    Selection::Tournament,
                    0.4,
                    &mut rng,
                ));
                picks.push(select_parent(
                    &classifiers,
                    &set,
                    Selection::Roulette,
                    0.4,
                    &mut rng,
                ));
            }
            let mut first = first_parent();
            let mut second = second_parent();
            crossover(&mut first, &mut second, &mut rng);
            let mut child = first.clone();
            for _ in 0..5 {
                mutate(&mut child, &row, &schema, 0.5, 4, &mut rng);
            }
            let trimmed = enforce_specificity_limit(&mut second, 1, &mut rng);
            (picks, first, second, child, trimmed)
        }
        assert_eq!(run(42), run(42));
        assert_eq!(run(7), run(7));
    }
}
