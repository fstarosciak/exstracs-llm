use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::classifier::{Classifier, Condition, Origin};
use crate::config::Config;
use crate::data::{AttributeKind, Dataset, Schema, Value};
use crate::error::{Error, Result};
use crate::ga;
use crate::metrics::Metrics;
use crate::population::Population;
use crate::rules::SeedRule;
use crate::snapshot::{ClassifierRecord, Snapshot, SNAPSHOT_FORMAT_VERSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prediction {
    pub class: u32,
    pub covered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TrainStats {
    pub iterations: u64,
    pub correct: u64,
    pub covering: u64,
    pub ga_runs: u64,
}

impl TrainStats {
    pub fn add(&mut self, other: &TrainStats) {
        self.iterations += other.iterations;
        self.correct += other.correct;
        self.covering += other.covering;
        self.ga_runs += other.ga_runs;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PopulationStats {
    pub macro_size: usize,
    pub micro_size: u64,
    pub mean_specificity: f64,
    pub seeded_alive: usize,
    pub seed_descendants: usize,
}

pub fn default_rule_specificity_limit(data: &Dataset) -> usize {
    let attributes = &data.schema.attributes;
    if attributes.is_empty() {
        return 1;
    }
    let total_states: f64 = attributes
        .iter()
        .map(|attribute| match &attribute.kind {
            AttributeKind::Discrete { values } => values.len().max(1) as f64,
            AttributeKind::Continuous { .. } => 2.0,
        })
        .sum();
    let average_states = total_states / attributes.len() as f64;
    let instances = data.len() as f64;
    let mut limit = 1;
    let mut combinations = average_states;
    while combinations < instances && limit < attributes.len() {
        limit += 1;
        combinations *= average_states;
    }
    limit
}

fn integer_power(base: f64, exponent: u32) -> f64 {
    let mut result = 1.0;
    let mut factor = base;
    let mut remaining = exponent;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result *= factor;
        }
        factor *= factor;
        remaining >>= 1;
    }
    result
}

#[derive(Debug, Clone)]
pub struct ExSTraCS {
    config: Config,
    schema: Schema,
    rsl: usize,
    class_counts: Vec<usize>,
    population: Population,
    rng: ChaCha8Rng,
    iteration: u64,
    next_id: u64,
    order: Vec<usize>,
    cursor: usize,
}

impl ExSTraCS {
    pub fn new(config: Config, train: &Dataset) -> Result<Self> {
        config.validate()?;
        if train.is_empty() {
            return Err(Error::Data("training data is empty".into()));
        }
        let rsl = config
            .rule_specificity_limit
            .unwrap_or_else(|| default_rule_specificity_limit(train))
            .min(train.schema.n_attributes())
            .max(1);
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        Ok(Self {
            schema: train.schema.clone(),
            class_counts: train.class_counts(),
            config,
            rsl,
            population: Population::default(),
            rng,
            iteration: 0,
            next_id: 0,
            order: Vec::new(),
            cursor: 0,
        })
    }

    pub fn from_snapshot(snapshot: Snapshot) -> Result<Self> {
        if snapshot.format_version != SNAPSHOT_FORMAT_VERSION {
            return Err(Error::Data(format!(
                "unsupported snapshot format version {}",
                snapshot.format_version
            )));
        }
        snapshot.config.validate()?;
        if snapshot.class_counts.len() != snapshot.schema.n_classes() {
            return Err(Error::Data(
                "snapshot class_counts do not match its classes".into(),
            ));
        }
        if snapshot.rule_specificity_limit == 0 {
            return Err(Error::Data(
                "snapshot rule_specificity_limit is zero".into(),
            ));
        }
        let mut population = Population::default();
        for record in &snapshot.classifiers {
            let classifier = record.to_classifier(&snapshot.schema)?;
            if classifier.numerosity == 0 {
                return Err(Error::Data(format!(
                    "classifier {} has zero numerosity",
                    classifier.id
                )));
            }
            if classifier.id >= snapshot.next_id {
                return Err(Error::Data(format!(
                    "classifier {} is not below next_id {}",
                    classifier.id, snapshot.next_id
                )));
            }
            if population.find_same_rule(&classifier).is_some() {
                return Err(Error::Data(format!(
                    "classifier {} repeats the condition and class of another classifier",
                    classifier.id
                )));
            }
            population.push(classifier);
        }
        let rng = ChaCha8Rng::seed_from_u64(snapshot.config.seed.wrapping_add(snapshot.iteration));
        Ok(Self {
            config: snapshot.config,
            schema: snapshot.schema,
            rsl: snapshot.rule_specificity_limit,
            class_counts: snapshot.class_counts,
            population,
            rng,
            iteration: snapshot.iteration,
            next_id: snapshot.next_id,
            order: Vec::new(),
            cursor: 0,
        })
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            format_version: SNAPSHOT_FORMAT_VERSION,
            config: self.config.clone(),
            schema: self.schema.clone(),
            rule_specificity_limit: self.rsl,
            class_counts: self.class_counts.clone(),
            iteration: self.iteration,
            next_id: self.next_id,
            classifiers: self
                .population
                .classifiers()
                .iter()
                .map(|classifier| ClassifierRecord::from_classifier(classifier, &self.schema))
                .collect(),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn rule_specificity_limit(&self) -> usize {
        self.rsl
    }

    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    pub fn population(&self) -> &Population {
        &self.population
    }

    pub fn population_stats(&self) -> PopulationStats {
        let classifiers = self.population.classifiers();
        let micro_size = self.population.micro_size();
        let specified: u64 = classifiers
            .iter()
            .map(|classifier| classifier.condition.len() as u64 * u64::from(classifier.numerosity))
            .sum();
        PopulationStats {
            macro_size: classifiers.len(),
            micro_size,
            mean_specificity: if micro_size == 0 {
                0.0
            } else {
                specified as f64 / micro_size as f64
            },
            seeded_alive: classifiers
                .iter()
                .filter(|classifier| classifier.origin.is_seeded())
                .count(),
            seed_descendants: classifiers
                .iter()
                .filter(|classifier| {
                    classifier.origin == Origin::Ga && !classifier.seed_lineage.is_empty()
                })
                .count(),
        }
    }

    pub fn seed(&mut self, rules: &[SeedRule], numerosity: u32) -> usize {
        assert!(numerosity > 0, "seed numerosity must be positive");
        let mut inserted = 0;
        for rule in rules {
            let mut classifier = Classifier::new(
                self.take_id(),
                rule.condition.clone(),
                rule.class,
                rule.origin,
                self.iteration,
                self.config.init_fitness,
                1.0,
            );
            classifier.numerosity = numerosity;
            classifier.external_id = Some(rule.external_id.clone());
            classifier.seed_lineage = vec![rule.external_id.clone()];
            let before = self.population.len();
            self.population.insert(classifier);
            if self.population.len() > before {
                inserted += 1;
            }
        }
        inserted
    }

    pub fn train(&mut self, data: &Dataset, iterations: u64) -> Result<TrainStats> {
        if data.schema != self.schema {
            return Err(Error::Data(
                "training data schema differs from the model schema".into(),
            ));
        }
        if data.is_empty() {
            return Err(Error::Data("training data is empty".into()));
        }
        if self.order.len() != data.len() {
            self.order = (0..data.len()).collect();
            self.order.shuffle(&mut self.rng);
            self.cursor = 0;
        }
        let mut stats = TrainStats::default();
        for _ in 0..iterations {
            let index = self.order[self.cursor];
            self.cursor = (self.cursor + 1) % self.order.len();
            self.learn(&data.rows[index], data.labels[index], &mut stats);
        }
        Ok(stats)
    }

    pub fn predict(&self, row: &[Value]) -> Prediction {
        self.vote(&self.population.match_set(row))
    }

    pub fn predict_all(&self, data: &Dataset) -> Vec<Prediction> {
        data.rows.iter().map(|row| self.predict(row)).collect()
    }

    pub fn evaluate(&self, data: &Dataset) -> Metrics {
        let predictions = self.predict_all(data);
        let predicted: Vec<u32> = predictions
            .iter()
            .map(|prediction| prediction.class)
            .collect();
        let covered: Vec<bool> = predictions
            .iter()
            .map(|prediction| prediction.covered)
            .collect();
        Metrics::compute(&data.labels, &predicted, &covered, self.schema.n_classes())
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn learn(&mut self, row: &[Value], label: u32, stats: &mut TrainStats) {
        let now = self.iteration;
        let mut match_set = self.population.match_set(row);
        if self.vote(&match_set).class == label {
            stats.correct += 1;
        }
        let mut correct_set: Vec<usize> = match_set
            .iter()
            .copied()
            .filter(|&index| self.population.classifiers()[index].class == label)
            .collect();
        if correct_set.is_empty() && !self.config.fixed_conditions {
            let index = self.cover(row, label, &match_set, now);
            match_set.push(index);
            correct_set.push(index);
            stats.covering += 1;
        }
        self.update_parameters(&match_set, label);
        if !self.config.fixed_conditions {
            if self.config.do_correct_set_subsumption {
                self.correct_set_subsumption(&mut correct_set);
            }
            if !correct_set.is_empty() && self.ga_due(&correct_set, now) {
                self.run_ga(&correct_set, row, label, now);
                stats.ga_runs += 1;
            }
            self.delete_excess();
            self.population.remove_empty();
        }
        self.iteration += 1;
        stats.iterations += 1;
    }

    fn cover(&mut self, row: &[Value], label: u32, match_set: &[usize], now: u64) -> usize {
        let match_numerosity = self.numerosity_sum(match_set);
        let condition = Condition::cover(row, &self.schema, self.rsl, &mut self.rng);
        let classifier = Classifier::new(
            self.take_id(),
            condition,
            label,
            Origin::Covering,
            now,
            self.config.init_fitness,
            match_numerosity + 1.0,
        );
        self.population.insert(classifier)
    }

    fn numerosity_sum(&self, set: &[usize]) -> f64 {
        let classifiers = self.population.classifiers();
        set.iter()
            .map(|&index| f64::from(classifiers[index].numerosity))
            .sum()
    }

    fn update_parameters(&mut self, match_set: &[usize], label: u32) {
        let match_numerosity = self.numerosity_sum(match_set);
        let nu = self.config.nu;
        let beta = self.config.beta;
        for &index in match_set {
            let classifier = self.population.get_mut(index);
            classifier.experience += 1;
            if classifier.class == label {
                classifier.correct_count += 1;
            }
            let experience = f64::from(classifier.experience);
            classifier.accuracy = f64::from(classifier.correct_count) / experience;
            classifier.fitness = integer_power(classifier.accuracy, nu);
            if experience < 1.0 / beta {
                classifier.avg_match_set_size +=
                    (match_numerosity - classifier.avg_match_set_size) / experience;
            } else {
                classifier.avg_match_set_size +=
                    beta * (match_numerosity - classifier.avg_match_set_size);
            }
        }
    }

    fn correct_set_subsumption(&mut self, correct_set: &mut Vec<usize>) {
        let theta_sub = self.config.theta_sub;
        let acc_sub = self.config.acc_sub;
        let classifiers = self.population.classifiers();
        let mut subsumer: Option<usize> = None;
        for &index in correct_set.iter() {
            let candidate = &classifiers[index];
            if !candidate.is_subsumer(theta_sub, acc_sub) {
                continue;
            }
            let better = match subsumer {
                None => true,
                Some(current) => {
                    let current = &classifiers[current];
                    candidate.condition.len() < current.condition.len()
                        || candidate.condition.is_more_general(&current.condition)
                }
            };
            if better {
                subsumer = Some(index);
            }
        }
        let Some(subsumer) = subsumer else {
            return;
        };
        let absorbed: Vec<usize> = correct_set
            .iter()
            .copied()
            .filter(|&index| {
                index != subsumer
                    && classifiers[subsumer]
                        .condition
                        .is_more_general(&classifiers[index].condition)
            })
            .collect();
        for index in absorbed {
            let numerosity = self.population.classifiers()[index].numerosity;
            self.population.get_mut(subsumer).numerosity += numerosity;
            self.population.get_mut(index).numerosity = 0;
        }
        let classifiers = self.population.classifiers();
        correct_set.retain(|&index| classifiers[index].numerosity > 0);
    }

    fn ga_due(&self, correct_set: &[usize], now: u64) -> bool {
        let classifiers = self.population.classifiers();
        let mut numerosity = 0.0;
        let mut weighted = 0.0;
        for &index in correct_set {
            let classifier = &classifiers[index];
            numerosity += f64::from(classifier.numerosity);
            weighted += classifier.ga_timestamp as f64 * f64::from(classifier.numerosity);
        }
        now as f64 - weighted / numerosity > self.config.theta_ga
    }

    fn run_ga(&mut self, correct_set: &[usize], row: &[Value], label: u32, now: u64) {
        for &index in correct_set {
            self.population.get_mut(index).ga_timestamp = now;
        }
        let first = ga::select_parent(
            self.population.classifiers(),
            correct_set,
            self.config.selection,
            self.config.theta_sel,
            &mut self.rng,
        );
        let second = ga::select_parent(
            self.population.classifiers(),
            correct_set,
            self.config.selection,
            self.config.theta_sel,
            &mut self.rng,
        );
        let parent_one = self.population.classifiers()[first].clone();
        let parent_two = self.population.classifiers()[second].clone();
        let mut condition_one = parent_one.condition.clone();
        let mut condition_two = parent_two.condition.clone();
        let crossed = first != second
            && self.rng.gen_bool(self.config.chi)
            && ga::crossover(&mut condition_one, &mut condition_two, &mut self.rng);
        for condition in [&mut condition_one, &mut condition_two] {
            ga::enforce_specificity_limit(condition, self.rsl, &mut self.rng);
            ga::mutate(
                condition,
                row,
                &self.schema,
                self.config.mu,
                self.rsl,
                &mut self.rng,
            );
        }
        let offspring = [
            (condition_one, &parent_one, &parent_two),
            (condition_two, &parent_two, &parent_one),
        ];
        for (condition, own, other) in offspring {
            let child = self.make_offspring(condition, label, own, other, crossed, now);
            self.insert_offspring(child, first, second);
        }
    }

    fn make_offspring(
        &mut self,
        condition: Condition,
        label: u32,
        own: &Classifier,
        other: &Classifier,
        crossed: bool,
        now: u64,
    ) -> Classifier {
        let (fitness, accuracy, match_set_size) = if crossed {
            (
                (own.fitness + other.fitness) / 2.0,
                (own.accuracy + other.accuracy) / 2.0,
                (own.avg_match_set_size + other.avg_match_set_size) / 2.0,
            )
        } else {
            (own.fitness, own.accuracy, own.avg_match_set_size)
        };
        let mut child = Classifier::new(
            self.take_id(),
            condition,
            label,
            Origin::Ga,
            now,
            self.config.fitness_reduction * fitness,
            match_set_size,
        );
        child.accuracy = accuracy;
        child.parents = vec![own.id];
        child.seed_lineage = own.seed_lineage.clone();
        if crossed {
            child.parents.push(other.id);
            child
                .seed_lineage
                .extend(other.seed_lineage.iter().cloned());
            child.seed_lineage.sort();
            child.seed_lineage.dedup();
        }
        child
    }

    fn insert_offspring(&mut self, child: Classifier, first: usize, second: usize) {
        if self.config.do_ga_subsumption {
            for parent in [first, second] {
                let candidate = &self.population.classifiers()[parent];
                if candidate.numerosity > 0
                    && candidate.subsumes(&child, self.config.theta_sub, self.config.acc_sub)
                {
                    self.population.get_mut(parent).numerosity += 1;
                    return;
                }
            }
        }
        self.population.insert(child);
    }

    fn delete_excess(&mut self) {
        let mut micro_size = self.population.micro_size();
        while micro_size > self.config.population_size {
            self.delete_one(micro_size);
            micro_size -= 1;
        }
    }

    fn delete_one(&mut self, micro_size: u64) {
        let classifiers = self.population.classifiers();
        let mean_fitness = classifiers
            .iter()
            .map(|classifier| classifier.fitness * f64::from(classifier.numerosity))
            .sum::<f64>()
            / micro_size as f64;
        let votes: Vec<f64> = classifiers
            .iter()
            .map(|classifier| self.deletion_vote(classifier, mean_fitness))
            .collect();
        let total: f64 = votes.iter().sum();
        let chosen = if total > 0.0 {
            let threshold = self.rng.gen::<f64>() * total;
            let mut cumulative = 0.0;
            let mut chosen = None;
            for (index, vote) in votes.iter().enumerate() {
                if *vote <= 0.0 {
                    continue;
                }
                cumulative += vote;
                chosen = Some(index);
                if cumulative > threshold {
                    break;
                }
            }
            chosen
        } else {
            let living: Vec<usize> = (0..classifiers.len())
                .filter(|&index| classifiers[index].numerosity > 0)
                .collect();
            living.get(self.rng.gen_range(0..living.len())).copied()
        };
        if let Some(index) = chosen {
            self.population.get_mut(index).numerosity -= 1;
        }
    }

    fn deletion_vote(&self, classifier: &Classifier, mean_fitness: f64) -> f64 {
        if classifier.numerosity == 0 {
            return 0.0;
        }
        let vote = classifier.avg_match_set_size * f64::from(classifier.numerosity);
        if classifier.experience > self.config.theta_del
            && classifier.fitness < self.config.delta * mean_fitness
        {
            vote * mean_fitness / classifier.fitness.max(self.config.init_fitness)
        } else {
            vote
        }
    }

    fn vote(&self, match_set: &[usize]) -> Prediction {
        let n_classes = self.schema.n_classes();
        let classifiers = self.population.classifiers();
        let mut votes = vec![0.0; n_classes];
        let mut present = vec![false; n_classes];
        for &index in match_set {
            let classifier = &classifiers[index];
            if classifier.numerosity == 0 {
                continue;
            }
            let class = classifier.class as usize;
            votes[class] += classifier.fitness * f64::from(classifier.numerosity);
            present[class] = true;
        }
        let mut best: Option<usize> = None;
        for class in 0..n_classes {
            if !present[class] {
                continue;
            }
            best = match best {
                None => Some(class),
                Some(current) => {
                    let better = votes[class] > votes[current]
                        || (votes[class] == votes[current]
                            && self.class_counts[class] > self.class_counts[current]);
                    Some(if better { class } else { current })
                }
            };
        }
        match best {
            Some(class) => Prediction {
                class: class as u32,
                covered: true,
            },
            None => Prediction {
                class: self.majority_class(),
                covered: false,
            },
        }
    }

    fn majority_class(&self) -> u32 {
        let mut best = 0;
        for (class, &count) in self.class_counts.iter().enumerate() {
            if count > self.class_counts[best] {
                best = class;
            }
        }
        best as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{LoadOptions, Table};

    fn dataset() -> Dataset {
        let table = Table {
            header: vec!["a".into(), "b".into(), "class".into()],
            rows: vec![
                vec!["0".into(), "0".into(), "0".into()],
                vec!["0".into(), "1".into(), "1".into()],
                vec!["1".into(), "0".into(), "1".into()],
                vec!["1".into(), "1".into(), "0".into()],
            ],
        };
        Dataset::from_table(&table, &LoadOptions::default()).unwrap()
    }

    fn trained() -> ExSTraCS {
        let data = dataset();
        let mut model = ExSTraCS::new(Config::default(), &data).unwrap();
        model.train(&data, 200).unwrap();
        model
    }

    #[test]
    fn integer_power_matches_repeated_multiplication() {
        for exponent in 0..12 {
            let expected = (0..exponent).fold(1.0, |product, _| product * 0.9);
            assert!((integer_power(0.9, exponent) - expected).abs() < 1e-12);
        }
        assert_eq!(integer_power(1.0, u32::MAX), 1.0);
    }

    #[test]
    fn snapshot_round_trip_keeps_population() {
        let model = trained();
        let restored = ExSTraCS::from_snapshot(model.snapshot()).unwrap();
        assert_eq!(restored.snapshot(), model.snapshot());
    }

    #[test]
    fn snapshot_with_duplicate_rule_is_rejected() {
        let mut snapshot = trained().snapshot();
        let mut copy = snapshot.classifiers[0].clone();
        copy.id = snapshot.next_id;
        snapshot.next_id += 1;
        snapshot.classifiers.push(copy);
        assert!(matches!(
            ExSTraCS::from_snapshot(snapshot),
            Err(Error::Data(_))
        ));
    }

    #[test]
    fn snapshot_with_stale_next_id_is_rejected() {
        let mut snapshot = trained().snapshot();
        snapshot.next_id = 0;
        assert!(matches!(
            ExSTraCS::from_snapshot(snapshot),
            Err(Error::Data(_))
        ));
    }
}
