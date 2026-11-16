use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Selection {
    Tournament,
    Roulette,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub population_size: u64,
    pub nu: u32,
    pub chi: f64,
    pub mu: f64,
    pub theta_ga: f64,
    pub theta_del: u32,
    pub theta_sub: u32,
    pub acc_sub: f64,
    pub beta: f64,
    pub delta: f64,
    pub init_fitness: f64,
    pub fitness_reduction: f64,
    pub do_ga_subsumption: bool,
    pub do_correct_set_subsumption: bool,
    pub selection: Selection,
    pub theta_sel: f64,
    pub rule_specificity_limit: Option<usize>,
    pub fixed_conditions: bool,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: 1000,
            nu: 1,
            chi: 0.8,
            mu: 0.04,
            theta_ga: 25.0,
            theta_del: 20,
            theta_sub: 20,
            acc_sub: 0.99,
            beta: 0.2,
            delta: 0.1,
            init_fitness: 0.01,
            fitness_reduction: 0.1,
            do_ga_subsumption: true,
            do_correct_set_subsumption: false,
            selection: Selection::Tournament,
            theta_sel: 0.5,
            rule_specificity_limit: None,
            fixed_conditions: false,
            seed: 42,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let probability = |name: &str, value: f64| {
            if (0.0..=1.0).contains(&value) {
                Ok(())
            } else {
                Err(Error::Config(format!("{name} must be in [0, 1], got {value}")))
            }
        };
        if self.population_size == 0 {
            return Err(Error::Config("population_size must be positive".into()));
        }
        if self.nu == 0 {
            return Err(Error::Config("nu must be at least 1".into()));
        }
        probability("chi", self.chi)?;
        probability("mu", self.mu)?;
        probability("acc_sub", self.acc_sub)?;
        probability("delta", self.delta)?;
        probability("fitness_reduction", self.fitness_reduction)?;
        if !(self.beta > 0.0 && self.beta <= 1.0) {
            return Err(Error::Config(format!("beta must be in (0, 1], got {}", self.beta)));
        }
        if !(self.theta_sel > 0.0 && self.theta_sel <= 1.0) {
            return Err(Error::Config(format!(
                "theta_sel must be in (0, 1], got {}",
                self.theta_sel
            )));
        }
        if !(self.theta_ga >= 0.0 && self.theta_ga.is_finite()) {
            return Err(Error::Config(format!(
                "theta_ga must be finite and non-negative, got {}",
                self.theta_ga
            )));
        }
        if !(self.init_fitness > 0.0 && self.init_fitness.is_finite()) {
            return Err(Error::Config(format!(
                "init_fitness must be positive, got {}",
                self.init_fitness
            )));
        }
        if self.rule_specificity_limit == Some(0) {
            return Err(Error::Config("rule_specificity_limit must be positive".into()));
        }
        Ok(())
    }
}
