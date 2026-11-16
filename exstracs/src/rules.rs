use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::classifier::{Condition, Origin, Predicate};
use crate::data::{AttributeKind, Schema};
use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    pub source: Origin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_hash: Option<String>,
    pub rules: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub conditions: Vec<RuleCondition>,
    pub class: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RuleCondition {
    Discrete {
        attribute: String,
        value: String,
    },
    Continuous {
        attribute: String,
        low: Option<f64>,
        high: Option<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum RejectReason {
    Malformed { message: String },
    UnknownAttribute { attribute: String },
    TypeMismatch { attribute: String },
    UnknownValue { attribute: String, value: String },
    NonFiniteBound { attribute: String },
    InvertedInterval { attribute: String },
    OutsideDataRange { attribute: String },
    RepeatedAttribute { attribute: String },
    UnknownClass { class: String },
    TooSpecific { specified: usize, limit: usize },
    Duplicate { of: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rejection {
    pub index: usize,
    pub id: String,
    #[serde(flatten)]
    pub reason: RejectReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub source: Origin,
    pub total: usize,
    pub accepted: usize,
    pub rejected: Vec<Rejection>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeedRule {
    pub external_id: String,
    pub condition: Condition,
    pub class: u32,
    pub origin: Origin,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Validated {
    pub rules: Vec<SeedRule>,
    pub report: ValidationReport,
}

impl RuleFile {
    pub fn from_json_str(text: &str) -> Result<RuleFile> {
        let file: RuleFile = serde_json::from_str(text)?;
        if !file.source.is_seeded() {
            return Err(Error::Data(format!(
                "rule file source must be llm, random, tree or expert, got {:?}",
                file.source
            )));
        }
        Ok(file)
    }

    pub fn load(path: &Path) -> Result<RuleFile> {
        let text = std::fs::read_to_string(path)?;
        RuleFile::from_json_str(&text)
    }
}

pub fn validate(file: &RuleFile, schema: &Schema, rsl: usize) -> Validated {
    let mut rules: Vec<SeedRule> = Vec::new();
    let mut rejected = Vec::new();
    for (index, value) in file.rules.iter().enumerate() {
        let spec: RuleSpec = match serde_json::from_value(value.clone()) {
            Ok(spec) => spec,
            Err(error) => {
                let id = value
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .map_or_else(|| format!("rule-{index}"), str::to_string);
                rejected.push(Rejection {
                    index,
                    id,
                    reason: RejectReason::Malformed {
                        message: error.to_string(),
                    },
                });
                continue;
            }
        };
        let id = spec.id.clone().unwrap_or_else(|| format!("rule-{index}"));
        match check_spec(&spec, schema, rsl, &rules) {
            Ok((condition, class)) => rules.push(SeedRule {
                external_id: id,
                condition,
                class,
                origin: file.source,
            }),
            Err(reason) => rejected.push(Rejection { index, id, reason }),
        }
    }
    let report = ValidationReport {
        source: file.source,
        total: file.rules.len(),
        accepted: rules.len(),
        rejected,
    };
    Validated { rules, report }
}

fn check_spec(
    spec: &RuleSpec,
    schema: &Schema,
    rsl: usize,
    accepted: &[SeedRule],
) -> std::result::Result<(Condition, u32), RejectReason> {
    let condition = to_condition(&spec.conditions, schema)?;
    let class = schema
        .class_index(&spec.class)
        .ok_or_else(|| RejectReason::UnknownClass {
            class: spec.class.clone(),
        })?;
    if condition.len() > rsl {
        return Err(RejectReason::TooSpecific {
            specified: condition.len(),
            limit: rsl,
        });
    }
    if let Some(earlier) = accepted
        .iter()
        .find(|rule| rule.class == class && rule.condition == condition)
    {
        return Err(RejectReason::Duplicate {
            of: earlier.external_id.clone(),
        });
    }
    Ok((condition, class))
}

pub fn to_condition(
    conditions: &[RuleCondition],
    schema: &Schema,
) -> std::result::Result<Condition, RejectReason> {
    let mut specs: Vec<(usize, Predicate)> = Vec::with_capacity(conditions.len());
    for condition in conditions {
        let name = match condition {
            RuleCondition::Discrete { attribute, .. }
            | RuleCondition::Continuous { attribute, .. } => attribute,
        };
        let index = schema
            .attribute_index(name)
            .ok_or_else(|| RejectReason::UnknownAttribute {
                attribute: name.clone(),
            })?;
        if specs.iter().any(|(used, _)| *used == index) {
            return Err(RejectReason::RepeatedAttribute {
                attribute: name.clone(),
            });
        }
        let attribute = &schema.attributes[index];
        let predicate = match (condition, &attribute.kind) {
            (RuleCondition::Discrete { value, .. }, AttributeKind::Discrete { .. }) => {
                let value_index =
                    attribute
                        .value_index(value)
                        .ok_or_else(|| RejectReason::UnknownValue {
                            attribute: name.clone(),
                            value: value.clone(),
                        })?;
                Predicate::Equals(value_index)
            }
            (
                RuleCondition::Continuous { low, high, .. },
                AttributeKind::Continuous { min, max },
            ) => interval(name, *low, *high, *min, *max)?,
            _ => {
                return Err(RejectReason::TypeMismatch {
                    attribute: name.clone(),
                })
            }
        };
        specs.push((index, predicate));
    }
    Ok(Condition::from_specs(specs))
}

fn interval(
    name: &str,
    low: Option<f64>,
    high: Option<f64>,
    min: f64,
    max: f64,
) -> std::result::Result<Predicate, RejectReason> {
    if low.is_some_and(|bound| !bound.is_finite()) || high.is_some_and(|bound| !bound.is_finite()) {
        return Err(RejectReason::NonFiniteBound {
            attribute: name.to_string(),
        });
    }
    if let (Some(low), Some(high)) = (low, high) {
        if low > high {
            return Err(RejectReason::InvertedInterval {
                attribute: name.to_string(),
            });
        }
    }
    let low = low.unwrap_or(min);
    let high = high.unwrap_or(max);
    if low > max || high < min {
        return Err(RejectReason::OutsideDataRange {
            attribute: name.to_string(),
        });
    }
    Ok(Predicate::Interval { low, high })
}

pub fn from_condition(condition: &Condition, schema: &Schema) -> Vec<RuleCondition> {
    condition
        .specs()
        .iter()
        .map(|(index, predicate)| {
            let attribute = &schema.attributes[*index];
            match *predicate {
                Predicate::Equals(value) => RuleCondition::Discrete {
                    attribute: attribute.name.clone(),
                    value: attribute
                        .value_name(value)
                        .expect("discrete predicate refers to a value outside the schema")
                        .to_string(),
                },
                Predicate::Interval { low, high } => RuleCondition::Continuous {
                    attribute: attribute.name.clone(),
                    low: Some(low),
                    high: Some(high),
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::data::Attribute;
    use crate::error::Error;

    const EXAMPLE: &str = r#"{"dataset":"adult","source":"llm","model":"m","prompt_hash":"sha256:x","rules":[{"id":"llm-001","conditions":[{"attribute":"education-num","type":"continuous","low":13,"high":null},{"attribute":"marital-status","type":"discrete","value":"Married-civ-spouse"}],"class":">50K","rationale":"..."}]}"#;

    fn schema() -> Schema {
        Schema {
            attributes: vec![
                Attribute {
                    name: "color".to_string(),
                    kind: AttributeKind::Discrete {
                        values: vec!["blue".to_string(), "red".to_string()],
                    },
                },
                Attribute {
                    name: "age".to_string(),
                    kind: AttributeKind::Continuous {
                        min: 0.0,
                        max: 100.0,
                    },
                },
            ],
            class_column: "class".to_string(),
            classes: vec!["no".to_string(), "yes".to_string()],
        }
    }

    fn discrete(attribute: &str, value: &str) -> RuleCondition {
        RuleCondition::Discrete {
            attribute: attribute.to_string(),
            value: value.to_string(),
        }
    }

    fn continuous(attribute: &str, low: Option<f64>, high: Option<f64>) -> RuleCondition {
        RuleCondition::Continuous {
            attribute: attribute.to_string(),
            low,
            high,
        }
    }

    fn reject(conditions: Vec<RuleCondition>) -> RejectReason {
        to_condition(&conditions, &schema()).unwrap_err()
    }

    fn file_with(rules: Vec<serde_json::Value>) -> RuleFile {
        RuleFile {
            dataset: None,
            source: Origin::Llm,
            model: None,
            prompt_hash: None,
            rules,
        }
    }

    fn single_rejection(rule: serde_json::Value, rsl: usize) -> Rejection {
        let validated = validate(&file_with(vec![rule]), &schema(), rsl);
        assert!(validated.rules.is_empty());
        assert_eq!(validated.report.accepted, 0);
        assert_eq!(validated.report.rejected.len(), 1);
        validated.report.rejected[0].clone()
    }

    #[test]
    fn parses_example_rule_file() {
        let file = RuleFile::from_json_str(EXAMPLE).unwrap();
        assert_eq!(file.dataset.as_deref(), Some("adult"));
        assert_eq!(file.source, Origin::Llm);
        assert_eq!(file.model.as_deref(), Some("m"));
        assert_eq!(file.prompt_hash.as_deref(), Some("sha256:x"));
        assert_eq!(file.rules.len(), 1);
        let spec: RuleSpec = serde_json::from_value(file.rules[0].clone()).unwrap();
        assert_eq!(spec.id.as_deref(), Some("llm-001"));
        assert_eq!(
            spec.conditions,
            vec![
                continuous("education-num", Some(13.0), None),
                discrete("marital-status", "Married-civ-spouse"),
            ]
        );
        assert_eq!(spec.class, ">50K");
        assert_eq!(spec.rationale.as_deref(), Some("..."));
    }

    #[test]
    fn invalid_json_is_a_json_error() {
        assert!(matches!(
            RuleFile::from_json_str("{\"source\":"),
            Err(Error::Json(_))
        ));
        assert!(matches!(
            RuleFile::from_json_str("{\"rules\":[]}"),
            Err(Error::Json(_))
        ));
    }

    #[test]
    fn loads_rule_file_from_disk() {
        let path = std::env::temp_dir().join(format!(
            "exstracs-rules-load-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, EXAMPLE).unwrap();
        let loaded = RuleFile::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.unwrap(), RuleFile::from_json_str(EXAMPLE).unwrap());
    }

    #[test]
    fn validate_accepts_well_formed_rules() {
        let file = RuleFile::from_json_str(
            r#"{"source":"llm","rules":[
                {"id":"llm-001","conditions":[{"attribute":"age","type":"continuous","low":40,"high":null},{"attribute":"color","type":"discrete","value":"red"}],"class":"yes"},
                {"conditions":[{"attribute":"color","type":"discrete","value":"blue"}],"class":"no"}
            ]}"#,
        )
        .unwrap();
        let validated = validate(&file, &schema(), 2);
        assert_eq!(
            validated.report,
            ValidationReport {
                source: Origin::Llm,
                total: 2,
                accepted: 2,
                rejected: Vec::new(),
            }
        );
        assert_eq!(
            validated.rules[0],
            SeedRule {
                external_id: "llm-001".to_string(),
                condition: Condition::from_specs(vec![
                    (
                        1,
                        Predicate::Interval {
                            low: 40.0,
                            high: 100.0,
                        },
                    ),
                    (0, Predicate::Equals(1)),
                ]),
                class: 1,
                origin: Origin::Llm,
            }
        );
        assert_eq!(validated.rules[1].external_id, "rule-1");
        assert_eq!(
            validated.rules[1].condition,
            Condition::from_specs(vec![(0, Predicate::Equals(0))])
        );
        assert_eq!(validated.rules[1].class, 0);
    }

    #[test]
    fn origin_follows_file_source() {
        let mut file = file_with(vec![json!({
            "conditions": [{"attribute": "color", "type": "discrete", "value": "red"}],
            "class": "yes"
        })]);
        file.source = Origin::Expert;
        let validated = validate(&file, &schema(), 2);
        assert_eq!(validated.report.source, Origin::Expert);
        assert_eq!(validated.rules[0].origin, Origin::Expert);
    }

    #[test]
    fn rejects_malformed_rule() {
        let rejection = single_rejection(
            json!({"id": "llm-007", "conditions": "age > 3", "class": "yes"}),
            2,
        );
        assert_eq!(rejection.index, 0);
        assert_eq!(rejection.id, "llm-007");
        assert!(matches!(rejection.reason, RejectReason::Malformed { .. }));

        let rejection = single_rejection(json!({"id": 5, "class": "yes"}), 2);
        assert_eq!(rejection.id, "rule-0");
        assert!(matches!(rejection.reason, RejectReason::Malformed { .. }));
    }

    #[test]
    fn rejects_unknown_attribute() {
        assert_eq!(
            reject(vec![discrete("shape", "round")]),
            RejectReason::UnknownAttribute {
                attribute: "shape".to_string(),
            }
        );
    }

    #[test]
    fn rejects_type_mismatch() {
        assert_eq!(
            reject(vec![discrete("age", "old")]),
            RejectReason::TypeMismatch {
                attribute: "age".to_string(),
            }
        );
        assert_eq!(
            reject(vec![continuous("color", Some(0.0), Some(1.0))]),
            RejectReason::TypeMismatch {
                attribute: "color".to_string(),
            }
        );
    }

    #[test]
    fn rejects_unknown_value() {
        assert_eq!(
            reject(vec![discrete("color", "green")]),
            RejectReason::UnknownValue {
                attribute: "color".to_string(),
                value: "green".to_string(),
            }
        );
    }

    #[test]
    fn rejects_non_finite_bound() {
        let expected = RejectReason::NonFiniteBound {
            attribute: "age".to_string(),
        };
        assert_eq!(
            reject(vec![continuous("age", Some(f64::NAN), None)]),
            expected
        );
        assert_eq!(
            reject(vec![continuous("age", Some(10.0), Some(f64::INFINITY))]),
            expected
        );
    }

    #[test]
    fn rejects_inverted_interval() {
        assert_eq!(
            reject(vec![continuous("age", Some(60.0), Some(30.0))]),
            RejectReason::InvertedInterval {
                attribute: "age".to_string(),
            }
        );
    }

    #[test]
    fn rejects_interval_outside_data_range() {
        let expected = RejectReason::OutsideDataRange {
            attribute: "age".to_string(),
        };
        assert_eq!(reject(vec![continuous("age", Some(150.0), None)]), expected);
        assert_eq!(reject(vec![continuous("age", None, Some(-5.0))]), expected);
        assert_eq!(
            reject(vec![continuous("age", Some(-20.0), Some(-10.0))]),
            expected
        );
    }

    #[test]
    fn rejects_repeated_attribute() {
        assert_eq!(
            reject(vec![discrete("color", "blue"), discrete("color", "red")]),
            RejectReason::RepeatedAttribute {
                attribute: "color".to_string(),
            }
        );
    }

    #[test]
    fn rejects_unknown_class() {
        let rejection = single_rejection(
            json!({
                "id": "llm-002",
                "conditions": [{"attribute": "color", "type": "discrete", "value": "red"}],
                "class": "maybe"
            }),
            2,
        );
        assert_eq!(rejection.id, "llm-002");
        assert_eq!(
            rejection.reason,
            RejectReason::UnknownClass {
                class: "maybe".to_string(),
            }
        );
    }

    #[test]
    fn rejects_too_specific_rule() {
        let rejection = single_rejection(
            json!({
                "conditions": [
                    {"attribute": "color", "type": "discrete", "value": "red"},
                    {"attribute": "age", "type": "continuous", "low": 10, "high": 20}
                ],
                "class": "yes"
            }),
            1,
        );
        assert_eq!(
            rejection.reason,
            RejectReason::TooSpecific {
                specified: 2,
                limit: 1,
            }
        );
    }

    #[test]
    fn rejects_duplicate_of_earlier_accepted_rule() {
        let rule = |id: Option<&str>, class: &str| {
            let mut value = json!({
                "conditions": [{"attribute": "age", "type": "continuous", "low": 30, "high": null}],
                "class": class
            });
            if let Some(id) = id {
                value["id"] = json!(id);
            }
            value
        };
        let file = file_with(vec![
            rule(Some("bad"), "maybe"),
            rule(None, "yes"),
            rule(Some("other-class"), "no"),
            rule(Some("copy"), "yes"),
        ]);
        let validated = validate(&file, &schema(), 2);
        assert_eq!(validated.report.accepted, 2);
        assert_eq!(
            validated
                .rules
                .iter()
                .map(|rule| rule.external_id.as_str())
                .collect::<Vec<_>>(),
            vec!["rule-1", "other-class"]
        );
        assert_eq!(validated.report.rejected.len(), 2);
        assert_eq!(validated.report.rejected[0].id, "bad");
        assert_eq!(
            validated.report.rejected[1],
            Rejection {
                index: 3,
                id: "copy".to_string(),
                reason: RejectReason::Duplicate {
                    of: "rule-1".to_string(),
                },
            }
        );
    }

    #[test]
    fn reports_first_failing_check_only() {
        let rejection = single_rejection(
            json!({
                "conditions": [
                    {"attribute": "color", "type": "discrete", "value": "green"},
                    {"attribute": "shape", "type": "discrete", "value": "round"}
                ],
                "class": "maybe"
            }),
            0,
        );
        assert_eq!(
            rejection.reason,
            RejectReason::UnknownValue {
                attribute: "color".to_string(),
                value: "green".to_string(),
            }
        );
    }

    #[test]
    fn missing_bounds_default_to_attribute_range() {
        let interval_of = |low, high| {
            to_condition(&[continuous("age", low, high)], &schema())
                .unwrap()
                .specs()
                .to_vec()
        };
        assert_eq!(
            interval_of(None, Some(40.0)),
            vec![(
                1,
                Predicate::Interval {
                    low: 0.0,
                    high: 40.0,
                },
            )]
        );
        assert_eq!(
            interval_of(Some(20.0), None),
            vec![(
                1,
                Predicate::Interval {
                    low: 20.0,
                    high: 100.0,
                },
            )]
        );
        assert_eq!(
            interval_of(None, None),
            vec![(
                1,
                Predicate::Interval {
                    low: 0.0,
                    high: 100.0,
                },
            )]
        );
    }

    #[test]
    fn malformed_rule_does_not_abort_file() {
        let file = file_with(vec![
            json!({
                "conditions": [{"attribute": "color", "type": "discrete", "value": "red"}],
                "class": "yes"
            }),
            json!(42),
            json!({
                "conditions": [{"attribute": "age", "type": "interval", "low": 1}],
                "class": "no"
            }),
            json!({
                "conditions": [{"attribute": "color", "type": "discrete", "value": "blue"}],
                "class": "no"
            }),
        ]);
        let validated = validate(&file, &schema(), 2);
        assert_eq!(validated.report.total, 4);
        assert_eq!(validated.report.accepted, 2);
        assert_eq!(validated.rules.len(), 2);
        assert_eq!(validated.rules[1].external_id, "rule-3");
        let rejected: Vec<(usize, &str)> = validated
            .report
            .rejected
            .iter()
            .map(|rejection| (rejection.index, rejection.id.as_str()))
            .collect();
        assert_eq!(rejected, vec![(1, "rule-1"), (2, "rule-2")]);
        assert!(validated
            .report
            .rejected
            .iter()
            .all(|rejection| matches!(rejection.reason, RejectReason::Malformed { .. })));
    }

    #[test]
    fn condition_round_trips_through_rule_conditions() {
        let schema = schema();
        let conditions = vec![
            discrete("color", "red"),
            continuous("age", Some(18.5), Some(65.0)),
        ];
        let condition = to_condition(&conditions, &schema).unwrap();
        assert_eq!(
            condition.specs(),
            &[
                (0, Predicate::Equals(1)),
                (
                    1,
                    Predicate::Interval {
                        low: 18.5,
                        high: 65.0,
                    },
                ),
            ]
        );
        let back = from_condition(&condition, &schema);
        assert_eq!(back, conditions);
        assert_eq!(to_condition(&back, &schema).unwrap(), condition);
    }

    #[test]
    fn empty_condition_round_trips() {
        let schema = schema();
        let condition = to_condition(&[], &schema).unwrap();
        assert!(condition.is_empty());
        assert!(from_condition(&condition, &schema).is_empty());
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;

    #[test]
    fn rule_file_with_learned_origin_is_rejected() {
        for source in ["covering", "ga"] {
            let text = format!(r#"{{"source":"{source}","rules":[]}}"#);
            assert!(matches!(
                RuleFile::from_json_str(&text),
                Err(Error::Data(_))
            ));
        }
        assert!(RuleFile::from_json_str(r#"{"source":"llm","rules":[]}"#).is_ok());
    }
}
