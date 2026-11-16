use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::classifier::{Classifier, Origin};
use crate::config::Config;
use crate::data::Schema;
use crate::error::{Error, Result};
use crate::rules::{self, RuleCondition};

pub const SNAPSHOT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub format_version: u32,
    pub config: Config,
    pub schema: Schema,
    pub rule_specificity_limit: usize,
    pub class_counts: Vec<usize>,
    pub iteration: u64,
    pub next_id: u64,
    pub classifiers: Vec<ClassifierRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassifierRecord {
    pub id: u64,
    pub conditions: Vec<RuleCondition>,
    pub class: String,
    pub numerosity: u32,
    pub experience: u32,
    pub correct_count: u32,
    pub accuracy: f64,
    pub fitness: f64,
    pub avg_match_set_size: f64,
    pub ga_timestamp: u64,
    pub birth_iteration: u64,
    pub origin: Origin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parents: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seed_lineage: Vec<String>,
}

impl ClassifierRecord {
    pub fn from_classifier(classifier: &Classifier, schema: &Schema) -> ClassifierRecord {
        ClassifierRecord {
            id: classifier.id,
            conditions: rules::from_condition(&classifier.condition, schema),
            class: schema.classes[classifier.class as usize].clone(),
            numerosity: classifier.numerosity,
            experience: classifier.experience,
            correct_count: classifier.correct_count,
            accuracy: classifier.accuracy,
            fitness: classifier.fitness,
            avg_match_set_size: classifier.avg_match_set_size,
            ga_timestamp: classifier.ga_timestamp,
            birth_iteration: classifier.birth_iteration,
            origin: classifier.origin,
            external_id: classifier.external_id.clone(),
            parents: classifier.parents.clone(),
            seed_lineage: classifier.seed_lineage.clone(),
        }
    }

    pub fn to_classifier(&self, schema: &Schema) -> Result<Classifier> {
        let id = self.id;
        let condition = rules::to_condition(&self.conditions, schema)
            .map_err(|reason| Error::Data(format!("classifier {id}: {reason:?}")))?;
        let class = schema.class_index(&self.class).ok_or_else(|| {
            Error::Data(format!("classifier {id}: unknown class {:?}", self.class))
        })?;
        Ok(Classifier {
            id,
            condition,
            class,
            numerosity: self.numerosity,
            experience: self.experience,
            correct_count: self.correct_count,
            accuracy: self.accuracy,
            fitness: self.fitness,
            avg_match_set_size: self.avg_match_set_size,
            ga_timestamp: self.ga_timestamp,
            birth_iteration: self.birth_iteration,
            origin: self.origin,
            external_id: self.external_id.clone(),
            parents: self.parents.clone(),
            seed_lineage: self.seed_lineage.clone(),
        })
    }
}

impl Snapshot {
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Snapshot> {
        let text = std::fs::read_to_string(path)?;
        let snapshot: Snapshot = serde_json::from_str(&text)?;
        if snapshot.format_version != SNAPSHOT_FORMAT_VERSION {
            return Err(Error::Data(format!(
                "unsupported snapshot format version {} (expected {SNAPSHOT_FORMAT_VERSION})",
                snapshot.format_version
            )));
        }
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::classifier::{Condition, Predicate};
    use crate::data::{Attribute, AttributeKind};

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

    fn classifier() -> Classifier {
        Classifier {
            id: 7,
            condition: Condition::from_specs(vec![
                (0, Predicate::Equals(1)),
                (
                    1,
                    Predicate::Interval {
                        low: 18.5,
                        high: 65.25,
                    },
                ),
            ]),
            class: 1,
            numerosity: 3,
            experience: 40,
            correct_count: 31,
            accuracy: 0.775,
            fitness: 0.123_456_789,
            avg_match_set_size: 12.5,
            ga_timestamp: 900,
            birth_iteration: 100,
            origin: Origin::Llm,
            external_id: Some("llm-001".to_string()),
            parents: vec![2, 5],
            seed_lineage: vec!["llm-001".to_string()],
        }
    }

    fn snapshot() -> Snapshot {
        let schema = schema();
        let classifiers = vec![ClassifierRecord::from_classifier(&classifier(), &schema)];
        Snapshot {
            format_version: SNAPSHOT_FORMAT_VERSION,
            config: Config::default(),
            schema,
            rule_specificity_limit: 2,
            class_counts: vec![10, 5],
            iteration: 1234,
            next_id: 8,
            classifiers,
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "exstracs-snapshot-{name}-{}-{nanos}.json",
            std::process::id()
        ))
    }

    #[test]
    fn classifier_record_round_trip() {
        let schema = schema();
        let original = classifier();
        let record = ClassifierRecord::from_classifier(&original, &schema);
        assert_eq!(record.id, 7);
        assert_eq!(record.class, "yes");
        assert_eq!(
            record.conditions,
            vec![
                RuleCondition::Discrete {
                    attribute: "color".to_string(),
                    value: "red".to_string(),
                },
                RuleCondition::Continuous {
                    attribute: "age".to_string(),
                    low: Some(18.5),
                    high: Some(65.25),
                },
            ]
        );
        assert_eq!(record.parents, vec![2, 5]);
        assert_eq!(record.to_classifier(&schema).unwrap(), original);
    }

    #[test]
    fn record_without_optional_fields_round_trips_through_json() {
        let schema = schema();
        let mut original = classifier();
        original.external_id = None;
        original.parents.clear();
        original.seed_lineage.clear();
        original.origin = Origin::Covering;
        let record = ClassifierRecord::from_classifier(&original, &schema);
        let text = serde_json::to_string(&record).unwrap();
        assert!(!text.contains("external_id"));
        assert!(!text.contains("parents"));
        let parsed: ClassifierRecord = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed, record);
        assert_eq!(parsed.to_classifier(&schema).unwrap(), original);
    }

    #[test]
    fn to_classifier_rejects_unknown_class() {
        let schema = schema();
        let mut record = ClassifierRecord::from_classifier(&classifier(), &schema);
        record.class = "maybe".to_string();
        match record.to_classifier(&schema) {
            Err(Error::Data(message)) => {
                assert!(message.contains("classifier 7"));
                assert!(message.contains("maybe"));
            }
            other => panic!("expected data error, got {other:?}"),
        }
    }

    #[test]
    fn to_classifier_rejects_invalid_condition() {
        let schema = schema();
        let mut record = ClassifierRecord::from_classifier(&classifier(), &schema);
        record.conditions.push(RuleCondition::Discrete {
            attribute: "shape".to_string(),
            value: "round".to_string(),
        });
        match record.to_classifier(&schema) {
            Err(Error::Data(message)) => {
                assert!(message.starts_with("classifier 7: "));
                assert!(message.contains("UnknownAttribute"));
            }
            other => panic!("expected data error, got {other:?}"),
        }
    }

    #[test]
    fn snapshot_save_load_round_trip() {
        let original = snapshot();
        let path = temp_path("round-trip");
        original.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let loaded = Snapshot::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(text.ends_with("}\n"));
        let loaded = loaded.unwrap();
        assert_eq!(loaded, original);
        assert_eq!(
            loaded.classifiers[0].to_classifier(&loaded.schema).unwrap(),
            classifier()
        );
    }

    #[test]
    fn snapshot_load_rejects_wrong_format_version() {
        let mut original = snapshot();
        original.format_version = SNAPSHOT_FORMAT_VERSION + 1;
        let path = temp_path("wrong-version");
        original.save(&path).unwrap();
        let loaded = Snapshot::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(loaded, Err(Error::Data(_))));
    }

    #[test]
    fn snapshot_load_reports_missing_file_and_bad_json() {
        let missing = temp_path("missing");
        assert!(matches!(Snapshot::load(&missing), Err(Error::Io(_))));
        let path = temp_path("bad-json");
        std::fs::write(&path, "{\"format_version\": 1").unwrap();
        let loaded = Snapshot::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(loaded, Err(Error::Json(_))));
    }
}
