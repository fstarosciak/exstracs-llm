use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

pub const UNSEEN_VALUE: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AttributeKind {
    Discrete { values: Vec<String> },
    Continuous { min: f64, max: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attribute {
    pub name: String,
    #[serde(flatten)]
    pub kind: AttributeKind,
}

impl Attribute {
    pub fn is_discrete(&self) -> bool {
        matches!(self.kind, AttributeKind::Discrete { .. })
    }

    pub fn range(&self) -> f64 {
        match self.kind {
            AttributeKind::Discrete { .. } => 0.0,
            AttributeKind::Continuous { min, max } => max - min,
        }
    }

    pub fn value_index(&self, value: &str) -> Option<u32> {
        match &self.kind {
            AttributeKind::Discrete { values } => values
                .iter()
                .position(|candidate| candidate == value)
                .map(|index| index as u32),
            AttributeKind::Continuous { .. } => None,
        }
    }

    pub fn value_name(&self, index: u32) -> Option<&str> {
        match &self.kind {
            AttributeKind::Discrete { values } => values.get(index as usize).map(String::as_str),
            AttributeKind::Continuous { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    pub attributes: Vec<Attribute>,
    pub class_column: String,
    pub classes: Vec<String>,
}

impl Schema {
    pub fn attribute_index(&self, name: &str) -> Option<usize> {
        self.attributes
            .iter()
            .position(|attribute| attribute.name == name)
    }

    pub fn class_index(&self, name: &str) -> Option<u32> {
        self.classes
            .iter()
            .position(|class| class == name)
            .map(|index| index as u32)
    }

    pub fn n_attributes(&self) -> usize {
        self.attributes.len()
    }

    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Missing,
    Discrete(u32),
    Continuous(f64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn read_csv(path: &Path) -> Result<Table> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .flexible(false)
            .from_path(path)?;
        let header = reader.headers()?.iter().map(String::from).collect();
        let mut rows = Vec::new();
        for record in reader.records() {
            rows.push(record?.iter().map(String::from).collect());
        }
        Ok(Table { header, rows })
    }

    pub fn write_csv(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record(&self.header)?;
        for row in &self.rows {
            writer.write_record(row)?;
        }
        writer.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadOptions {
    pub class_column: String,
    pub discrete_limit: usize,
    pub discrete: Vec<String>,
    pub continuous: Vec<String>,
    pub ignore: Vec<String>,
    pub missing_tokens: Vec<String>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            class_column: "class".into(),
            discrete_limit: 10,
            discrete: Vec::new(),
            continuous: Vec::new(),
            ignore: Vec::new(),
            missing_tokens: ["", "NA", "NaN", "?"].map(String::from).to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub schema: Schema,
    pub rows: Vec<Vec<Value>>,
    pub labels: Vec<u32>,
}

impl Dataset {
    pub fn from_table(table: &Table, options: &LoadOptions) -> Result<Dataset> {
        check_table(table)?;
        if table.rows.is_empty() {
            return Err(Error::Data("table has no data rows".into()));
        }
        let class_position = column_position(&table.header, &options.class_column)?;
        for name in &options.ignore {
            column_position(&table.header, name)?;
        }
        let positions: Vec<usize> = (0..table.header.len())
            .filter(|&position| {
                position != class_position && !options.ignore.contains(&table.header[position])
            })
            .collect();
        for name in options.discrete.iter().chain(&options.continuous) {
            if !positions
                .iter()
                .any(|&position| &table.header[position] == name)
            {
                return Err(Error::Data(format!(
                    "forced column '{name}' is not an attribute column"
                )));
            }
        }
        if let Some(name) = options
            .discrete
            .iter()
            .find(|name| options.continuous.contains(name))
        {
            return Err(Error::Data(format!(
                "column '{name}' is forced both discrete and continuous"
            )));
        }
        let mut class_names = BTreeSet::new();
        for (index, row) in table.rows.iter().enumerate() {
            let cell = row[class_position].as_str();
            if is_missing(cell, &options.missing_tokens) {
                return Err(missing_class(index));
            }
            class_names.insert(cell);
        }
        let classes: Vec<String> = class_names.into_iter().map(String::from).collect();
        let attributes = positions
            .iter()
            .map(|&position| infer_attribute(table, position, options))
            .collect::<Result<Vec<_>>>()?;
        let (rows, labels) = encode_rows(
            table,
            class_position,
            &positions,
            &attributes,
            &classes,
            &options.missing_tokens,
        )?;
        Ok(Dataset {
            schema: Schema {
                attributes,
                class_column: options.class_column.clone(),
                classes,
            },
            rows,
            labels,
        })
    }

    pub fn from_table_with_schema(
        table: &Table,
        schema: &Schema,
        options: &LoadOptions,
    ) -> Result<Dataset> {
        check_table(table)?;
        let class_position = column_position(&table.header, &schema.class_column)?;
        for (position, name) in table.header.iter().enumerate() {
            if position != class_position
                && schema.attribute_index(name).is_none()
                && !options.ignore.contains(name)
            {
                return Err(Error::Data(format!(
                    "column '{name}' is not an attribute of the schema"
                )));
            }
        }
        let positions = schema
            .attributes
            .iter()
            .map(|attribute| column_position(&table.header, &attribute.name))
            .collect::<Result<Vec<_>>>()?;
        let (rows, labels) = encode_rows(
            table,
            class_position,
            &positions,
            &schema.attributes,
            &schema.classes,
            &options.missing_tokens,
        )?;
        Ok(Dataset {
            schema: schema.clone(),
            rows,
            labels,
        })
    }

    pub fn from_csv(path: &Path, options: &LoadOptions) -> Result<Dataset> {
        Dataset::from_table(&Table::read_csv(path)?, options)
    }

    pub fn from_csv_with_schema(
        path: &Path,
        schema: &Schema,
        options: &LoadOptions,
    ) -> Result<Dataset> {
        Dataset::from_table_with_schema(&Table::read_csv(path)?, schema, options)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn class_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.schema.n_classes()];
        for &label in &self.labels {
            counts[label as usize] += 1;
        }
        counts
    }
}

fn is_missing(cell: &str, missing_tokens: &[String]) -> bool {
    missing_tokens.iter().any(|token| token == cell)
}

fn parse_finite(cell: &str) -> Option<f64> {
    cell.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn missing_class(index: usize) -> Error {
    Error::Data(format!("data row {}: missing class value", index + 1))
}

fn column_position(header: &[String], name: &str) -> Result<usize> {
    header
        .iter()
        .position(|column| column == name)
        .ok_or_else(|| Error::Data(format!("column '{name}' not found in header")))
}

fn check_table(table: &Table) -> Result<()> {
    for (index, name) in table.header.iter().enumerate() {
        if table.header[..index].contains(name) {
            return Err(Error::Data(format!("duplicate column '{name}' in header")));
        }
    }
    for (index, row) in table.rows.iter().enumerate() {
        if row.len() != table.header.len() {
            return Err(Error::Data(format!(
                "data row {}: expected {} cells, found {}",
                index + 1,
                table.header.len(),
                row.len()
            )));
        }
    }
    Ok(())
}

fn discrete_kind(cells: &[&str]) -> AttributeKind {
    let values: BTreeSet<&str> = cells.iter().copied().collect();
    AttributeKind::Discrete {
        values: values.into_iter().map(String::from).collect(),
    }
}

fn infer_attribute(table: &Table, position: usize, options: &LoadOptions) -> Result<Attribute> {
    let name = table.header[position].clone();
    let cells: Vec<&str> = table
        .rows
        .iter()
        .map(|row| row[position].as_str())
        .filter(|cell| !is_missing(cell, &options.missing_tokens))
        .collect();
    let forced_continuous = options.continuous.contains(&name);
    let kind = if options.discrete.contains(&name) {
        discrete_kind(&cells)
    } else {
        match cells
            .iter()
            .map(|cell| parse_finite(cell))
            .collect::<Option<Vec<f64>>>()
        {
            Some(mut numbers) => {
                numbers.sort_by(f64::total_cmp);
                numbers.dedup();
                if forced_continuous || numbers.len() > options.discrete_limit {
                    AttributeKind::Continuous {
                        min: numbers.first().copied().unwrap_or(0.0),
                        max: numbers.last().copied().unwrap_or(0.0),
                    }
                } else {
                    discrete_kind(&cells)
                }
            }
            None if forced_continuous => {
                let cell = cells
                    .iter()
                    .find(|cell| parse_finite(cell).is_none())
                    .copied()
                    .unwrap_or_default();
                return Err(Error::Data(format!(
                    "column '{name}' is forced continuous but has non-numeric value '{cell}'"
                )));
            }
            None => discrete_kind(&cells),
        }
    };
    Ok(Attribute { name, kind })
}

fn encode_rows(
    table: &Table,
    class_position: usize,
    positions: &[usize],
    attributes: &[Attribute],
    classes: &[String],
    missing_tokens: &[String],
) -> Result<(Vec<Vec<Value>>, Vec<u32>)> {
    let domains: Vec<Option<BTreeMap<&str, u32>>> = attributes
        .iter()
        .map(|attribute| match &attribute.kind {
            AttributeKind::Discrete { values } => Some(
                values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (value.as_str(), index as u32))
                    .collect(),
            ),
            AttributeKind::Continuous { .. } => None,
        })
        .collect();
    let class_indices: BTreeMap<&str, u32> = classes
        .iter()
        .enumerate()
        .map(|(index, class)| (class.as_str(), index as u32))
        .collect();
    let mut rows = Vec::with_capacity(table.rows.len());
    let mut labels = Vec::with_capacity(table.rows.len());
    for (index, row) in table.rows.iter().enumerate() {
        let class = row[class_position].as_str();
        if is_missing(class, missing_tokens) {
            return Err(missing_class(index));
        }
        let label = class_indices.get(class).copied().ok_or_else(|| {
            Error::Data(format!("data row {}: unknown class '{class}'", index + 1))
        })?;
        let mut values = Vec::with_capacity(attributes.len());
        for ((&position, attribute), domain) in positions.iter().zip(attributes).zip(&domains) {
            let cell = row[position].as_str();
            let value = if is_missing(cell, missing_tokens) {
                Value::Missing
            } else if let Some(domain) = domain {
                Value::Discrete(domain.get(cell).copied().unwrap_or(UNSEEN_VALUE))
            } else {
                Value::Continuous(parse_finite(cell).ok_or_else(|| {
                    Error::Data(format!(
                        "data row {}: column '{}' has non-numeric value '{cell}'",
                        index + 1,
                        attribute.name
                    ))
                })?)
            };
            values.push(value);
        }
        rows.push(values);
        labels.push(label);
    }
    Ok((rows, labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(header: &[&str], rows: &[&[&str]]) -> Table {
        Table {
            header: header.iter().map(|cell| cell.to_string()).collect(),
            rows: rows
                .iter()
                .map(|row| row.iter().map(|cell| cell.to_string()).collect())
                .collect(),
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "exstracs-data-test-{}-{name}.csv",
            std::process::id()
        ))
    }

    fn training_table() -> Table {
        let rows: Vec<Vec<String>> = (0..12)
            .map(|index| {
                vec![
                    format!("{}", index as f64 * 0.5),
                    format!("{}", index % 3),
                    ["red", "green", "blue"][index % 3].to_string(),
                    ["no", "yes"][index % 2].to_string(),
                ]
            })
            .collect();
        Table {
            header: strings(&["x", "y", "z", "class"]),
            rows,
        }
    }

    #[test]
    fn infers_types() {
        let dataset = Dataset::from_table(&training_table(), &LoadOptions::default()).unwrap();
        let attributes = &dataset.schema.attributes;
        assert_eq!(attributes.len(), 3);
        assert_eq!(attributes[0].name, "x");
        assert_eq!(
            attributes[0].kind,
            AttributeKind::Continuous { min: 0.0, max: 5.5 }
        );
        assert_eq!(attributes[0].range(), 5.5);
        assert_eq!(
            attributes[1].kind,
            AttributeKind::Discrete {
                values: strings(&["0", "1", "2"])
            }
        );
        assert_eq!(
            attributes[2].kind,
            AttributeKind::Discrete {
                values: strings(&["blue", "green", "red"])
            }
        );
        assert_eq!(dataset.len(), 12);
        assert_eq!(dataset.rows[1][0], Value::Continuous(0.5));
        assert_eq!(dataset.rows[1][1], Value::Discrete(1));
        assert_eq!(dataset.rows[1][2], Value::Discrete(1));
        assert_eq!(dataset.schema.class_column, "class");
    }

    #[test]
    fn discrete_limit_counts_distinct_parsed_values() {
        let source = table(
            &["x", "class"],
            &[&["1", "a"], &["1.0", "a"], &["2", "b"], &["10", "b"]],
        );
        let options = LoadOptions {
            discrete_limit: 2,
            ..LoadOptions::default()
        };
        let dataset = Dataset::from_table(&source, &options).unwrap();
        assert_eq!(
            dataset.schema.attributes[0].kind,
            AttributeKind::Continuous {
                min: 1.0,
                max: 10.0
            }
        );
        let options = LoadOptions {
            discrete_limit: 3,
            ..LoadOptions::default()
        };
        let dataset = Dataset::from_table(&source, &options).unwrap();
        assert_eq!(
            dataset.schema.attributes[0].kind,
            AttributeKind::Discrete {
                values: strings(&["1", "1.0", "10", "2"])
            }
        );
        assert_eq!(dataset.rows[3][0], Value::Discrete(2));
    }

    #[test]
    fn non_finite_values_are_not_numeric() {
        let rows: Vec<Vec<String>> = (0..12)
            .map(|index| vec![format!("{index}"), "a".to_string()])
            .chain(std::iter::once(vec!["inf".to_string(), "a".to_string()]))
            .collect();
        let source = Table {
            header: strings(&["x", "class"]),
            rows,
        };
        let dataset = Dataset::from_table(&source, &LoadOptions::default()).unwrap();
        assert!(dataset.schema.attributes[0].is_discrete());
        assert_eq!(dataset.schema.attributes[0].value_index("inf"), Some(12));
    }

    #[test]
    fn missing_tokens_become_missing() {
        let source = table(
            &["x", "y", "class"],
            &[
                &["?", "a", "c1"],
                &["NA", "", "c1"],
                &["NaN", "b", "c2"],
                &["1", "-", "c2"],
            ],
        );
        let dataset = Dataset::from_table(&source, &LoadOptions::default()).unwrap();
        assert_eq!(
            dataset.schema.attributes[0].kind,
            AttributeKind::Discrete {
                values: strings(&["1"])
            }
        );
        assert_eq!(
            dataset.schema.attributes[1].kind,
            AttributeKind::Discrete {
                values: strings(&["-", "a", "b"])
            }
        );
        assert_eq!(dataset.rows[0], vec![Value::Missing, Value::Discrete(1)]);
        assert_eq!(dataset.rows[1], vec![Value::Missing, Value::Missing]);
        assert_eq!(dataset.rows[2], vec![Value::Missing, Value::Discrete(2)]);
        assert_eq!(
            dataset.rows[3],
            vec![Value::Discrete(0), Value::Discrete(0)]
        );

        let options = LoadOptions {
            missing_tokens: strings(&["-"]),
            ..LoadOptions::default()
        };
        let dataset = Dataset::from_table(&source, &options).unwrap();
        assert_eq!(
            dataset.schema.attributes[1].kind,
            AttributeKind::Discrete {
                values: strings(&["", "a", "b"])
            }
        );
        assert_eq!(dataset.rows[3][1], Value::Missing);
        assert_eq!(dataset.rows[1][1], Value::Discrete(0));
    }

    #[test]
    fn all_missing_column_is_empty_discrete() {
        let source = table(&["x", "class"], &[&["?", "a"], &["", "b"]]);
        let dataset = Dataset::from_table(&source, &LoadOptions::default()).unwrap();
        assert_eq!(
            dataset.schema.attributes[0].kind,
            AttributeKind::Discrete { values: Vec::new() }
        );
        assert_eq!(
            dataset.rows,
            vec![vec![Value::Missing], vec![Value::Missing]]
        );
    }

    #[test]
    fn forced_types() {
        let options = LoadOptions {
            discrete: strings(&["x"]),
            continuous: strings(&["y"]),
            ..LoadOptions::default()
        };
        let dataset = Dataset::from_table(&training_table(), &options).unwrap();
        assert!(dataset.schema.attributes[0].is_discrete());
        assert_eq!(dataset.schema.attributes[0].value_name(0), Some("0"));
        assert_eq!(
            dataset.schema.attributes[1].kind,
            AttributeKind::Continuous { min: 0.0, max: 2.0 }
        );
        assert_eq!(dataset.rows[2][1], Value::Continuous(2.0));

        let options = LoadOptions {
            continuous: strings(&["z"]),
            ..LoadOptions::default()
        };
        assert!(matches!(
            Dataset::from_table(&training_table(), &options),
            Err(Error::Data(_))
        ));

        let source = table(&["x", "class"], &[&["?", "a"], &["NA", "b"]]);
        let options = LoadOptions {
            continuous: strings(&["x"]),
            ..LoadOptions::default()
        };
        let dataset = Dataset::from_table(&source, &options).unwrap();
        assert_eq!(
            dataset.schema.attributes[0].kind,
            AttributeKind::Continuous { min: 0.0, max: 0.0 }
        );
    }

    #[test]
    fn forced_type_errors() {
        for options in [
            LoadOptions {
                discrete: strings(&["missing"]),
                ..LoadOptions::default()
            },
            LoadOptions {
                continuous: strings(&["class"]),
                ..LoadOptions::default()
            },
            LoadOptions {
                discrete: strings(&["x"]),
                ignore: strings(&["x"]),
                ..LoadOptions::default()
            },
            LoadOptions {
                discrete: strings(&["x"]),
                continuous: strings(&["x"]),
                ..LoadOptions::default()
            },
        ] {
            assert!(matches!(
                Dataset::from_table(&training_table(), &options),
                Err(Error::Data(_))
            ));
        }
    }

    #[test]
    fn ignore_drops_columns() {
        let options = LoadOptions {
            ignore: strings(&["y"]),
            ..LoadOptions::default()
        };
        let dataset = Dataset::from_table(&training_table(), &options).unwrap();
        let names: Vec<&str> = dataset
            .schema
            .attributes
            .iter()
            .map(|attribute| attribute.name.as_str())
            .collect();
        assert_eq!(names, vec!["x", "z"]);
        assert_eq!(dataset.rows[0].len(), 2);

        let options = LoadOptions {
            ignore: strings(&["nope"]),
            ..LoadOptions::default()
        };
        assert!(matches!(
            Dataset::from_table(&training_table(), &options),
            Err(Error::Data(_))
        ));
    }

    #[test]
    fn classes_sorted_and_labels_indexed() {
        let source = table(
            &["label", "x"],
            &[&["c", "1"], &["a", "2"], &["b", "1"], &["c", "2"]],
        );
        let options = LoadOptions {
            class_column: "label".into(),
            ..LoadOptions::default()
        };
        let dataset = Dataset::from_table(&source, &options).unwrap();
        assert_eq!(dataset.schema.classes, strings(&["a", "b", "c"]));
        assert_eq!(dataset.labels, vec![2, 0, 1, 2]);
        assert_eq!(dataset.class_counts(), vec![1, 1, 2]);
        assert_eq!(dataset.schema.class_column, "label");
        assert_eq!(dataset.schema.attributes[0].name, "x");
    }

    #[test]
    fn table_errors() {
        let source = table(&["x", "class"], &[&["1", "a"], &["2", "?"]]);
        match Dataset::from_table(&source, &LoadOptions::default()) {
            Err(Error::Data(message)) => assert!(message.contains("row 2")),
            other => panic!("unexpected {other:?}"),
        }
        let source = table(&["x", "label"], &[&["1", "a"]]);
        assert!(matches!(
            Dataset::from_table(&source, &LoadOptions::default()),
            Err(Error::Data(_))
        ));
        let source = table(&["x", "class"], &[]);
        assert!(matches!(
            Dataset::from_table(&source, &LoadOptions::default()),
            Err(Error::Data(_))
        ));
        let source = table(&["x", "class"], &[&["1"]]);
        assert!(matches!(
            Dataset::from_table(&source, &LoadOptions::default()),
            Err(Error::Data(_))
        ));
        let source = table(&["x", "x", "class"], &[&["1", "2", "a"]]);
        assert!(matches!(
            Dataset::from_table(&source, &LoadOptions::default()),
            Err(Error::Data(_))
        ));
    }

    #[test]
    fn with_schema_maps_by_name() {
        let options = LoadOptions {
            ignore: strings(&["id"]),
            ..LoadOptions::default()
        };
        let train = Dataset::from_table(&training_table(), &LoadOptions::default()).unwrap();
        let test = table(
            &["class", "id", "z", "x", "y"],
            &[
                &["yes", "7", "green", "1.25", "2"],
                &["no", "8", "purple", "?", "5"],
                &["yes", "9", "?", "100", "0"],
            ],
        );
        let dataset = Dataset::from_table_with_schema(&test, &train.schema, &options).unwrap();
        assert_eq!(dataset.schema, train.schema);
        assert_eq!(dataset.labels, vec![1, 0, 1]);
        assert_eq!(
            dataset.rows[0],
            vec![
                Value::Continuous(1.25),
                Value::Discrete(2),
                Value::Discrete(1)
            ]
        );
        assert_eq!(
            dataset.rows[1],
            vec![
                Value::Missing,
                Value::Discrete(UNSEEN_VALUE),
                Value::Discrete(UNSEEN_VALUE)
            ]
        );
        assert_eq!(
            dataset.rows[2],
            vec![Value::Continuous(100.0), Value::Discrete(0), Value::Missing]
        );

        let empty = table(&["x", "y", "z", "class"], &[]);
        let dataset =
            Dataset::from_table_with_schema(&empty, &train.schema, &LoadOptions::default())
                .unwrap();
        assert!(dataset.is_empty());
    }

    #[test]
    fn with_schema_errors() {
        let train = Dataset::from_table(&training_table(), &LoadOptions::default()).unwrap();
        let cases: Vec<Table> = vec![
            table(&["x", "y", "class"], &[&["1", "1", "yes"]]),
            table(&["x", "y", "z"], &[&["1", "1", "red"]]),
            table(
                &["x", "y", "z", "id", "class"],
                &[&["1", "1", "red", "3", "yes"]],
            ),
            table(&["x", "y", "z", "class"], &[&["1", "1", "red", "maybe"]]),
            table(&["x", "y", "z", "class"], &[&["1", "1", "red", "?"]]),
            table(&["x", "y", "z", "class"], &[&["abc", "1", "red", "no"]]),
            table(&["x", "y", "z", "class"], &[&["inf", "1", "red", "no"]]),
        ];
        for case in cases {
            assert!(matches!(
                Dataset::from_table_with_schema(&case, &train.schema, &LoadOptions::default()),
                Err(Error::Data(_))
            ));
        }
    }

    #[test]
    fn csv_round_trip() {
        let path = temp_path("round-trip");
        let source = table(
            &["x", "name", "class"],
            &[&["1.5", "a, b", "yes"], &["", "\"quoted\"", "no"]],
        );
        source.write_csv(&path).unwrap();
        let loaded = Table::read_csv(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded, source);
    }

    #[test]
    fn csv_trims_and_rejects_ragged_rows() {
        let path = temp_path("trim");
        std::fs::write(&path, " x , class \n 1 , a \n2,b\n").unwrap();
        let loaded = Table::read_csv(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded, table(&["x", "class"], &[&["1", "a"], &["2", "b"]]));

        let path = temp_path("ragged");
        std::fs::write(&path, "x,class\n1,a\n2\n").unwrap();
        let result = Table::read_csv(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(result, Err(Error::Csv(_))));

        let path = temp_path("dataset");
        std::fs::write(&path, "x,class\n1,a\n2,b\n").unwrap();
        let dataset = Dataset::from_csv(&path, &LoadOptions::default()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(dataset.labels, vec![0, 1]);
    }
}
