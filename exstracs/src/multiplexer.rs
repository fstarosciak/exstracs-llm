use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::data::Table;
use crate::error::{Error, Result};

pub fn multiplexer_class(bits: &[u8], address_bits: usize) -> u8 {
    let expected = address_bits + (1usize << address_bits);
    assert_eq!(
        bits.len(),
        expected,
        "multiplexer with {address_bits} address bits needs {expected} bits"
    );
    let address = bits[..address_bits]
        .iter()
        .fold(0usize, |address, &bit| (address << 1) | usize::from(bit));
    bits[address_bits + address]
}

pub fn multiplexer_table(address_bits: usize, samples: Option<usize>, seed: u64) -> Result<Table> {
    if !(1..=7).contains(&address_bits) {
        return Err(Error::Config(format!(
            "multiplexer address bits must be in 1..=7, got {address_bits}"
        )));
    }
    let registers = 1usize << address_bits;
    let n = address_bits + registers;
    let mut header: Vec<String> = (0..address_bits).map(|index| format!("A{index}")).collect();
    header.extend((0..registers).map(|index| format!("R{index}")));
    header.push("class".into());
    let bit_rows: Vec<Vec<u8>> = match samples {
        None => {
            if n > 20 {
                return Err(Error::Config(
                    "complete table too large; pass samples".into(),
                ));
            }
            (0..1usize << n)
                .map(|value| {
                    (0..n)
                        .map(|column| u8::from((value >> (n - 1 - column)) & 1 == 1))
                        .collect()
                })
                .collect()
        }
        Some(count) => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut rows = Vec::with_capacity(count);
            for _ in 0..count {
                let row: Vec<u8> = (0..n).map(|_| u8::from(rng.gen_bool(0.5))).collect();
                rows.push(row);
            }
            rows
        }
    };
    let rows = bit_rows
        .iter()
        .map(|bits| {
            let class = multiplexer_class(bits, address_bits);
            bits.iter()
                .chain(std::iter::once(&class))
                .map(|bit| bit.to_string())
                .collect()
        })
        .collect();
    Ok(Table { header, rows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{AttributeKind, Dataset, LoadOptions};

    #[test]
    fn six_multiplexer_cases() {
        assert_eq!(multiplexer_class(&[0, 1, 0, 1, 0, 0], 2), 1);
        assert_eq!(multiplexer_class(&[1, 1, 0, 0, 0, 1], 2), 1);
        assert_eq!(multiplexer_class(&[0, 0, 1, 0, 0, 0], 2), 1);
        assert_eq!(multiplexer_class(&[0, 0, 0, 1, 1, 1], 2), 0);
        assert_eq!(multiplexer_class(&[1, 0, 1, 1, 0, 1], 2), 0);
        assert_eq!(multiplexer_class(&[1, 0, 0, 0, 1, 0], 2), 1);
        assert_eq!(multiplexer_class(&[1, 0, 1], 1), 1);
        assert_eq!(multiplexer_class(&[0, 0, 1], 1), 0);
    }

    #[test]
    #[should_panic]
    fn wrong_length_panics() {
        multiplexer_class(&[0, 1, 0, 1, 0], 2);
    }

    #[test]
    fn complete_six_multiplexer_table() {
        let table = multiplexer_table(2, None, 0).unwrap();
        assert_eq!(
            table.header,
            vec!["A0", "A1", "R0", "R1", "R2", "R3", "class"]
        );
        assert_eq!(table.rows.len(), 64);
        assert!(table.rows.iter().all(|row| row.len() == 7));
        assert_eq!(table.rows[0], vec!["0"; 7]);
        assert_eq!(table.rows[1], vec!["0", "0", "0", "0", "0", "1", "0"]);
        assert_eq!(table.rows[8], vec!["0", "0", "1", "0", "0", "0", "1"]);
        assert_eq!(table.rows[63], vec!["1"; 7]);
        let ones = table.rows.iter().filter(|row| row[6] == "1").count();
        assert_eq!(ones, 32);
        for row in &table.rows {
            let bits: Vec<u8> = row[..6].iter().map(|cell| cell.parse().unwrap()).collect();
            assert_eq!(row[6], multiplexer_class(&bits, 2).to_string());
        }
    }

    #[test]
    fn sampled_table_is_deterministic() {
        let first = multiplexer_table(3, Some(50), 7).unwrap();
        let second = multiplexer_table(3, Some(50), 7).unwrap();
        let other = multiplexer_table(3, Some(50), 8).unwrap();
        assert_eq!(first, second);
        assert_ne!(first.rows, other.rows);
        assert_eq!(first.rows.len(), 50);
        assert_eq!(first.header.len(), 12);
        assert_eq!(first.header[3], "R0");
        for row in &first.rows {
            let bits: Vec<u8> = row[..11].iter().map(|cell| cell.parse().unwrap()).collect();
            assert!(bits.iter().all(|&bit| bit <= 1));
            assert_eq!(row[11], multiplexer_class(&bits, 3).to_string());
        }
        assert!(multiplexer_table(3, Some(0), 7).unwrap().rows.is_empty());
    }

    #[test]
    fn six_multiplexer_dataset() {
        let table = multiplexer_table(2, None, 0).unwrap();
        let dataset = Dataset::from_table(&table, &LoadOptions::default()).unwrap();
        assert_eq!(dataset.schema.n_attributes(), 6);
        for attribute in &dataset.schema.attributes {
            assert_eq!(
                attribute.kind,
                AttributeKind::Discrete {
                    values: vec!["0".to_string(), "1".to_string()]
                }
            );
        }
        assert_eq!(dataset.schema.classes, vec!["0", "1"]);
        assert_eq!(dataset.class_counts(), vec![32, 32]);
    }

    #[test]
    fn invalid_sizes_are_rejected() {
        assert!(matches!(
            multiplexer_table(0, None, 0),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            multiplexer_table(8, Some(10), 0),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            multiplexer_table(5, None, 0),
            Err(Error::Config(_))
        ));
        let sampled = multiplexer_table(5, Some(10), 0).unwrap();
        assert_eq!(sampled.header.len(), 38);
        assert_eq!(sampled.rows.len(), 10);
        let largest = multiplexer_table(7, Some(2), 0).unwrap();
        assert_eq!(largest.header.len(), 136);
    }
}
