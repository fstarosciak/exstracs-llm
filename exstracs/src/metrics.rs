use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub n: usize,
    pub accuracy: f64,
    pub balanced_accuracy: f64,
    pub macro_f1: f64,
    pub coverage: f64,
}

impl Metrics {
    pub fn compute(
        truth: &[u32],
        predicted: &[u32],
        covered: &[bool],
        n_classes: usize,
    ) -> Metrics {
        assert_eq!(truth.len(), predicted.len());
        assert_eq!(truth.len(), covered.len());
        assert!(truth
            .iter()
            .chain(predicted)
            .all(|&label| (label as usize) < n_classes));
        let n = truth.len();
        if n == 0 {
            return Metrics::default();
        }
        let mut true_positives = vec![0usize; n_classes];
        let mut support = vec![0usize; n_classes];
        let mut predicted_count = vec![0usize; n_classes];
        let mut correct = 0usize;
        for (&actual, &guess) in truth.iter().zip(predicted) {
            support[actual as usize] += 1;
            predicted_count[guess as usize] += 1;
            if actual == guess {
                true_positives[actual as usize] += 1;
                correct += 1;
            }
        }
        let covered_count = covered.iter().filter(|&&flag| flag).count();
        let mut recall_sum = 0.0;
        let mut recall_classes = 0usize;
        let mut f1_sum = 0.0;
        let mut f1_classes = 0usize;
        for ((&hits, &actual), &guessed) in
            true_positives.iter().zip(&support).zip(&predicted_count)
        {
            let precision = ratio(hits, guessed);
            let recall = ratio(hits, actual);
            if actual > 0 {
                recall_sum += recall;
                recall_classes += 1;
            }
            if actual > 0 || guessed > 0 {
                f1_sum += if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };
                f1_classes += 1;
            }
        }
        Metrics {
            n,
            accuracy: ratio(correct, n),
            balanced_accuracy: mean(recall_sum, recall_classes),
            macro_f1: mean(f1_sum, f1_classes),
            coverage: ratio(covered_count, n),
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(sum: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn perfect_predictions_score_one() {
        let labels = [0, 1, 2, 1, 0];
        let metrics = Metrics::compute(&labels, &labels, &[true; 5], 3);
        assert_eq!(metrics.n, 5);
        close(metrics.accuracy, 1.0);
        close(metrics.balanced_accuracy, 1.0);
        close(metrics.macro_f1, 1.0);
        close(metrics.coverage, 1.0);
    }

    #[test]
    fn all_wrong_predictions_score_zero() {
        let metrics = Metrics::compute(
            &[0, 0, 1, 1],
            &[1, 1, 0, 0],
            &[true, false, false, false],
            2,
        );
        assert_eq!(metrics.n, 4);
        close(metrics.accuracy, 0.0);
        close(metrics.balanced_accuracy, 0.0);
        close(metrics.macro_f1, 0.0);
        close(metrics.coverage, 0.25);
    }

    #[test]
    fn imbalanced_example_matches_sklearn() {
        let truth = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2];
        let predicted = [0, 0, 0, 0, 0, 1, 1, 0, 0, 3];
        let covered = [true, true, true, true, true, true, true, true, false, false];
        let metrics = Metrics::compute(&truth, &predicted, &covered, 5);
        assert_eq!(metrics.n, 10);
        close(metrics.accuracy, 0.6);
        close(metrics.balanced_accuracy, 4.0 / 9.0);
        close(metrics.macro_f1, 33.0 / 104.0);
        close(metrics.coverage, 0.8);
    }

    #[test]
    fn empty_input_gives_default() {
        assert_eq!(Metrics::compute(&[], &[], &[], 3), Metrics::default());
    }

    #[test]
    #[should_panic]
    fn mismatched_lengths_panic() {
        Metrics::compute(&[0, 1], &[0], &[true, true], 2);
    }

    #[test]
    #[should_panic]
    fn out_of_range_label_panics() {
        Metrics::compute(&[0, 2], &[0, 1], &[true, true], 2);
    }
}
