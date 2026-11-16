use crate::classifier::Classifier;
use crate::data::Value;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Population {
    classifiers: Vec<Classifier>,
}

impl Population {
    pub fn classifiers(&self) -> &[Classifier] {
        &self.classifiers
    }

    pub fn len(&self) -> usize {
        self.classifiers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.classifiers.is_empty()
    }

    pub fn micro_size(&self) -> u64 {
        self.classifiers
            .iter()
            .map(|classifier| u64::from(classifier.numerosity))
            .sum()
    }

    pub fn match_set(&self, row: &[Value]) -> Vec<usize> {
        self.classifiers
            .iter()
            .enumerate()
            .filter(|(_, classifier)| classifier.condition.matches(row))
            .map(|(index, _)| index)
            .collect()
    }

    pub(crate) fn get_mut(&mut self, index: usize) -> &mut Classifier {
        &mut self.classifiers[index]
    }

    pub(crate) fn find_same_rule(&self, classifier: &Classifier) -> Option<usize> {
        self.classifiers
            .iter()
            .position(|candidate| candidate.numerosity > 0 && candidate.same_rule(classifier))
    }

    pub(crate) fn insert(&mut self, classifier: Classifier) -> usize {
        match self.find_same_rule(&classifier) {
            Some(index) => {
                self.classifiers[index].numerosity += classifier.numerosity;
                index
            }
            None => {
                self.classifiers.push(classifier);
                self.classifiers.len() - 1
            }
        }
    }

    pub(crate) fn push(&mut self, classifier: Classifier) {
        self.classifiers.push(classifier);
    }

    pub(crate) fn remove_empty(&mut self) {
        self.classifiers.retain(|classifier| classifier.numerosity > 0);
    }
}
