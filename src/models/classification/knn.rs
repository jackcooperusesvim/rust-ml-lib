use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone)]
struct Knn<C: Eq + Hash + Clone> {
    data: Option<Vec<(Vec<f32>, C)>>,
    k: usize,
}

impl<C: Clone + Eq + Hash> Knn<C> {
    fn new(k: usize) -> Self {
        if k == 0 {
            Knn { data: None, k: 1 }
        } else {
            Knn { data: None, k }
        }
    }

    fn train(&mut self, data: Vec<(Vec<f32>, C)>) {
        match self.data.take() {
            Some(mut pretrained_data) => {
                pretrained_data.extend(data.into_iter());
                self.data.replace(pretrained_data);
            }
            None => self.data = Some(data),
        };
    }

    fn eval(&self, input: Vec<f32>) -> Option<C> {
        match &self.data {
            Some(training_data) => {
                let mut dist: Vec<(f32, C)> = training_data
                    .iter()
                    .map(|(data, class)| {
                        (
                            data.iter()
                                .zip(input.clone())
                                .map(|(train, inp)| (train - inp) * (train - inp))
                                .sum(),
                            class.clone(),
                        )
                    })
                    .collect::<Vec<(f32, C)>>();

                dist.sort_by(|(dist1, _), (dist2, _)| dist1.total_cmp(&dist2));

                let ordered_classes: Vec<C> = dist.into_iter().map(|(_, class)| class).collect();
                let mut classmap: HashMap<C, isize> = HashMap::new();

                for class in ordered_classes.chunks(self.k).next().unwrap() {
                    classmap
                        .entry(class.clone())
                        .and_modify(|counter| *counter += 1)
                        .or_insert(1);
                }

                match classmap
                    .iter()
                    .max_by(|(_, aval), (_, bval)| aval.cmp(bval))
                {
                    Some(val) => Some(val.0.clone()),
                    None => None,
                }
                //TODO: Check for double winners
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    //TODO: WRITE TESTS
    use super::*;

    #[test]
    fn basic_knn() {
        let mut model = Knn::new(1);
        let training_data: Vec<(Vec<f32>, i32)> =
            vec![(vec![-1.0], 1), (vec![1.0], 1), (vec![2.0], 3)];
        model.train(training_data);
    }
}
