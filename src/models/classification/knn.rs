#[derive(Clone)]
struct Knn<C: Clone> {
    data: Vec<(Vec<f32>, C)>,
    k: int,
}

impl<C: Clone> Knn<C> {
    fn train(data: Vec<(Vec<f32>, C)>) -> Self {
        Self { data }
    }
    fn eval(&self, input: Vec<f32>) {
        self.data.clone().into_iter().map(|data, class| )
    }
}

