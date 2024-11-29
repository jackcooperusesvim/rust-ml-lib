pub trait Optimizer {
    fn partial_to_gradient(&self, input: Vec<f32>) -> Vec<f32>;
}
