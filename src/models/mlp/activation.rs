use std::f32::consts;

pub enum Activations {
    Sigmoid(Sigmoid),
}
struct Sigmoid {}
struct Relu {}

fn sig_raw(x: f32) -> f32 {
    1.0 / (1.0 + consts::E.powf(x as f32))
}

pub trait ActFunc {
    fn eval(&self, input_vec: Vec<f32>) -> Vec<f32>;
    fn deriv(&self, input_vec: Vec<f32>) -> Vec<f32>;
}

impl ActFunc for Sigmoid {
    fn eval(&self, input_vec: Vec<f32>) -> Vec<f32> {
        input_vec.into_iter().map(|x| sig_raw(x)).collect()
    }

    fn deriv(&self, input_vec: Vec<f32>) -> Vec<f32> {
        input_vec
            .into_iter()
            .map(|x| sig_raw(x) * (1.0 - sig_raw(x)))
            .collect()
    }
}
