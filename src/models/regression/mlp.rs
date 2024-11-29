extern crate nalgebra as na;
use na::{SVector,SMatrix};
pub mod activation;
pub mod optimization;
use optimization::Optimizer;
// use activation::Activation;

struct Dense<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    weights: [[f32; INPUT_SIZE]; OUTPUT_SIZE],
    //activation: Box<dyn Activation>
}

trait SingleDimLayer<const INPUT_SIZE:usize, const OUTPUT_SIZE: usize> {
    fn evaluate(input: SVector<f32,INPUT_SIZE>) -> SVector<f32,OUTPUT_SIZE>;
    fn partials(input: SVector<f32,INPUT_SIZE>) -> SVector<f32,OUTPUT_SIZE>;
    fn apply_gradients(grads: SMatrix<f32,INPUT_SIZE,OUTPUT_SIZE>, opt_alg: & dyn Optimizer) -> SVector<f32,OUTPUT_SIZE>;
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Dense<INPUT_SIZE,OUTPUT_SIZE> {
    fn new(input_size: usize, output_size: usize, rand_seed: Option<f32>) -> Self {
    };
    fn evaluate(input_size: usize, output_size: usize, activation) -> Self;
}
