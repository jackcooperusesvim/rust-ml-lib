pub mod activation;
pub mod optimization;
pub mod random;
pub mod vec_math;

use random::*;
use vec_math::*;

use activation::ActFunc;
use optimization::Optimizer;

pub struct Dense<'a, const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    activation: &'a mut dyn ActFunc,
}

struct DenseGradient {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

enum SingleDimGrad {
    Dense(DenseGradient),
}

trait SingleDimLayer<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    fn evaluate(&self, input: Box<[f32; INPUT_SIZE]>) -> Box<[f32; OUTPUT_SIZE]>;
    fn partials(&self, input: Box<[f32; INPUT_SIZE]>) -> SingleDimGrad;
    fn apply_gradients(&mut self, grads: SingleDimGrad, opt_alg: &mut dyn Optimizer);
}

//TODO: FIX RAND IN THIS

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Dense<'_, INPUT_SIZE, OUTPUT_SIZE> {
    fn new<'a>(
        activation: &'a mut dyn ActFunc,
        rand_seed: Option<u64>,
    ) -> Dense<'a, INPUT_SIZE, OUTPUT_SIZE> {
        Dense::<'a, INPUT_SIZE, OUTPUT_SIZE> {
            weights: rand_matrix((INPUT_SIZE, OUTPUT_SIZE), rand_seed),
            biases: rand_vec(OUTPUT_SIZE, rand_seed),
            activation,
        }
    }
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> SingleDimLayer<INPUT_SIZE, OUTPUT_SIZE>
    for Dense<'_, INPUT_SIZE, OUTPUT_SIZE>
{
    fn evaluate(&self, input: Box<[f32; INPUT_SIZE]>) -> Box<[f32; OUTPUT_SIZE]> {
        Box::new(constrain::<f32, OUTPUT_SIZE>(
            self.activation.eval(
                self.weights
                    .iter()
                    .zip(self.biases.clone())
                    .map(|(weight_vec, bias)| vec_dot(weight_vec.clone(), input.to_vec()) + bias)
                    .collect::<Vec<f32>>(),
            ),
        ))
    }

    fn partials(&self, input: Box<[f32; INPUT_SIZE]>) -> SingleDimGrad {
        let bias_grad: Vec<f32> = self.activation.deriv(
            self.weights
                .iter()
                .zip(self.biases.clone())
                .map(|(weight_vec, bias)| vec_dot(weight_vec.clone(), input.to_vec()) + bias)
                .collect::<Vec<f32>>(),
        );

        SingleDimGrad::Dense(DenseGradient {
            weights: outer_product(input.as_slice(), bias_grad.as_slice()),
            biases: bias_grad,
        })
    }

    //TODO: WRITE THIS
    fn apply_gradients(&mut self, grads: SingleDimGrad, opt_alg: &mut dyn Optimizer) {
        match grads {
            SingleDimGrad::Dense(grad_dta) => {}
            _ => panic!("improper gradient type applied to Dense"),
        }
    }
}
