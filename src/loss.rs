use super::*;
pub trait RegressionLossFunction {
    fn eval(&self, out: Vec<f32>, exp_out: Vec<f32>) -> f32;
    fn grads(&self, out: Vec<f32>, exp_out: Vec<f32>) -> Vec<f32>;
}

struct MeanSquaredError {}

impl RegressionLossFunction for MeanSquaredError {
    fn eval(&self, out: Vec<f32>, exp_out: Vec<f32>) -> f32 {
        out.into_iter()
            .zip(exp_out)
            //This is actually faster than exponentiation, because exponents with the exponentiation funtion can't be known at
            //compile time
            .map(|(iout, iexp_out)| (iexp_out - iout) * (iexp_out - iout))
            .sum()
    }
    fn grads(&self, out: Vec<f32>, exp_out: Vec<f32>) -> Vec<f32> {
        out.into_iter()
            .zip(exp_out)
            .map(|(iout, iexp_out)| iexp_out - iout)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_eval() {
        const MSE: MeanSquaredError = MeanSquaredError {};
        let should_be_0: f32 = MSE.eval(vec![1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0, 1.0]);
        let should_be_4 = MSE.eval(vec![2.0, 0.0, 2.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);
        let should_be_5 = MSE.eval(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(should_be_0, 0.0);
        assert_eq!(should_be_4, 4.0);
        assert_eq!(should_be_5, 5.0);
    }
    #[test]
    fn mse_grads() {
        const MSE: MeanSquaredError = MeanSquaredError {};
        let case1: Vec<f32> = MSE.grads(vec![1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0, 1.0]);
        let case2: Vec<f32> = MSE.grads(vec![2.0, 0.0, 2.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);
        let case3: Vec<f32> =
            MSE.grads(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(case1, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(case2, vec![-1.0, 1.0, -1.0, 1.0]);
        assert_eq!(case3, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }
}
