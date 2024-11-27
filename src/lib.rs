pub mod loss;
pub mod models;

// pub trait GradientEquation {
//     fn eval(&self, in:Vec<f32>) -> Vec<f32>;
//     fn grads(&self, in:Vec<f32>) -> Vec<f32>;
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(4, 4);
    }
}
