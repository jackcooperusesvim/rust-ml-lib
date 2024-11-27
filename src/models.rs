pub mod knn;
pub mod classification;

pub trait ClassificationModel {
    fn eval(&self, in:Vec<f32>) -> Vec<f32>;
    fn grads(&self, in:Vec<f32>) -> Vec<f32>;
}

