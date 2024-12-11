use std::fmt::Debug;
pub fn slice_dot<const len: usize>(vec1: [f32; len], vec2: [f32; len]) -> f32 {
    vec1.into_iter()
        .zip(vec2)
        .map(|(val1, val2)| val1 * val2)
        .sum()
}

pub fn vec_dot(vec1: Vec<f32>, vec2: Vec<f32>) -> f32 {
    vec1.into_iter()
        .zip(vec2)
        .map(|(val1, val2)| val1 * val2)
        .sum()
}

pub fn outer_product(hor: &[f32], ver: &[f32]) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = vec![vec![0.0; ver.len()]; hor.len()];
    for i in 0..hor.len() {
        for j in 0..ver.len() {
            out[j][i] = ver[j] * hor[i];
        }
    }
    out
}

pub fn mult_across(vec1: Vec<f32>, vec2: Vec<f32>) -> Vec<f32> {
    vec1.into_iter()
        .zip(vec2)
        .map(|(val1, val2)| val1 * val2)
        .collect()
}

pub fn add_across(vec1: Vec<f32>, vec2: Vec<f32>) -> Vec<f32> {
    vec1.into_iter()
        .zip(vec2)
        .map(|(val1, val2)| val1 + val2)
        .collect()
}

pub fn constrain<T, const L: usize>(vec: Vec<T>) -> [T; L]
where
    T: Debug,
{
    vec.try_into().unwrap()
}

pub fn constrain2d<T, const R: usize, const C: usize>(vec: Vec<Vec<T>>) -> [[T; C]; R]
where
    T: Debug,
{
    vec.into_iter()
        .map(|subvec| subvec.try_into().unwrap())
        .collect::<Vec<[T; C]>>()
        .try_into()
        .unwrap()
}
