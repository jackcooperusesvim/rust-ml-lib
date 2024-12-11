use fastrand;

const DEFAULT_SEED: u64 = 1234567890;
//TODO: Generate seeded semi-random numbers
//use rand_pcg::Pcg64Mcg;

pub fn default_seed() -> u64 {
    DEFAULT_SEED
}

pub fn rand_matrix(row_cols: (usize, usize), in_seed: Option<u64>) -> Vec<Vec<f32>> {
    let (rows, cols) = row_cols;
    let seed: u64 = match in_seed {
        Some(seed) => seed,
        None => DEFAULT_SEED,
    };
    fastrand::seed(seed);

    let mut out: Vec<Vec<f32>> = vec![vec![0.0; cols]; rows];

    for row_num in 0..rows {
        for col_num in 0..cols {
            *(out.get_mut(row_num).unwrap().get_mut(col_num).unwrap()) = fastrand::f32();
        }
    }
    out
}
pub fn rand_vec(rows: usize, in_seed: Option<u64>) -> Vec<f32> {
    let seed: u64 = match in_seed {
        Some(seed) => seed,
        None => DEFAULT_SEED,
    };

    fastrand::seed(seed);

    let mut out: Vec<f32> = vec![0.0; rows];

    for row_num in 0..rows {
        *(out.get_mut(row_num).unwrap()) = fastrand::f32();
    }
    out
}
