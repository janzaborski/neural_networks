use rand::Rng;
use rand::seq::SliceRandom;

use crate::data::Dataset;
use crate::math_utils::Scalar;
use crate::math_utils::matrix::Mat;
use crate::math_utils::matrix::layout::Shape;
use crate::neural_network::NeuralNetwork;
use crate::neural_network::activation::Activation;
use crate::neural_network::head::Head;

pub struct TrainConfig<T> {
    pub epochs: usize,
    pub batch: usize,
    pub lr: T,
    pub log_every: usize,
}

pub fn train<T, H, O>(
    net: &mut NeuralNetwork<T, H, O>,
    data: &Dataset<T>,
    cfg: &TrainConfig<T>,
    rng: &mut impl Rng,
    mut on_log: impl FnMut(usize, &NeuralNetwork<T, H, O>) -> Result<(), String>,
) -> Result<(), String>
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let n = data.len();
    if cfg.batch == 0 || cfg.batch > n {
        return Err(format!("batch size {} must be in 1..={n}", cfg.batch));
    }
    let log_every = cfg.log_every.max(1);

    let n_in = data.x().shape().cols();
    let n_out = data.y().shape().cols();

    let mut ws = net.workspace(cfg.batch);
    let mut grads = net.zero_grads();
    let mut x_b = Mat::<T>::zeros(Shape::new(cfg.batch, n_in));
    let mut y_b = Mat::<T>::zeros(Shape::new(cfg.batch, n_out));
    let mut idx: Vec<usize> = (0..n).collect();

    on_log(0, net)?;

    for epoch in 1..=cfg.epochs {
        idx.shuffle(rng);

        for chunk in idx.chunks_exact(cfg.batch) {
            {
                let mut xb = x_b.to_ref_mut();
                let mut yb = y_b.to_ref_mut();
                for (k, &i) in chunk.iter().enumerate() {
                    xb.row_mut(k).copy_from_slice(data.x().row(i));
                    yb.row_mut(k).copy_from_slice(data.y().row(i));
                }
            }

            net.forward(x_b.to_ref(), &mut ws);
            net.backward(x_b.to_ref(), y_b.to_ref(), &mut ws, &mut grads);
            net.sgd_step(&grads, cfg.lr);
        }

        if epoch % log_every == 0 || epoch == cfg.epochs {
            on_log(epoch, net)?;
        }
    }

    Ok(())
}
