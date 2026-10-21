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
    let remainder = n % cfg.batch;
    let mut tail_ws = (remainder > 0).then(|| net.workspace(remainder));
    let mut grads = net.zero_grads();
    let mut x_b = Mat::<T>::zeros(Shape::new(cfg.batch, n_in));
    let mut y_b = Mat::<T>::zeros(Shape::new(cfg.batch, n_out));
    let mut idx: Vec<usize> = (0..n).collect();

    on_log(0, net)?;

    for epoch in 1..=cfg.epochs {
        idx.shuffle(rng);

        for chunk in idx.chunks(cfg.batch) {
            {
                let mut xb = x_b.to_ref_mut();
                let mut yb = y_b.to_ref_mut();
                for (k, &i) in chunk.iter().enumerate() {
                    xb.row_mut(k).copy_from_slice(data.x().row(i));
                    yb.row_mut(k).copy_from_slice(data.y().row(i));
                }
            }

            let active_ws = if chunk.len() == cfg.batch {
                &mut ws
            } else {
                tail_ws.as_mut().unwrap()
            };
            let xb = x_b.to_ref().subrows(0..chunk.len());
            let yb = y_b.to_ref().subrows(0..chunk.len());
            net.forward(xb, active_ws);
            net.backward(xb, yb, active_ws, &mut grads);
            net.sgd_step(&grads, cfg.lr);
        }

        if epoch % log_every == 0 || epoch == cfg.epochs {
            on_log(epoch, net)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HeadKind, HiddenKind};
    use crate::data::{DataSpec, Source};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn partial_batch_matches_manual_sgd() {
        let path = std::env::temp_dir().join("nn_partial_batch.csv");
        std::fs::write(&path, "x,y\n1,2\n2,4\n3,6\n4,8\n5,10\n").unwrap();
        let spec: DataSpec = serde_json::from_value(serde_json::json!({
            "task": "regression", "train": path, "validation": 0.2
        }))
        .unwrap();
        let data = Dataset::<f64>::load(&spec, &Source::File(path.clone()), None).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut actual = NeuralNetwork::new(
            &[1, 1],
            true,
            HiddenKind::Tanh,
            HeadKind::IdentityMse,
            &mut rng,
        );
        let mut manual_rng = ChaCha8Rng::seed_from_u64(42);
        let mut expected = NeuralNetwork::new(
            &[1, 1],
            true,
            HiddenKind::Tanh,
            HeadKind::IdentityMse,
            &mut manual_rng,
        );
        let cfg = TrainConfig {
            epochs: 1,
            batch: 2,
            lr: 0.01,
            log_every: 1,
        };
        train(&mut actual, &data, &cfg, &mut rng, |_, _| Ok(())).unwrap();
        let mut indices: Vec<usize> = (0..5).collect();
        indices.shuffle(&mut manual_rng);
        for chunk in indices.chunks(2) {
            let batch = data.select(chunk);
            let mut ws = expected.workspace(chunk.len());
            let mut grads = expected.zero_grads();
            expected.forward(batch.x(), &mut ws);
            expected.backward(batch.x(), batch.y(), &mut ws, &mut grads);
            expected.sgd_step(&grads, cfg.lr);
        }
        let a = actual.layers()[0].params();
        let b = expected.layers()[0].params();
        assert_eq!(a.w().row(0), b.w().row(0));
        assert_eq!(a.b().unwrap().row(0), b.b().unwrap().row(0));
        std::fs::remove_file(path).unwrap();
    }
}
