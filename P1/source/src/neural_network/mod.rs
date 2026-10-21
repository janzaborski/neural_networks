use rand::{Rng, RngExt};
use rand_distr::Normal;

use crate::{
    math_utils::{
        Scalar,
        matrix::{Mat, MatRef, layout::Shape},
    },
    neural_network::{activation::Activation, gradient::Gradient, head::Head, layer::Layer},
};

pub mod activation;
pub mod cost;
pub mod gradient;
pub mod head;
pub mod layer;
pub mod params;

pub struct NeuralNetwork<T, H, O> {
    layers: Vec<Layer<T>>,
    hidden: H,
    head: O,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HeadKind, HiddenKind};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn propagated_deltas_match_pre_activation_derivatives() {
        let x = Mat::from_vec(vec![0.2, 0.7], Shape::new(2, 1));
        let y = Mat::from_vec(vec![-2.0, 3.0], Shape::new(2, 1));
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let net = NeuralNetwork::new(
            &[1, 2, 2, 1],
            true,
            HiddenKind::Tanh,
            HeadKind::IdentityMse,
            &mut rng,
        );
        let mut ws = net.workspace(2);
        let mut grads = net.zero_grads();
        net.forward(x.to_ref(), &mut ws);
        net.backward(x.to_ref(), y.to_ref(), &mut ws, &mut grads);

        for layer in 0..net.layers.len() {
            for sample in 0..2 {
                let expected = ws.delta(layer).row(sample).to_vec();
                for (neuron, analytic) in expected.iter().enumerate() {
                    let mut costs = Vec::new();
                    for offset in [1e-6, -1e-6] {
                        net.forward(x.to_ref(), &mut ws);
                        ws.z[layer].to_ref_mut().row_mut(sample)[neuron] += offset;
                        // Recompute from the perturbed pre-activation through to the output.
                        for l in layer..net.layers.len() {
                            if l > layer {
                                net.layers[l].forward(ws.a[l - 1].to_ref(), ws.z[l].to_ref_mut());
                            }
                            if l == net.layers.len() - 1 {
                                net.head.forward(ws.z[l].to_ref(), ws.a[l].to_ref_mut());
                            } else {
                                net.hidden.forward(ws.z[l].to_ref(), ws.a[l].to_ref_mut());
                            }
                        }
                        costs.push(net.cost(&ws, y.to_ref()));
                    }
                    let numerical = (costs[0] - costs[1]) / 2e-6;
                    assert!(
                        (numerical - analytic).abs() < 1e-6,
                        "layer {layer}, sample {sample}, neuron {neuron}: {numerical} != {analytic}"
                    );
                }
            }
        }
    }

    #[test]
    fn all_heads_match_finite_difference_gradients() {
        let x = Mat::from_vec(vec![0.2, -0.4, 0.7, 0.3], Shape::new(2, 2));
        for head in [
            HeadKind::IdentityMse,
            HeadKind::IdentityMae,
            HeadKind::SoftmaxCrossEntropy,
            HeadKind::SoftmaxMse,
        ] {
            let classification = head.task() == crate::data::TaskKind::Classification;
            let outputs = if classification { 2 } else { 1 };
            let y = Mat::from_vec(
                if classification {
                    vec![1.0, 0.0, 0.0, 1.0]
                } else {
                    vec![-3.0, 4.0]
                },
                Shape::new(2, outputs),
            );
            for hidden in [HiddenKind::Sigmoid, HiddenKind::Tanh, HiddenKind::Relu] {
                let mut rng = ChaCha8Rng::seed_from_u64(17);
                let mut net = NeuralNetwork::new(&[2, 3, outputs], true, hidden, head, &mut rng);
                let mut ws = net.workspace(2);
                let mut grads = net.zero_grads();
                net.forward(x.to_ref(), &mut ws);
                net.backward(x.to_ref(), y.to_ref(), &mut ws, &mut grads);
                for (l, grad) in grads.iter().enumerate() {
                    for bias in [false, true] {
                        let expected = if bias {
                            grad.p.b().unwrap()
                        } else {
                            grad.p.w()
                        }
                        .as_contiguous_slice()
                        .unwrap()
                        .to_vec();
                        for (i, analytic) in expected.iter().enumerate() {
                            let mut costs = Vec::new();
                            for offset in [1e-6, -2e-6, 1e-6] {
                                let p = net.layers[l].params_mut();
                                let values = if bias {
                                    p.b.as_mut().unwrap()
                                } else {
                                    &mut p.w
                                };
                                values.as_contiguous_slice_mut().unwrap()[i] += offset;
                                net.forward(x.to_ref(), &mut ws);
                                costs.push(net.cost(&ws, y.to_ref()));
                            }
                            let numerical = (costs[0] - costs[1]) / 2e-6;
                            assert!(
                                (numerical - analytic).abs() < 1e-6,
                                "{head:?} {hidden:?} layer {l} bias={bias}: {numerical} != {analytic}"
                            );
                        }
                    }
                }
            }
        }
    }
}

pub struct Workspace<T> {
    z: Vec<Mat<T>>,
    a: Vec<Mat<T>>,
    delta: Vec<Mat<T>>,
}

impl<T> Workspace<T> {
    /// d(batch loss)/d(z) for this layer; valid after NeuralNetwork::backward.
    pub fn delta(&self, layer: usize) -> MatRef<'_, T> {
        self.delta[layer].to_ref()
    }
}

impl<T: Scalar, H: Activation<T>, O: Head<T>> NeuralNetwork<T, H, O> {
    pub fn new(sizes: &[usize], bias: bool, hidden: H, head: O, rng: &mut impl Rng) -> Self {
        let n_layers = sizes.len() - 1;
        let layers = sizes
            .windows(2)
            .enumerate()
            .map(|(l, w)| {
                let (fan_in, fan_out) = (w[0], w[1]);
                let std = if l == n_layers - 1 {
                    head.init_std(fan_in, fan_out)
                } else {
                    hidden.init_std(fan_in, fan_out)
                };
                let dist = Normal::new(0f64, std).unwrap();
                Layer::new(fan_in, fan_out, bias, (&mut *rng).sample_iter(dist))
            })
            .collect();

        Self {
            layers,
            hidden,
            head,
        }
    }

    pub fn workspace(&self, batch: usize) -> Workspace<T> {
        let alloc = || {
            self.layers
                .iter()
                .map(|l| Mat::<T>::zeros(Shape::new(batch, l.params().fan_out())))
                .collect::<Vec<_>>()
        };

        Workspace {
            z: alloc(),
            a: alloc(),
            delta: alloc(),
        }
    }

    pub fn zero_grads(&self) -> Vec<Gradient<T>> {
        self.layers.iter().map(Layer::zero_grad).collect()
    }

    pub fn forward(&self, x: MatRef<'_, T>, ws: &mut Workspace<T>) {
        let n_layers = self.layers.len();

        for l in 0..n_layers {
            let input = if l == 0 { x } else { ws.a[l - 1].to_ref() };

            self.layers[l].forward(input, ws.z[l].to_ref_mut());

            if l < n_layers - 1 {
                self.hidden.forward(ws.z[l].to_ref(), ws.a[l].to_ref_mut());
            } else {
                self.head.forward(ws.z[l].to_ref(), ws.a[l].to_ref_mut());
            }
        }
    }

    pub fn backward(
        &self,
        x: MatRef<'_, T>,
        y: MatRef<'_, T>,
        ws: &mut Workspace<T>,
        grads: &mut [Gradient<T>],
    ) {
        let n_layers = self.layers.len();

        self.head.delta(
            ws.z[n_layers - 1].to_ref(),
            ws.a[n_layers - 1].to_ref(),
            y,
            ws.delta[n_layers - 1].to_ref_mut(),
        );

        for l in (0..n_layers).rev() {
            if l == 0 {
                self.layers[0].backward(x, ws.delta[0].to_ref(), &mut grads[0], None);
            } else {
                let (lower, upper) = ws.delta.split_at_mut(l);

                self.layers[l].backward(
                    ws.a[l - 1].to_ref(),
                    upper[0].to_ref(),
                    &mut grads[l],
                    Some(lower[l - 1].to_ref_mut()),
                );

                self.hidden.backward(
                    ws.z[l - 1].to_ref(),
                    ws.a[l - 1].to_ref(),
                    lower[l - 1].to_ref_mut(),
                );
            }
        }
    }

    pub fn output<'a>(&self, ws: &'a Workspace<T>) -> MatRef<'a, T> {
        ws.a.last().map(|a| a.to_ref()).unwrap()
    }

    pub fn cost(&self, ws: &Workspace<T>, y: MatRef<'_, T>) -> f64 {
        self.head.cost(self.output(ws), y)
    }

    pub fn sgd_step(&mut self, grads: &[Gradient<T>], lr: T) {
        self.layers
            .iter_mut()
            .zip(grads.iter())
            .for_each(|(l, g)| l.sgd_step(g, lr));
    }

    pub fn layers(&self) -> &[Layer<T>] {
        &self.layers
    }
}
