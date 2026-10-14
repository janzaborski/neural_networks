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

pub struct Workspace<T> {
    z: Vec<Mat<T>>,
    a: Vec<Mat<T>>,
    delta: Vec<Mat<T>>,
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
