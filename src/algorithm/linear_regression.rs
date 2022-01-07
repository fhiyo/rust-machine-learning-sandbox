use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::Solve;

pub struct LinearRegression {
    coefficients: Array1<f64>,
    phi_x: Array2<f64>,
}

impl LinearRegression {
    pub fn fit<'a>(
        x: ArrayView1<f64>,
        t: ArrayView1<f64>,
        f: Box<dyn Fn(f64) -> Vec<f64>>,
    ) -> Result<LinearRegression, Box<dyn std::error::Error>> {
        let phi = |v| {
            let mut vec = vec![1.0];
            vec.append(&mut f(v));
            vec
        };

        let col = f(f64::NAN).len() + 1;
        let phi_x = Array2::from_shape_vec(
            (x.len(), col),
            x.to_vec().into_iter().flat_map(phi).collect::<Vec<_>>(),
        )
        .unwrap();

        let w = phi_x.t().dot(&phi_x).solve(&phi_x.t().dot(&t)).unwrap();

        Ok(LinearRegression {
            coefficients: w,
            phi_x,
        })
    }

    pub fn predict(&self) -> Array1<f64> {
        self.phi_x.dot(&self.coefficients)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::LinearRegression;

    #[test]
    fn identity() {
        // y = x
        let x = arr1(&[1.0, 2.0]);
        let t = x.clone();
        let model = LinearRegression::fit(x.view(), t.view(), Box::new(|v| vec![v])).unwrap();
        let predict = model.predict();

        helper::assert_vec_eq_approx(t, predict, 1e-6);
    }

    #[test]
    fn linear() {
        // y = 2x + 3
        let x = arr1(&(0..=10).map(|i| i as f64).collect::<Vec<_>>());
        let t = x.clone().map(|f| 2.0 * f + 3.0);
        let model = LinearRegression::fit(x.view(), t.view(), Box::new(|v| vec![v])).unwrap();
        let predict = model.predict();

        helper::assert_vec_eq_approx(t, predict, 1e-6);
    }

    #[test]
    fn quad_ols() {
        // y = 3x^2 - 2x + 10
        let x = arr1(&(-100..=100).map(|i| i as f64).collect::<Vec<_>>());
        let t = x.clone().map(|f| 3.0 * f.powf(2.0) - 2.0 * f + 10.0);
        let model =
            LinearRegression::fit(x.view(), t.view(), Box::new(|v| vec![v, v.powf(2.0)])).unwrap();
        let predict = model.predict();

        helper::assert_vec_eq_approx(t, predict, 1e-6);
    }

    mod helper {
        use ndarray::Array1;

        pub fn assert_vec_eq_approx(v1: Array1<f64>, v2: Array1<f64>, threshold: f64) {
            assert!(v1
                .iter()
                .zip(v2.iter())
                .all(|(&a, &b)| (a - b).abs() < threshold));
        }
    }
}
