use fhiyo_machine_learning_sandbox::algorithm::LinearRegression;
use ndarray::Array1;

fn main() {
    run(Box::new(|v| vec![v]), "linear", "Linear");
    run(
        Box::new(|v| vec![v, v.powf(2.0), v.powf(3.0)]),
        "poly",
        "Polynomial",
    );

    let gaussians = |v| {
        (-2..=2)
            .map(|f| gaussian(v, f as f64, 5.0))
            .collect::<Vec<_>>()
    };
    run(Box::new(gaussians), "gaussian", "Gaussian");
}

fn run(f: Box<dyn Fn(f64) -> Vec<f64>>, image_name: &str, title: &str) {
    let x = Array1::from((-200..=200).map(|x| x as f64 / 50.0).collect::<Vec<_>>());
    let t = &x.map(|x| x.sin());

    let model = LinearRegression::fit(x.view(), t.view(), f).unwrap();
    let y = model.predict();

    plot(x.to_vec(), y.to_vec(), t.to_vec(), image_name, title).unwrap();
}

fn plot(
    xs: Vec<f64>,
    ys: Vec<f64>,
    ts: Vec<f64>,
    image_name: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    let (x_min, x_max) = plotting_ends(xs.iter());
    let (y_min, y_max) = plotting_ends(ys.iter().chain(ts.iter()));

    let image_path = format!("images/linear_regression/{}.png", image_name);
    let root = BitMapBackend::new(image_path.as_str(), (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            xs.clone().into_iter().zip(ts.into_iter()),
            &GREEN,
        ))?
        .label("y = sin(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .draw_series(LineSeries::new(xs.into_iter().zip(ys.into_iter()), &RED))?
        .label("y = f(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn plotting_ends<'a, I: Iterator<Item = &'a f64>>(vs: I) -> (f64, f64) {
    let (min, max) = vs.fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), b| {
        (min.min(*b), max.max(*b))
    });
    (min * 1.2, max * 1.2)
}

pub fn gaussian(x: f64, mean: f64, var: f64) -> f64 {
    f64::exp(-(x - mean).powf(2.0) / (2.0 * var))
}
