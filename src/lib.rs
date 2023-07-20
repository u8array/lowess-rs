fn lowest(
    x: &[f64],
    y: &[f64],
    xs: f64,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
    userw: bool,
    rw: &[f64],
) -> Option<f64> {
    // Calculate the bandwidth parameters
    let range = (nleft..=nright).into_iter();
    let h = f64::max(xs - x[nleft], x[nright] - xs);
    let (h9, h1) = compute_scaling_factors(h);

    let a: f64 = range.clone().fold(0.0, |acc, j| {
        let r = (x[j] - xs).abs();
        if r <= h9 {
            let wj = if r <= h1 {
                1.0
            } else {
                (1.0 - (r / h).powi(3)).powi(3)
            };
            let wj = if userw { wj * rw[j] } else { wj };
            w[j] = wj;
            acc + wj
        } else {
            acc
        }
    });

    if a <= 0.0 {
        return None;
    }

    // Normalize the weights
    w.iter_mut().for_each(|j| *j /= a);

    let ys = range.map(|j| w[j] * y[j]).sum();

    Some(ys)
}

fn lowess(params: &Loess) -> Vec<f64> {
    let Loess { x, y, f, nsteps, delta } = params;
    let n = x.len();
    if n < 2 {
        return y.to_vec();
    }

    // Determine the number of points used for each local regression
    let ns = (f * n as f64 + 1e-7).max(2.0).min(n as f64) as usize;

    let mut y = y.to_vec();
    let mut rw = vec![0.0; n];
    let mut res = vec![0.0; n];

    let mut last = 0;
    let mut iter = 1;

    while iter <= nsteps + 1 {
        let mut nleft = 1;
        let mut nright = ns;
        let mut i = 1;

        // Perform local regression for each segment
        loop {
            if nright < n {
                let d1 = x[i - 1] - x[nleft - 1];
                let d2 = x[nright] - x[i - 1];

                if d1 > d2 {
                    nleft += 1;
                    nright += 1;
                    continue;
                }
            }

            // Compute the local weighted regression
            if let Some(ys) = lowest(
                x,
                &y,
                x[i - 1],
                nleft - 1,
                nright - 1,
                &mut res,
                iter > 1,
                &rw,
            ) {
                if last < i - 1 {
                    // Perform linear interpolation within the segment
                    interpolate_segment(x, &mut y, &res, last - 1, i - 1);
                }

                y[i - 1] = ys;
                last = i;
            }

            let cut = x[last - 1] + delta;
            i = last;

            while i <= n {
                if x[i - 1] > cut {
                    break;
                }
                if x[i - 1] == x[last - 1] {
                    y[i - 1] = y[last - 1];
                    last = i;
                }
                i += 1;
            }

            i = last + 1;

            if last >= n {
                break;
            }
        }

        // Compute residuals for convergence check
        compute_residuals(&y, &mut res);

        let sc = res.iter().map(|x| x.abs()).sum::<f64>() / n as f64;

        if iter > *nsteps {
            break;
        }

        update_residual_weights(&mut rw, &res, n);

        let cmad = compute_cmad(&rw, n);

        if cmad < 1e-7 * sc {
            break;
        }

        update_residual_weights_with_cmad(&mut rw, &res, cmad);

        iter += 1;
    };

    y
}

fn interpolate_segment(x: &[f64], y: &mut Vec<f64>, res: &[f64], last: usize, i: usize) {
    let denom = x[i] - x[last];
    for j in last..i {
        let alpha = (x[j - 1] - x[last]) / denom;
        let interpolated = alpha * res[i] + (1.0 - alpha) * res[last];
        if interpolated.is_finite() {
            y[j - 1] = interpolated;
        } else {
            y[j - 1] = y[i - 2];
            break;
        }
    }
}

fn compute_residuals(y: &[f64], res: &mut Vec<f64>) {
    for i in 0..y.len() {
        res[i] = y[i] - y[i];
    }
}

fn update_residual_weights(rw: &mut Vec<f64>, res: &[f64], n: usize) {
    for i in 0..n {
        rw[i] = res[i].abs();
    }
    rw.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
}

fn compute_cmad(rw: &[f64], n: usize) -> f64 {
    let m1 = n / 2;
    if n % 2 == 0 {
        let m2 = n - m1 - 1;
        3.0 * (rw[m1] + rw[m2])
    } else {
        6.0 * rw[m1]
    }
}

fn update_residual_weights_with_cmad(rw: &mut Vec<f64>, res: &[f64], cmad: f64) {
    let (c9, c1) = compute_scaling_factors(cmad);
    rw.iter_mut().enumerate().for_each(|(i, rw)| {
        let r = res[i].abs();
        *rw = if r <= c1 {
            1.0
        } else if r <= c9 {
            (1.0 - (r / cmad).powi(2)).powi(2)
        } else {
            0.0
        };
    });
}

fn compute_scaling_factors(val: f64) -> (f64, f64) {
    let x9 = 0.999 * val;
    let x1 = 0.001 * val;
    (x9, x1)
}

impl Loess {
    pub fn new(x: &[f64], y: &[f64], f: f64, nsteps: usize, delta: f64) -> Vec<f64> {
        let params = Self {
            x: x.to_vec(),
            y: y.to_vec(),
            f,
            nsteps,
            delta
        };
        lowess(&params)
    }
}

pub struct Loess {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub f: f64,
    pub nsteps: usize,
    pub delta: f64,
}
