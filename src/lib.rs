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
    let Loess {
        x,
        y,
        f,
        nsteps,
        delta,
    } = params;
    let n = x.len();
    if n < 2 {
        return y.to_vec();
    }

    // Determine the number of points used for each local regression
    let ns = (f * n as f64 + 1e-7).max(2.0).min(n as f64) as usize;

    let y_original = y.to_vec();
    let mut y = y_original.clone();
    let mut rw = vec![0.0; n];
    let mut res = vec![0.0; n];

    let mut last = 0;

    for iter in 1..=nsteps + 1 {
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
            match lowest(
                x,
                &y,
                x[i - 1],
                nleft - 1,
                nright - 1,
                &mut res,
                iter > 1,
                &rw,
            ) {
                Some(ys) => {
                    if last < i - 1 {
                        interpolate_segment(x, &mut y, &res, last - 1, i - 1);
                    }

                    y[i - 1] = ys;
                    last = i;
                }
                None => break,
            }

            let cut = x[last - 1] + delta;

            for i in 1..=n {
                if x[i - 1] > cut {
                    break;
                }
                if x[i - 1] == x[last - 1] {
                    y[i - 1] = y[last - 1];
                    last = i;
                }
            }

            i = last + 1;

            if last >= n {
                break;
            }
        }

        // Compute residuals for convergence check
        for i in 0..n {
            res[i] = y_original[i] - y[i];
        }

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
    }

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

fn update_residual_weights(rw: &mut Vec<f64>, res: &[f64], n: usize) {
    for i in 0..n {
        rw[i] = res[i].abs();
    }
    rw.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            delta,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowess() {
        let y = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
            45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
        ];
        let x: Vec<f64> = (1..(y.len() + 1)).map(|x| x as f64).collect();

        let y_assert = [
            5.5037337686781305,
            6.114970675649464,
            6.658262751576383,
            7.1403695282317345,
            7.577255689830473,
            8.003882873231403,
            8.487312705799924,
            9.104067724255362,
            9.883408822063812,
            10.692715467291649,
            11.536654173833748,
            12.41420852412808,
            13.320802917860432,
            14.249910217842503,
            15.195352387869017,
            16.152895316931946,
            17.120064517689144,
            18.0948463179265,
            19.075531777772667,
            20.060753724127814,
            21.049471607765014,
            22.040899678791135,
            23.03443254949329,
            24.02959665501051,
            25.026023028668774,
            26.023426261087803,
            27.02158600478072,
            28.020331793653238,
            29.019531502277275,
            30.019082719566082,
            31.01890613178673,
            32.018940337133635,
            33.01913779273324,
            34.019461677573574,
            35.01988347315761,
            36.02038109324451,
            37.02093743410303,
            38.02153925039922,
            39.0221762844904,
            40.022840592418355,
            41.02352602208565,
            42.024227809117434,
            43.024942263837204,
            43.890130077204894,
            44.438386008569374,
            44.72861697990426,
            44.83840555704233,
            44.799400990727925,
            44.61138455825291,
            44.25380138245755,
        ];

        let f = 0.33333;
        let nsteps = 0;
        let delta = 0.0;
        let loess = Loess::new(&x, &y, f, nsteps, delta);
        println!("{:?}", loess);
        assert_eq!(loess, y_assert);
    }
}
