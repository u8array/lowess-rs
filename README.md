# LOcally Weighted Scatterplot Smoothing
This is a Rust port of LOcally Weighted Scatterplot Smoothing

## Usage
```Rust

use rand::Rng;
use std::ops::Range;

use loess::Lowess;

pub fn generate_random_series(length: usize, trend_slope: f64, noise: Range<f64>) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..length)
        .map(|_| {
            let random_noise = rng.gen_range(noise.clone());
            let value = rng.gen_range(0.0..1.0);
            value + trend_slope + random_noise
        })
        .collect()
}

fn main() {
    // Generate data
    let y = generate_random_series(50, 0.02, -0.1..0.1);
    let x: Vec<f64> = (1..(y.len()+1)).map(|x| x as f64).collect();


    // Use the generated random data to feed the LOWESS algorithm:
    let res = Lowess::new(&x, &y, 0.33333, 0, 0.0);

    println!("{:?}", res);
}

```
## Example

![example](https://github.com/u8array/lowess-rs/assets/104523700/e5d34b19-7c5e-48a1-a8f3-4bb86649f261)
