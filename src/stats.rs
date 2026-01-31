#![forbid(unsafe_code)]

pub fn zscore_in_place(points: &mut [Vec<f64>]) {
    if points.is_empty() {
        return;
    }
    let n = points.len();
    let d = points[0].len();

    let mut mean = vec![0.0; d];
    for p in points.iter() {
        for j in 0..d {
            mean[j] += p[j];
        }
    }
    for j in 0..d {
        mean[j] /= n as f64;
    }

    let mut var = vec![0.0; d];
    for p in points.iter() {
        for j in 0..d {
            let t = p[j] - mean[j];
            var[j] += t * t;
        }
    }
    for j in 0..d {
        var[j] /= (n as f64).max(1.0);
    }

    for p in points.iter_mut() {
        for j in 0..d {
            let sd = var[j].sqrt();
            if sd > 0.0 {
                p[j] = (p[j] - mean[j]) / sd;
            } else {
                p[j] = 0.0;
            }
        }
    }
}