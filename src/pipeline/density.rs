#![forbid(unsafe_code)]

use crate::backend::AnnBackend;
use crate::tomato::TomatoError;

#[derive(Debug, Clone)]
pub enum DensitySpec {
    KnnLog {
        k: usize,
        eps: f64,
    },
    KdeGaussianKnn {
        k: usize,
        bandwidth2: f64,
    },
    KdeGaussianFullBrute {
        bandwidth2: f64,
    },
}

pub fn estimate_density<B: AnnBackend>(backend: &B, spec: DensitySpec) -> Result<Vec<f64>, TomatoError> {
    let n = backend.len();
    let d = backend.dim();

    match spec {
        DensitySpec::KnnLog { k, eps } => {
            if k == 0 {
                return Err(TomatoError::InvalidGraph("k must be >= 1".to_string()));
            }
            let eps = eps.max(0.0);
            let knn = backend.knn_all_indices_dist2(k);
            let mut out = vec![0.0; n];
            for i in 0..n {
                let mut max_d2 = 0.0;
                for &(_j, d2) in &knn[i] {
                    if d2 > max_d2 {
                        max_d2 = d2;
                    }
                }
                let r = (max_d2 + eps).sqrt();
                let logr = r.ln();
                out[i] = - (d as f64) * logr;
            }
            Ok(out)
        }
        DensitySpec::KdeGaussianKnn { k, bandwidth2 } => {
            if !(bandwidth2 > 0.0) {
                return Err(TomatoError::InvalidGraph("bandwidth2 must be > 0".to_string()));
            }
            let knn = backend.knn_all_indices_dist2(k);
            let inv = 1.0 / (2.0 * bandwidth2);
            let mut out = vec![0.0; n];
            for i in 0..n {
                let mut s = 0.0;
                for &(_j, d2) in &knn[i] {
                    s += (-d2 * inv).exp();
                }
                out[i] = s;
            }
            Ok(out)
        }
        DensitySpec::KdeGaussianFullBrute { bandwidth2 } => {
            if !(bandwidth2 > 0.0) {
                return Err(TomatoError::InvalidGraph("bandwidth2 must be > 0".to_string()));
            }
            let inv = 1.0 / (2.0 * bandwidth2);
            let mut out = vec![0.0; n];
            for i in 0..n {
                let nbrs = backend.knn_indices_dist2(i, n.saturating_sub(1));
                let mut s = 0.0;
                for &(_j, d2) in &nbrs {
                    s += (-d2 * inv).exp();
                }
                out[i] = s;
            }
            Ok(out)
        }
    }
}