#![forbid(unsafe_code)]

use crate::backend::AnnBackend;

#[derive(Debug, Clone)]
pub struct BruteBackend {
    points: Vec<Vec<f64>>,
    dim: usize,
}

impl BruteBackend {
    pub fn new(points: Vec<Vec<f64>>) -> Result<Self, String> {
        if points.is_empty() {
            return Ok(Self { points, dim: 0 });
        }
        let dim = points[0].len();
        for (i, p) in points.iter().enumerate() {
            if p.len() != dim {
                return Err(format!("dimension mismatch at row {}", i));
            }
            for &x in p {
                if !x.is_finite() {
                    return Err("non finite point value".to_string());
                }
            }
        }
        Ok(Self { points, dim })
    }

    #[inline]
    fn dist2(&self, a: usize, b: usize) -> f64 {
        let pa = &self.points[a];
        let pb = &self.points[b];
        let mut s = 0.0;
        for j in 0..self.dim {
            let d = pa[j] - pb[j];
            s += d * d;
        }
        s
    }
}

impl AnnBackend for BruteBackend {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn knn_indices_dist2(&self, query_index: usize, k: usize) -> Vec<(usize, f64)> {
        let n = self.len();
        let kk = k.min(n.saturating_sub(1));
        let mut buf: Vec<(usize, f64)> = Vec::with_capacity(n.saturating_sub(1));
        for j in 0..n {
            if j == query_index {
                continue;
            }
            buf.push((j, self.dist2(query_index, j)));
        }
        buf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0)));
        buf.truncate(kk);
        buf
    }
}