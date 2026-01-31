#![forbid(unsafe_code)]

use crate::backend::AnnBackend;
use hnsw_rs::prelude::DistL2;
use hnsw_rs::hnsw::Hnsw;

#[derive(Debug, Clone)]
pub struct HnswParams {
    pub max_nb_connection: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_layer: usize,
    pub use_parallel_insert: bool,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_nb_connection: 24,
            ef_construction: 200,
            ef_search: 64,
            max_layer: 16,
            use_parallel_insert: true,
        }
    }
}

pub struct HnswBackend {
    points: Vec<Vec<f32>>,
    dim: usize,
    hnsw: Hnsw<'static, f32, DistL2>,
    ef_search: usize,
}

impl HnswBackend {
    pub fn new(points_f64: Vec<Vec<f64>>, params: HnswParams) -> Result<Self, String> {
        if points_f64.is_empty() {
            let hnsw = Hnsw::<f32, DistL2>::new(
                params.max_nb_connection,
                0,
                params.max_layer,
                params.ef_construction,
                DistL2 {},
            );
            return Ok(Self {
                points: vec![],
                dim: 0,
                hnsw,
                ef_search: params.ef_search,
            });
        }

        let dim = points_f64[0].len();
        for (i, p) in points_f64.iter().enumerate() {
            if p.len() != dim {
                return Err(format!("dimension mismatch at row {}", i));
            }
            for &x in p {
                if !x.is_finite() {
                    return Err("non finite point value".to_string());
                }
            }
        }

        let mut points: Vec<Vec<f32>> = Vec::with_capacity(points_f64.len());
        for p in points_f64 {
            let mut q = Vec::with_capacity(dim);
            for x in p {
                q.push(x as f32);
            }
            points.push(q);
        }

        let n = points.len();
        let mut hnsw = Hnsw::<f32, DistL2>::new(
            params.max_nb_connection,
            n,
            params.max_layer,
            params.ef_construction,
            DistL2 {},
        );

        if params.use_parallel_insert && n >= 2000 {
            let mut datas: Vec<(&[f32], usize)> = Vec::with_capacity(n);
            for i in 0..n {
                datas.push((&points[i], i));
            }
            hnsw.parallel_insert_slice(&datas);
            hnsw.set_searching_mode(true);
        } else {
            for i in 0..n {
                hnsw.insert_slice((&points[i], i));
            }
        }

        Ok(Self {
            points,
            dim,
            hnsw,
            ef_search: params.ef_search,
        })
    }
}

impl AnnBackend for HnswBackend {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn knn_indices_dist2(&self, query_index: usize, k: usize) -> Vec<(usize, f64)> {
        let n = self.len();
        if n == 0 {
            return vec![];
        }
        let kk = k.min(n.saturating_sub(1));

        let query = &self.points[query_index];

        let ef = self.ef_search.max(kk + 1);
        let mut ans = self.hnsw.search(query, kk + 1, ef);

        ans.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        let mut out: Vec<(usize, f64)> = Vec::with_capacity(kk);
        for nb in ans {
            let idx = nb.d_id;
            if idx == query_index {
                continue;
            }
            out.push((idx, (nb.distance as f64) * (nb.distance as f64)));
            if out.len() == kk {
                break;
            }
        }

        out
    }
}