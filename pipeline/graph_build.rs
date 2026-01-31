#![forbid(unsafe_code)]

use crate::backend::AnnBackend;
use crate::graph::Graph;
use crate::tomato::TomatoError;

#[derive(Debug, Clone)]
pub enum GraphSpec {
    Knn {
        k: usize,
        symmetrize: bool,
    },
    RipsBrute {
        radius2: f64,
    },
    RipsFromKnnApprox {
        k: usize,
        radius2: f64,
        symmetrize: bool,
    },
}

pub fn build_graph<B: AnnBackend>(backend: &B, spec: GraphSpec) -> Result<Graph, TomatoError> {
    let n = backend.len();
    match spec {
        GraphSpec::Knn { k, symmetrize } => {
            let knn = backend.knn_all_indices_dist2(k);
            let mut adj: Vec<Vec<usize>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut nbrs: Vec<usize> = knn[i].iter().map(|x| x.0).collect();
                nbrs.sort_unstable();
                nbrs.dedup();
                adj.push(nbrs);
            }
            if symmetrize {
                adj = Graph::symmetrize_and_dedup(adj);
            }
            Graph::new(adj)
        }
        GraphSpec::RipsBrute { radius2 } => {
            if !(radius2 >= 0.0) {
                return Err(TomatoError::InvalidGraph("radius2 must be >= 0".to_string()));
            }
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
            for i in 0..n {
                let mut nbrs: Vec<(usize, f64)> = backend.knn_indices_dist2(i, n.saturating_sub(1));
                nbrs.retain(|&(_j, d2)| d2 <= radius2);
                for (j, _) in nbrs {
                    adj[i].push(j);
                }
            }
            adj = Graph::symmetrize_and_dedup(adj);
            Graph::new(adj)
        }
        GraphSpec::RipsFromKnnApprox { k, radius2, symmetrize } => {
            if !(radius2 >= 0.0) {
                return Err(TomatoError::InvalidGraph("radius2 must be >= 0".to_string()));
            }
            let knn = backend.knn_all_indices_dist2(k);
            let mut adj: Vec<Vec<usize>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut nbrs: Vec<usize> = Vec::new();
                for &(j, d2) in &knn[i] {
                    if d2 <= radius2 {
                        nbrs.push(j);
                    }
                }
                nbrs.sort_unstable();
                nbrs.dedup();
                adj.push(nbrs);
            }
            if symmetrize {
                adj = Graph::symmetrize_and_dedup(adj);
            }
            Graph::new(adj)
        }
    }
}