#![forbid(unsafe_code)]

use crate::graph::Graph;
use crate::order::{higher, vertices_desc_by_density};
use crate::uf::UfTomato;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TomatoError {
    #[error("density has non finite value at index {0}")]
    NonFiniteDensity(usize),
    #[error("tau must be >= 0")]
    InvalidTau,
    #[error("density length mismatch")]
    DensityLengthMismatch,
    #[error("invalid graph: {0}")]
    InvalidGraph(String),
}

fn validate_density(density: &[f64]) -> Result<(), TomatoError> {
    for (i, &x) in density.iter().enumerate() {
        if !x.is_finite() {
            return Err(TomatoError::NonFiniteDensity(i));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct TomatoParams {
    pub tau: f64,
}

#[derive(Debug, Clone)]
pub struct TomatoResult {
    pub cluster_of: Vec<usize>,
    pub modes: Vec<usize>,
}

pub fn tomato_cluster(
    graph: &Graph,
    density: &[f64],
    params: TomatoParams,
) -> Result<TomatoResult, TomatoError> {
    validate_density(density)?;
    if density.len() != graph.n() {
        return Err(TomatoError::DensityLengthMismatch);
    }
    if !(params.tau >= 0.0) {
        return Err(TomatoError::InvalidTau);
    }

    let tau = params.tau;
    let n = graph.n();
    let ord = vertices_desc_by_density(density);
    let mut uf = UfTomato::new(n);

    let mut uniq_roots: Vec<usize> = Vec::new();

    for &v in &ord {
        uf.activate(v);

        uniq_roots.clear();
        let rv = uf.find(v);
        uniq_roots.push(rv);

        for &u in graph.neighbors(v) {
            if !uf.is_active(u) {
                continue;
            }
            let ru = uf.find(u);
            if !uniq_roots.iter().any(|&x| x == ru) {
                uniq_roots.push(ru);
            }
        }

        if uniq_roots.len() <= 1 {
            continue;
        }

        let mut winner_root = uniq_roots[0];
        let mut winner_mode = uf.mode_of_root(winner_root);

        for &r in uniq_roots[1..].iter() {
            let m = uf.mode_of_root(r);
            if higher(density, m, winner_mode) {
                winner_root = r;
                winner_mode = m;
            }
        }

        let fv = density[v];

        for &r0 in uniq_roots.clone().iter() {
            if r0 == winner_root {
                continue;
            }

            let r = uf.find(r0);
            let w = uf.find(winner_root);
            if r == w {
                winner_root = w;
                continue;
            }

            if uf.is_protected_root(r) {
                continue;
            }

            let m = uf.mode_of_root(r);
            let lifetime = density[m] - fv;

            if lifetime < tau {
                let w_after = uf.union_survivor(density, w, r);
                winner_root = w_after;
            } else {
                uf.protect_root(r);
            }
        }
    }

    let mut cluster_of = vec![0usize; n];
    for v in 0..n {
        let r = uf.find(v);
        cluster_of[v] = uf.mode_of_root(r);
    }

    let mut modes: Vec<usize> = Vec::new();
    for &m in &cluster_of {
        if !modes.iter().any(|&x| x == m) {
            modes.push(m);
        }
    }

    modes.sort_by(|&a, &b| {
        let fa = density[a];
        let fb = density[b];
        if fa > fb {
            std::cmp::Ordering::Less
        } else if fa < fb {
            std::cmp::Ordering::Greater
        } else {
            a.cmp(&b)
        }
    });

    Ok(TomatoResult { cluster_of, modes })
}