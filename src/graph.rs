#![forbid(unsafe_code)]

use crate::tomato::TomatoError;

#[derive(Debug, Clone)]
pub struct Graph {
    adj: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new(adj: Vec<Vec<usize>>) -> Result<Self, TomatoError> {
        let n = adj.len();
        for (u, nbrs) in adj.iter().enumerate() {
            for &v in nbrs {
                if v >= n {
                    return Err(TomatoError::InvalidGraph(format!(
                        "invalid edge {} -> {} for n={}",
                        u, v, n
                    )));
                }
            }
        }
        Ok(Self { adj })
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.adj.len()
    }

    #[inline]
    pub fn neighbors(&self, v: usize) -> &[usize] {
        &self.adj[v]
    }

    pub fn symmetrize_and_dedup(mut adj: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let n = adj.len();
        for u in 0..n {
            for &v in adj[u].clone().iter() {
                if v < n {
                    adj[v].push(u);
                }
            }
        }
        for u in 0..n {
            adj[u].sort_unstable();
            adj[u].dedup();
            adj[u].retain(|&v| v != u);
        }
        adj
    }
}