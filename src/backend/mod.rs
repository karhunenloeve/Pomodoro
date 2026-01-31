#![forbid(unsafe_code)]

pub mod brute;
pub mod hnsw;

pub use brute::BruteBackend;
pub use hnsw::{HnswBackend, HnswParams};

pub trait AnnBackend {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;

    fn knn_indices_dist2(&self, query_index: usize, k: usize) -> Vec<(usize, f64)>;

    fn knn_all_indices_dist2(&self, k: usize) -> Vec<Vec<(usize, f64)>> {
        let n = self.len();
        let mut all = Vec::with_capacity(n);
        for i in 0..n {
            all.push(self.knn_indices_dist2(i, k));
        }
        all
    }
}