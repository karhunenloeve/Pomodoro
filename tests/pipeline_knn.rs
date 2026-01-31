use tomato::backend::{BruteBackend, HnswBackend, HnswParams};
use tomato::pipeline::{build_graph, estimate_density, GraphSpec, DensitySpec};
use tomato::tomato::{tomato_cluster, TomatoParams};
use tomato::stats::zscore_in_place;

fn line_points(n: usize) -> Vec<Vec<f64>> {
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        pts.push(vec![i as f64]);
    }
    pts
}

#[test]
fn hnsw_knn_sanity_on_line() {
    let pts = line_points(200);
    let hnsw = HnswBackend::new(pts.clone(), HnswParams {
        ef_search: 256,
        ef_construction: 256,
        max_nb_connection: 32,
        max_layer: 16,
        use_parallel_insert: false,
    }).unwrap();

    let k = 5;
    let ans = hnsw.knn_indices_dist2(100, k);
    assert_eq!(ans.len(), k);
    for (idx, d2) in ans {
        assert_ne!(idx, 100);
        assert!(d2 >= 0.0);
    }
}

#[test]
fn pipeline_runs_brute() {
    let mut pts = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]];
    zscore_in_place(&mut pts);

    let brute = BruteBackend::new(pts).unwrap();

    let g = build_graph(&brute, GraphSpec::Knn { k: 2, symmetrize: true }).unwrap();
    let f = estimate_density(&brute, DensitySpec::KnnLog { k: 2, eps: 1e-12 }).unwrap();

    let res = tomato_cluster(&g, &f, TomatoParams { tau: 0.0 }).unwrap();
    assert_eq!(res.cluster_of.len(), 4);
}