use tomato::graph::Graph;
use tomato::tomato::{tomato_cluster, TomatoParams};

fn connected_components(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    let mut comp = vec![usize::MAX; n];
    let mut cid = 0usize;
    let mut stack: Vec<usize> = Vec::new();

    for i in 0..n {
        if comp[i] != usize::MAX {
            continue;
        }
        comp[i] = cid;
        stack.clear();
        stack.push(i);
        while let Some(v) = stack.pop() {
            for &u in &adj[v] {
                if comp[u] == usize::MAX {
                    comp[u] = cid;
                    stack.push(u);
                }
            }
        }
        cid += 1;
    }
    comp
}

#[test]
fn tau_huge_gives_graph_connected_components() {
    let adj = vec![
        vec![1],
        vec![0, 2],
        vec![1],
        vec![4],
        vec![3],
    ];
    let g = Graph::new(adj.clone()).unwrap();
    let f = vec![5.0, 4.0, 3.0, 2.0, 1.0];

    let res = tomato_cluster(&g, &f, TomatoParams { tau: 1e100 }).unwrap();

    let cc = connected_components(&adj);
    for i in 0..f.len() {
        for j in 0..f.len() {
            let same_cc = cc[i] == cc[j];
            let same_cluster = res.cluster_of[i] == res.cluster_of[j];
            assert_eq!(same_cc, same_cluster);
        }
    }
}

#[test]
fn batched_union_counterexample_regression() {
    let a = 0usize;
    let b = 1usize;
    let c = 2usize;
    let v = 3usize;

    let g = Graph::new(vec![
        vec![v],
        vec![v],
        vec![v],
        vec![b, a, c],
    ])
    .unwrap();

    let f = vec![4.0, 5.0, 10.0, 0.0];
    let tau = 4.5;
    let res = tomato_cluster(&g, &f, TomatoParams { tau }).unwrap();

    assert_eq!(res.cluster_of[a], res.cluster_of[c]);
    assert_ne!(res.cluster_of[b], res.cluster_of[c]);
}