# tomato

ToMATo clustering in Rust with a paper faithful pipeline: Vietoris–Rips graph plus Gaussian KDE density, then persistence based mode merging. Includes an HNSW acceleration backend for large datasets.

## What this crate implements

This crate provides two layers.

Core algorithm

- ToMATo core on a given undirected graph G and a density function f̂ on vertices
- deterministic behavior via a total order: higher density first, then smaller vertex id
- mathematically exact ToMATo merging semantics on the induced vertex superlevel filtration

Paper style pipeline

- build a Vietoris–Rips graph on a point cloud at radius r
- estimate density via Gaussian KDE
- run ToMATo with threshold tau

Two backends are available for neighbor queries

- BruteBackend for exact kNN and exact Rips, suitable for small to medium n
- HnswBackend for fast approximate kNN, suitable for large n

## Mathematical contract and guarantees

Inputs to the ToMATo core

- an undirected graph G on vertices V
- a density value f̂: V → R

Guarantee

- the ToMATo core computes exactly the thresholded 0 dimensional persistence merging on the vertex superlevel filtration of G induced by f̂

Key implementation detail

- merges at a vertex insertion level are processed in one batch
- this is required for correctness when a newly inserted vertex connects more than two active components at the same density level

Important note about HNSW

- HnswBackend typically returns approximate kNN lists
- the ToMATo core remains exact for the graph and density that you actually feed into it
- if you need exact kNN, exact Rips, and exact KDE sums, use BruteBackend

## Pipeline, paper faithful variants

This crate supports two paper faithful variants of the pipeline.

Speed variant, large n

- Rips graph is approximated by taking kNN edges and keeping only those with squared distance ≤ radius2
- KDE is estimated from kNN neighbors
- backend uses HNSW for kNN

Exact variant, small to medium n

- Rips graph is exact by testing all pairs for distance ≤ r
- KDE is exact by summing over all points
- backend uses brute force distances

These map to the following configuration.

Speed

- graph: GraphSpec::RipsFromKnnApprox
- density: DensitySpec::KdeGaussianKnn
- backend: HnswBackend

Exact

- graph: GraphSpec::RipsBrute
- density: DensitySpec::KdeGaussianFullBrute
- backend: BruteBackend

## API overview

Main entry points

- tomato::tomato::tomato_cluster for the ToMATo core
- tomato::pipeline::run_pipeline for the full graph plus density plus ToMATo pipeline

Key types

- GraphSpec selects how to build G
- DensitySpec selects how to estimate f̂
- TomatoParams holds tau

## Minimal usage, speed variant with HNSW

This is the paper style choice: Rips plus KDE plus ToMATo, accelerated.

~~~rust
use tomato::backend::{HnswBackend, HnswParams};
use tomato::pipeline::{run_pipeline, PipelineParams, GraphSpec, DensitySpec};
use tomato::tomato::TomatoParams;

let backend = HnswBackend::new(points, HnswParams::default())?;

let params = PipelineParams {
  graph: GraphSpec::RipsFromKnnApprox {
    k: 50,
    radius2: 1.50,
    symmetrize: true,
  },
  density: DensitySpec::KdeGaussianKnn {
    k: 50,
    bandwidth2: 0.20,
  },
  tomato: TomatoParams { tau: 0.15 },
};

let out = run_pipeline(&backend, params)?;
let labels = out.tomato.cluster_of;
let modes = out.tomato.modes;
~~~

## Minimal usage, exact variant with brute backend

This is the fully exact paper faithful pipeline on finite data.

~~~rust
use tomato::backend::BruteBackend;
use tomato::pipeline::{run_pipeline, PipelineParams, GraphSpec, DensitySpec};
use tomato::tomato::TomatoParams;

let backend = BruteBackend::new(points)?;

let params = PipelineParams {
  graph: GraphSpec::RipsBrute {
    radius2: 1.50,
  },
  density: DensitySpec::KdeGaussianFullBrute {
    bandwidth2: 0.20,
  },
  tomato: TomatoParams { tau: 0.15 },
};

let out = run_pipeline(&backend, params)?;
~~~

## Parameter guidance

radius2

- radius2 is r squared, measured in the squared distance of your metric
- in the speed variant, k in RipsFromKnnApprox must be large enough so most true neighbors within radius r appear in the kNN list

bandwidth2

- bandwidth2 is sigma squared in the Gaussian kernel exp of minus distance squared divided by 2 sigma squared
- bandwidth selection is data dependent, start by scaling points then tune

tau

- tau is the persistence threshold
- larger tau merges more modes
- tau zero keeps all discrete modes induced by G and f̂

Preprocessing

- standardize features before computing distances, the examples use z score scaling
- keep radius2 and bandwidth2 consistent with that scaling

## Examples

Iris dataset example using HNSW

- examples/iris_hnsw.rs downloads a public Iris CSV, scales features, builds approximate Rips, estimates kNN KDE, runs ToMATo

Run

~~~bash
cargo run --release --example iris_hnsw
~~~

Offline reproducibility

- vendor the Iris CSV into your repository
- replace the download step in the example with local file IO

## Testing

Run

~~~bash
cargo test
~~~

Test coverage includes

- regression test for batched merge correctness when one vertex connects multiple components
- tau very large yields connected components of the supplied graph
- pipeline smoke tests for brute and HNSW backends

## Performance

Let n be number of points and m be number of edges in the graph given to ToMATo.

- ToMATo core: O n log n plus m alpha n
- exact Rips: O n squared
- exact KDE: O n squared
- HNSW build and query: typically subquadratic in practice

For large n, the HNSW based speed variant is the intended choice.

## Citation

If you use this crate in academic work, cite the ToMATo paper.

~~~bibtex
@misc{martineau2025tomato_spike_sorting,
  title         = {ToMATo: an efficient and robust clustering algorithm for high dimensional datasets. An illustration with spike sorting.},
  author        = {Louise Martineau and Christophe Pouzat and S{\'e}gol{\`e}ne Geffray},
  year          = {2025},
  eprint        = {2509.17499},
  archivePrefix = {arXiv},
  primaryClass  = {stat.AP}
}
~~~
