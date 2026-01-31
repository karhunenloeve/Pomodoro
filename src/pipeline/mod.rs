#![forbid(unsafe_code)]

pub mod density;
pub mod graph_build;

pub use density::{estimate_density, DensitySpec};
pub use graph_build::{build_graph, GraphSpec};

use crate::backend::AnnBackend;
use crate::graph::Graph;
use crate::tomato::{tomato_cluster, TomatoError, TomatoParams, TomatoResult};

#[derive(Debug, Clone)]
pub struct PipelineParams {
    pub graph: GraphSpec,
    pub density: DensitySpec,
    pub tomato: TomatoParams,
}

#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub graph: Graph,
    pub density: Vec<f64>,
    pub tomato: TomatoResult,
}

pub fn run_pipeline<B: AnnBackend>(
    backend: &B,
    params: PipelineParams,
) -> Result<PipelineResult, TomatoError> {
    let graph = build_graph(backend, params.graph)?;
    let density = estimate_density(backend, params.density)?;
    let tomato = tomato_cluster(&graph, &density, params.tomato)?;
    Ok(PipelineResult { graph, density, tomato })
}