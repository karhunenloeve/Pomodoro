use std::io::Read;

use tomato::backend::{AnnBackend, HnswBackend, HnswParams};
use tomato::pipeline::{run_pipeline, DensitySpec, GraphSpec, PipelineParams};
use tomato::stats::zscore_in_place;
use tomato::tomato::TomatoParams;

const IRIS_CSV_URL: &str =
    "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv";

fn load_iris_points() -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let mut resp = reqwest::blocking::get(IRIS_CSV_URL)?;
    let mut body = String::new();
    resp.read_to_string(&mut body)?;

    let mut rdr = csv::Reader::from_reader(body.as_bytes());

    let mut pts: Vec<Vec<f64>> = Vec::new();
    for row in rdr.records() {
        let r = row?;
        let sl: f64 = r[0].parse()?;
        let sw: f64 = r[1].parse()?;
        let pl: f64 = r[2].parse()?;
        let pw: f64 = r[3].parse()?;
        pts.push(vec![sl, sw, pl, pw]);
    }
    Ok(pts)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut points = load_iris_points()?;
    zscore_in_place(&mut points);

    let backend = HnswBackend::new(points, HnswParams {
        max_nb_connection: 24,
        ef_construction: 200,
        ef_search: 128,
        max_layer: 16,
        use_parallel_insert: false,
    })?;

    // Paper style pipeline: Rips + KDE + ToMATo

    let radius2 = 1.50;
    let bandwidth2 = 0.20;

    let k_rips = 50usize;
    let k_kde = 50usize;

    let tau = 0.15;

    let params = PipelineParams {
        graph: GraphSpec::RipsFromKnnApprox {
            k: k_rips,
            radius2,
            symmetrize: true,
        },
        density: DensitySpec::KdeGaussianKnn {
            k: k_kde,
            bandwidth2,
        },
        tomato: TomatoParams { tau },
    };

    let out = run_pipeline(&backend, params)?;

    println!("n {}", backend.len());
    println!("modes {}", out.tomato.modes.len());

    let mut counts = std::collections::HashMap::<usize, usize>::new();
    for &m in &out.tomato.cluster_of {
        *counts.entry(m).or_insert(0) += 1;
    }
    let mut clusters: Vec<(usize, usize)> = counts.into_iter().collect();
    clusters.sort_by(|a, b| b.1.cmp(&a.1));

    for (mode, size) in clusters {
        println!("mode {} size {}", mode, size);
    }

    Ok(())
}