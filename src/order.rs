#![forbid(unsafe_code)]

#[inline]
pub fn higher(density: &[f64], a: usize, b: usize) -> bool {
    let fa = density[a];
    let fb = density[b];
    fa > fb || (fa == fb && a < b)
}

pub fn vertices_desc_by_density(density: &[f64]) -> Vec<usize> {
    let mut ord: Vec<usize> = (0..density.len()).collect();
    ord.sort_by(|&a, &b| {
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
    ord
}