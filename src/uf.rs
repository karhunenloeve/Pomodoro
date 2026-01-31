#![forbid(unsafe_code)]

use crate::order::higher;

#[derive(Debug, Clone)]
pub struct UfTomato {
    parent: Vec<usize>,
    size: Vec<usize>,
    mode: Vec<usize>,
    protected: Vec<bool>,
    active: Vec<bool>,
}

impl UfTomato {
    pub fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        let mut size = Vec::with_capacity(n);
        let mut mode = Vec::with_capacity(n);
        let mut protected = Vec::with_capacity(n);
        let mut active = Vec::with_capacity(n);

        for i in 0..n {
            parent.push(i);
            size.push(1);
            mode.push(i);
            protected.push(false);
            active.push(false);
        }

        Self {
            parent,
            size,
            mode,
            protected,
            active,
        }
    }

    #[inline]
    pub fn is_active(&self, v: usize) -> bool {
        self.active[v]
    }

    #[inline]
    pub fn activate(&mut self, v: usize) {
        self.active[v] = true;
        self.parent[v] = v;
        self.size[v] = 1;
        self.mode[v] = v;
        self.protected[v] = false;
    }

    #[inline]
    pub fn is_protected_root(&self, r: usize) -> bool {
        self.protected[r]
    }

    #[inline]
    pub fn protect_root(&mut self, r: usize) {
        self.protected[r] = true;
    }

    #[inline]
    pub fn mode_of_root(&self, r: usize) -> usize {
        self.mode[r]
    }

    pub fn find(&mut self, mut v: usize) -> usize {
        while self.parent[v] != v {
            let p = self.parent[v];
            let gp = self.parent[p];
            self.parent[v] = gp;
            v = p;
        }
        v
    }

    pub fn union_survivor(
        &mut self,
        density: &[f64],
        survivor_root: usize,
        other_root: usize,
    ) -> usize {
        debug_assert_eq!(self.parent[survivor_root], survivor_root);
        debug_assert_eq!(self.parent[other_root], other_root);
        debug_assert_ne!(survivor_root, other_root);

        let survivor_mode = self.mode[survivor_root];
        debug_assert!(higher(density, survivor_mode, self.mode[other_root]));

        let mut new_root = survivor_root;
        let mut child = other_root;

        if self.size[new_root] < self.size[child] {
            std::mem::swap(&mut new_root, &mut child);
        }

        self.parent[child] = new_root;
        self.size[new_root] += self.size[child];

        self.mode[new_root] = survivor_mode;
        self.protected[new_root] = self.protected[survivor_root];

        new_root
    }
}