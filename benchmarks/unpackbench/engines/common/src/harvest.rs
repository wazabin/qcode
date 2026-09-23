//! The unpack example's harvest (`graph.rs`, `artifact.rs`) over what a
//! baseline's hooks recorded, writing the same `graph.json` schema, the same
//! `regions/<0xstart>-g<gen>.bin` files, and the contract's `meta.json`.
//!
//! One difference, by construction: a baseline has no IR, so its nodes are
//! the blocks the engine translated (Unicorn: entered) rather than every
//! lifted block, and its control-flow edges are the observed ones only;
//! `flags.static_edges` says so.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::record::{Layout, Record};

const MAX_REGION: u64 = 4 << 20;
const CHUNK: u64 = 4096;

#[derive(Debug, Clone)]
pub struct Node {
    pub addr: u64,
    pub end: u64,
    pub generated: bool,
    pub generation: u32,
    pub sites: Vec<u16>,
    pub first_entry: Option<usize>,
    pub executed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    pub kind: &'static str,
    pub origin: Option<&'static str>,
    pub site: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Region {
    pub start: u64,
    pub end: u64,
    pub generation: u32,
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub regions: Vec<Region>,
    pub warnings: Vec<String>,
}

fn covering(nodes: &[Node], a: u64) -> Option<usize> {
    let at = nodes.partition_point(|n| n.addr <= a);
    (1..=at.min(32)).find_map(|back| (a < nodes[at - back].end).then_some(at - back))
}

fn site_generation(nodes: &[Node], owner: &[Option<usize>], id: u16, of: Option<usize>) -> u32 {
    let parent = usize::from(id).checked_sub(1).and_then(|i| owner.get(i).copied().flatten());
    match parent {
        Some(p) if Some(p) != of => nodes[p].generation + 1,
        _ => 1,
    }
}

pub fn harvest(rec: &Record, layout: &Layout, edges_recorded: bool) -> Graph {
    let shadow = &rec.shadow;
    let ids = |s: u64, e: u64| layout.ids(shadow, s, e);
    let mut warnings = Vec::new();

    // Nodes: one per block address, the widest range seen for it.
    let mut ranges: BTreeMap<u64, u64> = BTreeMap::new();
    for b in &rec.blocks {
        let e = ranges.entry(b.addr).or_insert(b.end);
        *e = (*e).max(b.end);
    }
    let mut nodes: Vec<Node> = ranges
        .into_iter()
        .map(|(addr, end)| Node {
            addr,
            end: end.max(addr + 1),
            generated: false,
            generation: 0,
            sites: Vec::new(),
            first_entry: None,
            executed: false,
        })
        .collect();
    let index: BTreeMap<u64, usize> = nodes.iter().enumerate().map(|(i, n)| (n.addr, i)).collect();
    let node_of_block = |k: u32| rec.blocks.get(k as usize).and_then(|b| index.get(&b.addr).copied());

    for n in nodes.iter_mut() {
        if let Some(v) = ids(n.addr, n.end) {
            let seen: BTreeSet<u16> = v.into_iter().filter(|&i| i != 0).collect();
            n.generated = !seen.is_empty();
            n.sites = seen.into_iter().collect();
        }
    }
    for (pos, &(k, _)) in rec.log.iter().enumerate() {
        if let Some(i) = node_of_block(k)
            && !nodes[i].executed
        {
            nodes[i].executed = true;
            nodes[i].first_entry = Some(pos);
        }
    }

    let owner: Vec<Option<usize>> = rec.sites.iter().map(|s| covering(&nodes, s.pc)).collect();

    // generation = 1 + max(generation of the writers), by iteration.
    let generated: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].generated).collect();
    for &i in &generated {
        nodes[i].generation = 1;
    }
    let mut settled = generated.is_empty();
    for _ in 0..generated.len() + 1 {
        let mut changed = false;
        for &i in &generated {
            let want = nodes[i]
                .sites
                .iter()
                .map(|&id| site_generation(&nodes, &owner, id, Some(i)))
                .max()
                .unwrap_or(1);
            if want != nodes[i].generation {
                nodes[i].generation = want;
                changed = true;
            }
        }
        if !changed {
            settled = true;
            break;
        }
    }
    if !settled {
        warnings.push("the generation fixed point did not settle: the write graph has a cycle, and the generations reported are a lower bound".into());
    }

    let outside: Vec<u64> = nodes.iter().filter(|n| n.executed && !layout.in_window(n.addr)).map(|n| n.addr).collect();
    if let Some(first) = outside.first() {
        warnings.push(format!(
            "{} executed node(s) lie outside the provenance windows, the first at {first:#x}; their bytes carry no shadow and read as ungenerated",
            outside.len()
        ));
    }
    for n in &nodes {
        if let Some(pc) = n
            .sites
            .iter()
            .filter_map(|&id| rec.sites.get(usize::from(id).checked_sub(1)?))
            .map(|s| s.pc)
            .find(|&pc| pc >= n.addr && pc < n.end)
        {
            warnings.push(format!("the node at {:#x} holds the store at {pc:#x} that wrote it: it modified its own code", n.addr));
        }
    }
    let stamped: BTreeSet<u16> = nodes.iter().flat_map(|n| n.sites.iter().copied()).collect();
    let unattributed = stamped
        .iter()
        .filter(|&&id| usize::from(id).checked_sub(1).and_then(|i| owner.get(i).copied().flatten()).is_none())
        .count();
    if unattributed > 0 {
        warnings.push(format!(
            "{unattributed} store site(s) that wrote executed code are inside no node; what they wrote is one generation deep at most"
        ));
    }

    // Edges: generated_by from the shadow, control_flow from the log.
    let mut edges = BTreeSet::new();
    for n in &nodes {
        for &id in &n.sites {
            if let Some(p) = usize::from(id).checked_sub(1).and_then(|i| owner.get(i).copied().flatten()) {
                edges.insert(Edge { from: nodes[p].addr, to: n.addr, kind: "generated_by", origin: None, site: Some(id) });
            }
        }
    }
    if edges_recorded {
        for &(k, pred) in &rec.log {
            if let (Some(to), Some(from)) = (node_of_block(k), pred.and_then(node_of_block)) {
                edges.insert(Edge {
                    from: nodes[from].addr,
                    to: nodes[to].addr,
                    kind: "control_flow",
                    origin: Some("observed"),
                    site: None,
                });
            }
        }
    }

    // Regions: maximal runs of written bytes around executed generated
    // nodes, cut where the writer's generation changes.
    let window_of = |a: u64| layout.windows().into_iter().find(|w| w.contains(a)).map(|w| (w.start, w.end()));
    let fitting = |room: u64, read: &dyn Fn(u64) -> Option<Vec<u16>>| {
        let mut step = CHUNK.min(room);
        while step > 0 {
            if let Some(v) = read(step) {
                return Some(v);
            }
            step /= 2;
        }
        None
    };
    let grow_down = |from: u64| {
        let Some((start, _)) = window_of(from.saturating_sub(1)) else { return from };
        let mut at = from;
        loop {
            let room = (at - start).min(MAX_REGION - (from - at));
            let Some(v) = fitting(room, &|step| ids(at - step, at)) else { return at };
            let w = v.iter().rev().take_while(|&&i| i != 0).count();
            at -= w as u64;
            if w < v.len() {
                return at;
            }
        }
    };
    let grow_up = |from: u64, lo: u64| {
        let Some((_, end)) = window_of(from) else { return from };
        let mut at = from;
        loop {
            let room = (end - at).min(MAX_REGION - (at - lo));
            let Some(v) = fitting(room, &|step| ids(at, at + step)) else { return at };
            let w = v.iter().take_while(|&&i| i != 0).count();
            at += w as u64;
            if w < v.len() {
                return at;
            }
        }
    };
    let seeds: Vec<&Node> = nodes.iter().filter(|n| n.executed && n.generated).collect();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for n in &seeds {
        let Some(v) = ids(n.addr, n.end) else { continue };
        let mut at = 0;
        while at < v.len() {
            if v[at] == 0 {
                at += 1;
                continue;
            }
            let mut to = at;
            while to < v.len() && v[to] != 0 {
                to += 1;
            }
            let (lo, hi) = (n.addr + at as u64, n.addr + to as u64);
            at = to;
            if runs.iter().any(|&(a, b)| lo >= a && hi <= b) {
                continue;
            }
            let lo = grow_down(lo);
            let hi = grow_up(hi, lo);
            if hi - lo >= MAX_REGION {
                warnings.push(format!("the generated run at {lo:#x} reached the {MAX_REGION:#x}-byte harvest limit and is reported truncated"));
            }
            runs.push((lo, hi));
        }
    }
    runs.sort_unstable();
    runs.dedup();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (lo, hi) in runs {
        match merged.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    let mut regions = Vec::new();
    for (lo, hi) in merged {
        let Some(v) = ids(lo, hi) else { continue };
        let mut at = 0;
        while at < v.len() {
            let g = site_generation(&nodes, &owner, v[at], None);
            let mut to = at;
            while to < v.len() && site_generation(&nodes, &owner, v[to], None) == g {
                to += 1;
            }
            let r = Region { start: lo + at as u64, end: lo + to as u64, generation: g };
            if seeds.iter().any(|n| n.addr < r.end && r.start < n.end) {
                regions.push(r);
            }
            at = to;
        }
    }
    regions.sort_unstable();

    Graph { nodes, edges: edges.into_iter().collect(), regions, warnings }
}

/// One finished run, as the artifact writer needs it.
pub struct Run {
    pub engine: String,
    pub strategy: String,
    pub path: String,
    pub entry: u64,
    pub exit: Option<i32>,
    pub stop_reason: String,
    pub crashed: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub steps: Option<u64>,
    pub layout: Layout,
    pub hooks: bool,
    pub edges: bool,
    pub record: Option<Record>,
    pub warnings: Vec<String>,
}

fn hex(v: u64) -> String {
    format!("{v:#x}")
}

pub fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/// Writes `graph.json`, `regions/`, `stderr.txt` and `meta.json` into `dir`.
/// `read` reads the final guest memory; `meta` is merged into `meta.json`.
pub fn write(dir: &Path, run: &Run, read: &mut dyn FnMut(u64, &mut [u8]) -> bool, meta: Value) -> Result<(), String> {
    let fail = |what: &str, e: std::io::Error| format!("cannot write {what}: {e}");
    // A rerun into the same directory must not leave an old run's regions.
    let _ = fs::remove_dir_all(dir.join("regions"));
    fs::create_dir_all(dir.join("regions")).map_err(|e| fail("regions/", e))?;
    let edges_recorded = run.hooks && run.edges;
    let graph = match &run.record {
        Some(rec) => harvest(rec, &run.layout, edges_recorded),
        None => Graph { nodes: vec![], edges: vec![], regions: vec![], warnings: vec![] },
    };
    let mut warnings = run.warnings.clone();
    warnings.extend(graph.warnings.iter().cloned());
    for r in &graph.regions {
        let name = format!("{:#x}-g{}.bin", r.start, r.generation);
        let mut bytes = vec![0u8; (r.end - r.start) as usize];
        if !read(r.start, &mut bytes) {
            warnings.push(format!("the generated region at {:#x} is no longer readable; regions/{name} holds zeroes", r.start));
        }
        fs::write(dir.join("regions").join(&name), &bytes).map_err(|e| fail(&name, e))?;
    }
    let rec = run.record.as_ref();
    let report = json!({
        "program": {
            "path": run.path, "entry": hex(run.entry), "exit_status": run.exit,
            "stopped": run.stop_reason, "steps": run.steps, "strategy": run.strategy,
            "engine": run.engine,
        },
        "io": {
            "stdout": String::from_utf8_lossy(&run.stdout),
            "stderr": String::from_utf8_lossy(&run.stderr),
        },
        "windows": run.layout.windows().iter().map(|w| json!({
            "start": hex(w.start), "end": hex(w.end()), "bytes_len": w.len,
        })).collect::<Vec<_>>(),
        "nodes": graph.nodes.iter().map(|n| {
            let mut v = json!({
                "addr": hex(n.addr), "range": [hex(n.addr), hex(n.end)],
                "generated": n.generated, "generation": n.generation, "sites": n.sites,
                "executed": n.executed,
            });
            if let Some(f) = n.first_entry { v["first_entry"] = json!(f); }
            v
        }).collect::<Vec<_>>(),
        "edges": graph.edges.iter().map(|e| {
            let mut v = json!({"from": hex(e.from), "to": hex(e.to), "kind": e.kind});
            if let Some(o) = e.origin { v["origin"] = json!(o); }
            if let Some(s) = e.site { v["site"] = json!(s); }
            v
        }).collect::<Vec<_>>(),
        "sites": rec.map(|r| r.sites.iter().enumerate().map(|(i, s)| json!({
            "id": (i + 1).min(u16::MAX as usize), "pc": hex(s.pc), "size": s.size,
        })).collect::<Vec<_>>()).unwrap_or_default(),
        "regions": graph.regions.iter().map(|r| json!({
            "start": hex(r.start), "end": hex(r.end), "generation": r.generation,
            "bytes_len": r.end - r.start,
        })).collect::<Vec<_>>(),
        "flags": {
            "hooks": run.hooks, "edges_recorded": edges_recorded, "crashed": run.crashed,
            "sites_saturated": rec.is_some_and(|r| r.sites_saturated),
            "blocks_saturated": rec.is_some_and(|r| r.blocks_saturated),
            "evicted": 0, "static_edges": false,
        },
        "warnings": warnings,
    });
    let text = serde_json::to_string_pretty(&report).unwrap() + "\n";
    fs::write(dir.join("graph.json"), text).map_err(|e| fail("graph.json", e))?;
    fs::write(dir.join("stderr.txt"), &run.stderr).map_err(|e| fail("stderr.txt", e))?;

    let mut m = json!({
        "exit": run.exit,
        "stdout_sha256": sha256_hex(&run.stdout),
        "crashed": run.crashed,
        "stop_reason": run.stop_reason,
        "regions": graph.regions.iter().map(|r| json!({
            "start": hex(r.start), "end": hex(r.end), "generation": r.generation,
        })).collect::<Vec<_>>(),
    });
    if let (Value::Object(m), Value::Object(extra)) = (&mut m, meta) {
        m.extend(extra);
    }
    fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&m).unwrap() + "\n")
        .map_err(|e| fail("meta.json", e))?;
    Ok(())
}

/// `meta.json` for a binary the engine cannot run: no graph.
pub fn write_unsupported(dir: &Path, meta: Value, reason: &str, stderr: &[u8]) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let _ = fs::remove_file(dir.join("graph.json"));
    let _ = fs::remove_dir_all(dir.join("regions"));
    let mut m = meta;
    m["unsupported"] = json!(reason);
    fs::write(dir.join("stderr.txt"), stderr).map_err(|e| e.to_string())?;
    fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&m).unwrap() + "\n").map_err(|e| e.to_string())
}

/// The median of `v`.
pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() { 0.0 } else { s[s.len() / 2] }
}
