//! landscape_native: the native computations of the landscape package -- required by it; the pure-Python
//! functions of `landscape/` that it computes are the references of the checks.
//!
//! `expand_series(...)` (module `series`) is the expansion engine: the exponential of the explicit itotal
//! (`landscape.form.itotal_terms`), truncated at the t-power limit and the expansion order, in the irreducible mode
//! (the default) or in the slot engine (`engine="slot"`, the checks' reference), with the flavor projection above t^6,
//! the 64-bit coefficient path and the exact truncation as parameters.
//!
//! `lie_run(lcode, timeout)`, `dom_char(group, label)` and `lie_dim` (module `lie`) are the character-arithmetic engine:
//! the character stores' `Adams` and `tensor` lcode forms answered as LiE prints them, byte for byte, and the dominant
//! characters; `sym_singlet` (module `project`) the singlet count of a product of symmetric powers.
//!
//! `FieldResolvedRows` (module `post`) is the native post-processing pass: `expand_series` returns its rows as this object
//! when asked (`native_rows`), and it answers the reduced index, its flavor projection, the net index and the physical
//! index over them.

mod lie;
mod post;
mod project;
mod series;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::BTreeMap;

/// {k: m} -> [m_1, ..., m_N] without a term context (the series engine).
pub fn key_vec_of_map(adams: &BTreeMap<i64, i64>) -> Result<Vec<i64>, String> {
    let nonzero: Vec<(i64, i64)> = adams.iter().filter(|(_, m)| **m != 0).map(|(k, m)| (*k, *m)).collect();
    if nonzero.is_empty() {
        return Ok(Vec::new());
    }
    let length: i64 = nonzero.iter().map(|(k, m)| k * m).sum();
    if length < 0 {
        return Err("negative Adams degree".into());
    }
    let mut vec = vec![0i64; length as usize];
    for (k, m) in nonzero {
        if k < 1 || (k - 1) as usize >= vec.len() {
            return Err(format!("Adams index {k} outside the key vector"));
        }
        vec[(k - 1) as usize] = m;
    }
    Ok(vec)
}

pub type NodeChars = Vec<(Vec<i64>, Vec<i64>)>;

fn int_tuple<'py>(py: Python<'py>, v: &[i64]) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyTuple::new(py, v.iter().copied())?.into_any())
}

pub fn chars_object<'py>(py: Python<'py>, chars: &[NodeChars]) -> PyResult<Bound<'py, PyAny>> {
    let mut nodes: Vec<Bound<'py, PyAny>> = Vec::with_capacity(chars.len());
    for node in chars {
        let mut entries: Vec<Bound<'py, PyAny>> = Vec::with_capacity(node.len());
        for (label, key) in node {
            let pair = PyTuple::new(py, [int_tuple(py, label)?, int_tuple(py, key)?])?;
            entries.push(pair.into_any());
        }
        nodes.push(PyTuple::new(py, entries)?.into_any());
    }
    Ok(PyTuple::new(py, nodes)?.into_any())
}

#[pymodule]
fn landscape_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lie::lie_run, m)?)?;
    m.add_function(wrap_pyfunction!(lie::lie_dim, m)?)?;
    m.add_function(wrap_pyfunction!(lie::dom_char, m)?)?;
    m.add_function(wrap_pyfunction!(project::sym_singlet, m)?)?;
    m.add_function(wrap_pyfunction!(series::expand_series, m)?)?;
    m.add_function(wrap_pyfunction!(series::last_work, m)?)?;
    m.add_class::<post::FieldResolvedRows>()?;
    m.add("__version__", "0.12.0")?;
    Ok(())
}
