//! The native post-processing pass (step 43): the linear passes of `landscape.post` and of
//! `landscape.model.project` over the rows of a field-resolved expansion, the rows kept in the
//! extension.
//!
//! `FieldResolvedRows` holds the rows (coefficient, milli exponent, y power, marker exponents)
//! that the expansion engine or the combined pass produced -- one row per (milli, y, markers),
//! sorted, integer coefficients, none zero -- and answers:
//!
//!   * `reduce(basis, milli_limit, block_limit)` -> (vanishing, fullpower, blocks, index2, net):
//!     the kernel (1 - t^3 y)(1 - t^3/y) applied to the expansion minus 1 (`post.reduced_index`
//!     with its strict truncation milli < milli_limit); `vanishing` whether the expansion minus 1
//!     is empty (`post.scan`'s c4 flag and `post.decouple`'s first test: the kernel is injective on
//!     polynomials, so the untruncated reduced index is empty iff the expansion minus 1 is);
//!     `fullpower` the largest y power of the reduced index below the limit (None when it is
//!     empty); `blocks` its rows at 0 < milli <= block_limit (the exponents the operator
//!     extraction reads, before the y-peel that Python performs); `index2` its flavor projection
//!     basis . markers (`post.project_flavor`) at every exponent below the limit; `net` the
//!     unrefined reduced index without truncation (`post.scan`'s net).  Zero sums are dropped
//!     everywhere, as `post._add` drops them; the half-even rounding of `reduced_index` is the
//!     identity on the integer coefficients.
//!   * `project(basis)` -> the physical index of `model.project`: the coefficient sums per
//!     (milli, y, basis . markers) over the rows, sorted by (milli, flavor, y) as Python sorts them.
//!   * `rows()`, `row(i)`, `__len__`, `n_fields`: the rows for the Python path, from which
//!     `model.FieldResolvedExpansion` builds FieldResolvedTerms on demand.
//!
//! Sums are 128-bit integers; an overflow raises OverflowError and the caller runs the Python
//! functions on the same terms.  `from_rows` builds the object from (numerator, denominator,
//! milli, y, markers) tuples (the combined pass's row format) for the comparisons of the check.
//!
//! With the flavor projection of step 44 (`expand_series(..., basis)`), the rows come in two
//! kinds: field-resolved through t^6 (`rows`) and flavor-refined above (`frows`, (coefficient,
//! milli, y, flavor)), the object recording the basis of the projection.  `reduce` then builds
//! the blocks at E <= 6 from the field-resolved rows (their sources lie at E <= 6, the kernel
//! raising exponents), the flavor projection, the net index and the y maximum from both kinds;
//! `project` sums both; `rows()` and `row(i)` raise TypeError, the field-resolved terms existing
//! through t^6 only, and `basis` must equal the projection's.

use pyo3::exceptions::{PyIndexError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rustc_hash::FxHashMap;
use std::hash::Hash;

const KERNEL: [(i64, i64, i128); 4] = [(0, 0, 1), (3000, 1, -1), (3000, -1, -1), (6000, 0, 1)];

pub struct Row {
    pub coeff: i128,
    pub milli: i64,
    pub ypow: i64,
    pub markers: Vec<i64>,
}

pub struct FlavorRow {
    pub coeff: i128,
    pub milli: i64,
    pub ypow: i64,
    pub flavor: Vec<i64>,
}

#[pyclass(frozen, name = "FieldResolvedRows")]
pub struct FieldResolvedRows {
    n_fields: usize,
    rows: Vec<Row>,
    frows: Vec<FlavorRow>,
    basis: Option<Vec<Vec<i64>>>,
}

fn overflow() -> PyErr {
    PyOverflowError::new_err("coefficient overflow in the native post-processing")
}

fn add_to<K: Hash + Eq>(map: &mut FxHashMap<K, i128>, key: K, c: i128) -> PyResult<()> {
    match map.get_mut(&key) {
        Some(e) => {
            *e = e.checked_add(c).ok_or_else(overflow)?;
        }
        None => {
            map.insert(key, c);
        }
    }
    Ok(())
}

/// basis . markers, over the shorter of a basis row and the markers (Python's zip).
fn flavor(basis: &[Vec<i64>], markers: &[i64]) -> PyResult<Vec<i64>> {
    let mut out = Vec::with_capacity(basis.len());
    for row in basis {
        let mut s: i64 = 0;
        for (b, e) in row.iter().zip(markers.iter()) {
            s = s.checked_add(b.checked_mul(*e).ok_or_else(overflow)?).ok_or_else(overflow)?;
        }
        out.push(s);
    }
    Ok(out)
}

fn int_tuple<'py>(py: Python<'py>, v: &[i64]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, v.iter().copied())
}

/// The rows of `expand` / `expand_series` -- sorted ((milli, y, markers), (numerator, denominator)),
/// and the flavor-refined rows ((milli, y, flavor), (numerator, denominator)) of a projected expansion
/// -- as the list of (numerator, denominator, milli, y, markers) tuples the Python path reads (no
/// flavor-refined rows then), or as a `FieldResolvedRows` object when `native_rows` is set; a
/// non-integral coefficient is an error either way (as `field_resolved` asserts).
pub fn emit_rows(py: Python<'_>, rows: Vec<((i128, i64, Vec<i64>), (i128, i128))>, frows: Vec<((i128, i64, Vec<i64>), (i128, i128))>,
                 n_fields: usize, basis: Option<Vec<Vec<i64>>>, native_rows: bool) -> PyResult<Py<PyAny>> {
    if native_rows {
        let mut out: Vec<Row> = Vec::with_capacity(rows.len());
        for ((milli, ypow, markers), (num, den)) in rows {
            if den != 1 {
                return Err(PyValueError::new_err(format!(
                    "non-integral coefficient {num}/{den} at t^{} y^{ypow} {markers:?}", milli as f64 / 1000.0)));
            }
            let milli: i64 = milli.try_into().map_err(|_| PyValueError::new_err("milli exponent out of range"))?;
            out.push(Row { coeff: num, milli, ypow, markers });
        }
        let mut fout: Vec<FlavorRow> = Vec::with_capacity(frows.len());
        for ((milli, ypow, flavor), (num, den)) in frows {
            if den != 1 {
                return Err(PyValueError::new_err(format!(
                    "non-integral coefficient {num}/{den} at t^{} y^{ypow} flavor {flavor:?}", milli as f64 / 1000.0)));
            }
            let milli: i64 = milli.try_into().map_err(|_| PyValueError::new_err("milli exponent out of range"))?;
            fout.push(FlavorRow { coeff: num, milli, ypow, flavor });
        }
        return Ok(Py::new(py, FieldResolvedRows { n_fields, rows: out, frows: fout, basis })?.into_any());
    }
    if !frows.is_empty() {
        return Err(PyValueError::new_err("flavor-refined rows need the native post pass (native_rows)"));
    }
    let out = PyList::empty(py);
    for ((milli, ypow, markers), (num, den)) in rows {
        if den != 1 {
            return Err(PyValueError::new_err(format!(
                "non-integral coefficient {num}/{den} at t^{} y^{ypow} {markers:?}", milli as f64 / 1000.0)));
        }
        let m = PyTuple::new(py, markers.iter().copied())?;
        out.append((num, den, milli, ypow, m))?;
    }
    Ok(out.into_any().unbind())
}

impl FieldResolvedRows {
    fn check_basis(&self, basis: &[Vec<i64>]) -> PyResult<()> {
        if let Some(b) = &self.basis {
            if b.as_slice() != basis {
                return Err(PyValueError::new_err("the flavor basis differs from the projection's"));
            }
        }
        Ok(())
    }

    fn only_field_resolved(&self) -> PyResult<()> {
        if !self.frows.is_empty() {
            return Err(PyTypeError::new_err("the expansion is flavor-refined above t^6 (LANDSCAPE_NATIVE_FLAVOR): its field-resolved terms exist through t^6 only"));
        }
        Ok(())
    }
}

#[pymethods]
impl FieldResolvedRows {
    /// from_rows(rows, n_fields): (numerator, denominator, milli, y, markers) tuples -> the object,
    /// equal keys summed (as `field_resolved` sums them), zero sums dropped, sorted.
    #[staticmethod]
    fn from_rows(rows: Vec<(i128, i128, i64, i64, Vec<i64>)>, n_fields: usize) -> PyResult<Self> {
        let mut acc: FxHashMap<(i64, i64, Vec<i64>), i128> = FxHashMap::default();
        for (num, den, milli, ypow, markers) in rows {
            if den != 1 {
                return Err(PyValueError::new_err(format!("non-integral coefficient {num}/{den} at t^{} y^{ypow} {markers:?}", milli as f64 / 1000.0)));
            }
            if markers.len() != n_fields {
                return Err(PyValueError::new_err(format!("{} markers for {n_fields} fields", markers.len())));
            }
            add_to(&mut acc, (milli, ypow, markers), num)?;
        }
        let mut keyed: Vec<((i64, i64, Vec<i64>), i128)> = acc.into_iter().filter(|(_, c)| *c != 0).collect();
        keyed.sort();
        Ok(FieldResolvedRows { n_fields, rows: keyed.into_iter().map(|((milli, ypow, markers), coeff)| Row { coeff, milli, ypow, markers }).collect(),
                               frows: Vec::new(), basis: None })
    }

    fn __len__(&self) -> usize {
        self.rows.len() + self.frows.len()
    }

    #[getter]
    fn n_fields(&self) -> usize {
        self.n_fields
    }

    /// The number of flavor-refined rows (above t^6; zero for an expansion without the projection).
    #[getter]
    fn n_flavor_rows(&self) -> usize {
        self.frows.len()
    }

    /// The basis of the projection (rows = U(1) generators, columns = fields), None without one.
    #[getter]
    fn basis(&self) -> Option<Vec<Vec<i64>>> {
        self.basis.clone()
    }

    /// The field-resolved rows as (numerator, 1, milli, y, markers) whatever the object holds above
    /// t^6 (through t^6 only for a projected expansion): for the comparisons of the checks.
    fn field_resolved_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for r in &self.rows {
            out.append((r.coeff, 1i128, r.milli, r.ypow, int_tuple(py, &r.markers)?))?;
        }
        Ok(out)
    }

    /// The flavor-refined rows as (numerator, milli, y, flavor), sorted.
    fn flavor_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for r in &self.frows {
            out.append((r.coeff, r.milli, r.ypow, int_tuple(py, &r.flavor)?))?;
        }
        Ok(out)
    }

    /// (numerator, milli, y, markers) of row i (negative i from the end); IndexError outside.
    fn row<'py>(&self, py: Python<'py>, i: isize) -> PyResult<(i128, i64, i64, Bound<'py, PyTuple>)> {
        self.only_field_resolved()?;
        let n = self.rows.len() as isize;
        let j = if i < 0 { i + n } else { i };
        if j < 0 || j >= n {
            return Err(PyIndexError::new_err("row index out of range"));
        }
        let r = &self.rows[j as usize];
        Ok((r.coeff, r.milli, r.ypow, int_tuple(py, &r.markers)?))
    }

    /// Every row as (numerator, 1, milli, y, markers), the format of the combined pass.
    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.only_field_resolved()?;
        let out = PyList::empty(py);
        for r in &self.rows {
            out.append((r.coeff, 1i128, r.milli, r.ypow, int_tuple(py, &r.markers)?))?;
        }
        Ok(out)
    }

    /// The physical index: [(milli, y, flavor, coefficient)] sorted by (milli, flavor, y), zero sums dropped.
    fn project<'py>(&self, py: Python<'py>, basis: Vec<Vec<i64>>) -> PyResult<Bound<'py, PyList>> {
        self.check_basis(&basis)?;
        let mut acc: FxHashMap<(i64, Vec<i64>, i64), i128> = FxHashMap::default();
        for r in &self.rows {
            let fl = flavor(&basis, &r.markers)?;
            add_to(&mut acc, (r.milli, fl, r.ypow), r.coeff)?;
        }
        for r in &self.frows {
            add_to(&mut acc, (r.milli, r.flavor.clone(), r.ypow), r.coeff)?;
        }
        let mut keyed: Vec<((i64, Vec<i64>, i64), i128)> = acc.into_iter().filter(|(_, c)| *c != 0).collect();
        keyed.sort();
        let out = PyList::empty(py);
        for ((milli, fl, ypow), c) in keyed {
            out.append((milli, ypow, int_tuple(py, &fl)?, c))?;
        }
        Ok(out)
    }

    /// reduce(basis, milli_limit, block_limit=6000) -> (vanishing, fullpower, blocks, index2, net); see the module.
    #[pyo3(signature = (basis, milli_limit, block_limit=6000))]
    fn reduce<'py>(&self, py: Python<'py>, basis: Vec<Vec<i64>>, milli_limit: i64, block_limit: i64)
        -> PyResult<(bool, Option<i64>, Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>)> {
        self.check_basis(&basis)?;
        let zeros = vec![0i64; self.n_fields];
        // the expansion minus 1 is empty iff the rows are exactly the constant 1 (or none): one row per key
        let vanishing = self.frows.is_empty() && self.rows.iter().all(|r| r.milli == 0 && r.ypow == 0 && r.coeff == 1 && r.markers == zeros);
        // marker vectors interned, so that the kernel's four images of a row share one key vector
        let mut ids: FxHashMap<&[i64], u32> = FxHashMap::default();
        let mut table: Vec<&[i64]> = Vec::new();
        let mut out: FxHashMap<(i64, i64, u32), i128> = FxHashMap::default();
        out.reserve(2 * self.rows.len());
        let mut net: FxHashMap<(i64, i64), i128> = FxHashMap::default();
        for r in &self.rows {
            let id = match ids.get(r.markers.as_slice()) {
                Some(id) => *id,
                None => {
                    let id = table.len() as u32;
                    ids.insert(r.markers.as_slice(), id);
                    table.push(r.markers.as_slice());
                    id
                }
            };
            for (dm, dy, s) in KERNEL {
                let m = r.milli + dm;
                let c = s.checked_mul(r.coeff).ok_or_else(overflow)?;
                add_to(&mut net, (m, r.ypow + dy), c)?;
                if m < milli_limit {
                    add_to(&mut out, (m, r.ypow + dy, id), c)?;
                }
            }
        }
        if !self.rows.is_empty() {
            let zero_id = match ids.get(zeros.as_slice()) {
                Some(id) => *id,
                None => {
                    let id = table.len() as u32;
                    table.push(zeros.as_slice());
                    id
                }
            };
            for (dm, dy, s) in KERNEL {
                add_to(&mut net, (dm, dy), -s)?;
                if dm < milli_limit {
                    add_to(&mut out, (dm, dy, zero_id), -s)?;
                }
            }
        }
        let mut reduced: Vec<((i64, i64, u32), i128)> = out.into_iter().filter(|(_, c)| *c != 0).collect();
        reduced.sort_by(|a, b| (a.0 .0, a.0 .1, table[a.0 .2 as usize]).cmp(&(b.0 .0, b.0 .1, table[b.0 .2 as usize])));
        let mut fullpower = reduced.iter().map(|((_, y, _), _)| *y).max();
        let blocks = PyList::empty(py);
        let mut index2: FxHashMap<(i64, i64, Vec<i64>), i128> = FxHashMap::default();
        for ((m, y, id), c) in &reduced {
            let markers = table[*id as usize];
            if *m > 0 && *m <= block_limit {
                blocks.append((*m, *y, int_tuple(py, markers)?, *c))?;
            }
            add_to(&mut index2, (*m, *y, flavor(&basis, markers)?), *c)?;
        }
        // the flavor-refined rows (above t^6): their kernel images join the flavor projection, the
        // net index and the y maximum (an upper bound of the reduced index's y support, which is
        // all the y-peel of the blocks needs)
        if !self.frows.is_empty() {
            let mut out_fl: FxHashMap<(i64, i64, Vec<i64>), i128> = FxHashMap::default();
            for r in &self.frows {
                for (dm, dy, s) in KERNEL {
                    let m = r.milli + dm;
                    let c = s.checked_mul(r.coeff).ok_or_else(overflow)?;
                    add_to(&mut net, (m, r.ypow + dy), c)?;
                    if m < milli_limit {
                        add_to(&mut out_fl, (m, r.ypow + dy, r.flavor.clone()), c)?;
                    }
                }
            }
            for ((m, y, fl), c) in out_fl {
                if c != 0 {
                    fullpower = Some(fullpower.map_or(y, |p| p.max(y)));
                    add_to(&mut index2, (m, y, fl), c)?;
                }
            }
        }
        let mut refined: Vec<((i64, i64, Vec<i64>), i128)> = index2.into_iter().filter(|(_, c)| *c != 0).collect();
        refined.sort();
        let index2_list = PyList::empty(py);
        for ((m, y, fl), c) in refined {
            index2_list.append((m, y, int_tuple(py, &fl)?, c))?;
        }
        let mut net_rows: Vec<((i64, i64), i128)> = net.into_iter().filter(|(_, c)| *c != 0).collect();
        net_rows.sort();
        let net_list = PyList::empty(py);
        for ((m, y), c) in net_rows {
            net_list.append((m, y, c))?;
        }
        Ok((vanishing, fullpower, blocks, index2_list, net_list))
    }
}
