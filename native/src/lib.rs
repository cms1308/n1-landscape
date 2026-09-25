//! landscape_native: native implementations of the measured hot spots of the landscape
//! package.  Optional at import; the pure-Python functions of `landscape/` are the
//! reference and stay.
//!
//! `parse_terms(text, n_nodes, term_cls, fraction_cls)` parses a FORM output (the
//! '+'-separated terms of `FormRunner.run`, no whitespace) into the same `Term`
//! namedtuples that `landscape.singlet.parse_term` builds term by term:
//!
//!   Term(coeff: Fraction, milli: int, ypow: int,
//!        fug: tuple[(symbol, exponent), ...] sorted,
//!        chars: tuple[per node: tuple[(label tuple, Adams key vector), ...] sorted])
//!
//! Factor semantics, mirrored from `parse_term`: `d(n,m)` multiplies the coefficient by
//! n/m; a bare integer (optionally negative, optionally `^e`) multiplies it by its e-th
//! power; `t^a`, `s^b`, `r^c` accumulate exponent digits, the milli exponent is the
//! half-up rounding of 1000 (a/500 + b/2,500,000 + c/12,500,000); `y[^e]` accumulates the
//! y power; `C<node>L<l1>x<l2>..(k)[^e]` adds e to the Adams multiplicity k of that label
//! at that node (an unknown function or a node beyond n_nodes is a ValueError); every
//! other base is a fugacity symbol (field markers) whose exponents are summed, zero sums
//! dropped, the tuple sorted by (symbol, exponent).  Per node the labels are sorted and
//! the key vector of a label is [m_1, ..., m_N] with N = sum_k k m_k (zero multiplicities
//! dropped).  Coefficients, fugacity tuples and character tuples that repeat within one
//! call are shared objects (equal values).
//!
//! `expand_series(...)` (module `series`) is the expansion engine in place of FORM: the
//! exponential of the explicit itotal, truncated as FORM truncates it, with the refinements of
//! step 44 as parameters (the flavor projection above t^6, the 64-bit coefficient path, the exact
//! truncation).
//!
//! `lie_run(lcode, timeout)` (module `lie`) is the character-arithmetic engine: LiE's stdout
//! for the `Adams` and `tensor` lcode forms of the pipeline, byte for byte, NotImplementedError
//! for anything else (the caller falls back to the LiE subprocess).
//!
//! `expand(text, n_nodes, n_fields, t_order, lookup)` is the combined pass: it parses the
//! same output, drops the terms above the truncation (milli > 1000 t_order), obtains the
//! singlet multiplicity of every distinct character product through one call of `lookup`
//! (the projector's `multiplicity`, given the product as the same tuple `parse_terms`
//! builds) and sums coefficient x multiplicity per (milli, y power, marker exponents), the
//! marker symbols `f<i>` mapped to field i - 1; it returns the sorted list of
//! (numerator, denominator, milli, y power, markers) with nonzero sums, the field-resolved
//! expansion of `landscape.index.field_resolved` without the Term objects in between.
//!
//! `FieldResolvedRows` (module `post`) is the native post-processing pass of step 43: `expand` and
//! `expand_series` return their rows as this object when asked (`native_rows`), and it answers the
//! reduced index, its flavor projection, the net index and the physical index over them.

mod lie;
mod post;
mod series;

use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use std::collections::{BTreeMap, HashMap};

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn normalize(num: i128, den: i128) -> PyResult<(i128, i128)> {
    if den == 0 {
        return Err(PyZeroDivisionError::new_err(format!("Fraction({num}, {den})")));   // as Fraction(n, 0) raises
    }
    let g = gcd(num, den);
    let (mut n, mut d) = if g > 1 { (num / g, den / g) } else { (num, den) };
    if d < 0 {
        n = -n;
        d = -d;
    }
    Ok((n, d))
}

fn checked_mul(a: i128, b: i128) -> PyResult<i128> {
    a.checked_mul(b)
        .ok_or_else(|| PyValueError::new_err("coefficient overflow in the native parser"))
}

fn parse_i128(s: &str, what: &str, term: &str) -> PyResult<i128> {
    s.parse::<i128>()
        .map_err(|_| PyValueError::new_err(format!("bad {what} '{s}' in term '{term}'")))
}

fn parse_i64(s: &str, what: &str, term: &str) -> PyResult<i64> {
    s.parse::<i64>()
        .map_err(|_| PyValueError::new_err(format!("bad {what} '{s}' in term '{term}'")))
}

fn is_int_literal(s: &str) -> bool {
    let digits = s.strip_prefix('-').unwrap_or(s);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// `C<node>L<l1>x<l2>...` -> (node, label); None when the name does not match.
fn parse_symbol(name: &str) -> Option<(usize, Vec<i64>)> {
    let rest = name.strip_prefix('C')?;
    let lpos = rest.find('L')?;
    let node_s = &rest[..lpos];
    let labels_s = &rest[lpos + 1..];
    if node_s.is_empty() || !node_s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if labels_s.is_empty() {
        return None;
    }
    let mut label = Vec::new();
    for part in labels_s.split('x') {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        label.push(part.parse::<i64>().ok()?);
    }
    Some((node_s.parse::<usize>().ok()?, label))
}

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

/// {k: m} -> [m_1, ..., m_N], N = sum k m_k (zero multiplicities dropped); () when empty.
fn key_vec_of(adams: &BTreeMap<i64, i64>, term: &str) -> PyResult<Vec<i64>> {
    let nonzero: Vec<(i64, i64)> = adams.iter().filter(|(_, m)| **m != 0).map(|(k, m)| (*k, *m)).collect();
    if nonzero.is_empty() {
        return Ok(Vec::new());
    }
    let length: i64 = nonzero.iter().map(|(k, m)| k * m).sum();
    if length < 0 {
        return Err(PyValueError::new_err(format!("negative Adams degree in term '{term}'")));
    }
    let mut vec = vec![0i64; length as usize];
    for (k, m) in nonzero {
        if k < 1 || (k - 1) as usize >= vec.len() {
            return Err(PyValueError::new_err(format!("Adams index {k} outside the key vector in term '{term}'")));
        }
        vec[(k - 1) as usize] = m;
    }
    Ok(vec)
}

/// Half-up rounding of 1000 (a/500 + b/2,500,000 + c/12,500,000), exactly.
fn milli_exponent(tpow: i64, spow: i64, rpow: i64) -> i128 {
    let n: i128 = 25_000_000i128 * tpow as i128 + 5_000i128 * spow as i128 + rpow as i128;
    let d: i128 = 12_500_000;
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

pub type NodeChars = Vec<(Vec<i64>, Vec<i64>)>;

struct Parsed {
    num: i128,
    den: i128,
    milli: i128,
    ypow: i64,
    fug: Vec<(String, i64)>,
    chars: Vec<NodeChars>,
}

fn parse_one(term: &str, n_nodes: usize) -> PyResult<Parsed> {
    let mut num: i128 = 1;
    let mut den: i128 = 1;
    let (mut tpow, mut spow, mut rpow, mut ypow) = (0i64, 0i64, 0i64, 0i64);
    let mut fug: BTreeMap<String, i64> = BTreeMap::new();
    let mut per_node: Vec<BTreeMap<Vec<i64>, BTreeMap<i64, i64>>> = (0..n_nodes).map(|_| BTreeMap::new()).collect();
    for factor in term.split('*') {
        if factor.starts_with("d(") && factor.ends_with(')') {
            let inner = &factor[2..factor.len() - 1];
            let (n_str, m_str) = inner
                .split_once(',')
                .ok_or_else(|| PyValueError::new_err(format!("bad coefficient '{factor}' in term '{term}'")))?;
            let n = parse_i128(n_str, "coefficient numerator", term)?;
            let m = parse_i128(m_str, "coefficient denominator", term)?;
            let (n, m) = normalize(n, m)?;
            num = checked_mul(num, n)?;
            den = checked_mul(den, m)?;
            let (a, b) = normalize(num, den)?;
            num = a;
            den = b;
            continue;
        }
        let (base, exp) = match factor.split_once('^') {
            Some((b, e)) => (b, parse_i64(e.trim_matches(|c| c == '(' || c == ')'), "exponent", term)?),
            None => (factor, 1i64),
        };
        if base.contains('(') {
            if !base.ends_with(')') {
                return Err(PyValueError::new_err(format!("unknown character function '{factor}'")));
            }
            let body = &base[..base.len() - 1];
            let (name, arg) = body
                .split_once('(')
                .ok_or_else(|| PyValueError::new_err(format!("unknown character function '{factor}'")))?;
            let (node, label) = parse_symbol(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown character function '{factor}'")))?;
            if node >= n_nodes {
                return Err(PyValueError::new_err(format!("node {node} of '{factor}' beyond {n_nodes} nodes")));
            }
            let k = parse_i64(arg, "Adams index", term)?;
            let entry = per_node[node].entry(label).or_default();
            *entry.entry(k).or_insert(0) += exp;
        } else if base == "t" {
            tpow += exp;
        } else if base == "s" {
            spow += exp;
        } else if base == "r" {
            rpow += exp;
        } else if base == "y" {
            ypow += exp;
        } else if is_int_literal(base) {
            let b = parse_i128(base, "integer factor", term)?;
            if exp >= 0 {
                for _ in 0..exp {
                    num = checked_mul(num, b)?;
                }
            } else {
                for _ in 0..(-exp) {
                    den = checked_mul(den, b)?;
                }
            }
            let (a, c) = normalize(num, den)?;
            num = a;
            den = c;
        } else if !base.is_empty() {
            *fug.entry(base.to_string()).or_insert(0) += exp;
        } else {
            return Err(PyValueError::new_err(format!("empty factor in term '{term}'")));
        }
    }
    let fug: Vec<(String, i64)> = fug.into_iter().filter(|(_, e)| *e != 0).collect(); // BTreeMap: sorted by symbol; one entry per symbol
    let mut chars: Vec<NodeChars> = Vec::with_capacity(n_nodes);
    for per_label in per_node.iter() {
        let mut node: NodeChars = Vec::new();
        for (label, adams) in per_label.iter() {
            if adams.values().any(|m| *m != 0) {
                node.push((label.clone(), key_vec_of(adams, term)?));
            }
        }
        node.sort(); // (label, key vector) lexicographic, as Python's sorted on the tuples
        chars.push(node);
    }
    Ok(Parsed { num, den, milli: milli_exponent(tpow, spow, rpow), ypow, fug, chars })
}

struct Caches<'py> {
    coeff: HashMap<(i128, i128), Bound<'py, PyAny>>,
    fug: HashMap<Vec<(String, i64)>, Bound<'py, PyAny>>,
    chars: HashMap<Vec<NodeChars>, Bound<'py, PyAny>>,
}

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

fn fug_object<'py>(py: Python<'py>, fug: &[(String, i64)]) -> PyResult<Bound<'py, PyAny>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(fug.len());
    for (sym, e) in fug {
        let pair = PyTuple::new(py, [PyString::new(py, sym).into_any(), e.into_pyobject(py)?.into_any()])?;
        items.push(pair.into_any());
    }
    Ok(PyTuple::new(py, items)?.into_any())
}

/// parse_terms(text, n_nodes, term_cls, fraction_cls) -> list[Term]
#[pyfunction]
fn parse_terms<'py>(
    py: Python<'py>,
    text: &str,
    n_nodes: usize,
    term_cls: Bound<'py, PyAny>,
    fraction_cls: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let mut caches = Caches { coeff: HashMap::new(), fug: HashMap::new(), chars: HashMap::new() };
    let out = PyList::empty(py);
    for term in text.split('+') {
        if term.is_empty() {
            continue;
        }
        let p = parse_one(term, n_nodes)?;
        let coeff = match caches.coeff.get(&(p.num, p.den)) {
            Some(c) => c.clone(),
            None => {
                let c = fraction_cls.call1((p.num, p.den))?;
                caches.coeff.insert((p.num, p.den), c.clone());
                c
            }
        };
        let fug = match caches.fug.get(&p.fug) {
            Some(f) => f.clone(),
            None => {
                let f = fug_object(py, &p.fug)?;
                caches.fug.insert(p.fug.clone(), f.clone());
                f
            }
        };
        let chars = match caches.chars.get(&p.chars) {
            Some(c) => c.clone(),
            None => {
                let c = chars_object(py, &p.chars)?;
                caches.chars.insert(p.chars.clone(), c.clone());
                c
            }
        };
        let term_obj = term_cls.call1((coeff, p.milli, p.ypow, fug, chars))?;
        out.append(term_obj)?;
    }
    Ok(out)
}

fn add_fraction(acc: &mut (i128, i128), num: i128, den: i128) -> PyResult<()> {
    // acc += num/den, both reduced with positive denominators
    let (a, b) = *acc;
    let n = checked_mul(a, den)?.checked_add(checked_mul(num, b)?)
        .ok_or_else(|| PyValueError::new_err("coefficient overflow in the native expansion"))?;
    let d = checked_mul(b, den)?;
    *acc = normalize(n, d)?;
    Ok(())
}

/// expand(text, n_nodes, n_fields, t_order, lookup, native_rows=False) -> list[(num, den, milli, ypow, markers)],
/// or the rows as a FieldResolvedRows object when native_rows is set
#[pyfunction]
#[pyo3(signature = (text, n_nodes, n_fields, t_order, lookup, native_rows=false))]
fn expand<'py>(
    py: Python<'py>,
    text: &str,
    n_nodes: usize,
    n_fields: usize,
    t_order: i64,
    lookup: Bound<'py, PyAny>,
    native_rows: bool,
) -> PyResult<Py<PyAny>> {
    let limit: i128 = 1000i128 * t_order as i128;
    let mut mults: HashMap<Vec<NodeChars>, i128> = HashMap::new();
    let mut acc: HashMap<(i128, i64, Vec<i64>), (i128, i128)> = HashMap::new();
    let empty: Vec<NodeChars> = (0..n_nodes).map(|_| Vec::new()).collect();
    for term in text.split('+') {
        if term.is_empty() {
            continue;
        }
        let p = parse_one(term, n_nodes)?;
        if p.milli > limit {
            continue;
        }
        let mult: i128 = if p.chars == empty {
            1
        } else if let Some(m) = mults.get(&p.chars) {
            *m
        } else {
            let obj = chars_object(py, &p.chars)?;
            let m: i128 = lookup.call1((obj,))?.extract()?;
            mults.insert(p.chars.clone(), m);
            m
        };
        if mult == 0 {
            continue;
        }
        let mut markers = vec![0i64; n_fields];
        for (sym, e) in &p.fug {
            let idx = sym
                .strip_prefix('f')
                .and_then(|d| if !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()) { d.parse::<usize>().ok() } else { None })
                .filter(|i| *i >= 1 && *i <= n_fields)
                .ok_or_else(|| PyValueError::new_err(format!("unknown fugacity symbol '{sym}' in term '{term}'")))?;
            markers[idx - 1] = *e;
        }
        let entry = acc.entry((p.milli, p.ypow, markers)).or_insert((0, 1));
        add_fraction(entry, checked_mul(p.num, mult)?, p.den)?;
    }
    let mut rows: Vec<((i128, i64, Vec<i64>), (i128, i128))> = acc.into_iter().filter(|(_, (n, _))| *n != 0).collect();
    rows.sort();
    post::emit_rows(py, rows, Vec::new(), n_fields, None, native_rows)
}

#[pymodule]
fn landscape_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_terms, m)?)?;
    m.add_function(wrap_pyfunction!(expand, m)?)?;
    m.add_function(wrap_pyfunction!(lie::lie_run, m)?)?;
    m.add_function(wrap_pyfunction!(lie::lie_dim, m)?)?;
    m.add_function(wrap_pyfunction!(series::expand_series, m)?)?;
    m.add_class::<post::FieldResolvedRows>()?;
    m.add("__version__", "0.6.0")?;
    Ok(())
}
