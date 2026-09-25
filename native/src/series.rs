//! The native expansion engine (step 39f): the exponential of the explicit itotal of
//! `landscape.form.program`, truncated at the t-power limit and at `max_order` powers, in
//! place of the FORM run -- the same polynomial FORM prints as `result`.
//!
//! A monomial is an exponent vector [t, s, r, y, f_1..f_n, c_1..c_m] (`form.Series`); the
//! product of two monomials adds the vectors; a coefficient is a reduced 128-bit rational
//! (an overflow raises, and the caller falls back to FORM).  With P_1 = itotal and
//! P_k = P_{k-1} itotal / k, the result is 1 + sum_k P_k for k = 1..max_order, every product
//! whose t-power exceeds the limit dropped before it is formed (the terms of itotal sorted by
//! t-power, the multiplication of a term of P_{k-1} stopping at the bound) -- the products FORM's
//! bounded loop generates and keeps, and none of the ones it discards.  The products of the
//! terms of P_{k-1} run in parallel (rayon), the partial maps merged; a deadline is checked
//! between chunks and raises subprocess.TimeoutExpired.
//!
//! `expand_series(..., monomials=true)` returns the polynomial itself, (numerator,
//! denominator, exponents) per monomial, for the comparison with FORM's parsed output;
//! `monomials=false` returns the rows of the field-resolved expansion as `expand` does: the
//! terms above t_order dropped by their milli exponent, the singlet multiplicity of every
//! distinct character product obtained through the projector's `multiplicity`, coefficient x
//! multiplicity summed per (milli, y power, markers).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::time::Instant;

type Exps = Vec<i32>;
type Coef = (i128, i128);

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

#[inline]
fn reduce(n: i128, d: i128) -> Result<Coef, String> {
    if d == 0 {
        return Err("zero denominator".into());
    }
    let g = gcd(n, d);
    let (mut n, mut d) = if g > 1 { (n / g, d / g) } else { (n, d) };
    if d < 0 {
        n = -n;
        d = -d;
    }
    Ok((n, d))
}

#[inline]
fn mul(a: Coef, b: Coef) -> Result<Coef, String> {
    let n = a.0.checked_mul(b.0).ok_or("overflow")?;
    let d = a.1.checked_mul(b.1).ok_or("overflow")?;
    reduce(n, d)
}

#[inline]
fn add(a: Coef, b: Coef) -> Result<Coef, String> {
    if a.1 == b.1 {
        return reduce(a.0.checked_add(b.0).ok_or("overflow")?, a.1);
    }
    let n = a.0.checked_mul(b.1).ok_or("overflow")?.checked_add(b.0.checked_mul(a.1).ok_or("overflow")?).ok_or("overflow")?;
    let d = a.1.checked_mul(b.1).ok_or("overflow")?;
    reduce(n, d)
}

fn check_deadline(deadline: Option<Instant>) -> Result<(), String> {
    if let Some(d) = deadline {
        if Instant::now() > d {
            return Err("__timeout__".into());
        }
    }
    Ok(())
}

fn merge_into(into: &mut FxHashMap<Exps, Coef>, from: FxHashMap<Exps, Coef>) -> Result<(), String> {
    for (k, c) in from {
        match into.get_mut(&k) {
            Some(e) => {
                *e = add(*e, c)?;
            }
            None => {
                into.insert(k, c);
            }
        }
    }
    Ok(())
}

/// P_{k-1} x itotal with the t-bound, divided by k.
fn step(prev: &[(Exps, Coef)], itotal: &[(Exps, Coef)], t_limit: i32, k: i128, deadline: Option<Instant>) -> Result<FxHashMap<Exps, Coef>, String> {
    let chunk = ((prev.len() + 63) / 64).max(64);
    let parts: Vec<FxHashMap<Exps, Coef>> = prev
        .par_chunks(chunk)
        .map(|rows| -> Result<FxHashMap<Exps, Coef>, String> {
            check_deadline(deadline)?;
            let mut local: FxHashMap<Exps, Coef> = FxHashMap::default();
            let width = itotal.first().map(|(e, _)| e.len()).unwrap_or(0);
            let mut key: Exps = vec![0; width];
            for (ep, cp) in rows {
                let bound = t_limit - ep[0];
                for (ei, ci) in itotal {
                    if ei[0] > bound {
                        break;
                    }
                    for j in 0..width {
                        key[j] = ep[j] + ei[j];
                    }
                    let c = mul(*cp, *ci)?;
                    match local.get_mut(&key) {
                        Some(e) => {
                            *e = add(*e, c)?;
                        }
                        None => {
                            local.insert(key.clone(), c);
                        }
                    }
                }
            }
            Ok(local)
        })
        .collect::<Result<Vec<_>, String>>()?;
    // merge pairwise in parallel, then divide by k
    let merged = parts
        .into_par_iter()
        .map(Ok)
        .reduce(|| Ok(FxHashMap::default()), |a: Result<FxHashMap<Exps, Coef>, String>, b| {
            let mut a = a?;
            let b = b?;
            if a.len() < b.len() {
                let mut b = b;
                merge_into(&mut b, a)?;
                Ok(b)
            } else {
                merge_into(&mut a, b)?;
                Ok(a)
            }
        })?;
    let mut out: FxHashMap<Exps, Coef> = FxHashMap::default();
    out.reserve(merged.len());
    for (key, c) in merged {
        if c.0 == 0 {
            continue;
        }
        let d = c.1.checked_mul(k).ok_or("overflow")?;
        let r = reduce(c.0, d)?;
        out.insert(key, r);
    }
    Ok(out)
}

/// The truncated exponential: {monomial: coefficient}, the constant term included.
fn exponential(terms: &[(Exps, Coef)], t_limit: i32, max_order: usize, deadline: Option<Instant>) -> Result<FxHashMap<Exps, Coef>, String> {
    let trace = std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok();
    let mut itotal: Vec<(Exps, Coef)> = terms.iter().filter(|(e, _)| e[0] <= t_limit).cloned().collect();
    itotal.sort_by_key(|(e, _)| e[0]);
    let width = itotal.first().map(|(e, _)| e.len()).unwrap_or(0);
    let mut result: FxHashMap<Exps, Coef> = FxHashMap::default();
    result.insert(vec![0; width], (1, 1));
    let mut power: Vec<(Exps, Coef)> = itotal.clone();
    for k in 1..=max_order {
        if k > 1 {
            let t0 = Instant::now();
            let next = step(&power, &itotal, t_limit, k as i128, deadline)?;
            power = next.into_iter().collect();
            if trace {
                eprintln!("[series] k={k}: {} terms, {:.3} s", power.len(), t0.elapsed().as_secs_f64());
            }
            if power.is_empty() {
                break;
            }
        }
        for (e, c) in &power {
            match result.get_mut(e) {
                Some(x) => {
                    *x = add(*x, *c)?;
                }
                None => {
                    result.insert(e.clone(), *c);
                }
            }
        }
        check_deadline(deadline)?;
    }
    result.retain(|_, c| c.0 != 0);
    Ok(result)
}

#[inline]
fn milli_exponent(tpow: i64, spow: i64, rpow: i64) -> i128 {
    let n: i128 = 25_000_000i128 * tpow as i128 + 5_000i128 * spow as i128 + rpow as i128;
    let d: i128 = 12_500_000;
    if n >= 0 {
        (n + d / 2) / d
    } else {
        -((-n + d / 2) / d)
    }
}

fn timeout_error(py: Python<'_>, timeout: Option<f64>) -> PyErr {
    match py.import("subprocess").and_then(|m| m.getattr("TimeoutExpired")).and_then(|cls| cls.call1(("landscape_native.expand_series", timeout.unwrap_or(0.0)))) {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

/// expand_series(terms, t_limit, max_order, n_fields, slots, n_nodes, t_order, lookup, timeout, monomials)
#[pyfunction]
#[pyo3(signature = (terms, t_limit, max_order, n_fields, slots, n_nodes, t_order, lookup, timeout=None, monomials=false))]
#[allow(clippy::too_many_arguments)]
pub fn expand_series<'py>(
    py: Python<'py>,
    terms: Vec<(i128, i128, Vec<i32>)>,
    t_limit: i32,
    max_order: usize,
    n_fields: usize,
    slots: Vec<(usize, Vec<i64>, i64)>,
    n_nodes: usize,
    t_order: i64,
    lookup: Bound<'py, PyAny>,
    timeout: Option<f64>,
    monomials: bool,
) -> PyResult<Bound<'py, PyList>> {
    let deadline = timeout.map(|t| Instant::now() + std::time::Duration::from_secs_f64(t.max(0.0)));
    let width = 4 + n_fields + slots.len();
    let mut series: Vec<(Exps, Coef)> = Vec::with_capacity(terms.len());
    for (n, d, e) in terms {
        if e.len() != width {
            return Err(PyValueError::new_err("exponent vector of the wrong length"));
        }
        series.push((e, reduce(n, d).map_err(PyValueError::new_err)?));
    }
    let t_exp = Instant::now();
    let result = py.allow_threads(|| exponential(&series, t_limit, max_order, deadline));
    if std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok() {
        eprintln!("[series] exponential {:.3} s", t_exp.elapsed().as_secs_f64());
    }
    let result = match result {
        Ok(r) => r,
        Err(m) if m == "__timeout__" => return Err(timeout_error(py, timeout)),
        Err(m) => return Err(PyValueError::new_err(m)),
    };
    let out = PyList::empty(py);
    if monomials {
        let mut rows: Vec<(&Exps, &Coef)> = result.iter().collect();
        rows.sort();
        for (e, c) in rows {
            out.append((c.0, c.1, PyList::new(py, e.iter().copied())?))?;
        }
        return Ok(out);
    }
    // the rows of the field-resolved expansion
    let limit: i128 = 1000i128 * t_order as i128;
    let mut mults: FxHashMap<Vec<i32>, i128> = FxHashMap::default();
    let mut acc: FxHashMap<(i128, i64, Vec<i64>), Coef> = FxHashMap::default();
    for (e, c) in result.iter() {
        let milli = milli_exponent(e[0] as i64, e[1] as i64, e[2] as i64);
        if milli > limit {
            continue;
        }
        let cexps: Vec<i32> = e[4 + n_fields..].to_vec();
        let mult: i128 = if cexps.iter().all(|x| *x == 0) {
            1
        } else if let Some(m) = mults.get(&cexps) {
            *m
        } else {
            // per node: label -> {k: m}, then the sorted (label, key vector) tuples of parse_term
            let mut per_node: Vec<BTreeMap<Vec<i64>, BTreeMap<i64, i64>>> = (0..n_nodes).map(|_| BTreeMap::new()).collect();
            for (j, x) in cexps.iter().enumerate() {
                if *x != 0 {
                    let (node, label, k) = &slots[j];
                    *per_node[*node].entry(label.clone()).or_default().entry(*k).or_insert(0) += *x as i64;
                }
            }
            let mut chars: Vec<crate::NodeChars> = Vec::with_capacity(n_nodes);
            for per_label in per_node.iter() {
                let mut node: crate::NodeChars = Vec::new();
                for (label, adams) in per_label.iter() {
                    if adams.values().any(|m| *m != 0) {
                        node.push((label.clone(), crate::key_vec_of_map(adams).map_err(PyValueError::new_err)?));
                    }
                }
                node.sort();
                chars.push(node);
            }
            let obj = crate::chars_object(py, &chars)?;
            let m: i128 = lookup.call1((obj,))?.extract()?;
            mults.insert(cexps.clone(), m);
            m
        };
        if mult == 0 {
            continue;
        }
        let markers: Vec<i64> = e[4..4 + n_fields].iter().map(|x| *x as i64).collect();
        let cm = mul(*c, (mult, 1)).map_err(PyValueError::new_err)?;
        let key = (milli, e[3] as i64, markers);
        match acc.get_mut(&key) {
            Some(x) => {
                *x = add(*x, cm).map_err(PyValueError::new_err)?;
            }
            None => {
                acc.insert(key, cm);
            }
        }
    }
    let mut rows: Vec<((i128, i64, Vec<i64>), Coef)> = acc.into_iter().filter(|(_, c)| c.0 != 0).collect();
    rows.sort();
    for ((milli, ypow, markers), (num, den)) in rows {
        if den != 1 {
            return Err(PyValueError::new_err(format!("non-integral coefficient {num}/{den} at t^{} y^{ypow} {markers:?}", milli as f64 / 1000.0)));
        }
        let m = PyTuple::new(py, markers.iter().copied())?;
        out.append((num, den, milli, ypow, m))?;
    }
    Ok(out)
}
