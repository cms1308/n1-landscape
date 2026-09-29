//! The native expansion engine (step 39f; refined in step 44): the exponential of the explicit
//! itotal of `landscape.form.program`, truncated at the t-power limit and at `max_order` powers,
//! in place of the FORM run -- the same polynomial FORM prints as `result`.
//!
//! A monomial is an exponent vector [t, s, r, y, f_1..f_n, c_1..c_m] (`form.Series`); the
//! product of two monomials adds the vectors; a coefficient is a reduced 128-bit rational, and
//! when one overflows the same expansion runs again with arbitrary-precision rationals (step 54;
//! the factors 1/k give denominators of the order of k!, beyond 128 bits above an expansion order
//! of about 25-34).  With P_1 = itotal and
//! P_k = P_{k-1} itotal / k, the result is 1 + sum_k P_k for k = 1..max_order, every product
//! whose t-power exceeds the limit dropped before it is formed (the terms of itotal sorted by
//! t-power, the multiplication of a term of P_{k-1} stopping at the bound) -- the products FORM's
//! bounded loop generates and keeps, and none of the ones it discards.  The products of the
//! terms of P_{k-1} run in parallel (rayon), the partial maps merged; a deadline is checked
//! inside the loops and raises subprocess.TimeoutExpired; a power beyond the monomial cap raises
//! "capacity" and the caller falls back to FORM.
//!
//! The refinements of step 44, each a parameter, off by default:
//!   * `basis` (the flavor projection above t^6): a monomial whose milli exponent -- from its
//!     t, s, r sum, the rule of the rows -- exceeds `project_above` (6000) holds the flavor
//!     exponents basis . markers in place of the markers, [t, s, r, y, x_1..x_rank, c_1..c_m].
//!     The projection is linear on the exponents, so a product is projected as soon as its milli
//!     crosses the bound and a projected factor projects the product; the terms of itotal and of
//!     every power are kept in two lists, field-resolved and flavor-refined, and the rows come in
//!     the two kinds (`post::emit_rows`).  Weights are positive, so a field-resolved product arises
//!     from field-resolved factors only: the field-resolved part is complete through t^6.
//!   * `coef64`: the coefficient arithmetic without a 128-bit division where both operands fit
//!     in 62 bits (a binary gcd on 64-bit integers) and without any where a denominator is one;
//!     the same reduced rationals by another route.
//!   * `exact`: a product whose milli exceeds `milli_limit` (1000 t_order) is not formed; the
//!     integer t-power bound stays the outer filter (a monomial above the limit contributes to no
//!     row at or below it, weights being positive).
//!
//! `expand_series(..., monomials=true)` returns the polynomial itself, (numerator,
//! denominator, exponents) per monomial, for the comparison with FORM's parsed output (without a
//! basis and without `exact`, the polynomial FORM prints); `monomials=false` returns the rows of
//! the field-resolved expansion as `expand` does (as a `FieldResolvedRows` object when
//! `native_rows` is set; with a basis, the object only): the terms above t_order dropped by their
//! milli exponent, the singlet multiplicity of every distinct character product obtained through
//! the projector's `multiplicity`, coefficient x multiplicity summed per (milli, y power, markers)
//! or per (milli, y power, flavor).

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

type Exps = Vec<i32>;
type Coef = (i128, i128);
type Term<C> = (Exps, C);
type Map<C> = FxHashMap<Exps, C>;

/// The overflow of a 128-bit coefficient: the expansion runs again with `Big`.
const COEF_OVERFLOW: &str = "coefficient overflow";

// --------------------------------------------------------------------------- //
// coefficients
// --------------------------------------------------------------------------- //
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

/// Binary gcd on 64-bit integers (no division).
#[inline]
fn gcd64(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

#[inline]
fn fits62(x: i128) -> bool {
    x.unsigned_abs() < (1u128 << 62)
}

#[inline]
fn reduce(n: i128, d: i128, fast: bool) -> Result<Coef, String> {
    if d == 0 {
        return Err("zero denominator".into());
    }
    if fast {
        if d == 1 {
            return Ok((n, 1));
        }
        if n == 0 {
            return Ok((0, 1));
        }
        if fits62(n) && fits62(d) {
            let g = gcd64(n.unsigned_abs() as u64, d.unsigned_abs() as u64) as i128;
            let (mut n, mut d) = if g > 1 { (n / g, d / g) } else { (n, d) };
            if d < 0 {
                n = -n;
                d = -d;
            }
            return Ok((n, d));
        }
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
fn mul(a: Coef, b: Coef, fast: bool) -> Result<Coef, String> {
    let n = a.0.checked_mul(b.0).ok_or("overflow")?;
    let d = a.1.checked_mul(b.1).ok_or("overflow")?;
    if fast && d == 1 {
        return Ok((n, 1));
    }
    reduce(n, d, fast)
}

#[inline]
fn add(a: Coef, b: Coef, fast: bool) -> Result<Coef, String> {
    if a.1 == b.1 {
        return reduce(a.0.checked_add(b.0).ok_or("overflow")?, a.1, fast);
    }
    let n = a.0.checked_mul(b.1).ok_or("overflow")?.checked_add(b.0.checked_mul(a.1).ok_or("overflow")?).ok_or("overflow")?;
    let d = a.1.checked_mul(b.1).ok_or("overflow")?;
    reduce(n, d, fast)
}

/// The coefficient arithmetic of the engine: a reduced rational of two `i128` (`Coef`, checked, an
/// overflow raising COEF_OVERFLOW) or of two `BigInt` (`Big`).  `fast` is the `coef64` route of the
/// 128-bit arithmetic (step 44); `Big` ignores it.
pub(crate) trait Coeff: Clone + Send + Sync {
    fn from_pair(n: i128, d: i128) -> Result<Self, String>;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn times(&self, b: &Self, fast: bool) -> Result<Self, String>;
    fn plus(&self, b: &Self, fast: bool) -> Result<Self, String>;
    fn over(&self, k: i128, fast: bool) -> Result<Self, String>;
    fn times_int(&self, m: i128) -> Result<Self, String>;
    /// The value as two `i128`, or COEF_OVERFLOW.
    fn pair(&self) -> Result<Coef, String>;
    fn big(&self) -> (BigInt, BigInt);
}

fn coef_overflow<E>(_: E) -> String {
    COEF_OVERFLOW.to_string()
}

impl Coeff for Coef {
    #[inline]
    fn from_pair(n: i128, d: i128) -> Result<Self, String> {
        reduce(n, d, false)
    }
    #[inline]
    fn one() -> Self {
        (1, 1)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    #[inline]
    fn times(&self, b: &Self, fast: bool) -> Result<Self, String> {
        mul(*self, *b, fast).map_err(coef_overflow)
    }
    #[inline]
    fn plus(&self, b: &Self, fast: bool) -> Result<Self, String> {
        add(*self, *b, fast).map_err(coef_overflow)
    }
    #[inline]
    fn over(&self, k: i128, fast: bool) -> Result<Self, String> {
        let d = self.1.checked_mul(k).ok_or(COEF_OVERFLOW)?;
        reduce(self.0, d, fast)
    }
    #[inline]
    fn times_int(&self, m: i128) -> Result<Self, String> {
        mul(*self, (m, 1), false).map_err(coef_overflow)
    }
    #[inline]
    fn pair(&self) -> Result<Coef, String> {
        Ok(*self)
    }
    fn big(&self) -> (BigInt, BigInt) {
        (BigInt::from(self.0), BigInt::from(self.1))
    }
}

/// An arbitrary-precision reduced rational (denominator positive).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Big {
    n: BigInt,
    d: BigInt,
}

impl Big {
    fn reduced(mut n: BigInt, mut d: BigInt) -> Result<Big, String> {
        if d.is_zero() {
            return Err("zero denominator".into());
        }
        if n.is_zero() {
            return Ok(Big { n, d: BigInt::one() });
        }
        let g = n.gcd(&d);
        if !g.is_one() {
            n /= &g;
            d /= &g;
        }
        if d.is_negative() {
            n = -n;
            d = -d;
        }
        Ok(Big { n, d })
    }
}

impl Coeff for Big {
    fn from_pair(n: i128, d: i128) -> Result<Self, String> {
        Big::reduced(BigInt::from(n), BigInt::from(d))
    }
    fn one() -> Self {
        Big { n: BigInt::one(), d: BigInt::one() }
    }
    fn is_zero(&self) -> bool {
        self.n.is_zero()
    }
    fn times(&self, b: &Self, _fast: bool) -> Result<Self, String> {
        Big::reduced(&self.n * &b.n, &self.d * &b.d)
    }
    fn plus(&self, b: &Self, _fast: bool) -> Result<Self, String> {
        if self.d == b.d {
            return Big::reduced(&self.n + &b.n, self.d.clone());
        }
        Big::reduced(&self.n * &b.d + &b.n * &self.d, &self.d * &b.d)
    }
    fn over(&self, k: i128, _fast: bool) -> Result<Self, String> {
        Big::reduced(self.n.clone(), &self.d * BigInt::from(k))
    }
    fn times_int(&self, m: i128) -> Result<Self, String> {
        Big::reduced(&self.n * BigInt::from(m), self.d.clone())
    }
    fn pair(&self) -> Result<Coef, String> {
        match (self.n.to_i128(), self.d.to_i128()) {
            (Some(n), Some(d)) => Ok((n, d)),
            _ => Err(COEF_OVERFLOW.to_string()),
        }
    }
    fn big(&self) -> (BigInt, BigInt) {
        (self.n.clone(), self.d.clone())
    }
}

fn check_deadline(deadline: Option<Instant>) -> Result<(), String> {
    if let Some(d) = deadline {
        if Instant::now() > d {
            return Err("__timeout__".into());
        }
    }
    Ok(())
}

fn merge_into<C: Coeff>(into: &mut Map<C>, from: Map<C>, fast: bool) -> Result<(), String> {
    for (k, c) in from {
        match into.get_mut(&k) {
            Some(e) => {
                *e = e.plus(&c, fast)?;
            }
            None => {
                into.insert(k, c);
            }
        }
    }
    Ok(())
}

/// The monomial cap of the engine (LANDSCAPE_NATIVE_MAX_TERMS, default 8,000,000): a power of the
/// series, or the truncated exponential, with more monomials than this raises "__capacity__" and
/// the caller falls back to FORM, which sorts on disk.  Found on the June-2026 Sp2nf5 landscape
/// (a theory of 25 fields and expansion order 38: 10 M monomials at k = 5 and 20 GB at k = 6;
/// FORM had computed it), whose orders lie above the 18 of the engine's evidence sets.
fn capacity() -> usize {
    std::env::var("LANDSCAPE_NATIVE_MAX_TERMS").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or(8_000_000)
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

/// The scaled weight 25,000,000 t + 5,000 s + r of a monomial (milli = floor((n + 6,250,000) / 12,500,000));
/// additive under products, so a product's milli is classified by one comparison of the sum with a
/// threshold (no division per product).
#[inline]
fn weight_n(e: &[i32]) -> i64 {
    25_000_000 * e[0] as i64 + 5_000 * e[1] as i64 + e[2] as i64
}

/// The smallest scaled weight whose milli exceeds `milli`.
#[inline]
fn threshold_n(milli: i64) -> i64 {
    (milli + 1) * 12_500_000 - 6_250_000
}

// --------------------------------------------------------------------------- //
// the layout of the monomials
// --------------------------------------------------------------------------- //
/// The exponent layouts: field-resolved [t, s, r, y, f_1..f_n, c_1..c_m] and, with a basis,
/// flavor-refined [t, s, r, y, x_1..x_rank, c_1..c_m]; the parameters of the refinements.
struct Layout {
    n_fields: usize,
    n_slots: usize,
    basis: Option<Vec<Vec<i64>>>,
    rank: usize,
    project_above: i64,
    above_n: i64,       // threshold_n(project_above)
    limit_n: i64,       // threshold_n(milli_limit)
    exact: bool,
    fast: bool,
}

impl Layout {
    #[inline]
    fn classify(&self) -> bool {
        self.basis.is_some() || self.exact
    }

    /// basis . markers of a field-resolved monomial.
    fn flavor(&self, e: &[i32]) -> Result<Vec<i32>, String> {
        let basis = self.basis.as_ref().expect("a basis");
        let mut out = Vec::with_capacity(self.rank);
        for row in basis {
            let mut s: i64 = 0;
            for (b, x) in row.iter().zip(e[4..4 + self.n_fields].iter()) {
                s = s.checked_add(b.checked_mul(*x as i64).ok_or("overflow")?).ok_or("overflow")?;
            }
            out.push(i32::try_from(s).map_err(|_| "flavor exponent out of range".to_string())?);
        }
        Ok(out)
    }

    /// A field-resolved term as a flavor-refined one.
    fn to_flavor(&self, e: &[i32]) -> Result<Exps, String> {
        let mut out = Vec::with_capacity(4 + self.rank + self.n_slots);
        out.extend_from_slice(&e[..4]);
        out.extend(self.flavor(e)?);
        out.extend_from_slice(&e[4 + self.n_fields..]);
        Ok(out)
    }
}

/// The two lists of a power (or of itotal): field-resolved terms and flavor-refined terms, the
/// former with their flavor images (computed once per term) when a basis is set.
struct Power<C> {
    fr: Vec<Term<C>>,
    fr_flavor: Vec<Exps>,
    fr_n: Vec<i64>,      // the scaled weights of the field-resolved terms (the letters; a power's are computed per row)
    fl: Vec<Term<C>>,
    fl_n: Vec<i64>,
}

struct Maps<C> {
    fr: Map<C>,
    fl: Map<C>,
}

impl<C> Maps<C> {
    fn new() -> Self {
        Maps { fr: FxHashMap::default(), fl: FxHashMap::default() }
    }
    fn len(&self) -> usize {
        self.fr.len() + self.fl.len()
    }
}

#[inline]
fn insert<C: Coeff>(map: &mut Map<C>, key: &Exps, c: C, fast: bool) -> Result<(), String> {
    match map.get_mut(key) {
        Some(e) => {
            *e = e.plus(&c, fast)?;
        }
        None => {
            map.insert(key.clone(), c);
        }
    }
    Ok(())
}

/// The products of one chunk of terms of P_{k-1} (of one kind) with itotal.  `live` counts the
/// monomials held by every chunk of the step so far; past twice the cap the step stops with
/// "__capacity__" before the merge (the guard of Q-6.6, on the sum instead of a per-chunk share:
/// the field-resolved and flavor-refined lists of the projection make the chunks unequal).
#[allow(clippy::too_many_arguments)]
fn products<C: Coeff>(rows: &[Term<C>], rows_flavor: &[Exps], rows_fr: bool, itotal: &Power<C>, layout: &Layout, t_limit: i32,
                      deadline: Option<Instant>, cap: usize, live: &AtomicUsize) -> Result<Maps<C>, String> {
    check_deadline(deadline)?;
    let mut reported = 0usize;
    let n = layout.n_fields;
    let r = layout.rank;
    let m = layout.n_slots;
    let fast = layout.fast;
    let mut local = Maps::new();
    let width_fr = 4 + n + m;
    let width_fl = 4 + r + m;
    let mut key_fr: Exps = vec![0; width_fr];
    let mut key_fl: Exps = vec![0; width_fl];
    for (row, (ep, cp)) in rows.iter().enumerate() {
        if row % 256 == 255 {
            check_deadline(deadline)?;
            let held = local.len();
            if held > reported {
                live.fetch_add(held - reported, Ordering::Relaxed);
                reported = held;
            }
            if live.load(Ordering::Relaxed) > 2 * cap {
                return Err("__capacity__".into());
            }
        }
        let bound = t_limit - ep[0];
        let n_ep = weight_n(ep);
        let ep_flavor: &[i32] = if rows_fr { if layout.basis.is_some() { &rows_flavor[row] } else { &[] } } else { &ep[4..4 + r] };
        let ep_chars: &[i32] = if rows_fr { &ep[4 + n..] } else { &ep[4 + r..] };
        // x field-resolved letters
        for (j, (ei, ci)) in itotal.fr.iter().enumerate() {
            if ei[0] > bound {
                break;
            }
            let mut n_p = 0i64;
            if layout.classify() {
                n_p = n_ep + itotal.fr_n[j];
                if layout.exact && n_p >= layout.limit_n {
                    continue;
                }
            }
            let c = cp.times(ci, fast)?;
            if rows_fr && (layout.basis.is_none() || n_p < layout.above_n) {
                for j in 0..width_fr {
                    key_fr[j] = ep[j] + ei[j];
                }
                insert(&mut local.fr, &key_fr, c, fast)?;
            } else {
                for j in 0..4 {
                    key_fl[j] = ep[j] + ei[j];
                }
                let ei_flavor = &itotal.fr_flavor[j];
                for j in 0..r {
                    key_fl[4 + j] = ep_flavor[j] + ei_flavor[j];
                }
                let ei_chars = &ei[4 + n..];
                for j in 0..m {
                    key_fl[4 + r + j] = ep_chars[j] + ei_chars[j];
                }
                insert(&mut local.fl, &key_fl, c, fast)?;
            }
        }
        // x flavor-refined letters (a basis only)
        for (j, (ei, ci)) in itotal.fl.iter().enumerate() {
            if ei[0] > bound {
                break;
            }
            if layout.exact && n_ep + itotal.fl_n[j] >= layout.limit_n {
                continue;
            }
            let c = cp.times(ci, fast)?;
            for j in 0..width_fl {
                key_fl[j] = if j < 4 { ep[j] + ei[j] } else if j < 4 + r { ep_flavor[j - 4] + ei[j] } else { ep_chars[j - 4 - r] + ei[j] };
            }
            insert(&mut local.fl, &key_fl, c, fast)?;
        }
    }
    Ok(local)
}

/// P_{k-1} x itotal with the t-bound, divided by k.  The deadline is checked every 256 rows of
/// every chunk and at every merge; the chunks together past twice the cap during the step, or
/// past the cap after it, stop with "__capacity__" before the merge.
fn step<C: Coeff>(prev: &Power<C>, itotal: &Power<C>, layout: &Layout, t_limit: i32, k: i128, deadline: Option<Instant>,
                  cap: usize) -> Result<Maps<C>, String> {
    let total = prev.fr.len() + prev.fl.len();
    let chunk = ((total + 63) / 64).max(64);
    let fast = layout.fast;
    let live = AtomicUsize::new(0);
    let parts_fr: Vec<Maps<C>> = prev.fr
        .par_chunks(chunk)
        .enumerate()
        .map(|(i, rows)| {
            let flav: &[Exps] = if layout.basis.is_some() { &prev.fr_flavor[i * chunk..i * chunk + rows.len()] } else { &[] };
            products(rows, flav, true, itotal, layout, t_limit, deadline, cap, &live)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let parts_fl: Vec<Maps<C>> = prev.fl
        .par_chunks(chunk)
        .map(|rows| products(rows, &[], false, itotal, layout, t_limit, deadline, cap, &live))
        .collect::<Result<Vec<_>, String>>()?;
    let mut parts = parts_fr;
    parts.extend(parts_fl);
    if parts.iter().map(|p| p.len()).sum::<usize>() > cap {
        return Err("__capacity__".into());
    }
    // merge pairwise in parallel, then divide by k
    let merged = parts
        .into_par_iter()
        .map(Ok)
        .reduce(|| Ok(Maps::new()), |a: Result<Maps<C>, String>, b| {
            check_deadline(deadline)?;
            let mut a = a?;
            let mut b = b?;
            if a.len() < b.len() {
                merge_into(&mut b.fr, a.fr, fast)?;
                merge_into(&mut b.fl, a.fl, fast)?;
                Ok(b)
            } else {
                merge_into(&mut a.fr, b.fr, fast)?;
                merge_into(&mut a.fl, b.fl, fast)?;
                Ok(a)
            }
        })?;
    let mut out = Maps::new();
    for (src, dst) in [(merged.fr, &mut out.fr), (merged.fl, &mut out.fl)] {
        dst.reserve(src.len());
        for (key, c) in src {
            if c.is_zero() {
                continue;
            }
            dst.insert(key, c.over(k, fast)?);
        }
    }
    Ok(out)
}

fn power_of<C: Coeff>(maps: Maps<C>, layout: &Layout) -> Result<Power<C>, String> {
    let fr: Vec<Term<C>> = maps.fr.into_iter().collect();
    let fr_flavor = if layout.basis.is_some() { fr.iter().map(|(e, _)| layout.flavor(e)).collect::<Result<Vec<_>, String>>()? } else { Vec::new() };
    Ok(Power { fr, fr_flavor, fr_n: Vec::new(), fl: maps.fl.into_iter().collect(), fl_n: Vec::new() })
}

/// The truncated exponential: {monomial: coefficient} of the two kinds, the constant term included.
fn exponential<C: Coeff>(terms: &[Term<C>], t_limit: i32, max_order: usize, layout: &Layout, deadline: Option<Instant>) -> Result<Maps<C>, String> {
    let trace = std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok();
    let fast = layout.fast;
    // itotal: the terms within the limit, sorted by t-power, in two lists when a basis is set
    let mut itotal_fr: Vec<Term<C>> = Vec::new();
    let mut itotal_fl: Vec<Term<C>> = Vec::new();
    for (e, c) in terms.iter().filter(|(e, _)| e[0] <= t_limit) {
        let above = layout.basis.is_some() && milli_exponent(e[0] as i64, e[1] as i64, e[2] as i64) as i64 > layout.project_above;
        if above {
            itotal_fl.push((layout.to_flavor(e)?, c.clone()));
        } else {
            itotal_fr.push((e.clone(), c.clone()));
        }
    }
    itotal_fr.sort_by_key(|(e, _)| e[0]);
    itotal_fl.sort_by_key(|(e, _)| e[0]);
    let itotal_fr_flavor = if layout.basis.is_some() { itotal_fr.iter().map(|(e, _)| layout.flavor(e)).collect::<Result<Vec<_>, String>>()? } else { Vec::new() };
    let itotal_fr_n: Vec<i64> = itotal_fr.iter().map(|(e, _)| weight_n(e)).collect();
    let itotal_fl_n: Vec<i64> = itotal_fl.iter().map(|(e, _)| weight_n(e)).collect();
    let itotal = Power { fr: itotal_fr, fr_flavor: itotal_fr_flavor, fr_n: itotal_fr_n, fl: itotal_fl, fl_n: itotal_fl_n };
    let width_fr = 4 + layout.n_fields + layout.n_slots;
    let mut result = Maps::new();
    result.fr.insert(vec![0; width_fr], C::one());
    let mut power = Power { fr: itotal.fr.clone(), fr_flavor: itotal.fr_flavor.clone(), fr_n: Vec::new(), fl: itotal.fl.clone(), fl_n: Vec::new() };
    let cap = capacity();
    for k in 1..=max_order {
        if k > 1 {
            let t0 = Instant::now();
            let next = step(&power, &itotal, layout, t_limit, k as i128, deadline, cap)?;
            power = power_of(next, layout)?;
            if trace {
                eprintln!("[series] k={k}: {} + {} terms, {:.3} s", power.fr.len(), power.fl.len(), t0.elapsed().as_secs_f64());
            }
            if power.fr.is_empty() && power.fl.is_empty() {
                break;
            }
            let n = power.fr.len() + power.fl.len();
            if n > cap || result.len() + n > cap {
                return Err("__capacity__".into());
            }
        }
        for (rows, dst) in [(&power.fr, &mut result.fr), (&power.fl, &mut result.fl)] {
            for (row, (e, c)) in rows.iter().enumerate() {
                if row % 65536 == 65535 {
                    check_deadline(deadline)?;
                }
                match dst.get_mut(e) {
                    Some(x) => {
                        *x = x.plus(c, fast)?;
                    }
                    None => {
                        dst.insert(e.clone(), c.clone());
                    }
                }
            }
        }
        check_deadline(deadline)?;
    }
    result.fr.retain(|_, c| !c.is_zero());
    result.fl.retain(|_, c| !c.is_zero());
    Ok(result)
}

fn timeout_error(py: Python<'_>, timeout: Option<f64>) -> PyErr {
    match py.import("subprocess").and_then(|m| m.getattr("TimeoutExpired")).and_then(|cls| cls.call1(("landscape_native.expand_series", timeout.unwrap_or(0.0)))) {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
}

/// The character product of a monomial's slot exponents as the projector's tuple, for the lookup.
fn chars_of<'py>(py: Python<'py>, cexps: &[i32], slots: &[(usize, Vec<i64>, i64)], n_nodes: usize) -> PyResult<Bound<'py, PyAny>> {
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
    crate::chars_object(py, &chars)
}

/// expand_series(terms, t_limit, max_order, n_fields, slots, n_nodes, t_order, lookup, timeout, monomials, native_rows,
///               basis, exact, coef64): the rows as a FieldResolvedRows object when native_rows is set (the native
/// post-processing pass reads them there; with a basis, the only form); see the module.
#[pyfunction]
#[pyo3(signature = (terms, t_limit, max_order, n_fields, slots, n_nodes, t_order, lookup, timeout=None, monomials=false, native_rows=false,
                    basis=None, exact=false, coef64=false))]
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
    native_rows: bool,
    basis: Option<Vec<Vec<i64>>>,
    exact: bool,
    coef64: bool,
) -> PyResult<Py<PyAny>> {
    let deadline = timeout.map(|t| Instant::now() + std::time::Duration::from_secs_f64(t.max(0.0)));
    let width = 4 + n_fields + slots.len();
    if basis.is_some() && (monomials || !native_rows) {
        return Err(PyValueError::new_err("a flavor basis needs native rows (the rows above t^6 are flavor-refined)"));
    }
    for (_, _, e) in terms.iter() {
        if e.len() != width {
            return Err(PyValueError::new_err("exponent vector of the wrong length"));
        }
    }
    let limit: i64 = 1000 * t_order;
    let layout = Layout { n_fields, n_slots: slots.len(), rank: basis.as_ref().map(|b| b.len()).unwrap_or(0), basis,
                          project_above: 6000, above_n: threshold_n(6000), limit_n: threshold_n(limit), exact, fast: coef64 };
    crate::lie::init_pool();
    let trace = std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok();
    let t_exp = Instant::now();
    // the 128-bit engine first; on a coefficient overflow the same expansion with arbitrary precision (step 54)
    let small = py.allow_threads(|| series_of::<Coef>(&terms).and_then(|s| exponential(&s, t_limit, max_order, &layout, deadline)));
    let out = match small {
        // the 128-bit rows' sums may overflow where the expansion did not: the rows again from the same expansion, exactly
        Ok(r) => match finish(py, &r, monomials, native_rows, &layout, &slots, n_nodes, n_fields, limit, &lookup) {
            Err(e) if e.to_string().contains(COEF_OVERFLOW) => {
                let r = to_big(&r).map_err(PyValueError::new_err)?;
                finish(py, &r, monomials, native_rows, &layout, &slots, n_nodes, n_fields, limit, &lookup)
            }
            other => other,
        },
        Err(m) if m == COEF_OVERFLOW => {
            if trace {
                eprintln!("[series] 128-bit overflow after {:.3} s: arbitrary precision", t_exp.elapsed().as_secs_f64());
            }
            let big = py.allow_threads(|| series_of::<Big>(&terms).and_then(|s| exponential(&s, t_limit, max_order, &layout, deadline)));
            match big {
                Ok(r) => finish(py, &r, monomials, native_rows, &layout, &slots, n_nodes, n_fields, limit, &lookup),
                Err(m) => Err(engine_error(py, m, timeout)),
            }
        }
        Err(m) => Err(engine_error(py, m, timeout)),
    };
    if trace {
        eprintln!("[series] exponential {:.3} s", t_exp.elapsed().as_secs_f64());
    }
    out
}

fn to_big<C: Coeff>(m: &Maps<C>) -> Result<Maps<Big>, String> {
    let conv = |src: &Map<C>| -> Result<Map<Big>, String> {
        let mut out: Map<Big> = FxHashMap::default();
        out.reserve(src.len());
        for (e, c) in src {
            let (n, d) = c.big();
            out.insert(e.clone(), Big::reduced(n, d)?);
        }
        Ok(out)
    };
    Ok(Maps { fr: conv(&m.fr)?, fl: conv(&m.fl)? })
}

fn series_of<C: Coeff>(terms: &[(i128, i128, Vec<i32>)]) -> Result<Vec<Term<C>>, String> {
    terms.iter().map(|(n, d, e)| Ok((e.clone(), C::from_pair(*n, *d)?))).collect()
}

fn engine_error(py: Python<'_>, m: String, timeout: Option<f64>) -> PyErr {
    if m == "__timeout__" {
        return timeout_error(py, timeout);
    }
    if m == "__capacity__" {
        return PyValueError::new_err(format!("capacity: the expansion exceeds {} monomials (LANDSCAPE_NATIVE_MAX_TERMS)", capacity()));
    }
    PyValueError::new_err(m)
}

/// The polynomial (monomials) or the rows of the two kinds from the truncated exponential.
#[allow(clippy::too_many_arguments)]
fn finish<'py, C: Coeff>(py: Python<'py>, result: &Maps<C>, monomials: bool, native_rows: bool, layout: &Layout,
                          slots: &[(usize, Vec<i64>, i64)], n_nodes: usize, n_fields: usize, limit: i64,
                          lookup: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    if monomials {
        let out = PyList::empty(py);
        let mut rows: Vec<(&Exps, &C)> = result.fr.iter().collect();
        rows.sort_by(|a, b| a.0.cmp(b.0));
        for (e, c) in rows {
            let (n, d) = c.big();
            out.append((n, d, PyList::new(py, e.iter().copied())?))?;
        }
        return Ok(out.into_any().unbind());
    }
    // the rows of the two kinds
    let rank = layout.rank;
    let mut mults: FxHashMap<Vec<i32>, i128> = FxHashMap::default();
    let mut acc_fr: FxHashMap<(i128, i64, Vec<i64>), C> = FxHashMap::default();
    let mut acc_fl: FxHashMap<(i128, i64, Vec<i64>), C> = FxHashMap::default();
    for (fr, map) in [(true, &result.fr), (false, &result.fl)] {
        let cstart = if fr { 4 + n_fields } else { 4 + rank };
        for (e, c) in map.iter() {
            let milli = milli_exponent(e[0] as i64, e[1] as i64, e[2] as i64);
            if milli > limit as i128 {
                continue;
            }
            let cexps: Vec<i32> = e[cstart..].to_vec();
            let mult: i128 = if cexps.iter().all(|x| *x == 0) {
                1
            } else if let Some(m) = mults.get(&cexps) {
                *m
            } else {
                let obj = chars_of(py, &cexps, slots, n_nodes)?;
                let m: i128 = lookup.call1((obj,))?.extract()?;
                mults.insert(cexps.clone(), m);
                m
            };
            if mult == 0 {
                continue;
            }
            let xs: Vec<i64> = e[4..cstart].iter().map(|x| *x as i64).collect();
            let cm = c.times_int(mult).map_err(PyValueError::new_err)?;
            let key = (milli, e[3] as i64, xs);
            let acc = if fr { &mut acc_fr } else { &mut acc_fl };
            match acc.get_mut(&key) {
                Some(x) => {
                    *x = x.plus(&cm, false).map_err(PyValueError::new_err)?;
                }
                None => {
                    acc.insert(key, cm);
                }
            }
        }
    }
    // integral after the projection: back to 128 bits (a row beyond raises the overflow; the caller's FORM path follows)
    let pairs = |acc: FxHashMap<(i128, i64, Vec<i64>), C>| -> PyResult<Vec<((i128, i64, Vec<i64>), Coef)>> {
        let mut rows = Vec::with_capacity(acc.len());
        for (k, c) in acc {
            if !c.is_zero() {
                rows.push((k, c.pair().map_err(PyValueError::new_err)?));
            }
        }
        rows.sort();
        Ok(rows)
    };
    let rows_fr = pairs(acc_fr)?;
    let rows_fl = pairs(acc_fl)?;
    crate::post::emit_rows(py, rows_fr, rows_fl, n_fields, layout.basis.clone(), native_rows)
}
