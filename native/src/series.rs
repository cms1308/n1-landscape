//! The native expansion engine (step 39f; refined in steps 44, 54 and 56b): the exponential of the
//! explicit itotal of `landscape.form.program`, truncated at the t-power limit and at `max_order`
//! powers, in place of the FORM run -- the same polynomial FORM prints as `result`, and from it the
//! rows of the field-resolved expansion.
//!
//! A monomial is an exponent vector [t, s, r, y, f_1..f_n, c_1..c_m] (`form.Series`); the product of
//! two monomials adds the vectors.  With P_1 = itotal and P_k = P_{k-1} itotal / k, the result is
//! 1 + sum_k P_k for k = 1..max_order, every product whose t-power exceeds the limit dropped before
//! it is formed (the terms of itotal sorted by t-power, the multiplication of a term of P_{k-1}
//! stopping at the bound) -- the products FORM's bounded loop generates and keeps, and none of the
//! ones it discards.  The products of the terms of P_{k-1} run in parallel (rayon), the partial maps
//! merged; a deadline is checked inside the loops and raises subprocess.TimeoutExpired; a power
//! beyond the monomial cap raises "capacity" and the caller falls back to FORM.
//!
//! Step 56b (extension 0.8.0), the memory of the engine:
//!   * the rows are accumulated from every power as it is formed; no map of the whole exponential
//!     is kept (the monomial mode, the comparison with FORM, keeps it), so the cap applies to one
//!     power;
//!   * a monomial is a fixed-width array of machine words: each exponent a bit field with an offset,
//!     its width from a bound valid for every monomial within the t-limit (a field is at most the
//!     t-limit times the largest ratio of that exponent to the t-power over the letters, and at most
//!     max_order times the largest letter), so that the product of two monomials is one addition and
//!     one subtraction per word with no carry between the fields;
//!   * a coefficient is a reduced rational of two `i128`, promoted in place to arbitrary precision
//!     when an operation overflows (step 54 ran the whole expansion a second time instead);
//!   * the singlet multiplicity of a character product is computed in the extension (`project.rs`),
//!     the Python projector (`lookup`) its fallback and, with LANDSCAPE_NATIVE_VERIFY_MULT=1, the
//!     check of every value.
//!
//! The refinements of step 44, each a parameter, off by default unless the caller sets it:
//!   * `basis` (the flavor projection above t^6): a monomial whose milli exponent -- from its
//!     t, s, r sum, the rule of the rows -- exceeds `project_above` (6000) holds the flavor
//!     exponents basis . markers in place of the markers, [t, s, r, y, x_1..x_rank, c_1..c_m].
//!     The projection is linear on the exponents, so a product is projected as soon as its milli
//!     crosses the bound and a projected factor projects the product; the terms of itotal and of
//!     every power are kept in two lists, field-resolved and flavor-refined, and the rows come in
//!     the two kinds (`post::emit_rows`).  Weights are positive, so a field-resolved product arises
//!     from field-resolved factors only: the field-resolved part is complete through t^6.
//!   * `coef64`: the 128-bit arithmetic without a 128-bit division where both operands fit in 62
//!     bits (a binary gcd on 64-bit integers) and without any where a denominator is one.
//!   * `exact`: a product whose milli exceeds `milli_limit` (1000 t_order) is not formed; the
//!     integer t-power bound stays the outer filter.
//!
//! `expand_series(..., monomials=true)` returns the polynomial itself, (numerator, denominator,
//! exponents) per monomial, for the comparison with FORM's parsed output (without a basis and
//! without `exact`, the polynomial FORM prints); `monomials=false` returns the rows of the
//! field-resolved expansion (as a `FieldResolvedRows` object when `native_rows` is set; with a
//! basis, the object only): the terms above t_order dropped by their milli exponent,
//! coefficient x singlet multiplicity summed per (milli, y power, markers) or per (milli, y power,
//! flavor).

use crate::project::Projector;
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

type Coef = (i128, i128);

/// A row value beyond 128 bits after the projection: the caller's FORM path follows.
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
fn mul(a: Coef, b: Coef, fast: bool) -> Option<Coef> {
    let n = a.0.checked_mul(b.0)?;
    let d = a.1.checked_mul(b.1)?;
    if fast && d == 1 {
        return Some((n, 1));
    }
    reduce(n, d, fast).ok()
}

#[inline]
fn add(a: Coef, b: Coef, fast: bool) -> Option<Coef> {
    if a.1 == b.1 {
        return reduce(a.0.checked_add(b.0)?, a.1, fast).ok();
    }
    let n = a.0.checked_mul(b.1)?.checked_add(b.0.checked_mul(a.1)?)?;
    let d = a.1.checked_mul(b.1)?;
    reduce(n, d, fast).ok()
}

/// An arbitrary-precision reduced rational (denominator positive).
#[derive(Clone, Debug, PartialEq)]
struct Big {
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

/// The engine's coefficient: a reduced rational of two `i128`, or, once an operation on it has
/// overflowed, of two `BigInt` (brought back to `i128` whenever the reduced value fits).
#[derive(Clone, Debug)]
enum Q {
    S(i128, i128),
    B(Box<Big>),
}

impl Q {
    fn from_pair(n: i128, d: i128) -> Result<Q, String> {
        reduce(n, d, false).map(|(n, d)| Q::S(n, d))
    }
    fn one() -> Q {
        Q::S(1, 1)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        match self {
            Q::S(n, _) => *n == 0,
            Q::B(b) => b.n.is_zero(),
        }
    }
    fn big(&self) -> Big {
        match self {
            Q::S(n, d) => Big { n: BigInt::from(*n), d: BigInt::from(*d) },
            Q::B(b) => (**b).clone(),
        }
    }
    fn of_big(b: Big) -> Q {
        match (b.n.to_i128(), b.d.to_i128()) {
            (Some(n), Some(d)) => Q::S(n, d),
            _ => Q::B(Box::new(b)),
        }
    }
    #[inline]
    fn times(&self, o: &Q, fast: bool) -> Result<Q, String> {
        if let (Q::S(a, b), Q::S(c, d)) = (self, o) {
            if let Some((n, d)) = mul((*a, *b), (*c, *d), fast) {
                return Ok(Q::S(n, d));
            }
        }
        let (x, y) = (self.big(), o.big());
        Ok(Q::of_big(Big::reduced(&x.n * &y.n, &x.d * &y.d)?))
    }
    #[inline]
    fn plus(&self, o: &Q, fast: bool) -> Result<Q, String> {
        if let (Q::S(a, b), Q::S(c, d)) = (self, o) {
            if let Some((n, d)) = add((*a, *b), (*c, *d), fast) {
                return Ok(Q::S(n, d));
            }
        }
        let (x, y) = (self.big(), o.big());
        if x.d == y.d {
            return Ok(Q::of_big(Big::reduced(&x.n + &y.n, x.d)?));
        }
        Ok(Q::of_big(Big::reduced(&x.n * &y.d + &y.n * &x.d, &x.d * &y.d)?))
    }
    fn over(&self, k: i128, fast: bool) -> Result<Q, String> {
        if let Q::S(n, d) = self {
            if let Some(dk) = d.checked_mul(k) {
                return reduce(*n, dk, fast).map(|(n, d)| Q::S(n, d));
            }
        }
        let x = self.big();
        Ok(Q::of_big(Big::reduced(x.n, &x.d * BigInt::from(k))?))
    }
    fn times_int(&self, m: i128) -> Result<Q, String> {
        self.times(&Q::S(m, 1), false)
    }
    /// The value as two `i128`, or COEF_OVERFLOW.
    fn pair(&self) -> Result<Coef, String> {
        match self {
            Q::S(n, d) => Ok((*n, *d)),
            Q::B(_) => Err(COEF_OVERFLOW.to_string()),
        }
    }
    fn big_pair(&self) -> (BigInt, BigInt) {
        let b = self.big();
        (b.n, b.d)
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

/// The monomial cap of the engine (LANDSCAPE_NATIVE_MAX_TERMS, default 8,000,000): a power of the
/// series with more monomials than this raises "__capacity__" and the caller falls back to FORM,
/// which sorts on disk.  Found on the June-2026 Sp2nf5 landscape (a theory of 25 fields and
/// expansion order 38: 10 M monomials at k = 5 and 20 GB at k = 6 before the flavor projection;
/// FORM had computed it).  Since step 56b the cap applies to one power, the whole exponential no
/// longer being held.
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
fn weight_n(t: i64, s: i64, r: i64) -> i64 {
    25_000_000 * t + 5_000 * s + r
}

/// The smallest scaled weight whose milli exceeds `milli`.
#[inline]
fn threshold_n(milli: i64) -> i64 {
    (milli + 1) * 12_500_000 - 6_250_000
}

// --------------------------------------------------------------------------- //
// packed monomials
// --------------------------------------------------------------------------- //
/// The bounds [lo, hi] of every exponent over the monomials within the t-limit, from the letters.
fn field_bounds(letters: &[Vec<i64>], t_limit: i64, max_order: i64) -> (Vec<i64>, Vec<i64>) {
    let w = letters.first().map(|e| e.len()).unwrap_or(0);
    let mut lo = vec![0i64; w];
    let mut hi = vec![0i64; w];
    if w == 0 {
        return (lo, hi);
    }
    hi[0] = t_limit.max(0);
    for j in 1..w {
        let (mut maxpos, mut minneg) = (0i64, 0i64);
        let (mut rpos, mut rneg) = (0i64, 0i64);
        let mut zero_t = false;
        for e in letters {
            let x = e[j];
            maxpos = maxpos.max(x);
            minneg = minneg.min(x);
            if e[0] > 0 {
                let prod = t_limit as i128 * x as i128;
                let d = e[0] as i128;
                if x > 0 {
                    rpos = rpos.max(((prod + d - 1) / d) as i64);
                } else if x < 0 {
                    rneg = rneg.min(-(((-prod) + d - 1) / d) as i64);
                }
            } else if x != 0 {
                zero_t = true;
            }
        }
        hi[j] = max_order.saturating_mul(maxpos);
        lo[j] = max_order.saturating_mul(minneg);
        if !zero_t {
            hi[j] = hi[j].min(rpos);
            lo[j] = lo[j].max(rneg);
        }
    }
    (lo, hi)
}

/// The fields of a layout: (word, shift, width in bits) for the bounds, stored value = v + offset,
/// the width holding every intermediate sum of two stored values of a valid product.
fn field_layout(lo: &[i64], hi: &[i64]) -> Result<(Vec<(usize, u32, u32, i64)>, usize), String> {
    let mut out = Vec::with_capacity(lo.len());
    let (mut w, mut used) = (0usize, 0u32);
    for (l, h) in lo.iter().zip(hi.iter()) {
        let off = (-*l).max(0);
        let top = (*h as i128) + 2 * (off as i128);
        if top < 0 || top >= (1i128 << 62) {
            return Err("__capacity__".into());
        }
        let bits = (128 - (top as u128).leading_zeros()).max(1);
        if used + bits > 64 {
            w += 1;
            used = 0;
        }
        out.push((w, used, bits, off));
        used += bits;
    }
    Ok((out, if lo.is_empty() { 0 } else { w + 1 }))
}

struct Packer<const N: usize> {
    fields: Vec<(usize, u32, u64, i64)>, // (word, shift, mask, offset)
    bias: [u64; N],
}

impl<const N: usize> Packer<N> {
    fn new(lo: &[i64], hi: &[i64]) -> Result<Self, String> {
        let (layout, words) = field_layout(lo, hi)?;
        if words > N {
            return Err("__capacity__".into());
        }
        let mut bias = [0u64; N];
        let mut fields = Vec::with_capacity(layout.len());
        for (w, s, bits, off) in layout {
            let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
            bias[w] |= (off as u64) << s;
            fields.push((w, s, mask, off));
        }
        Ok(Packer { fields, bias })
    }
    #[inline]
    fn pack(&self, v: &[i64]) -> [u64; N] {
        let mut k = [0u64; N];
        for ((w, s, _, off), x) in self.fields.iter().zip(v.iter()) {
            k[*w] |= ((x + off) as u64) << s;
        }
        k
    }
    #[inline]
    fn get(&self, k: &[u64; N], j: usize) -> i64 {
        let (w, s, m, off) = self.fields[j];
        ((k[w] >> s) & m) as i64 - off
    }
    /// The words with only the fields `from..` kept (the character slots of a layout).
    fn mask_from(&self, from: usize) -> [u64; N] {
        let mut m = [0u64; N];
        for (w, s, mask, _) in self.fields[from..].iter() {
            m[*w] |= mask << s;
        }
        m
    }
    fn unpack(&self, k: &[u64; N]) -> Vec<i64> {
        (0..self.fields.len()).map(|j| self.get(k, j)).collect()
    }
    #[inline]
    fn add(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut o = [0u64; N];
        for i in 0..N {
            o[i] = a[i].wrapping_add(b[i]).wrapping_sub(self.bias[i]);
        }
        o
    }
}

type Map<const N: usize> = FxHashMap<[u64; N], Q>;

// --------------------------------------------------------------------------- //
// the layout of the monomials
// --------------------------------------------------------------------------- //
struct Ctx<const N: usize> {
    pf: Packer<N>,       // field-resolved [t, s, r, y, f_1..f_n, c_1..c_m]
    pl: Packer<N>,       // flavor-refined [t, s, r, y, x_1..x_rank, c_1..c_m] (a basis only)
    n_fields: usize,
    n_slots: usize,
    basis: Option<Vec<Vec<i64>>>,
    rank: usize,
    above_n: i64,
    limit_n: i64,
    exact: bool,
    fast: bool,
}

impl<const N: usize> Ctx<N> {
    #[inline]
    fn classify(&self) -> bool {
        self.basis.is_some() || self.exact
    }

    /// basis . markers of an unpacked field-resolved vector, as the flavor-refined vector.
    fn to_flavor(&self, e: &[i64]) -> Result<Vec<i64>, String> {
        let basis = self.basis.as_ref().expect("a basis");
        let mut out = Vec::with_capacity(4 + self.rank + self.n_slots);
        out.extend_from_slice(&e[..4]);
        for row in basis {
            let mut s: i64 = 0;
            for (b, x) in row.iter().zip(e[4..4 + self.n_fields].iter()) {
                s = s.checked_add(b.checked_mul(*x).ok_or("overflow")?).ok_or("overflow")?;
            }
            out.push(s);
        }
        out.extend_from_slice(&e[4 + self.n_fields..]);
        Ok(out)
    }

    fn image(&self, k: &[u64; N]) -> Result<[u64; N], String> {
        Ok(self.pl.pack(&self.to_flavor(&self.pf.unpack(k))?))
    }
}

/// itotal's letters sorted by t-power: the field-resolved ones with their flavor images, the
/// flavor-refined ones (a basis only).
struct Letters<const N: usize> {
    fr: Vec<([u64; N], Q)>,
    fr_t: Vec<i64>,
    fr_n: Vec<i64>,
    fr_img: Vec<[u64; N]>,
    fl: Vec<([u64; N], Q)>,
    fl_t: Vec<i64>,
    fl_n: Vec<i64>,
}

struct Power<const N: usize> {
    fr: Vec<([u64; N], Q)>,
    fl: Vec<([u64; N], Q)>,
}

impl<const N: usize> Power<N> {
    fn len(&self) -> usize {
        self.fr.len() + self.fl.len()
    }
}

struct Maps<const N: usize> {
    fr: Map<N>,
    fl: Map<N>,
}

impl<const N: usize> Maps<N> {
    fn new() -> Self {
        Maps { fr: FxHashMap::default(), fl: FxHashMap::default() }
    }
    fn len(&self) -> usize {
        self.fr.len() + self.fl.len()
    }
}

#[inline]
fn insert<const N: usize>(map: &mut Map<N>, key: [u64; N], c: Q, fast: bool) -> Result<(), String> {
    match map.get_mut(&key) {
        Some(e) => {
            *e = e.plus(&c, fast)?;
        }
        None => {
            map.insert(key, c);
        }
    }
    Ok(())
}

fn merge_into<const N: usize>(into: &mut Map<N>, from: Map<N>, fast: bool) -> Result<(), String> {
    for (k, c) in from {
        insert(into, k, c, fast)?;
    }
    Ok(())
}

// --------------------------------------------------------------------------- //
// the powers
// --------------------------------------------------------------------------- //
/// The products of one chunk of terms of P_{k-1} (of one kind) with itotal.  `live` counts the
/// monomials held by every chunk of the step so far; past twice the cap the step stops with
/// "__capacity__" before the merge.
#[allow(clippy::too_many_arguments)]
fn products<const N: usize>(rows: &[([u64; N], Q)], rows_fr: bool, it: &Letters<N>, ctx: &Ctx<N>, t_limit: i64,
                            deadline: Option<Instant>, cap: usize, live: &AtomicUsize) -> Result<Maps<N>, String> {
    check_deadline(deadline)?;
    let mut reported = 0usize;
    let fast = ctx.fast;
    let p = if rows_fr { &ctx.pf } else { &ctx.pl };
    let mut local = Maps::new();
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
        let t_ep = p.get(ep, 0);
        let bound = t_limit - t_ep;
        let n_ep = if ctx.classify() { weight_n(t_ep, p.get(ep, 1), p.get(ep, 2)) } else { 0 };
        let ep_img: Option<[u64; N]> = if rows_fr && ctx.basis.is_some() { Some(ctx.image(ep)?) } else { None };
        // x field-resolved letters
        for j in 0..it.fr.len() {
            if it.fr_t[j] > bound {
                break;
            }
            let mut n_p = 0i64;
            if ctx.classify() {
                n_p = n_ep + it.fr_n[j];
                if ctx.exact && n_p >= ctx.limit_n {
                    continue;
                }
            }
            let c = cp.times(&it.fr[j].1, fast)?;
            if rows_fr && (ctx.basis.is_none() || n_p < ctx.above_n) {
                insert(&mut local.fr, ctx.pf.add(ep, &it.fr[j].0), c, fast)?;
            } else {
                let a = if rows_fr { ep_img.as_ref().expect("an image") } else { ep };
                insert(&mut local.fl, ctx.pl.add(a, &it.fr_img[j]), c, fast)?;
            }
        }
        // x flavor-refined letters (a basis only)
        for j in 0..it.fl.len() {
            if it.fl_t[j] > bound {
                break;
            }
            if ctx.exact && n_ep + it.fl_n[j] >= ctx.limit_n {
                continue;
            }
            let c = cp.times(&it.fl[j].1, fast)?;
            let a = if rows_fr { ep_img.as_ref().expect("an image") } else { ep };
            insert(&mut local.fl, ctx.pl.add(a, &it.fl[j].0), c, fast)?;
        }
    }
    Ok(local)
}

/// P_{k-1} x itotal with the t-bound, divided by k.  The deadline is checked every 256 rows of
/// every chunk and at every merge; the chunks together past twice the cap during the step, or
/// past the cap after it, stop with "__capacity__" before the merge.
fn step<const N: usize>(prev: &Power<N>, it: &Letters<N>, ctx: &Ctx<N>, t_limit: i64, k: i128, deadline: Option<Instant>,
                        cap: usize) -> Result<Power<N>, String> {
    let total = prev.len();
    let chunk = ((total + 63) / 64).max(64);
    let fast = ctx.fast;
    let live = AtomicUsize::new(0);
    let mut parts: Vec<Maps<N>> = prev.fr
        .par_chunks(chunk)
        .map(|rows| products(rows, true, it, ctx, t_limit, deadline, cap, &live))
        .collect::<Result<Vec<_>, String>>()?;
    let parts_fl: Vec<Maps<N>> = prev.fl
        .par_chunks(chunk)
        .map(|rows| products(rows, false, it, ctx, t_limit, deadline, cap, &live))
        .collect::<Result<Vec<_>, String>>()?;
    parts.extend(parts_fl);
    if parts.iter().map(|p| p.len()).sum::<usize>() > 2 * cap {
        return Err("__capacity__".into());
    }
    let merged = parts
        .into_par_iter()
        .map(Ok)
        .reduce(|| Ok(Maps::new()), |a: Result<Maps<N>, String>, b| {
            check_deadline(deadline)?;
            let mut a = a?;
            let mut b = b?;
            if a.len() < b.len() {
                std::mem::swap(&mut a, &mut b);
            }
            merge_into(&mut a.fr, b.fr, fast)?;
            merge_into(&mut a.fl, b.fl, fast)?;
            Ok(a)
        })?;
    let mut out = Power { fr: Vec::with_capacity(merged.fr.len()), fl: Vec::with_capacity(merged.fl.len()) };
    for (src, dst) in [(merged.fr, &mut out.fr), (merged.fl, &mut out.fl)] {
        for (key, c) in src {
            if c.is_zero() {
                continue;
            }
            dst.push((key, c.over(k, fast)?));
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------------- //
// the multiplicities and the rows
// --------------------------------------------------------------------------- //
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

struct Mults<'a, 'py> {
    py: Python<'py>,
    lookup: &'a Bound<'py, PyAny>,
    slots: &'a [(usize, Vec<i64>, i64)],
    n_nodes: usize,
    native: Option<Projector>,
    timeout: Option<f64>,
    verify: bool,
    verified: usize,
    fallback: usize,
}

impl<'a, 'py> Mults<'a, 'py> {
    fn python(&self, cexps: &[i32]) -> PyResult<i128> {
        let obj = chars_of(self.py, cexps, self.slots, self.n_nodes)?;
        self.lookup.call1((obj,))?.extract()
    }

    fn get(&mut self, cexps: &[i32], deadline: Option<Instant>) -> PyResult<i128> {
        let native = match self.native.as_mut() {
            Some(p) => p.multiplicity(cexps, deadline).map_err(|m| engine_error(self.py, m, self.timeout))?,
            None => None,
        };
        match native {
            Some(m) => {
                if self.verify {
                    let want = self.python(cexps)?;
                    if want != m {
                        return Err(PyValueError::new_err(format!(
                            "multiplicity check: the extension gives {m}, the projector {want}, for slot exponents {cexps:?}")));
                    }
                    self.verified += 1;
                }
                Ok(m)
            }
            None => {
                self.fallback += 1;
                self.python(cexps)
            }
        }
    }
}

type Acc = FxHashMap<(i128, i64, Vec<i64>), Q>;

/// The rows of one power, added into the accumulators: the terms above the limit dropped by their
/// milli exponent, coefficient x multiplicity per (milli, y power, markers or flavor).
#[allow(clippy::too_many_arguments)]
fn accumulate<const N: usize>(power: &Power<N>, ctx: &Ctx<N>, limit: i64, mults: &mut Mults<'_, '_>, acc_fr: &mut Acc, acc_fl: &mut Acc,
                              deadline: Option<Instant>) -> PyResult<()> {
    for (fr, list) in [(true, &power.fr), (false, &power.fl)] {
        if list.is_empty() {
            continue;                   // without a basis the flavor-refined layout has no fields
        }
        let p = if fr { &ctx.pf } else { &ctx.pl };
        let cstart = if fr { 4 + ctx.n_fields } else { 4 + ctx.rank };
        let width = cstart + ctx.n_slots;
        let acc = if fr { &mut *acc_fr } else { &mut *acc_fl };
        // the distinct character products of this power, keyed by the words of their slot fields
        let slot_mask = p.mask_from(cstart);
        let mut seen: FxHashMap<[u64; N], i128> = FxHashMap::default();
        for (row, (k, c)) in list.iter().enumerate() {
            if row % 65536 == 65535 {
                check_deadline(deadline).map_err(|m| engine_error(mults.py, m, mults.timeout))?;
            }
            let milli = milli_exponent(p.get(k, 0), p.get(k, 1), p.get(k, 2));
            if milli > limit as i128 {
                continue;
            }
            let mut ck = [0u64; N];
            for i in 0..N {
                ck[i] = k[i] & slot_mask[i];
            }
            let mult: i128 = if let Some(m) = seen.get(&ck) {
                *m
            } else {
                let cexps: Vec<i32> = (cstart..width).map(|j| p.get(k, j) as i32).collect();
                let m = if cexps.iter().all(|x| *x == 0) { 1 } else { mults.get(&cexps, deadline)? };
                seen.insert(ck, m);
                m
            };
            if mult == 0 {
                continue;
            }
            let xs: Vec<i64> = (4..cstart).map(|j| p.get(k, j)).collect();
            let cm = c.times_int(mult).map_err(PyValueError::new_err)?;
            let key = (milli, p.get(k, 3), xs);
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
    Ok(())
}

fn timeout_error(py: Python<'_>, timeout: Option<f64>) -> PyErr {
    match py.import("subprocess").and_then(|m| m.getattr("TimeoutExpired")).and_then(|cls| cls.call1(("landscape_native.expand_series", timeout.unwrap_or(0.0)))) {
        Ok(exc) => PyErr::from_value(exc),
        Err(e) => e,
    }
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

// --------------------------------------------------------------------------- //
// the engine
// --------------------------------------------------------------------------- //
struct Input<'a> {
    terms: &'a [(i128, i128, Vec<i32>)],
    t_limit: i64,
    max_order: usize,
    n_fields: usize,
    slots: &'a [(usize, Vec<i64>, i64)],
    n_nodes: usize,
    limit: i64,
    basis: Option<Vec<Vec<i64>>>,
    exact: bool,
    fast: bool,
    lo_fr: Vec<i64>,
    hi_fr: Vec<i64>,
    lo_fl: Vec<i64>,
    hi_fl: Vec<i64>,
}

#[allow(clippy::too_many_arguments)]
fn run<'py, const N: usize>(py: Python<'py>, inp: &Input, lookup: &Bound<'py, PyAny>, groups: &Option<Vec<(String, String, usize)>>,
                       timeout: Option<f64>, deadline: Option<Instant>, monomials: bool, native_rows: bool) -> PyResult<Py<PyAny>> {
    let err = |m: String| engine_error(py, m, timeout);
    let rank = inp.basis.as_ref().map(|b| b.len()).unwrap_or(0);
    let ctx: Ctx<N> = Ctx {
        pf: Packer::new(&inp.lo_fr, &inp.hi_fr).map_err(err)?,
        pl: Packer::new(&inp.lo_fl, &inp.hi_fl).map_err(err)?,
        n_fields: inp.n_fields,
        n_slots: inp.slots.len(),
        basis: inp.basis.clone(),
        rank,
        above_n: threshold_n(6000),
        limit_n: threshold_n(inp.limit),
        exact: inp.exact,
        fast: inp.fast,
    };
    let trace = std::env::var("LANDSCAPE_NATIVE_TRACE").is_ok();
    let verify = std::env::var("LANDSCAPE_NATIVE_VERIFY_MULT").map(|v| v.trim() == "1").unwrap_or(false);
    // itotal: the terms within the limit, sorted by t-power, in two lists when a basis is set
    let mut sorted: Vec<&(i128, i128, Vec<i32>)> = inp.terms.iter().filter(|(_, _, e)| e[0] as i64 <= inp.t_limit).collect();
    sorted.sort_by_key(|(_, _, e)| e[0]);
    let mut it = Letters { fr: Vec::new(), fr_t: Vec::new(), fr_n: Vec::new(), fr_img: Vec::new(), fl: Vec::new(), fl_t: Vec::new(), fl_n: Vec::new() };
    for (n, d, e) in sorted {
        let v: Vec<i64> = e.iter().map(|x| *x as i64).collect();
        let c = Q::from_pair(*n, *d).map_err(err)?;
        let above = ctx.basis.is_some() && milli_exponent(v[0], v[1], v[2]) as i64 > 6000;
        let wn = weight_n(v[0], v[1], v[2]);
        if above {
            it.fl.push((ctx.pl.pack(&ctx.to_flavor(&v).map_err(err)?), c));
            it.fl_t.push(v[0]);
            it.fl_n.push(wn);
        } else {
            let key = ctx.pf.pack(&v);
            if ctx.basis.is_some() {
                it.fr_img.push(ctx.pl.pack(&ctx.to_flavor(&v).map_err(err)?));
            }
            it.fr.push((key, c));
            it.fr_t.push(v[0]);
            it.fr_n.push(wn);
        }
    }
    // the multiplicities: the extension, the Python projector as the fallback and the check
    let native = match groups {
        Some(g) => Projector::new(g, inp.slots, deadline).map_err(err)?,
        None => None,
    };
    let mut mults = Mults { py, lookup, slots: inp.slots, n_nodes: inp.n_nodes, native, timeout, verify, verified: 0, fallback: 0 };
    let mut acc_fr: Acc = FxHashMap::default();
    let mut acc_fl: Acc = FxHashMap::default();
    let mut result: Map<N> = FxHashMap::default();
    // the constant term
    let zero_fr = ctx.pf.pack(&vec![0i64; 4 + inp.n_fields + inp.slots.len()]);
    acc_fr.insert((0, 0, vec![0i64; inp.n_fields]), Q::one());
    if monomials {
        result.insert(zero_fr, Q::one());
    }
    let cap = capacity();
    let t_exp = Instant::now();
    let mut power = Power { fr: it.fr.clone(), fl: it.fl.clone() };
    for k in 1..=inp.max_order {
        if k > 1 {
            let t0 = Instant::now();
            let prev = std::mem::replace(&mut power, Power { fr: Vec::new(), fl: Vec::new() });
            let next = py.allow_threads(|| step(&prev, &it, &ctx, inp.t_limit, k as i128, deadline, cap));
            drop(prev);
            power = next.map_err(err)?;
            if trace {
                eprintln!("[series] k={k}: {} + {} terms, {:.3} s", power.fr.len(), power.fl.len(), t0.elapsed().as_secs_f64());
            }
            if power.len() == 0 {
                break;
            }
            if power.len() > cap || (monomials && result.len() + power.len() > cap) {
                return Err(err("__capacity__".into()));
            }
        }
        if monomials {
            for (key, c) in power.fr.iter() {
                insert(&mut result, *key, c.clone(), ctx.fast).map_err(err)?;
            }
        } else {
            accumulate(&power, &ctx, inp.limit, &mut mults, &mut acc_fr, &mut acc_fl, deadline)?;
        }
        check_deadline(deadline).map_err(err)?;
    }
    drop(power);
    if trace {
        eprintln!("[series] exponential {:.3} s", t_exp.elapsed().as_secs_f64());
    }
    if verify {
        eprintln!("[mult] verified {}, fallback {}", mults.verified, mults.fallback);
    }
    if monomials {
        let out = PyList::empty(py);
        let mut rows: Vec<(Vec<i64>, &Q)> = result.iter().filter(|(_, c)| !c.is_zero()).map(|(k, c)| (ctx.pf.unpack(k), c)).collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for (e, c) in rows {
            let (n, d) = c.big_pair();
            out.append((n, d, PyList::new(py, e.iter().map(|x| *x as i32))?))?;
        }
        return Ok(out.into_any().unbind());
    }
    // integral after the projection: back to 128 bits (a row beyond raises the overflow; the caller's FORM path follows)
    let pairs = |acc: Acc| -> PyResult<Vec<((i128, i64, Vec<i64>), Coef)>> {
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
    crate::post::emit_rows(py, rows_fr, rows_fl, inp.n_fields, inp.basis.clone(), native_rows)
}

/// expand_series(terms, t_limit, max_order, n_fields, slots, n_nodes, t_order, lookup, timeout, monomials, native_rows,
///               basis, exact, coef64, groups): the rows as a FieldResolvedRows object when native_rows is set (the
/// native post-processing pass reads them there; with a basis, the only form); see the module.  `groups[node]` =
/// (LiE group name, type letter, rank) of the node's store, for the multiplicities in the extension; without it
/// every multiplicity comes from `lookup`.
#[pyfunction]
#[pyo3(signature = (terms, t_limit, max_order, n_fields, slots, n_nodes, t_order, lookup, timeout=None, monomials=false, native_rows=false,
                    basis=None, exact=false, coef64=false, groups=None))]
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
    groups: Option<Vec<(String, String, usize)>>,
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
    crate::lie::init_pool();
    // the bounds of the two layouts from the letters within the limit
    let letters: Vec<Vec<i64>> = terms.iter().filter(|(_, _, e)| e[0] <= t_limit).map(|(_, _, e)| e.iter().map(|x| *x as i64).collect()).collect();
    let (lo_fr, hi_fr) = field_bounds(&letters, t_limit as i64, max_order as i64);
    let (lo_fl, hi_fl) = match &basis {
        Some(b) => {
            let rank = b.len();
            let mut images = Vec::with_capacity(letters.len());
            for e in &letters {
                let mut v = Vec::with_capacity(4 + rank + slots.len());
                v.extend_from_slice(&e[..4]);
                for row in b {
                    let mut s: i64 = 0;
                    for (x, y) in row.iter().zip(e[4..4 + n_fields].iter()) {
                        s = s.checked_add(x.checked_mul(*y).ok_or_else(|| PyValueError::new_err("overflow"))?)
                            .ok_or_else(|| PyValueError::new_err("overflow"))?;
                    }
                    v.push(s);
                }
                v.extend_from_slice(&e[4 + n_fields..]);
                images.push(v);
            }
            field_bounds(&images, t_limit as i64, max_order as i64)
        }
        None => (Vec::new(), Vec::new()),
    };
    let words = field_layout(&lo_fr, &hi_fr).map_err(|m| engine_error(py, m, timeout))?.1
        .max(field_layout(&lo_fl, &hi_fl).map_err(|m| engine_error(py, m, timeout))?.1);
    let inp = Input { terms: &terms, t_limit: t_limit as i64, max_order, n_fields, slots: &slots, n_nodes, limit: 1000 * t_order, basis,
                      exact, fast: coef64, lo_fr, hi_fr, lo_fl, hi_fl };
    macro_rules! go {
        ($n:literal) => {
            run::<$n>(py, &inp, &lookup, &groups, timeout, deadline, monomials, native_rows)
        };
    }
    match words {
        0..=1 => go!(1),
        2 => go!(2),
        3 => go!(3),
        4 => go!(4),
        5 => go!(5),
        6 => go!(6),
        7 => go!(7),
        8 => go!(8),
        9..=10 => go!(10),
        11..=12 => go!(12),
        13..=16 => go!(16),
        17..=24 => go!(24),
        25..=32 => go!(32),
        _ => Err(engine_error(py, "__capacity__".into(), timeout)),
    }
}
