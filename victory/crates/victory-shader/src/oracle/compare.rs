//! The comparison of a run's output with another's (a host's read-back), dword by dword, in
//! classes kept apart so that noise in one never hides a difference in another:
//!
//! - **exact**: integer, bit, mask, address and memory contents, and the floats the reference
//!   pins (additions, multiplications, conversions it gives as 0.5 ULP or exact): bit for bit.
//! - **approximate**: values made by an instruction whose rounding the reference does not pin
//!   or that the console fuses where a host may not ([`super::semantics::own_tolerance`] names
//!   each and its ULPs), and everything computed from them: within the ULPs (or the absolute
//!   error) the oracle carried with the value; any NaN matches a NaN whose bits are not pinned.
//! - **order-dependent**: what lanes or waves raced for (an atomic's return, a dword several of
//!   them stored to). Where the oracle knows the values one of which landed (exact whole-dword
//!   stores: [`super::Memory::choices`]), the other side's must be one of them; else counted,
//!   not compared.
//! - **undefined**: read from a register or LDS nothing wrote. Counted, not compared.
//!
//! The two sides must be of one length, with a trust for every dword: a missing or extra dword
//! is a disagreement, not a dword skipped. [`Diff::complete`] says whether every dword was
//! checked.

use super::Tol;

/// One dword that differs.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    pub dword: usize,
    pub oracle: u32,
    pub other: u32,
    pub tol: Tol,
}

/// The comparison of two outputs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Diff {
    pub exact: usize,
    pub exact_differ: Vec<Mismatch>,
    pub approximate: usize,
    /// The largest ULPs any approximate dword was allowed.
    pub approximate_ulps: u32,
    pub approximate_differ: Vec<Mismatch>,
    /// Order-dependent dwords, and of them those checked against the values that could land.
    pub unordered: usize,
    pub unordered_checked: usize,
    pub unordered_differ: Vec<Mismatch>,
    pub undefined: usize,
    /// Why the two sides could not be compared dword by dword: not of one length, or the
    /// oracle's trusts not as many as its dwords.
    pub shape: Option<String>,
}

impl Diff {
    pub fn agrees(&self) -> bool {
        self.shape.is_none()
            && self.exact_differ.is_empty()
            && self.approximate_differ.is_empty()
            && self.unordered_differ.is_empty()
    }

    /// Whether every dword was checked (none order-dependent without its values, none
    /// undefined): an agreement that is not complete is a partial one.
    pub fn complete(&self) -> bool {
        self.shape.is_none() && self.unordered == self.unordered_checked && self.undefined == 0
    }

    /// Adds `d`'s counts to these.
    pub fn add(&mut self, d: &Diff) {
        self.exact += d.exact;
        self.approximate += d.approximate;
        self.approximate_ulps = self.approximate_ulps.max(d.approximate_ulps);
        self.unordered += d.unordered;
        self.unordered_checked += d.unordered_checked;
        self.undefined += d.undefined;
    }

    /// The comparison in one line, each class on its own.
    pub fn line(&self) -> String {
        format!(
            "{}exact {} dwords, {} differ; approximate {} dwords (up to {} ULPs), {} beyond; order-dependent {} ({} checked against the values that could land, {} outside them); undefined {}; {}",
            self.shape.as_ref().map_or(String::new(), |why| format!("{why}; ")),
            self.exact,
            self.exact_differ.len(),
            self.approximate,
            self.approximate_ulps,
            self.approximate_differ.len(),
            self.unordered,
            self.unordered_checked,
            self.unordered_differ.len(),
            self.undefined,
            if self.complete() { "complete" } else { "PARTIAL" }
        )
    }
}

/// ULPs between two single-precision bit patterns (on the line of ordered floats, so that -0 and
/// +0 are one apart and the integers that are bit patterns of small floats compare by value).
pub fn ulps(a: u32, b: u32) -> u64 {
    let key = |x: u32| -> i64 {
        if x >> 31 == 1 { -i64::from(x & 0x7FFF_FFFF) - 1 } else { i64::from(x) }
    };
    key(a).abs_diff(key(b))
}

fn nan(x: u32) -> bool {
    f32::from_bits(x).is_nan()
}

fn infinite(x: u32) -> bool {
    f32::from_bits(x).is_infinite()
}

/// Whether `b` is within `t` of `a`: the same bits; a NaN for a NaN whose bits are not pinned;
/// else, two finite values within the ULPs or the absolute error (an infinity or a NaN is never
/// near anything but itself).
fn within(a: u32, b: u32, t: Tol) -> bool {
    if a == b {
        return true;
    }
    if nan(a) || nan(b) {
        return t.nan && nan(a) && nan(b);
    }
    if infinite(a) || infinite(b) {
        return false;
    }
    ulps(a, b) <= u64::from(t.ulps) || (f32::from_bits(a) - f32::from_bits(b)).abs() <= t.abs
}

/// Compares the oracle's dwords (with their trust) with `other`'s.
pub fn compare(oracle: &[u32], tol: &[Tol], other: &[u32]) -> Diff {
    compare_choices(oracle, tol, &[], other)
}

/// [`compare`], with the values each order-dependent dword may hold (`choices`, empty where the
/// oracle does not know them).
pub fn compare_choices(oracle: &[u32], tol: &[Tol], choices: &[Vec<u32>], other: &[u32]) -> Diff {
    let mut d = Diff::default();
    if tol.len() != oracle.len() {
        d.shape = Some(format!("{} TRUSTS FOR {} DWORDS", tol.len(), oracle.len()));
        return d;
    }
    if oracle.len() != other.len() {
        d.shape = Some(format!("LENGTHS DIFFER (oracle {}, other {})", oracle.len(), other.len()));
        return d;
    }
    for (i, (&a, &b)) in oracle.iter().zip(other).enumerate() {
        let t = tol[i];
        let miss = || Mismatch { dword: i, oracle: a, other: b, tol: t };
        if t.undefined {
            d.undefined += 1;
        } else if t.unordered {
            d.unordered += 1;
            if let Some(values) = choices.get(i).filter(|c| !c.is_empty()) {
                d.unordered_checked += 1;
                if !values.contains(&b) {
                    d.unordered_differ.push(miss());
                }
            }
        } else if t.exact() {
            d.exact += 1;
            if a != b {
                d.exact_differ.push(miss());
            }
        } else {
            d.approximate += 1;
            d.approximate_ulps = d.approximate_ulps.max(t.ulps);
            if !within(a, b, t) {
                d.approximate_differ.push(miss());
            }
        }
    }
    d
}
