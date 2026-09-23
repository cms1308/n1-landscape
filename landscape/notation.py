"""Index-string notation of the package: the exact polynomial representation of index
strings and operator monomials, the parser of the Mathematica InputForm strings of the
legacy records and of this notation, and the printer.  `mma_key` reproduces Mathematica's
canonical order of symbol names (used by the monomial renderer of convert.py).

Rules:

- U(1) flavor fugacities are printed ``x1, x2, ...`` (parsed from ``g<i>`` or
  ``x<i>``); the internal names ``g<i>`` are untouched elsewhere.
- Factors of a term appear in the order t, x_i (numeric index order), y, then
  field symbols (operator monomials) in natural order: alphabetic prefix
  case-insensitively, then case-sensitively, then the numeric suffix.
- Terms are ordered by ascending t exponent; ties by the sequence of
  (index, exponent) pairs of the x factors, lexicographically; then by the y
  exponent; then by the field factors.
- t exponents live on the 1/1000 grid: integers are printed without a decimal
  point (``t^6``), other values with the minimal decimals (``t^3.703``).
- Negative powers are printed as negative exponents (``x4^-5``), never as
  denominators; no parentheses occur.
- Coefficients are integers, omitted when +-1 (the sign is kept); the zero
  polynomial prints as ``0``.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Tuple

_TOKEN = re.compile(r"\d+\.\d*|\d+|[A-Za-z]+\d*|[()*/^+-]")

# symbol = (kind, *key); kind 0 = x_i, 1 = y, 2 = field symbol
Symbol = tuple
Monomial = Tuple[int, Tuple[Tuple[Symbol, int], ...]]  # (t_milli, sorted factors)
Poly = Dict[Monomial, Fraction]


def _symbol(name: str) -> Symbol:
    m = re.fullmatch(r"([A-Za-z]+)(\d*)", name)
    if not m:
        raise ValueError(f"bad symbol {name!r}")
    prefix, digits = m.group(1), m.group(2)
    if prefix in ("g", "x") and digits:
        return (0, int(digits))
    if name == "y":
        return (1,)
    return (2, prefix.lower(), prefix, int(digits) if digits else -1, name)


def _symbol_name(sym: Symbol) -> str:
    if sym[0] == 0:
        return f"x{sym[1]}"
    if sym[0] == 1:
        return "y"
    return sym[4]


class _Parser:
    def __init__(self, text: str):
        self.toks = _TOKEN.findall(text)
        if "".join(self.toks) != text:
            raise ValueError(f"unparsed characters in {text!r}")
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, tok=None):
        cur = self.peek()
        if cur is None or (tok is not None and cur != tok):
            raise ValueError(f"expected {tok!r}, got {cur!r} at {self.i}")
        self.i += 1
        return cur

    # expr := ['-'] term (('+'|'-') term)*
    def expr(self) -> Poly:
        poly: Poly = {}
        sign = 1
        if self.peek() == "-":
            self.take(); sign = -1
        elif self.peek() == "+":
            self.take()
        self._add(poly, self.term(), sign)
        while self.peek() in ("+", "-"):
            sign = 1 if self.take() == "+" else -1
            self._add(poly, self.term(), sign)
        if self.peek() is not None:
            raise ValueError(f"trailing token {self.peek()!r}")
        return poly

    @staticmethod
    def _add(poly: Poly, term, sign):
        coeff, mono = term
        poly[mono] = poly.get(mono, Fraction(0)) + sign * coeff
        if poly[mono] == 0:
            del poly[mono]

    # term := unit ('/' divisor)*   unit := '(' term ')' | product
    # divisor := '(' product ')' | factor      (InputForm also prints -(a/b))
    def term(self):
        coeff, factors = self._unit()
        while self.peek() == "/":
            self.take("/")
            if self.peek() == "(":
                self.take("("); dc, df = self.product(); self.take(")")
            else:
                dc, df = self.factor()
            coeff /= dc
            for sym, p in df.items():
                factors[sym] = factors.get(sym, 0) - p
        return coeff, _monomial(factors)

    def _unit(self):
        if self.peek() == "(":
            self.take("(")
            coeff, mono = self.term()
            self.take(")")
            factors = dict(mono[1])
            if mono[0]:
                factors["t"] = mono[0]
            return coeff, factors
        return self.product()

    # product := factor ('*' factor)*
    def product(self):
        coeff, factors = self.factor()
        while self.peek() == "*":
            self.take("*")
            c2, f2 = self.factor()
            coeff *= c2
            for sym, p in f2.items():
                factors[sym] = factors.get(sym, 0) + p
        return coeff, factors

    # factor := number | symbol ('^' ['-'] number)?
    def factor(self):
        tok = self.take()
        if tok[0].isdigit():
            if "." in tok:
                raise ValueError(f"non-integer coefficient {tok!r}")
            return Fraction(int(tok)), {}
        if not tok[0].isalpha():
            raise ValueError(f"unexpected token {tok!r}")
        power = Fraction(1)
        if self.peek() == "^":
            self.take("^")
            neg = False
            if self.peek() == "-":
                self.take(); neg = True
            num = self.take()
            if not num[0].isdigit():
                raise ValueError(f"bad exponent {num!r}")
            power = Fraction(num.rstrip("."))  # exact decimal ("6." -> 6)
            if neg:
                power = -power
        if tok == "t":
            milli = power * 1000
            if milli.denominator != 1:
                raise ValueError(f"t exponent {power} is not on the 1/1000 grid")
            return Fraction(1), {"t": int(milli)}
        if power.denominator != 1:
            raise ValueError(f"non-integer exponent {power} of {tok}")
        return Fraction(1), {_symbol(tok): int(power)}


def _monomial(factors: dict) -> Monomial:
    t_milli = factors.get("t", 0)
    rest = tuple(sorted((sym, p) for sym, p in factors.items()
                        if sym != "t" and p != 0))
    return (t_milli, rest)


def parse(text: str) -> Poly:
    """Parse a Mathematica InputForm string or a new-notation string exactly."""
    text = text.strip()
    if text == "0":
        return {}
    return _Parser(text).expr()


def _t_exponent(milli: int) -> str:
    sign = "-" if milli < 0 else ""
    milli = abs(milli)
    whole, frac = divmod(milli, 1000)
    if frac == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}." + f"{frac:03d}".rstrip("0")


def _sort_key(mono: Monomial):
    t_milli, rest = mono
    xs = tuple((s[1], p) for s, p in rest if s[0] == 0)
    ys = tuple(p for s, p in rest if s[0] == 1)
    fields = tuple((s, p) for s, p in rest if s[0] == 2)
    return (t_milli, xs, ys, fields)


def format_term(coeff: Fraction, mono: Monomial) -> str:
    if coeff.denominator != 1:
        raise ValueError(f"non-integer coefficient {coeff}")
    t_milli, rest = mono
    factors: List[str] = []
    if t_milli != 0:
        factors.append("t" if t_milli == 1000 else f"t^{_t_exponent(t_milli)}")
    for sym, p in rest:  # already sorted: x_i by index, then y, then fields
        name = _symbol_name(sym)
        factors.append(name if p == 1 else f"{name}^{p}")
    c = int(coeff)
    if not factors:
        return str(c)
    body = "*".join(factors)
    if c == 1:
        return body
    if c == -1:
        return "-" + body
    return f"{c}*{body}"


def format_poly(poly: Poly) -> str:
    """Print a polynomial in the new notation."""
    if not poly:
        return "0"
    out = ""
    for mono in sorted(poly, key=_sort_key):
        term = format_term(poly[mono], mono)
        if out and not term.startswith("-"):
            out += "+"
        out += term
    return out


def convert(text: str) -> str:
    """Mathematica InputForm string -> new notation (exact)."""
    return format_poly(parse(text))


def convert_list(items: List[str]) -> List[str]:
    """Operator lists: each element converted; order and multiplicity kept."""
    return [convert(s) for s in items]


def mma_key(name: str):
    """Sort key reproducing Mathematica's canonical order of symbol names."""
    return (name.lower(), name.swapcase())
