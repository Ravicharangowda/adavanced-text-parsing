"""
Advanced Text Parsing - Mini Project
------------------------------------
Pure standard library (no installs needed). Four modules in one file:

1. Entity extractor  - emails, URLs, phones, dates, IPs, hashtags, mentions (regex)
2. Tokenizer         - regex master-pattern lexer with named groups
3. Log parser        - structured parsing + key=value extraction + statistics
4. Expression parser - recursive-descent calculator built on the tokenizer

Usage:
    python text_parser.py demo
    python text_parser.py extract "Mail me at a@b.com or visit https://x.org #python"
    python text_parser.py tokens  "3 + 4 * (2 - 1)"
    python text_parser.py calc    "2 * (3 + 4) ^ 2 - sqrt(16)"
    python text_parser.py logs    path/to/app.log
"""

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterator


# ---------------------------------------------------------------------------
# 1. ENTITY EXTRACTOR
# ---------------------------------------------------------------------------
PATTERNS = {
    "email":   r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
    "url":     r"https?://[^\s<>\"']+",
    "phone":   r"(?<![\w.-])(?:\+\d{1,3}[ -]?)?\d(?:[ -]?\d){8,11}(?!\d)",
    "date":    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b",
    "ipv4":    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "hashtag": r"(?<!\w)#\w+",
    "mention": r"(?<!\w)@\w+",
    "money":   r"(?:₹|\$|€|£)\s?\d[\d,]*(?:\.\d+)?",
}
_COMPILED = {name: re.compile(p) for name, p in PATTERNS.items()}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Return {entity_type: [unique matches in order of appearance]}."""
    found = {}
    for name, rx in _COMPILED.items():
        seen, out = set(), []
        for m in rx.finditer(text):
            val = m.group().rstrip(".,;:!?)")
            if val not in seen:
                seen.add(val)
                out.append(val)
        if out:
            found[name] = out
    # An email's domain part can look like a mention; drop those false positives.
    if "mention" in found and "email" in found:
        domains = {e.split("@")[1] for e in found["email"]}
        found["mention"] = [m for m in found["mention"] if m[1:] not in domains]
        if not found["mention"]:
            del found["mention"]
    return found


# ---------------------------------------------------------------------------
# 2. TOKENIZER
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int

    def __repr__(self):
        return f"{self.kind}({self.value!r})@{self.pos}"


TOKEN_SPEC = [
    ("NUMBER", r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"),
    ("STRING", r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"),
    ("IDENT",  r"[A-Za-z_]\w*"),
    ("OP",     r"\*\*|[+\-*/%^=<>!]=?"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA",  r","),
    ("SKIP",   r"\s+"),
    ("ERROR",  r"."),
]
_MASTER = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC))


def tokenize(text: str) -> Iterator[Token]:
    for m in _MASTER.finditer(text):
        kind = m.lastgroup
        if kind == "SKIP":
            continue
        if kind == "ERROR":
            raise SyntaxError(f"Unexpected character {m.group()!r} at position {m.start()}")
        yield Token(kind, m.group(), m.start())


# ---------------------------------------------------------------------------
# 3. LOG PARSER
# ---------------------------------------------------------------------------
LOG_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+"
    r"\[(?P<level>[A-Z]+)\]\s+"
    r"(?P<module>[\w.]+):\s+"
    r"(?P<msg>.*)$"
)
KV_RE = re.compile(r'(?P<key>\w+)=(?P<val>"[^"]*"|\S+)')


def parse_log_line(line: str) -> dict | None:
    m = LOG_RE.match(line.strip())
    if not m:
        return None
    rec = m.groupdict()
    rec["fields"] = {k: v.strip('"') for k, v in KV_RE.findall(rec["msg"])}
    return rec


def analyze_logs(lines) -> dict:
    records, bad = [], 0
    for line in lines:
        if not line.strip():
            continue
        rec = parse_log_line(line)
        if rec:
            records.append(rec)
        else:
            bad += 1
    levels = Counter(r["level"] for r in records)
    by_module = Counter(r["module"] for r in records)
    errors_by_ip = defaultdict(int)
    for r in records:
        if r["level"] in ("ERROR", "CRITICAL") and "ip" in r["fields"]:
            errors_by_ip[r["fields"]["ip"]] += 1
    return {
        "parsed": len(records),
        "unparsed": bad,
        "levels": dict(levels),
        "top_modules": by_module.most_common(3),
        "errors_by_ip": dict(errors_by_ip),
        "time_range": (records[0]["ts"], records[-1]["ts"]) if records else None,
    }


# ---------------------------------------------------------------------------
# 4. EXPRESSION PARSER (recursive descent)
# ---------------------------------------------------------------------------
#   expr    := term (('+'|'-') term)*
#   term    := unary (('*'|'/'|'%') unary)*
#   unary   := '-' unary | pow
#   pow     := primary (('^'|'**') unary)?      (right-associative, so -2^2 = -4)
#   primary := NUMBER | IDENT '(' args ')' | IDENT | '(' expr ')'
FUNCS = {"sqrt": math.sqrt, "abs": abs, "sin": math.sin, "cos": math.cos,
         "log": math.log, "round": round, "max": max, "min": min}
CONSTS = {"pi": math.pi, "e": math.e}


class Parser:
    def __init__(self, text: str):
        self.tokens = list(tokenize(text))
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def eat(self, kind=None, value=None):
        t = self.peek()
        if t is None or (kind and t.kind != kind) or (value and t.value != value):
            raise SyntaxError(f"Expected {value or kind}, got {t}")
        self.i += 1
        return t

    def parse(self):
        result = self.expr()
        if self.peek():
            raise SyntaxError(f"Unexpected token {self.peek()}")
        return result

    def expr(self):
        v = self.term()
        while (t := self.peek()) and t.kind == "OP" and t.value in "+-":
            self.i += 1
            r = self.term()
            v = v + r if t.value == "+" else v - r
        return v

    def term(self):
        v = self.power()
        while (t := self.peek()) and t.kind == "OP" and t.value in ("*", "/", "%"):
            self.i += 1
            r = self.power()
            if t.value == "*":
                v *= r
            elif r == 0:
                raise ZeroDivisionError("Division by zero")
            else:
                v = v / r if t.value == "/" else v % r
        return v

    def power(self):
        base = self.unary()
        return base

    def unary(self):
        if (t := self.peek()) and t.kind == "OP" and t.value == "-":
            self.i += 1
            return -self.unary()
        return self.pow_()

    def pow_(self):
        base = self.primary()
        if (t := self.peek()) and t.kind == "OP" and t.value in ("^", "**"):
            self.i += 1
            return base ** self.unary()  # right-assoc; allows 2 ^ -1
        return base

    def primary(self):
        t = self.peek()
        if t is None:
            raise SyntaxError("Unexpected end of expression")
        if t.kind == "NUMBER":
            self.i += 1
            return float(t.value) if any(c in t.value for c in ".eE") else int(t.value)
        if t.kind == "IDENT":
            self.i += 1
            if self.peek() and self.peek().kind == "LPAREN":
                if t.value not in FUNCS:
                    raise NameError(f"Unknown function '{t.value}'")
                self.eat("LPAREN")
                args = []
                if self.peek() and self.peek().kind != "RPAREN":
                    args.append(self.expr())
                    while self.peek() and self.peek().kind == "COMMA":
                        self.i += 1
                        args.append(self.expr())
                self.eat("RPAREN")
                return FUNCS[t.value](*args)
            if t.value in CONSTS:
                return CONSTS[t.value]
            raise NameError(f"Unknown name '{t.value}'")
        if t.kind == "LPAREN":
            self.i += 1
            v = self.expr()
            self.eat("RPAREN")
            return v
        raise SyntaxError(f"Unexpected token {t}")


def calculate(text: str):
    return Parser(text).parse()


# ---------------------------------------------------------------------------
# CLI + DEMO
# ---------------------------------------------------------------------------
SAMPLE_TEXT = """Contact Priya at priya.k@example.com or support@acme.io.
Call +91 98765 4321 or 080-2345 6789. Meeting on 2026-11-20 (also 20/11/2026).
Server 192.168.1.10 and 10.0.0.256(invalid) were checked. Budget: ₹1,50,000 or $2,500.50.
Docs: https://docs.python.org/3/library/re.html. Follow @pythondev #regex #parsing"""

SAMPLE_LOG = """2026-11-13 09:00:01 [INFO] auth: Login ok user=alice ip=10.0.0.5
2026-11-13 09:00:07 [WARNING] db: Slow query ms=1840 table="user sessions"
2026-11-13 09:01:12 [ERROR] auth: Login failed user=bob ip=10.0.0.9 reason="bad password"
2026-11-13 09:01:15 [ERROR] auth: Login failed user=bob ip=10.0.0.9 reason="bad password"
this line is garbage and will not parse
2026-11-13 09:02:30 [INFO] api: GET /items status=200 ms=34
2026-11-13 09:03:44 [CRITICAL] db: Connection lost host=db1 ip=10.0.0.20"""


def _print_dict(d: dict, indent=2):
    for k, v in d.items():
        print(f"{' ' * indent}{k}: {v}")


def run_demo():
    print("=== 1. ENTITY EXTRACTION ===")
    _print_dict(extract_entities(SAMPLE_TEXT))

    print("\n=== 2. TOKENIZER ===")
    print(" ", list(tokenize('total = price * (1 + 0.18) - discount("VIP")')))

    print("\n=== 3. LOG ANALYSIS ===")
    _print_dict(analyze_logs(SAMPLE_LOG.splitlines()))
    print("  sample record:", parse_log_line(SAMPLE_LOG.splitlines()[2]))

    print("\n=== 4. EXPRESSION PARSER ===")
    for expr in ["2 + 3 * 4", "2 * (3 + 4) ^ 2 - sqrt(16)", "-2 ^ 2", "max(3, 9, 4) / 2",
                 "10 / (5 - 5)", "2 +", "foo(3)"]:
        try:
            print(f"  {expr:32} = {calculate(expr)}")
        except Exception as e:
            print(f"  {expr:32} ! {type(e).__name__}: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Advanced text parsing mini project")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run all demos")
    for name, help_ in [("extract", "extract entities from text"),
                        ("tokens", "tokenize text"),
                        ("calc", "evaluate a math expression"),
                        ("logs", "analyze a log file")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("arg", help="text (or file path for 'logs')")
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        run_demo()
    elif args.cmd == "extract":
        _print_dict(extract_entities(args.arg))
    elif args.cmd == "tokens":
        print(list(tokenize(args.arg)))
    elif args.cmd == "calc":
        try:
            print(calculate(args.arg))
        except Exception as e:
            sys.exit(f"{type(e).__name__}: {e}")
    elif args.cmd == "logs":
        with open(args.arg, encoding="utf-8") as f:
            _print_dict(analyze_logs(f))


if __name__ == "__main__":
    main()
