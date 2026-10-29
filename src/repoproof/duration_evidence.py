"""Bounded evidence checks for English duration descriptions.

Deterministic rules, not an additional neural model. No training labels, sentence
IDs, or fixture names are consulted. Results remain review candidates. Code is
parsed, never executed; units are taken only from parameter-bound docstrings.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal
import re

from .core import MarkdownBlock, ParameterChange

UNITS = {"ms": Decimal("0.001"), "millisecond": Decimal("0.001"), "milliseconds": Decimal("0.001"),
         "second": Decimal(1), "seconds": Decimal(1), "sec": Decimal(1), "secs": Decimal(1),
         "minute": Decimal(60), "minutes": Decimal(60), "min": Decimal(60), "mins": Decimal(60),
         "hour": Decimal(3600), "hours": Decimal(3600)}
WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), map(Decimal, range(20))))
WORDS.update({"twenty": Decimal(20), "thirty": Decimal(30), "forty": Decimal(40), "fifty": Decimal(50),
              "sixty": Decimal(60), "seventy": Decimal(70), "eighty": Decimal(80), "ninety": Decimal(90),
              "half": Decimal("0.5"), "quarter": Decimal("0.25")})
_UNIT = "|".join(sorted(UNITS, key=len, reverse=True))
_NUMBER = "|".join(sorted(WORDS, key=len, reverse=True))
_DURATION = re.compile(r"(?<![\w.+-])(?P<amount>\d+(?:\.\d+)?|" + _NUMBER + r")\s+(?:of\s+)?(?:(?:a|an)\s+)?(?P<unit>" + _UNIT + r")\b", re.I)
_WAIT = re.compile(r"\b(?:timeout|time\s*out|waiting(?:\s+time)?|wait|response\s+wait|deadline|abandoned\s+after)\b", re.I)
_DEFAULT = re.compile(r"\b(?:by\s+default|default\s+(?:waiting\s+time|timeout|response\s+wait)|unless\s+overridden)\b", re.I)
_HISTORY = re.compile(r"\b(?:migration|history|historical|previously|formerly|old\s+default|version\s+\d|used\s+to)\b", re.I)
_QUALIFIED = re.compile(r"\b(?:if|may|might|not|never|roughly|approximately|about|at\s+least|at\s+most|up\s+to|more\s+than|less\s+than|explicitly|isn['’]t|doesn['’]t|don['’]t|can['’]t)\b", re.I)


def identifier(name: str, text: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", text) is not None


@dataclass(frozen=True)
class FunctionDoc:
    symbol: str
    parameters: tuple[str, ...]
    text: str
    line: int


def function_docs(source: str) -> dict[str, FunctionDoc]:
    tree = ast.parse(source)
    found = {}
    duplicates = set()

    def walk(nodes, scope):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, [*scope, node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = ".".join([*scope, node.name])
                text = ast.get_docstring(node, clean=False) or ""
                line = node.body[0].lineno if text else node.lineno
                params = tuple(arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs])
                if symbol in found:
                    duplicates.add(symbol)
                found[symbol] = FunctionDoc(symbol, params, text, line)
                walk(node.body, [*scope, node.name, "<locals>"])
            else:
                for _, value in ast.iter_fields(node):
                    if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                        walk(value, scope)
    walk(tree.body, [])
    return {key: value for key, value in found.items() if key not in duplicates}


def unit_binding(doc: FunctionDoc, parameter: str) -> dict | None:
    matches = []
    for offset, line in enumerate(doc.text.splitlines()):
        for clause in re.split(r"[;.]\s+", line):
            if not identifier(parameter, clause) or not _WAIT.search(clause):
                continue
            if any(identifier(other, clause) for other in doc.parameters if other != parameter):
                continue
            units = list(re.finditer(r"\b(?:in|measured\s+in)\s+(" + _UNIT + r")\b", clause, re.I))
            if len(units) == 1:
                matches.append({"unit": units[0].group(1).lower(), "line": doc.line + offset, "text": line.strip()})
    return matches[0] if len(matches) == 1 else None


def durations(text: str) -> list[dict]:
    # Normalize only hyphenated words, preserving signed numeric syntax.
    normalized = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)
    rows = []
    for match in _DURATION.finditer(normalized):
        prefix = normalized[:match.start()].rstrip()
        previous = re.search(r"([A-Za-z]+|[+\-/\d.])$", prefix)
        # Do not read the tail of 'twenty five', 'one hundred thirty', 1/2,
        # or a signed/range expression as an independent simple quantity.
        if previous and (previous.group().lower() in set(WORDS) | {"hundred", "thousand", "million", "minus", "plus"}
                         or re.fullmatch(r"[+\-/\d.]", previous.group())):
            return []
        word = match.group("amount").lower()
        amount = WORDS[word] if word in WORDS else Decimal(word)
        rows.append({"text": text[match.start():match.end()],
                     "seconds": str(amount * UNITS[match.group("unit").lower()]),
                     "offset": match.start()})
    return rows


class DurationEvidenceChecker:
    def __init__(self, before: str, after: str):
        self.before = function_docs(before)
        self.after = function_docs(after)
        self.symbols = set(self.before) | set(self.after)

    def check(self, change: ParameterChange, block: MarkdownBlock) -> dict:
        result = {"symbol": change.symbol, "parameter": change.parameter, "block_id": block.block_id,
                  "status": "manual_review", "reason": "", "evidence": [],
                  "method": "bounded_duration_rules_v1", "automatic_edit": False}
        evidence = result["evidence"]

        def finish(reason, status="manual_review"):
            result.update(reason=reason, status=status)
            return result

        if block.kind != "paragraph":
            return finish("non_prose_block")
        scoped = []
        for heading in block.headings:
            text = heading.text.strip().strip("`")
            known = {symbol for symbol in self.symbols if identifier(symbol, text) or identifier(symbol.rsplit(".", 1)[-1], text)}
            if len(known) > 1:
                return finish("ambiguous_function")
            if known:
                scoped.append((next(iter(known)), heading))
            elif re.fullmatch(r"[A-Za-z_]\w*(?:\.\w+)*(?:\(\))?", text) and ("_" in text or text.endswith("()")):
                scoped.append((text.removesuffix("()"), heading))
        body_symbols = {symbol for symbol in self.symbols if identifier(symbol, block.text) or identifier(symbol.rsplit(".", 1)[-1], block.text)}
        if len(body_symbols) > 1:
            return finish("ambiguous_function")
        if scoped and scoped[-1][0] != change.symbol:
            other, heading = scoped[-1]
            evidence.append({"kind": "documented_symbol", "text": heading.text, "line": heading.line})
            return finish("different_documented_symbol", "excluded_from_change")
        if body_symbols and body_symbols != {change.symbol}:
            return finish("different_function_in_text", "excluded_from_change")
        if not scoped and body_symbols != {change.symbol}:
            return finish("function_not_established")
        if scoped:
            evidence.append({"kind": "documented_symbol", "text": scoped[-1][1].text, "line": scoped[-1][1].line})
        else:
            evidence.append({"kind": "documented_symbol", "text": block.text, "line": block.start_line})
        context = " / ".join(block.heading_path) + "\n" + block.text
        if _HISTORY.search(context):
            return finish("historical_context", "excluded_from_change")
        if _QUALIFIED.search(block.text):
            return finish("qualified_or_override_claim")
        if not _DEFAULT.search(block.text):
            return finish("default_behavior_not_established")
        if not _WAIT.search(block.text):
            return finish("waiting_behavior_not_established")
        if len(re.split(r"(?<=[.!?])\s+", block.text.strip())) > 1:
            return finish("multiple_sentences_require_review")
        if change.symbol not in self.before or change.symbol not in self.after:
            return finish("source_function_ambiguous")
        old_doc, new_doc = self.before[change.symbol], self.after[change.symbol]
        old_unit, new_unit = unit_binding(old_doc, change.parameter), unit_binding(new_doc, change.parameter)
        if old_unit is None or new_unit is None:
            return finish("source_unit_not_established")
        related = [param for param in new_doc.parameters if unit_binding(new_doc, param)]
        if len(related) > 1 and not identifier(change.parameter, block.text):
            return finish("ambiguous_wait_parameter")
        if any(identifier(param, block.text) for param in related if param != change.parameter):
            return finish("different_or_multiple_wait_parameters")
        if type(change.old.value) not in (int, float) or type(change.new.value) not in (int, float):
            return finish("non_numeric_code_default")
        old_seconds = Decimal(str(change.old.value)) * UNITS[old_unit["unit"]]
        new_seconds = Decimal(str(change.new.value)) * UNITS[new_unit["unit"]]
        if not old_seconds.is_finite() or not new_seconds.is_finite() or min(old_seconds, new_seconds) < 0:
            return finish("unsupported_code_duration")
        evidence.extend([{"kind": "old_source_unit", **old_unit}, {"kind": "new_source_unit", **new_unit},
                         {"kind": "code_defaults_in_seconds", "old": str(old_seconds), "new": str(new_seconds),
                          "old_line": change.before_line, "new_line": change.after_line}])
        values = durations(block.text)
        if len(values) != 1:
            return finish("ambiguous_or_unsupported_document_duration")
        duration = values[0]
        evidence.append({"kind": "document_duration", **duration,
                         "line": block.start_line + block.text[:duration["offset"]].count("\n")})
        documented_seconds = Decimal(duration["seconds"])
        if documented_seconds == new_seconds:
            return finish("matches_new_declared_default", "consistent_with_new_default")
        if documented_seconds == old_seconds:
            return finish("matches_old_declared_default", "needs_update_candidate")
        return finish("document_value_matches_neither_version")
