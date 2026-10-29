"""Conservative, non-executing evidence collection for documentation review.

This module reports review candidates, never proves a natural-language contradiction.
It uses Python's AST and literal evaluator only; inspected code is never imported.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class _Sentinel(Enum):
    UNKNOWN = "unknown"
    MISSING = "missing"


UNKNOWN = _Sentinel.UNKNOWN
MISSING = _Sentinel.MISSING


def _literal_key(value: Any) -> Any:
    """Preserve types: bool, int, None, and strings are not interchangeable."""
    name = type(value).__name__
    if isinstance(value, (list, tuple)):
        return name, tuple(_literal_key(item) for item in value)
    if isinstance(value, dict):
        return name, tuple((_literal_key(k), _literal_key(v)) for k, v in value.items())
    if isinstance(value, (set, frozenset)):
        return name, tuple(sorted((_literal_key(item) for item in value), key=repr))
    return name, value


def _json_literal(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"type": "float", "display": repr(value)}
    if type(value) is list:
        return [_json_literal(item) for item in value]
    if type(value) is tuple:
        return {"type": "tuple", "items": [_json_literal(item) for item in value]}
    if type(value) is dict:
        return {"type": "dict", "items": [[_json_literal(k), _json_literal(v)] for k, v in value.items()]}
    if type(value) is set:
        return {"type": "set", "items": [_json_literal(item) for item in sorted(value, key=repr)]}
    if type(value) is bytes:
        return {"type": "bytes", "hex": value.hex()}
    if type(value) is complex:
        return {"type": "complex", "real": _json_literal(value.real), "imag": _json_literal(value.imag)}
    if value is Ellipsis:
        return {"type": "ellipsis", "display": "..."}
    raise TypeError(f"Unexpected literal type: {type(value).__name__}")


@dataclass(frozen=True)
class DefaultValue:
    value: Any
    expression: str
    line: int
    end_line: int

    @property
    def known(self) -> bool:
        return self.value is not UNKNOWN and self.value is not MISSING

    @property
    def type(self) -> str:
        return type(self.value).__name__ if self.known else self.value.value

    @property
    def display(self) -> str:
        return repr(self.value) if self.known else self.expression

    def to_dict(self) -> dict[str, Any]:
        result = {
            "state": "known" if self.known else self.value.value,
            "type": self.type,
            "display": self.display,
            "expression": self.expression,
            "line": self.line,
            "end_line": self.end_line,
        }
        if self.known:
            result["value"] = _json_literal(self.value)
        return result


@dataclass(frozen=True)
class ParameterChange:
    symbol: str
    parameter: str
    old: DefaultValue
    new: DefaultValue
    before_line: int
    after_line: int
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "parameter": self.parameter,
            "old": self.old.to_dict(),
            "new": self.new.to_dict(),
            "before_line": self.before_line,
            "after_line": self.after_line,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class Heading:
    text: str
    level: int
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "level": self.level, "line": self.line}


@dataclass(frozen=True)
class MarkdownBlock:
    block_id: str
    text: str
    start_line: int
    end_line: int
    headings: tuple[Heading, ...]
    kind: str = "paragraph"

    @property
    def heading_path(self) -> tuple[str, ...]:
        return tuple(heading.text for heading in self.headings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "headings": [heading.to_dict() for heading in self.headings],
            "heading_path": list(self.heading_path),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Evidence:
    kind: str
    text: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "start_line": self.start_line, "end_line": self.end_line}


@dataclass(frozen=True)
class Candidate:
    change: ParameterChange
    block_id: str
    evidence: tuple[Evidence, ...]
    status: str = "needs_update_candidate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "change": self.change.to_dict(),
            "block_id": self.block_id,
            "status": self.status,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class Abstention:
    reason: str
    symbol: str = ""
    parameter: str = ""
    block_id: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "symbol": self.symbol, "parameter": self.parameter, "block_id": self.block_id, "detail": self.detail}


@dataclass(frozen=True)
class AnalysisResult:
    changes: tuple[ParameterChange, ...]
    blocks: tuple[MarkdownBlock, ...]
    candidates: tuple[Candidate, ...]
    abstentions: tuple[Abstention, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": [item.to_dict() for item in self.changes],
            "blocks": [item.to_dict() for item in self.blocks],
            "candidates": [item.to_dict() for item in self.candidates],
            "abstentions": [item.to_dict() for item in self.abstentions],
        }


@dataclass(frozen=True)
class _Function:
    symbol: str
    parameters: dict[str, DefaultValue]


def _default(source: str, node: ast.expr | None, arg: ast.arg) -> DefaultValue:
    if node is None:
        return DefaultValue(MISSING, "", arg.lineno, arg.end_lineno or arg.lineno)
    expression = ast.get_source_segment(source, node) or ""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
        value = UNKNOWN
    return DefaultValue(value, expression, node.lineno, node.end_lineno or node.lineno)


def _functions(source: str) -> tuple[dict[str, _Function], set[str]]:
    """Find lexical qualified names; duplicate definitions remain ambiguous."""
    tree = ast.parse(source)
    found: dict[str, _Function] = {}
    duplicate: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            for child in node.body:
                self.visit(child)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            symbol = ".".join(self.scope + [node.name])
            positional = list(node.args.posonlyargs) + list(node.args.args)
            defaults: list[ast.expr | None] = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
            parameters = {arg.arg: _default(source, default, arg) for arg, default in zip(positional, defaults)}
            parameters.update({arg.arg: _default(source, default, arg) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)})
            if symbol in found:
                duplicate.add(symbol)
            found[symbol] = _Function(symbol, parameters)
            self.scope.extend([node.name, "<locals>"])
            for child in node.body:
                self.visit(child)
            del self.scope[-2:]

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    return found, duplicate


def split_markdown(markdown: str) -> tuple[MarkdownBlock, ...]:
    """Split ATX/single-line Setext headings and paragraphs, with 1-based lines.

    Fences, block quotes, indented code, and HTML comments stay available for
    retrieval, but the evidence rule does not interpret them as factual prose.
    CRLF/LF are normalized in block text without changing line numbers.
    """
    lines = markdown.splitlines()
    headings: list[Heading] = []
    blocks: list[MarkdownBlock] = []
    buffer: list[str] = []
    start = 0

    def flush(kind: str = "paragraph") -> None:
        nonlocal buffer
        if not buffer:
            return
        if kind == "paragraph" and all(line.lstrip().startswith(">") for line in buffer if line.strip()):
            kind = "quote"
        elif kind == "paragraph" and all(line.startswith(("    ", "\t")) for line in buffer if line.strip()):
            kind = "code"
        blocks.append(MarkdownBlock(f"b{len(blocks) + 1:04d}", "\n".join(buffer), start, start + len(buffer) - 1, tuple(headings), kind))
        buffer = []

    def add_heading(text: str, level: int, line: int) -> None:
        while headings and headings[-1].level >= level:
            headings.pop()
        headings.append(Heading(text.strip(), level, line))

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            flush()
            start = i + 1
            delimiter = fence.group(1)
            buffer = [line]
            i += 1
            while i < len(lines):
                buffer.append(lines[i])
                is_end = re.match(r"^ {0,3}" + re.escape(delimiter[0]) + "{" + str(len(delimiter)) + r",}\s*$", lines[i])
                i += 1
                if is_end:
                    break
            flush("code")
            continue
        if line.lstrip().startswith("<!--"):
            flush()
            start = i + 1
            buffer = [line]
            i += 1
            while "-->" not in buffer[-1] and i < len(lines):
                buffer.append(lines[i])
                i += 1
            flush("html_comment")
            continue
        heading = re.match(r"^ {0,3}(#{1,6})(?:\s+|$)(.*)$", line)
        if heading:
            flush()
            text = re.sub(r"\s+#+\s*$", "", heading.group(2))
            add_heading(text, len(heading.group(1)), i + 1)
        elif line.strip() and not buffer and i + 1 < len(lines) and re.match(r"^ {0,3}(=+|-+)\s*$", lines[i + 1]):
            add_heading(line, 1 if lines[i + 1].lstrip().startswith("=") else 2, i + 1)
            i += 1
        elif not line.strip():
            flush()
        else:
            if not buffer:
                start = i + 1
            buffer.append(line)
        i += 1
    flush()
    return tuple(blocks)


def _identifier_matches(name: str, text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", text))


def _symbol_context(block: MarkdownBlock, symbols: set[str]) -> tuple[set[str], bool]:
    text = "\n".join(block.heading_path) + "\n" + block.text
    qualified = {symbol for symbol in symbols if "." in symbol and _identifier_matches(symbol, text)}
    resolved = set(qualified)
    ambiguous = False
    leaves: dict[str, set[str]] = {}
    for symbol in symbols:
        leaves.setdefault(symbol.rsplit(".", 1)[-1], set()).add(symbol)
    for leaf, owners in leaves.items():
        if not _identifier_matches(leaf, text):
            continue
        if len(owners) == 1:
            resolved.update(owners)
        elif not (owners & qualified):
            ambiguous = True
    return resolved, ambiguous


_DEFAULT_CLAIM = re.compile(
    r"(?<![\w.])`?(?P<parameter>[A-Za-z_][A-Za-z0-9_]*)`?"
    r"(?:\s*(?:'s|的))?\s*"
    r"(?:(?:has|with|uses?)\s+(?:a\s+)?)?"
    r"(?:defaults?\s+to\s+|default(?:\s+value)?\s*(?:is|of|:|=)\s*|"
    r"默认(?:值)?\s*(?:为|是|:|：|=)\s*)",
    re.IGNORECASE,
)

_DEFAULT_FIRST_CLAIM = re.compile(
    r"\bdefault\s+`?(?P<parameter>[A-Za-z_][A-Za-z0-9_]*)`?"
    r"(?:\s+value)?"
    r"(?:\s+for\s+`?[A-Za-z_][A-Za-z0-9_.]*`?)?"
    r"\s*(?:is|of|:|=)\s*",
    re.IGNORECASE,
)

_VALUE_TOKEN = re.compile(
    r"`(?P<code>[^`\n]+)`|"
    r"(?P<quoted>(?:[bB])?(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'))|"
    r"(?P<bare>(?:None|True|False|[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?))(?![\w]|\.[\w.])"
)
_HISTORICAL = re.compile(
    r"\b(?:previously|formerly|historical|history|used\s+to|prior\s+to|before\s+(?:v\d|version)|no\s+longer)\b|"
    r"旧版|原先|过去|曾经|变更前|此前|不再|迁移指南|历史记录", re.IGNORECASE,
)
_UNCERTAIN = re.compile(
    r"\b(?:if|may|might|perhaps|not|never|incorrect|hypothetical)\b|如果|假如|可能|并非|不是|不应|错误", re.IGNORECASE,
)
_EXPRESSION_CONTINUATION = re.compile(r"\s*(?:[+*/%|&^<>=-]|(?:and|or)\b)")


@dataclass(frozen=True)
class _Claim:
    parameter: str
    value: Any
    start: int
    end: int


def _claims(text: str) -> list[_Claim]:
    claims: list[_Claim] = []
    matches = sorted(
        [match for pattern in (_DEFAULT_CLAIM, _DEFAULT_FIRST_CLAIM)
         for match in pattern.finditer(text)], key=lambda item: item.start()
    )
    for match in matches:
        token = _VALUE_TOKEN.match(text, match.end())
        if token is None:
            claims.append(_Claim(match.group("parameter"), UNKNOWN, match.start(), match.end()))
            continue
        literal = token.group("code") or token.group("quoted") or token.group("bare")
        try:
            value = ast.literal_eval(literal)
        except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
            value = UNKNOWN
        if _EXPRESSION_CONTINUATION.match(text, token.end()):
            value = UNKNOWN
        claims.append(_Claim(match.group("parameter"), value, match.start(), token.end()))
    return claims


def _symbol_evidence(block: MarkdownBlock, symbol: str) -> Evidence:
    names = [symbol, symbol.rsplit(".", 1)[-1]]
    for name in names:
        for heading in reversed(block.headings):
            if _identifier_matches(name, heading.text):
                return Evidence("symbol_context", heading.text, heading.line, heading.line)
        matches = _identifier_matches(name, block.text)
        if matches:
            match = matches[0]
            line = block.start_line + block.text[:match.start()].count("\n")
            return Evidence("symbol_context", match.group(), line, line)
    raise AssertionError("Resolved symbol has no lexical evidence")


def analyze(
    before_source: str,
    after_source: str,
    markdown: str,
    *,
    source_path: str = "",
    document_path: str = "",
) -> AnalysisResult:
    """Compare one Python file's defaults against one Markdown document.

    Only modified literal defaults on a matched function and parameter produce
    changes. Added/removed parameters/defaults and unknown expressions abstain.
    Document identity uses lexical qualified names; aliases and semantics abstain.
    ``document_path`` is accepted for callers; block line numbers refer to that
    input, and no filesystem reads or writes are performed here.
    """
    blocks = split_markdown(markdown)
    abstentions: list[Abstention] = []
    functions: list[dict[str, _Function]] = []
    duplicates: list[set[str]] = []
    for label, source in (("before", before_source), ("after", after_source)):
        try:
            found, duplicate = _functions(source)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
            line = getattr(error, "lineno", None)
            abstentions.append(Abstention("source_parse_error", detail=f"{label}: {type(error).__name__}, line {line}"))
            functions.append({})
            duplicates.append(set())
        else:
            functions.append(found)
            duplicates.append(duplicate)
    if any(item.reason == "source_parse_error" for item in abstentions):
        return AnalysisResult((), blocks, (), tuple(abstentions))
    before, after = functions
    ambiguous_symbols = duplicates[0] | duplicates[1]
    all_symbols = set(before) | set(after)
    changes: list[ParameterChange] = []
    for symbol in sorted(all_symbols):
        if symbol in ambiguous_symbols:
            abstentions.append(Abstention("duplicate_symbol_definition", symbol=symbol, detail="Repeated lexical definitions cannot be uniquely paired."))
            continue
        if symbol not in before or symbol not in after:
            abstentions.append(Abstention("unmatched_symbol", symbol=symbol, detail="Added or removed functions are outside the literal-default comparison."))
            continue
        old_params, new_params = before[symbol].parameters, after[symbol].parameters
        for parameter in sorted(set(old_params) | set(new_params)):
            if parameter not in old_params or parameter not in new_params:
                abstentions.append(Abstention("parameter_added_or_removed", symbol, parameter))
                continue
            old, new = old_params[parameter], new_params[parameter]
            if old.value is UNKNOWN or new.value is UNKNOWN:
                abstentions.append(Abstention("non_literal_default", symbol, parameter, detail="At least one default is not safely literal-evaluable; no repository code was executed."))
                continue
            if old.value is MISSING or new.value is MISSING:
                if old.value is not new.value:
                    abstentions.append(Abstention("default_added_or_removed", symbol, parameter))
                continue
            if _literal_key(old.value) != _literal_key(new.value):
                changes.append(ParameterChange(symbol, parameter, old, new, old.line, new.line, source_path))

    candidates: list[Candidate] = []
    for change in changes:
        relevant_blocks = 0
        for block in blocks:
            context = "\n".join(block.heading_path) + "\n" + block.text
            parameter_present = bool(_identifier_matches(change.parameter, block.text))
            symbol_present = bool(_identifier_matches(change.symbol, context) or _identifier_matches(change.symbol.rsplit(".", 1)[-1], context))
            if not (parameter_present or symbol_present):
                continue
            relevant_blocks += 1

            def abstain(reason: str, detail: str = "") -> None:
                abstentions.append(Abstention(reason, change.symbol, change.parameter, block.block_id, detail))

            if block.kind != "paragraph":
                abstain("non_prose_block", "Code, quotations, and comments require a separate interpretation.")
                continue
            identities, ambiguous = _symbol_context(block, all_symbols)
            if ambiguous:
                abstain("ambiguous_symbol_identity", "An unqualified function name identifies multiple lexical symbols.")
                continue
            if len(identities) > 1:
                abstain("multiple_symbol_context", "The paragraph or heading context mentions several functions.")
                continue
            if identities != {change.symbol}:
                abstain("symbol_identity_not_established", "No unique lexical identity for the changed function in this block.")
                continue
            if _HISTORICAL.search(context):
                abstain("historical_context", "Historical or migration wording may intentionally describe an older default.")
                continue
            if _UNCERTAIN.search(block.text):
                abstain("qualified_or_negated_claim", "Conditional, uncertain, or negative wording needs semantic review.")
                continue
            claims = [claim for claim in _claims(block.text) if claim.parameter == change.parameter]
            if not claims:
                abstain("explicit_default_claim_missing", "Requires a supported parameter-to-default-value statement, not token co-occurrence.")
                continue
            if any(claim.value is UNKNOWN for claim in claims):
                abstain("document_literal_unknown", "A documented value lacks an unambiguous Python literal type.")
                continue
            keys = {_literal_key(claim.value) for claim in claims}
            if len(keys) > 1:
                abstain("conflicting_document_defaults", "Several different defaults are stated for the same parameter.")
                continue
            matching = [claim for claim in claims if _literal_key(claim.value) == _literal_key(change.old.value)]
            if not matching:
                abstain("document_default_not_old_value", "Explicit documented value does not match the old value and its type.")
                continue
            evidence = [_symbol_evidence(block, change.symbol)]
            for claim in matching:
                evidence.append(Evidence(
                    "parameter_default_old_value",
                    block.text[claim.start:claim.end],
                    block.start_line + block.text[:claim.start].count("\n"),
                    block.start_line + block.text[:claim.end].count("\n"),
                ))
            candidates.append(Candidate(change, block.block_id, tuple(evidence)))
        if not relevant_blocks:
            abstentions.append(Abstention("no_lexical_evidence", change.symbol, change.parameter, detail="All document blocks remain available for later semantic retrieval."))
    return AnalysisResult(tuple(changes), blocks, tuple(candidates), tuple(abstentions))


analyze_changes = analyze
