"""Traceable, label-free retrieval of prose from Markdown and reStructuredText.

This conservative splitter is not a full Markdown/RST interpreter. It never
executes directives, includes, substitutions, examples, or repository code.
Retrieval scores locate text; they do not establish documentation inconsistency.
"""
from __future__ import annotations

import hashlib
import json
import re
from time import perf_counter

from .bm25 import BM25Index
from .cli import lexical_rank
from .lesson import reciprocal_rank_fusion
from .semantic import MODEL_ID, REVISION, SemanticRetriever, SemanticUnavailable


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def split_document(text: str, path: str, format: str) -> list[dict]:
    """Keep every nonblank physical line once, with one-based inclusive ranges.

    Blank separators are omitted except inside contiguous code/directive blocks.
    ``text`` preserves source indentation and CR characters. Only LF defines
    physical lines, so Unicode paragraph separators cannot change source offsets.
    Ordinary list items include their indented/blank-separated continuation prose.
    Markdown lists containing embedded code or nested structures remain whole as
    ``other`` with ``incomplete=True``; their leading sentence is not isolated as
    a complete claim. RST directive bodies are retained as directives (not assumed
    prose); unsupported indented structures/tables are ``other``.
    """
    if not isinstance(text, str) or not isinstance(path, str):
        raise TypeError("text and path must be strings")
    fmt = format.lower().lstrip(".")
    if fmt in {"md", "markdown"}:
        fmt = "markdown"
    elif fmt in {"rst", "rest", "restructuredtext"}:
        fmt = "rst"
    else:
        raise ValueError("supported formats are markdown and rst")
    raw = text.split("\n")
    lines = [line.removesuffix("\r") for line in raw]
    blocks, headings, rst_levels = [], [], []
    n, i = len(lines), 0

    def add(start, stop, kind, **metadata):
        body = "\n".join(raw[start:stop])
        identity = {"path": path, "start_line": start + 1, "end_line": stop,
                    "kind": kind, "text": body}
        blocks.append({"block_id": _digest(identity), **identity,
                       "headings": [title for _, title in headings], "format": fmt, **metadata})

    def adornment(line):
        value = line.strip()
        return value[0] if len(value) >= 3 and len(set(value)) == 1 and value[0] in "=-~^\"'`:+*#_" else None

    def heading_at(pos):
        line = lines[pos]
        if fmt == "markdown":
            match = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                return len(match[1]), match[2], 1
            if (line.strip() and not line.startswith((" ", "\t")) and pos + 1 < n
                    and re.fullmatch(r" {0,3}(?:=+|-+)\s*", lines[pos + 1])):
                return (1 if lines[pos + 1].strip().startswith("=") else 2), line.strip(), 2
        else:
            char = adornment(line)
            if (char and pos + 2 < n and lines[pos + 1].strip()
                    and adornment(lines[pos + 2]) == char
                    and len(lines[pos + 2].strip()) >= len(lines[pos + 1].strip())):
                return (char, True), lines[pos + 1].strip(), 3
            if (line.strip() and not line.startswith((" ", "\t")) and not char and pos + 1 < n
                    and adornment(lines[pos + 1])
                    and len(lines[pos + 1].strip()) >= len(line.strip())):
                return (adornment(lines[pos + 1]), False), line.strip(), 2
        return None

    def indented_stop(start):
        stop, last, minimum_indent = start, start, None
        while stop < n:
            if lines[stop].strip() and not lines[stop].startswith((" ", "\t")):
                break
            if lines[stop].strip():
                expanded = lines[stop].expandtabs(8)
                indent = len(expanded) - len(expanded.lstrip())
                if minimum_indent is None:
                    minimum_indent = indent
                elif indent < minimum_indent:
                    break
                last = stop + 1
            stop += 1
        return last

    def list_start(line):
        return bool(re.match(r"^ {0,3}(?:[-+*]|\d+[.)])\s+", line))

    def markdown_list_item(start):
        """Conservative item boundary; indentation is relative to its marker.

        Four spaces are not inherently code inside a list. Lazy continuation is
        accepted before a blank; after a blank, continuation must be indented to
        the item content column. Mixed code/nested items stay opaque as a whole.
        """
        match = re.match(r"^( {0,3})(?:[-+*]|\d+[.)])([ \t]+)(.*)$", lines[start])
        base_indent = len(match[1])
        content_indent = len(lines[start][:match.start(3)].expandtabs(8))
        stop, reasons, open_fence = start + 1, [], None
        opening = re.match(r"(`{3,}|~{3,})", match[3])
        if opening:
            open_fence = opening[1]
            reasons.append("embedded fenced code")
        while stop < n:
            after_blank = not lines[stop].strip()
            candidate = stop
            while candidate < n and not lines[candidate].strip():
                candidate += 1
            if candidate == n:
                break
            current = lines[candidate]
            expanded = current.expandtabs(8)
            indent = len(expanded) - len(expanded.lstrip())
            stripped = current.lstrip()
            if open_fence:
                close = re.fullmatch(re.escape(open_fence[0]) + "{" + str(len(open_fence)) + r",}\s*", stripped)
                stop = candidate + 1
                if close:
                    open_fence = None
                continue
            next_item = re.match(r"^( *)(?:[-+*]|\d+[.)])\s+", expanded)
            if next_item and indent <= base_indent:
                break
            if after_blank and indent < content_indent:
                break
            if indent < content_indent and (heading_at(candidate) or stripped.startswith(("```", "~~~", "<", "|"))):
                break
            if next_item:
                reasons.append("nested list structure")
            opening = re.match(r"(`{3,}|~{3,})", stripped)
            if opening:
                open_fence = opening[1]
                reasons.append("embedded fenced code")
            elif after_blank and indent >= content_indent + 4:
                reasons.append("embedded indented code")
            elif stripped.startswith(("<", "|", "+--", "+==")):
                reasons.append("embedded opaque structure")
            stop = candidate + 1
        if open_fence:
            reasons.append("unclosed embedded fence")
        return stop, sorted(set(reasons))

    pending_literal = False
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        start, line = i, lines[i]
        # RST literal bodies may be separated from the introducing :: by blanks.
        if fmt == "rst" and pending_literal and line.startswith((" ", "\t")):
            i = indented_stop(i)
            add(start, i, "code")
            pending_literal = False
            continue
        pending_literal = False
        if fmt == "markdown":
            fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            if fence:
                marker, i = fence[1], i + 1
                closing = re.compile(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$")
                while i < n:
                    close = bool(closing.match(lines[i]))
                    i += 1
                    if close:
                        break
                add(start, i, "code")
                continue
        heading = heading_at(i)
        if heading:
            level, title, width = heading
            if fmt == "rst":
                if level not in rst_levels:
                    rst_levels.append(level)
                level = rst_levels.index(level) + 1
            headings[:] = [(depth, old) for depth, old in headings if depth < level]
            headings.append((level, title))
            i += width
            add(start, i, "heading")
            continue
        if fmt == "rst" and line.startswith(".. "):
            # Keep an unknown directive/comment/target and its body opaque.
            is_code = bool(re.match(r"^\.\.\s+(?:code|code-block|sourcecode)::", line))
            i += 1
            stop = indented_stop(i)
            i = max(i, stop)
            add(start, i, "code" if is_code else "directive")
            continue
        if fmt == "markdown" and list_start(line):
            i, reasons = markdown_list_item(i)
            if reasons:
                add(start, i, "other", incomplete=True,
                    parse_status="unsupported_mixed_list_item", parse_notes=reasons)
            else:
                add(start, i, "prose")
            continue
        if line.startswith(("    ", "\t")):
            i = indented_stop(i)
            add(start, i, "code" if fmt == "markdown" else "other")
            continue
        # Opaque table/HTML/adornment lines are not automatically natural prose.
        if (line.lstrip().startswith(("|", "<", "+--", "+==")) or adornment(line)
                or (fmt == "rst" and line.startswith((" ", "\t")))):
            i += 1
            while i < n and lines[i].strip() and (lines[i].lstrip().startswith(("|", "<", "+--", "+==")) or adornment(lines[i])):
                i += 1
            add(start, i, "other")
            continue
        # Split separate list items while retaining their wrapped continuation.
        i += 1
        while i < n and lines[i].strip():
            next_line = lines[i]
            if (heading_at(i) or list_start(next_line)
                    or next_line.startswith((".. ", "    ", "\t"))
                    or next_line.lstrip().startswith(("```", "~~~", "|", "<", "+--", "+=="))
                    or adornment(next_line)):
                break
            i += 1
        add(start, i, "prose")
        pending_literal = fmt == "rst" and lines[i - 1].rstrip().endswith("::")
    return blocks


_FORBIDDEN = {"label", "labels", "answer", "answers", "expected", "expected_answer",
              "gold", "gold_label", "ground_truth", "reference_label", "reference_labels",
              "correct_answer", "correct_answers", "pr_diff", "diff"}


def rank_documents(documents: list[dict], queries: list[dict], with_semantic=True) -> dict:
    """Rank an identical prose corpus for each method, without correctness labels.

    Only query_id/text and the documented block fields are consumed. Known label
    and diff fields are rejected, unrelated metadata is ignored. ``index`` always
    refers to the original input block, ``corpus_index`` to the prose-only corpus.
    Missing semantic infrastructure is reported, never replaced by lexical scores.
    """
    started = perf_counter()
    if not isinstance(documents, list) or not isinstance(queries, list):
        raise TypeError("documents and queries must be lists")
    blocks, query_rows, seen_blocks, seen_queries = [], [], set(), set()
    for block in documents:
        if not isinstance(block, dict) or _FORBIDDEN.intersection(block):
            raise ValueError("blocks must contain document content only; labels/diffs are forbidden")
        row = {key: block[key] for key in ("block_id", "path", "start_line", "end_line", "text", "headings", "kind")}
        if (any(not isinstance(row[key], str) for key in ("block_id", "path", "text", "kind"))
                or not isinstance(row["headings"], list) or any(not isinstance(x, str) for x in row["headings"])
                or not isinstance(row["start_line"], int) or not isinstance(row["end_line"], int)
                or row["start_line"] < 1 or row["end_line"] < row["start_line"]):
            raise ValueError("invalid document block fields")
        if row["block_id"] in seen_blocks:
            raise ValueError("duplicate block_id")
        seen_blocks.add(row["block_id"])
        if "format" in block:
            row["format"] = block["format"]
        for key in ("incomplete", "parse_status", "parse_notes"):
            if key in block:
                row[key] = block[key]
        blocks.append(row)
    for query in queries:
        if not isinstance(query, dict) or _FORBIDDEN.intersection(query):
            raise ValueError("queries must contain retrieval text only; labels/diffs are forbidden")
        row = {key: query[key] for key in ("query_id", "text")}
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError("query_id and text must be nonempty strings")
        if row["query_id"] in seen_queries:
            raise ValueError("duplicate query_id")
        seen_queries.add(row["query_id"])
        query_rows.append(row)
    corpus, excluded = [], []
    for index, block in enumerate(blocks):
        if block["kind"] == "prose" and block["text"].strip() and not block.get("incomplete"):
            corpus.append({"index": index, "corpus_index": len(corpus), "block_id": block["block_id"],
                           "retrieval_text": "\n".join([*block["headings"], block["text"]])})
        else:
            excluded.append({"index": index, "block_id": block["block_id"], "kind": block["kind"],
                             "reason": "incomplete/unsupported structure" if block.get("incomplete") else "only nonempty prose blocks are ranked"})
    texts = [item["retrieval_text"] for item in corpus]
    bm25 = BM25Index(texts, k1=1.2, b=0.75)
    model = {"requested": bool(with_semantic), "initialized": False, "encode_attempted": False,
             "actual_encode_completed": False, "model_id": MODEL_ID, "revision": REVISION,
             "status": "disabled" if not with_semantic else "not_called_empty_corpus_or_queries",
             "manifest": None, "manifest_sha256": None, "model_file_sha256": None}
    times = {"semantic_initialization_seconds": 0.0, "semantic_encoding_seconds": 0.0,
             "semantic_scoring_seconds": 0.0, "lexical_seconds": 0.0,
             "bm25_ranking_seconds": 0.0, "rrf_seconds": 0.0}
    similarities = None
    if with_semantic and texts and query_rows:
        phase = perf_counter()
        try:
            encoder = SemanticRetriever()
            times["semantic_initialization_seconds"] = perf_counter() - phase
            model.update(initialized=True, manifest=encoder.manifest,
                         manifest_sha256=_digest(encoder.manifest),
                         model_file_sha256=encoder.manifest["files"]["onnx/model.onnx"]["sha256"],
                         providers=encoder.session.get_providers())
            phase = perf_counter()
            model["encode_attempted"] = True
            vectors = encoder.encode(texts + [query["text"] for query in query_rows])
            times["semantic_encoding_seconds"] = perf_counter() - phase
            model.update(actual_encode_completed=True, status="completed",
                         encoded_document_count=len(texts), encoded_query_count=len(query_rows))
            phase = perf_counter()
            similarities = vectors[len(texts):] @ vectors[:len(texts)].T
            times["semantic_scoring_seconds"] = perf_counter() - phase
        except SemanticUnavailable as exc:
            times["semantic_encoding_seconds" if model["encode_attempted"] else "semantic_initialization_seconds"] = perf_counter() - phase
            model.update(status="unavailable", error=str(exc))
    records = []
    for qi, query in enumerate(query_rows):
        phase = perf_counter()
        rankings = {"lexical": lexical_rank(query["text"], texts)}
        times["lexical_seconds"] += perf_counter() - phase
        phase = perf_counter()
        rankings["bm25"] = bm25.rank(query["text"])
        times["bm25_ranking_seconds"] += perf_counter() - phase
        if similarities is not None:
            phase = perf_counter()
            rankings["semantic"] = sorted([(j, float(similarities[qi, j])) for j in range(len(texts))],
                                          key=lambda item: (-item[1], item[0]))
            times["semantic_scoring_seconds"] += perf_counter() - phase
            phase = perf_counter()
            rankings["lexical_semantic_rrf"] = reciprocal_rank_fusion([rankings["lexical"], rankings["semantic"]], k=60)
            rankings["bm25_semantic_rrf"] = reciprocal_rank_fusion([rankings["bm25"], rankings["semantic"]], k=60)
            times["rrf_seconds"] += perf_counter() - phase
        records.append({**query, "rankings": {
            method: [{"rank": rank, "index": corpus[j]["index"], "corpus_index": j,
                      "block_id": corpus[j]["block_id"], "score": score}
                     for rank, (j, score) in enumerate(ranking, 1)]
            for method, ranking in rankings.items()}})
    times["total_seconds"] = perf_counter() - started
    return {"schema_version": 1, "purpose": "development retrieval diagnostic; no inconsistency verdict",
            "documents": blocks, "corpus": corpus, "excluded_blocks": excluded, "records": records,
            "hashes": {"documents_sha256": _digest(blocks), "corpus_sha256": _digest(corpus),
                       "queries_sha256": _digest(query_rows),
                       "serialization": "UTF-8 canonical JSON: sorted keys, compact separators, ensure_ascii=False"},
            "parameters": {"candidate_kinds": ["prose"], "retrieval_text": "headings joined by LF, then original block text",
                           "lexical": "existing ASCII-token set Jaccard", "bm25": bm25.specification(),
                           "rrf": {"k": 60, "weights": [1, 1], "complete_rankings": True},
                           "semantic": {"threads": 4, "batch_size": 16, "max_sequence_length": 256,
                                        "pooling": "mean; L2 normalized", "similarity": "cosine"},
                           "tie_break": "input corpus order"},
            "model": model, "timing": times,
            "unavailable_methods": [] if similarities is not None else ["semantic", "lexical_semantic_rrf", "bm25_semantic_rrf"]}
