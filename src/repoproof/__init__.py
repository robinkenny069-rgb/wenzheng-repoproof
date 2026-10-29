"""RepoProof prototype: static evidence for reviewing changed API defaults."""

from .core import (
    MISSING,
    UNKNOWN,
    Abstention,
    AnalysisResult,
    Candidate,
    DefaultValue,
    Evidence,
    Heading,
    MarkdownBlock,
    ParameterChange,
    analyze,
    analyze_changes,
    split_markdown,
)

__all__ = [
    "MISSING", "UNKNOWN", "Abstention", "AnalysisResult", "Candidate",
    "DefaultValue", "Evidence", "Heading", "MarkdownBlock", "ParameterChange",
    "analyze", "analyze_changes", "split_markdown",
]
