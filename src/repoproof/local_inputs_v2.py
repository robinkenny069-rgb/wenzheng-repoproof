"""Bounded, deterministic, read-only collection of local Python and prose files.

This loader does not import modules, interpret documents, follow links, or apply
reference labels. Limits apply to all eligible files, including encoding errors.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat


EXCLUDED_DIRECTORIES = frozenset({
    '.git', '.deps', '.venv', 'venv', 'node_modules', '__pycache__', 'output',
})
SUPPORTED_SUFFIXES = frozenset({'.py', '.rst', '.md'})


class SnapshotInputError(ValueError):
    """An input boundary or resource limit prevents a complete snapshot."""


def _link_or_reparse(path: Path, info: os.stat_result) -> bool:
    # On Windows this also rejects junctions and other reparse points; on Unix
    # S_ISLNK covers symlinks. Do not rely on is_dir(), which follows links.
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, 'st_file_attributes', 0)
        & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    )


def load_snapshot(directory: Path, *, max_files: int = 128,
                  max_file_bytes: int = 2 * 1024 * 1024,
                  max_total_bytes: int = 16 * 1024 * 1024) -> dict:
    """Collect UTF-8 .py/.rst/.md files beneath one explicit local directory.

    Directory names are excluded case-insensitively. Internal links are listed
    as excluded and never followed; links escaping the root fail the snapshot.
    Decode failures are retained in ``excluded`` and still consume limits. Any
    limit failure raises rather than returning a silently truncated snapshot.
    Empty means no successfully decoded, supported file (not zero text length).
    """
    for name, value in [('max_files', max_files), ('max_file_bytes', max_file_bytes),
                        ('max_total_bytes', max_total_bytes)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SnapshotInputError(f'{name} must be a positive integer')
    selected = Path(directory)
    try:
        original_info = selected.lstat()
        if _link_or_reparse(selected, original_info):
            raise SnapshotInputError('Snapshot root must be a real directory, not a linked/reparse path')
        root = selected.resolve(strict=True)
        if not root.is_dir():
            raise SnapshotInputError('Snapshot root is not a directory')
    except OSError as exc:
        raise SnapshotInputError(f'Cannot access snapshot root: {exc}') from exc

    sources, documents, files, excluded = {}, {}, [], []
    candidate_count, candidate_bytes, encoding_errors = 0, 0, 0

    def checked_path(path: Path, relative: str) -> Path:
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SnapshotInputError(f'Cannot resolve input path {relative}: {exc}') from exc
        if not resolved.is_relative_to(root):
            raise SnapshotInputError(f'Input path escapes snapshot root: {relative}')
        return resolved

    def visit(folder: Path):
        nonlocal candidate_count, candidate_bytes, encoding_errors
        try:
            with os.scandir(folder) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise SnapshotInputError(f'Cannot list input directory {folder.relative_to(root).as_posix()}: {exc}') from exc
        for entry in children:
            path = folder / entry.name
            relative = path.relative_to(root).as_posix()
            try:
                info = path.lstat()
            except OSError as exc:
                raise SnapshotInputError(f'Cannot inspect input path {relative}: {exc}') from exc
            # Check links before treating them as a file/directory or recursing.
            if _link_or_reparse(path, info):
                checked_path(path, relative)
                excluded.append({'path': relative, 'reason': 'linked_or_reparse_path_not_followed'})
                continue
            checked_path(path, relative)
            if stat.S_ISDIR(info.st_mode):
                if entry.name.casefold() in EXCLUDED_DIRECTORIES:
                    excluded.append({'path': relative, 'reason': 'excluded_directory'})
                else:
                    visit(path)
                continue
            if not stat.S_ISREG(info.st_mode):
                excluded.append({'path': relative, 'reason': 'unsupported_file_type'})
                continue
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                excluded.append({'path': relative, 'reason': 'unsupported_extension'})
                continue
            candidate_count += 1
            if candidate_count > max_files:
                raise SnapshotInputError(f'File count limit exceeded at {relative}: more than {max_files} eligible files')
            if info.st_size > max_file_bytes:
                raise SnapshotInputError(f'Per-file byte limit exceeded: {relative} ({info.st_size} > {max_file_bytes})')
            if candidate_bytes + info.st_size > max_total_bytes:
                raise SnapshotInputError(f'Total byte limit exceeded at {relative}: more than {max_total_bytes}')
            try:
                # Bounded read also detects growth between stat and open.
                with path.open('rb') as stream:
                    raw = stream.read(max_file_bytes + 1)
            except OSError as exc:
                raise SnapshotInputError(f'Cannot read input file {relative}: {exc}') from exc
            checked_path(path, relative)
            if len(raw) > max_file_bytes:
                raise SnapshotInputError(f'Per-file byte limit exceeded during read: {relative}')
            candidate_bytes += len(raw)
            if candidate_bytes > max_total_bytes:
                raise SnapshotInputError(f'Total byte limit exceeded during read: {relative}')
            digest = hashlib.sha256(raw).hexdigest()
            try:
                text = raw.decode('utf-8-sig')
            except UnicodeDecodeError as exc:
                encoding_errors += 1
                excluded.append({'path': relative, 'reason': 'invalid_utf8', 'bytes': len(raw),
                                 'sha256': digest, 'error_byte_offset': exc.start})
                continue
            kind = 'source' if suffix == '.py' else 'document'
            (sources if kind == 'source' else documents)[relative] = text
            files.append({'path': relative, 'kind': kind, 'bytes': len(raw), 'sha256': digest})

    visit(root)
    if not files:
        raise SnapshotInputError('Snapshot has no readable UTF-8 .py/.rst/.md files')
    # Stable globally sorted output, independent of directory traversal details.
    files.sort(key=lambda item: item['path'])
    excluded.sort(key=lambda item: (item['path'], item['reason']))
    return {'sources': dict(sorted(sources.items())), 'documents': dict(sorted(documents.items())),
            'inventory': {'source_files': len(sources), 'document_files': len(documents),
                          'accepted_files': len(files), 'scanned_candidate_files': candidate_count,
                          'scanned_candidate_bytes': candidate_bytes,
                          'excluded_entries': len(excluded), 'encoding_errors': encoding_errors},
            'files': files, 'excluded': excluded}
