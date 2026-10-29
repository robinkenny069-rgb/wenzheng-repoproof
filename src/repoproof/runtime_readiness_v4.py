"""Startup-only readiness checks; no retrieval scoring or method changes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .semantic import SemanticRetriever


def exception_chain(error):
    items, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        items.append({'type': type(error).__name__, 'message': str(error)[:3000]})
        error = error.__cause__ or error.__context__
    return items


def prepare_search_runtime(index, report_path, *, encoder_factory=SemanticRetriever):
    """Probe real inference before opening a page; retain the same encoder.

    A startup probe is not a successful user search, benchmark, or human review.
    An unsuccessful probe prevents the normal semantic launcher from serving a
    misleading ready page. --static-only remains an explicit separate mode.
    """
    record = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'python_executable': sys.executable, 'python_version': sys.version.split()[0],
              'isolated_python': bool(sys.flags.isolated), 'probe_is_user_query': False,
              'scope': 'Model integrity and one actual CPU inference only; not task quality.',
              'requested': bool(index.model['requested']), 'status': 'disabled'}
    if index.model['requested']:
        try:
            encoder = encoder_factory()
            vectors = encoder.encode(['Local model readiness check.'])
            if tuple(vectors.shape) != (1, 384) or not bool(encoder.np.isfinite(vectors).all()):
                raise ValueError('Startup probe returned invalid vectors')
            index.encoder = encoder
            index.model.update(status='ready', actual_encode_completed=False,
                model_file_sha256=encoder.manifest['files']['onnx/model.onnx']['sha256'])
            record.update(status='ready', probe_shape=list(vectors.shape),
                model_revision=encoder.manifest['revision'],
                model_file_sha256=index.model['model_file_sha256'],
                providers=encoder.session.get_providers())
        except Exception as error:
            record.update(status='unavailable', errors=exception_chain(error))
            index.model.update(status='unavailable', actual_encode_completed=False,
                               error=' → '.join(x['message'] for x in record['errors']))
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    return record
