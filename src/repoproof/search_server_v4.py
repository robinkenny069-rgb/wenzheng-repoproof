"""Loopback-only, session-scoped UI for a preloaded search index.

No path, file upload, source execution, or mutation endpoint is exposed.
The session URL is intentionally absent from persisted metadata and access logs.
"""
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit


MAX_BODY_BYTES = 8192
MAX_QUERY_CHARS = 500
MAX_REQUESTS_PER_MINUTE = 30


class SearchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, index, template_html, *, port=0, session_token=None,
                 workbench_html=None, max_requests=MAX_REQUESTS_PER_MINUTE):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError('port must be an integer from 0 to 65535')
        if not isinstance(max_requests, int) or max_requests < 1:
            raise ValueError('max_requests must be positive')
        token = session_token or secrets.token_urlsafe(32)
        if not re.fullmatch(r'[A-Za-z0-9_-]{24,128}', token):
            raise ValueError('session token must have at least 24 URL-safe characters')
        self.index = index
        self.template_html = template_html.encode('utf-8') if isinstance(template_html, str) else bytes(template_html)
        self.workbench_html = (workbench_html.encode('utf-8') if isinstance(workbench_html, str) else workbench_html)
        self.session_prefix = '/' + token
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.search_lock = threading.Lock()
        self.rate_lock = threading.Lock()
        self.request_times = deque()
        self.max_requests = max_requests
        super().__init__(('127.0.0.1', port), SearchHandler)
        self.origin = f'http://127.0.0.1:{self.server_port}'
        self.expected_host = f'127.0.0.1:{self.server_port}'
        self.session_url = self.origin + self.session_prefix + '/'

    def public_metadata(self):
        """Safe to persist: no capability URL, session token, or query text."""
        return {'schema_version': 1, 'pid': os.getpid(), 'host': '127.0.0.1',
                'port': self.server_port, 'started_at_utc': self.started_at,
                'session_url_persisted': False, 'read_only': True,
                'session_url_location': 'launcher stdout only',
                'body_limit_bytes': MAX_BODY_BYTES, 'query_limit_characters': MAX_QUERY_CHARS,
                'requests_per_minute': self.max_requests}

    def consume_request(self):
        now = time.monotonic()
        with self.rate_lock:
            while self.request_times and now - self.request_times[0] >= 60:
                self.request_times.popleft()
            if len(self.request_times) >= self.max_requests:
                return False
            self.request_times.append(now)
        return True


class SearchHandler(BaseHTTPRequestHandler):
    server_version = 'WenzhengLocal/4'
    sys_version = ''
    protocol_version = 'HTTP/1.0'

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_args):
        # BaseHTTPRequestHandler logs request paths, which contain the session token.
        pass

    def _reply(self, status, data, content_type='application/json; charset=utf-8'):
        body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _guard(self, *, post=False):
        hosts = self.headers.get_all('Host', [])
        if hosts != [self.server.expected_host]:
            self._reply(403, {'error': '仅允许本机会话地址。'})
            return False
        origins = self.headers.get_all('Origin', [])
        if (post and origins != [self.server.origin]) or (origins and origins != [self.server.origin]):
            self._reply(403, {'error': '请求来源不属于当前本机会话。'})
            return False
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self._reply(403, {'error': '拒绝跨站请求。'})
            return False
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            self._reply(404, {'error': '未找到这个会话页面。'})
            return False
        return True

    def do_GET(self):
        if not self._guard():
            return
        prefix = self.server.session_prefix
        if self.path == prefix + '/':
            self._reply(200, self.server.template_html, 'text/html; charset=utf-8')
        elif self.path == prefix + '/api/catalog':
            data = dict(self.server.index.catalog())
            data['server'] = {'query_limit_characters': MAX_QUERY_CHARS,
                              'original_workbench_available': self.server.workbench_html is not None}
            self._reply(200, data)
        elif self.path == prefix + '/workbench' and self.server.workbench_html is not None:
            self._reply(200, self.server.workbench_html, 'text/html; charset=utf-8')
        else:
            self._reply(404, {'error': '未找到这个会话页面。'})

    def do_POST(self):
        if not self._guard(post=True):
            return
        if self.path != self.server.session_prefix + '/api/search':
            self._reply(404, {'error': '未找到这个会话接口。'})
            return
        if self.headers.get_all('Transfer-Encoding'):
            self._reply(400, {'error': '不支持分块请求。'})
            return
        lengths = self.headers.get_all('Content-Length', [])
        if len(lengths) != 1 or not re.fullmatch(r'[0-9]+', lengths[0]):
            self._reply(411, {'error': '必须提供单一有效的请求长度。'})
            return
        length = int(lengths[0])
        if not 0 < length <= MAX_BODY_BYTES:
            self._reply(413, {'error': '请求内容超过限制。'})
            return
        if self.headers.get_content_type() != 'application/json':
            self._reply(415, {'error': '请求必须使用 JSON。'})
            return
        if not self.server.consume_request():
            self._reply(429, {'error': '查询过于频繁，请稍后再试。'})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('incomplete request')
            data = json.loads(raw.decode('utf-8'))
            if not isinstance(data, dict) or set(data) - {'query', 'case_id', 'snapshot', 'k'}:
                raise ValueError('unexpected request fields')
            query, case_id = data.get('query'), data.get('case_id')
            snapshot = data.get('snapshot', 'after')
            if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
                raise ValueError('query must contain 1 to 500 characters')
            if not isinstance(case_id, str) or not 1 <= len(case_id) <= 120:
                raise ValueError('invalid case identifier')
            if snapshot not in ('before', 'after') or type(data.get('k', 6)) is not int or data.get('k', 6) != 6:
                raise ValueError('snapshot must be before/after and k must be 6')
        except (UnicodeError, ValueError, TypeError):
            self._reply(400, {'error': '输入无效：请选择案例、快照，输入 1–500 个字符的用途描述。'})
            return
        except (TimeoutError, socket.timeout):
            self._reply(408, {'error': '接收请求超时。'})
            return
        if not self.server.search_lock.acquire(blocking=False):
            self._reply(429, {'error': '上一项查询仍在运行，请稍后再试。'})
            return
        try:
            result = self.server.index.search(query.strip(), case_id, snapshot=snapshot, k=6)
            self._reply(200, result)
        except (KeyError, ValueError) as exc:
            self._reply(400, {'error': '无法使用这组查询设置：' + str(exc)[:300]})
        except Exception as exc:
            self._reply(500, {'error': '检索未完成，请保留错误类型供排查：' + type(exc).__name__})
        finally:
            self.server.search_lock.release()


def make_server(index, template_html, **kwargs):
    return SearchHTTPServer(index, template_html, **kwargs)
