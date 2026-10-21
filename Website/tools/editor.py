"""Local-only portfolio editor. Run from any directory; Python standard library only."""
from __future__ import annotations

import argparse
import base64
import binascii
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / 'content'
PUBLIC = ROOT / 'public'
DIST = ROOT / 'dist'
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.RLock()
MAX_REQUEST = 112 * 1024 * 1024


def safe_path(base: Path, raw: str, suffix: str | None = None) -> Path:
    if not isinstance(raw, str) or not raw or '\\' in raw or '\x00' in raw:
        raise ValueError('Choose a valid relative file path.')
    if raw.startswith('/') or any(part in ('.', '..') for part in raw.split('/')):
        raise ValueError('The file path must stay inside the designated folder.')
    target = (base / raw).resolve()
    if not target.is_relative_to(base.resolve()) or target == base.resolve():
        raise ValueError('The file path must stay inside the designated folder.')
    if suffix and target.suffix.lower() != suffix:
        raise ValueError('This endpoint only edits ' + suffix + ' files.')
    return target


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.editor-tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def image_dimensions(data: bytes, extension: str) -> tuple[int, int]:
    if extension == '.png' and data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) >= 24:
        width, height = struct.unpack('>II', data[16:24])
    elif extension in ('.jpg', '.jpeg') and data.startswith(b'\xff\xd8'):
        offset = 2
        width = height = 0
        while offset < len(data) - 9:
            if data[offset] != 255:
                offset += 1
                continue
            while offset < len(data) and data[offset] == 255:
                offset += 1
            marker = data[offset]
            offset += 1
            if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            length = int.from_bytes(data[offset:offset+2], 'big')
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                height = int.from_bytes(data[offset+3:offset+5], 'big')
                width = int.from_bytes(data[offset+5:offset+7], 'big')
                break
            if length < 2:
                break
            offset += length
    elif extension == '.webp' and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        chunk = data[12:16]
        if chunk == b'VP8X' and len(data) >= 30:
            width = 1 + int.from_bytes(data[24:27], 'little')
            height = 1 + int.from_bytes(data[27:30], 'little')
        elif chunk == b'VP8L' and len(data) >= 25 and data[20] == 0x2F:
            bits = int.from_bytes(data[21:25], 'little')
            width, height = 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
        elif chunk == b'VP8 ' and len(data) >= 30 and data[23:26] == b'\x9d\x01\x2a':
            width = int.from_bytes(data[26:28], 'little') & 0x3FFF
            height = int.from_bytes(data[28:30], 'little') & 0x3FFF
        else:
            raise ValueError('This WebP format could not be read. Export a PNG or JPEG instead.')
    else:
        raise ValueError('The image contents do not match its PNG, JPEG, or WebP extension.')
    if not (0 < width <= 50000 and 0 < height <= 50000):
        raise ValueError('The image has invalid dimensions.')
    return width, height


def projects() -> list[dict]:
    source = CONTENT / 'projects.json'
    value = json.loads(source.read_text(encoding='utf-8')) if source.exists() else []
    if not isinstance(value, list):
        raise ValueError('content/projects.json must contain a list of projects.')
    return value


def preview_path(relative: str) -> str:
    extra_config = CONTENT / 'extra-pages.json'
    if extra_config.exists():
        for extra_page in json.loads(extra_config.read_text(encoding='utf-8')):
            if extra_page.get('file') == relative:
                return extra_page['path']
    parts = relative.split('/')
    if len(parts) > 2 and parts[0] == 'projects':
        anchor = ''
        for project in projects():
            for section in project.get('sections', []):
                if section.get('file') == relative:
                    anchor = '#' + section.get('id', '')
        return '/projects/' + parts[1] + '/' + anchor
    name = Path(relative).stem
    return '/' if name == 'home' else '/' + name + '/'


def build(origin: str) -> dict:
    environment = dict(os.environ, PORTFOLIO_BASE_PATH='', PORTFOLIO_ORIGIN=origin)
    messages = []
    for script in (ROOT / 'build_portfolio.py', ROOT / 'tools' / 'validate_site.py'):
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=environment,
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
        messages.append(script.name + '\n' + result.stdout + result.stderr)
        if result.returncode:
            return {'ok': False, 'saved': True, 'output': '\n'.join(messages)}
    return {'ok': True, 'saved': True, 'output': '\n'.join(messages)}


class EditorHandler(BaseHTTPRequestHandler):
    server_version = 'PortfolioLocalEditor/1.0'

    @property
    def origin(self) -> str:
        return 'http://127.0.0.1:' + str(self.server.server_port)

    def same_origin(self) -> bool:
        expected = '127.0.0.1:' + str(self.server.server_port)
        if self.headers.get('Host') != expected:
            return False
        if self.headers.get('Origin') not in (None, self.origin):
            return False
        referer = self.headers.get('Referer')
        if referer and (urlsplit(referer).scheme + '://' + urlsplit(referer).netloc) != self.origin:
            return False
        return self.headers.get('Sec-Fetch-Site') not in ('cross-site',)

    def json_response(self, value: dict | list, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def authorized(self) -> bool:
        if not self.same_origin() or not secrets.compare_digest(self.headers.get('X-Editor-Token', ''), TOKEN):
            self.json_response({'error': 'Open this editor from its local launcher. The session token or request origin is invalid.'}, 403)
            return False
        return True

    def do_GET(self) -> None:
        route = urlsplit(self.path)
        if not self.same_origin():
            self.json_response({'error': 'Only requests from this local editor are accepted.'}, 403)
            return
        try:
            if route.path in ('/editor', '/editor/'):
                document = (ROOT / 'tools' / 'editor.html').read_text(encoding='utf-8')
                document = document.replace('__EDITOR_TOKEN__', TOKEN)
                self.serve_bytes(document.encode('utf-8'), 'text/html; charset=utf-8')
                return
            if route.path.startswith('/api/'):
                if not self.authorized():
                    return
                if route.path == '/api/list':
                    entries = []
                    if CONTENT.exists():
                        for source in sorted(CONTENT.rglob('*.html')):
                            if source.resolve().is_relative_to(CONTENT.resolve()):
                                relative = source.relative_to(CONTENT).as_posix()
                                entries.append({'path': relative, 'preview': preview_path(relative)})
                    self.json_response({'files': entries, 'projects': projects(), 'repository': str(ROOT.parent)})
                elif route.path == '/api/file':
                    relative = parse_qs(route.query).get('path', [''])[0]
                    source = safe_path(CONTENT, relative, '.html')
                    self.json_response({'path': relative, 'html': source.read_text(encoding='utf-8'), 'preview': preview_path(relative)})
                elif route.path == '/api/projects':
                    self.json_response({'projects': projects()})
                else:
                    self.json_response({'error': 'Unknown API route.'}, 404)
                return
            relative = unquote(route.path).lstrip('/') or 'index.html'
            if any(part.startswith('.') for part in relative.split('/')):
                raise ValueError('Hidden files are not served.')
            source = safe_path(DIST, relative)
            if source.is_dir():
                source = safe_path(DIST, relative.rstrip('/') + '/index.html')
            if not source.is_file() or source.suffix.lower() in ('.py', '.cmd', '.env', '.json', '.md'):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(str(source))[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(source.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            if self.command != 'HEAD':
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, self.wfile)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            self.json_response({'error': str(error)}, 400)

    def serve_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        if not self.authorized():
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_REQUEST:
                raise ValueError('Choose a smaller file (maximum request size: 112 MB).')
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Expected a JSON request.')
            request = json.loads(self.rfile.read(length).decode('utf-8'))
            if not isinstance(request, dict):
                raise ValueError('Expected a JSON object.')
            with LOCK:
                result = self.mutate(urlsplit(self.path).path, request)
            self.json_response(result, 200 if result.get('ok', True) else 422)
        except (ValueError, OSError, json.JSONDecodeError, binascii.Error, TypeError, KeyError) as error:
            self.json_response({'error': str(error)}, 400)
        except subprocess.TimeoutExpired:
            self.json_response({'error': 'The file was saved, but the build timed out. Review the local console.', 'saved': True}, 500)

    def mutate(self, route: str, request: dict) -> dict:
        if route == '/api/save':
            relative = request.get('path', '')
            destination = safe_path(CONTENT, relative, '.html')
            if not destination.is_file():
                raise ValueError('Select an existing content file before saving.')
            value = request.get('html')
            if not isinstance(value, str):
                raise ValueError('HTML content is required.')
            atomic_text(destination, value)
            return {**build(self.origin), 'preview': preview_path(relative)}
        if route == '/api/build':
            return build(self.origin)
        if route == '/api/upload':
            original_name = request.get('name', '')
            if not isinstance(original_name, str) or Path(original_name).name != original_name or '\\' in original_name:
                raise ValueError('Upload a filename without a folder path.')
            extension = Path(original_name).suffix.lower()
            is_pdf = extension == '.pdf'
            if extension not in ('.png', '.jpg', '.jpeg', '.webp', '.pdf'):
                raise ValueError('Supported uploads: PNG, JPEG, WebP, and PDF.')
            raw = request.get('dataBase64', '')
            if not isinstance(raw, str):
                raise ValueError('The upload must be base64-encoded.')
            data = base64.b64decode(raw, validate=True)
            if len(data) > (80 if is_pdf else 25) * 1024 * 1024:
                raise ValueError('Upload limit: 25 MB per image, 80 MB per PDF.')
            if is_pdf and not data.startswith(b'%PDF-'):
                raise ValueError('The file contents do not match a PDF.')
            size = None if is_pdf else image_dimensions(data, extension)
            stem = re.sub(r'[^a-z0-9-]+', '-', Path(original_name).stem.lower()).strip('-')[:70] or 'upload'
            filename = stem + '-' + secrets.token_hex(3) + extension
            folder = 'documents/uploads' if is_pdf else 'assets/uploads'
            destination = safe_path(PUBLIC, folder + '/' + filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            if size:
                manifest = ROOT / 'asset-dimensions.json'
                dimensions = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
                dimensions['uploads/' + filename] = list(size)
                atomic_text(manifest, json.dumps(dimensions, indent=2, ensure_ascii=False) + '\n')
            # Make the new upload immediately visible even before the next full build.
            visible = safe_path(DIST, folder + '/' + filename)
            visible.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(destination, visible)
            return {'ok': True, 'url': '/' + folder + '/' + filename, 'name': original_name,
                    'kind': 'pdf' if is_pdf else 'image', 'width': size[0] if size else None, 'height': size[1] if size else None}
        if route == '/api/section':
            slug = request.get('slug', '')
            identifier = request.get('id', '')
            title = request.get('title', '')
            if not isinstance(slug, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug):
                raise ValueError('Choose an existing project.')
            if not isinstance(identifier, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', identifier):
                raise ValueError('Section ID needs lowercase letters, numbers, and hyphens.')
            if not isinstance(title, str) or not title.strip():
                raise ValueError('A section title is required.')
            collection = projects()
            project = next((item for item in collection if item.get('slug') == slug), None)
            if project is None:
                raise ValueError('Project not found.')
            relative = 'projects/' + slug + '/' + identifier + '.html'
            destination = safe_path(CONTENT, relative, '.html')
            if destination.exists() or any(section.get('id') == identifier for section in project.get('sections', [])):
                raise ValueError('That section ID already exists.')
            atomic_text(destination, '<p>Describe the engineering work, decisions, and results here.</p>\n')
            project.setdefault('sections', []).append({'id': identifier, 'title': title.strip(), 'file': relative})
            atomic_text(CONTENT / 'projects.json', json.dumps(collection, indent=2, ensure_ascii=False) + '\n')
            return {**build(self.origin), 'path': relative, 'preview': '/projects/' + slug + '/#' + identifier}
        if route == '/api/project':
            collection = projects()
            project = next((item for item in collection if item.get('slug') == request.get('slug')), None)
            if project is None:
                raise ValueError('Project not found.')
            fields = ('title', 'lede', 'hero_caption', 'kind', 'category', 'status', 'image', 'alt', 'summary', 'role', 'period', 'tools', 'stage')
            changes = request.get('fields', {})
            if not isinstance(changes, dict):
                raise ValueError('Project fields must be an object.')
            for name, value in changes.items():
                if name not in fields or not isinstance(value, str):
                    raise ValueError('Only the displayed text fields can be changed here.')
                project[name] = value
            atomic_text(CONTENT / 'projects.json', json.dumps(collection, indent=2, ensure_ascii=False) + '\n')
            return {**build(self.origin), 'preview': '/projects/' + project['slug'] + '/'}
        raise ValueError('Unknown API route. Publishing is performed separately in GitHub Desktop.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), EditorHandler)
    except OSError:
        if args.port != 8765:
            raise
        server = ThreadingHTTPServer(('127.0.0.1', 0), EditorHandler)
    local_origin = 'http://127.0.0.1:' + str(server.server_port)
    address = local_origin + '/editor'
    print('Building and checking the local preview...', flush=True)
    try:
        startup = build(local_origin)
    except (OSError, subprocess.SubprocessError) as error:
        startup = {'ok': False, 'output': str(error)}
    if startup.get('ok'):
        print('Local preview is ready at ' + local_origin + '/', flush=True)
    else:
        print('Local preview is not ready: the startup build or validation failed.', flush=True)
        print(startup.get('output', ''), flush=True)
        print('The editor remains available. Correct the issue, then use Save & build.', flush=True)
    print('Portfolio editor: ' + address, flush=True)
    print('This server is local to your computer. Close this window or press Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
