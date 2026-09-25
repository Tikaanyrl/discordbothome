import io
import json
import os
import secrets
import signal
import threading
import uuid
import zipfile

from flask import Flask, abort, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from core import Manager, Problem, atomic


def create_app(manager, local=False):
    app = Flask(__name__, static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024**2
    csrf = secrets.token_urlsafe(32)

    @app.before_request
    def access():
        allowed = {'127.0.0.1', '::1'} if local else {'172.30.32.2'}
        if request.remote_addr not in allowed:
            abort(403)
        if request.method != 'GET' and not secrets.compare_digest(request.headers.get('X-CSRF-Token', ''), csrf):
            abort(403)

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'self'"
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        status = 400 if isinstance(exc, (Problem, ValueError, OSError, zipfile.BadZipFile)) else 500
        manager.log(str(exc))
        return jsonify(error=manager.redact(str(exc))), status

    @app.get('/')
    def index():
        return app.send_static_file('index.html')

    @app.get('/api/state')
    def state():
        with manager.lock:
            return jsonify(status=manager.status(), settings=manager.public_settings(), csrf=csrf)

    @app.post('/api/control/<action>')
    def control(action):
        with manager.lock:
            if action == 'start':
                manager.start()
            elif action == 'stop':
                manager.stop()
            elif action == 'restart':
                manager.stop()
                manager.start()
            elif action == 'install':
                manager.install()
            else:
                abort(404)
        return jsonify(ok=True)

    @app.post('/api/settings')
    def settings():
        manager.save_settings(request.get_json())
        return jsonify(ok=True)

    @app.get('/api/logs')
    def logs():
        search = request.args.get('q', '').lower()
        return jsonify(lines=[manager.redact(line) for line in list(manager.lines) if search in line.lower()])

    @app.get('/api/files')
    def files():
        with manager.lock:
            path = manager.safe(request.args.get('path', ''), allow_root=True)
            result = []
            for p in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if p.is_symlink():
                    continue
                result.append({'name': p.name, 'path': p.relative_to(manager.root).as_posix(),
                               'directory': p.is_dir(), 'size': p.stat().st_size if p.is_file() else 0})
            return jsonify(files=result)

    @app.get('/api/file')
    def read():
        with manager.lock:
            path = manager.safe(request.args['path'])
            if request.args.get('download') == '1':
                if path.suffix == '.log':
                    return send_file(io.BytesIO(manager.redact(path.read_text(errors='replace')).encode()),
                                     download_name=path.name, as_attachment=True)
                return send_file(path, as_attachment=True, download_name=path.name)
            if path.stat().st_size > 2 * 1024**2:
                raise Problem('Editor unterstützt Dateien bis 2 MiB.')
            return jsonify(text=path.read_text(encoding='utf-8'))

    @app.post('/api/file')
    def write():
        with manager.lock:
            manager.stopped()
            body = request.get_json()
            p = manager.editable(body['path'])
            if p.exists() and not body.get('overwrite'):
                raise Problem('Datei existiert bereits. Überschreiben ausdrücklich bestätigen.')
            atomic(p, body['text'].encode('utf-8'))
        return jsonify(ok=True)

    @app.post('/api/upload')
    def upload():
        with manager.lock:
            manager.stopped()
            p = manager.editable(request.form['path'])
            if p.exists() and request.form.get('overwrite') != 'true':
                raise Problem('Datei existiert bereits.')
            atomic(p, request.files['file'].read())
        return jsonify(ok=True)

    @app.post('/api/files/<action>')
    def change(action):
        with manager.lock:
            manager.stopped()
            body = request.get_json()
            if action == 'mkdir':
                manager.editable(body['path']).mkdir()
            elif action == 'rename':
                source = manager.editable(body['path'])
                target = manager.editable(body['target'])
                if target.exists():
                    raise Problem('Ziel existiert bereits.')
                source.rename(target)
            elif action == 'trash':
                source = manager.safe(body['path'])
                if len(source.relative_to(manager.root).parts) < 2 or source.is_relative_to(manager.root / 'trash'):
                    raise Problem('Dieser Pfad kann nicht in den Papierkorb verschoben werden.')
                # Original path preserved inside a unique trash folder.
                target = manager.root / 'trash' / uuid.uuid4().hex / source.relative_to(manager.root)
                target.parent.mkdir(parents=True)
                source.rename(target)
            elif action == 'recover':
                source = manager.safe(body['path'])
                if not source.is_relative_to(manager.root / 'trash') or source == manager.root / 'trash':
                    raise Problem('Nur Dateien aus dem Papierkorb können zurückgeholt werden.')
                target = manager.editable(body['target'])
                if target.exists():
                    raise Problem('Ziel existiert bereits.')
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
            elif action == 'delete':
                source = manager.safe(body['path'])
                if body.get('confirm') != body['path'] or source == manager.root / 'trash' or not source.is_relative_to(manager.root / 'trash'):
                    raise Problem('Dauerhaftes Löschen erfordert den Papierkorb und eine Bestätigung.')
                # Files only; no recursive delete that could follow a symlink.
                if source.is_dir():
                    source.rmdir()
                else:
                    source.unlink()
            else:
                abort(404)
        return jsonify(ok=True)

    @app.post('/api/backup')
    def backup():
        with manager.lock:
            if manager.job:
                raise Problem('Bitte laufende Aufgabe abwarten.')
            return jsonify(path=manager.backup())

    @app.post('/api/backup/upload')
    def backup_upload():
        with manager.lock:
            manager.stopped()
            path = 'backups/import-' + uuid.uuid4().hex + '.zip'
            atomic(manager.safe(path), request.files['file'].read())
            return jsonify(path=path, **manager.archive_preview(path))

    @app.post('/api/backup/preview')
    def preview():
        with manager.lock:
            return jsonify(manager.archive_preview(request.get_json()['path']))

    @app.post('/api/backup/restore')
    def restore():
        body = request.get_json()
        if body.get('confirm') is not True:
            raise Problem('Wiederherstellung muss bestätigt werden.')
        with manager.lock:
            manager.restore(body['path'], body['sha256'])
        return jsonify(ok=True)

    return app


if __name__ == '__main__':
    from waitress import serve
    local = os.getenv('MANAGER_LOCAL') == '1'
    m = Manager(os.getenv('BOT_ROOT', '/share/discord-bot-manager'),
                os.getenv('MANAGER_DATA', '/data'),
                os.getenv('BOT_SEED', '/opt/seed'))
    def shutdown(*_):
        m.closing = True
        m.stop()
        if m.install_proc:
            m.terminate(m.install_proc)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    def initialize():
        try:
            m.prepare_runtime()
            if m.settings['autostart']:
                m.start()
        except Exception as exc:
            m.last_error = str(exc)
            m.log(exc)
    threading.Thread(target=initialize, daemon=True).start()
    serve(create_app(m, local), host='127.0.0.1' if local else '0.0.0.0', port=8099, threads=8)
