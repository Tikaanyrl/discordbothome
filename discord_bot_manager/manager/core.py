"""Persistent storage and single-process bot supervision. No automatic data pruning."""
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import venv
import zipfile


class Problem(Exception):
    pass


def atomic(path, data, mode=0o600):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            os.fchmod(f.fileno(), mode)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)  # Incomplete internal write only; never user data.


class Manager:
    roots = {'code', 'storage', 'logs', 'backups', 'trash'}

    def __init__(self, root, data, seed, runtime_python=None):
        self.root, self.data = Path(root).absolute(), Path(data).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in self.roots:
            p = self.root / name
            if p.is_symlink():
                raise Problem('Speicherordner darf kein Symlink sein.')
            p.mkdir(exist_ok=True)
        for source in Path(seed).iterdir():
            target = self.root / 'code' / source.name
            if source.is_file() and not target.exists() and not target.is_symlink():
                # Exclusive creation: an update never replaces existing bot code.
                with target.open('xb') as f:
                    f.write(source.read_bytes())
        self.settings_path = self.data / 'settings.json'
        self.settings = {'entrypoint': 'bot.py', 'token': '', 'autostart': False,
                         'auto_restart': True, 'environment': {}}
        if self.settings_path.exists():
            self.settings.update(json.loads(self.settings_path.read_text()))
        self.lock = threading.RLock()
        self.runtime_lock = threading.Lock()
        self.proc = None
        self.install_proc = None
        self.wanted = False
        self.closing = False
        self.retries = 0
        self.started = None
        self.last_error = None
        self.job = None
        self.lines = collections.deque(maxlen=1000)
        self.python = runtime_python
        self.status_file = self.data / 'connection.json'
        self.log_path = self.root / 'logs' / f'manager-{time.time_ns()}.log'
        self.redactions = set()
        self.remember_secrets()

    def remember_secrets(self):
        self.redactions.update(str(v) for v in [self.settings.get('token'),
                               *self.settings.get('environment', {}).values()] if v)

    def redact(self, value):
        for secret in sorted(self.redactions, key=len, reverse=True):
            value = value.replace(secret, '[REDACTED]')
        return value

    def log(self, text):
        text = self.redact(str(text))
        self.lines.append(text)
        try:
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(text + '\n')
        except OSError as exc:
            self.last_error = f'Log konnte nicht gespeichert werden: {exc}'

    def safe(self, value, allow_root=False):
        if not isinstance(value, str) or '\\' in value or '\x00' in value:
            raise Problem('Ungültiger Pfad.')
        parts = PurePosixPath(value).parts
        if not parts and allow_root:
            return self.root
        if not parts or parts[0] not in self.roots or '..' in parts or value.startswith('/') or PurePosixPath(value).as_posix() != value:
            raise Problem('Pfad außerhalb des Dateibereichs.')
        p = self.root
        for part in parts:
            p = p / part
            if p.is_symlink():
                raise Problem('Symlinks sind im Dateimanager nicht erlaubt.')
        if not p.resolve().is_relative_to(self.root.resolve()):
            raise Problem('Pfad außerhalb des Dateibereichs.')
        return p

    def stopped(self):
        if self.proc is not None or self.job:
            raise Problem('Bitte zuerst den Bot stoppen und laufende Aufgaben abwarten.')

    def editable(self, path):
        p = self.safe(path)
        if len(p.relative_to(self.root).parts) < 2 or p.parts[len(self.root.parts)] not in {'code', 'storage'}:
            raise Problem('Schreiben ist nur innerhalb von code/ und storage/ erlaubt.')
        return p

    def save_settings(self, values):
        with self.lock:
            self.stopped()
            entry = values.get('entrypoint', self.settings['entrypoint'])
            self.safe('code/' + entry)
            if not entry.endswith('.py'):
                raise Problem('Einstiegspunkt muss eine Python-Datei sein.')
            env = values.get('environment', self.settings['environment'])
            if not isinstance(env, dict) or any(not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', k)
                                               or not isinstance(v, str) or '\x00' in v for k, v in env.items()):
                raise Problem('Umgebungsvariablen müssen ein JSON-Objekt aus Textwerten sein.')
            reserved = {'DISCORD_TOKEN', 'BOT_STORAGE_DIR', 'BOT_STATUS_FILE', 'PYTHONPATH',
                        'PYTHONHOME', 'LD_PRELOAD', 'PATH', 'SUPERVISOR_TOKEN'}
            if reserved.intersection(env):
                raise Problem('Reservierte Umgebungsvariable.')
            updated = dict(self.settings, entrypoint=entry, environment=env)
            for key in ('autostart', 'auto_restart'):
                if key in values:
                    if type(values[key]) is not bool:
                        raise Problem('Ungültiger Schalter.')
                    updated[key] = values[key]
            if values.get('clear_token'):
                updated['token'] = ''
            elif values.get('token'):
                if not isinstance(values['token'], str) or '\x00' in values['token']:
                    raise Problem('Ungültiges Token.')
                updated['token'] = values['token']
            atomic(self.settings_path, json.dumps(updated).encode())
            self.settings = updated
            self.remember_secrets()

    def public_settings(self):
        return {**self.settings, 'token': '', 'token_set': bool(self.settings['token'])}

    def prepare_runtime(self):
        with self.runtime_lock:
            self._prepare_runtime()

    def _prepare_runtime(self):
        if self.python:
            return
        version = os.getenv('MANAGER_RUNTIME_VERSION', 'dev')
        runtime = self.data / f'python-{sys.version_info.major}.{sys.version_info.minor}-{version}'
        if not (runtime / 'ready').exists():
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(runtime)
            atomic(runtime / 'ready', b'ready')
        self.python = str(runtime / 'bin/python')

    def child_env(self):
        return {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(self.root / 'storage'),
                'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg',
                **self.settings['environment'], 'DISCORD_TOKEN': self.settings['token'],
                'BOT_STORAGE_DIR': str(self.root / 'storage'), 'BOT_STATUS_FILE': str(self.status_file)}

    def start(self, retry=False):
        with self.lock:
            if self.closing:
                raise Problem('Manager wird beendet.')
            if self.proc is not None:
                return
            if self.job:
                raise Problem('Eine Aufgabe läuft noch.')
            entry = self.safe('code/' + self.settings['entrypoint'])
            if not entry.is_file():
                raise Problem('Einstiegspunkt fehlt.')
            if not self.python:
                raise Problem('Python-Umgebung wird noch vorbereitet.')
            if not retry:
                self.retries = 0
            atomic(self.status_file, b'{}')
            self.proc = subprocess.Popen([self.python, '-u', str(entry)], cwd=self.root / 'storage',
                                         env=self.child_env(), stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            self.wanted, self.started = True, time.time()
            self.last_error = None
            threading.Thread(target=self.watch, args=(self.proc,), daemon=True).start()

    @staticmethod
    def terminate(proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        # Also stop descendants if the parent exited before them.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def stop(self):
        with self.lock:
            self.wanted = False
            if self.proc:
                self.terminate(self.proc)
                self.proc = None
            self.started = None

    def watch(self, proc):
        # Read bounded fragments, including output without newlines.
        while True:
            chunk = proc.stdout.readline(16384)
            if not chunk:
                break
            self.log(chunk.decode('utf-8', errors='replace').rstrip())
        code = proc.wait()
        with self.lock:
            if self.proc is not proc:
                return
            self.terminate(proc)
            self.proc = None
            self.started = None
            if code != 0:
                self.last_error = f'Bot mit Exit-Code {code} beendet. Details im Log.'
            self.log(f'Bot beendet (Exit-Code {code}).')
            again = self.wanted and code != 0 and self.settings['auto_restart'] and self.retries < 3
            if again:
                self.retries += 1
                delay = 5 * self.retries
            else:
                self.wanted = False
        if again:
            time.sleep(delay)
            with self.lock:
                if self.wanted and not self.closing and self.proc is None:
                    try:
                        self.start(retry=True)
                    except Exception as exc:
                        self.last_error = str(exc)
                        self.log(exc)

    def background(self, title, callback):
        with self.lock:
            self.stopped()
            self.job = title
        def run():
            try:
                callback()
                self.log(title + ': abgeschlossen.')
            except Exception as exc:
                self.last_error = self.redact(str(exc))
                self.log(title + ': ' + str(exc))
            finally:
                with self.lock:
                    self.job = None
        threading.Thread(target=run, daemon=True).start()

    def install(self):
        requirements = self.safe('code/requirements.txt')
        if not requirements.is_file():
            raise Problem('requirements.txt fehlt.')
        def run():
            self.prepare_runtime()
            self.install_proc = subprocess.Popen([self.python, '-m', 'pip', 'install',
                '--disable-pip-version-check', '-r', str(requirements)], cwd=self.root / 'code',
                env={**self.child_env(), 'PIP_NO_INPUT': '1'}, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True)
            proc = self.install_proc
            try:
                for line in iter(lambda: proc.stdout.readline(16384), b''):
                    self.log(line.decode('utf-8', errors='replace').rstrip())
                if proc.wait() != 0:
                    raise Problem('Paketinstallation fehlgeschlagen. Details im Log.')
            finally:
                self.install_proc = None
        self.background('Paketinstallation', run)

    def status(self):
        usage = shutil.disk_usage(self.root)
        connected = None
        if self.proc:
            try:
                state = json.loads(self.status_file.read_text())
                if time.time() - state.get('updated', 0) < 30:
                    connected = bool(state.get('connected'))
            except (OSError, ValueError):
                pass
        return {'running': self.proc is not None, 'connected': connected,
                'uptime': round(time.time() - self.started) if self.started else 0,
                'retries': self.retries, 'last_error': self.last_error, 'job': self.job,
                'disk_free': usage.free, 'disk_total': usage.total,
                'disk_warning': usage.free < max(1024**3, usage.total * .05)}

    def backup(self):
        # Caller holds lock; stop before reading any bot files.
        was_running = self.proc is not None
        self.stop()
        self.stopped()
        name = f'backup-{time.time_ns()}.zip'
        target = self.root / 'backups' / name
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix='.backup-')
        os.close(fd)
        try:
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                for area in ('code', 'storage'):
                    for p in (self.root / area).rglob('*'):
                        self.safe(p.relative_to(self.root).as_posix())
                        if p.is_file():
                            archive.write(p, p.relative_to(self.root).as_posix())
                archive.writestr('settings.json', json.dumps(self.settings))
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if was_running:
            self.start()
        return 'backups/' + name

    def archive_preview(self, path):
        p = self.safe(path)
        if p.parent != self.root / 'backups' or not p.is_file():
            raise Problem('Archiv muss in backups/ liegen.')
        result, seen, total = [], set(), 0
        with zipfile.ZipFile(p) as archive:
            if len(archive.infolist()) > 10000:
                raise Problem('Zu viele Archiveinträge.')
            for info in archive.infolist():
                name = info.filename
                if info.is_dir():
                    continue
                if name in seen or stat.S_ISLNK(info.external_attr >> 16):
                    raise Problem('Doppelte Pfade oder Symlinks im Archiv.')
                seen.add(name)
                if name != 'settings.json':
                    self.editable(name)
                total += info.file_size
                if total > 1024**3 or info.file_size > 256 * 1024**2:
                    raise Problem('Archiv zu groß (max. 1 GiB entpackt, 256 MiB pro Datei).')
                result.append({'path': name, 'size': info.file_size,
                               'overwrite': name == 'settings.json' or self.safe(name).exists()})
            if not result:
                raise Problem('Archiv ist leer.')
        with p.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        return {'files': result, 'sha256': digest}

    def restore(self, path, digest):
        self.stopped()
        preview = self.archive_preview(path)
        if preview['sha256'] != digest:
            raise Problem('Archiv wurde verändert. Vorschau erneut öffnen.')
        self.backup()  # Preserve overwritten content and settings unconditionally.
        with zipfile.ZipFile(self.safe(path)) as archive:
            # Validate CRC of every entry before modifying data.
            if archive.testzip():
                raise Problem('Beschädigtes Archiv.')
            settings = None
            if 'settings.json' in archive.namelist():
                settings = json.loads(archive.read('settings.json'))
                self.save_settings({**settings, 'clear_token': not settings.get('token')})
            for item in preview['files']:
                name = item['path']
                if name != 'settings.json':
                    target = self.editable(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    atomic(target, archive.read(name))
        return preview
