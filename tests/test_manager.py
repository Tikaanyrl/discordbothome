import io
import json
from pathlib import Path
import signal
import stat
import sys
import time
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'discord_bot_manager/manager'))
from core import Manager, Problem
from app import create_app


@pytest.fixture
def manager(tmp_path):
    seed = tmp_path / 'seed'
    seed.mkdir()
    (seed / 'bot.py').write_text('import time\nprint("ready", flush=True)\ntime.sleep(60)\n')
    (seed / 'requirements.txt').write_text('')
    m = Manager(tmp_path / 'share', tmp_path / 'data', seed, sys.executable)
    yield m
    m.closing = True
    m.stop()


@pytest.fixture
def client(manager):
    c = create_app(manager, local=True).test_client()
    token = c.get('/api/state').json['csrf']
    c.environ_base['HTTP_X_CSRF_TOKEN'] = token
    return c


def wait_until(test, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if test():
            return
        time.sleep(.02)
    assert test()


def test_process_unique_stop_and_persistence(manager):
    data = manager.root / 'storage/keep.json'
    data.write_text('{"keep":true}')
    manager.start()
    pid = manager.proc.pid
    manager.start()
    assert manager.proc.pid == pid
    wait_until(lambda: 'ready' in manager.lines)
    manager.stop()
    assert not manager.status()['running']
    assert not manager.wanted
    manager.start()
    manager.stop()
    assert data.read_text() == '{"keep":true}'


def test_shutdown_allows_bot_to_flush(manager):
    (manager.root / 'code/bot.py').write_text('''import signal, time
from pathlib import Path
def stop(*args):
    Path('saved.txt').write_text('flushed')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
time.sleep(60)
''')
    manager.start()
    wait_until(lambda: 'ready' in manager.lines)
    manager.stop()
    assert (manager.root / 'storage/saved.txt').read_text() == 'flushed'


def test_crash_keeps_ui_alive(manager, client):
    (manager.root / 'code/bot.py').write_text('raise RuntimeError("example failure")')
    manager.settings['auto_restart'] = False
    manager.start()
    wait_until(lambda: manager.proc is None)
    assert 'Exit-Code 1' in manager.last_error
    assert client.get('/').status_code == 200
    assert client.get('/api/state').json['status']['running'] is False


def test_restart_limit_and_manual_stop(manager, monkeypatch):
    # Make backoff immediate without changing the retry policy.
    original_sleep = time.sleep
    monkeypatch.setattr('core.time.sleep', lambda seconds: original_sleep(.01))
    (manager.root / 'code/bot.py').write_text('raise SystemExit(1)')
    manager.start()
    wait_until(lambda: manager.retries == 3 and not manager.wanted)
    assert manager.proc is None
    assert manager.retries == 3
    manager.start()
    manager.stop()
    original_sleep(.1)
    assert manager.proc is None and not manager.wanted


@pytest.mark.parametrize('path', ['../env', '/etc/passwd', 'code/../../env', 'storage/../code/a', 'code\\a'])
def test_path_escape(manager, path):
    with pytest.raises(Problem):
        manager.safe(path)


def test_symlink_escape(manager, tmp_path, client):
    (manager.root / 'code/link').symlink_to(tmp_path, target_is_directory=True)
    assert client.get('/api/file?path=code/link/secret').status_code == 400
    assert client.post('/api/file', json={'path': 'code/link/secret', 'text': 'oops'}).status_code == 400
    assert not (tmp_path / 'secret').exists()


def test_auth_and_csrf(manager):
    c = create_app(manager).test_client()
    assert c.get('/api/state').status_code == 403
    assert c.get('/api/state', environ_overrides={'REMOTE_ADDR':'172.30.32.2'}).status_code == 200
    assert c.post('/api/control/start', environ_overrides={'REMOTE_ADDR':'172.30.32.2'}).status_code == 403
    assert c.get('/api/state', headers={'X-Forwarded-For':'172.30.32.2'}).status_code == 403


def test_edit_stop_requirement_and_trash(manager, client):
    body = {'path': 'storage/test.txt', 'text': 'precious'}
    assert client.post('/api/file', json=body).status_code == 200
    assert client.post('/api/file', json=body).status_code == 400
    manager.start()
    assert client.post('/api/file', json={**body, 'overwrite':True}).status_code == 400
    manager.stop()
    assert client.post('/api/files/trash', json={'path':body['path']}).status_code == 200
    saved = next((manager.root / 'trash').rglob('test.txt'))
    assert saved.read_text() == 'precious'
    assert client.post('/api/files/recover', json={'path':saved.relative_to(manager.root).as_posix(), 'target':body['path']}).status_code == 200
    assert (manager.root / body['path']).read_text() == 'precious'


def test_backup_restore_preserves_extra_and_previous(manager):
    file = manager.root / 'storage/data.json'
    file.write_text('original')
    manager.save_settings({'token':'secret-token'})
    archive = manager.backup()
    file.write_text('changed')
    (manager.root / 'storage/extra.txt').write_text('extra')
    preview = manager.archive_preview(archive)
    manager.restore(archive, preview['sha256'])
    assert file.read_text() == 'original'
    assert (manager.root / 'storage/extra.txt').read_text() == 'extra'
    other = [p for p in (manager.root / 'backups').glob('*.zip') if p.name != Path(archive).name]
    assert len(other) == 1
    with zipfile.ZipFile(other[0]) as z:
        assert z.read('storage/data.json') == b'changed'
    assert manager.settings['token'] == 'secret-token'


@pytest.mark.parametrize('name', ['../escape', 'code/../../escape', '/storage/x', 'logs/overwrite.log'])
def test_bad_archive(manager, name):
    p = manager.root / 'backups/evil.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr(name, 'evil')
    with pytest.raises(Problem):
        manager.archive_preview('backups/evil.zip')


def test_archive_symlink(manager):
    p = manager.root / 'backups/evil.zip'
    with zipfile.ZipFile(p,'w') as z:
        info = zipfile.ZipInfo('code/link')
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, '/etc')
    with pytest.raises(Problem):
        manager.archive_preview('backups/evil.zip')


def test_token_mask_and_settings_permissions(manager, client):
    assert client.post('/api/settings',json={'token':'test-super-secret'}).status_code == 200
    assert 'test-super-secret' not in client.get('/api/state').text
    manager.log('oops test-super-secret')
    assert 'test-super-secret' not in manager.log_path.read_text()
    assert 'test-super-secret' not in client.get('/api/logs').text
    assert stat.S_IMODE(manager.settings_path.stat().st_mode) == 0o600


def test_seed_update_never_overwrites(manager, tmp_path):
    p = manager.root / 'code/bot.py'
    p.write_text('custom code')
    data = manager.root / 'storage/precious.json'
    data.write_text('original')
    other = Manager(manager.root, manager.data, tmp_path / 'seed', sys.executable)
    assert p.read_text() == 'custom code'
    assert data.read_text() == 'original'


def test_failed_atomic_write_preserves_old_file(manager, client, monkeypatch):
    p = manager.root / 'storage/data.txt'
    p.write_text('original')
    def fail(*args):
        raise OSError('No space left on device')
    monkeypatch.setattr('core.os.replace', fail)
    response = client.post('/api/file',json={'path':'storage/data.txt','text':'changed','overwrite':True})
    assert response.status_code == 400
    assert 'No space' in response.json['error']
    assert p.read_text() == 'original'


def test_restore_digest_required(manager):
    archive = manager.backup()
    with pytest.raises(Problem):
        manager.restore(archive,'wrong')


def test_restore_empty_token_clears_current_secret(manager):
    archive = manager.backup()
    manager.save_settings({'token':'later-secret'})
    manager.restore(archive, manager.archive_preview(archive)['sha256'])
    assert manager.settings['token'] == ''


def test_noncanonical_archive_paths_rejected(manager):
    with pytest.raises(Problem):
        manager.safe('code//file.py')
    with pytest.raises(Problem):
        manager.safe('code/./file.py')


def test_restore_refuses_running_bot(manager):
    archive = manager.backup()
    digest = manager.archive_preview(archive)['sha256']
    manager.start()
    with pytest.raises(Problem):
        manager.restore(archive, digest)
    assert manager.proc is not None


def test_backup_resumes_running_bot(manager):
    manager.start()
    archive = manager.backup()
    assert manager.proc is not None
    assert manager.safe(archive).is_file()


def test_migration_zip_matches_originals():
    root = Path(__file__).resolve().parents[1]
    archives = sorted((root / 'private').glob('bot-daten-*.zip'))
    if not archives:
        pytest.skip('Private migration files not present in public repository')
    with zipfile.ZipFile(archives[-1]) as z:
        assert all(n.startswith('storage/user_') and n.endswith('.json') for n in z.namelist())
        for name in z.namelist():
            assert z.read(name) == (root / Path(name).name).read_bytes()
