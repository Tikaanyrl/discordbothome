"""Create a private migration ZIP from existing JSON files; never include env/token."""
from pathlib import Path
import json
import time
import zipfile

root = Path(__file__).resolve().parents[1]
files = sorted(root.glob('user_*.json'))
if not files:
    raise SystemExit('Keine vorhandenen user_*.json-Dateien gefunden.')
for p in files:
    json.loads(p.read_text(encoding='utf-8'))
destination = root / 'private'
destination.mkdir(mode=0o700, exist_ok=True)
archive = destination / f'bot-daten-{time.time_ns()}.zip'
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
    for p in files:
        output.write(p, 'storage/' + p.name)
archive.chmod(0o600)
print(archive)
