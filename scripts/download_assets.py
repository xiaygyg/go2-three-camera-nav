"""Download public, revision-pinned runtime files and verify their SHA-256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]


def fetch(item):
    destination = (ROOT / item['destination']).resolve()
    if not destination.is_relative_to(ROOT / 'assets'):
        raise ValueError('Asset destination escapes assets/')
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == item['sha256']:
        return
    response = requests.get(item['url'], timeout=120)
    response.raise_for_status()
    content = response.content
    if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
        raise ValueError('Asset verification failed: ' + item['destination'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(destination)


def main():
    items = json.loads((ROOT / 'assets/sources.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, items))
    print(f'Verified {len(items)} pinned public assets. See THIRD_PARTY.md for source terms.')


if __name__ == '__main__':
    main()
