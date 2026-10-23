"""Download public, revision-pinned runtime files and verify their SHA-256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def fetch(item):
    destination = (ROOT / item['destination']).resolve()
    if not destination.is_relative_to(ROOT / 'assets'):
        raise ValueError('Asset destination escapes assets/')
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == item['sha256']:
        return
    # Some networks cannot reach raw.githubusercontent.com. Both routes refer
    # to exactly the same pinned public revision; neither bypasses hash checks.
    owner, repo, revision, path = urlsplit(item['url']).path.lstrip('/').split('/', 3)
    api = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={revision}'
    failures = []
    for url in [api, item['url']]:
        try:
            response = requests.get(url, headers={'Accept': 'application/vnd.github.raw+json'},
                                    timeout=(7, 45))
            response.raise_for_status()
            content = response.content
            if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('Asset verification failed')
            break
        except (requests.RequestException, ValueError) as error:
            failures.append(type(error).__name__)
    else:
        raise RuntimeError('Could not fetch verified asset: ' + item['destination'] + ' / ' + ', '.join(failures))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(destination)
    print('Verified', item['destination'], flush=True)


def main():
    items = json.loads((ROOT / 'assets/sources.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, items))
    print(f'Verified {len(items)} pinned public assets. See THIRD_PARTY.md for source terms.')


if __name__ == '__main__':
    main()
