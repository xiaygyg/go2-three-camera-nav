"""Audit an explicit publication file list; never print secret values."""
import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private_robot': re.compile(r'dog' + r'70|xt' + r'70|hrl_' + r'xt', re.I),
    'private_address': re.compile(r'(?<!\d)(?:192\.168\.\d{1,3}\.\d{1,3}|100\.78\.\d{1,3}\.\d{1,3})(?!\d)'),
    'local_home': re.compile(r'/home/(?!runner\b|user\b|USER\b)[A-Za-z0-9_.-]+|[CD]:[\\/](?:Users|codex|文档)[\\/]'),
    'token': re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----'),
    'credential_url': re.compile(r'https?://[^\s/]+:[^\s/]+@'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tracked', action='store_true')
    a = p.parse_args()
    if a.tracked:
        names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        files = [ROOT / n for n in names if n]
    else:
        files = [f for f in ROOT.rglob('*') if f.is_file() and not any(
            part in ['.git', '.venv', '.runtime', '__pycache__', 'downloaded', 'pretrained', 'results', 'runs'] for part in f.relative_to(ROOT).parts)
            and f.name != 'go2_initial.pt']
    issues = []
    for f in files:
        if f.stat().st_size > 50 * 1024**2:
            issues.append((str(f.relative_to(ROOT)), 'large_file'))
        if f.suffix in ['.pt', '.gif', '.png', '.jpg']:
            continue
        try:
            content = f.read_text(encoding='utf-8-sig')
        except UnicodeError:
            issues.append((str(f.relative_to(ROOT)), 'unreviewed_binary'))
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(content):
                issues.append((str(f.relative_to(ROOT)), name))
    if issues:
        for path, kind in issues:
            print(path, kind)
        raise SystemExit('Publication audit failed; no matching secret values were printed.')
    print(f'Publication text/size audit passed for {len(files)} files. Review checkpoint metadata separately.')


if __name__ == '__main__':
    main()
