"""Optional one-time local copy of data from a previous CampusPulse folder.
Existing destination files are preserved. The source is never removed.
"""
from pathlib import Path
import shutil


def main():
    text = input('Paste your OLD campuspulse folder path (not the new one): ').strip().strip('"')
    source = Path(text).expanduser().resolve() / 'data'
    target = Path(__file__).resolve().parent / 'data'
    if not source.is_dir():
        raise SystemExit('No data folder found at that path. No files copied.')
    if source == target.resolve() or source in target.resolve().parents or target.resolve() in source.parents:
        raise SystemExit('Choose a separate old project folder. No files copied.')
    copied = skipped = 0
    for path in source.rglob('*'):
        if path.is_symlink() or not path.is_file():
            continue
        destination = target / path.relative_to(source)
        if destination.exists():
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    print(f'Copied {copied} files; kept {skipped} existing files. Old folder unchanged.')
    print('Research data stays on your computer. Do not publish private data folders.')


if __name__ == '__main__':
    main()
