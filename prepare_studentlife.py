"""Fetch only four public archive members, check CRCs, and prepare measured rows."""
import argparse
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib
from campuspulse.studentlife import prepare, read_rds

ROOT = Path(__file__).resolve().parent
URL = 'https://zenodo.org/records/3529253/files/dataset_rds.zip'

def download_member(name, entry):
    # Immutable archive manifest. Request one header+payload instead of the 230 MB archive.
    offset = entry['offset']
    stop = offset + entry['compressed_size'] + 300
    request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{stop}'})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise RuntimeError('Server did not honor byte ranges. Download the full archive and use --archive PATH.')
        data = response.read(stop - offset + 1)
    fields = struct.unpack_from('<4s5H3I2H', data)
    if fields[0] != b'PK\x03\x04':
        raise ValueError('Archive member header mismatch.')
    filename_size, extra_size = fields[-2:]
    if data[30:30 + filename_size].decode() != name:
        raise ValueError('Archive member name mismatch.')
    begin = 30 + filename_size + extra_size
    raw = data[begin:begin + entry['compressed_size']]
    if entry['method'] == 8:
        raw = zlib.decompress(raw, -15)
    if len(raw) != entry['size'] or zlib.crc32(raw) != entry['crc32']:
        raise ValueError('Archive member integrity check failed.')
    return raw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, help='Optional locally downloaded dataset_rds.zip')
    parser.add_argument('--rds-dir', type=Path, help='Use previously extracted RDS files')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/private_studentlife')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.rds_dir or args.output / 'raw'
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'configs/studentlife_members.json').read_text())
    tables = {}
    for name, entry in manifest.items():
        path = cache / Path(name).name
        if not path.exists():
            if args.rds_dir:
                raise FileNotFoundError(path)
            print('Fetching', Path(name).name, flush=True)
            if args.archive:
                with zipfile.ZipFile(args.archive) as archive:
                    raw = archive.read(name)
            else:
                raw = download_member(name, entry)
            path.write_bytes(raw)
        raw = path.read_bytes()
        if len(raw) != entry['size'] or zlib.crc32(raw) != entry['crc32']:
            raise ValueError(f'Cached file failed integrity check: {path}')
        tables[Path(name).stem] = read_rds(path)
    features, metadata, coverage = prepare(tables['LonelinessScale'], tables['Sleep'],
                                          tables['conversation'], tables['phonelock'])
    features.to_csv(args.output / 'features.csv', index=False)
    coverage.to_csv(args.output / 'coverage.csv', index=False)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Prepared {len(features)} real participants from {metadata["post_surveys"]} complete post surveys.')
    print('Output:', args.output)

if __name__ == '__main__':
    main()
