"""Run from the project folder: python train.py --demo"""
import argparse
import json
from pathlib import Path
import pandas as pd
from campuspulse.data import synthetic_data
from campuspulse.model import train

def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true')
    source.add_argument('--csv', type=Path)
    parser.add_argument('--target-description')
    parser.add_argument('--output', default='artifacts')
    args = parser.parse_args()
    if args.csv and not args.target_description:
        parser.error('--csv requires --target-description describing measurement and timing')
    if args.demo:
        df = synthetic_data()
        Path('data').mkdir(exist_ok=True)
        df.to_csv('data/synthetic_demo.csv', index=False)
    else:
        df = pd.read_csv(args.csv)
    report = train(df, args.output, 'synthetic' if args.demo else 'user_supplied_unverified',
                   args.target_description or 'Invented 20–80 score; not a real survey')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
