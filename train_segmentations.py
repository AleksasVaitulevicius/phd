import os
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--min', type=int, required=True)
    parser.add_argument('--max', type=int, required=True)
    args = parser.parse_args()

    for segment in range(args.min, args.max + 1):
        print(segment)
        os.system(f'python ./train.py --segment {segment}')