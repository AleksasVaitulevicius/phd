import os
import pandas as pd
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_segments', type=int, required=True)
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    args = parser.parse_args()

    path_processed = './processed/'
    metadata = pd.read_csv(f'{path_processed}metadata.csv')
    # metadata = pd.read_csv('./inpsecting-1.csv')

    for patient, slice_no in metadata.query(args.query)[['patient', 'slice']].drop_duplicates().sort_values(['patient', 'slice']).to_records(index=False):
    # for patient, slice_no in metadata.query(args.query)[['patient_id', 'slice_id']].drop_duplicates().to_records(index=False):
        os.system(
            f'python3 ./slic_slice-v1.py --patient {patient} --slice {slice_no} --n_segments {args.n_segments} --dest {args.dest}'
        )