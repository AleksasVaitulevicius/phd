import os
import pandas as pd
import argparse
import csv

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--dest', type=str, required=True)
	parser.add_argument('--query', type=str, default='slic_size == slic_size')
	args = parser.parse_args()


	for i in range(0, 10):
		dest = f'{args.dest}-{i}.csv'
		if not os.path.exists(dest):
			with open(dest, mode='w', newline='') as test_file:
				file_writer = csv.writer(test_file)
				file_writer.writerow(
					[
						'img_type', 'patient_id', 'cycle_id', 'slice_id', 'mask_int_mean',
						'confirmed_cancer', 'confirmed_cancer_less',
						'unconfirmed_cancer', 'unconfirmed_cancer_less', 'benign', 'segment',
					]
				)
			break

	metadata = (
		pd.read_csv('./processed/metadata.csv')
			.query('slic_size != 0 and slic_size == slic_size and has_prostate')
			.assign(slic_size=lambda x: x.slic_size.astype(int))
	)

	for patient, slice_no, n_segments in (
		metadata
			.query(args.query)
			[['patient', 'slice', 'slic_size']]
			.drop_duplicates()
			.sort_values(['patient', 'slice'])
			.to_records(index=False)
	):
		os.system(
			f'python3 ./slic_slice_new_labels.py --patient {patient} --slice {slice_no} --n_segments {n_segments} --dest {dest}'
		)