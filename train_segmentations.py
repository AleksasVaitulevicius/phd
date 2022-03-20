import os
import argparse
import csv

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--min', type=int, default=0)
	parser.add_argument('--max', type=int, default=24)
	parser.add_argument('--model', type=str, default='svm')
	args = parser.parse_args()

	for i in range(1, 10):
		dest = f'{args.model}_metrics_v{i}.csv'
		if not os.path.exists(dest):
			with open(dest, mode='w', newline='') as test_file:
				file_writer = csv.writer(test_file)
				file_writer.writerow(
					[
						'segmentation', 'F1', 'precision', 'recall', 'accuracy', 'specificity',
						'train_F1', 'train_precision', 'train_recall', 'train_accuracy', 'train_specificity',
					]
				)
			break

	for segment in range(args.min, args.max + 1):
		print(segment)
		os.system(f'python ./train.py --segment {segment} --model {args.model} --dest {dest}')