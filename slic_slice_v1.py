import os
from PIL import Image
import argparse
import numpy as np
import pandas as pd
import itertools

from skimage.segmentation import slic, mark_boundaries

path_norm =      './processed/full_normalized/'
path_maskP =     './processed/mask_prostate/'
path_segments =  './processed/segmented/'


def get_cycles(patient, slice_no, path_norm):
	return [
		int(os.path.splitext(file)[0].split('-')[0])
		for file in os.listdir(f'{path_norm}{patient}')
		if '-' in file and os.path.splitext(file)[0].split('-')[1] == str(slice_no)
	]


def slic_slice(
	patient, slice_no, n_segments, to_number=False
):
	functions = {
		10: lambda x: f'00{x}',
		100: lambda x: f'0{x}',
		1000: lambda x: str(x),
	}
	patient_str = [func(patient) for limit, func in functions.items() if patient < limit][0]
	prostate_mask = load_image(f'{path_maskP}/{patient}/{slice_no}.png')
	if np.all(prostate_mask == np.zeros(prostate_mask.shape)):
		return None

	cancer_mask = load_image(f'./filtered_cancer_masks/{patient_str}/regionMaskByType/{slice_no}.png')

	if not os.path.exists(f'{path_segments}{patient}'):
		os.makedirs(f'{path_segments}{patient}')

	cycle_ids = get_cycles(patient, slice_no, path_norm)
	return [
		slic_cycle(
			patient, slice_no, cycle_id, n_segments, prostate_mask, cancer_mask, to_number,
		)
		for cycle_id in sorted(cycle_ids)
	]


def slic_cycle(
	patient, slice_no, cycle_id, n_segments, prostate_mask, cancer_mask, to_number=False,
):
	slice_i = load_image(
		f'{path_norm}{patient}/{cycle_id}-{slice_no}.png',
		convert=lambda x: x.convert('RGB'),
	)
	segments = slic(
		slice_i,
		n_segments=n_segments,
		compactness=7,
		start_label=1,
		mask=prostate_mask,
	)

	malignant = cancer_mask.copy()
	malignant[malignant == 1] = 0
	malignant[malignant == 2] = 1
	slice_i, confirmed_cancer = color_segments(
		n_segments, malignant, slice_i, segments, (0.5, 1), (1, 0, 0), to_number,
	)
	slice_i, confirmed_cancer_less = color_segments(
		n_segments, malignant, slice_i, segments, (0, 0.5), (1, 1, 1), to_number,
	)
	undetermined = cancer_mask.copy()
	undetermined[undetermined == 2] = 0
	slice_i, unconfirmed_cancer = color_segments(
		n_segments, undetermined, slice_i, segments, (0.5, 1), (0, 0, 1), to_number,
	)
	slice_i, unconfirmed_cancer_less = color_segments(
		n_segments, undetermined, slice_i, segments, (0, 0.5), (1, 1, 1), to_number,
	)
	both = cancer_mask.copy()
	both[both == 2] = 1
	slice_i, benign_zones = color_segments(
		n_segments, both, slice_i, segments, (-1, 0), (0, 1, 0), to_number,
	)
	# slice_i = mark_boundaries(slice_i, malignant, color=(1, 0, 1))
	# slice_i = mark_boundaries(slice_i, undetermined, color=(0, 1, 1))
	# with Image.fromarray(np.uint8(slice_i * 255)) as img:
	# 	img.save(
	# 		f'{path_segments}{patient}/{cycle_id}-{slice_no}.png'
	# 	)
	return (
        segments, cycle_id,
        confirmed_cancer, confirmed_cancer_less,
        unconfirmed_cancer, unconfirmed_cancer_less,
        benign_zones
    )


def color_segments(
	n_segments, cancer_mask, image, segments,
	cancer_coverage=(0.8, 1), to_color=(1, 0, 0), to_number=False,
):
	marked = image
	marked_segments = []
	for segment_id in list(range(1, n_segments + 1)):
		region = image[np.where(segments == segment_id)]
		cancer_in_region = image[np.where(cancer_mask * segments == segment_id)]
		if (
			region.shape[0] != 0
			and cancer_in_region.shape[0] / region.shape[0] > cancer_coverage[0]
			and cancer_in_region.shape[0] / region.shape[0] <= cancer_coverage[1]
		):
			marked_segments += [segment_id]
			# marked = mark_boundaries(marked, segments == segment_id, color=to_color)
			# if to_number:
			# 	coords = np.argwhere(segments == segment_id)
			# 	x = np.quantile([axis for axis, _ in coords], 0.5)
			# 	y = np.quantile([axis_y for axis_x, axis_y in coords], 0.5)
			# 	cv.putText(
			# 		marked, str(segment_id), (int(y), int(x)),
			# 		fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
			# 		fontScale=0.35,
			# 		color=(0, 1, 1),
			# 	)

	return marked, marked_segments


def aggregate_segments(
	patient, slice_no, segments, n_segments,
	confirmed_cancer, confirmed_cancer_less,
	unconfirmed_cancer, unconfirmed_cancer_less, benign_zones,
	agg_func=np.median, segmented_in_cycle=None
):
	cycles = get_cycles(patient, slice_no, path_norm)
	slices = {
		cycle_id: load_image(f'{path_norm}{patient}/{cycle_id}-{slice_no}.png')
		for cycle_id in sorted(cycles)
	}
	segment_means = [
		(
			segment_id, patient, cycle_id, slice_no,
			aggregate_segment(slices[cycle_id], segment_id, segments, cycle_id, agg_func),
			segment_id in confirmed_cancer,
			segment_id in confirmed_cancer_less,
			segment_id in unconfirmed_cancer,
			segment_id in unconfirmed_cancer_less,
			segment_id in benign_zones,
			segmented_in_cycle,
		)
		for segment_id, cycle_id  in itertools.product(range(1, n_segments + 1), cycles)
	]
	return segment_means


def load_image(path, convert=lambda x: x):
	with Image.open(path) as img:
		matrix = np.array(convert(img))
	return matrix


def aggregate_segment(slice_i, seg_i, segments, cycle_id, agg_func=np.median):
	selected_region = slice_i[np.where(segments == seg_i)]
	if selected_region.shape[0] == 0:
		return None
	return agg_func(selected_region)


def process_slic(patient, slice_no, n_segments, dest):
	print(f'patient {patient}, slice={slice_no}')
	segments_list = slic_slice(patient, slice_no, n_segments)
	if not segments_list:
		return
	segment_means = [
		aggregate_segments(
			patient, slice_no, segments, n_segments,
			confirmed_cancer, confirmed_cancer_less,
			unconfirmed_cancer, unconfirmed_cancer_less,
			benign_zones,
			np.median, segmented_in_cycle=cycle_id,
		)
		for segments, cycle_id,
			confirmed_cancer, confirmed_cancer_less,
			unconfirmed_cancer, unconfirmed_cancer_less,
			benign_zones
		in segments_list
	]

	(
		pd.DataFrame(
			np.concatenate(segment_means),
			columns=[
				'img_type', 'patient_id', 'cycle_id', 'slice_id', 'mask_int_mean',
				'confirmed_cancer', 'confirmed_cancer_less',
				'unconfirmed_cancer', 'unconfirmed_cancer_less', 'benign', 'segment',
			]
		).assign(
			confirmed_cancer=lambda y: y.confirmed_cancer.astype(int),
			confirmed_cancer_less=lambda y: y.confirmed_cancer_less.astype(int),
			unconfirmed_cancer=lambda y: y.unconfirmed_cancer.astype(int),
			unconfirmed_cancer_less=lambda y: y.unconfirmed_cancer_less.astype(int),
			benign=lambda y: y.benign.astype(int),
		)
		.to_csv(dest, index=False, mode='a', header=False)
	)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--patient', type=int, required=True)
	parser.add_argument('--slice', type=int, required=True)
	parser.add_argument('--dest', type=str, required=True)
	parser.add_argument('--n_segments', type=int, default=50)

	args = parser.parse_args()
	process_slic(args.patient, args.slice, args.n_segments, args.dest)
