import os
import cv2 as cv
import argparse
import numpy as np
import pandas as pd
import itertools

from matplotlib import pyplot as plt
from skimage.segmentation import slic, mark_boundaries

path_original =  './data/'
path_processed = './processed/'
path_norm =      './processed/full_normalized/'
path_segments =  './processed/segmented/'

path_maskP =     './processed/mask_prostate/'
path_maskC =     './processed/mask_cancer/'
path_maskB =     './processed/mask_biopsy/'


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
	prostate_mask = cv.imread(f'{path_maskP}/{patient}/{slice_no}.png', cv.IMREAD_GRAYSCALE)
	if np.all(prostate_mask == 0):
		return None
	cancer_mask = cv.imread(f'{path_maskC}/{patient}/{slice_no}.png', cv.IMREAD_GRAYSCALE)
	benign, malignant, undetermined = get_biopsy_masks(patient_str, slice_no)
	if not os.path.exists(f'{path_segments}{patient}-{n_segments}'):
		os.makedirs(f'{path_segments}{patient}-{n_segments}')
		os.makedirs(f'{path_segments}{patient}-{n_segments}/contour')

	cycle_ids = get_cycles(patient, slice_no, path_norm)
	return [
		slic_cycle(
			patient, slice_no, cycle_id, n_segments,
			prostate_mask, cancer_mask, benign, malignant, undetermined, to_number,
		)
		for cycle_id in sorted(cycle_ids)
	]


def get_biopsy_masks(patient, slice_no):
	biopsy_mask = cv.imread(
		f'{path_original}{patient}/biopsyMask/{slice_no}.png', cv.IMREAD_ANYDEPTH
	)
	biopsies = (
		pd.read_csv(
			f'./biopsies/{patient}.csv',
			header=None,
			names=['biopsy_id', 'type', 'a', 'b'],
		)
	)
	malignant = biopsy_mask.copy()
	for biopsy in biopsies.query("type != ' Malignant'")['biopsy_id'].to_list():
		malignant[malignant == biopsy] = 0
	malignant[malignant != 0] = 255
	benign = biopsy_mask.copy()
	for biopsy in biopsies.query("type != ' Benign'")['biopsy_id'].to_list():
		benign[benign == biopsy] = 0
	benign[benign != 0] = 255
	undetermined = biopsy_mask.copy()
	for biopsy in biopsies.query("type != ' Benign'")['biopsy_id'].to_list():
		undetermined[undetermined == biopsy] = 0
	undetermined[undetermined != 0] = 255
	return benign, malignant, undetermined


def slic_cycle(
	patient, slice_no, cycle_id, n_segments,
	prostate_mask, cancer_mask, benign, malignant, undetermined,
	to_number=False,
):
	cycle_slice_i = f'{cycle_id}-{slice_no}.png'
	slice_i = cv.imread(f'{path_norm}{patient}/{cycle_slice_i}')
	segments = slic(
		slice_i,
		n_segments=n_segments,
		compactness=7,
		start_label=1,
		mask=prostate_mask,
	)

	slice_i, unconfirmed_cancer = color_segments(
		cycle_id, n_segments, cancer_mask, malignant, slice_i, segments,
		(0.5, 1), (-1, 0), (1, 0.5, 0), to_number, # SLIGHTLY RED UNCONFIRMED
		# (0.5, 1), (-1, 0), (1, 1, 1), to_number, # WHITE UNCONFIRMED
		# (0.5, 1), (-1, 0), (0, 0, 1), to_number, # BLUE UNCONFIRMED
	)
	slice_i, _ = color_segments(
		cycle_id, n_segments, cancer_mask, malignant, slice_i, segments,
		(-1, 0.5), (-1, 1), (1, 1, 1), to_number, # WHITE NEITHER
		# (-1, 0.5), (-1, 1), (0, 0, 1), to_number, # BLUE NEITHER
	)
	slice_i, confirmed_cancer = color_segments(
		cycle_id, n_segments, cancer_mask, malignant, slice_i, segments,
		(0.5, 1), (0, 1), (1, 0, 0), to_number, # RED CONFIRMED
		# (0.5, 1), (0, 1), (0, 0, 1), to_number, # BLUE CONFIRMED
	)
	slice_i, benign_zones = color_segments(
		cycle_id, n_segments, cancer_mask, malignant + undetermined, slice_i, segments,
		(-1, 0), (-1, 0), (0, 1, 0), to_number, # GREEN BENIGN
		# (-1, 0), (-1, 0), (0, 0, 1), to_number, # BLUE BENIGN
	)
	slice_i = mark_boundaries(slice_i, cancer_mask, color=(0, 1, 1))
	slice_i = mark_boundaries(slice_i, malignant, color=(1, 0, 0))
	slice_i = mark_boundaries(slice_i, benign, color=(0, 1, 0))
	slice_i = mark_boundaries(slice_i, undetermined, color=(0, 0, 1))
	save_image(
		mark_boundaries(slice_i, cancer_mask, color=(1, 0, 1)),
		f'{path_segments}{patient}-{n_segments}/contour/{cycle_id}-{slice_no}.png',
	)
	return (segments, cycle_id, confirmed_cancer, unconfirmed_cancer, benign_zones)


def color_segments(
	cycle_id, n_segments, cancer_mask, biopsy, image, segments,
	cancer_coverage=(0.8, 1), biopsy_coverage=(0, 1), to_color=(1, 0, 0),
	to_number=False,
):
	marked = image
	marked_segments = []
	for segment_id in list(range(1, n_segments + 1)):
		region = image[np.where(segments == segment_id)]
		cancer_in_region = image[np.where(cancer_mask * segments / 255 == segment_id)]
		biopsy_in_region = image[np.where(biopsy * segments / 255 == segment_id)]
		if (
			region.shape[0] != 0
			and cancer_in_region.shape[0] / region.shape[0] > cancer_coverage[0]
			and cancer_in_region.shape[0] / region.shape[0] <= cancer_coverage[1]
			and biopsy_in_region.shape[0] / region.shape[0] > biopsy_coverage[0]
			and biopsy_in_region.shape[0] / region.shape[0] <= biopsy_coverage[1]
		):
			marked_segments += [segment_id]
			marked = mark_boundaries(marked, segments == segment_id, color=to_color)
			if to_number:
				coords = np.argwhere(segments == segment_id)
				x = np.quantile([axis for axis, _ in coords], 0.5)
				y = np.quantile([axis_y for axis_x, axis_y in coords], 0.5)
				cv.putText(
					marked, str(segment_id), (int(y), int(x)),
					fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
					fontScale=0.35,
					color=(0, 1, 1),
				)
	
	return marked, marked_segments


def save_image(image, path):
	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(image)
	fig.savefig(path, dpi=150)
	plt.close()

def segments_means_foreach_slice(
	patient, slice_no, segments, n_segments,
	confirmed_cancer, unconfirmed_cancer, benign_zones,
	agg_func=np.median, normalize_to=None, drop_start_by=0,drop_end_by=0,
	segmented_in_cycle=None
):
	cycles = get_cycles(patient, slice_no, path_norm)
	slices = {
		cycle_id: cv.imread(f'{path_norm}{patient}/{cycle_id}-{slice_no}.png')
		for cycle_id in sorted(cycles)
	}
	segment_means = [
		(
			segment_id, patient, cycle_id, slice_no,
			get_segments_mean(slices[cycle_id], segment_id, segments, cycle_id, agg_func),
			segment_id in confirmed_cancer,
			segment_id in unconfirmed_cancer,
			segment_id in benign_zones,
			segmented_in_cycle,
		)
		for segment_id, cycle_id  in itertools.product(
			range(1, n_segments + 1),
			cycles
		)
	]
	if normalize_to:
		max_vals = segment_means.groupby('img_type').mask_int_mean.transform('max')
		segment_means['mask_int_mean'] = segment_means.mask_int_mean / max_vals
		segment_means['mask_int_mean'] = segment_means.mask_int_mean * normalize_to
	return segment_means


def get_segments_mean(slice_i, seg_i, segments, cycle_id, agg_func=np.median):
	selected_region = slice_i[np.where(segments == seg_i)]
	if selected_region.shape[0] == 0:
		return None
	if agg_func:
		return agg_func(selected_region)
	return ','.join(selected_region.reshape(-1).astype('str'))


def process_slic(patient, slice_no, n_segments, dest):
	print(f'patient {patient}, slice={slice_no}')
	segments_list = slic_slice(patient, slice_no, n_segments)
	if not segments_list:
		return
	segment_means = [
		segments_means_foreach_slice(
			patient, slice_no, segments, n_segments,
			confirmed_cancer, unconfirmed_cancer, benign_zones,
			np.median, segmented_in_cycle=cycle_id,
		)
		for segments, cycle_id, confirmed_cancer, unconfirmed_cancer, benign_zones in segments_list
	]

	(
        pd.DataFrame(
		np.concatenate(segment_means),
		columns=[
			'img_type', 'patient_id', 'cycle_id', 'slice_id', 'mask_int_mean',
			'confirmed_cancer', 'unconfirmed_cancer', 'benign', 'segment',
		]
        ).assign(
            confirmed_cancer=lambda y: y.confirmed_cancer.astype(int),
            unconfirmed_cancer=lambda y: y.unconfirmed_cancer.astype(int),
            benign=lambda y: y.benign.astype(int),
        )
        .to_csv(dest, index=False, mode='a', header=False)
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', type=int, required=True)
    parser.add_argument('--slice', type=int, required=True)
    parser.add_argument('--n_segments', type=int, required=True)
    parser.add_argument('--dest', type=str, required=True)

    args = parser.parse_args()
    process_slic(args.patient, args.slice, args.n_segments, args.dest)