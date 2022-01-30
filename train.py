import os
import numpy as np
import pandas as pd
import argparse

from skfda import FDataGrid
from skfda.representation.basis import BSpline
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import ElasticRegistration, landmark_registration

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth, BandDepth
from sklearn.model_selection import StratifiedKFold
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import (
	precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
)


agg_columns = ['patient_id', 'slice_id', 'img_type']

n_basis=18
order=4

prc_rm=0.05
n_points=100

train_n_points = 10

# FILE_OUTPUT = './svm_metrics.csv'
# FILE_OUTPUT = './rf_metrics.csv'
FILE_OUTPUT = './xgb_metrics.csv'

basis = BSpline(domain_range=(0, 1), n_basis=n_basis, order=order)
smoother = BasisSmoother(basis=basis, return_basis=True, method='svd')

registration = ElasticRegistration()
ID = IntegratedDepth()
MBD = ModifiedBandDepth()

# scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
l_enc = LabelEncoder()

kfolds = StratifiedKFold(n_splits=5)

def get_model():
	# return SVC(class_weight='balanced')
	# return RandomForestClassifier(class_weight='balanced', n_jobs=4)
    return xgb.XGBClassifier(tree_method='gpu_hist', eval_metric='logloss', use_label_encoder=False)


def cut_ends(bsplined, order=0, prc_rm_start=prc_rm, prc_rm_end=prc_rm, n_points=n_points):
	bsplined_grid = bsplined.derivative(order=order).to_grid(np.linspace(0, 1, n_points))
	return FDataGrid(
		data_matrix=bsplined_grid.data_matrix[
			..., int(n_points * prc_rm_start): int(n_points * (1 - prc_rm_end)), 0
		],
		grid_points=bsplined_grid.grid_points[0][
			int(n_points * prc_rm_start): int(n_points * (1 - prc_rm_end))
		]
	)


def get_landmark_registration(bsplined, order=0):
	bsplined_grid = cut_ends(bsplined, order)
	landmark_indexes = cut_ends(bsplined, order, prc_rm_end=0.5).data_matrix.argmax(axis=1)
	grid_points = bsplined_grid.grid_points[0]
	landmarks = [grid_points[index] for index in np.concatenate(landmark_indexes)]
	return landmark_registration(bsplined_grid, landmarks)


def get_descrete_points(fd_smooth):
	t_cut = np.linspace(
		start=0,
		stop=fd_smooth.data_matrix.shape[1] - 1,
		num=train_n_points, endpoint=True, dtype=int,
	)
	return scaler.fit_transform(fd_smooth.data_matrix[:, t_cut, 0])


def specificity(y_true, y_pred, zero_division=0):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	if tn+fp == 0 and zero_division:
		return zero_division
	return tn / (tn+fp)


def train_kfolds(Y, X):
	f1s = []
	precisions = []
	recalls = []
	accuracies = []
	specificities = []

	train_f1s = []
	train_precisions = []
	train_recalls = []
	train_accuracies = []
	train_specificities = []

	i = 0
	for train_index, test_index in kfolds.split(X, Y):
		print(i)
		i += 1
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
        
		model = get_model()
		model.fit(X_train, y_train)
		preds = model.predict(X_test)
        
		f1s += [f1_score(y_test, preds, zero_division=0)]
		precisions += [precision_score(y_test, preds, zero_division=0)]
		recalls += [recall_score(y_test, preds, zero_division=0)]
		accuracies += [balanced_accuracy_score(y_test, preds)]
		specificities += [specificity(y_test, preds)]

		preds = model.predict(X_train)
		train_f1s += [f1_score(y_train, preds, zero_division=0)]
		train_precisions += [precision_score(y_train, preds, zero_division=0)]
		train_recalls += [recall_score(y_train, preds, zero_division=0)]
		train_accuracies += [balanced_accuracy_score(y_train, preds)]
		train_specificities += [specificity(y_train, preds)]
	
	return (
		np.mean(f1s),
		np.mean(precisions),
		np.mean(recalls),
		np.mean(accuracies),
		np.mean(specificities),
		
		np.mean(train_f1s),
		np.mean(train_precisions),
		np.mean(train_recalls),
		np.mean(train_accuracies),
		np.mean(train_specificities),
	)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation', type=int, required=True)

    args = parser.parse_args()
    
    segmentation_id = args.segmentation
    file = f'./segmentations/segmentation-{segmentation_id}.csv'
    dataset = (
        pd.read_csv(
            file,
			dtype={
				'img_type': int,
				'patient_id': int,
				'cycle_id': int,
				'slice_id': int,
				'label': str,
				'mask_int_mean': float,
				'segment': int,
			},
        )
            .assign(label=lambda x: x.label.astype(str) == 'True')
            .drop_duplicates()
            .sort_values(agg_columns + ['cycle_id'])
    )
    ts = (
        dataset[['patient_id', 'cycle_id']].drop_duplicates()
            .groupby('patient_id').cycle_id.count()
            .apply(lambda x: np.linspace(0, 1, int(x)))
            .reset_index()
    )

    dataset = dataset.groupby(agg_columns + ['label']).mask_int_mean.apply(list).reset_index()
    bsplined = dataset.groupby('patient_id').mask_int_mean.apply(list).reset_index().merge(ts)
    bsplined = bsplined.apply(
        lambda x: smoother.fit_transform(
            FDataGrid(data_matrix=x['mask_int_mean'], grid_points=x['cycle_id'])
        ),
        axis='columns',
    )
    print('registering ...')
    bsplined = [get_landmark_registration(fd_smooth, 1) for fd_smooth in bsplined]
    print('getting depths ...')
    ids = np.vstack(
        [scaler.fit_transform(ID(fd_smooth).reshape(-1, 1)) for fd_smooth in bsplined]
    )
    mbds = np.vstack(
        [scaler.fit_transform(MBD(fd_smooth).reshape(-1, 1)) for fd_smooth in bsplined]
    )
    print('getting other features ...')
    max_values = np.vstack(
        [
            scaler.fit_transform(fd_smooth.data_matrix.max(axis=1).reshape(-1, 1))
            for fd_smooth in bsplined
        ]
    )
    max_points = np.vstack(
        [
            scaler.fit_transform(
                (fd_smooth.data_matrix.argmax(axis=1) / n_points).reshape(-1, 1)
            )
            for fd_smooth in bsplined
        ]
    )

    descret_points = np.vstack([get_descrete_points(fd_smooth) for fd_smooth in bsplined])
    train_set = np.hstack([ids, mbds, max_values, max_points, descret_points])
    # labels = l_enc.fit_transform(dataset.label.astype(int).tolist())
    labels = dataset.label.astype(int).to_numpy()

    pd.DataFrame(
        [(segmentation_id,) + train_kfolds(labels, train_set)],
        columns=[
            'segmentation', 'F1', 'precision', 'recall', 'accuracy', 'specificity',
            'train_F1', 'train_precision', 'train_recall', 'train_accuracy', 'train_specificity',
        ]
    ).to_csv(FILE_OUTPUT, index=False, mode='a', header=False)

