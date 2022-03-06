import os
import math
import numpy as np
import pandas as pd
import argparse

from skfda import FDataGrid
from skfda.representation.basis import BSpline
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import ElasticRegistration, landmark_registration

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import (
	precision_score, recall_score, f1_score, balanced_accuracy_score, roc_curve, confusion_matrix
)


agg_columns = ['patient_id', 'slice_id', 'img_type']

n_basis=18
order=4

prc_rm=0.05
n_points=100

train_n_points = 10

MODEL = 'svm'
# MODEL = 'rf'
# MODEL = 'xgb'

basis = BSpline(domain_range=(0, 1), n_basis=n_basis, order=order)
smoother = BasisSmoother(basis=basis, return_basis=True, method='svd')

registration = ElasticRegistration()
ID = IntegratedDepth()
MBD = ModifiedBandDepth()

scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
l_enc = LabelEncoder()


def get_model():
	return {
		'svm': SVC(class_weight='balanced', probability=True),
		'rf': RandomForestClassifier(class_weight='balanced', n_jobs=4),
		'xgb': xgb.XGBClassifier(tree_method='gpu_hist', eval_metric='logloss', use_label_encoder=False),
	}[MODEL]


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
	return fd_smooth.data_matrix[:, t_cut, 0]


def specificity(y_true, y_pred, zero_division=0):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	if tn+fp == 0 and zero_division:
		return zero_division
	return tn / (tn+fp)


def kfolds(dataset, nfolds=5):
	class_ratios = (
		dataset
			.query('label')
			.patient_id
			.value_counts()
			.reset_index()
			.rename(
				{'patient_id': 'cancerous_regions', 'index': 'patient_id'},
				axis='columns',
			)
			.merge(
				dataset
					.patient_id
					.value_counts()
					.reset_index()
					.rename({'patient_id': 'n', 'index': 'patient_id'}, axis='columns')
			)
		.assign(patient_ratio=lambda x: x.cancerous_regions / x.n)
		.sort_values('patient_ratio', ascending=False)
		.assign(
			fold_id=lambda x:
				(
					(list(range(0, nfolds)) + list(reversed(range(0, nfolds)))) *
					math.ceil(x.shape[0] / (nfolds * 2))
				)
				[:x.shape[0]]
		)
	)
	return class_ratios.groupby('fold_id').patient_id.apply(list).to_numpy()


def train_kfolds(Y, X, folds, segmentation_id):
	ax = None
	roc_scores = []

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

	patients = pd.DataFrame(X)[13]
	X = pd.DataFrame(
		np.vstack(
            pd.DataFrame(X).groupby([13]).apply(scaler.fit_transform).to_numpy()
        )[:,0:-1]
	).assign(Y=Y, patient=patients)

	for i, test_patients in enumerate(folds):
		print(i)
		X_train = X[X.patient.apply(lambda x: x not in test_patients)]
		X_test = X[X.patient.apply(lambda x: x in test_patients)]

		y_train = X_train.Y
		y_test = X_test.Y
		X_train = X_train.drop(['Y', 'patient'], axis='columns')
		X_test = X_test.drop(['Y', 'patient'], axis='columns')

		model = get_model()
		model.fit(X_train, y_train)
		preds = model.predict_proba(X_test)[:,1]

		fpr, tpr, thresholds = roc_curve(y_test, preds)
		roc_scores = pd.DataFrame(
			{
				'fpr': fpr,
				'tpr': tpr,
				'threshold': thresholds,
				'fold': i,
				'segmentation_id': segmentation_id,
			}
		)
		if not os.path.exists(f'./roc/{MODEL}.csv'):
			roc_scores.to_csv(f'./roc/{MODEL}.csv', index=False)
		else:
			roc_scores.to_csv(f'./roc/{MODEL}.csv', index=False, mode='a', header=False)
		ax = sns.scatterplot(x=fpr, y=tpr, ax=ax)
		preds = np.reshape(preds >= 0.5, (-1)).astype(int)

		f1s += [f1_score(y_test, preds, zero_division=0)]
		precisions += [precision_score(y_test, preds, zero_division=0)]
		recalls += [recall_score(y_test, preds, zero_division=0)]
		accuracies += [balanced_accuracy_score(y_test, preds)]
		specificities += [specificity(y_test, preds)]

		preds = model.predict(X_train)
		preds = np.reshape(preds >= 0.5, (-1)).astype(int)
		train_f1s += [f1_score(y_train, preds, zero_division=0)]
		train_precisions += [precision_score(y_train, preds, zero_division=0)]
		train_recalls += [recall_score(y_train, preds, zero_division=0)]
		train_accuracies += [balanced_accuracy_score(y_train, preds)]
		train_specificities += [specificity(y_train, preds)]

	ax.set_ylabel('False positive rate')
	ax.set_xlabel('True positive rate')
	plt.legend(labels=['TPR x FPR', 'threshold'] * 5)
	plt.savefig(f'./roc/{MODEL}_segmentation-{segmentation_id}.png')
	return (
        segmentation_id,
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
    parser.add_argument('--model', type=str, default=MODEL)

    args = parser.parse_args()
    
    MODEL = args.model
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
    dataset = dataset[dataset.patient_id.apply(lambda x: x not in [3, 6, 7, 8, 10, 11, 14, 19, 27, 29, 37, 60, 66, 76, 86, 108, 119, 122, 123, 128, 133, 140])]
    # dataset = dataset[dataset.patient_id.apply(lambda x: x not in [2, 32, 35, 40, 41, 45, 52, 92, 107, 110, 114, 116, 139])]
    dataset = dataset.merge(dataset.query('label == True').patient_id.drop_duplicates())
    # print('using patients:')
    # print(dataset.patient_id.drop_duplicates())
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
    ids = np.vstack([ID(fd_smooth).reshape(-1, 1) for fd_smooth in bsplined])
    mbds = np.vstack([MBD(fd_smooth).reshape(-1, 1) for fd_smooth in bsplined])
    print('getting other features ...')
    max_values = np.vstack(
        [fd_smooth.data_matrix.max(axis=1).reshape(-1, 1) for fd_smooth in bsplined]
    )
    descret_points = np.vstack([get_descrete_points(fd_smooth) for fd_smooth in bsplined])
    patients = dataset.patient_id.to_numpy().reshape(-1, 1)
    slices = dataset.slice_id.to_numpy().reshape(-1, 1)
    
    train_set = np.hstack([ids, mbds, max_values, descret_points, patients])
    labels = l_enc.fit_transform(dataset.label.astype(int).tolist())
    if MODEL == 'xgb':
        labels = dataset.label.astype(int).to_numpy()

    pd.DataFrame(
        [train_kfolds(labels, train_set, kfolds(dataset, 5), segmentation_id)],
        columns=[
            'segmentation', 'F1', 'precision', 'recall', 'accuracy', 'specificity',
            'train_F1', 'train_precision', 'train_recall', 'train_accuracy', 'train_specificity',
        ]
    ).to_csv(f'./{MODEL}_metrics_v3.csv', index=False, mode='a', header=False)
