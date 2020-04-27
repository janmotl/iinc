import openml
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import iinc


# Measures
def brier_multi(targets, probs):
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ohe = enc.fit_transform(targets.reshape(-1, 1))
    return np.mean(np.sum((ohe - probs)**2, axis=1))

# Dataset list
openml_list = openml.datasets.list_datasets()
datalist = pd.DataFrame.from_dict(openml_list, orient='index')
filtered = datalist.query('NumberOfClasses > 2')
filtered = filtered.query('NumberOfInstances <= 3000')
filtered = filtered.query('NumberOfFeatures <= 30')
filtered = filtered.query('format == "ARFF"') # Avoid sparse data sets
filtered = filtered.query('NumberOfSymbolicFeatures <= 1') # The label is included in the count
filtered = filtered.query('MajorityClassSize/MinorityClassSize > 2')
filtered = filtered.query('did not in [1528, 1529, 1530, 1543, 1544, 1545, 1546]') # Almost duplicates


logger = []
for did in filtered.did:
    try:
        # Download dataset
        dataset = openml.datasets.get_dataset(did)
        X_in, y_in, categorical_indicator, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='array')
        X_in[np.isnan(X_in)] = -1  # Very simple missing value treatment
        print('Dataset', dataset.name, did, flush=True) # For progress indication

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size = 0.33, random_state = 42, stratify=y_in)

        # Score
        clf = KNeighborsClassifier(n_neighbors=3)
        probs =  clf.fit(X_train, y_train).predict_proba(X_test)
        prediction = clf.fit(X_train, y_train).predict(X_test)
        baseline_auc = metrics.roc_auc_score(y_test, probs, multi_class='ovo')
        baseline_kappa = metrics.cohen_kappa_score(y_test, prediction)
        baseline_brier = brier_multi(y_test, probs)

        probs, prediction = sandbox_iinc.iinc(X_train, y_train, X_test, prior_weight='raw')
        iinc_raw_auc = metrics.roc_auc_score(y_test, probs, multi_class='ovo')
        iinc_raw_kappa = metrics.cohen_kappa_score(y_test, prediction)
        iinc_raw_brier = brier_multi(y_test, probs)

        probs, prediction = sandbox_iinc.iinc(X_train, y_train, X_test, prior_weight='ovo')
        iinc_ovo_auc = metrics.roc_auc_score(y_test, probs, multi_class='ovo')
        iinc_ovo_kappa = metrics.cohen_kappa_score(y_test, prediction)
        iinc_ovo_brier = brier_multi(y_test, probs)

        probs, prediction = sandbox_iinc.iinc(X_train, y_train, X_test, prior_weight='ovr')
        iinc_ovr_auc = metrics.roc_auc_score(y_test, probs, multi_class='ovo')
        iinc_ovr_kappa = metrics.cohen_kappa_score(y_test, prediction)
        iinc_ovr_brier = brier_multi(y_test, probs)

        logger.append([dataset.name, did, dataset.qualities.get('MajorityClassSize') / dataset.qualities.get('MinorityClassSize'), baseline_kappa, iinc_raw_kappa, iinc_ovo_kappa, iinc_ovr_kappa, baseline_auc, iinc_raw_auc, iinc_ovo_auc, iinc_ovr_auc, baseline_brier, iinc_raw_brier, iinc_ovo_brier, iinc_ovr_brier])
    except Exception:
        continue

result = pd.DataFrame(logger, columns=['dataset', 'did', 'class_ratio', 'baseline_kappa', 'raw_kappa', 'ovo_kappa', 'ovr_kappa', 'baseline_auc', 'raw_auc', 'ovo_auc', 'ovr_auc', 'baseline_brier', 'raw_brier', 'ovo_brier', 'ovr_brier'])
result.to_csv('~/Downloads/results.csv')

pd.set_option('display.width', 1600)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(result)
