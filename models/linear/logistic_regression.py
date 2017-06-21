
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


DATA_DIR = os.path.realpath("../../data/")

WM_DATA_FILE = os.path.join(DATA_DIR, "watermelon3_0_En.csv")
WM_DATA = pd.read_csv(WM_DATA_FILE).set_index('id')
WM_LABELS = ['label']
WM_FEATURES = [_ for _ in WM_DATA.columns if _.lower() not in WM_LABELS]
WM_TRAIN_RATIO = 0.9


def digitize(f, inplace=False, start_from=1):
    ff = f if inplace else f.copy(deep=True)
    m = dict()
    for c in ff.columns:
        if ff[c].dtype.name == 'object':
            names = list(ff[c].unique())
            m[c] = {name: start_from + names.index(name) for name in names}
            for n, i in m[c].items():
                ff.loc[ff[c] == n, c] = i
    return ff, m


lr = LogisticRegression()
data, mapped = digitize(WM_DATA)

# sampling
train_rows = []
while len(train_rows) != int(len(data) * WM_TRAIN_RATIO):
    ix = data.iloc[np.random.randint(0, len(data))].name
    if ix not in train_rows:
        train_rows.append(ix)

test_rows = list(set(data.index) - set(train_rows))

lr.fit(
    data.loc[train_rows, WM_FEATURES].as_matrix(),
    data.loc[train_rows, WM_LABELS].as_matrix()
)

print(lr.predict_proba(
    data.loc[test_rows, WM_FEATURES].as_matrix()
))

print(lr.score(
    data.loc[test_rows, WM_FEATURES].as_matrix(),
    data.loc[test_rows, WM_LABELS].as_matrix()
))

print(data.loc[test_rows, WM_LABELS])

pass
