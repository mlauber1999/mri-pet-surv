import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2]]*10)
y = np.array([0]*5 + [1]*5)
skf = StratifiedKFold(n_splits=5)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=120)
# skf.get_n_splits(X, y)


print(skf)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
