# coding=utf-8

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from treelib import Ardennes

h = .02  # step size in the mesh

# X, y = make_classification(
#     n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
# )

my_dataset = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=np.float32)

X, y = my_dataset[:, :-1], my_dataset[:, -1]

datasets = [
    my_dataset
]

figure = plt.figure(figsize=(14, 9))
i = 1

# classifiers = [('best individual', False), ('last population', True)]
classifiers = [('best individual', False)]
# classifiers = [('last population', True)]

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# iterate over datasets
for ds_cnt, ds in enumerate(datasets):

    X = StandardScaler().fit_transform(X)

    X_train, y_train = ds[:, :-1], ds[:, -1]
    X_test, y_test = ds[:, :-1], ds[:, -1]

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    clf = Ardennes(
        n_individuals=10, n_iterations=10,
        initial_tree_size=7, distribution='univariate',
        class_probability='decreased'
    )
    clf.fit(train=(X_train, y_train), verbose=True)
    print '-- training complete --'

    # iterate over classifiers
    for name, ensemble in classifiers:
        ax = plt.subplot(len(datasets), len(classifiers), i)

        score = clf.validate(X_test=X_test, y_test=y_test, ensemble=ensemble)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if not ensemble:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()], ensemble=ensemble)
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()], ensemble=ensemble)[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, s=45, linewidth=0., label='train')
        # testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, s=45, linewidth=1., label='test')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

        ax.legend(loc=3)

        i += 1

plt.tight_layout()
plt.show()
