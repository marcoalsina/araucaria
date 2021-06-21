import matplotlib.pyplot as plt
from araucaria.testdata import get_testpath
from araucaria import Dataset
from araucaria.io import read_collection_hdf5
from araucaria.xas import pre_edge
from araucaria.stats import pca, target_transform
from araucaria.plot import fig_target_transform
fpath      = get_testpath('Fe_database.h5')
collection = read_collection_hdf5(fpath)
collection.apply(pre_edge)
model      = pca(collection, pca_region='xanes', ncomps=3,
                 pca_range=[7050,7300])
data       = target_transform(model, collection)
fig, axes  = fig_target_transform(data, model)
legend     = axes[0,0].legend(loc='upper right', edgecolor='k')
fig.tight_layout()
plt.show(block=False)
