import matplotlib.pyplot as plt
from araucaria.testdata import get_testpath
from araucaria.xas import pre_edge
from araucaria.stats import pca
from araucaria.io import read_collection_hdf5
from araucaria.plot import fig_pca
fpath      = get_testpath('Fe_database.h5')
collection = read_collection_hdf5(fpath)
collection.apply(pre_edge)
model      = pca(collection, pca_region='xanes',
                 pca_range=[7050, 7300])
fig, axes = fig_pca(model, fontsize=8)
fig.tight_layout()
plt.show(block=False)
