import matplotlib.pyplot as plt
from araucaria.testdata import get_testpath
from araucaria.stats import cluster
from araucaria.io import read_collection_hdf5
from araucaria.plot import fig_cluster
fpath      = get_testpath('Fe_database.h5')
collection = read_collection_hdf5(fpath)
datgroup   = cluster(collection, cluster_region='xanes', pre_edge_kws={})
fig, ax    = fig_cluster(datgroup)
fig.tight_layout()
plt.show(block=False)
