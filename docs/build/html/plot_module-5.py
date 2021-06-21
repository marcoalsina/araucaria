import matplotlib.pyplot as plt
from araucaria.testdata import get_testpath
from araucaria.xas import pre_edge
from araucaria.stats import cluster
from araucaria.io import read_collection_hdf5
from araucaria.plot import fig_cluster
fpath      = get_testpath('Fe_database.h5')
collection = read_collection_hdf5(fpath)
collection.apply(pre_edge)
datgroup   = cluster(collection, cluster_region='xanes')
fig, ax    = fig_cluster(datgroup)
fig.tight_layout()
plt.show(block=False)
