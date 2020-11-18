import matplotlib.pyplot as plt
from araucaria import Collection
from araucaria.testdata import get_testpath
from araucaria.io import read_dnd
from araucaria.xas import merge
from araucaria.plot import fig_merge
collection = Collection()
files = ['dnd_testfile.dat' , 'dnd_testfile2.dat', 'dnd_testfile3.dat']
for file in files:
    fpath = get_testpath(file)
    group_mu = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
    collection.add_group(group_mu)         # adding group to collection
report, merge = merge(collection)
fig, ax = fig_merge(merge, collection)
plt.show(block=False)
