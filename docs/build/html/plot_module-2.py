import matplotlib.pyplot as plt
from araucaria.testdata import get_testpath
from araucaria.io import read_dnd
from araucaria.xas import pre_edge
from araucaria.plot import fig_pre_edge
fpath   = get_testpath('dnd_testfile.dat')
group   = read_dnd(fpath, scan='mu')
pre     = pre_edge(group, update=True)
fig, ax = fig_pre_edge(group)
plt.show(block=False)
