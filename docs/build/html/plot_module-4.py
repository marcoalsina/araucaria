import matplotlib.pyplot as plt
from araucaria.testdata import get_testpath
from araucaria.io import read_dnd
from araucaria.xas import pre_edge, autobk
from araucaria.plot import fig_autobk
fpath   = get_testpath('dnd_testfile.dat')
group   = read_dnd(fpath, scan='mu')
pre     = pre_edge(group, update=True)
bkg     = autobk(group, update=True)
fig, ax = fig_autobk(group)
plt.show(block=False)
