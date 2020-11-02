from araucaria.testdata import get_testpath
from araucaria import Group
from araucaria.io import read_dnd
from araucaria.xas import pre_edge, autobk
from araucaria.utils import check_objattrs
fpath    = get_testpath('dnd_testfile.dat')
group    = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
pre      = pre_edge(group, update=True)
attrs    = ['e0', 'edge_step', 'bkg', 'chie', 'chi', 'k']
autbk    = autobk(group, update=True)
check_objattrs(group, Group, attrs)
# [True, True, True, True, True, True]

# plotting original and background spectrum
import matplotlib.pyplot as plt
from araucaria.plot import fig_xas_template
fig, ax = fig_xas_template(panels='xe')
line = ax[0].plot(group.energy, group.mu, label='mu')
line = ax[0].plot(group.energy, group.bkg, label='bkg', zorder=-1)
text = ax[0].set_ylabel('Absorbance')
leg  = ax[0].legend()
line = ax[1].plot(group.k, group.k**2 * group.chi, label='k^2 chi')
leg  = ax[1].legend()
fig.tight_layout()
plt.show(block=False)
