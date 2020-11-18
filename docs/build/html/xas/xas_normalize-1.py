from araucaria.testdata import get_testpath
from araucaria import Group
from araucaria.io import read_dnd
from araucaria.xas import pre_edge
from araucaria.utils import check_objattrs
fpath    = get_testpath('dnd_testfile.dat')
group = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
attrs = ['e0', 'edge_step', 'pre_edge', 'post_edge', 'norm', 'flat']
pre    = pre_edge(group, update=True)
check_objattrs(group, Group, attrs)
# [True, True, True, True, True, True]

# plotting original and normalized spectrum
import matplotlib.pyplot as plt
from araucaria.plot import fig_xas_template
fig, ax = fig_xas_template(panels='xx')
line = ax[0].plot(group.energy, group.pre_edge,
                  color='gray', ls='--', label='pre-edge')
line = ax[0].plot(group.energy, group.post_edge,
                  color='gray', ls=':', label='post-edge')
line = ax[0].plot(group.energy, group.mu, label='mu')
text = ax[0].set_ylabel('Absorbance')
leg  = ax[0].legend()
line = ax[1].plot(group.energy, group.norm, label='norm')
line = ax[1].plot(group.energy, group.flat, label='flat')
line = ax[1].axhline(0, color='gray', ls=':')
line = ax[1].axhline(1, color='gray', ls=':')
leg  = ax[1].legend()
fig.tight_layout()
plt.show(block=False)
