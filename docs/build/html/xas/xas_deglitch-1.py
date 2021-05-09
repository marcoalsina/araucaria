from numpy import allclose
from araucaria.testdata import get_testpath
from araucaria import Group
from araucaria.io import read_dnd
from araucaria.xas import deglitch, pre_edge, autobk
from araucaria.utils import check_objattrs
fpath  = get_testpath('dnd_glitchfile.dat')
group  = read_dnd(fpath, scan='fluo')  # extracting fluo and mu_ref scans
cgroup = group.copy()
degli  = deglitch(cgroup, update=True)
attrs  = ['index_glitches', 'energy_glitches', 'deglitch_pars']
check_objattrs(cgroup, Group, attrs)
# [True, True, True]
allclose(cgroup.energy_glitches, group.energy[cgroup.index_glitches])
# True
print(cgroup.energy_glitches)
# [7552.2789 7548.1747 7390.512  7387.2613]

# plotting original and deglitched spectrum
from araucaria.plot import fig_xas_template
import matplotlib.pyplot as plt
for g in [group, cgroup]:
    pre   = pre_edge(g, update=True)
    autbk = autobk(g, update=True)
fig, ax = fig_xas_template(panels='xe')
line = ax[0].plot(group.energy,  group.norm,  label='original', color='tab:red')
line = ax[0].plot(cgroup.energy, cgroup.norm, label ='degliched', color='k')
line = ax[1].plot(group.k, group.k**2 * group.chi, color='tab:red')
line = ax[1].plot(cgroup.k, cgroup.k**2 * cgroup.chi, color='k')
leg  = ax[0].legend()
fig.tight_layout()
plt.show(block=False)
