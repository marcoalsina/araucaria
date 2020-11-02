from araucaria.testdata import get_testpath
from araucaria import Group
from araucaria.io import read_dnd
from araucaria.xas import pre_edge, autobk, xftf
from araucaria.utils import check_objattrs
kw      = 2
k_range = [2,10]
fpath   = get_testpath('dnd_testfile.dat')
group   = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
pre     = pre_edge(group, update=True)
autbk   = autobk(group, update=True)
fft     = xftf(group, k_range=k_range, kweight=kw, update=True)
attrs = ['kwin', 'r', 'chir', 'chir_mag', 'chir_re', 'chir_im']
check_objattrs(group, Group, attrs)
# [True, True, True, True, True, True]

# plotting forward FFT signal
import matplotlib.pyplot as plt
from araucaria.plot import fig_xas_template
fig, ax = fig_xas_template(panels='er', pars={'kweight':kw})
line = ax[0].plot(group.k, group.k**kw*group.chi)
line = ax[0].plot(group.k, group.kwin, color='firebrick')
xlim = ax[0].set_xlim(0,12)
line = ax[1].plot(group.r, group.chir_mag)
xlim = ax[1].set_xlim(0,6)
fig.tight_layout()
plt.show(block=False)
