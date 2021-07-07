from araucaria import Group
from araucaria.testdata import get_testpath
from araucaria.io import read_xmu
from araucaria.stats import rebin
from araucaria.utils import check_objattrs
fpath = get_testpath('xmu_testfile.xmu')
# extracting mu and mu_ref scans
group_mu = read_xmu(fpath, scan='mu')
bins    = 600             # number of bins
regroup = group_mu.copy() # rebinning copy of group
rebin   = rebin(regroup, bins=bins, update=True)
attrs   = ['energy', 'mu', 'mu_ref', 'rebin_stats']
check_objattrs(regroup, Group, attrs)
# [True, True, True, True]

# plotting rebinned spectrum
from araucaria.plot import fig_xas_template
import matplotlib.pyplot as plt
figpars = {'e_range' : (11850, 11900)}   # energy range
fig, ax = fig_xas_template(panels='x', fig_pars=figpars)
stdev = regroup.rebin_stats['mu_std']    # std of rebinned mu
line  = ax.plot(group_mu.energy, group_mu.mu, label='original')
line  = ax.errorbar(regroup.energy, regroup.mu, yerr=stdev, marker='o',
                    capsize=3.0, label='rebinned')
leg   = ax.legend(edgecolor='k')
lab   = ax.set_ylabel('abs [a.u]')
plt.show(block=False)
