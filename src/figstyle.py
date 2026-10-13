import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager
fam = 'Liberation Sans' if any('Liberation Sans' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': fam, 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'figure.dpi': 100, 'savefig.dpi': 300,
                     'savefig.bbox': 'tight', 'pdf.fonttype': 42})
C = dict(phys='#E69F00', ml='#0072B2', hyb='#009E73', fault='#D55E00', l2='#56B4E9', grey='#7F7F7F', purple='#CC79A7', black='#222222')
W1, W2 = 3.54, 7.48   # 90 mm and 190 mm
def save(fig, name):
    import os
    FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
    fig.savefig(os.path.join(FIG, name + '.png')); fig.savefig(os.path.join(FIG, name + '.pdf')); plt.close(fig)
def panel(ax, s): ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
