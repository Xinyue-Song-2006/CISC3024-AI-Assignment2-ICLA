"""Generate report figures from saved predictions; no retraining."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import roc_curve, roc_auc_score
from experiment import METHODS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial', 'font.size':11, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.linewidth':1.1,
                     'legend.frameon':False, 'svg.fonttype':'none'})
COLORS = ['#4D4D4D','#42949E','#B64342','#0F4D92']
STYLES = ['-','--','-.',':']

def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

digits = load_digits()
splits = json.loads((ROOT/'results/splits.json').read_text())
test = np.array(splits['ten_class_0']['test'])
x = digits.data[test]/16
noisy = np.clip(x + np.random.default_rng(10000).normal(0,.25,x.shape),0,1)
fig, axes = plt.subplots(2, 10, figsize=(9.5, 2.5))
for digit in range(10):
    idx = np.flatnonzero(digits.target[test]==digit)[0]
    for row, data in enumerate([x,noisy]):
        axes[row,digit].imshow(data[idx].reshape(8,8),cmap='gray_r',vmin=0,vmax=1)
        axes[row,digit].set_xticks([]); axes[row,digit].set_yticks([])
        for spine in axes[row,digit].spines.values(): spine.set_visible(False)
    axes[0,digit].set_title(str(digit))
axes[0,0].set_ylabel('Clean'); axes[1,0].set_ylabel('Noise')
fig.tight_layout(pad=.7)
save(fig,'digit_examples')

data = np.load(ROOT/'results/predictions.npz')
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
for method, color, style in zip(METHODS,COLORS,STYLES):
    p = data[f'open_set_0_clean_{method}']; po = data[f'open_set_0_ood_{method}']
    labels=np.r_[np.zeros(len(p)),np.ones(len(po))]
    scores=np.r_[1-p.max(axis=1),1-po.max(axis=1)]
    fpr,tpr,_ = roc_curve(labels,scores)
    auc=roc_auc_score(labels,scores)
    axes[0].plot(fpr,tpr,color=color,ls=style,label=f'{method}: {auc:.3f}',lw=1.8)
    prob = data[f'ten_class_0_clean_{method}']
    correct = prob.argmax(axis=1)==data['ten_class_0_clean_labels']
    conf = prob.max(axis=1)
    bins=np.minimum((10*conf).astype(int),9)
    xs=[]; ys=[]
    for b in range(10):
        mask=bins==b
        if mask.sum()>=5:
            xs.append(conf[mask].mean()); ys.append(correct[mask].mean())
    axes[1].plot(xs,ys,marker='o',color=color,ls=style,label=method,lw=1.6,ms=4)
axes[0].plot([0,1],[0,1],color='.7',ls='--',lw=1)
axes[0].set(xlabel='False positive rate (ID rejected)',ylabel='True positive rate (OOD detected)',
            title='(a) Open-set ROC, seed 0',xlim=(0,1),ylim=(0,1.02))
axes[0].legend(fontsize=8,loc='lower right')
axes[1].plot([0,1],[0,1],color='.7',ls='--',lw=1)
axes[1].set(xlabel='Mean confidence',ylabel='Observed accuracy',
            title='(b) Clean-digit reliability, seed 0',xlim=(0,1),ylim=(0,1.05))
axes[1].legend(fontsize=8,loc='lower right')
fig.tight_layout(pad=1)
save(fig,'uncertainty')

rows = list(csv.DictReader((ROOT/'results/validation_grid.csv').open()))
fig, axes = plt.subplots(1,2,figsize=(9.4,3.1))
metadata = json.loads((ROOT/'results/metadata.json').read_text())
for ax,task,title in zip(axes,['ten_class','open_set'],['(a) Ten-class validation','(b) Five-class validation']):
    for seed in range(5):
        subset=[r for r in rows if r['task']==task and int(r['seed'])==seed]
        ax.semilogx([float(r['precision']) for r in subset], [float(r['validation_nll']) for r in subset],
                    lw=1.4,label=f'Seed {seed}')
    ax.set(xlabel='ICLA precision (lambda)',ylabel='Validation NLL',title=title)
    ax.legend(fontsize=7,ncol=2)
fig.tight_layout(pad=1)
save(fig,'precision_sensitivity')
print('Created 3 figures, PNG and SVG.')
