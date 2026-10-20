import pandas as pd, numpy as np
from scipy import stats
f=pd.read_csv('results/cv_folds.csv'); f=f[f.rep<5]
print(f.groupby('model').size().to_dict())
ref=f[f.model=='AG-StackNet'].set_index(['rep','fold'])
ntr,nte=ref.n_tr.mean(),ref.n_te.mean()
for m,g in f.groupby('model',sort=False):
    g=g.set_index(['rep','fold']).loc[ref.index]; x=g.RMSE.values; J=len(x)
    h=stats.t.ppf(.975,J-1)*x.std(ddof=1)/np.sqrt(J)
    row=f"{m:22s} RMSE {x.mean():.2f}+-{x.std(ddof=1):.2f} CI[{x.mean()-h:.2f},{x.mean()+h:.2f}] R2 {g.R2.mean():.2f}+-{g.R2.std(ddof=1):.2f} MAE {g.MAE.mean():.2f} MAPE {g.MAPE.mean():.2f}"
    if m!='AG-StackNet':
        dd=ref.RMSE.values-x; se=np.sqrt((1/J+nte/ntr)*dd.var(ddof=1)); t=dd.mean()/se; pt=2*stats.t.sf(abs(t),J-1); hh=stats.t.ppf(.975,J-1)*se
        w=stats.wilcoxon(ref.RMSE.values,x); dz=(x-ref.RMSE.values).mean()/(x-ref.RMSE.values).std(ddof=1)
        row+=f" | d={dd.mean():.2f} CI[{dd.mean()-hh:.2f},{dd.mean()+hh:.2f}] p_corr={pt:.3g} W={w.statistic:.0f} p_w={w.pvalue:.3g} dz={dz:.2f} wins={(ref.RMSE.values<x).sum()}/{J}"
    print(row)
