import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
d=pd.read_csv('data/WheatDataset.csv').drop_duplicates().dropna()
d['Temperature_Difference']=d.Max_Temperature-d.Min_Temperature
F=['Precipitation','Min_Temperature','Max_Temperature','Mean_Temperature','Temperature_Difference','RH','Wind_Speed','DVI','NDVI','EC','pH','Organic_Matter','Available_Phosphorus','Available_Potassium','Saturation']
X,y=d[F],d.Yield
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
nns=[MLPRegressor(hidden_layer_sizes=(64,32),max_iter=2000,random_state=s,learning_rate_init=0.01).fit(Xtr,ytr) for s in range(3)]
Pte=np.column_stack([m.predict(Xte) for m in nns]); Ptr=np.column_stack([m.predict(Xtr) for m in nns])
def rep(name,p):
    print(f"{name:55s} RMSE={np.sqrt(mean_squared_error(yte,p)):7.2f} MAE={mean_absolute_error(yte,p):7.2f} R2={r2_score(yte,p):.2f}")
rep("Published protocol: RF meta fitted on TEST set", RandomForestRegressor(n_estimators=100,random_state=42).fit(Pte,yte).predict(Pte))
rep("Correct protocol: RF meta fitted on TRAIN preds", RandomForestRegressor(n_estimators=100,random_state=42).fit(Ptr,ytr).predict(Pte))
rep("Predict training mean", np.full(len(yte),ytr.mean()))
