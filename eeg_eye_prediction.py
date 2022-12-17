from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split


plt.close('all')

##

df = pd.read_excel(r'/Users/charalamposp/Documents/MATLAB/eyepca.xlsx')
print(df)

df.head()

nndata=np.array(df)
X=nndata

plt.close('all')

X_train, X_test, y_train, y_test = train_test_split(X, nndata[:,3], test_size=0.8, random_state=4)
clf = RandomForestClassifier(n_estimators = 2, random_state = 30,max_features=2)
clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)  # test the output by changing values

diff=abs(Y_pred-y_test)

print('acc = ',100*(y_test.shape[0]-np.count_nonzero(diff))/y_test.shape[0],' %')


fig = plt.figure()
ax = plt.axes(projection='3d')
sc=ax.scatter(X_test[:,0],X_test[:,1],X_test[:,2],'*b',c=diff)
fig.colorbar(sc, ax=ax)
