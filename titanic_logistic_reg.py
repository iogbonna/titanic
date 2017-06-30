from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import pandas
from pandas.tools.plotting import scatter_matrix

# S=1
# C=2
# Q=3
#
# male=1
# female=2

#print(boston.target)
#print(boston.DESCR)

dt=pandas.read_csv("train.csv",header=0)
dt.fillna(0)

scatter_matrix(dt)
#plt.show()

df=pandas.read_csv("dev_test.csv",header=0)
df.fillna(0)

df_actual=pandas.read_csv("test.csv",header=0)
#train_x=dt.ix[:,5:9]
#train_y=dt.ix[:,1]
#print(np.array(train_x))

#plt.plot(train_x,train_y,'.k')
#plt.show()
X_scaler=StandardScaler()
Y_scaler=StandardScaler()


linear_model=LogisticRegression(C=100.0,random_state=0)

quadratic_featurizer=PolynomialFeatures(degree=2)

X_train=dt[[2,4,5,7,11,10]].as_matrix()
X_quadratic_train=quadratic_featurizer.fit_transform(X_train)

X_test=df[[2,4,5,7,11,10]].as_matrix()
X_quadratic_test=quadratic_featurizer.fit_transform(X_test)

X_actual_test=df_actual[[1,3,4,6,10,9]].as_matrix()
X_actual_quadratic_test=quadratic_featurizer.fit_transform(X_actual_test)



linear_model.fit(X_quadratic_train,dt.iloc[:,1].to_frame())
print(linear_model)
l_predicted=linear_model.predict(X_quadratic_test)
l_expected=df.iloc[:,1].to_frame()

print(metrics.classification_report(l_expected,l_predicted))
print(metrics.confusion_matrix(l_expected,l_predicted))
print(metrics.accuracy_score(l_expected,l_predicted))

print('Logistic regression %s' % linear_model.score(X_quadratic_test,df.iloc[:,1].to_frame()))
#linear_model.predict(df[[2,4,5,7]])
ids=df_actual["PassengerId"]
actual_predictions=linear_model.predict(X_actual_quadratic_test)
output=pandas.DataFrame({'PassengerId':ids,'Survived':actual_predictions})
output.to_csv('titanic-predictions.csv', index = False)
output.head()
