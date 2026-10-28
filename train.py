import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df=pd.read_csv('Iris - Iris.csv')
df=df.drop("Id",axis=1)
X=df.drop("Species",axis=1)
y=df["Species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
pred=model.predict(X_test)
accuracy=accuracy_score(pred,y_test)
print("Accuracy score",accuracy)
print(pred)

joblib.dump(model,"model.pkl")
print("model saved")