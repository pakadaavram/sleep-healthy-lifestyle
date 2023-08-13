import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
print(df.head())
print(df.describe())
print(df.columns.values)
column_names = ['Person ID','Gender','Age','Occupation','Sleep Duration','Quality of Sleep','Physical Activity Level','Stress Level','BMI Category','Blood Pressure','Heart Rate','Daily Steps','Sleep Disorder']
df.columns = column_names

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
x = df[['Age','Sleep Duration','Physical Activity Level','Stress Level']] #המידע שלפיו נחזה את ערך ה-Y (איכות השינה)
print(x.head())
y = df[['Quality of Sleep']]
print(y.head())

#split the data to testing data and training data for assess the modele's accuaracy on data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test) #array with values with predictions for y of x_test
sns.regplot(x=y_test,y=y_pred)
plt.show()

#Calculate the RMSE and the smaller it is, the more accurate our model is
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(rmse)

#We will see how important each characteristic was in relation to the objective function
print(reg.coef_)
df1 = pd.DataFrame(['Age','Sleep Duration','Physical Activity Level','Stress Level'], reg.coef_.flatten(),columns=["feature"])
print(df1)