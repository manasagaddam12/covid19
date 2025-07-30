import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
import joblib
df=pd.read_csv("C:\\Users\\Manasa\\Documents\\covid19\\pneumonia_covid_diagnosis_dataset.csv")
columns=["Gender","Fever","Cough","Fatigue","Breathlessness","Comorbidity","Type","Stage"]
# to convert str to num..every column.
for col in columns:
    le=LabelEncoder()
    # updates previous existing value.
    df[col]=le.fit_transform(df[col])
df=df.drop("Is_Curable",axis=1)
# for removing survival_rate to get all inputs
x=df.drop(columns=["Survival_Rate"],axis=1)
# this is the output
y=df["Survival_Rate"]
# to split and train.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestRegressor()
# to fit
model.fit(x_train,y_train)
# predict
prd=model.predict(x_test)
# save model
joblib.dump(model,"covid_diag.pkl")
