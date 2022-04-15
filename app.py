from sklearn.preprocessing import MinMaxScaler

import numpy as np
import datetime
import streamlit as st
import pandas as pd
import pickle
import joblib
import boto3
# Header 
st.write('''
# *Loyalty program* 
##### Sébastien Lozano Forero 
''')



Edad = st.slider('Edad', 0, 130, 25)
Patrimonio = st.number_input('Patrimonio')
Ingresos_Mensuales = st.number_input('Ingresos Mensuales')
No_hijos = st.number_input('No_hijos')
fecha_ult_desembolso = st.date_input("Fecha último desembolso (yyyy/mm/dd)", datetime.date(2019, 7, 6))
monto_credito = st.number_input('Monto del credito')
tasa = st.number_input('tasa')
saldo_capital = st.number_input('saldo capital')
saldo_Ahorro = st.number_input('saldo ahorro')
Antiguedad_en_meses = st.number_input('Antiguedad en meses')
Max_dias_mora = st.number_input('Máximo días en mora')
plazo_dias = st.number_input('plazo días')
mes_ult_desembolso = fecha_ult_desembolso.month

ciudad = st.selectbox(
     'Ciudad',
     ('Barranquilla', 'Bogotá', 'Cali','Cartagena'))

C_Barranquilla = 1 if ciudad == 'Barranquilla' else 0
C_Bogotá = 1 if ciudad == 'Bogotá' else 0
C_Cali= 1 if ciudad == 'Cali' else 0
C_Cartagena= 1 if ciudad == 'Cartagena' else 0

test = {'Edad':[Edad],
        'Patrimonio':[Patrimonio],
        'Ingresos_Mensuales':[Ingresos_Mensuales],
        'No_hijos':[No_hijos],
        'fecha_ult_desembolso':[fecha_ult_desembolso],
        'monto_credito':[monto_credito],
        'tasa':[tasa],
        'saldo_capital':[saldo_capital],
        'saldo_Ahorro':[saldo_Ahorro],
        'Antiguedad_en_meses':[Antiguedad_en_meses],
        'Max_dias_mora':[Max_dias_mora],
        'plazo_dias':[plazo_dias],        
        'mes_ult_desembolso':[mes_ult_desembolso],
        'C_Barranquilla' : [C_Barranquilla],
        'C_Bogotá' : [C_Bogotá],
        'C_Cali' : [C_Cali],
        'C_Cartagena' : [C_Cartagena]       
        }

nueva = pd.DataFrame.from_dict(test,orient='columns')


# This section conducts the test using the original dataset
# nueva['mes_ult_desembolso'] = nueva['fecha_ult_desembolso'].dt.month
# nueva


mms = joblib.load('parameter/Patrimonio.sav')
nueva['Patrimonio'] = mms.transform( nueva[['Patrimonio']].values )

mms = joblib.load('parameter/Ingresos_Mensuales.sav')
nueva['Ingresos_Mensuales'] = mms.transform( nueva[['Ingresos_Mensuales']].values )

mms = joblib.load('parameter/No_hijos.sav')
nueva['No_hijos'] = mms.transform( nueva[['No_hijos']].values )

mms = joblib.load('parameter/monto_credito.sav')
nueva['monto_credito'] = mms.transform( nueva[['monto_credito']].values )

mms = joblib.load('parameter/tasa.sav')
nueva['tasa'] = mms.transform( nueva[['tasa']].values )

mms = joblib.load('parameter/saldo_capital.sav')
nueva['saldo_capital'] = mms.transform( nueva[['saldo_capital']].values )

mms = joblib.load('parameter/saldo_Ahorro.sav')
nueva['saldo_Ahorro'] = mms.transform( nueva[['saldo_Ahorro']].values )

mms = joblib.load('parameter/Antiguedad_en_meses.sav')
nueva['Antiguedad_en_meses'] = mms.transform( nueva[['Antiguedad_en_meses']].values )

mms = joblib.load('parameter/Max_dias_mora.sav')
nueva['Max_dias_mora'] = mms.transform( nueva[['Max_dias_mora']].values )

mms = joblib.load('parameter/Edad.sav')
nueva['Edad'] = mms.transform( nueva[['Edad']].values )

mms = joblib.load('parameter/plazo_dias.sav')
nueva['plazo_dias'] = mms.transform( nueva[['plazo_dias']].values )


X = nueva[[ 'Patrimonio', 'Ingresos_Mensuales', 'No_hijos',
       'monto_credito', 'tasa',
       'saldo_capital', 'saldo_Ahorro', 'Antiguedad_en_meses', 'Max_dias_mora',
       'plazo_dias', 'Edad', 'mes_ult_desembolso',
       'C_Barranquilla','C_Bogotá','C_Cartagena','C_Cali']]


classifier2 = joblib.load('model/final_model.pkl')
y_xgb = classifier2.predict(X)
y_xgb_proba = classifier2.predict_proba(X)
X = X.join(pd.DataFrame(y_xgb_proba))
# X = X.join(pd.DataFrame(y_xgb))
# st.write(y_xgb)
X = X.rename(columns={0:'Prob_no_aceptar',1:'Prob_aceptar'})
X['Prob_no_aceptar'] = np.round(X['Prob_no_aceptar'].astype(float),4)
X['Prob_aceptar'] = np.round(X['Prob_aceptar'].astype(float),4)
X = X.sort_values('Prob_aceptar', ascending = False)
st.write('La probabilidad de qué el cliente acepte ser parte de la campaña es de',100*X.loc[0,'Prob_aceptar'],'%')


