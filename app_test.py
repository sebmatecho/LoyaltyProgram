from sklearn.preprocessing import MinMaxScaler

import numpy as np
import datetime
import streamlit as st
import pandas as pd
import pickle
import joblib
import boto3
import tempfile
# Header 
st.markdown('''
# *Loyalty program* 
### Proposed by [Sébastien Lozano Forero](https://www.linkedin.com/in/sebastienlozanoforero/)


This dataset was provided by a Colombian bank who used it for technical assessments while looking to fill a Data Scientist position. 
#### The described scenario
The bank has a loyalty program and would only invite new customers to join every year. 
Customers in such program would have access to exclusive services within the bank portfolio. 
Formal invitation to the program is made through sending letter and some presents to potential customers. 
Now, the bank would like to be as effective as possible when selecting customers to invite to the program.
#### The data
This way, a dataset containing information on customers and the outcome on wheather he/she accepts or decline the invitation to join the program, based on prior years information, is presented.
#### The task
To design a tool to better select customers when sending invitations out to such program. 
''')



Edad = st.slider('Age', 0, 130, 25)
Patrimonio = st.number_input('Wealth')
Ingresos_Mensuales = st.number_input('Montly Income (COP)')
No_hijos = st.number_input('Dependants')
fecha_ult_desembolso = st.date_input("Last disbursement date (yyyy/mm/dd)", datetime.date(2019, 7, 6))
monto_credito = st.number_input('Loan amount')
tasa = st.number_input('Interest Rate (year)')
saldo_capital = st.number_input('Principal Balance')
saldo_Ahorro = st.number_input('Savings Balance')
Antiguedad_en_meses = st.number_input('Antiquity (months)')
Max_dias_mora = st.number_input('Max days after due date')
plazo_dias = st.number_input('days until due date')
mes_ult_desembolso = fecha_ult_desembolso.month

ciudad = st.selectbox(
     'City',
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

client = boto3.client('s3',
            aws_access_key_id = 'AKIAW6IPH3CTUNR7ATJS',
            aws_secret_access_key = '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')

list_names = ['Patrimonio','Ingresos_Mensuales','No_hijos','monto_credito',
               'tasa', 'saldo_capital','saldo_Ahorro','Antiguedad_en_meses',
               'Max_dias_mora','Edad','plazo_dias']
for name in list_names:

    with tempfile.TemporaryFile() as fp: 
        client.download_fileobj(Fileobj=fp, 
                                Bucket='loyaltyprogrammodelandparameters',
                                Key=name+'.sav')
        fp.seek(0)
        mms = joblib.load(fp)
    nueva[name] = mms.transform( nueva[[name]].values )




X = nueva[[ 'Patrimonio', 'Ingresos_Mensuales', 'No_hijos',
       'monto_credito', 'tasa',
       'saldo_capital', 'saldo_Ahorro', 'Antiguedad_en_meses', 'Max_dias_mora',
       'plazo_dias', 'Edad', 'mes_ult_desembolso',
       'C_Barranquilla','C_Bogotá','C_Cartagena','C_Cali']]




with tempfile.TemporaryFile() as fp: 
    client.download_fileobj(Fileobj=fp, 
                            Bucket='loyaltyprogrammodelandparameters',
                            Key='final_model.pkl')
    fp.seek(0)
    classifier2 = joblib.load(fp)

y_xgb = classifier2.predict(X)
y_xgb_proba = classifier2.predict_proba(X)
X = X.join(pd.DataFrame(y_xgb_proba))
# X = X.join(pd.DataFrame(y_xgb))
# st.write(y_xgb)
X = X.rename(columns={0:'Prob_no_aceptar',1:'Prob_aceptar'})
X['Prob_no_aceptar'] = np.round(X['Prob_no_aceptar'].astype(float),4)
X['Prob_aceptar'] = np.round(X['Prob_aceptar'].astype(float),4)
X = X.sort_values('Prob_aceptar', ascending = False)
st.write('Chances customer accepts invitation to join the program',100*X.loc[0,'Prob_aceptar'],'%')

