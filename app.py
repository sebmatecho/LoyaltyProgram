from sklearn.preprocessing import MinMaxScaler

import datetime
import streamlit as st
import pandas as pd
import pickle
import joblib
# Header 
st.write('''
# *Loyalty program* 
##### Sébastien Lozano Forero 
''')


nueva = pd.read_csv('Base_nueva_DC.csv',encoding = "ISO-8859-1", sep =';',parse_dates=['fecha_ult_desembolso'], 
                  dtype = {
                      'Id_Cliente':str,
                      'Ciudad':str,
                      ' Patrimonio ':int,
                      ' Ingresos_Mensuales ':int,
                      'No_hijos':int,
                      'oficina':str,
                      'monto_credito':int,
                      'tasa':float,
                      'saldo_capital':int,
                      'Antiguedad_en_meses':int,
                      'Max_dias_mora':int,
                      'plazo_dias':int,
                      'Edad':int
                  })

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

X = nueva[[ 'Patrimonio', 'Ingresos_Mensuales', 'No_hijos',
       'monto_credito', 'tasa',
       'saldo_capital', 'saldo_Ahorro', 'Antiguedad_en_meses', 'Max_dias_mora',
       'plazo_dias', 'Edad', 'mes_ult_desembolso',
       'C_Barranquilla','C_Bogotá','C_Cartagena','C_Cali']]

X.head().T
# test = joblib.load('parameter/Patrimonio.gz',mmap_mode = None).fit_transform( X[['Patrimonio']].values )
# test
X[['Patrimonio']].values 
test_trans = joblib.load(open('parameter/Patrimonio.gz', 'rb'))
X[['Patrimonio']] = test_trans.transform( X[['Patrimonio']].values )
#  = joblib.load('parameter/Patrimonio.gz',mmap_mode = None).fit_transform( X[['Patrimonio']].values )
#
# X['Ingresos_Mensuales'] = mms.fit_transform( X[['Ingresos_Mensuales']].values )
# mms =pickle.load( open('parameter/No_hijos.pkl', 'rb') )
# X['No_hijos'] = mms.fit_transform( X[['No_hijos']].values )
# mms =pickle.load( open('parameter/monto_credito.pkl', 'rb') )
# X['monto_credito'] = mms.fit_transform( X[['monto_credito']].values )
# mms =pickle.load( open('parameter/tasa.pkl', 'rb') )
# X['tasa'] = mms.fit_transform( X[['tasa']].values )
# mms =pickle.load( open('parameter/saldo_capital.pkl', 'rb') )
# X['saldo_capital'] = mms.fit_transform( X[['saldo_capital']].values )
# mms =pickle.load( open('parameter/saldo_Ahorro.pkl', 'rb') )
# X['saldo_Ahorro'] = mms.fit_transform( X[['saldo_Ahorro']].values )
# mms =pickle.load( open('parameter/Antiguedad_en_meses.pkl', 'rb') )
# X['Antiguedad_en_meses'] = mms.fit_transform( X[['Antiguedad_en_meses']].values )
# mms =pickle.load( open('parameter/Max_dias_mora.pkl', 'rb') )
# X['Max_dias_mora'] = mms.fit_transform( X[['Max_dias_mora']].values )
# mms =pickle.load( open('parameter/Edad.pkl', 'rb') )
# X['Edad'] = mms.fit_transform( X[['Edad']].values )
# mms =pickle.load( open('parameter/plazo_dias.pkl', 'rb') )
# X['plazo_dias'] = mms.fit_transform( X[['plazo_dias']].values )
# mms =pickle.load( open('parameter/mes_ult_desembolso.pkl', 'rb') )
# X['mes_ult_desembolso'] = mms.fit_transform( X[['mes_ult_desembolso']].values )
# X['semana_ult_desembolso'] = mms.fit_transform( X[['semana_ult_desembolso']].values )

X.head().T
# classifier2 = pickle.load( open( '/Users/Windows/Desktop/Repos/Robayo/model/final_model.pkl', 'rb') )
# y_xgb = classifier2.predict(X)
# y_xgb_proba = classifier2.predict_proba(X)
# X = X.join(pd.DataFrame(y_xgb_proba))
# nueva = nueva.join(pd.DataFrame(y_xgb))


# X = X.rename(columns={0:'Prob_no_aceptar',1:'Prob_aceptar'})
# X['Prob_aceptar']

# nueva['Prob_aceptar']


