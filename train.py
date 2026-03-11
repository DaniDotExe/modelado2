import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Ignorar advertencias de statsmodels sobre frecuencias inferidas
warnings.filterwarnings("ignore")

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_models():
    # Cargar datos limpios
    df = pd.read_csv('data_cleaned.csv')
    
    # Limpiar la columna Periodo 
    df['Year'] = df['Periodo'].astype(str).str[:4].astype(int)
    
    # Convertir 'Year' y 'Trimestre' a fechas 
    # Mes de inicio: Trimestre 1=Ene(1), 2=Abr(4), 3=Jul(7), 4=Oct(10)
    df['Month'] = (df['Trimestre'] - 1) * 3 + 1
    df['Date'] = pd.to_datetime({'year': df['Year'], 'month': df['Month'], 'day': 1})
    
    # Ordenar y definir el índice
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df.index.freq = pd.infer_freq(df.index) or 'QS-JAN'
    
    ts = df['PIB a precios constantes del 2015']
    
    # Entrenar desde 2016-1 (2016-01-01) hasta 2023-2 (2023-04-01)
    train_end_date = '2023-04-01'
    train = ts.loc[:train_end_date]
    
    # Obtener los próximos 7 periodos como conjunto de prueba
    test = ts.loc[train.index[-1]:].iloc[1:8]  # Tomar los 7 que siguen al final del train
    
    print(f"Tamano del set de entrenamiento: {len(train)} trimestres (hasta {train.index[-1].year}-Q{train.index[-1].quarter})")
    print(f"Tamano del set de prueba: {len(test)} trimestres (desde {test.index[0].year}-Q{test.index[0].quarter} hasta {test.index[-1].year}-Q{test.index[-1].quarter})")
    print("-" * 50)
    
    # 1. Configuración Completamente Aditiva
    modelo_aditivo = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=4).fit()
    pred_aditivo = modelo_aditivo.forecast(7)
    mape_aditivo = mape(test, pred_aditivo)
    
    # 2. Configuración Completamente Multiplicativa
    modelo_multiplicativo = ExponentialSmoothing(train, trend='mul', seasonal='mul', seasonal_periods=4).fit()
    pred_multiplicativo = modelo_multiplicativo.forecast(7)
    mape_multiplicativo = mape(test, pred_multiplicativo)
    
    print(f"MAPE Modelo Aditivo (Aditivo - Aditivo):            {mape_aditivo:.3f}%")
    print(f"MAPE Modelo Multiplicativo (Multiplicativo - Multiplicativo): {mape_multiplicativo:.3f}%")
    print("-" * 50)
    
    if mape_aditivo < mape_multiplicativo:
        print("=> El modelo con configuracion ADITIVA tuvo el menor MAPE.")
    else:
        print("=> El modelo con configuracion MULTIPLICATIVA tuvo el menor MAPE.")

if __name__ == "__main__":
    evaluate_models()
