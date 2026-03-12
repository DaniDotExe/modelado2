import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Ignorar advertencias de statsmodels sobre frecuencias inferidas
warnings.filterwarnings("ignore")

def make_prediction():
    # 1. Cargar datos limpios
    df = pd.read_csv('data_cleaned.csv')
    
    # Limpiar columnas para facilitar el manejo temporal
    df['Year'] = df['Periodo'].astype(str).str[:4].astype(int)
    df['Month'] = (df['Trimestre'] - 1) * 3 + 1
    df['Date'] = pd.to_datetime({'year': df['Year'], 'month': df['Month'], 'day': 1})
    
    # Ordenar y definir el índice
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df.index.freq = pd.infer_freq(df.index) or 'QS-JAN'
    
    # 2. Obtener la serie de tiempo para entrenar (hasta 2025-1)
    ts = df['PIB a precios constantes del 2015']
    
    print("Entrenando modelo Holt-Winters (Multiplicativo) con todos los datos históricos...")
    # Entrenar el modelo con configuración Multiplicativa (según resultados del super_train.py)
    modelo = ExponentialSmoothing(ts, trend='mul', seasonal='mul', seasonal_periods=4)
    modelo_ajustado = modelo.fit(smoothing_level=0.1, smoothing_trend=0.1, smoothing_seasonal=0.5, optimized=False)
    
    # 3. Predecir los próximos 7 periodos (de 2025-2 a 2026-4)
    pasos = 7
    predicciones = modelo_ajustado.forecast(pasos)
    
    # 4. Formatear los datos combinando Histórico y Predicción
    # DataFrame con datos históricos
    df_historico = pd.DataFrame({
        'Año': ts.index.year,
        'Trimestre': ts.index.quarter,
        'PIB a precios constantes del 2015': ts.values.round(2),
        'Tipo': 'Historico'
    })
    
    # DataFrame con predicciones
    df_prediccion = pd.DataFrame({
        'Año': predicciones.index.year,
        'Trimestre': predicciones.index.quarter,
        'PIB a precios constantes del 2015': predicciones.values.round(2),
        'Tipo': 'Prediccion'
    })
    
    # Unir ambos DataFrames (los originales arriba, las predicciones debajo)
    df_final = pd.concat([df_historico, df_prediccion], ignore_index=True)
    
    # 5. Exportar el resultado final
    archivo_salida = 'output.csv'
    df_final.to_csv(archivo_salida, index=False)
    
    print(f"\nPrediccion completada.")
    print(f"Total registros históricos: {len(df_historico)}")
    print(f"Total registros de prediccion: {len(df_prediccion)} (desde 2025-Q2 hasta 2026-Q4)")
    print(f"Los resultados combinados se han guardado en '{archivo_salida}'.")

if __name__ == "__main__":
    make_prediction()
