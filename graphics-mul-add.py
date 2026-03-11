import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Ignorar advertencias de statsmodels
warnings.filterwarnings("ignore")

def plot_models_comparison():
    # 1. Cargar y preparar datos (mismo proceso que train.py)
    df = pd.read_csv('data_cleaned.csv')
    df['Year'] = df['Periodo'].astype(str).str[:4].astype(int)
    df['Month'] = (df['Trimestre'] - 1) * 3 + 1
    df['Date'] = pd.to_datetime({'year': df['Year'], 'month': df['Month'], 'day': 1})
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df.index.freq = pd.infer_freq(df.index) or 'QS-JAN'
    
    ts = df['PIB a precios constantes del 2015']
    
    # 2. Dividir en Entrenamiento y Prueba
    train_end_date = '2023-04-01'
    train = ts.loc[:train_end_date]
    test = ts.loc[train.index[-1]:].iloc[1:8] # Próximos 7 periodos
    
    # 3. Entrenar y predecir ambos modelos
    # Aditivo
    modelo_aditivo = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=4).fit()
    pred_aditivo = modelo_aditivo.forecast(7)
    
    # Multiplicativo
    modelo_multiplicativo = ExponentialSmoothing(train, trend='mul', seasonal='mul', seasonal_periods=4).fit()
    pred_multiplicativo = modelo_multiplicativo.forecast(7)
    
    # 4. Crear el gráfico
    plt.figure(figsize=(15, 8))
    
    # Para que las líneas de predicción se conecten con el último dato real de entrenamiento
    ultimo_real = pd.Series([train.iloc[-1]], index=[train.index[-1]])
    
    pred_aditiva_continua = pd.concat([ultimo_real, pred_aditivo])
    pred_multiplicativa_continua = pd.concat([ultimo_real, pred_multiplicativo])
    
    # Toda la serie real (train + test) para que se vea el fondo completo
    todo_real_eval = pd.concat([train, test])
    
    # Eje X en formato legible
    x_labels = [f"{d.year}-Q{d.quarter}" for d in todo_real_eval.index]
    
    # Graficar Entrenamiento
    plt.plot(x_labels[:len(train)], train.values, 
             label='Entrenamiento (Real)', color='#1f77b4', linewidth=3)
             
    # Graficar Prueba (Parte que intentamos predecir)
    plt.plot(x_labels[len(train)-1:], pd.concat([ultimo_real, test]).values, 
             label='Prueba (Real)', color='black', linewidth=3, linestyle='-', marker='o')
    
    # Graficar predicción Aditiva
    plt.plot(x_labels[len(train)-1:], pred_aditiva_continua.values, 
             label='Predicción Aditiva', color='#2ca02c', linewidth=2.5, linestyle='--', marker='^')
             
    # Graficar predicción Multiplicativa
    plt.plot(x_labels[len(train)-1:], pred_multiplicativa_continua.values, 
             label='Predicción Multiplicativa', color='#d62728', linewidth=2.5, linestyle='-.', marker='s')
    
    # Configuración de diseño
    plt.title('Comparación Holt-Winters: Aditivo vs Multiplicativo', fontsize=16, pad=15)
    plt.xlabel('Período', fontsize=12)
    plt.ylabel('PIB a precios constantes del 2015', fontsize=12)
    plt.xticks(rotation=45)
    
    # Filtro de etiquetas del eje X para mejor legibilidad (mostrar 1 de cada 3)
    ax = plt.gca()
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 3 != 0:
            label.set_visible(False)
            
    # Resaltar la zona de predicción (fondo semitransparente)
    ax.axvspan(len(train)-1, len(x_labels)-1, alpha=0.1, color='gray', label='Zona de Evaluación')
            
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    
    # Guardar la imagen
    plt.savefig('comparacion_modelos.png', dpi=300)
    print("Gráfico guardado como 'comparacion_modelos.png'")
    
    # Mostrar el gráfico (esto puede requerir cerrar la ventana anterior del OS si quedó abierta)
    plt.show()

if __name__ == "__main__":
    plot_models_comparison()
