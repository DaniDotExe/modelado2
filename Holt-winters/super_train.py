import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import itertools

# Ignorar advertencias de statsmodels sobre frecuencias inferidas
warnings.filterwarnings("ignore")

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def grid_search():
    # Lista para capturar los resultados de la consola
    output_lines = []
    
    def log_print(message):
        print(message)
        output_lines.append(message)

    # Cargar datos limpios
    df = pd.read_csv('data_cleaned.csv')
    
    # Limpiar columnas
    df['Year'] = df['Periodo'].astype(str).str[:4].astype(int)
    df['Month'] = (df['Trimestre'] - 1) * 3 + 1
    df['Date'] = pd.to_datetime({'year': df['Year'], 'month': df['Month'], 'day': 1})
    
    # Ordenar y definir el índice
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df.index.freq = pd.infer_freq(df.index) or 'QS-JAN'
    
    ts = df['PIB a precios constantes del 2015']
    
    # Entrenar desde 2016-1 hasta 2023-2
    train_end_date = '2023-04-01'
    train = ts.loc[:train_end_date]
    
    # Obtener los próximos 7 periodos como conjunto de prueba
    test = ts.loc[train.index[-1]:].iloc[1:8]
    
    log_print(f"Tamano del set de entrenamiento: {len(train)} trimestres (hasta {train.index[-1].year}-Q{train.index[-1].quarter})")
    log_print(f"Tamano del set de prueba: {len(test)} trimestres (desde {test.index[0].year}-Q{test.index[0].quarter} hasta {test.index[-1].year}-Q{test.index[-1].quarter})")
    log_print("-" * 50)
    
    # Definir un grid de parámetros para (alpha, beta, gamma)
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    param_grid = list(itertools.product(alphas, betas, gammas))
    log_print(f"Iniciando Grid Search con {len(param_grid)} combinaciones posibles por modelo...")
    log_print(f"Cota Superior iterando sobre alphas, betas y gammas de {min(alphas)} a {max(alphas)}")
    log_print("-" * 50)
    
    best_aditivo = {'mape': float('inf'), 'params': None}
    best_multiplicativo = {'mape': float('inf'), 'params': None}
    
    # Configuración de base Aditiva
    modelo_base_aditivo = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=4)
    # Configuración de base Multiplicativa
    modelo_base_multiplicativo = ExponentialSmoothing(train, trend='mul', seasonal='mul', seasonal_periods=4)
    
    for alpha, beta, gamma in param_grid:
        # Evaluar Aditivo
        try:
            fit_add = modelo_base_aditivo.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
            pred_add = fit_add.forecast(7)
            mape_add = mape(test, pred_add)
            
            if mape_add < best_aditivo['mape']:
                best_aditivo['mape'] = mape_add
                best_aditivo['params'] = (alpha, beta, gamma)
        except Exception:
            pass
            
        # Evaluar Multiplicativo
        try:
            fit_mul = modelo_base_multiplicativo.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
            pred_mul = fit_mul.forecast(7)
            mape_mul = mape(test, pred_mul)
            
            if mape_mul < best_multiplicativo['mape']:
                best_multiplicativo['mape'] = mape_mul
                best_multiplicativo['params'] = (alpha, beta, gamma)
        except Exception:
            pass

    log_print("Mejor resultado Modelo Aditivo (Aditivo - Aditivo):")
    log_print(f"  MAPE:  {best_aditivo['mape']:.3f}%")
    log_print(f"  Alpha (nivel):      {best_aditivo['params'][0]}")
    log_print(f"  Beta (tendencia):   {best_aditivo['params'][1]}")
    log_print(f"  Gamma (estacional): {best_aditivo['params'][2]}")
    log_print("-" * 50)
    
    log_print("Mejor resultado Modelo Multiplicativo (Multiplicativo - Multiplicativo):")
    log_print(f"  MAPE:  {best_multiplicativo['mape']:.3f}%")
    log_print(f"  Alpha (nivel):      {best_multiplicativo['params'][0]}")
    log_print(f"  Beta (tendencia):   {best_multiplicativo['params'][1]}")
    log_print(f"  Gamma (estacional): {best_multiplicativo['params'][2]}")
    log_print("-" * 50)
    
    if best_aditivo['mape'] < best_multiplicativo['mape']:
        log_print("=> El modelo con configuracion ADITIVA y parametros manuales obtuvo el menor MAPE optimizado.")
    else:
        log_print("=> El modelo con configuracion MULTIPLICATIVA y parametros manuales obtuvo el menor MAPE optimizado.")

    # Guardar en archivo .txt
    with open('resultados_superentrenamiento.txt', 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nResultados del Grid Search guardados exitosamente en 'resultados_superentrenamiento.txt'")

if __name__ == "__main__":
    grid_search()
