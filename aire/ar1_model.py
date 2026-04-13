import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

def main():
    print("Cargando datos...")
    # Leer el archivo CSV que contiene la columna 'Valor'
    df = pd.read_csv('data_page_27.csv')
    
    print("\nAjustando modelo AR(1)...")
    # Ajustar un modelo Autoregresivo de orden 1 - AR(1)
    # lags=1 indica que es de orden 1
    modelo = AutoReg(df['Valor'], lags=1)
    resultado = modelo.fit()
    
    # Imprimir los parámetros y detalles estadísticos en pantalla
    print("\n" + "="*80)
    print("RESUMEN DEL MODELO AR(1):")
    print("="*80)
    print(resultado.summary())
    print("="*80 + "\n")
    
    # Obtener las predicciones sobre el propio set de datos para ver el ajuste
    # El ajuste comienza en el índice 1 porque el lag de 1 necesita el valor t-1
    df['Ajuste_AR1'] = resultado.predict(start=1, end=len(df)-1)
    
    # Hacer una proyeccción al futuro de 5 pasos
    forecast = resultado.predict(start=len(df), end=len(df)+4)
    
    # Crear la figura
    plt.figure(figsize=(10, 6))
    
    # Graficar la serie original
    plt.plot(df.index, df['Valor'], label='Datos Originales', marker='o', color='blue', linewidth=2)
    
    # Graficar el ajuste dentro de la muestra
    plt.plot(df.index, df['Ajuste_AR1'], label='Ajuste AR(1) (En muestra)', color='red', marker='x', linestyle='--')
    
    # Graficar la predicción de valores futuros
    plt.plot(forecast.index, forecast.values, label='Pronóstico a Futuro (5 pasos)', color='orange', marker='s', linestyle='-.')
    
    # Ajustes estéticos y de texto
    plt.title('Estimación y Pronóstico con Modelo AR(1)')
    plt.xlabel('Observación (Índice)')
    plt.ylabel('Valor')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    print("Mostrando gráfica en pantalla...")
    plt.show()

if __name__ == "__main__":
    main()
