import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

def main():
    print("Cargando datos de la derivada...")
    # Leer el archivo CSV con la columna derivada
    df = pd.read_csv('data_page_27_derivada.csv')
    
    # Eliminar el primer valor (NaN) que resulta de la diferenciación
    serie_derivada = df['derivada'].dropna()
    
    print("\nAjustando modelo AR(1) sobre la columna 'derivada'...")
    # Ajustar un modelo Autoregresivo de orden 1 - AR(1)
    modelo = AutoReg(serie_derivada, lags=1)
    resultado = modelo.fit()
    
    # Imprimir resumen del modelo en consola
    print("\n" + "="*80)
    print("RESUMEN DEL MODELO AR(1) [COLUMNA DERIVADA]:")
    print("="*80)
    print(resultado.summary())
    print("="*80 + "\n")
    
    # Parámetros para la ecuación
    c = resultado.params['const']
    phi = resultado.params.iloc[1]
    
    print(f"Ecuación del proceso:")
    print(f"y_t = {c:.4f} + ({phi:.4f}) * y_{{t-1}}")
    print(f"Forma con operador B: (1 - ({phi:.4f})B) y_t = {c:.4f}\n")

    # Predicciones
    ajuste = resultado.predict(start=serie_derivada.index[1], end=serie_derivada.index[-1])
    forecast = resultado.predict(start=len(df), end=len(df) + 4)
    
    # Gráfica - Solo la derivada
    plt.figure(figsize=(10, 6))
    plt.plot(serie_derivada.index, serie_derivada, label='Serie Derivada', marker='o', color='purple', linewidth=2)
    
    plt.title('Gráfica de la Derivada ($y_t - y_{t-1}$)')
    plt.axhline(0, color='black', alpha=0.5, linestyle='-') # Eje neutral
    plt.xlabel('Observación')
    plt.ylabel('Valor')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
