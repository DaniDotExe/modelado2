import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

def main():
    print("Cargando datos de consumo de energía...")
    df = pd.read_csv('consumo_energia.csv')
    
    # Filtrar solo la Ciudad A
    df_a = df[df['Ciudad'] == 'Ciudad A'].copy()
    
    # Convertir la columna Fecha a datetime
    df_a['Fecha'] = pd.to_datetime(df_a['Fecha'])
    
    # Dado que algunas fechas pueden estar duplicadas, agrupamos y promediamos por mes
    df_a = df_a.groupby('Fecha').mean(numeric_only=True).reset_index()
    df_a = df_a.sort_values(by='Fecha')
    df_a.set_index('Fecha', inplace=True)
    
    # Asegurar una frecuencia mensual regular
    df_a = df_a.asfreq('MS')
    
    # Interpolamos por si quedó algún mes vacío por la frecuencia regular
    serie_total = df_a['Consumo_Total(kWh)'].interpolate()
    
    print("\nAjustando modelo AR(2) sobre el Consumo Total de la Ciudad A...")
    # Ajustar modelo AR de orden 2
    modelo = AutoReg(serie_total, lags=2)
    resultado = modelo.fit()
    
    # Imprimir resumen
    print("\n" + "="*80)
    print("RESUMEN DEL MODELO AR(2) [CIUDAD A]:")
    print("="*80)
    print(resultado.summary())
    print("="*80 + "\n")
    
    # Extraer parámetros (la constante y los dos rezagos)
    const = resultado.params['const']
    phi_1 = resultado.params.iloc[1]
    phi_2 = resultado.params.iloc[2]
    
    print("Ecuación del proceso:")
    print(f"y_t = {const:.4f} + ({phi_1:.4f}) * y_{{t-1}} + ({phi_2:.4f}) * y_{{t-2}}\n")
    
    # Predecir valores para visualizar el ajuste dentro de la muestra
    # Empezamos en el índice 2 porque es un AR(2) y requiere 2 valores previos
    predicciones = resultado.predict(start=serie_total.index[2], end=serie_total.index[-1])
    
    # Proyectar algunos periodos hacia adelante (ej. 3 meses)
    # forecast = resultado.predict(start=len(serie_total), end=len(serie_total)+3)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(serie_total.index, serie_total, label='Consumo Total Real (Ciudad A)', marker='o', color='blue')
    plt.plot(predicciones.index, predicciones, label='Ajuste Modelo AR(2)', color='red', linestyle='--', linewidth=2)
    plt.title('Modelo Autoregresivo AR(2) - Consumo de Energía en Ciudad A')
    plt.xlabel('Fecha')
    plt.ylabel('Consumo Total (kWh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar gráfico
    output_filename = 'modelo_ar2_ciudad_A.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Gráfica del modelo generada y guardada como '{output_filename}'")

if __name__ == '__main__':
    main()
