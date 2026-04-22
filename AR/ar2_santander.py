import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

def main():
    print("Cargando datos de loterias...")
    df = pd.read_csv('loterias.csv')
    
    # Filtrar solo Loteria Santander, Tipo de Premio 'Mayor'
    df_santander = df[(df['Lotería'] == 'Loteria Santander') & (df['Tipo de Premio'] == 'Mayor')].copy()
    
    # Asegurar que el número de billete sea numérico
    if df_santander['Numero billete ganador'].dtype == object:
        df_santander['Numero billete ganador'] = df_santander['Numero billete ganador'].astype(str).str.replace(',', '').astype(float)
    else:
        df_santander['Numero billete ganador'] = df_santander['Numero billete ganador'].astype(float)
        
    # Convertir la columna Fecha a datetime
    df_santander['Fecha del Sorteo'] = pd.to_datetime(df_santander['Fecha del Sorteo'], format='%d/%m/%Y', errors='coerce')
    
    # Eliminar posibles nulos y ordenar por fecha
    df_santander = df_santander.dropna(subset=['Fecha del Sorteo'])
    df_santander = df_santander.sort_values(by='Fecha del Sorteo')
    
    # Promediar en caso de fechas duplicadas (no debería haber para 'Mayor', pero por seguridad)
    df_santander = df_santander.groupby('Fecha del Sorteo')['Numero billete ganador'].mean().reset_index()
    df_santander.set_index('Fecha del Sorteo', inplace=True)
    
    serie_total = df_santander['Numero billete ganador']
    serie_valores = serie_total.values
    
    # Gráfica 1: Solo Loteria Santander
    print("\nGenerando gráfica original...")
    plt.figure(figsize=(12, 6))
    plt.plot(serie_total.index, serie_total.values, label='Billete Ganador (Santander)', marker='o', color='blue')
    plt.title('Serie Original - Lotería Santander (Premio Mayor)')
    plt.xlabel('Fecha del Sorteo')
    plt.ylabel('Número Ganador')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('loteria_santander_original.png', dpi=300)
    print("-> Guardada como 'loteria_santander_original.png'")
    
    # Ajuste AR(2)
    print("\nAjustando modelo AR(2) sobre el billete ganador de Loteria Santander...")
    modelo = AutoReg(serie_valores, lags=2)
    resultado = modelo.fit()
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DEL MODELO AR(2) [LOTERIA SANTANDER]:")
    print("="*80)
    print(resultado.summary())
    print("="*80 + "\n")
    
    const = resultado.params[0]
    phi_1 = resultado.params[1]
    phi_2 = resultado.params[2]
    
    print("Ecuación del proceso:")
    print(f"y_t = {const:.4f} + ({phi_1:.4f}) * y_{{t-1}} + ({phi_2:.4f}) * y_{{t-2}}\n")
    
    # Predicción dentro de muestra (índice 2 en adelante para AR(2))
    predicciones = resultado.predict(start=2, end=len(serie_valores)-1)
    fechas_prediccion = serie_total.index[2:]
    
    # Gráfica 2: Loteria Santander vs AR(2)
    print("Generando gráfica comparativa...")
    plt.figure(figsize=(12, 6))
    plt.plot(serie_total.index, serie_total.values, label='Real (Santander)', marker='o', color='blue', alpha=0.5)
    plt.plot(fechas_prediccion, predicciones, label='Ajuste AR(2)', color='red', linestyle='-', linewidth=2)
    plt.title('Modelo Autoregresivo AR(2) vs Lotería Santander')
    plt.xlabel('Fecha del Sorteo')
    plt.ylabel('Número Ganador')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('loteria_santander_vs_ar2.png', dpi=300)
    print("-> Guardada como 'loteria_santander_vs_ar2.png'")

if __name__ == '__main__':
    main()
