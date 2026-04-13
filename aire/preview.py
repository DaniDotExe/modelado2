import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Leer el archivo CSV
    df = pd.read_csv('calidad_aire.csv')
    
    # Limpiar y convertir la columna PM10 a numérico
    # Reemplazar comas por puntos en los valores de la columna PM10 si son strings
    if df['PM10'].dtype == 'O':
        df['PM10'] = df['PM10'].str.replace(',', '.').astype(float)
        
    # Convertir 'Fecha inicial' a datetime
    df['Fecha inicial'] = pd.to_datetime(df['Fecha inicial'])
    
    # Ordenar por fecha
    df = df.sort_values('Fecha inicial')
    
    # Crear la gráfica
    plt.figure(figsize=(12, 6))
    
    # Agrupar por estación por si hay más de una
    estaciones = df['Estacion'].unique()
    for estacion in estaciones:
        df_est = df[df['Estacion'] == estacion]
        plt.plot(df_est['Fecha inicial'], df_est['PM10'], marker='.', linestyle='-', linewidth=1, label=estacion)
        
    plt.title('Niveles de PM10 en Calidad del Aire')
    plt.xlabel('Fecha')
    plt.ylabel('PM10')
    plt.grid(True, linestyle='--', alpha=0.7)
    if len(estaciones) > 1:
        plt.legend()
    plt.xticks(rotation=45)
    
    # Ajustar márgenes
    plt.tight_layout()
    
    # Mostrar la gráfica
    plt.show()

if __name__ == '__main__':
    main()
