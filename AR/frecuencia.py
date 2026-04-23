import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Cargando datos de loterias...")
    # Asegurarnos de que funcione sin importar si se ejecuta desde modelado2/ o modelado2/AR/
    try:
        df = pd.read_csv('loterias.csv')
    except FileNotFoundError:
        df = pd.read_csv('AR/loterias.csv')
        
    # Filtrar solo Loteria Santander, usando TODOS los datos
    df_santander = df[df['Lotería'] == 'Loteria Santander'].copy()
    
    if df_santander.empty:
        print("No se encontraron datos para la Lotería Santander.")
        return
        
    # Asegurar que el número de billete sea numérico
    if df_santander['Numero billete ganador'].dtype == object:
        df_santander['Numero billete ganador'] = df_santander['Numero billete ganador'].astype(str).str.replace(',', '').astype(float)
    else:
        df_santander['Numero billete ganador'] = df_santander['Numero billete ganador'].astype(float)
        
    # Eliminar nulos en el número ganador
    df_santander = df_santander.dropna(subset=['Numero billete ganador'])
    numeros = df_santander['Numero billete ganador']
    
    # 1. Tabla de distribución de frecuencias (agrupada en intervalos de 1000)
    bins = range(0, 10001, 1000) # De 0 a 10000 de 1000 en 1000
    df_santander['Rango'] = pd.cut(numeros, bins=bins, right=False)
    
    frecuencias = df_santander['Rango'].value_counts().sort_index().reset_index()
    frecuencias.columns = ['Rango de Números', 'Frecuencia Absoluta']
    frecuencias['Frecuencia Relativa'] = frecuencias['Frecuencia Absoluta'] / len(numeros)
    frecuencias['Frecuencia Relativa (%)'] = frecuencias['Frecuencia Relativa'] * 100
    
    print("\n" + "="*65)
    print("DISTRIBUCIÓN DE FRECUENCIAS (INTERVALOS) - LOTERIA SANTANDER")
    print("="*65)
    print(frecuencias.to_string(index=False))
    print("="*65 + "\n")
    
    # 2. Top 10 números más frecuentes (frecuencia exacta)
    top_frecuentes = numeros.value_counts().head(10).reset_index()
    top_frecuentes.columns = ['Número', 'Frecuencia']
    # Convertir el número a entero para que se vea sin decimales si es posible
    top_frecuentes['Número'] = top_frecuentes['Número'].astype(int).astype(str).str.zfill(4)
    
    print("TOP 10 NÚMEROS MÁS FRECUENTES (EXACTOS):")
    print(top_frecuentes.to_string(index=False))
    print("="*65 + "\n")
    
    # Análisis de los últimos 3, 2 y 1 dígitos
    str_numeros = numeros.astype(int).astype(str).str.zfill(4)
    
    ultimos_3 = str_numeros.str[-3:]
    top_ultimos_3 = ultimos_3.value_counts().head(10).reset_index()
    top_ultimos_3.columns = ['Últimos 3 Dígitos', 'Frecuencia']
    print("TOP 10 MÁS FRECUENTES (ÚLTIMOS 3 DÍGITOS):")
    print(top_ultimos_3.to_string(index=False))
    print("="*65 + "\n")
    
    ultimos_2 = str_numeros.str[-2:]
    top_ultimos_2 = ultimos_2.value_counts().head(10).reset_index()
    top_ultimos_2.columns = ['Últimos 2 Dígitos', 'Frecuencia']
    print("TOP 10 MÁS FRECUENTES (ÚLTIMOS 2 DÍGITOS):")
    print(top_ultimos_2.to_string(index=False))
    print("="*65 + "\n")
    
    ultimo_1 = str_numeros.str[-1:]
    top_ultimo_1 = ultimo_1.value_counts().head(10).reset_index()
    top_ultimo_1.columns = ['Último Dígito', 'Frecuencia']
    print("TOP 10 MÁS FRECUENTES (ÚLTIMO DÍGITO):")
    print(top_ultimo_1.to_string(index=False))
    print("="*65 + "\n")
    
    # Guardar tabla de frecuencias agrupadas a CSV
    try:
        frecuencias.to_csv('frecuencias_rango_santander.csv', index=False)
        print("Tabla de frecuencias guardada en 'frecuencias_rango_santander.csv'")
    except Exception as e:
        pass
        
    # Graficar el Histograma
    plt.figure(figsize=(10, 6))
    
    # Usar los mismos bins para el histograma
    counts, edges, bars = plt.hist(numeros, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
    
    # Añadir las cantidades sobre las barras
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height + (max(counts) * 0.01), 
                     str(int(height)), ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
    plt.title('Histograma de Frecuencias - Lotería Santander (Todos los premios)')
    plt.xlabel('Rango de Número Ganador')
    plt.ylabel('Frecuencia Absoluta')
    plt.xticks(bins)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Guardar gráfico
    output_filename = 'frecuencia_loteria_santander.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Histograma generado y guardado como '{output_filename}'")

if __name__ == '__main__':
    main()
