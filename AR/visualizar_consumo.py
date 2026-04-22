import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Cargando datos de consumo de energía...")
    # Leer el archivo CSV
    df = pd.read_csv('consumo_energia.csv')
    
    # Convertir la columna Fecha a tipo datetime para que se ordene y grafique correctamente
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Ordenar por fecha para evitar líneas que crucen el gráfico de atrás hacia adelante
    df = df.sort_values(by='Fecha')
    
    # Crear una figura con dos subgráficos (uno para total y otro para promedio)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Obtener la lista de ciudades únicas
    ciudades = df['Ciudad'].unique()
    
    # Gráfico 1: Consumo Total por Ciudad
    for ciudad in ciudades:
        datos_ciudad = df[df['Ciudad'] == ciudad]
        ax1.plot(datos_ciudad['Fecha'], datos_ciudad['Consumo_Total(kWh)'], marker='o', label=ciudad)
        
    ax1.set_title('Consumo Total de Energía (kWh) por Ciudad')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Consumo Total (kWh)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico 2: Consumo Promedio por Hogar por Ciudad
    for ciudad in ciudades:
        datos_ciudad = df[df['Ciudad'] == ciudad]
        ax2.plot(datos_ciudad['Fecha'], datos_ciudad['Consumo_Promedio_Hogar(kWh)'], marker='s', label=ciudad)
        
    ax2.set_title('Consumo Promedio por Hogar (kWh) por Ciudad')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Consumo Promedio (kWh)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar el espaciado para que no se superpongan textos
    plt.tight_layout()
    
    # Guardar la figura como imagen PNG
    output_filename = 'consumo_energia_visualizacion.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Visualización generada y guardada exitosamente como '{output_filename}'.")

    # --- Gráfica para Ciudad A solamente ---
    df_ciudad_a = df[df['Ciudad'] == 'Ciudad A']
    
    fig_a, (ax1_a, ax2_a) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gráfico 1: Consumo Total Ciudad A
    ax1_a.plot(df_ciudad_a['Fecha'], df_ciudad_a['Consumo_Total(kWh)'], marker='o', color='blue')
    ax1_a.set_title('Consumo Total de Energía (kWh) - Ciudad A')
    ax1_a.set_xlabel('Fecha')
    ax1_a.set_ylabel('Consumo Total (kWh)')
    ax1_a.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico 2: Consumo Promedio Ciudad A
    ax2_a.plot(df_ciudad_a['Fecha'], df_ciudad_a['Consumo_Promedio_Hogar(kWh)'], marker='s', color='orange')
    ax2_a.set_title('Consumo Promedio por Hogar (kWh) - Ciudad A')
    ax2_a.set_xlabel('Fecha')
    ax2_a.set_ylabel('Consumo Promedio (kWh)')
    ax2_a.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_a = 'consumo_energia_ciudad_A.png'
    plt.savefig(output_a, dpi=300)
    print(f"Visualización de Ciudad A generada y guardada exitosamente como '{output_a}'.")


if __name__ == '__main__':
    main()
