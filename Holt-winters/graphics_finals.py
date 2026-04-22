import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions():
    # Cargar los datos desde output.csv
    df = pd.read_csv('output.csv')
    
    # Crear una columna de etiqueta para el eje X (ejemplo: "2016-Q1")
    df['Periodo_Eje'] = df['Año'].astype(str) + '-Q' + df['Trimestre'].astype(str)
    
    # Separar los datos por tipo
    historico = df[df['Tipo'] == 'Historico']
    prediccion = df[df['Tipo'] == 'Prediccion']
    
    # Para que la línea se vea continua, incluimos el último punto histórico 
    # en la serie temporal de la predicción
    ultimo_historico = historico.iloc[-1:]
    pred_continua = pd.concat([ultimo_historico, prediccion])
    
    # Configurar el tamaño del gráfico
    plt.figure(figsize=(14, 7))
    
    # Trazar los datos históricos (línea azul, sólida, marcadores circulares)
    plt.plot(historico['Periodo_Eje'], 
             historico['PIB a precios constantes del 2015'], 
             label='Histórico original', 
             color='#1f77b4', 
             linewidth=2.5, 
             marker='o',
             markersize=4)
    
    # Trazar los datos predichos (línea naranja/roja, punteada, marcadores en forma de x)
    plt.plot(pred_continua['Periodo_Eje'], 
             pred_continua['PIB a precios constantes del 2015'], 
             label='Predicción (> 2025-Q1)', 
             color='#ff7f0e', 
             linewidth=2.5, 
             linestyle='--', 
             marker='s',
             markersize=5)
    
    # Diseño adicional del gráfico
    plt.title('Evolución y Predicción del PIB en Educación (Precios Constantes 2015)', fontsize=15, pad=15)
    plt.xlabel('Período (Año y Trimestre)', fontsize=12)
    plt.ylabel('PIB a precios constantes', fontsize=12)
    
    # Rotar las etiquetas del eje X para mejor lectura
    plt.xticks(rotation=45)
    
    # Reducir la cantidad de etiquetas visibles en el eje X para que no se superpongan
    # Tomamos 1 de cada 4 etiquetas (1 por año aprox)
    ax = plt.gca()
    todas_las_x = ax.get_xticks()
    if len(todas_las_x) > 10:
        ax.set_xticks(todas_las_x[::4])
    
    # Mostrar la cuadrícula y leyenda
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Ajustar para que no se recorten las etiquetas
    plt.tight_layout()
    
    # Guardar la imagen
    nombre_imagen = 'grafico_prediccion.png'
    plt.savefig(nombre_imagen, dpi=300)
    print(f"El gráfico se ha generado exitosamente y guardado como '{nombre_imagen}'")
    
    # Mostrar el gráfico de manera interactiva
    plt.show()

if __name__ == "__main__":
    plot_predictions()
