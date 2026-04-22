import pandas as pd

def clean_data(input_file='data_raw.xlsx', output_file='data_cleaned.csv'):
    print(f"Leyendo los datos de {input_file}...")
    df = pd.read_excel(input_file)
    
    # Extraer el año como número entero (para manejar valores como '2023p' o '2024pr')
    df['Periodo_num'] = df['Periodo'].astype(str).str[:4].astype(int)
    
    # Filtrar por actividad económica "Educacion" y años del 2016 al 2025
    df_filtered = df[
        (df['Actividad economica'] == 'Educacion') & 
        (df['Periodo_num'] >= 2016) & 
        (df['Periodo_num'] <= 2025)
    ]
    
    # Seleccionar las columnas solicitadas
    columnas = ['Periodo', 'Trimestre', 'Actividad economica', 'PIB a precios constantes del 2015']
    
    # Asegurar de que las columnas existan, ignorando posibles diferencias de mayúsculas en el archivo original
    columnas_existentes = []
    for col in columnas:
        # Encontrar el nombre real de la columna para evitar errores por espacios o mayúsculas
        match = [c for c in df.columns if c.strip().lower() == col.lower()]
        if match:
            columnas_existentes.append(match[0])
        else:
            print(f"Advertencia: No se encontró la columna '{col}'.")
            
    df_final = df_filtered[columnas_existentes]
    
    # Guardar a un nuevo archivo
    df_final.to_csv(output_file, index=False)
    print(f"Datos guardados exitosamente en {output_file}")
    print(f"Total de registros procesados: {len(df_final)}")
    
    return df_final

if __name__ == "__main__":
    clean_data()
