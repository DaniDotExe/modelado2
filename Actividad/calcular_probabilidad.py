import csv
from collections import Counter

def calcular_probabilidad_empirica(archivo_csv):
    total_lanzamientos = 0
    sumas = []

    # Leer el archivo CSV
    with open(archivo_csv, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sumas.append(int(row['suma']))
            total_lanzamientos += 1

    # Contar la frecuencia de cada suma
    frecuencias = Counter(sumas)

    print(f"Total de lanzamientos: {total_lanzamientos}")
    print(f"{'Suma':<10} | {'Frecuencia':<15} | {'Probabilidad Empírica'}")
    print("-" * 55)

    # Calcular y mostrar la probabilidad empírica para cada posible suma (del 2 al 12)
    for suma in range(2, 13):
        frecuencia = frecuencias.get(suma, 0)
        # Dividir la cantidad de veces que salió cada suma entre el total de lanzamientos
        prob_empirica = frecuencia / total_lanzamientos
        
        print(f"{suma:<10} | {frecuencia:<15} | {prob_empirica:.2f}")

if __name__ == "__main__":
    calcular_probabilidad_empirica('100.csv')
