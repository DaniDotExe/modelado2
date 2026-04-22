import csv
from collections import Counter

def calcular_esperanza(archivo_csv):
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
    
    # La esperanza matemática empírica es la suma de (x * P(x))
    esperanza_empirica = 0.0
    
    print("Cálculo de la Esperanza Matemática Empírica: E(x) = Σ [ x * P(x) ]")
    print("-" * 65)
    print(f"{'Suma (x)':<10} | {'Prob. Empírica P(x)':<20} | {'Cálculo (x * P(x))'}")
    print("-" * 65)

    for suma in range(2, 13):
        frecuencia = frecuencias.get(suma, 0)
        # Probabilidad empírica
        prob_empirica = frecuencia / total_lanzamientos
        
        # Multiplicación de Suma (x) por Probabilidad Empírica P(x)
        valor_esperado_parcial = suma * prob_empirica
        esperanza_empirica += valor_esperado_parcial
        
        # Formateamos a 2 decimales para que coincida con el código de probabilidad
        print(f"{suma:<10} | {prob_empirica:<20.2f} | {suma} * {prob_empirica:.2f} = {valor_esperado_parcial:.2f}")

    print("-" * 65)
    print(f"Esperanza Matemática Empírica E(x): {esperanza_empirica:.2f}")

if __name__ == "__main__":
    calcular_esperanza('100.csv')
