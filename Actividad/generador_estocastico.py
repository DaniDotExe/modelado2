import csv
import random

def generar_datos_estocasticos(nombre_archivo, num_lanzamientos=100):
    # Valores posibles para la suma
    sumas_posibles = list(range(2, 13))
    
    # Pesos (numeradores de las probabilidades dadas) para x(t) = 2...12
    # 2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36, 7: 6/36
    # 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
    pesos_probabilidad = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1] 

    # Mapeo de cada suma a todas las combinaciones posibles de dos dados (dado1, dado2)
    # que dan como resultado esa suma exacta.
    combinaciones_por_suma = {
        2: [(1, 1)],
        3: [(1, 2), (2, 1)],
        4: [(1, 3), (2, 2), (3, 1)],
        5: [(1, 4), (2, 3), (3, 2), (4, 1)],
        6: [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)],
        7: [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)],
        8: [(2, 6), (3, 5), (4, 4), (5, 3), (6, 2)],
        9: [(3, 6), (4, 5), (5, 4), (6, 3)],
        10: [(4, 6), (5, 5), (6, 4)],
        11: [(5, 6), (6, 5)],
        12: [(6, 6)]
    }

    with open(nombre_archivo, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['tiro', 'dado1', 'dado2', 'suma'])

        for tiro in range(1, num_lanzamientos + 1):
            # 1. Seleccionar la suma aleatoriamente, usando una distribución ponderada (ruleta rusa)
            # de acuerdo a las probabilidades teóricas.
            suma_elegida = random.choices(sumas_posibles, weights=pesos_probabilidad, k=1)[0]
            
            # 2. Una vez que sabemos la suma, seleccionamos al azar qué combinación 
            # de dados formó esa suma.
            dado1, dado2 = random.choice(combinaciones_por_suma[suma_elegida])
            
            # 3. Guardamos esto en nuestro archivo
            writer.writerow([tiro, dado1, dado2, suma_elegida])

if __name__ == "__main__":
    generar_datos_estocasticos('100.csv', num_lanzamientos=100)
    print("El archivo '100.csv' se ha generado correctamente basado en las probabilidades teóricas.")
