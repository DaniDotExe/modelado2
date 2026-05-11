import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Cargar el dataset Tecator (solo los 100 canales de absorbancia)
print("Cargando datos de Tecator...")
tecator = fetch_openml(data_id=505, as_frame=True, parser='auto')
X = tecator.data.iloc[:, :100]  # Tomamos estrictamente los 100 canales de luz
y = np.array(tecator.target).astype(float) # Porcentaje de grasa

# 2. División de datos y Estandarización
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenamiento: Regresión Lineal vs PLS (con 6 componentes para mayor suavidad)
# Modelo 1: Regresión Lineal
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Modelo 2: PLS
n_comp = 6 
pls = PLSRegression(n_components=n_comp)
pls.fit(X_train_scaled, y_train)
y_pred_pls = pls.predict(X_test_scaled)

# 4. Cálculo de precisiones (Métricas)
def calcular_metricas(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return r2, rmse

r2_lr, rmse_lr = calcular_metricas(y_test, y_pred_lr)
r2_pls, rmse_pls = calcular_metricas(y_test, y_pred_pls)

print("\n" + "="*40)
print(f"PRECISIONES EN EL SET DE PRUEBA")
print("="*40)
print(f"Regresión Lineal (100 variables):")
print(f"  - R2:   {r2_lr:.4f}")
print(f"  - RMSE: {rmse_lr:.2f}% de grasa")
print("-" * 40)
print(f"PLS ({n_comp} componentes):")
print(f"  - R2:   {r2_pls:.4f}")
print(f"  - RMSE: {rmse_pls:.2f}% de grasa")
print("="*40)

# 5. Gráficas de Coeficientes Separadas
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Gráfico Regresión Lineal
ax1.plot(lr.coef_, color='#d32f2f', linewidth=1)
ax1.set_title('Coeficientes: Regresión Lineal (Ruido Matemático)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Peso del coeficiente')
ax1.grid(alpha=0.3)

# Gráfico PLS
ax2.plot(pls.coef_.ravel(), color='#1976d2', linewidth=3)
ax2.set_title(f'Coeficientes: PLS con {n_comp} componentes (Huella Química Suave)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Peso del coeficiente')
ax2.set_xlabel('Longitud de onda (Canales 0-99)')
ax2.grid(alpha=0.3)
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout(pad=3.0)
plt.show()