import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# 1. Cargar los datos de la UJED
df = pd.read_csv('alumnos_ujed_actualizado.csv')
print("--- Datos Originales ---")
print(df.head())

# 2. Preprocesamiento: Convertir texto (Sexo y Trabaja) a números
le = LabelEncoder()
df['Sexo_N'] = le.fit_transform(df['Sexo'])        
df['Trabaja_N'] = le.fit_transform(df['Trabaja'])  

print("\n--- Datos Preprocesados (Listos para ML) ---")
print(df[['Nombre', 'Sexo', 'Sexo_N', 'Trabaja', 'Trabaja_N', 'Matematicas']])

# 3. SIMULACIÓN: Añadir la variable objetivo futura (Target)
df['Reprobo_Estructura_Datos'] = [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]  # 1 = Reprobó en el futuro, 0 = Aprobó en el futuro

# 4. Seleccionar Características (X) y Objetivo (y)
X = df[['Matematicas', 'Contabilidad_I', 'Habilidades_Pensamiento', 'Sexo_N', 'Trabaja_N']]
y = df['Reprobo_Estructura_Datos']

# 5. Entrenar el modelo (Árbol de Decisión)
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X, y)

print("\n¡Modelo entrenado con éxito!\n")

# 6. Mostrar las reglas que el modelo descubrió
reglas = export_text(modelo, feature_names=list(X.columns))
print("--- REGLAS DE DECISIÓN DEL MODELO ---")
print(reglas)

# 7. Evaluar al alumno de nuevo ingreso
# Perfil: Matemáticas=4.5, Contabilidad=8.0, Habilidades=7.5, Sexo_N=0, Trabaja_N=0
# 7. Evaluar al alumno de nuevo ingreso
# En lugar de una lista simple, lo envolvemos en un DataFrame con los mismos nombres de columnas
nuevo_alumno = pd.DataFrame([[3.0, 8.0, 7.5, 0, 1]], 
                            columns=['Matematicas', 'Contabilidad_I', 'Habilidades_Pensamiento', 'Sexo_N', 'Trabaja_N'])

# Realizar la predicción (Ahora sí, sin warnings)
prediccion = modelo.predict(nuevo_alumno)
probabilidad = modelo.predict_proba(nuevo_alumno)

# Realizar la predicción
prediccion = modelo.predict(nuevo_alumno)
probabilidad = modelo.predict_proba(nuevo_alumno)

print("--- RESULTADO PARA EL ALUMNO DE NUEVO INGRESO ---")
if prediccion[0] == 1:
    print("Resultado: ¡ALERTA DE RIESGO!")
    print("El modelo predice que este alumno REPROBARÁ 'Estructura de Datos' en el futuro.")
else:
    print("Resultado: Alumno a salvo.")
    print("El modelo predice que el alumno aprobará la materia futura.")

print(f"\nProbabilidad de aprobación: {probabilidad[0][0] * 100:.1f}%")
print(f"Probabilidad de reprobación: {probabilidad[0][1] * 100:.1f}%")