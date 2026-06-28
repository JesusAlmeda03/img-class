import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# 1. Cargar tus datos de la UJED
df = pd.read_csv('alumnos_ujed_actualizado.csv')

# 2. Preprocesamiento: Convertir texto a números
le = LabelEncoder()
df['Sexo_N'] = le.fit_transform(df['Sexo'])
df['Trabaja_N'] = le.fit_transform(df['Trabaja'])

# 3. SIMULACIÓN: Añadimos la columna histórica "Reprobo_Estructura_Datos" (Nuestra Meta/Target)
# 1 = Reprobó en el futuro, 0 = Aprobó en el futuro
# Asignamos valores lógicos: los que tienen notas bajas o trabajan y tienen poco tiempo, suelen reprobar.
df['Reprobo_Estructura_Datos'] = [0, 1, 1, 0, 1, 0, 0, 0, 1, 0]

# 4. Seleccionar Características (X) y Objetivo (y)
# Usaremos: Matemáticas, Contabilidad, Habilidades, Sexo_N y Trabaja_N
X = df[['Matematicas', 'Contabilidad_I', 'Habilidades_Pensamiento', 'Sexo_N', 'Trabaja_N']]
y = df['Reprobo_Estructura_Datos']

# 5. Entrenar el modelo (Árbol de Decisión)
# Usamos un árbol porque nos permite entender las "reglas" que la IA inventó
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X, y)

print("¡Modelo entrenado con éxito!\n")

# 6. Mostrar las reglas que el modelo descubrió
reglas = export_text(modelo, feature_names=list(X.columns))
print("--- REGLAS DE DECISIÓN DEL MODELO ---")
print(reglas)