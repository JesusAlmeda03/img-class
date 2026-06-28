import numpy as np

# 1. Definimos el perfil del nuevo alumno según tus requerimientos:
# - No trabaja ('No' -> se convierte en 0)
# - No es bueno en matemáticas (le asignamos un 4.5)
# - Agregamos notas intermedias/buenas en las otras materias para ver el peso de las matemáticas
# - Supondremos que es mujer ('F' -> se convierte en 0)

# El orden exacto de las columnas que tu modelo espera es:
# ['Matematicas', 'Contabilidad_I', 'Habilidades_Pensamiento', 'Sexo_N', 'Trabaja_N']
nuevo_alumno = [[4.5, 8.0, 7.5, 0, 0]]

# 2. El modelo realiza la predicción (Inferencia)
prediccion = modelo.predict(nuevo_alumno)
probabilidad = modelo.predict_proba(nuevo_alumno)

print("--- RESULTADO PARA EL ALUMNO DE NUEVO INGRESO ---")
if prediccion[0] == 1:
    print("Resultado: ¡ALERTA DE RIESGO!")
    print("El modelo predice que este alumno REPROBARÁ 'Estructura de Datos' en el futuro.")
else:
    print("Resultado: Alumno a salvo.")
    print("El modelo predice que el alumno aprobará la materia futura.")

# 3. Ver el porcentaje de certeza del modelo
print(f"\nProbabilidad de aprobación: {probabilidad[0][0] * 100:.1f}%")
print(f"Probabilidad de reprobación: {probabilidad[0][1] * 100:.1f}%")