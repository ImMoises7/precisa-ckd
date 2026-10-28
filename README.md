\# PRECISA: Diagnóstico de ERC con Machine Learning



Proyecto de aula de Inteligencia Artificial, Ingeniería de Sistemas, Universidad Popular del Cesar. Grupo 03.



Sistema de clasificación del diagnóstico actual de enfermedad renal crónica (ERC) a partir de variables clínicas, comparando regresión logística (baseline), Random Forest y XGBoost.



\## Alcance



La predicción de progresión a 6 y 12 meses requiere datos longitudinales, que no están disponibles en los datasets públicos. Por eso este repositorio implementa diagnóstico actual; la predicción temporal se considera trabajo futuro.



\## Datos



Dataset Chronic Kidney Disease de la UCI Machine Learning Repository (400 pacientes, 24 variables y la etiqueta `class`). Se encuentra en `data/chronic\_kidney\_disease.arff`.



\## Instalación



Requiere Python 3.11.



```bash

python -m venv venv

venv\\Scripts\\activate          # Windows

source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

```



\## Ejecución



Los scripts se ejecutan en este orden:



1\. `python explorar.py`: carga y revisión inicial del dataset.

2\. `python fuga.py`: poder de separación de cada variable.

3\. `python preprocesamiento.py`: limpieza y partición 85/15 estratificada. Genera `data/procesado/`.

4\. `python entrenar\_regresion.py`: regresión logística con validación cruzada.

5\. `python sensibilidad.py`: prueba de robustez sin `hemo`, `pcv` y `sc`.

6\. `python comparar\_modelos.py`: comparación de los tres modelos.



\## Metodología



\- Partición 85 % entrenamiento / 15 % prueba, estratificada, con semilla 42.

\- Validación cruzada estratificada de 5 pliegues sobre el 85 %.

\- Imputación por mediana, escalado estándar y pesos de clase, ajustados dentro de cada pliegue mediante `Pipeline`.

\- Métricas: AUC, exactitud, F1, sensibilidad y especificidad.

\- El conjunto de prueba se reserva para la evaluación final.



\## Limitaciones



El dataset no documenta sus criterios diagnósticos, por lo que no puede garantizarse que la etiqueta sea independiente de las variables predictoras. Ver el análisis de `fuga.py` y `sensibilidad.py`.



\## Autores



Santiago Alfonso García Salazar, Nataly Sofia Camelo Hernandez, Moises Araujo.

