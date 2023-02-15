![HenryLogo](https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png)

​
# <h1 align="center">**`Proyecto Machine Learning - Estancia Hospitalaria`**

<p align="center">
<img src="https://www.ibm.com/blogs/client-voices/wp-content/uploads/2019/09/Glinnt.jpg"   
>
</p>

 
## 🏥 **Estancia hospitalaria** 🏥

La hospitalización, o estancia hospitalaria, cuando es prolongada constituye una preocupación a nivel mundial debido a sus efectos negativos en el sistema de salud, aumentando los costos, generando deficiencia en la accesibilidad de prestación de servicios de salud, saturación de unidades de hospitalización y urgencias, por consiguiente, mayores efectos adversos como lo son las enfermedades intrahospitalarias.

El estudio de los procesos de atención en salud, así como el conocimiento de las características y perfiles de los usuarios con el objetivo de predecir la ocupación hospitalaria, es uno de los aspectos al que las autoridades de salud han prestado gran interés, pues permite no sólo garantizar los recursos necesarios para la atención del paciente, sino realizar ajustes respecto a la oferta y demanda de los servicios de salud y los implementos asociados.

 
## **Introducción**
En un principio, nos encontramos ante un problema de clasificación, tenemos que predecir ante nuevos registros de pacientes si su estadía va a ser larga o corta(Clasificación binaria). Estadia corta: cuando dura 8 o menos días, y larga se considera a las estadias que duran mas de 8 dias.

El objetivo de esta clasificacion es que en base a los registros historicos, podamos administrar la demanda de camas segun la condicion en la que llegan los pacientes recien ingresados. Para qué? Para mejorar la eficiencia en la prestacion de servicios de salud, disminuir los costos y la saturacion de hospitales, evitar enfermedades intrahospitalarias, que son todos efectos negativos derivados de las estancias hospitalarias prolongadas.
 
Tecnologías utilizadas: el proyecto fue desarrollado exclusivamente en Python con variadas librerías(Pandas, Numpy,Matplotlib Sklearn y Seaborn). 
***

## **EDA y Preprocesamiento de datos**
En esta parte, trabajamos en el notebook llamado 'Proyecto.ipynb', en el cual están detallados y en orden cada uno de los pasos del proceso. En primer lugar, ingestamos la data y obtuvimos un pantallazo(exploración de los datos) para empezar a entrar en confianza con la misma. Pudimos ver el tamaño de los datos(filas y columnas), en detalle cada una de las variables y el tipo de datos de cada columna, y un resumen estadístico de las variables numericas.

Seguimos con la búsqueda y el tratamiento de valores nulos(no había), de valores duplicados(tampoco había) y de los valores outliers que los detectamos a simple vista con graficos como el de boxplot(de caja y bigotes/brazos).

Escalamos(con el metodo StandardScaler) algunas variables y normalizamos las categóricas con métodos como labelEncoder y OrdinalEncoder según si estas eran nominales u ordinales con el fin de obtener un dataframe con exclusivamente números. Tambien renombramos algunas columnas para facilitar el manejo de las variables.

Visualizacion de datos: Mediante graficos countplot de la librería seaborn, vimos la distribucion de algunas variables respecto a la variable target y sacamos buenas conclusiones como por ejemplo: en el area/departamento de anestesia y cirugía solo se presentaban casos de estadias largas, había 2 de los doctores(Simon y Isaac) que a todos los pacientes que atendian derivaban en una estadia larga. Tambien, que los de edad>50 años terminaban siendo siempre casos de estadia larga y otras conclusiones mas que nos ayudaron o guiaron para ver qué variables podían pesar mas o tener mas importancia a la hora de realizar un modelo de machine learning.

 En este gráfico observamos que los pacientes con más de 50 años derivan siempre en una estadía larga.
 [![imagen-2023-02-15-012352804.png](https://i.postimg.cc/MT56zWxd/imagen-2023-02-15-012352804.png)](https://postimg.cc/Fdf5ZXYL)
 
Utilizamos otros gráficos como Pairplot y la matriz de Correlación que tambien te permiten observar a simple vista, algunas relaciones y la fuerza de esas relaciones entre variables.

Finalmente decidí utilizar un dataframe con estas columnas:(['habitaciones_disponibles', 'area', 'doctor', 'personal_disponibles', 'visitas','seguro', 'deposito', 'target', 'gravedad_enc', 'edad_enc']

***

## **Machine Learning**
Ante el problema de clasificación, decidimos utlizar algunos modelos los cuales fuimos probando y observando como actuaban al predecir. En un principio, empezamos con modelos mas simples(arbol de decision, regresion logistica) y fuimos avanzando con modelos mas complejos con los cuales nos quedamos.

### **Random Forest:**
 Este fue el modelo definitivo, lo elegimos porque una de sus ventajas es que maneja bien hasta miles de variables e identifica las mas importantes, siendo un metodo de reduccion de dimensionalidad. La separación de nuestra data la realizamos con train_test_split.

#### **Parámetros**

- La cantidad de arboles que va a tener el bosque y elegí 100 que es un buen valor por defecto(n_estimators). 

- n_jobs = -1 que indica que va a utilizar tantos cores como tiene la máquina.

- random_state=42 que es el tipo de aleatoriedad
- max_features='sqrt' que se refiere a tomar las n_features que tengas.

 Además podiamos jugar con otros parametros como max_depth, min_samples_split y min_samples_leaf.


El score del bosque(porque son varios arboles) sobre la importancia de las variables, es un promedio que se normaliza a partir de la desviación estandar. En mi caso decidí mostrar la importancia de los gráficos tambien en un grafico de barras hecho en seaborn(librería de Python).Adjunto la imagen donde se pueden aprenciar las variables más importantes.


[![imagen-2023-02-14-181503894.png](https://i.postimg.cc/Ssds3yJ7/imagen-2023-02-14-181503894.png)](https://postimg.cc/627tGxVy)

**Modelos Extra**

1- Modelo de ensamble Bagging para el arbol de decision : decidimos utilizar este porque es un conjunto de modelos de ML, que se combinan para obtener una unica prediccion y su principal ventaja es que al ser diferentes modelos, los errores tienden a compensarse y obteniendo asi un mejor error de generalizacion.


2- Por ultimo, utilizamos un Arbol de decision con max_depth=10 y tambien aplicamos el GridSearch para ajustar parámetros.
 
 
## **Métrica a utilizar**

Como método de evaluación del desempeño del modelo, se utilizará la métrica de Exhaustividad (Recall) para las estadías hospitalarias largas, a partir de la matriz de confusión (Confusion Matrix). Cómo métrica adicional utilizamos la precisión(accuracy) para verificar el desempeño del modelo.
 
 $$ Recall=\frac{TP}{TP+FN}$$
 
 Donde $TP$ son los verdaderos positivos y $FN$ los falsos negativos.
 
 $$ Accuracy=\frac{TP+TN}{P+N}$$

Siendo $TP$ los verdaderos positivos, $TN$ verdaderos negativos y $P+N$ población total.

## **Resultado obtenido**
 
 Para finalizar y plasmar los resultados conseguidos, generamos un archivo .csv con las predicciones, el cual contiene una sola columna con todos los valores que predecimos. Con el mismo, obtuvimos un accuracy de 0.72 y un recall de 0.74

 [![Whats-App-Image-2023-02-14-at-6-04-49-PM.jpg](https://i.postimg.cc/6pMMWZgD/Whats-App-Image-2023-02-14-at-6-04-49-PM.jpg)](https://postimg.cc/qNCsQNkL)























