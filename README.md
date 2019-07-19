# ECI2019-Competencia_Despegar
Código utilizado por el participante Cristian Yones, en la competencia organizada por Despegar en el marco de la 33ra
Escuela de Ciencias de la Informática 2019.

Para ejecutar el código, en primer lugar es necesario construir la imágen de docker, con

```bash
docker build -t eci2019 .
```
## Preparar las imágenes
Las imágenes utilizadas para entrenamiento y prueba deben estar en formato jpg, libre de errores. Para comprobar errores
en las imágenes se puede utilizar el script repair_images.py. A veces, es necesario ejecutar el script más de una vez
para corregir todos los errores en lás imágenes. Las imágenes de entrenamiento y prueba se deben guardar en dos carpetas
llamadas train y test, respectivamente. Las dos carpetas se deben colocar dentro de una misma carpeta. Por ejemplo,
podemos colocar las imágenes de entrenamiento en ./images/train y las de prueba en ./images/test.
Es importante aclarar ademas que el código está preparado para realizar inferencias en el conjunto de imágenes final de
la competencia, con cualquier otro conjunto de imaǵenes distinto probablemente falle. Para realizar inferencias en otros
conjuntos de datos ver el archivo eval_test.py

## Entrenamiento de modelos
Para ejecutar el contenedor con soporte para GPU es necesario utilizar nvidia-docker en lugar del tradicional docker. El
comando completo es

```bash
nvidia-docker run --ipc=host --mount source=<path_a_carpeta_de_imágenes>,target=/images,type=bind eci2019
```

## Utilizar modelos ya entrenados

El entrenamiento de los modelos tarda aproximadamente 20 horas, y la inferencia sobre los datos de test 1 hora más. Para
ahorrar el tiempo de entrenamiento, podemos descargar los modelos ya entrenados
desde [aqui](https://drive.google.com/drive/folders/1OdR6JMtI4_f-SdSQuRaxsQBVsYrCiy7E?usp=sharing).

Y ejecutar el contenedor pasandole la carpeta con estos con el siguiente comando:

```bash
nvidia-docker run --ipc=host --mount source=<carpeta_de_imágenes>,target=/images,type=bind \
		             --mount source=<carpeta_con_modelos>,target=/models,type=bind eci2019
```
