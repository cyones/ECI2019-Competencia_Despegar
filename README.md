# ECI2019-Competencia_Despegar
Código utilizado por el participante Cristian Yones, en la competencia organizada por Despegar en el marco de la 33ra Escuela de Ciencias de la Informática 2019.

Para ejecutar el código, en primer lugar es necesario construir la imágen de docker, con

docker build -t eci2019 .

Luego, se deben colocar las imágenes de entrenamiento y prueba en la carpeta images/train y images/test respectivamente.
Estas imágenes deben estar en formato jpg, libre de errores. Para comprobar errores en las imágenes se puede utilizar el
script repair_images.py. A veces, es necesario ejecutar el script más de una vez para corregir todos los errores en lás
imágenes.

Para ejecutar el contenedor con soporte para GPU es necesario utilizar nvidia-docker en lugar del tradicional docker. El
comando completo es

nvidia-docker run --ipc=host --mount source=<path_a_carpeta_de_imágenes>,target=/images,type=bind eci2019

El entrenamiento de los modelos tarda aproximadamente 20 horas, y la inferencia sobre los datos de test 1 hora más. Para
ahorrar el tiempo de entrenamiento, podemos descargar los modelos ya entrenados de:

<subiendo_modelos>

Y ejecutar el contenedor pasandole la carpeta con estos con el siguiente comando:

nvidia-docker run --ipc=host --mount source=<carpeta_de_imágenes>,target=/images,type=bind \
		             --mount source=<carpeta_con_modelos>,target=/models,type=bind eci2019
