import tensorflow as tf
import tensorflow_datasets as tfds

datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)

tfds.show_examples(datos['train'], metadatos)

import cv2

datos_entrenamiento = []
TAMANO_IMG = 100
for i, (imagen, etiqueta) in enumerate(datos['train']): #Todos los datos
  imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
  imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
  imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1) #Cambiar tamano a 100,100,1
  datos_entrenamiento.append([imagen, etiqueta])

imagenes_entrada = [] #pixeles de las imagenes
etiquetas = [] #gato o perro

for imagen, etiqueta in datos_entrenamiento:
  imagenes_entrada.append(imagen)
  etiquetas.append(etiqueta)

#Normalizar los datos de las X (imagenes). Se pasan a numero flotante y dividen entre 255 para quedar de 0-1 en lugar de 0-255
import numpy as np

imagenes_entrada = np.array(imagenes_entrada).astype(float) / 255
#Convertir etiquetas en arreglo simple
etiquetas = np.array(etiquetas)

#Realizar el aumento de datos con varias transformaciones. Al final, graficar 10 como ejemplo
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(imagenes_entrada)

#Separar los datos de entrenamiento y los datos de pruebas en variables diferentes

imagenes_entrada_entrenamiento = imagenes_entrada[:19700]
imagenes_entrada_validacion = imagenes_entrada[19700:]

etiquetas_entrenamiento = etiquetas[:19700]
etiquetas_validacion = etiquetas[19700:]

#Usar la funcion flow del generador para crear los datos de entrenamiento para la funcion FIT del modelo
data_gen_entrenamiento = datagen.flow(imagenes_entrada_entrenamiento, etiquetas_entrenamiento, batch_size=32)

#modelo convolucional
modeloCNN2_AD = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(250, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
modeloCNN2_AD.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

#importar tensorboar para visualizar los resultados de la ia
from tensorflow.keras.callbacks import TensorBoard
tensorboardCNN = TensorBoard(log_dir='logs/cnn')

#entrenar el modelo
modeloCNN2_AD.fit(
    data_gen_entrenamiento,
    epochs=200, batch_size=32,
    validation_data=(imagenes_entrada_validacion, etiquetas_validacion),
    steps_per_epoch=int(np.ceil(len(imagenes_entrada_validacion) / float(32))),
    validation_steps=int(np.ceil(len(etiquetas_validacion) / float(32))),
    callbacks=[tensorboardCNN]
)

# Commented out IPython magic to ensure Python compatibility.
#Cargar la extension de tensorboard de colab
# %load_ext tensorboard
#Ejecutar tensorboard e indicarle que lea la carpeta "logs"
# %tensorboard --logdir logs

#guardar el modelo
modeloCNN2_AD.save('perros-gatos-cnn-ad.h5')
!pip install tensorflowjs
!mkdir carpeta_salida
!tensorflowjs_converter --input_format keras perros-gatos-cnn-ad.h5 carpeta_salida