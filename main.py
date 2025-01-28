# Importación de bibliotecas necesarias
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Definir los directorios donde están tus imágenes
train_dir = '/path/to/train'  # Ruta para las imágenes de entrenamiento
validation_dir = '/path/to/validation'  # Ruta para las imágenes de validación

# Preprocesamiento de las imágenes con ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizar las imágenes
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Cargar las imágenes desde los directorios
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Ajustar el tamaño de las imágenes
    batch_size=32,
    class_mode='binary')  # Cambiar a 'categorical' si tienes más de dos clases

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Cargar el modelo preentrenado MobileNetV2 sin la capa superior (top)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas del modelo base
base_model.trainable = False

# Construir el modelo agregando nuevas capas
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # Usar 'softmax' para más de dos clases
])

# Compilar el modelo
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento inicial del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Ajustar el número de épocas según sea necesario
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Descongelar algunas capas del modelo base para realizar el fine-tuning
base_model.trainable = True
fine_tune_at = 100  # Especificar desde qué capa hacer el fine-tuning

# Congelar las primeras capas y dejar entrenables las siguientes
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompilar el modelo con una tasa de aprendizaje más baja
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Continuar el entrenamiento con fine-tuning
history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,  # Ajustar el número de épocas
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Evaluar el modelo en el conjunto de validación
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc}")

# Realizar predicciones en el conjunto de validación
predictions = model.predict(validation_generator)
