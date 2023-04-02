# методы для отрисовки изображений
#from PIL import Image

# Для отрисовки графиков
import matplotlib.pyplot as plt

# Для генерации случайных чисел
import random

# Библиотека работы с массивами
import numpy as np

# Для работы с файлами
import os

# для создания сети
from tensorflow.keras.models import Sequential

# для создания слоев
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, SpatialDropout2D

# для работы с изображениями
from tensorflow.keras.preprocessing import image

# оптимизатор
from tensorflow.keras.optimizers import Adam

# для формирования тестовой выборки
from sklearn.model_selection import train_test_split


# Папка с папками картинок, рассортированных по категориям
IMAGE_PATH = 'data'
# Определение списка имен классов
CLASS_LIST = sorted(os.listdir(IMAGE_PATH))
# Определение количества классов
CLASS_COUNT = len(CLASS_LIST)

train_files = []                           # Cписок путей к файлам картинок
train_labels = []                          # Список меток классов, соответствующих файлам

for class_label in range(CLASS_COUNT):    # Для всех классов по порядку номеров (их меток)
    class_name = CLASS_LIST[class_label]  # Выборка имени класса из списка имен
    class_path = IMAGE_PATH + class_name  # Формирование полного пути к папке с изображениями класса
    class_files = os.listdir(class_path)  # Получение списка имен файлов с изображениями текущего класса
    print(f'Размер класса {class_name} составляет {len(class_files)} фото')

    # Добавление к общему списку всех файлов класса с добавлением родительского пути
    train_files += [f'{class_path}/{file_name}' for file_name in class_files]

    # Добавление к общему списку меток текущего класса - их ровно столько, сколько файлов в классе
    train_labels += [class_label] * len(class_files)

print()
print('Общий размер базы для обучения:', len(train_labels))

# Задание высоты и ширины загружаемых изображений
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANELS = 1

# Пустой список для данных изображений
train_images = []

for file_name in train_files:
    # Открытие и смена размера изображения
    img = Image.open(file_name).resize((IMG_HEIGHT, IMG_WIDTH)).convert("L")
    img_np = np.array(img)                # Перевод в numpy-массив
    train_images.append(img_np)            # Добавление изображения в виде numpy-массива к общему списку

x = np.array(train_images).reshape(-1,128,128,1)            # Перевод общего списка изображений в numpy-массив
y = np.array(train_labels)            # Перевод общего списка меток класса в numpy-массив

print(f'В массив собрано {len(train_images)} фотографий следующей формы: {img_np.shape}')
print(f'Общий массив данных изображений следующей формы: {x.shape}')
print(f'Общий массив меток классов следующей формы: {y.shape}')

# Нормируем данные
x = x / 255.

# Разбиваем данные на тестовые и обучающие
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Задаем размер вхлдящего тензора
input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANELS)

# Создаем линейную модель CNN
model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(input_shape)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(SpatialDropout2D(0.5))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(SpatialDropout2D(0.6))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Компилируем модель, выбираем функцию ошибки и оптимизатор и шаг обучения
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])
# Запускаем обучение, определяем размер батча, количество эпох
store = model.fit(x_train, y_train, shuffle=True, batch_size=32, epochs=30, validation_split=0.2, verbose=1)
# Отрисовываем процесс и результат
plt.plot(store.history['accuracy'], label='Обучающая')
plt.plot(store.history['val_accuracy'], label='Проверочная')
plt.legend()
plt.title('Точность')
plt.show()
plt.plot(store.history['loss'], label='Обучающая')
plt.plot(store.history['val_loss'], label='Проверочная')
plt.legend()
plt.title('Ошибка')
plt.show()

# Компилируем модель, уменьшаем шаг обучения
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.00001),
              metrics=['accuracy'])
# Дообучаем модель определяем размер батча, количество эпох
store = model.fit(x_train, y_train, shuffle=True, batch_size=32, epochs=10, validation_split=0.2, verbose=1)
# Отрисовываем процесс и результат
plt.plot(store.history['accuracy'], label='Обучающая')
plt.plot(store.history['val_accuracy'], label='Проверочная')
plt.legend()
plt.title('Точность')
plt.show()
plt.plot(store.history['loss'], label='Обучающая')
plt.plot(store.history['val_loss'], label='Проверочная')
plt.legend()
plt.title('Ошибка')
plt.show()
# Компилируем модель, уменьшаем шаг обучения
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.000001),
              metrics=['accuracy'])
# Дообучаем модель определяем размер батча, количество эпох
store = model.fit(x_train, y_train, shuffle=True, batch_size=32, epochs=10, validation_split=0.2, verbose=1)
# Отрисовываем процесс и результат
plt.plot(store.history['accuracy'], label='Обучающая')
plt.plot(store.history['val_accuracy'], label='Проверочная')
plt.legend()
plt.title('Точность')
plt.show()
plt.plot(store.history['loss'], label='Обучающая')
plt.plot(store.history['val_loss'], label='Проверочная')
plt.legend()
plt.title('Ошибка')
plt.show()
# Запускаем проверку на тестовых данных, которые модель не видела ранее.
scores = model.evaluate(x_test, y_test, verbose=1)
# Отрисовываем результат
print(f"Результат проверочной базы: {round(store.history['val_accuracy'][-1] * 100)} %")
print(f'Результат тестовой базы: {round(scores[1] * 100)} %')
# Сохраняем результат
model.save('pneumonia_6_full_98%.h5')