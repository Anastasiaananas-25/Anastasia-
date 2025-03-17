import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Загрузка данных
df = pd.read_csv('датасет.csv', sep=';')

# Посмотрим на первые 7 строк
print(df.head())

# Проверяем типы данных
print(df.dtypes)

# Преобразование типа данных в 'price' если это необходимо
df['price'] = df['price'].str.replace(',', '.')
df['price'] = df['price'].astype(float)

# Визуализация данных
plt.scatter(df.area, df.price, color='red')
plt.xlabel('площадь(кв.м.)')
plt.ylabel('стоимость(млн.руб)')
plt.show()

# Создание модели
reg = linear_model.LinearRegression()

# Обучение модели
reg.fit(df[['area']], df.price)

# Предсказание цены квартиры
print(reg.predict([[38]]))  # Предсказание для квартиры 38 м2
print(reg.predict(df[['area']]))  # Предсказанные цены для всех квартир

# Получение коэффициентов модели
print(reg.coef_)  # a
print(reg.intercept_)  # b

# Построение линии регрессии
plt.scatter(df.area, df.price, color='red')
plt.xlabel('площадь(кв.м.)')
plt.ylabel('стоимость(млн.руб)')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()

# Загружаем новый файл с данными для предсказания
pred = pd.read_csv('prediction_price.csv', sep=';')

# Преобразуем данные аналогичным образом
pred['price'] = pred['price'].str.replace(',', '.')
pred['price'] = pred['price'].astype(float)

# Предсказание цен для новых данных
p = reg.predict(pred[['area']])

# Добавление новой колонки с предсказанными ценами
pred['predicted prices'] = p

# Вывод результата на экран
print(pred)

# Сохранение результата в формате Excel
pred.to_excel('new.xlsx', index=False)  # сохраняем файл в Excel без первой колонки

