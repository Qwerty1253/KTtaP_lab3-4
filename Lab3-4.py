import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys
import io

# Зміна кодування виводу на UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Генерація випадкових даних для таблиці 5.2
np.random.seed(42)  # Фіксація випадковості для відтворюваності
data = {
    'Змінна 1': np.random.rand(30),
    'Змінна 2': np.random.rand(30),
    'Змінна 3': np.random.randint(1, 100, 30),
    'Змінна 4': np.random.randint(1, 50, 30),
    'Змінна 5': np.random.rand(30) * 100,
    'Змінна 6': np.random.randint(1, 200, 30)
}

# Створення DataFrame
df = pd.DataFrame(data)
print("Оригінальні дані:")
print(df)

# Збереження даних у файл Excel
df.to_excel('data.xlsx', index=False)

# Нормування даних
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

# Створення нового DataFrame з нормованими даними
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
print("\nНормовані дані:")
print(normalized_df)

# Збереження нормованих даних у файл Excel
normalized_df.to_excel('normalized_data.xlsx', index=False)

# Завантаження датасету Iris для демонстрації алгоритму Decision Tree
iris = load_iris()
X, y = iris.data, iris.target

# Розділення даних на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Налаштування параметрів алгоритму Decision Tree
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}

# Налаштування моделі за допомогою GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Найкращі параметри
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Найкращі параметри: {best_params}")
print(f"Найкраща точність на крос-валідації: {best_score:.2f}")

# Навчання моделі з найкращими параметрами
clf = DecisionTreeClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)

# Прогнозування та оцінка точності
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі на тестовому наборі: {accuracy:.6f}")
