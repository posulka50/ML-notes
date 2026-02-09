# Missing Values (Пропущені дані)

## Що це?

Пропущені дані (Missing Values) — це відсутні значення в наборі даних, позначені як NaN (Not a Number), None, null або інші спеціальні значення.

## Чому виникають? 🤔

- 📝 Помилки при введенні даних
- 🔧 Технічні проблеми при збиранні
- 🙅 Респонденти не відповіли на питання
- 🔄 Об'єднання різних джерел даних
- ⏳ Дані ще не доступні

---

## Типи пропущених даних

### 1. MCAR (Missing Completely At Random)

**Випадкові пропуски без зв'язку з даними**
- Приклад: технічна помилка датчика
- Найпростіший випадок
- Безпечно видаляти

### 2. MAR (Missing At Random)

**Пропуски залежать від інших змінних**
- Приклад: чоловіки рідше вказують вагу
- Можна моделювати залежності
- Потрібна обережність при видаленні

### 3. MNAR (Missing Not At Random)

**Пропуски залежать від самих відсутніх значень**
- Приклад: люди з низьким доходом не вказують зарплату
- Найскладніший випадок
- Видалення може призвести до bias

---

## Виявлення пропусків

### Перевірка наявності

```python
import pandas as pd
import numpy as np

# Підрахунок пропусків
print(df.isnull().sum())

# Відсоток пропусків
print(df.isnull().sum() / len(df) * 100)

# Візуалізація
import missingno as msno
msno.matrix(df)
```

### Детальний аналіз

```python
# Створення звіту
missing_report = pd.DataFrame({
    'Кількість': df.isnull().sum(),
    'Відсоток': df.isnull().sum() / len(df) * 100
})
missing_report = missing_report[missing_report['Кількість'] > 0]
print(missing_report.sort_values('Відсоток', ascending=False))
```

### Візуалізація патернів

```python
import missingno as msno
import matplotlib.pyplot as plt

# Матриця пропусків
msno.matrix(df)

# Heatmap кореляцій пропусків
msno.heatmap(df)

# Dendrogram для групування
msno.dendrogram(df)
```

---

## Методи обробки

## 1. Видалення (Deletion)

### A) Видалення рядків

```python
# Видалити всі рядки з будь-яким NaN
df_clean = df.dropna()

# Видалити рядки, де всі значення NaN
df_clean = df.dropna(how='all')

# Видалити рядки з NaN у конкретних стовпцях
df_clean = df.dropna(subset=['вік', 'зарплата'])

# Поріг: залишити рядки з мінімум N не-NaN значень
df_clean = df.dropna(thresh=5)
```

### B) Видалення стовпців

```python
# Видалити стовпці з будь-яким NaN
df_clean = df.dropna(axis=1)

# Видалити стовпці з >50% пропусків
threshold = 0.5
df_clean = df.loc[:, df.isnull().mean() < threshold]
```

### ⚠️ Коли використовувати?

✅ Малий відсоток пропусків (< 5%)
✅ MCAR тип пропусків
✅ Багато даних (можна дозволити втрату)

❌ Великий відсоток пропусків
❌ MAR/MNAR типи (ризик bias)

---

## 2. Imputation (Заповнення)

### A) Константою

```python
# Заповнити нулями
df['вік'].fillna(0, inplace=True)

# Заповнити власним значенням
df['категорія'].fillna('Невідомо', inplace=True)

# Заповнити різними значеннями для різних стовпців
df.fillna({
    'вік': 0,
    'зарплата': df['зарплата'].median(),
    'місто': 'Інше'
}, inplace=True)
```

### B) Статистичними показниками

#### Середнє (Mean)

```python
# Pandas
df['вік'].fillna(df['вік'].mean(), inplace=True)

# Scikit-learn
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df[['вік', 'зарплата']] = imputer.fit_transform(df[['вік', 'зарплата']])
```

#### Медіана (Median)

```python
# Pandas
df['ціна'].fillna(df['ціна'].median(), inplace=True)

# Scikit-learn
imputer = SimpleImputer(strategy='median')
df[['ціна']] = imputer.fit_transform(df[['ціна']])
```

#### Мода (Most Frequent)

```python
# Pandas
df['категорія'].fillna(df['категорія'].mode()[0], inplace=True)

# Scikit-learn
imputer = SimpleImputer(strategy='most_frequent')
df[['категорія']] = imputer.fit_transform(df[['категорія']])
```

### Порівняння стратегій

| Стратегія | Тип даних | Переваги | Недоліки |
|-----------|-----------|----------|----------|
| Mean | Числові | Проста | Чутлива до викидів |
| Median | Числові | Робастна до викидів | Може не відображати розподіл |
| Mode | Категоріальні | Найбільш вірогідне значення | Може бути багато мод |
| Константа | Будь-які | Повний контроль | Довільний вибір |

---

### C) Forward Fill / Backward Fill

**Для часових рядів**

```python
# Forward fill (попереднє значення)
df['температура'].fillna(method='ffill', inplace=True)

# Backward fill (наступне значення)
df['температура'].fillna(method='bfill', inplace=True)

# З обмеженням кількості заповнень
df['температура'].fillna(method='ffill', limit=2, inplace=True)
```

### D) Інтерполяція

```python
# Лінійна інтерполяція
df['значення'].interpolate(method='linear', inplace=True)

# Поліноміальна
df['значення'].interpolate(method='polynomial', order=2, inplace=True)

# Для часових рядів
df['дата'] = pd.to_datetime(df['дата'])
df.set_index('дата', inplace=True)
df['значення'].interpolate(method='time', inplace=True)
```

---

### E) KNN Imputation

**Заповнення на основі k найближчих сусідів**

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

### Як працює?

1. Для кожного пропуску знаходить k найближчих рядків (без пропусків)
2. Обчислює середнє значення з цих k сусідів
3. Заповнює пропуск цим середнім

### Коли використовувати?

✅ Дані мають сильні кореляції між ознаками
✅ Невеликі/середні датасети (обчислювально дорого)
✅ Складні залежності між змінними

❌ Дуже великі датасети (повільно)
❌ Категоріальні дані (потребують кодування)

---

### F) Iterative Imputation (MICE)

**Multiple Imputation by Chained Equations**

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    max_iter=10,
    random_state=42
)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

### Як працює?

1. Початкове заповнення (median/mean)
2. Для кожної змінної:
   - Використовує інші змінні для предикції
   - Навчає модель регресії
   - Оновлює пропущені значення
3. Повторює цикл кілька разів

### Коли використовувати?

✅ Складні залежності між змінними
✅ MAR тип пропусків
✅ Потрібна висока точність

❌ Обчислювально дорого
❌ Можливий overfitting

---

## 3. Створення індикатора пропусків

### Чому важливо?

Сам факт пропуску може бути інформативним!

```python
# Створення бінарного індикатора
df['вік_відсутній'] = df['вік'].isnull().astype(int)

# Потім заповнення
df['вік'].fillna(df['вік'].median(), inplace=True)
```

### Приклад з MissingIndicator

```python
from sklearn.impute import MissingIndicator

indicator = MissingIndicator()
indicator_array = indicator.fit_transform(df)

# Додавання до основного датафрейму
df_with_indicators = pd.concat([
    df,
    pd.DataFrame(indicator_array, columns=[f'{col}_missing' 
                                           for col in df.columns])
], axis=1)
```

---

## Повний Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Завантаження даних
df = pd.read_csv('data.csv')

# 2. Аналіз пропусків
print("Пропуски до обробки:")
print(df.isnull().sum())

# 3. Розділення на числові та категоріальні
numeric_features = df.select_dtypes(include=[np.number]).columns
categorical_features = df.select_dtypes(include=['object']).columns

# 4. Розділення на train/test
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Imputation для числових змінних
numeric_imputer = SimpleImputer(strategy='median')
X_train[numeric_features] = numeric_imputer.fit_transform(
    X_train[numeric_features]
)
X_test[numeric_features] = numeric_imputer.transform(
    X_test[numeric_features]
)

# 6. Imputation для категоріальних змінних
categorical_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_features] = categorical_imputer.fit_transform(
    X_train[categorical_features]
)
X_test[categorical_features] = categorical_imputer.transform(
    X_test[categorical_features]
)

# 7. Перевірка
print("\nПропуски після обробки:")
print(X_train.isnull().sum())
print(X_test.isnull().sum())
```

---

## Pipeline з sklearn

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Визначення трансформерів для різних типів даних
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Об'єднання
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Повний pipeline з моделлю
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Навчання
full_pipeline.fit(X_train, y_train)

# Предикція
y_pred = full_pipeline.predict(X_test)
```

---

## Вибір методу: Decision Tree 🌳

```
                       Пропуски < 5%?
                      /              \
                   Так                Ні
                    |                  |
              Видалити          Тип даних?
                                /         \
                         Числові      Категоріальні
                             |               |
                      Викиди?         Most Frequent
                        /    \
                     Так      Ні
                      |        |
                  Median    Mean/KNN
```

---

## Поради та Best Practices 💡

### 1. Завжди аналізуйте патерни

```python
# Кореляція між пропусками
import seaborn as sns

missing_corr = df.isnull().corr()
sns.heatmap(missing_corr, annot=True)
```

### 2. Fit на train, transform на test

```python
# ✅ ПРАВИЛЬНО
imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

# ❌ НЕПРАВИЛЬНО
imputer.fit(X_test)  # Витік даних!
```

### 3. Документуйте рішення

```python
# Створюємо словник стратегій
imputation_strategy = {
    'вік': 'median',
    'зарплата': 'mean',
    'місто': 'most_frequent',
    'категорія': 'constant:Невідомо'
}

# Зберігаємо metadata
import json
with open('imputation_config.json', 'w') as f:
    json.dump(imputation_strategy, f)
```

### 4. Перевіряйте результати

```python
# Порівняння розподілів до/після
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# До imputation
df['вік'].dropna().hist(bins=30, ax=axes[0])
axes[0].set_title('До imputation')

# Після imputation
df_imputed['вік'].hist(bins=30, ax=axes[1])
axes[1].set_title('Після imputation')

plt.show()
```

### 5. Експериментуйте

```python
# Порівняння різних стратегій
from sklearn.model_selection import cross_val_score

strategies = ['mean', 'median', 'most_frequent']
results = {}

for strategy in strategies:
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X_train)
    
    score = cross_val_score(model, X_imputed, y_train, cv=5).mean()
    results[strategy] = score

print(results)
```

---

## Поширені помилки ❌

### 1. Витік даних

```python
# ❌ НЕПРАВИЛЬНО
imputer.fit(pd.concat([X_train, X_test]))

# ✅ ПРАВИЛЬНО
imputer.fit(X_train)
```

### 2. Ігнорування типу пропусків

```python
# Якщо MNAR, простий imputation може дати bias
# Краще використовувати індикатори або моделювання
```

### 3. Використання mean при викидах

```python
# ❌ Mean при викидах
df['ціна'].fillna(df['ціна'].mean())

# ✅ Median при викидах
df['ціна'].fillna(df['ціна'].median())
```

---

## Пов'язані теми

- [[01_Feature_Scaling]] — масштабування після imputation
- [[02_Categorical_Encoding]] — кодування категорій
- [[04_Outliers_Handling]] — робота з викидами

## Ресурси

- [Scikit-learn Imputation](https://scikit-learn.org/stable/modules/impute.html)
- [Missing Data: A Gentle Introduction](https://stefvanbuuren.name/fimd/)
- [Missingno Library](https://github.com/ResidentMario/missingno)

---

#ml #preprocessing #missing-values #imputation #datascience
