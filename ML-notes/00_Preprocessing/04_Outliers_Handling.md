# Outliers Handling (Робота з викидами)

## Що таке викиди?

**Викиди (Outliers)** — це спостереження, які значно відрізняються від інших даних у наборі. Вони можуть бути результатом помилок або представляти рідкісні, але реальні події.

## Типи викидів

### 1. Univariate Outliers (Одновимірні)

**Екстремальні значення в одній змінній**
- Приклад: вік = 200 років
- Визначаються для кожної ознаки окремо

### 2. Multivariate Outliers (Багатовимірні)

**Незвичайна комбінація значень**
- Приклад: вік = 15, зарплата = $200,000
- Окремо значення нормальні, але разом — ні
- Важче виявити

---

## Чому виникають? 🤔

### Помилки (видалити!)

- 📝 Помилки вводу даних (1000 замість 100)
- 🔧 Збої в обладнанні
- 🐛 Баги в коді обробки
- 🔄 Помилки при об'єднанні даних

### Реальні події (зберегти!)

- 💎 Рідкісні, але легітимні випадки
- 🌟 VIP клієнти з надвисокими покупками
- 🏆 Виняткові досягнення
- 📊 Природна варіативність даних

---

## Виявлення викидів

## 1. Візуальні методи

### A) Box Plot (Ящик з вусами)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Простий box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='зарплата')
plt.title('Box Plot зарплати')
plt.show()

# Для багатьох змінних
df[['вік', 'зарплата', 'досвід']].boxplot(figsize=(12, 6))
plt.show()
```

**Як читати:**
- Коробка: 25%-75% квартилі (IQR)
- Лінія всередині: медіана
- Вуса: 1.5 × IQR
- Точки за вусами: **викиди**

### B) Scatter Plot (Діаграма розсіювання)

```python
# Для виявлення багатовимірних викидів
plt.figure(figsize=(10, 6))
plt.scatter(df['вік'], df['зарплата'], alpha=0.5)
plt.xlabel('Вік')
plt.ylabel('Зарплата')
plt.title('Співвідношення віку та зарплати')
plt.show()
```

### C) Histogram (Гістограма)

```python
plt.figure(figsize=(10, 6))
df['ціна'].hist(bins=50)
plt.xlabel('Ціна')
plt.ylabel('Частота')
plt.title('Розподіл цін')
plt.show()
```

### D) Violin Plot

```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, y='зарплата')
plt.title('Violin Plot зарплати')
plt.show()
```

---

## 2. Статистичні методи

### A) IQR Method (Метод міжквартильного розмаху)

**Найпопулярніший метод**

```python
import numpy as np

def detect_outliers_iqr(data, column):
    """
    Виявлення викидів методом IQR
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Індекси викидів
    outliers = data[(data[column] < lower_bound) | 
                    (data[column] > upper_bound)]
    
    return outliers, lower_bound, upper_bound

# Використання
outliers, lower, upper = detect_outliers_iqr(df, 'зарплата')
print(f"Нижня межа: {lower}")
print(f"Верхня межа: {upper}")
print(f"Кількість викидів: {len(outliers)}")
```

**Формула:**

```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

де IQR = Q3 - Q1
```

### B) Z-Score Method

**Для нормального розподілу**

```python
from scipy import stats

def detect_outliers_zscore(data, column, threshold=3):
    """
    Виявлення викидів методом Z-score
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    outliers = data[z_scores > threshold]
    
    return outliers

# Використання
outliers = detect_outliers_zscore(df, 'зарплата', threshold=3)
print(f"Кількість викидів (Z-score > 3): {len(outliers)}")
```

**Формула:**

```
Z = (x - μ) / σ

де:
- x: значення
- μ: середнє
- σ: стандартне відхилення
```

**Інтерпретація:**
- |Z| > 2: можливий викид (95% даних)
- |Z| > 3: сильний викид (99.7% даних)

### C) Modified Z-Score (Robust)

**Робастна версія для ненормальних розподілів**

```python
def detect_outliers_modified_zscore(data, column, threshold=3.5):
    """
    Modified Z-score використовує медіану замість середнього
    """
    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))
    
    modified_z_scores = 0.6745 * (data[column] - median) / mad
    outliers = data[np.abs(modified_z_scores) > threshold]
    
    return outliers

# Використання
outliers = detect_outliers_modified_zscore(df, 'ціна')
```

---

### D) Isolation Forest

**Machine Learning підхід**

```python
from sklearn.ensemble import IsolationForest

def detect_outliers_isolation_forest(data, columns, contamination=0.1):
    """
    Виявлення багатовимірних викидів
    """
    iso_forest = IsolationForest(
        contamination=contamination,  # Очікуваний % викидів
        random_state=42
    )
    
    # Предикція: 1 = нормальні, -1 = викиди
    predictions = iso_forest.fit_predict(data[columns])
    
    # Додавання результатів до датафрейму
    data['outlier'] = predictions
    outliers = data[data['outlier'] == -1]
    
    return outliers

# Використання
outliers = detect_outliers_isolation_forest(
    df, 
    columns=['вік', 'зарплата', 'досвід'],
    contamination=0.05
)
```

**Переваги:**
- ✅ Працює з багатовимірними даними
- ✅ Не припускає конкретний розподіл
- ✅ Швидкий

**Коли використовувати:**
- Багато ознак
- Складні залежності
- Невідомий розподіл

---

### E) Local Outlier Factor (LOF)

**На основі локальної щільності**

```python
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_lof(data, columns, n_neighbors=20):
    """
    LOF виявляє викиди на основі локальної щільності
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination='auto'
    )
    
    predictions = lof.fit_predict(data[columns])
    
    data['outlier_lof'] = predictions
    outliers = data[data['outlier_lof'] == -1]
    
    return outliers

# Використання
outliers = detect_outliers_lof(
    df, 
    columns=['вік', 'зарплата', 'досвід']
)
```

---

### F) DBSCAN Clustering

**Виявлення через кластеризацію**

```python
from sklearn.cluster import DBSCAN

def detect_outliers_dbscan(data, columns, eps=0.5, min_samples=5):
    """
    Точки, що не входять в жоден кластер = викиди
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data[columns])
    
    # -1 означає викид (noise)
    data['cluster'] = clusters
    outliers = data[data['cluster'] == -1]
    
    return outliers

# Використання
outliers = detect_outliers_dbscan(
    df,
    columns=['feature1', 'feature2']
)
```

---

## Обробка викидів

## 1. Видалення (Removal)

```python
# Видалення рядків з викидами
df_clean = df[~df.index.isin(outliers.index)]

# Або за умовою
Q1 = df['зарплата'].quantile(0.25)
Q3 = df['зарплата'].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[
    (df['зарплата'] >= Q1 - 1.5 * IQR) & 
    (df['зарплата'] <= Q3 + 1.5 * IQR)
]
```

### ⚠️ Коли використовувати?

✅ Викиди — це помилки
✅ Невеликий % викидів (< 1-5%)
✅ Багато даних

❌ Викиди реальні та інформативні
❌ Малий датасет

---

## 2. Capping/Clipping (Обрізання)

### Winsorization
**Заміна викидів на граничні значення**

```python
from scipy.stats.mstats import winsorize

# Обрізання 5% з кожного боку
df['зарплата_winsorized'] = winsorize(
    df['зарплата'], 
    limits=[0.05, 0.05]
)
```

### Вручну
```python
def cap_outliers(data, column, lower_percentile=5, upper_percentile=95):
    """
    Обрізає викиди до вказаних перцентилів
    """
    lower = data[column].quantile(lower_percentile / 100)
    upper = data[column].quantile(upper_percentile / 100)
    
    data[f'{column}_capped'] = data[column].clip(lower, upper)
    
    return data

# Використання
df = cap_outliers(df, 'ціна', lower_percentile=1, upper_percentile=99)
```

### За допомогою IQR
```python
def cap_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    data[f'{column}_capped'] = data[column].clip(
        lower_bound, 
        upper_bound
    )
    
    return data

df = cap_outliers_iqr(df, 'зарплата')
```

### Коли використовувати?
✅ Хочемо зберегти всі дані
✅ Викиди впливають на модель
✅ Розподіл важливий, але екстремуми — ні

---

## 3. Трансформація (Transformation)

### A) Log Transformation
**Для правоскошених розподілів**

```python
import numpy as np

# Log трансформація
df['ціна_log'] = np.log1p(df['ціна'])  # log1p = log(1 + x)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['ціна'].hist(bins=50, ax=axes[0])
axes[0].set_title('Оригінальні дані')

df['ціна_log'].hist(bins=50, ax=axes[1])
axes[1].set_title('Після log трансформації')

plt.show()
```

### B) Square Root Transformation
```python
df['значення_sqrt'] = np.sqrt(df['значення'])
```

### C) Box-Cox Transformation
**Автоматично підбирає найкращу трансформацію**

```python
from scipy import stats

# Box-Cox (тільки для позитивних значень)
transformed_data, lambda_param = stats.boxcox(df['ціна'])
df['ціна_boxcox'] = transformed_data

print(f"Оптимальна lambda: {lambda_param}")
```

### D) Yeo-Johnson Transformation
**Box-Cox, але працює з негативними значеннями**

```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='yeo-johnson')
df['ціна_yeojohnson'] = transformer.fit_transform(df[['ціна']])
```

### Коли використовувати?
✅ Скошений розподіл
✅ Нормальність важлива для моделі
✅ Хочемо зменшити вплив викидів, але зберегти їх

---

## 4. Окреме моделювання (Separate Treatment)

### Бінаризація
```python
# Створення індикатора "чи є викидом"
threshold = df['зарплата'].quantile(0.95)
df['висока_зарплата'] = (df['зарплата'] > threshold).astype(int)
```

### Створення нової категорії
```python
def categorize_by_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    def categorize(value):
        if value < lower:
            return 'нижче_норми'
        elif value > upper:
            return 'вище_норми'
        else:
            return 'норма'
    
    data[f'{column}_category'] = data[column].apply(categorize)
    
    return data

df = categorize_by_outliers(df, 'зарплата')
```

---

## 5. Робастні моделі

**Використання алгоритмів, стійких до викидів**

```python
# Дерева рішень (стійкі до викидів)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Робастна регресія
from sklearn.linear_model import HuberRegressor

model = HuberRegressor(epsilon=1.35)
model.fit(X_train, y_train)
```

### Робастні алгоритми:
- ✅ Random Forest
- ✅ Gradient Boosting (XGBoost, LightGBM)
- ✅ Huber Regression
- ✅ RANSAC

### Чутливі алгоритми:
- ❌ Linear Regression
- ❌ Logistic Regression
- ❌ SVM
- ❌ K-Nearest Neighbors

---

## Порівняння методів

| Метод | Переваги | Недоліки | Коли використовувати |
|-------|----------|----------|----------------------|
| Видалення | Просто | Втрата даних | Викиди = помилки |
| Capping | Зберігаємо дані | Штучні межі | Екстремуми неважливі |
| Трансформація | Зберігає порядок | Складна інтерпретація | Скошений розподіл |
| Робастні моделі | Не потребує обробки | Обмежений вибір | Багато викидів |

---

## Повний Workflow

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Завантаження даних
df = pd.read_csv('data.csv')

# 2. Візуальний аналіз
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Box plot
sns.boxplot(data=df, y='зарплата', ax=axes[0, 0])
axes[0, 0].set_title('Box Plot')

# Histogram
df['зарплата'].hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Histogram')

# Scatter
axes[1, 0].scatter(df['вік'], df['зарплата'], alpha=0.5)
axes[1, 0].set_title('Вік vs Зарплата')

# Q-Q plot
stats.probplot(df['зарплата'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# 3. Виявлення викидів (IQR)
Q1 = df['зарплата'].quantile(0.25)
Q3 = df['зарплата'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_mask = (df['зарплата'] < lower_bound) | (df['зарплата'] > upper_bound)
print(f"Кількість викидів: {outliers_mask.sum()}")
print(f"Відсоток викидів: {outliers_mask.sum() / len(df) * 100:.2f}%")

# 4. Візуалізація викидів
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['зарплата'], c=outliers_mask, cmap='coolwarm', alpha=0.6)
plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
plt.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
plt.xlabel('Індекс')
plt.ylabel('Зарплата')
plt.title('Виявлені викиди')
plt.legend()
plt.show()

# 5. Обробка (вибираємо метод)
# Варіант А: Видалення
df_no_outliers = df[~outliers_mask]

# Варіант Б: Capping
df_capped = df.copy()
df_capped['зарплата'] = df_capped['зарплата'].clip(lower_bound, upper_bound)

# Варіант В: Log трансформація
df_transformed = df.copy()
df_transformed['зарплата_log'] = np.log1p(df_transformed['зарплата'])

# 6. Порівняння результатів
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df_no_outliers['зарплата'].hist(bins=30, ax=axes[0])
axes[0].set_title('Після видалення викидів')

df_capped['зарплата'].hist(bins=30, ax=axes[1])
axes[1].set_title('Після capping')

df_transformed['зарплата_log'].hist(bins=30, ax=axes[2])
axes[2].set_title('Після log трансформації')

plt.tight_layout()
plt.show()

# 7. Вибір найкращого підходу
print("\n=== Статистика ===")
print(f"Оригінал - Mean: {df['зарплата'].mean():.2f}, Std: {df['зарплата'].std():.2f}")
print(f"Без викидів - Mean: {df_no_outliers['зарплата'].mean():.2f}, Std: {df_no_outliers['зарплата'].std():.2f}")
print(f"Capped - Mean: {df_capped['зарплата'].mean():.2f}, Std: {df_capped['зарплата'].std():.2f}")
```

---

## Best Practices 💡

### 1. Завжди візуалізуйте

```python
# Перед і після обробки
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(data=df, y='ціна', ax=axes[0])
axes[0].set_title('До обробки')

sns.boxplot(data=df_processed, y='ціна', ax=axes[1])
axes[1].set_title('Після обробки')
```

### 2. Розумійте контекст

**Не всі викиди потрібно видаляти!**
- Висока ціна будинку в елітному районі — це нормально
- Аномальна транзакція може бути шахрайством (важлива!)

### 3. Документуйте рішення

```python
# Створюємо звіт
outlier_report = {
    'метод_виявлення': 'IQR',
    'метод_обробки': 'capping',
    'кількість_викидів': outliers_mask.sum(),
    'відсоток': f"{outliers_mask.sum() / len(df) * 100:.2f}%",
    'нижня_межа': lower_bound,
    'верхня_межа': upper_bound
}

import json
with open('outlier_report.json', 'w') as f:
    json.dump(outlier_report, f, indent=2)
```

### 4. Використовуйте domain knowledge
Консультуйтеся з експертами в предметній області!

### 5. A/B тестування

```python
# Порівняйте моделі з різною обробкою викидів
from sklearn.model_selection import cross_val_score

# З викидами
score_with = cross_val_score(model, X_with_outliers, y, cv=5).mean()

# Без викидів
score_without = cross_val_score(model, X_without_outliers, y_clean, cv=5).mean()

print(f"З викидами: {score_with:.4f}")
print(f"Без викидів: {score_without:.4f}")
```

---

## Поширені помилки ❌

### 1. Автоматичне видалення всіх викидів

```python
# ❌ НЕ робіть так без аналізу!
df = df[df['ціна'] < df['ціна'].quantile(0.99)]
```

### 2. Обробка після train-test split

```python
# ❌ НЕПРАВИЛЬНО
X_train, X_test = train_test_split(X, y)
X_train = remove_outliers(X_train)  # Витік інформації!

# ✅ ПРАВИЛЬНО
X_clean = remove_outliers(X)
X_train, X_test = train_test_split(X_clean, y)
```

### 3. Ігнорування багатовимірних викидів

```python
# Окремо значення OK, але разом — викид
# Використовуйте Isolation Forest або LOF
```

---

## Пов'язані теми

- [[01_Feature_Scaling]] — масштабування після обробки викидів
- [[03_Missing_Values]] — пропуски можуть бути пов'язані з викидами
- [[Feature_Engineering]] — створення ознак з викидів

## Ресурси
- [Scikit-learn Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [PyOD Library](https://pyod.readthedocs.io/)

---

#ml #preprocessing #outliers #anomaly-detection #datascience
