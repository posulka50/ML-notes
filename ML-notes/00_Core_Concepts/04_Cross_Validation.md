# Cross-Validation (Крос-валідація)

## Що це?

**Cross-Validation (CV)** — це техніка **оцінки моделі**, яка розділяє датасет на **кілька частин** (folds), навчає модель на одних частинах та тестує на інших, **повторюючи процес** кілька разів.

**Головна ідея:** замість одного розділення на train/test, робимо **багато розділень** і усереднюємо результати. Це дає **більш надійну оцінку** продуктивності моделі.

## Навіщо потрібно?

- 🎯 **Надійніша оцінка** — менша залежність від конкретного розділення
- 📊 **Оцінка variance** — як стабільна модель
- 🔍 **Виявлення overfitting** — краще, ніж один train/test split
- ⚙️ **Hyperparameter tuning** — GridSearchCV, RandomizedSearchCV
- 💡 **Максимальне використання даних** — всі зразки використовуються для train і test
- 📉 **Зменшення bias оцінки** — особливо на малих датасетах

## Коли використовувати?

**Потрібно:**

- Малі та середні датасети (< 100k зразків)
- Hyperparameter tuning
- Порівняння моделей
- Коли потрібна **надійна оцінка** перед production

**Не потрібно:**

- Дуже великі датасети (> 1M) — занадто повільно
- Time series (використовуй TimeSeriesSplit)
- Коли є окремий великий test set

---

## K-Fold Cross-Validation

### Як працює?

```
Dataset розділяється на K частин (folds):

╔════╦════╦════╦════╦════╗
║ F1 ║ F2 ║ F3 ║ F4 ║ F5 ║  K = 5 folds
╚════╩════╩════╩════╩════╝

Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]

Кожен fold використовується як test рівно 1 раз
Всі інші folds — train set

Final Score = Average(Fold1, Fold2, Fold3, Fold4, Fold5)
```

### Базове використання

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import numpy as np

# Дані
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Модель
model = LogisticRegression(max_iter=10000, random_state=42)

# 5-Fold Cross-Validation
scores = cross_val_score(
    model,           # Модель
    X, y,           # Дані
    cv=5,           # Кількість folds
    scoring='accuracy'  # Метрика
)

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
print(f"95% CI: [{scores.mean() - 2*scores.std():.4f}, "
      f"{scores.mean() + 2*scores.std():.4f}]")

# Приклад виводу:
# CV Scores: [0.956 0.965 0.973 0.964 0.973]
# Mean: 0.9662
# Std: 0.0065
# 95% CI: [0.9532, 0.9792]
```

### Manual K-Fold

```python
from sklearn.model_selection import KFold

# K-Fold splitter
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    # Розділити дані
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Навчити модель
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    
    # Оцінити
    score = model.score(X_test, y_test)
    scores.append(score)
    
    print(f"Fold {fold}: {score:.4f}")

print(f"\nMean: {np.mean(scores):.4f}")
print(f"Std: {np.std(scores):.4f}")
```

### Параметри KFold

```python
kf = KFold(
    n_splits=5,        # Кількість folds (зазвичай 5 або 10)
    shuffle=True,      # Перемішати перед розділенням
    random_state=42    # Seed для відтворюваності
)

# shuffle=True — ВАЖЛИВО!
# Без shuffle, якщо дані відсортовані → погані результати
```

---

## Stratified K-Fold (для класифікації)

### Навіщо?

**Проблема:** несбалансовані класи можуть розподілятися нерівномірно по folds.

```python
# Дані: 90% клас 0, 10% клас 1
y = np.array([0]*900 + [1]*100)

# Звичайний KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    y_test = y[test_idx]
    class_dist = np.bincount(y_test) / len(y_test)
    print(f"Fold {fold}: {class_dist}")

# Можливий вивід:
# Fold 1: [0.85, 0.15]  ← 85% клас 0, 15% клас 1
# Fold 2: [0.93, 0.07]  ← 93% клас 0, 7% клас 1
# Неконсистентно! ❌
```

### ✅ Stratified K-Fold

**Зберігає пропорції класів у кожному fold**.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    y_test = y[test_idx]
    class_dist = np.bincount(y_test) / len(y_test)
    print(f"Fold {fold}: {class_dist}")

# Вивід:
# Fold 1: [0.90, 0.10]  ← Точно як в оригіналі!
# Fold 2: [0.90, 0.10]
# Fold 3: [0.90, 0.10]
# Fold 4: [0.90, 0.10]
# Fold 5: [0.90, 0.10]
# Консистентно! ✓
```

### Використання з cross_val_score

```python
from sklearn.model_selection import cross_val_score

# Автоматично використовує StratifiedKFold для classification
scores = cross_val_score(
    model, X, y,
    cv=5,  # Для classification → StratifiedKFold автоматично
    scoring='accuracy'
)

# Або явно:
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model, X, y,
    cv=skf,  # Передати об'єкт явно
    scoring='accuracy'
)
```

**Рекомендація:** для класифікації **завжди використовуй StratifiedKFold**!

---

## Leave-One-Out Cross-Validation (LOOCV)

### Що це?

**K = n** (де n — кількість зразків). Кожен зразок по черзі є test set.

```
Dataset: 10 зразків

Fold 1:  [TEST][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2:  [TRAIN][TEST][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 3:  [TRAIN][TRAIN][TEST][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN]
...
Fold 10: [TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TEST]

10 ітерацій для 10 зразків
```

### Використання

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()

# Для малого датасету
X_small = X[:100]  # Тільки 100 зразків
y_small = y[:100]

scores = cross_val_score(
    model, X_small, y_small,
    cv=loo,  # LOOCV
    scoring='accuracy'
)

print(f"Number of folds: {len(scores)}")  # 100
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

### Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Максимальне використання даних | ❌ **Дуже повільно** (n ітерацій) |
| ✅ Детермінований результат | ❌ Висока variance оцінки |
| ✅ Підходить для дуже малих датасетів | ❌ Не працює для великих датасетів |

**Коли використовувати:**
- Дуже малий датасет (< 100 зразків)
- Коли computational cost не важливий

**Коли НЕ використовувати:**
- Середні та великі датасети (> 100)
- Коли потрібна швидкість

---

## Time Series Cross-Validation

### Чому не K-Fold для time series?

```python
# ❌ ПОГАНО: звичайний K-Fold для time series
# Порушує часову послідовність!

Dates: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct]

K-Fold може дати:
Train: [Feb, Apr, Jun, Jul, Sep, Oct]
Test:  [Jan, Mar, May, Aug]

Модель навчається на майбутньому (Oct) 
і тестується на минулому (Jan)! ❌
```

### ✅ TimeSeriesSplit

**Expanding window:** train set росте, test завжди в майбутньому.

```
Dataset: 10 місяців

Fold 1: [TRAIN][TRAIN] [TEST]
Fold 2: [TRAIN][TRAIN][TRAIN] [TEST]
Fold 3: [TRAIN][TRAIN][TRAIN][TRAIN] [TEST]
Fold 4: [TRAIN][TRAIN][TRAIN][TRAIN][TRAIN] [TEST]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN] [TEST]

Train set росте
Test завжди ПІСЛЯ train (в майбутньому)
```

### Використання

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Time series дані (відсортовані за часом!)
n_samples = 100
X = np.random.randn(n_samples, 5)
y = np.random.randn(n_samples)

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"\nFold {fold}:")
    print(f"  Train: indices {train_idx[0]} to {train_idx[-1]} "
          f"({len(train_idx)} samples)")
    print(f"  Test:  indices {test_idx[0]} to {test_idx[-1]} "
          f"({len(test_idx)} samples)")

# Вивід:
# Fold 1:
#   Train: indices 0 to 49 (50 samples)
#   Test:  indices 50 to 59 (10 samples)
# Fold 2:
#   Train: indices 0 to 59 (60 samples)
#   Test:  indices 60 to 69 (10 samples)
# ...
```

### З cross_val_score

```python
from sklearn.linear_model import Ridge

model = Ridge()

scores = cross_val_score(
    model, X, y,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='r2'
)

print(f"Time Series CV Scores: {scores}")
print(f"Mean R²: {scores.mean():.4f}")
```

### Параметри TimeSeriesSplit

```python
tscv = TimeSeriesSplit(
    n_splits=5,        # Кількість splits
    max_train_size=None,  # Максимальний розмір train (None = необмежений)
    test_size=None,    # Фіксований розмір test set
    gap=0              # Gap між train і test (кількість пропущених зразків)
)

# З gap (корисно для forecasting)
tscv_gap = TimeSeriesSplit(n_splits=5, gap=5)
# Train: [0...49], Test: [55...64] (5 зразків пропущено)
```

---

## Hyperparameter Tuning з CV

### GridSearchCV

**Перебір всіх комбінацій гіперпараметрів** з крос-валідацією.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Модель
rf = RandomForestClassifier(random_state=42)

# Параметри для перебору
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV
grid_search = GridSearchCV(
    rf,                      # Модель
    param_grid,              # Параметри
    cv=5,                    # 5-fold CV
    scoring='accuracy',      # Метрика
    n_jobs=-1,               # Паралелізація
    verbose=2                # Виводити прогрес
)

# Навчання (перебирає всі комбінації)
grid_search.fit(X, y)

# Результати
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Найкраща модель
best_model = grid_search.best_estimator_

# Всі результати
results = pd.DataFrame(grid_search.cv_results_)
print(results[['params', 'mean_test_score', 'std_test_score']].head(10))
```

**Кількість комбінацій:**
```python
# 3 * 4 * 3 * 3 = 108 комбінацій
# З cv=5 → 108 * 5 = 540 fits!
```

### RandomizedSearchCV

**Випадковий вибір комбінацій** — швидше за GridSearch.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Розподіли параметрів
param_distributions = {
    'n_estimators': randint(50, 300),       # Ціле число від 50 до 300
    'max_depth': [5, 10, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)       # Float від 0.1 до 1.0
}

random_search = RandomizedSearchCV(
    rf,
    param_distributions,
    n_iter=50,           # Кількість випадкових комбінацій
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

**Переваги RandomizedSearchCV:**
- ✅ Швидше (менше fits)
- ✅ Може знайти краще рішення (випадковий пошук іноді кращий)
- ✅ Підходить для continuous параметрів

### Порівняння Grid vs Random

| Аспект | GridSearchCV | RandomizedSearchCV |
|--------|--------------|-------------------|
| **Швидкість** | Повільно (всі комбінації) | Швидко (n_iter комбінацій) |
| **Покриття** | Вичерпний пошук | Випадковий пошук |
| **Continuous params** | Потрібно дискретизувати | Працює з розподілами |
| **Коли використовувати** | Малий param_grid | Великий param space |

---

## cross_validate (розширена версія)

**cross_val_score** повертає тільки одну метрику. **cross_validate** — багато метрик і додаткову інформацію.

```python
from sklearn.model_selection import cross_validate

# Кілька метрик
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring=scoring,
    return_train_score=True,  # Повернути train scores
    return_estimator=True     # Повернути fitted моделі
)

# Результати
print("Test scores:")
for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"  {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")

print("\nTrain scores:")
for metric in scoring.keys():
    scores = cv_results[f'train_{metric}']
    print(f"  {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Fitted моделі
models = cv_results['estimator']
print(f"\nNumber of fitted models: {len(models)}")

# Час виконання
print(f"Fit time: {cv_results['fit_time'].mean():.3f}s")
print(f"Score time: {cv_results['score_time'].mean():.3f}s")
```

---

## Вибір кількості folds (K)

### Загальні рекомендації

| Розмір датасету | Рекомендація | Причина |
|-----------------|--------------|---------|
| **Малий** (< 1000) | K=10 або LOOCV | Максимальне використання даних |
| **Середній** (1k-10k) | K=5 або K=10 | Баланс між bias і variance |
| **Великий** (> 10k) | K=3 або K=5 | Швидкість |

### K=5 vs K=10

```python
# K=5: швидше, більша variance оцінки
scores_5 = cross_val_score(model, X, y, cv=5)

# K=10: повільніше, менша variance оцінки
scores_10 = cross_val_score(model, X, y, cv=10)

print(f"K=5:  Mean={scores_5.mean():.4f}, Std={scores_5.std():.4f}")
print(f"K=10: Mean={scores_10.mean():.4f}, Std={scores_10.std():.4f}")

# Зазвичай K=10 має меншу std
```

### Компроміс

```
K малий (наприклад, K=3):
✅ Швидше
✅ Більший train set в кожному fold
❌ Більша variance оцінки
❌ Менше використання даних для тестування

K великий (наприклад, K=10):
✅ Менша variance оцінки
✅ Краще використання даних
❌ Повільніше
❌ Менший train set в кожному fold
```

**Стандарт:** **K=5 або K=10**

---

## Nested Cross-Validation

### Навіщо?

**Проблема:** якщо використовуємо CV і для model selection, і для оцінки → завищена оцінка (optimistic bias).

**Рішення:** два рівні CV:
- **Outer CV** — оцінка узагальнення
- **Inner CV** — hyperparameter tuning

```
Outer Loop (5 folds):
  Fold 1: [TRAIN+VAL][TRAIN+VAL][TRAIN+VAL][TRAIN+VAL][TEST]
           └── Inner CV (для tuning на TRAIN+VAL)
  Fold 2: [TRAIN+VAL][TRAIN+VAL][TRAIN+VAL][TEST][TRAIN+VAL]
           └── Inner CV
  ...
```

### Реалізація

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Модель з tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

# GridSearchCV (Inner CV)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,  # Inner CV: 3 folds
    scoring='accuracy'
)

# Outer CV
outer_scores = cross_val_score(
    grid_search,  # GridSearchCV як estimator
    X, y,
    cv=5,  # Outer CV: 5 folds
    scoring='accuracy'
)

print(f"Nested CV scores: {outer_scores}")
print(f"Mean: {outer_scores.mean():.4f}")
print(f"Std: {outer_scores.std():.4f}")

# Це ЧЕСНА оцінка узагальнення!
```

### Схема

```
Total: 5 (outer) × 3 (inner) × 9 (param combinations) = 135 fits

Outer Fold 1:
  Inner CV на 80% даних:
    - Перебрати всі параметри (3×3=9)
    - Вибрати найкращі
  Оцінити на 20% (test для outer fold 1)

Outer Fold 2:
  Inner CV на 80% даних:
    - Знову перебрати (можуть бути інші найкращі!)
  Оцінити на 20%

...
```

**Результат:** надійна оцінка того, як модель (з автоматичним tuning) працюватиме на нових даних.

---

## Метрики для CV

### Regression

```python
from sklearn.model_selection import cross_val_score

# R² (за замовчуванням)
scores = cross_val_score(model, X, y, cv=5)

# MSE
scores_mse = cross_val_score(
    model, X, y, cv=5, 
    scoring='neg_mean_squared_error'  # Негативний MSE
)
mse_scores = -scores_mse  # Конвертувати назад

# MAE
scores_mae = cross_val_score(
    model, X, y, cv=5,
    scoring='neg_mean_absolute_error'
)

# Кілька метрик
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X, y, cv=5,
    scoring={
        'r2': 'r2',
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
)
```

### Classification

```python
# Accuracy (за замовчуванням для classification)
scores = cross_val_score(model, X, y, cv=5)

# Precision
scores_prec = cross_val_score(model, X, y, cv=5, scoring='precision')

# Recall
scores_rec = cross_val_score(model, X, y, cv=5, scoring='recall')

# F1
scores_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')

# ROC-AUC
scores_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

# Для multiclass
scores_f1_weighted = cross_val_score(
    model, X, y, cv=5, 
    scoring='f1_weighted'  # Weighted F1
)
```

### Повний список

```python
from sklearn.metrics import get_scorer_names

# Всі доступні scoring functions
all_scorers = get_scorer_names()
print(f"Available scorers: {len(all_scorers)}")
print(all_scorers[:20])  # Перші 20
```

---

## Візуалізація CV результатів

### Boxplot

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Кілька моделей
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# CV scores для кожної моделі
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Візуалізація
plt.figure(figsize=(10, 6))

plt.boxplot(cv_results.values(), labels=cv_results.keys())
plt.ylabel('Accuracy', fontsize=12)
plt.title('10-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15, ha='right')

# Середні значення
means = [scores.mean() for scores in cv_results.values()]
plt.plot(range(1, len(means)+1), means, 'r^', markersize=10, label='Mean')
plt.legend()

plt.tight_layout()
plt.show()
```

### Learning Curve з CV

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Усереднення по folds
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Візуалізація
plt.figure(figsize=(10, 6))

plt.plot(train_sizes, train_mean, 'o-', linewidth=2, label='Train Score')
plt.fill_between(train_sizes, train_mean - train_std, 
                 train_mean + train_std, alpha=0.1)

plt.plot(train_sizes, val_mean, 's-', linewidth=2, label='CV Score')
plt.fill_between(train_sizes, val_mean - val_std,
                 val_mean + val_std, alpha=0.1)

plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Learning Curves with 5-Fold CV', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Поширені помилки ❌

### 1. Preprocessing перед CV

```python
# ❌ ПОГАНО: витік інформації
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Використав ВЕСЬ датасет!

scores = cross_val_score(model, X_scaled, y, cv=5)
# CV folds бачать інформацію один від одного через scaling!

# ✅ ДОБРЕ: використовуй Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5)
# Scaling виконується ВСЕРЕДИНІ кожного fold!
```

### 2. Оцінка на всьому датасеті після CV

```python
# ❌ ПОГАНО
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean()}")

# Потім навчити на ВСЬОМУ датасеті і оцінити
model.fit(X, y)
final_score = model.score(X, y)  # ❌ Завищена оцінка!

# ✅ ДОБРЕ: зберегти окремий test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CV на train
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Score: {scores.mean()}")

# Фінальна модель на ВСЬОМУ train
model.fit(X_train, y_train)

# Оцінка на test (не використовувався в CV!)
final_score = model.score(X_test, y_test)
```

### 3. Використання CV для model selection БЕЗ nested CV

```python
# ❌ ПОГАНО: optimistic bias
models = [LogisticRegression(), RandomForest(), GradientBoosting()]

best_score = 0
best_model = None

for model in models:
    score = cross_val_score(model, X, y, cv=5).mean()
    if score > best_score:
        best_score = score
        best_model = model

print(f"Best CV score: {best_score}")  # Завищена!

# ✅ ДОБРЕ: nested CV або окремий test set
# Варіант 1: окремий test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CV тільки на train для вибору
best_score = 0
for model in models:
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    if score > best_score:
        best_score = score
        best_model = model

# Фінальна оцінка на test
best_model.fit(X_train, y_train)
final_score = best_model.score(X_test, y_test)
```

### 4. Shuffle для time series

```python
# ❌ ПОГАНО
scores = cross_val_score(model, X_timeseries, y_timeseries, cv=5)
# KFold за замовчуванням shuffle=True → порушує часовий порядок!

# ✅ ДОБРЕ
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_timeseries, y_timeseries, cv=tscv)
```

### 5. Забувати про stratification для несбалансованих класів

```python
# Дані: 95% клас 0, 5% клас 1

# ❌ ПОГАНО: звичайний KFold
scores = cross_val_score(model, X, y, cv=5)
# Деякі folds можуть не містити клас 1!

# ✅ ДОБРЕ: StratifiedKFold (автоматично для classification)
scores = cross_val_score(model, X, y, cv=5)
# Або явно:
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

---

## Практичні поради 💡

### 1. Завжди використовуй Pipeline для preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Pipeline гарантує, що preprocessing робиться ВСЕРЕДИНІ CV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier())
])

scores = cross_val_score(pipeline, X, y, cv=5)
```

### 2. Фіксуй random_state для відтворюваності

```python
from sklearn.model_selection import StratifiedKFold

# Фіксований seed
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv)
# Завжди однакові результати
```

### 3. Зберігай результати CV

```python
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)

# Конвертувати в DataFrame
df_results = pd.DataFrame(cv_results)
df_results.to_csv('cv_results.csv', index=False)

print(df_results.describe())
```

### 4. Використовуй cross_validate замість cross_val_score

```python
# cross_validate дає більше інформації
cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring='accuracy',
    return_train_score=True,
    return_estimator=True  # Зберегти fitted моделі
)

# Можна аналізувати train vs test
train_scores = cv_results['train_score']
test_scores = cv_results['test_score']

print(f"Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
print(f"Test: {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")

# Діагностика overfitting
if train_scores.mean() - test_scores.mean() > 0.1:
    print("⚠️ Possible overfitting")
```

### 5. Перевіряй час виконання

```python
import time

start = time.time()
scores = cross_val_score(model, X, y, cv=10)
elapsed = time.time() - start

print(f"CV Time: {elapsed:.2f}s")
print(f"Time per fold: {elapsed/10:.2f}s")

# Якщо занадто повільно → зменш K або використовуй RandomizedSearchCV
```

---

## Реальний приклад: Model Selection з CV

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

print("="*70)
print("MODEL SELECTION WITH CROSS-VALIDATION")
print("="*70)

# Дані
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.unique(y)}")

# CV strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Моделі для порівняння
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=10000, random_state=42))
    ]),
    
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(random_state=42, probability=True))
    ])
}

# Метрики
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Оцінка кожної моделі
results = []

print("\n" + "="*70)
print("EVALUATING MODELS")
print("="*70)

for name, model in models.items():
    print(f"\n{name}...")
    
    # Cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Зберегти результати
    result = {'Model': name}
    
    for metric in scoring:
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        result[f'{metric}_mean'] = test_scores.mean()
        result[f'{metric}_std'] = test_scores.std()
        result[f'train_{metric}_mean'] = train_scores.mean()
        result[f'gap_{metric}'] = train_scores.mean() - test_scores.mean()
    
    result['fit_time'] = cv_results['fit_time'].mean()
    results.append(result)
    
    print(f"  Accuracy: {result['accuracy_mean']:.4f} (+/- {result['accuracy_std']:.4f})")
    print(f"  ROC-AUC:  {result['roc_auc_mean']:.4f} (+/- {result['roc_auc_std']:.4f})")

# DataFrame з результатами
df_results = pd.DataFrame(results)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Топ моделі за accuracy
print("\nRanked by Accuracy:")
print(df_results[['Model', 'accuracy_mean', 'accuracy_std']]\
      .sort_values('accuracy_mean', ascending=False)\
      .to_string(index=False))

print("\nRanked by ROC-AUC:")
print(df_results[['Model', 'roc_auc_mean', 'roc_auc_std']]\
      .sort_values('roc_auc_mean', ascending=False)\
      .to_string(index=False))

# Overfitting analysis
print("\nOverfitting Analysis (Gap Train-Test):")
for _, row in df_results.iterrows():
    gap = row['gap_accuracy']
    status = "✅" if gap < 0.05 else "⚠️" if gap < 0.1 else "🔴"
    print(f"  {status} {row['Model']}: {gap:.4f}")

# Візуалізація
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy comparison
model_names = df_results['Model'].values
acc_means = df_results['accuracy_mean'].values
acc_stds = df_results['accuracy_std'].values

x = np.arange(len(model_names))
axes[0, 0].bar(x, acc_means, yerr=acc_stds, alpha=0.7, capsize=5)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('10-Fold CV: Accuracy Comparison', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_ylim([0.85, 1.0])

# 2. Multiple metrics heatmap
metrics_to_plot = ['accuracy_mean', 'precision_mean', 'recall_mean', 
                   'f1_mean', 'roc_auc_mean']
heatmap_data = df_results[['Model'] + metrics_to_plot].set_index('Model')

import seaborn as sns
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', 
            ax=axes[0, 1], cbar_kws={'label': 'Score'})
axes[0, 1].set_title('All Metrics Heatmap', fontsize=13, fontweight='bold')
axes[0, 1].set_yticklabels(axes[0, 1].get_yticklabels(), rotation=0)

# 3. Train vs Test (overfitting check)
train_acc = df_results['train_accuracy_mean'].values
test_acc = df_results['accuracy_mean'].values

width = 0.35
axes[1, 0].bar(x - width/2, train_acc, width, label='Train', alpha=0.8)
axes[1, 0].bar(x + width/2, test_acc, width, label='Test', alpha=0.8)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
axes[1, 0].set_ylabel('Accuracy', fontsize=11)
axes[1, 0].set_title('Train vs Test Accuracy', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Fit time comparison
fit_times = df_results['fit_time'].values

axes[1, 1].barh(model_names, fit_times, alpha=0.7)
axes[1, 1].set_xlabel('Time (seconds)', fontsize=11)
axes[1, 1].set_title('Average Fit Time per Fold', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Найкраща модель
best_idx = df_results['accuracy_mean'].idxmax()
best_model_name = df_results.loc[best_idx, 'Model']
best_accuracy = df_results.loc[best_idx, 'accuracy_mean']

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print(f"Best model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"ROC-AUC: {df_results.loc[best_idx, 'roc_auc_mean']:.4f}")

# Hyperparameter tuning для найкращої моделі
if best_model_name == 'Random Forest':
    print("\nPerforming hyperparameter tuning for Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

print("="*70)
```

---

## Пов'язані теми

- [[01_Bias_Variance_Tradeoff]] — CV допомагає оцінити variance
- [[02_Overfitting_Underfitting]] — діагностика через CV
- [[03_Train_Test_Split]] — базовий метод валідації
- [[Hyperparameter_Tuning]] — GridSearchCV, RandomizedSearchCV
- [[Model_Selection]] — порівняння моделей

## Ресурси

- [Scikit-learn: Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw)
- [Machine Learning Mastery: k-Fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)

---

## Ключові висновки

> Cross-Validation — це техніка розділення датасету на кілька частин (folds), навчання та тестування моделі на різних комбінаціях цих частин, і усереднення результатів. Це дає більш надійну оцінку, ніж один train-test split.

**Типи CV:**

- **K-Fold** — стандарт (K=5 або K=10)
- **Stratified K-Fold** — для класифікації (зберігає пропорції класів)
- **TimeSeriesSplit** — для часових рядів
- **LOOCV** — для дуже малих датасетів

**Best Practices:**

```python
# 1. Використовуй Pipeline для preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# 2. Stratified для класифікації
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Cross-validate замість cross_val_score
cv_results = cross_validate(pipeline, X, y, cv=cv, 
                            return_train_score=True)

# 4. Nested CV для model selection
```

**Переваги:**

- ✅ Надійніша оцінка (менша variance)
- ✅ Максимальне використання даних
- ✅ Оцінка stability моделі
- ✅ Виявлення overfitting

**Недоліки:**
- ❌ Повільніше (K × fits)
- ❌ Не підходить для великих датасетів
- ❌ Складніша реалізація

**Золоте правило:** завжди використовуй CV для model selection і hyperparameter tuning, але зберігай окремий test set для фінальної оцінки!

---

#ml #core-concepts #cross-validation #model-evaluation #hyperparameter-tuning #k-fold #stratified
