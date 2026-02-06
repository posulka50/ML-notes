
Повний набір класів, методів та функцій scikit-learn для роботи з лінійною регресією та суміжними темами.

---

## 📦 Імпорти

```python
# Основні моделі
from sklearn.linear_model import (
    LinearRegression,      # Звичайна лінійна регресія
    Ridge,                 # Ridge регресія (L2)
    Lasso,                 # Lasso регресія (L1)
    ElasticNet,            # Elastic Net (L1 + L2)
    LogisticRegression,    # Логістична регресія
    RidgeCV,               # Ridge з вбудованою cross-validation
    LassoCV,               # Lasso з вбудованою CV
    ElasticNetCV,          # ElasticNet з CV
    SGDRegressor,          # Стохастичний градієнтний спуск
    SGDClassifier          # SGD для класифікації
)

# Поліноміальні ознаки
from sklearn.preprocessing import (
    PolynomialFeatures,    # Генерація поліноміальних ознак
    StandardScaler,        # Нормалізація (mean=0, std=1)
    MinMaxScaler,          # Масштабування [0, 1]
    RobustScaler,          # Робастне масштабування (стійке до викидів)
    LabelEncoder,          # Кодування міток
    OneHotEncoder          # One-hot кодування
)

# Розділення даних
from sklearn.model_selection import (
    train_test_split,      # Розділення на train/test
    cross_val_score,       # Cross-validation з оцінкою
    cross_validate,        # CV з детальними результатами
    KFold,                 # K-Fold CV
    StratifiedKFold,       # Stratified K-Fold
    TimeSeriesSplit,       # Time series split
    LeaveOneOut,           # Leave-One-Out CV
    GroupKFold,            # Group K-Fold
    GridSearchCV,          # Grid search для гіперпараметрів
    RandomizedSearchCV,    # Random search
    learning_curve,        # Криві навчання
    validation_curve       # Криві валідації
)

# Метрики регресії
from sklearn.metrics import (
    mean_squared_error,           # MSE
    mean_absolute_error,          # MAE
    r2_score,                     # R²
    mean_absolute_percentage_error, # MAPE
    explained_variance_score      # Explained variance
)

# Метрики класифікації
from sklearn.metrics import (
    accuracy_score,        # Accuracy
    precision_score,       # Precision
    recall_score,          # Recall
    f1_score,              # F1-score
    roc_auc_score,         # ROC-AUC
    roc_curve,             # ROC крива
    confusion_matrix,      # Confusion matrix
    classification_report  # Повний звіт
)

# Pipelines
from sklearn.pipeline import Pipeline, make_pipeline

# Utilities
from sklearn.datasets import (
    make_regression,       # Генерація регресійних даних
    make_classification,   # Генерація класифікаційних даних
    load_diabetes,         # Датасет діабету
    load_boston            # Датасет цін будинків (deprecated)
)

# Збереження моделей
import pickle
import joblib
```

---

## 1️⃣ LinearRegression — Звичайна лінійна регресія

### Основні параметри

```python
LinearRegression(
    fit_intercept=True,    # Чи обчислювати β₀ (intercept)
    normalize=False,       # Deprecated (використовуй StandardScaler)
    copy_X=True,           # Копіювати X (щоб не змінювати оригінал)
    n_jobs=None,           # Кількість ядер (-1 = всі)
    positive=False         # Примусити коефіцієнти бути позитивними
)
```

### Методи

```python
model = LinearRegression()

# Навчання
model.fit(X_train, y_train)

# Передбачення
y_pred = model.predict(X_test)

# Оцінка (R²)
score = model.score(X_test, y_test)

# Доступ до коефіцієнтів
model.coef_          # β₁, β₂, ..., βₚ
model.intercept_     # β₀

# Кількість ознак
model.n_features_in_
model.feature_names_in_  # Якщо X був DataFrame
```

### Приклад

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Генерація даних
X = np.random.randn(100, 5)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)*0.5

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Модель
model = LinearRegression()
model.fit(X_train, y_train)

# Результати
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R² train: {model.score(X_train, y_train):.3f}")
print(f"R² test: {model.score(X_test, y_test):.3f}")

# Передбачення
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
```

---

## 2️⃣ Ridge — L2 регуляризація

### Основні параметри

```python
Ridge(
    alpha=1.0,             # λ - сила регуляризації (більше = сильніша)
    fit_intercept=True,
    copy_X=True,
    max_iter=None,         # Максимум ітерацій (для solver='sag')
    tol=1e-4,              # Толерантність для зупинки
    solver='auto',         # 'auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'
    positive=False,
    random_state=None
)
```

### Методи (аналогічні LinearRegression)

```python
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
```

### Приклад з підбором alpha

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Підбір alpha
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

ridge = Ridge()
grid_search = GridSearchCV(
    ridge, param_grid, cv=5, scoring='r2'
)
grid_search.fit(X_train, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best R²: {grid_search.best_score_:.3f}")

# Використання найкращої моделі
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

### RidgeCV — автоматичний підбір alpha

```python
from sklearn.linear_model import RidgeCV

# Автоматичний вибір alpha
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
model = RidgeCV(alphas=alphas, cv=5)
model.fit(X_train, y_train)

print(f"Best alpha: {model.alpha_}")
print(f"R² test: {model.score(X_test, y_test):.3f}")
```

---

## 3️⃣ Lasso — L1 регуляризація

### Основні параметри

```python
Lasso(
    alpha=1.0,             # λ - сила регуляризації
    fit_intercept=True,
    max_iter=1000,         # Максимум ітерацій
    tol=1e-4,
    positive=False,
    selection='cyclic',    # 'cyclic' або 'random'
    random_state=None
)
```

### Особливість: відбір ознак

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Які коефіцієнти зануляться?
print("Coefficients:", model.coef_)
print("Non-zero features:", np.sum(model.coef_ != 0))

# Відбір найважливіших ознак
important_features = np.where(model.coef_ != 0)[0]
print(f"Important features: {important_features}")
```

### LassoCV

```python
from sklearn.linear_model import LassoCV

model = LassoCV(
    alphas=None,           # Автоматично генерує alphas
    cv=5,
    max_iter=10000,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Best alpha: {model.alpha_}")
print(f"Non-zero coefs: {np.sum(model.coef_ != 0)}")
```

---

## 4️⃣ ElasticNet — L1 + L2

### Основні параметри

```python
ElasticNet(
    alpha=1.0,             # Загальна сила регуляризації
    l1_ratio=0.5,          # Баланс L1/L2: 0=Ridge, 1=Lasso
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    positive=False,
    selection='cyclic',
    random_state=None
)
```

### Приклад

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=0.1,      # Сила регуляризації
    l1_ratio=0.5    # 50% L1 + 50% L2
)
model.fit(X_train, y_train)

print(f"R² test: {model.score(X_test, y_test):.3f}")
print(f"Non-zero coefs: {np.sum(model.coef_ != 0)}")
```

### ElasticNetCV

```python
from sklearn.linear_model import ElasticNetCV

model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
    alphas=[0.001, 0.01, 0.1, 1],
    cv=5,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Best alpha: {model.alpha_}")
print(f"Best l1_ratio: {model.l1_ratio_}")
```

---

## 5️⃣ LogisticRegression — Логістична регресія

### Основні параметри

```python
LogisticRegression(
    penalty='l2',          # 'l1', 'l2', 'elasticnet', None
    C=1.0,                 # Inverse of λ (менше = сильніша регуляризація)
    solver='lbfgs',        # 'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'
    max_iter=100,
    multi_class='auto',    # 'ovr', 'multinomial'
    class_weight=None,     # 'balanced' для незбалансованих класів
    random_state=None,
    l1_ratio=None          # Для penalty='elasticnet'
)
```

### Методи

```python
model = LogisticRegression()
model.fit(X_train, y_train)

# Передбачення класів
y_pred = model.predict(X_test)

# Передбачення ймовірностей
y_proba = model.predict_proba(X_test)
# [[P(class=0), P(class=1)], ...]

# Log-odds
y_decision = model.decision_function(X_test)

# Accuracy
score = model.score(X_test, y_test)

# Коефіцієнти
model.coef_        # β₁, β₂, ..., βₚ
model.intercept_   # β₀
model.classes_     # Унікальні класи
```

### Приклад з метриками

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

# Модель
model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Передбачення
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Метрики
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc_score(y_test, y_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 6️⃣ SGDRegressor / SGDClassifier — Стохастичний градієнтний спуск

### SGDRegressor

```python
from sklearn.linear_model import SGDRegressor

SGDRegressor(
    loss='squared_error',  # 'squared_error', 'huber', 'epsilon_insensitive'
    penalty='l2',          # 'l1', 'l2', 'elasticnet'
    alpha=0.0001,          # Регуляризація
    l1_ratio=0.15,         # Для elasticnet
    max_iter=1000,
    tol=1e-3,
    learning_rate='invscaling',  # 'constant', 'optimal', 'invscaling', 'adaptive'
    eta0=0.01,             # Початковий learning rate
    random_state=None
)
```

### Приклад

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

print(f"R² test: {model.score(X_test, y_test):.3f}")
```

### SGDClassifier

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    loss='log_loss',       # 'hinge' (SVM), 'log_loss' (LogReg)
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
```

---

## 7️⃣ PolynomialFeatures — Поліноміальна регресія

### Параметри

```python
PolynomialFeatures(
    degree=2,              # Степінь полінома
    interaction_only=False, # Тільки взаємодії (x1*x2), без x1², x2²
    include_bias=True      # Додавати колонку з 1 (для intercept)
)
```

### Приклад

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Створення поліноміальних ознак
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")
print(f"Feature names: {poly.get_feature_names_out()}")

# Або з Pipeline
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Приклад з різними степенями

```python
from sklearn.metrics import mean_squared_error
import numpy as np

degrees = [1, 2, 3, 4, 5]
results = []

for d in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('linear', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    
    results.append({
        'degree': d,
        'train_mse': train_mse,
        'test_mse': test_mse
    })
    
    print(f"Degree {d}: Train MSE={train_mse:.3f}, Test MSE={test_mse:.3f}")
```

---

## 8️⃣ Preprocessing — Масштабування та кодування

### StandardScaler — нормалізація

```python
from sklearn.preprocessing import StandardScaler

# Перетворення: (x - mean) / std
scaler = StandardScaler()

# Fit + transform на train
X_train_scaled = scaler.fit_transform(X_train)

# Тільки transform на test (використовуємо статистики train!)
X_test_scaled = scaler.transform(X_test)

# Доступ до параметрів
print(f"Mean: {scaler.mean_}")
print(f"Std: {scaler.scale_}")
```

### MinMaxScaler — масштабування [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

# Перетворення: (x - min) / (max - min)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

### RobustScaler — стійке до викидів

```python
from sklearn.preprocessing import RobustScaler

# Використовує медіану та IQR замість mean/std
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### OneHotEncoder — кодування категоріальних

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(X_categorical)

# Альтернатива: pandas get_dummies
import pandas as pd
X_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)
```

---

## 9️⃣ Model Selection — Розділення та валідація

### train_test_split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% на тест
    random_state=42,       # Відтворюваність
    shuffle=True,          # Перемішати
    stratify=y             # Зберегти пропорції класів (для класифікації)
)
```

### cross_val_score — проста CV

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

# 5-fold cross-validation
scores = cross_val_score(
    model, X, y,
    cv=5,                  # Кількість folds
    scoring='r2'           # Метрика: 'r2', 'neg_mean_squared_error', etc.
)

print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

### cross_validate — детальна CV

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y,
    cv=5,
    scoring=['r2', 'neg_mean_squared_error'],
    return_train_score=True
)

print(f"Train R²: {results['train_r2'].mean():.3f}")
print(f"Val R²: {results['test_r2'].mean():.3f}")
print(f"Val MSE: {-results['test_neg_mean_squared_error'].mean():.3f}")
```

### KFold та StratifiedKFold

```python
from sklearn.model_selection import KFold, StratifiedKFold

# KFold для регресії
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Навчання та оцінка
    model.fit(X_train_fold, y_train_fold)
    score = model.score(X_val_fold, y_val_fold)
    print(f"Fold score: {score:.3f}")

# StratifiedKFold для класифікації
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

### TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
```

---

## 🔟 GridSearchCV та RandomizedSearchCV

### GridSearchCV — перебір усіх комбінацій

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'svd', 'lsqr']
}

grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,             # Використовувати всі ядра
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

# Результати
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Використання найкращої моделі
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.3f}")

# Всі результати
import pandas as pd
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['params', 'mean_test_score', 'std_test_score']])
```

### RandomizedSearchCV — випадковий пошук

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'alpha': uniform(0.001, 100),     # Неперервний розподіл
    'solver': ['auto', 'svd', 'lsqr']
}

random_search = RandomizedSearchCV(
    Ridge(),
    param_distributions,
    n_iter=20,             # Кількість комбінацій для пробування
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

### Pipeline з GridSearch

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

param_grid = {
    'ridge__alpha': [0.1, 1, 10],     # Використовуй назву кроку + '__'
    'ridge__solver': ['auto', 'svd']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

---

## 1️⃣1️⃣ Metrics — Метрики оцінки

### Метрики регресії

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

y_pred = model.predict(X_test)

# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.3f}")

# RMSE
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.3f}")

# MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.3f}")

# R²
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.3f}")

# MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape:.3f}")

# Explained Variance
ev = explained_variance_score(y_test, y_pred)
print(f"Explained Variance: {ev:.3f}")
```

### Метрики класифікації

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Ймовірність класу 1

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")

# Precision, Recall, F1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")

# ROC-AUC
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## 1️⃣2️⃣ Learning Curves та Validation Curves

### learning_curve — криві навчання

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='r2',
    n_jobs=-1
)

# Візуалізація
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Train Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, val_mean, label='Validation Score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.grid()
plt.show()
```

### validation_curve — криві валідації

```python
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores, val_scores = validation_curve(
    Ridge(), X, y,
    param_name='alpha',
    param_range=param_range,
    cv=5,
    scoring='r2'
)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_scores.mean(axis=1), label='Train Score')
plt.plot(param_range, val_scores.mean(axis=1), label='Validation Score')
plt.xscale('log')
plt.xlabel('Alpha (regularization)')
plt.ylabel('R²')
plt.title('Validation Curve for Ridge')
plt.legend()
plt.grid()
plt.show()
```

---

## 1️⃣3️⃣ Pipeline — Конвеєри обробки

### Простий Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Крок 1: масштабування
    ('ridge', Ridge(alpha=1.0))        # Крок 2: модель
])

# Використання як звичайна модель
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)

# Доступ до окремих кроків
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['ridge']
print(f"Ridge coefficients: {model.coef_}")
```

### make_pipeline — скорочений синтаксис

```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2),
    Ridge(alpha=1.0)
)

pipeline.fit(X_train, y_train)
```

### Pipeline з поліноміальними ознаками

```python
from sklearn.preprocessing import PolynomialFeatures

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## 1️⃣4️⃣ Збереження та завантаження моделей

### pickle

```python
import pickle

# Збереження
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Завантаження
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

y_pred = loaded_model.predict(X_test)
```

### joblib — краще для sklearn

```python
import joblib

# Збереження
joblib.dump(model, 'model.joblib')

# Завантаження
loaded_model = joblib.load('model.joblib')

y_pred = loaded_model.predict(X_test)
```

### Збереження Pipeline

```python
# Pipeline також зберігається цілком
joblib.dump(pipeline, 'pipeline.joblib')
loaded_pipeline = joblib.load('pipeline.joblib')
```

---

## 1️⃣5️⃣ Генерація синтетичних даних

### make_regression

```python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000,        # Кількість зразків
    n_features=10,         # Кількість ознак
    n_informative=5,       # Інформативні ознаки
    n_targets=1,           # Одна цільова змінна
    noise=10.0,            # Шум
    random_state=42
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
```

### make_classification

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],    # Незбалансовані класи
    random_state=42
)
```

### Реальні датасети

```python
from sklearn.datasets import load_diabetes, load_iris

# Регресія: діабет
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

# Класифікація: ірис
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names
```

---

## 1️⃣6️⃣ Повний приклад: End-to-End Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Генерація даних
X, y = make_regression(
    n_samples=1000, 
    n_features=20, 
    n_informative=10,
    noise=10, 
    random_state=42
)

# 2. Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# 4. Grid Search
param_grid = {
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# 5. Навчання
grid_search.fit(X_train, y_train)

# 6. Результати
print(f"Best alpha: {grid_search.best_params_['ridge__alpha']}")
print(f"Best CV R²: {grid_search.best_score_:.3f}")

# 7. Оцінка на test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\n=== Test Set Results ===")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# 8. Збереження
joblib.dump(best_model, 'best_ridge_model.joblib')
print("\nModel saved to 'best_ridge_model.joblib'")
```

---

## 1️⃣7️⃣ Корисні поради

### 1. Завжди масштабуй дані для регуляризованих моделей

```python
# ПОГАНО (без масштабування)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ДОБРЕ (з масштабуванням)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
```

### 2. Використовуй Pipeline для уникнення data leakage

```python
# ПОГАНО (масштабування до розділення)
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, ...)

# ДОБРЕ (Pipeline автоматично правильно)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])
```

### 3. Фіксуй random_state для відтворюваності

```python
# Всюди використовуй однаковий random_state
train_test_split(X, y, random_state=42)
cross_val_score(model, X, y, cv=KFold(random_state=42))
model = Ridge(random_state=42)
```

### 4. Використовуй n_jobs=-1 для прискорення

```python
# Використовує всі доступні ядра
cross_val_score(model, X, y, cv=5, n_jobs=-1)
GridSearchCV(model, param_grid, n_jobs=-1)
```

### 5. Перевіряй на overfitting

```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

if train_score - test_score > 0.1:
    print("⚠️ Overfitting detected!")
```

---

## 1️⃣8️⃣ Чек-лист для роботи з регресією

```python
# ✅ 1. Завантажити дані
X, y = load_data()

# ✅ 2. Розділити на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 3. Створити Pipeline з масштабуванням
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# ✅ 4. Grid Search для гіперпараметрів
param_grid = {'model__alpha': [0.1, 1, 10]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# ✅ 5. Оцінити на test
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)

# ✅ 6. Перевірити на overfitting
train_score = best_model.score(X_train, y_train)
print(f"Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")

# ✅ 7. Зберегти модель
joblib.dump(best_model, 'model.joblib')
```

---

## Резюме основних класів

|Клас|Призначення|Ключові параметри|
|---|---|---|
|`LinearRegression`|Звичайна лінійна регресія|`fit_intercept`|
|`Ridge`|L2 регуляризація|`alpha`|
|`Lasso`|L1 регуляризація (відбір ознак)|`alpha`|
|`ElasticNet`|L1 + L2|`alpha`, `l1_ratio`|
|`LogisticRegression`|Бінарна класифікація|`C`, `penalty`|
|`SGDRegressor`|SGD для регресії|`loss`, `penalty`, `alpha`|
|`PolynomialFeatures`|Поліноміальні ознаки|`degree`|
|`StandardScaler`|Нормалізація|-|
|`Pipeline`|Конвеєр обробки|-|
|`GridSearchCV`|Підбір гіперпараметрів|`param_grid`, `cv`|
|`cross_val_score`|Cross-validation|`cv`, `scoring`|

---

## Корисні посилання

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Linear Models User Guide](https://scikit-learn.org/stable/modules/linear_model.html)
- [Model Selection Guide](https://scikit-learn.org/stable/model_selection.html)
- [Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
