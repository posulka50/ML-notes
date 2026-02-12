# Gradient Boosting (Градієнтний бустинг)

## Що це?

**Gradient Boosting** — це потужний ансамблевий алгоритм, який **послідовно** будує слабкі моделі (зазвичай дерева), де кожна наступна модель виправляє помилки попередніх, рухаючись у напрямку градієнта функції втрат.

**Головна ідея:** навчати нові моделі на помилках (residuals) попередніх моделей, поступово покращуючи передбачення.

## Навіщо потрібен?

- 🏆 **Найвища точність** — SOTA на табличних даних (Kaggle winner)
- 🎯 **Гнучкість** — різні функції втрат для різних задач
- 📊 **Універсальність** — регресія, класифікація, ранжування
- 💡 **Feature importance** — аналіз важливості ознак
- 🔧 **Контроль** — багато параметрів для fine-tuning
- ⚡ **Сучасні реалізації** — XGBoost, LightGBM, CatBoost

## Коли використовувати?

**Потрібно:**
- Потрібна **максимальна точність** на табличних даних
- **Kaggle competitions** — стандарт для табличних даних
- Складні нелінійні залежності
- **Production ML** — висока якість передбачень
- Готовність витратити час на **tuning**

**Не потрібно:**
- **Інтерпретованість критична** → Decision Tree, Linear Models
- **Дуже швидкий baseline** → Random Forest (легше в tuning)
- Зображення, текст, аудіо → Deep Learning
- **Дуже малі дані** (< 1000 зразків) → Linear Models
- Потрібна швидкість inference → Linear Models

---

## Відмінність від Random Forest

### Random Forest (Bagging)

```
Паралельний підхід:

Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┼─→ Average/Vote → Prediction
Tree 4 ─┤
Tree 5 ─┘

Кожне дерево незалежне
Навчаються паралельно
```

### Gradient Boosting (Boosting)

```
Послідовний підхід:

Data → Tree 1 → Residuals₁ → Tree 2 → Residuals₂ → Tree 3 → ... → Final
         ↓                      ↓                      ↓
       Pred₁                  Pred₂                  Pred₃

Кожне дерево залежить від попереднього
Навчаються послідовно
```

**Ключова різниця:**
- **Random Forest:** незалежні дерева, виправляють variance
- **Gradient Boosting:** залежні дерева, виправляють bias

---

## Як працює Gradient Boosting?

### Інтуїція

**Задача:** передбачити зарплату.

**Крок 1:** Просте передбачення (середнє)

```
Реальна зарплата: [50k, 60k, 70k, 80k]
Передбачення (середнє): [65k, 65k, 65k, 65k]
Помилки (residuals): [-15k, -5k, +5k, +15k]
```

**Крок 2:** Навчити дерево на помилках

```
Модель 2 передбачає помилки: [-14k, -6k, +6k, +14k]
Нове передбачення: 65k + (-14k) = 51k, 65k + (-6k) = 59k, ...
Нові помилки: [-1k, +1k, -1k, +1k]  ← Менші!
```

**Крок 3:** Повторюємо...

```
Модель 3 передбачає нові помилки: [-1k, +1k, -1k, +1k]
Фінальне передбачення: [50k, 60k, 70k, 80k]  ← Ідеально!
```

### Математика

**Загальна форма:**

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu \cdot h_m(x)$$

де:
- $F_M(x)$ — фінальне передбачення після $M$ ітерацій
- $F_0(x)$ — початкове передбачення (зазвичай константа)
- $h_m(x)$ — $m$-те дерево (слабкий learner)
- $\nu$ — **learning rate** (0 < ν ≤ 1)
- $M$ — кількість дерев

### Алгоритм (для регресії)

**Вхід:** датасет $(x_i, y_i)$, функція втрат $L$, кількість ітерацій $M$

**1. Ініціалізація:** початкове передбачення (константа)
$$F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$$

Для MSE: $F_0(x) = \text{mean}(y) = \bar{y}$

**2. Для $m = 1$ до $M$:**

   **a) Обчислити псевдо-residuals (негативний градієнт):**
   $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$
   
   Для MSE: $r_{im} = y_i - F_{m-1}(x_i)$ (просто помилки)

   **b) Навчити дерево $h_m(x)$ передбачати residuals $r_m$**

   **c) Оновити модель:**
   $$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

**3. Вихід:** $F_M(x)$

---

## Learning Rate (швидкість навчання)

### Що це?

**Learning rate** $\nu$ контролює, наскільки ми довіряємо кожному новому дереву.

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

### Ефект різних $\nu$

```
ν = 1.0 (агресивне навчання):
    Швидка збіжність
    Високий ризик overfitting
    Мало дерев потрібно

ν = 0.1 (помірне навчання):
    Середня швидкість
    Баланс між точністю та overfitting
    Рекомендовано за замовчуванням

ν = 0.01 (повільне навчання):
    Дуже повільна збіжність
    Дуже робастно до overfitting
    Багато дерев потрібно
```

### Trade-off: Learning Rate vs Number of Trees

```
Висока точність досягається через:
    Малий ν + багато дерев (M)
    або
    Великий ν + мало дерев (M)

Але:
    Малий ν + багато M → краще узагальнення ✓
    Великий ν + мало M → ризик overfitting ✗
```

**Практичне правило:**
- $\nu = 0.1$ та $M = 100-500$ — хороший старт
- $\nu = 0.01$ та $M = 1000-5000$ — для максимальної якості

---

## Функції втрат

### Для регресії

#### 1. MSE (L2 Loss)

$$L(y, F(x)) = \frac{1}{2}(y - F(x))^2$$

**Градієнт (residuals):**
$$r = y - F(x)$$

**Використання:** звичайна регресія, чутлива до викидів

#### 2. MAE (L1 Loss)

$$L(y, F(x)) = |y - F(x)|$$

**Градієнт:**
$$r = \text{sign}(y - F(x))$$

**Використання:** робастна до викидів

#### 3. Huber Loss (комбінація L1 + L2)

$$L_\delta(y, F) = \begin{cases}
\frac{1}{2}(y - F)^2 & \text{якщо } |y - F| \leq \delta \\
\delta(|y - F| - \frac{\delta}{2}) & \text{інакше}
\end{cases}$$

**Використання:** компроміс між MSE та MAE

### Для класифікації

#### 1. Log Loss (Binary Cross-Entropy)

$$L(y, F(x)) = -[y \log(p) + (1-y) \log(1-p)]$$

де $p = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$

**Градієнт:**
$$r = y - p$$

**Використання:** бінарна класифікація

#### 2. Multinomial Deviance

Для багатокласової класифікації (softmax).

---

## Regularization (регуляризація)

### 1. Shrinkage (Learning Rate)

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

Малий $\nu$ → сильніша регуляризація

### 2. Subsampling (Stochastic GB)

**Випадкова підмножина даних** для кожного дерева:

```python
GradientBoostingClassifier(subsample=0.8)  # 80% даних на дерево
```

**Переваги:**
- Зменшує overfitting
- Прискорює навчання
- Додає стохастичність (як у SGD)

**Типові значення:** 0.5 - 1.0

### 3. Tree Constraints

**Обмеження складності дерев:**
- `max_depth` — глибина дерев (зазвичай 3-10)
- `min_samples_split` — мін. зразків для розбиття
- `min_samples_leaf` — мін. зразків у листку
- `max_features` — підмножина ознак

**Gradient Boosting використовує МІЛКІ дерева!**
- Random Forest: глибокі дерева (max_depth=None)
- Gradient Boosting: мілкі дерева (max_depth=3-5)

### 4. Early Stopping

**Зупинка при погіршенні на validation set:**

```python
gb = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.1,  # 10% для валідації
    n_iter_no_change=50,      # Зупинка після 50 ітерацій без покращення
    tol=1e-4
)
```

---

## Код (scikit-learn)

### Класифікація

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Генерація даних
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=100,         # Кількість дерев
    learning_rate=0.1,        # Швидкість навчання (ν)
    max_depth=3,              # Глибина дерев (мілкі!)
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,            # Stochastic GB (80% даних)
    max_features='sqrt',      # Підмножина ознак
    random_state=42,
    verbose=0
)

# Навчання
gb_clf.fit(X_train, y_train)

# Передбачення
y_pred = gb_clf.predict(X_test)
y_pred_proba = gb_clf.predict_proba(X_test)[:, 1]

# Метрики
print("=== Gradient Boosting Classification ===")
print(f"Train Accuracy: {gb_clf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n" + classification_report(y_test, y_pred))

# Feature Importance
importances = gb_clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n=== Top 10 Features ===")
for i in range(min(10, len(importances))):
    print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
```

### Регресія

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Дані
X, y = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=8,
    noise=10,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    loss='squared_error',    # або 'absolute_error', 'huber'
    random_state=42,
    verbose=0
)

# Навчання
gb_reg.fit(X_train, y_train)

# Передбачення
y_pred_train = gb_reg.predict(X_train)
y_pred_test = gb_reg.predict(X_test)

# Метрики
print("=== Gradient Boosting Regression ===")
print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title('Gradient Boosting Regression', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Staged Predictions (поетапні передбачення)

### Моніторинг навчання

**Отримання передбачень після кожного дерева:**

```python
import matplotlib.pyplot as plt

# Навчання
gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
gb.fit(X_train, y_train)

# Поетапні передбачення
train_scores = []
test_scores = []

for i, (train_pred, test_pred) in enumerate(zip(
    gb.staged_predict(X_train),
    gb.staged_predict(X_test)
)):
    train_scores.append(mean_squared_error(y_train, train_pred))
    test_scores.append(mean_squared_error(y_test, test_pred))

# Візуалізація
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_scores) + 1), train_scores, 
         label='Train MSE', linewidth=2)
plt.plot(range(1, len(test_scores) + 1), test_scores, 
         label='Test MSE', linewidth=2)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Gradient Boosting: MSE vs Number of Trees', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Оптимальна кількість дерев
optimal_n_trees = np.argmin(test_scores) + 1
print(f"Optimal number of trees: {optimal_n_trees}")
print(f"Best Test MSE: {test_scores[optimal_n_trees - 1]:.2f}")
```

**Спостереження:**
- Train MSE монотонно зменшується
- Test MSE спочатку зменшується, потім може зростати (overfitting)
- Оптимальна кількість дерев — мінімум Test MSE

---

## Підбір гіперпараметрів

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Сітка параметрів
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# Grid Search
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)
print(f"\nBest CV score: {grid_search.best_score_:.4f}")

# Тест
best_gb = grid_search.best_estimator_
print(f"Test score: {best_gb.score(X_test, y_test):.4f}")
```

### Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.5, 0.5),  # 0.5-1.0
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
```

---

## XGBoost, LightGBM, CatBoost

### Порівняння реалізацій

| Характеристика | scikit-learn GB | XGBoost | LightGBM | CatBoost |
|----------------|-----------------|---------|----------|----------|
| **Швидкість** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Точність** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Категоріальні дані** | ❌ | ⚠️ | ⚠️ | ✅✅ |
| **GPU підтримка** | ❌ | ✅ | ✅ | ✅ |
| **Regularization** | Базова | ✅✅ | ✅ | ✅ |
| **Використання** | Навчання | Production | Дуже великі дані | Категоріальні дані |

### XGBoost

**Extreme Gradient Boosting** — найпопулярніша реалізація.

```python
import xgboost as xgb

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,     # Feature sampling
    reg_alpha=0.1,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization
    random_state=42,
    eval_metric='logloss'
)

# Навчання з early stopping
xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

print(f"Best iteration: {xgb_clf.best_iteration}")
print(f"Test Accuracy: {xgb_clf.score(X_test, y_test):.4f}")
```

### LightGBM

**Light Gradient Boosting Machine** — найшвидша реалізація.

```python
import lightgbm as lgb

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,            # Унікально для LightGBM
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(10)]
)

print(f"Best iteration: {lgb_clf.best_iteration_}")
print(f"Test Accuracy: {lgb_clf.score(X_test, y_test):.4f}")
```

### CatBoost

**Categorical Boosting** — найкраще для категоріальних даних.

```python
from catboost import CatBoostClassifier

# CatBoost Classifier
cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    subsample=0.8,
    l2_leaf_reg=1.0,
    random_state=42,
    verbose=False
)

cat_clf.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=10
)

print(f"Best iteration: {cat_clf.get_best_iteration()}")
print(f"Test Accuracy: {cat_clf.score(X_test, y_test):.4f}")
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Найвища точність** | SOTA на табличних даних |
| **Гнучкість** | Різні функції втрат |
| **Feature importance** | Аналіз важливості ознак |
| **Missing values** | Може працювати (XGBoost, LightGBM) |
| **Regularization** | Багато способів контролю overfitting |
| **Нелінійні залежності** | Автоматично виявляє |
| **Робастність** | Менш чутливі до викидів за лінійні моделі |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Складний tuning** | Багато гіперпараметрів |
| **Повільне навчання** | Послідовна природа |
| **Overfitting** | Легко переобучитися без регуляризації |
| **Інтерпретованість** | Чорна скринька |
| **Не паралелиться** | На відміну від Random Forest |
| **Потребує підготовки** | Категоріальні дані, масштабування (іноді) |
| **Чутливість до шуму** | У даних з label noise |

---

## Random Forest vs Gradient Boosting

### Коли Random Forest краще?

✅ Потрібен **швидкий baseline**
✅ Мало часу на tuning
✅ Паралелізація важлива
✅ Дуже шумні дані
✅ Робастність > точність

### Коли Gradient Boosting краще?

✅ Потрібна **максимальна точність**
✅ Є час на fine-tuning
✅ Kaggle competition
✅ Production ML system
✅ Точність > швидкість навчання

### Порівняльна таблиця

| Критерій | Random Forest | Gradient Boosting |
|----------|---------------|-------------------|
| **Точність** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Швидкість навчання** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Tuning складність** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Overfitting ризик** | ⭐⭐ | ⭐⭐⭐⭐ |
| **Паралелізація** | ✅ Так | ❌ Ні (складно) |
| **Використання** | Baseline | Production |

---

## Практичні поради 💡

1. **Почни з малого learning rate** — 0.1 та 100-200 дерев
2. **Мілкі дерева** — max_depth=3-5 (не як у RF!)
3. **Subsample** — 0.8 для stochastic GB
4. **Early stopping** — завжди використовуй
5. **Staged predictions** — моніторинг навчання
6. **XGBoost/LightGBM** для production — швидше та точніше
7. **CatBoost** для категоріальних даних
8. **Feature engineering** — GB любить якісні ознаки
9. **Grid/Random Search** — інвестуй час у tuning
10. **Порівняй з Random Forest** — іноді RF достатньо

---

## Коли використовувати Gradient Boosting

### Ідеально підходить ✓

- **Kaggle competitions** — стандарт для табличних даних
- **Production ML** з високими вимогами до точності
- Складні нелінійні залежності
- Є час та ресурси для **tuning**
- Табличні дані (structured data)
- Потрібна **максимальна якість** передбачень

### Краще використати інше ✗

- **Інтерпретованість** → Decision Tree, Linear Models
- **Швидкий baseline** → Random Forest
- Зображення/Текст → Deep Learning
- **Дуже малі дані** → Linear Models, SVM
- Немає часу на tuning → Random Forest
- **Реал-тайм inference** → Linear Models

---

## Реальний приклад: Передбачення відтоку клієнтів

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Синтетичні дані (customer churn)
np.random.seed(42)
n_samples = 5000

data = {
    'tenure_months': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'total_charges': np.random.uniform(100, 8000, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'payment_method': np.random.choice(['Auto', 'Manual'], n_samples),
    'customer_service_calls': np.random.randint(0, 10, n_samples),
    'num_products': np.random.randint(1, 5, n_samples),
}

# Target (симулюємо churn)
churn_prob = (
    (data['tenure_months'] < 12) * 0.3 +
    (data['monthly_charges'] > 80) * 0.2 +
    (data['customer_service_calls'] > 5) * 0.3 +
    np.random.uniform(0, 0.2, n_samples)
)
data['churn'] = (churn_prob > 0.5).astype(int)

df = pd.DataFrame(data)

# Encoding категоріальних ознак
df['contract_month_to_month'] = (df['contract_type'] == 'Month-to-month').astype(int)
df['contract_one_year'] = (df['contract_type'] == 'One year').astype(int)
df['payment_auto'] = (df['payment_method'] == 'Auto').astype(int)

# Підготовка даних
X = df.drop(['churn', 'contract_type', 'payment_method'], axis=1)
y = df['churn']

print(f"Dataset shape: {X.shape}")
print(f"Churn rate: {y.mean():.2%}")

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gradient Boosting з оптимальними параметрами
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
    verbose=0
)

# Навчання
print("\nTraining Gradient Boosting...")
gb.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Передбачення
y_pred = gb.predict(X_test)
y_pred_proba = gb.predict_proba(X_test)[:, 1]

# Метрики
print("\n" + "="*60)
print("=== Model Performance ===")
print("="*60)
print(f"Train Accuracy: {gb.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {gb.score(X_test, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n" + classification_report(y_test, y_pred, 
                                   target_names=['No Churn', 'Churn']))

# Feature Importance
print("\n" + "="*60)
print("=== Top 5 Most Important Features ===")
print("="*60)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Візуалізації
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Feature Importance
axes[0, 0].barh(feature_importance['feature'][:10][::-1], 
                feature_importance['importance'][:10][::-1])
axes[0, 0].set_xlabel('Importance', fontsize=12)
axes[0, 0].set_title('Top 10 Feature Importances', 
                     fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# 2. Learning Curve (Staged Predictions)
train_scores_staged = []
test_scores_staged = []

for train_pred, test_pred in zip(gb.staged_predict_proba(X_train),
                                  gb.staged_predict_proba(X_test)):
    train_scores_staged.append(roc_auc_score(y_train, train_pred[:, 1]))
    test_scores_staged.append(roc_auc_score(y_test, test_pred[:, 1]))

axes[0, 1].plot(range(1, len(train_scores_staged) + 1), 
                train_scores_staged, label='Train', linewidth=2)
axes[0, 1].plot(range(1, len(test_scores_staged) + 1), 
                test_scores_staged, label='Test', linewidth=2)
axes[0, 1].set_xlabel('Number of Trees', fontsize=12)
axes[0, 1].set_ylabel('ROC-AUC', fontsize=12)
axes[0, 1].set_title('Learning Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
axes[1, 0].set_xlabel('False Positive Rate', fontsize=12)
axes[1, 0].set_ylabel('True Positive Rate', fontsize=12)
axes[1, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# 4. Prediction Distribution
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, 
                label='No Churn', color='blue', edgecolor='black')
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, 
                label='Churn', color='red', edgecolor='black')
axes[1, 1].set_xlabel('Predicted Probability', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Поширені помилки ❌

### 1. Використання глибоких дерев

```python
# ❌ НЕПРАВИЛЬНО (як у Random Forest)
gb = GradientBoostingClassifier(max_depth=None)

# ✅ ПРАВИЛЬНО (мілкі дерева!)
gb = GradientBoostingClassifier(max_depth=3)  # або 4, 5
```

### 2. Високий learning rate без регуляризації

```python
# ❌ РИЗИК OVERFITTING
gb = GradientBoostingClassifier(
    learning_rate=1.0,
    n_estimators=50
)

# ✅ КРАЩЕ
gb = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8
)
```

### 3. Не використовувати early stopping

```python
# ❌ МОЖЛИВИЙ OVERFITTING
gb = GradientBoostingClassifier(n_estimators=1000)
gb.fit(X_train, y_train)

# ✅ З EARLY STOPPING
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(n_estimators=1000)
xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50
)
```

### 4. Не моніторити навчання

```python
# ✅ ЗАВЖДИ МОНІТОРИТИ
train_scores = []
test_scores = []

for pred_train, pred_test in zip(gb.staged_predict(X_train),
                                   gb.staged_predict(X_test)):
    train_scores.append(accuracy_score(y_train, pred_train))
    test_scores.append(accuracy_score(y_test, pred_test))

# Якщо test_scores зростає → overfitting!
```

---

## Пов'язані теми

- [[01_Decision_Trees]] — базові будівельні блоки
- [[02_Random_Forest]] — альтернативний ансамбль
- [[04_AdaBoost]] — інший boosting алгоритм
- [[05_Ensemble_Methods]] — теорія ансамблів
- [[05_Gradient_Descent]] — концепція градієнта

## Ресурси

- [Scikit-learn: Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Original Paper: Friedman (2001)](https://jerryfriedman.su.domains/ftp/trebst.pdf)

---

## Ключові висновки

> Gradient Boosting послідовно будує слабкі моделі (мілкі дерева), де кожна наступна модель виправляє помилки (residuals) попередніх, рухаючись у напрямку градієнта функції втрат.

**Основні принципи:**
- **Послідовне навчання** — кожне дерево залежить від попереднього
- **Gradient descent в функціональному просторі** — мінімізація loss через додавання моделей
- **Learning rate** контролює швидкість навчання та overfitting
- **Мілкі дерева** (max_depth=3-5) як слабкі learners

**Формула:**
$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu \cdot h_m(x)$$

**Коли використовувати:**
- Максимальна точність на табличних даних = Gradient Boosting ✓
- Швидкий baseline без tuning = Random Forest ✓
- Production ML з XGBoost/LightGBM = Gradient Boosting ✓

**Найважливіше:**
- learning_rate=0.1, max_depth=3-5, subsample=0.8
- Завжди використовуй early stopping
- XGBoost/LightGBM для production

---

#ml #supervised-learning #ensemble #gradient-boosting #boosting #xgboost #lightgbm #catboost #kaggle #tree-based
