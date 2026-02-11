# Random Forest (Випадковий ліс)

## Що це?

**Random Forest** — це **ансамблевий алгоритм**, який будує множину decision trees на різних підмножинах даних та об'єднує їх передбачення для отримання більш точних і стабільних результатів.

**Головна ідея:** "мудрість натовпу" — багато простих моделей разом працюють краще за одну складну модель.

## Навіщо потрібен?

- 🎯 **Висока точність** — один з найкращих "out-of-the-box" алгоритмів
- 🛡️ **Робастність** — менш схильний до overfitting ніж одне дерево
- 📊 **Універсальність** — регресія та класифікація
- 🚀 **Простота** — мало гіперпараметрів для налаштування
- 💡 **Feature importance** — показує важливість ознак
- ⚡ **Паралелізація** — дерева навчаються незалежно

## Коли використовувати?

**Потрібно:**
- Потрібна **висока точність** без складного tuning
- **Табличні дані** (structured data)
- Категоріальні + числові ознаки
- **Baseline** перед складнішими моделями
- **Feature selection** — аналіз важливості ознак
- Робастність до шуму та викидів

**Не потрібно:**
- **Потрібна інтерпретованість** → Decision Tree (одне дерево)
- Дуже великі дані (>1M рядків) → Gradient Boosting (LightGBM, XGBoost)
- Зображення, текст → Deep Learning
- Потрібна максимальна точність на табличних даних → **Gradient Boosting**

---

## Як працює Random Forest?

### Основна ідея: Bootstrap + Feature Randomness

**Два рівні рандомізації:**

1. **Bootstrap Aggregating (Bagging)**
   - Для кожного дерева створюємо випадкову підмножину даних (з поверненням)
   - Кожне дерево бачить ~63% унікальних зразків

2. **Feature Randomness**
   - При кожному розбитті розглядаємо випадкову підмножину ознак
   - Зазвичай $\sqrt{p}$ для класифікації, $p/3$ для регресії

### Алгоритм

```
Для i = 1 до n_estimators:
    1. Створити bootstrap sample (випадкова вибірка з поверненням)
    2. Побудувати дерево:
        - На кожному вузлі:
            a. Вибрати випадкову підмножину ознак
            b. Знайти найкраще розбиття серед цих ознак
            c. Розділити вузол
    3. Зберегти дерево

Передбачення:
    - Класифікація: голосування більшості (majority vote)
    - Регресія: усереднення передбачень
```

### Візуалізація

```
Dataset (1000 samples)
        |
        |-- Bootstrap Sample 1 (1000 samples with replacement)
        |   → Train Tree 1 (using random features at each split)
        |
        |-- Bootstrap Sample 2 (1000 samples with replacement)
        |   → Train Tree 2 (using random features at each split)
        |
        |-- ...
        |
        |-- Bootstrap Sample 100
            → Train Tree 100 (using random features at each split)

Prediction for new sample X:
    Tree 1: Class A    Tree 2: Class B    ...    Tree 100: Class A
    
    Majority Vote: Class A (60 votes) > Class B (40 votes)
    → Final Prediction: Class A
```

---

## Bootstrap Aggregating (Bagging)

### Що таке Bootstrap?

**Вибірка з поверненням** — випадково обираємо $n$ зразків з $n$ доступних, але **дозволяємо повторення**.

**Приклад:**
```
Original dataset: [A, B, C, D, E]

Bootstrap sample 1: [A, B, A, C, E]  ← A двічі, D відсутній
Bootstrap sample 2: [D, D, B, C, A]  ← D двічі, E відсутній
Bootstrap sample 3: [B, E, C, A, B]  ← B двічі, D відсутній
```

### Статистика Bootstrap

**Ймовірність, що зразок НЕ буде обраний:**
$$P(\text{not selected}) = \left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$$

**Результат:**
- ~**63.2%** унікальних зразків у bootstrap sample
- ~**36.8%** зразків не увійшли (Out-Of-Bag samples)

### Out-Of-Bag (OOB) Error

**OOB зразки** — зразки, які НЕ використовувалися для навчання конкретного дерева.

**Використання:**
- Безкоштовна валідація без окремого test set!
- Для кожного зразка використовуємо дерева, які його не бачили
- OOB error ≈ тестова помилка

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_:.4f}")  # Оцінка на OOB зразках
```

---

## Feature Randomness

### Навіщо?

**Проблема корельованих дерев:**
- Якщо одна ознака дуже сильна → всі дерева використають її спочатку
- Дерева стануть схожими → мало різноманітності
- Ансамбль працює гірше

**Рішення: випадкові підмножини ознак**

### Скільки ознак вибирати?

| Задача | max_features | Пояснення |
|--------|--------------|-----------|
| **Класифікація** | $\sqrt{p}$ | За замовчуванням |
| **Регресія** | $p/3$ або $p$ | За замовчуванням |

де $p$ — загальна кількість ознак.

**Приклад:**
- 100 ознак, класифікація → $\sqrt{100} = 10$ ознак на розбиття
- 100 ознак, регресія → $100/3 \approx 33$ ознаки на розбиття

### Ефект max_features

```
max_features = 1:      Дуже різні дерева, висока variance
max_features = sqrt(p): Баланс (за замовчуванням для класифікації)
max_features = p:       Схожі дерева, менше різноманітності
```

---

## Передбачення (Aggregation)

### Класифікація: Majority Voting

**Кожне дерево голосує за клас:**

```
100 дерев передбачають для зразка X:

Tree 1: Class A
Tree 2: Class A
Tree 3: Class B
...
Tree 100: Class A

Votes:
    Class A: 65 votes
    Class B: 30 votes
    Class C: 5 votes

Final Prediction: Class A (majority)
```

**Формула:**
$$\hat{y} = \text{mode}\{h_1(x), h_2(x), ..., h_T(x)\}$$

### Регресія: Averaging

**Усереднення передбачень:**

```
100 дерев передбачують для зразка X:

Tree 1: 50.2
Tree 2: 51.8
Tree 3: 49.5
...
Tree 100: 50.7

Average: (50.2 + 51.8 + ... + 50.7) / 100 = 50.4

Final Prediction: 50.4
```

**Формула:**
$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} h_t(x)$$

### Ймовірності (для класифікації)

```python
# Predict probabilities
probas = rf.predict_proba(X_test)

# Кожне дерево дає soft vote (ймовірність)
# Фінальна ймовірність = середнє по деревах
```

---

## Код (Python + scikit-learn)

### Класифікація

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Завантаження даних
wine = load_wine()
X = wine.data
y = wine.target

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Модель Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=100,        # Кількість дерев
    max_depth=None,          # Необмежена глибина
    min_samples_split=2,     # Мін. зразків для розбиття
    min_samples_leaf=1,      # Мін. зразків у листку
    max_features='sqrt',     # sqrt(n_features) на розбиття
    bootstrap=True,          # Використовувати bootstrap
    oob_score=True,          # Обчислювати OOB score
    n_jobs=-1,               # Паралелізація (всі ядра)
    random_state=42
)

# 3. Навчання
rf_clf.fit(X_train, y_train)

# 4. Передбачення
y_pred = rf_clf.predict(X_test)
y_pred_proba = rf_clf.predict_proba(X_test)

# 5. Оцінка
print("=== Metrics ===")
print(f"Train Accuracy: {rf_clf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score: {rf_clf.oob_score_:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 6. Feature Importance
print("\n=== Feature Importance ===")
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(X.shape[1]):
    print(f"{i+1}. {wine.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Візуалізація Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), 
           [wine.feature_names[i] for i in indices], 
           rotation=45, ha='right')
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importances - Random Forest', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Регресія

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Генерація даних
X, y = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=7,
    noise=10,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,       # Всі ознаки (або 'sqrt', 0.5)
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

# Навчання
rf_reg.fit(X_train, y_train)

# Передбачення
y_pred_train = rf_reg.predict(X_train)
y_pred_test = rf_reg.predict(X_test)

# Метрики
print("=== Regression Metrics ===")
print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"OOB Score: {rf_reg.oob_score_:.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.2f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, s=50)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('True Values', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title('Random Forest Regression: Predictions vs True Values', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Гіперпараметри

### Основні параметри

| Параметр | Опис | Типові значення | Ефект |
|----------|------|-----------------|-------|
| **n_estimators** | Кількість дерев | 100-500 | Більше → точніше, але повільніше |
| **max_depth** | Макс. глибина дерева | None (необмежена) | Обмежує складність |
| **min_samples_split** | Мін. зразків для розбиття | 2-20 | Контроль overfitting |
| **min_samples_leaf** | Мін. зразків у листку | 1-10 | Згладжування |
| **max_features** | Макс. ознак на розбиття | 'sqrt', 'log2', 0.5 | Різноманітність дерев |
| **bootstrap** | Використовувати bootstrap | True | Bagging |
| **oob_score** | Обчислювати OOB error | False | Валідація |
| **n_jobs** | Паралелізація | -1 (всі ядра) | Швидкість |

### Параметри дерев

**Random Forest успадковує параметри Decision Tree:**
- `max_leaf_nodes`
- `min_impurity_decrease`
- `criterion` ('gini', 'entropy', 'squared_error')

---

## Підбір гіперпараметрів

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Сітка параметрів
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Кращі параметри
print("Best parameters:")
print(grid_search.best_params_)
print(f"\nBest CV score: {grid_search.best_score_:.4f}")

# Кращa модель
best_rf = grid_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

### Randomized Search (швидше)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Розподіли параметрів
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(randint(5, 50).rvs(10)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2'] + list(uniform(0.1, 0.9).rvs(5))
}

# Randomized Search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
```

---

## Bias-Variance Tradeoff

### Як Random Forest зменшує Variance

**Одне дерево:**
- **Високий variance** — малі зміни даних → різні дерева
- Схильне до overfitting

**Random Forest:**
- **Зменшує variance** через усереднення
- Кожне дерево має високий variance, але некорельовані помилки
- **Усереднення некорельованих помилок** → зменшення variance

### Математика

Для некорельованих моделей з variance $\sigma^2$:

**Variance одного дерева:** $\sigma^2$

**Variance усереднення T дерев:**
$$\text{Var}(\text{average}) = \frac{\sigma^2}{T}$$

**При корельованих моделях** (кореляція $\rho$):
$$\text{Var}(\text{average}) = \rho \sigma^2 + \frac{1 - \rho}{T} \sigma^2$$

**Висновок:**
- Більше дерев (T ↑) → менше variance ✓
- Менша кореляція ($\rho$ ↓) → менше variance ✓
- **Feature randomness знижує $\rho$!**

---

## Вплив n_estimators

### Експеримент

```python
from sklearn.model_selection import cross_val_score

# Тестуємо різну кількість дерев
n_estimators_range = [1, 5, 10, 20, 50, 100, 200, 500]
train_scores = []
test_scores = []
oob_scores = []

for n_est in n_estimators_range:
    rf = RandomForestClassifier(
        n_estimators=n_est,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    oob_scores.append(rf.oob_score_)

# Візуалізація
plt.figure(figsize=(12, 6))
plt.plot(n_estimators_range, train_scores, 'o-', 
         linewidth=2, label='Train Score')
plt.plot(n_estimators_range, test_scores, 's-', 
         linewidth=2, label='Test Score')
plt.plot(n_estimators_range, oob_scores, '^-', 
         linewidth=2, label='OOB Score')
plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Random Forest: Performance vs Number of Trees', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()
plt.show()
```

**Спостереження:**
- **1 дерево:** висока variance, низька accuracy
- **10-50 дерев:** швидке покращення
- **100+ дерев:** стабілізація (diminishing returns)
- **500+ дерев:** майже немає покращення, але повільніше

**Висновок:** n_estimators=100-200 зазвичай достатньо.

---

## Feature Importance

### Два типи важливості

### 1. Mean Decrease Impurity (MDI)

**Як обчислюється:**
- Сума зменшень impurity (Gini/Entropy) по всіх деревах
- Зважена на кількість зразків

**Формула:**
$$\text{Importance}(f) = \frac{1}{T} \sum_{t=1}^{T} \sum_{n \in \text{splits using } f} \frac{n_{\text{samples}}}{n_{\text{total}}} \Delta I_n$$

**У scikit-learn:**
```python
importances = rf.feature_importances_  # MDI за замовчуванням
```

**Переваги:**
- ✅ Швидко обчислюється
- ✅ Вже є в моделі

**Недоліки:**
- ❌ Bias в сторону числових ознак
- ❌ Bias в сторону ознак з багатьма категоріями

### 2. Permutation Importance

**Як обчислюється:**
- Перемішати одну ознаку → виміряти падіння accuracy
- Більше падіння → ознака важливіша

**Код:**
```python
from sklearn.inspection import permutation_importance

# Обчислити permutation importance
perm_importance = permutation_importance(
    rf_clf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Результати
for i in perm_importance.importances_mean.argsort()[::-1]:
    print(f"{wine.feature_names[i]}: "
          f"{perm_importance.importances_mean[i]:.4f} "
          f"+/- {perm_importance.importances_std[i]:.4f}")
```

**Переваги:**
- ✅ Не має bias в сторону конкретних типів ознак
- ✅ Працює з будь-якою моделлю

**Недоліки:**
- ❌ Повільніше обчислюється

### Візуалізація порівняння

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MDI
indices_mdi = np.argsort(rf_clf.feature_importances_)[::-1]
axes[0].bar(range(X.shape[1]), rf_clf.feature_importances_[indices_mdi])
axes[0].set_xticks(range(X.shape[1]))
axes[0].set_xticklabels([wine.feature_names[i] for i in indices_mdi], 
                        rotation=45, ha='right')
axes[0].set_ylabel('Importance', fontsize=12)
axes[0].set_title('Mean Decrease Impurity (MDI)', 
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Permutation
indices_perm = perm_importance.importances_mean.argsort()[::-1]
axes[1].bar(range(X.shape[1]), 
            perm_importance.importances_mean[indices_perm])
axes[1].errorbar(range(X.shape[1]), 
                perm_importance.importances_mean[indices_perm],
                yerr=perm_importance.importances_std[indices_perm],
                fmt='none', ecolor='black', capsize=3)
axes[1].set_xticks(range(X.shape[1]))
axes[1].set_xticklabels([wine.feature_names[i] for i in indices_perm], 
                        rotation=45, ha='right')
axes[1].set_ylabel('Importance', fontsize=12)
axes[1].set_title('Permutation Importance', 
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Висока точність** | Один з найкращих "out-of-the-box" |
| **Робастність до overfitting** | Усереднення зменшує variance |
| **Універсальність** | Регресія + класифікація |
| **Мало tuning** | Працює добре з defaults |
| **Не потрібна нормалізація** | Працює з різними масштабами |
| **Категоріальні дані** | Обробляє без One-Hot |
| **Нелінійні залежності** | Автоматично виявляє |
| **Feature importance** | Аналіз важливості ознак |
| **OOB error** | Безкоштовна валідація |
| **Паралелізація** | Швидке навчання (n_jobs=-1) |
| **Робастність до викидів** | Менш чутливі |
| **Missing values** | Може працювати (з обмеженнями) |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Інтерпретованість** | Складно пояснити (чорна скринька) |
| **Розмір моделі** | Багато пам'яті (зберігає всі дерева) |
| **Повільні передбачення** | Потрібно пройти всі дерева |
| **Не екстраполює** | Погано за межами train даних |
| **Табличні дані** | Для зображень/тексту → CNN/RNN |
| **Точність** | Gradient Boosting часто точніше |
| **Реал-тайм** | Повільніше за linear models |

---

## Порівняння з іншими моделями

### Random Forest vs Decision Tree

| Критерій | Decision Tree | Random Forest |
|----------|---------------|---------------|
| **Точність** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Overfitting** | Високий ризик | Низький ризик |
| **Інтерпретованість** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Швидкість навчання** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Швидкість передбачення** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Стабільність** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### Random Forest vs Gradient Boosting

| Критерій | Random Forest | Gradient Boosting |
|----------|---------------|-------------------|
| **Точність** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Tuning** | Легкий | Складніший |
| **Overfitting** | Робастний | Може overfitting |
| **Швидкість навчання** | ⭐⭐⭐⭐ (паралельно) | ⭐⭐ (послідовно) |
| **Паралелізація** | ✅ Так | ❌ Складно |
| **Використання** | Baseline, features | Production, Kaggle |

---

## Коли використовувати Random Forest

### Ідеально підходить ✓

- Потрібен **швидкий baseline** з хорошою точністю
- **Табличні дані** (structured data)
- Категоріальні + числові ознаки
- **Feature importance** для аналізу
- Невелика кількість гіперпараметрів для tuning
- **Робастність до шуму** важлива
- Достатньо пам'яті та обчислювальних ресурсів

### Краще використати інше ✗

- **Максимальна точність** → Gradient Boosting (XGBoost, LightGBM, CatBoost)
- **Інтерпретованість** → Decision Tree (одне), Logistic Regression
- **Дуже великі дані** (>10M) → Linear models, LightGBM
- **Зображення/Текст** → Deep Learning (CNN, RNN)
- **Реал-тайм inference** → Linear models (швидші)
- **Мало пам'яті** → Linear models, одне дерево

---

## Практичні поради 💡

1. **Почни з defaults** — n_estimators=100 зазвичай добре
2. **Збільш n_estimators** — до 200-500 для покращення
3. **OOB score** — використовуй для швидкої валідації
4. **n_jobs=-1** — завжди паралелізуй!
5. **max_features='sqrt'** — класифікація (за замовчуванням)
6. **Не обмежуй глибину** — RF робастний до overfitting
7. **Feature importance** — видаляй непотрібні ознаки
8. **Порівняй з Gradient Boosting** — можливо точніше
9. **class_weight='balanced'** для незбалансованих класів
10. **Збережи модель** — `joblib.dump(rf, 'model.pkl')`

---

## Реальний приклад: Передбачення хвороби серця

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

# Синтетичні дані (в реальності використай UCI Heart Disease Dataset)
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(30, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),  # chest pain type
    'trestbps': np.random.randint(90, 200, n_samples),  # blood pressure
    'chol': np.random.randint(120, 400, n_samples),  # cholesterol
    'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar
    'restecg': np.random.randint(0, 3, n_samples),
    'thalach': np.random.randint(70, 200, n_samples),  # max heart rate
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6, n_samples),
}

# Target (симулюємо залежність)
data['target'] = (
    (data['age'] > 55).astype(int) +
    (data['chol'] > 240).astype(int) +
    (data['thalach'] < 120).astype(int) +
    np.random.randint(0, 2, n_samples)
) > 1

df = pd.DataFrame(data)

X = df.drop('target', axis=1)
y = df['target']

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    oob_score=True,
    class_weight='balanced',  # Незбалансовані класи
    n_jobs=-1,
    random_state=42
)

# Навчання
rf.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Оцінка
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("=== Model Performance ===")
print("="*60)
print(f"Train Accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n" + classification_report(y_test, y_pred, 
                                   target_names=['No Disease', 'Disease']))

# Feature Importance
print("\n" + "="*60)
print("=== Top 5 Most Important Features ===")
print("="*60)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importances.head(5).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature Importance
axes[0].barh(importances['feature'][:10], importances['importance'][:10])
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('Top 10 Feature Importances', 
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Number of Trees vs Performance
n_trees = list(range(10, 201, 10))
train_scores_prog = []
test_scores_prog = []

for n in n_trees:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    train_scores_prog.append(rf_temp.score(X_train, y_train))
    test_scores_prog.append(rf_temp.score(X_test, y_test))

axes[1].plot(n_trees, train_scores_prog, 'o-', label='Train', linewidth=2)
axes[1].plot(n_trees, test_scores_prog, 's-', label='Test', linewidth=2)
axes[1].set_xlabel('Number of Trees', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Performance vs Number of Trees', 
                  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Поширені помилки ❌

### 1. Занадто мало дерев

```python
# ❌ НЕПРАВИЛЬНО
rf = RandomForestClassifier(n_estimators=10)  # Занадто мало

# ✅ ПРАВИЛЬНО
rf = RandomForestClassifier(n_estimators=100)  # Мінімум 100
```

### 2. Не використовувати паралелізацію

```python
# ❌ ПОВІЛЬНО
rf = RandomForestClassifier(n_jobs=1)

# ✅ ШВИДКО
rf = RandomForestClassifier(n_jobs=-1)  # Всі ядра
```

### 3. Обмежувати глибину без причини

```python
# ❌ БЕЗ ПОТРЕБИ
rf = RandomForestClassifier(max_depth=5)  # RF робастний до overfitting

# ✅ КРАЩЕ
rf = RandomForestClassifier(max_depth=None)  # Необмежена глибина
```

### 4. Ігнорувати OOB score

```python
# ❌ ВТРАЧЕНА МОЖЛИВІСТЬ
rf = RandomForestClassifier(oob_score=False)

# ✅ ВИКОРИСТОВУЙ
rf = RandomForestClassifier(oob_score=True)
print(f"OOB Score: {rf.oob_score_}")  # Безкоштовна валідація!
```

---

## Збереження та завантаження моделі

```python
import joblib

# Збереження
joblib.dump(rf, 'random_forest_model.pkl')
print("Model saved!")

# Завантаження
rf_loaded = joblib.load('random_forest_model.pkl')
print("Model loaded!")

# Використання
predictions = rf_loaded.predict(X_new)
```

---

## Пов'язані теми

- [[01_Decision_Trees]] — базовий блок RF
- [[03_Gradient_Boosting]] — альтернативний ансамбль
- [[05_Ensemble_Methods]] — теорія ансамблів
- [[06_Feature_Importance]] — аналіз ознак
- [[Cross_Validation]] — оцінка моделей

## Ресурси

- [Scikit-learn: Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Original Paper: Breiman (2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [StatQuest: Random Forest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

---

## Ключові висновки

> Random Forest — це ансамбль Decision Trees, навчених на різних bootstrap samples з випадковим вибором ознак, що об'єднує їх передбачення через voting/averaging.

**Основні принципи:**
- **Bootstrap Aggregating (Bagging)** — різні підмножини даних
- **Feature Randomness** — різні підмножини ознак
- **Majority Voting** (класифікація) або **Averaging** (регресія)
- **OOB Error** — безкоштовна валідація

**Формула (класифікація):**
$$\hat{y} = \text{mode}\{h_1(x), h_2(x), ..., h_T(x)\}$$

**Коли використовувати:**
- Табличні дані + швидкий baseline + робастність = Random Forest ✓
- Максимальна точність на табличних → Gradient Boosting ✓

**Найважливіше:**
- n_estimators=100-200, n_jobs=-1, oob_score=True
- Добре працює "out-of-the-box"
- Для production розгляньте Gradient Boosting

---

#ml #supervised-learning #ensemble #random-forest #bagging #classification #regression #tree-based
