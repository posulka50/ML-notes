# Random Forest (Випадковий ліс)

## Що це?

**Random Forest** — це **ансамбль** багатьох Decision Trees, які навчаються незалежно на різних підмножинах даних та ознак, а їх передбачення об'єднуються через голосування (класифікація) або усереднення (регресія).

**Головна ідея:** "мудрість натовпу" — багато незалежних простих моделей разом дають кращий результат, ніж одна складна модель.

## Навіщо потрібний?

- 🎯 **Висока точність** — один з найкращих out-of-the-box алгоритмів
- 🛡️ **Робастність** — стійкий до overfitting
- ⚡ **Універсальність** — класифікація та регресія
- 📊 **Feature importance** — автоматичний відбір важливих ознак
- 🚫 **Мінімальна підготовка** — не потребує нормалізації
- 🔧 **Мало гіперпараметрів** — легко налаштувати
- 💪 **Паралелізація** — швидке навчання на багатьох ядрах

## Коли використовувати?

**Потрібно:**
- Потрібна **висока точність** на табличних даних
- **Robust baseline** — швидкий старт
- Мало часу на feature engineering
- Нелінійні залежності
- Важливі feature importance
- Стабільні передбачення

**Не потрібно:**
- **Інтерпретованість критична** → Decision Tree
- Дуже великі дані → XGBoost, LightGBM (швидші)
- Лінійні залежності → Linear models
- Зображення, текст → Neural Networks
- Потрібна калібрована ймовірність → Logistic Regression

---

## Як працює Random Forest?

### Схема ансамблю

```
                    [Тренувальні дані]
                            |
         ┌──────────────────┼──────────────────┐
         |                  |                   |
    Bootstrap 1        Bootstrap 2         Bootstrap N
         |                  |                   |
   [Decision Tree 1]  [Decision Tree 2]  [Decision Tree N]
         |                  |                   |
    Prediction 1       Prediction 2        Prediction N
         |                  |                   |
         └──────────────────┼──────────────────┘
                            ↓
                    [Voting/Averaging]
                            ↓
                  [Final Prediction]
```

### Два ключові компоненти

#### 1. Bagging (Bootstrap Aggregating)

**Bootstrap sampling:** з тренувальних даних (n прикладів) створюємо k підвибірок розміром n **з поверненням**.

```
Оригінальні дані (100 прикладів):
[1, 2, 3, 4, ..., 100]

Bootstrap 1: [1, 1, 5, 7, 10, ..., 99]  ← може повторювати
Bootstrap 2: [2, 3, 3, 15, 20, ..., 100]
Bootstrap 3: [1, 4, 8, 8, 11, ..., 98]
...

Кожна підвибірка: ~63% унікальних прикладів
Out-of-Bag (OOB): ~37% не використані
```

#### 2. Random Feature Selection

**При кожному розбитті вузла:**
- Розглядаємо тільки **випадкову підмножину** з $m$ ознак
- Зазвичай: $m = \sqrt{p}$ для класифікації, $m = p/3$ для регресії
- Це **декорелює** дерева → більше різноманітності

```
Усього ознак: 10
При розбитті розглядаємо: sqrt(10) ≈ 3 випадкові ознаки

Дерево 1, вузол 1: розглядає ознаки [2, 5, 8]
Дерево 1, вузол 2: розглядає ознаки [1, 3, 9]
Дерево 2, вузол 1: розглядає ознаки [4, 6, 7]
...

Результат: різні дерева → різні помилки → усереднення покращує
```

### Об'єднання передбачень

**Класифікація (Voting):**
```
Приклад: 100 дерев передбачають клас для нового об'єкта

Дерево 1 → Клас A
Дерево 2 → Клас B
Дерево 3 → Клас A
...
Дерево 100 → Клас A

Результат:
Клас A: 65 голосів → ПЕРЕМОЖЕЦЬ ✓
Клас B: 35 голосів
```

**Регресія (Averaging):**
```
Дерево 1 → 150.2
Дерево 2 → 148.5
Дерево 3 → 152.1
...
Дерево 100 → 149.8

Фінальне передбачення = mean([150.2, 148.5, ..., 149.8]) = 150.1
```

---

## Математика

### Variance Reduction

**Чому ансамбль краще?**

Для N незалежних моделей з variance σ²:

$$\text{Var}(\text{average}) = \frac{\sigma^2}{N}$$

**Приклад:**
- Одне дерево: variance = 100
- 100 незалежних дерев: variance = 100/100 = 1 ✓

**Проблема:** дерева не повністю незалежні (корелюють).

Для корельованих моделей з кореляцією ρ:

$$\text{Var}(\text{average}) = \rho \sigma^2 + \frac{1-\rho}{N}\sigma^2$$

**Рішення Random Forest:**
- Random feature selection → зменшує кореляцію ρ
- Bagging → збільшує різноманітність

### Out-of-Bag (OOB) Error

**OOB дані:** приклади, які не потрапили у bootstrap вибірку (~37%).

$$\text{OOB Error} = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{y}_i^{\text{OOB}})$$

де $\hat{y}_i^{\text{OOB}}$ — передбачення тільки тих дерев, які не бачили приклад $i$.

**Перевага:** безкоштовна валідація без окремого test set!

---

## Простий приклад: Передбачення захворювання

### Дані

50 пацієнтів:

| Вік | Тиск | Глюкоза | Куріння | Хворий |
|-----|------|---------|---------|--------|
| 45  | 120  | 100     | Так     | Так    |
| 30  | 110  | 85      | Ні      | Ні     |
| 60  | 150  | 130     | Так     | Так    |
| ...

### Random Forest з 3 дерев

**Дерево 1** (навчається на bootstrap 1, розглядає ознаки: Вік, Глюкоза)
```
      Вік >= 50?
      /        \
    Ні          Так
    /            \
Здоровий      Хворий
```

**Дерево 2** (навчається на bootstrap 2, розглядає ознаки: Тиск, Куріння)
```
     Тиск >= 140?
      /         \
    Ні           Так
    /              \
Куріння?         Хворий
/      \
Так    Ні
/        \
Хворий Здоровий
```

**Дерево 3** (навчається на bootstrap 3, розглядає ознаки: Вік, Тиск, Куріння)
```
       Куріння?
       /      \
     Так      Ні
     /          \
Хворий      Вік >= 55?
            /        \
          Так        Ні
          /            \
      Хворий        Здоровий
```

### Передбачення для нового пацієнта

**Пацієнт:** Вік=55, Тиск=145, Глюкоза=115, Куріння=Так

- **Дерево 1:** Вік=55 >= 50 → **Хворий**
- **Дерево 2:** Тиск=145 >= 140 → **Хворий**
- **Дерево 3:** Куріння=Так → **Хворий**

**Голосування:** 3/3 за "Хворий" → **Фінальне передбачення: Хворий** ⚠️

---

## Код (Python + scikit-learn)

### Класифікація

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer

# 1. Завантаження даних
data = load_breast_cancer()
X = data.data
y = data.target

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Модель
rf = RandomForestClassifier(
    n_estimators=100,        # кількість дерев
    max_depth=10,            # максимальна глибина кожного дерева
    min_samples_split=5,     # мін. прикладів для розбиття
    min_samples_leaf=2,      # мін. прикладів у листі
    max_features='sqrt',     # sqrt(p) ознак при розбитті
    bootstrap=True,          # використовувати bootstrap
    oob_score=True,          # обчислювати OOB error
    n_jobs=-1,               # використовувати всі ядра
    random_state=42
)

# 3. Навчання
rf.fit(X_train, y_train)

# 4. Передбачення
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# 5. Оцінка
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 6. Feature Importance
feature_importance = rf.feature_importances_
indices = np.argsort(feature_importance)[::-1]

print("\n=== Top 10 Important Features ===")
for i in range(10):
    print(f"{i+1}. {data.feature_names[indices[i]]}: {feature_importance[indices[i]]:.4f}")
```

### Регресія

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Генерація даних
from sklearn.datasets import make_regression
X, y = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    noise=10,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Модель
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

# Навчання
rf_reg.fit(X_train, y_train)

# Передбачення
y_pred = rf_reg.predict(X_test)

# Оцінка
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"OOB Score: {rf_reg.oob_score_:.4f}")
```

---

## Гіперпараметри

### Основні параметри

| Параметр | Значення за замовчуванням | Рекомендації |
|----------|---------------------------|--------------|
| **n_estimators** | 100 | Більше = краще (але повільніше). Спробуй 100-500 |
| **max_depth** | None | Обмеж (10-30) для запобігання overfitting |
| **min_samples_split** | 2 | Збільш (5-10) для простіших дерев |
| **min_samples_leaf** | 1 | Збільш (2-5) для згладжування |
| **max_features** | 'sqrt' (clf), 'sqrt' (reg) | 'sqrt' для класифікації, 'log2' або p/3 для регресії |
| **bootstrap** | True | Завжди True для Random Forest |
| **oob_score** | False | True для безкоштовної валідації |
| **n_jobs** | None | -1 для використання всіх ядер |

### Підбір гіперпараметрів

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Простір параметрів
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Randomized Search (швидше за Grid Search)
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,              # кількість комбінацій
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print(f"Best CV score: {random_search.best_score_:.4f}")

# Найкраща модель
best_rf = random_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

---

## OOB Score vs Cross-Validation

### Out-of-Bag Error

```python
# OOB score (безкоштовно під час навчання)
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_:.4f}")

# OOB передбачення для кожного прикладу
oob_predictions = rf.oob_decision_function_
print(f"OOB predictions shape: {oob_predictions.shape}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Cross-validation (точніше, але повільніше)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')

print(f"CV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### Порівняння

| Метод | Швидкість | Точність оцінки | Використання |
|-------|-----------|-----------------|--------------|
| **OOB Score** | ✅ Швидко (під час навчання) | ⚠️ Добра | Швидка оцінка, великі дані |
| **Cross-Validation** | ❌ Повільно (навчає k разів) | ✅ Найточніша | Остаточна оцінка, підбір параметрів |

**Рекомендація:**
- **OOB** для швидкого моніторингу та великих даних
- **CV** для фінального підбору гіперпараметрів

---

## Feature Importance

### Розрахунок

**Важливість ознаки** = середнє зменшення impurity при розбитті по цій ознаці, усереднене по всіх деревах.

```python
# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Стандартне відхилення (між деревами)
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

# Візуалізація
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), importances[indices], 
        yerr=std[indices], align='center')
plt.xticks(range(X.shape[1]), 
           [data.feature_names[i] for i in indices], 
           rotation=90)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importance with Error Bars', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Відбір найважливіших ознак
n_important = 10
important_features = indices[:n_important]
print(f"Selected {n_important} most important features")
print([data.feature_names[i] for i in important_features])
```

### Permutation Importance

**Більш надійний метод:**

```python
from sklearn.inspection import permutation_importance

# Обчислення
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Результати
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.boxplot(perm_importance.importances[sorted_idx].T,
            labels=[data.feature_names[i] for i in sorted_idx],
            vert=False)
plt.xlabel('Permutation Importance', fontsize=12)
plt.title('Permutation Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Переваги Permutation Importance:**
- ✅ Не має bias до ознак з багатьма значеннями
- ✅ Працює на test даних (вимірює реальну важливість)
- ✅ Model-agnostic (працює для будь-якої моделі)

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Висока точність** | Один з найкращих out-of-the-box алгоритмів |
| **Robust** | Стійкий до overfitting (в порівнянні з одним деревом) |
| **Універсальність** | Класифікація + регресія |
| **Не потребує нормалізації** | Стійкий до масштабу ознак |
| **Feature importance** | Автоматичний відбір важливих ознак |
| **OOB validation** | Безкоштовна оцінка без test set |
| **Паралелізація** | Швидке навчання на багатьох ядрах |
| **Мікс-дані** | Числові + категоріальні |
| **Мало гіперпараметрів** | Легко налаштувати |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Інтерпретованість** | Важче пояснити ніж одне дерево |
| **Розмір моделі** | Багато дерев → багато пам'яті |
| **Повільне передбачення** | Потрібно пройти всі дерева |
| **Екстраполяція** | Не може передбачати за межами train даних |
| **Bias до категоріальних** | Ознаки з багатьма значеннями мають вищий importance |
| **Імбаланс класів** | Потребує додаткової обробки |
| **Не оптимальний для дуже великих даних** | XGBoost/LightGBM швидші |

---

## Random Forest vs Decision Tree

| Критерій | Decision Tree | Random Forest |
|----------|---------------|---------------|
| **Точність** | ⚠️ Середня | ✅ Висока |
| **Overfitting** | ⚠️ Високий ризик | ✅ Низький ризик |
| **Стабільність** | ❌ Нестабільний | ✅ Стабільний |
| **Інтерпретованість** | ✅ Висока | ⚠️ Середня |
| **Швидкість навчання** | ✅ Швидко | ⚠️ Повільніше |
| **Швидкість передбачення** | ✅ Дуже швидко | ⚠️ Повільніше |
| **Розмір моделі** | ✅ Малий | ❌ Великий |

### Приклад порівняння

```python
from sklearn.tree import DecisionTreeClassifier

# Одне дерево
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_acc = tree.score(X_test, y_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)

print(f"Decision Tree Accuracy: {tree_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"Improvement: {(rf_acc - tree_acc) * 100:.2f}%")
```

**Типові результати:**
```
Decision Tree Accuracy: 0.8800
Random Forest Accuracy: 0.9500
Improvement: 7.00%
```

---

## Незбалансовані класи

### Проблема

```python
# Незбалансовані дані
# Клас 0: 9000 прикладів
# Клас 1: 1000 прикладів

# Random Forest може ігнорувати minority class
```

### Рішення

#### 1. Class Weights

```python
# Автоматичний підбір ваг
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Автоматично
    random_state=42
)

# Або вручну
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight={0: 1, 1: 9},  # Клас 1 в 9 разів важливіший
    random_state=42
)

rf.fit(X_train, y_train)
```

#### 2. Balanced Random Forest

```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Автоматично балансує кожен bootstrap
brf = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='auto',  # Автоматичне балансування
    replacement=True,
    random_state=42
)

brf.fit(X_train, y_train)
```

#### 3. Resampling перед навчанням

```python
from imblearn.over_sampling import SMOTE

# SMOTE для oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Навчання на збалансованих даних
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)
```

---

## Візуалізація окремого дерева

```python
from sklearn.tree import plot_tree

# Візуалізація одного дерева з лісу
estimator = rf.estimators_[0]  # Перше дерево

plt.figure(figsize=(20, 10))
plot_tree(
    estimator,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title('First Decision Tree from Random Forest', 
          fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Порівняння з іншими ансамблями

| Метод | Підхід | Переваги | Недоліки |
|-------|--------|----------|----------|
| **Random Forest** | Паралельні незалежні дерева | Швидко, robust, мало overfitting | Велика модель |
| **Gradient Boosting** | Послідовні дерева виправляють помилки | Вища точність | Повільніше, overfitting |
| **AdaBoost** | Послідовно зважує помилки | Простота | Чутливий до outliers |
| **XGBoost** | Оптимізований Gradient Boosting | Дуже швидко, висока точність | Складніше налаштувати |

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import time

# Порівняння
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    accuracy = model.score(X_test, y_test)
    
    results[name] = {
        'Accuracy': accuracy,
        'Train Time': train_time
    }
    
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Train Time: {train_time:.2f}s\n")
```

---

## Практичні поради 💡

1. **Почни з дефолтами** — 100 дерев, sqrt features працює добре
2. **n_jobs=-1** — використовуй всі ядра для швидкості
3. **OOB score** — безкоштовна валідація для великих даних
4. **Не перебільшуй n_estimators** — після 100-500 покращення мінімальне
5. **max_depth=10-30** — обмеж для запобігання overfitting
6. **Feature importance** — використай для відбору ознак
7. **Не нормалізуй** — Random Forest не потребує нормалізації
8. **Class weights** — для незбалансованих класів
9. **Baseline** — відмінний стартовий алгоритм
10. **Порівнюй з XGBoost** — якщо RF недостатньо

---

## Коли використовувати Random Forest

### Ідеально підходить ✓

- Потрібна **висока точність** на табличних даних
- **Robust baseline** без багато feature engineering
- Нелінійні залежності
- Мікс числових та категоріальних ознак
- **Feature importance** важливі
- Середні датасети (10k-1M прикладів)
- Паралелізація доступна (багато ядер)

### Краще використати інше ✗

- **Інтерпретованість критична** → Decision Tree
- **Дуже великі дані** → XGBoost, LightGBM
- Лінійні залежності → Linear models
- **Зображення, текст, послідовності** → Neural Networks
- Потрібна калібрована ймовірність → Logistic Regression + calibration
- **Екстремально мало даних** → Regularized linear models

---

## Поширені помилки ❌

### 1. Занадто багато дерев без причини

```python
# ❌ НЕПРАВИЛЬНО (марна трата часу)
rf = RandomForestClassifier(n_estimators=10000)

# ✅ ПРАВИЛЬНО (достатньо 100-500)
rf = RandomForestClassifier(n_estimators=100)
```

### 2. Не використовувати паралелізацію

```python
# ❌ НЕПРАВИЛЬНО (повільно)
rf = RandomForestClassifier(n_estimators=100)

# ✅ ПРАВИЛЬНО (використати всі ядра)
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
```

### 3. Нормалізувати дані (непотрібно)

```python
# ❌ НЕПОТРІБНО
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf.fit(X_scaled, y)

# ✅ ПРАВИЛЬНО (використовуй оригінальні дані)
rf.fit(X, y)
```

### 4. Забути про random_state

```python
# ❌ НЕПРАВИЛЬНО (нестабільні результати)
rf = RandomForestClassifier(n_estimators=100)

# ✅ ПРАВИЛЬНО (відтворювані результати)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```

---

## Збереження та завантаження

```python
import joblib

# Збереження моделі
joblib.dump(rf, 'random_forest_model.pkl')

# Завантаження
loaded_rf = joblib.load('random_forest_model.pkl')

# Використання
predictions = loaded_rf.predict(X_new)
```

---

## Пов'язані теми

- [[01_Decision_Trees]] — базовий будівельний блок
- [[03_Gradient_Boosting]] — альтернативний ансамбль
- [[05_Ensemble_Methods]] — теорія ансамблів
- [[06_Feature_Importance]] — відбір ознак
- [[Cross_Validation]] — оцінка моделі
- [[Hyperparameter_Tuning]] — підбір параметрів

## Ресурси

- [Scikit-learn: Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Leo Breiman: Random Forests (Original Paper)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [Random Forest Interpretation](https://explained.ai/rf-importance/)

---

## Ключові висновки

> Random Forest — це ансамбль багатьох Decision Trees, навчених на різних підвибірках даних та ознак, з об'єднанням передбачень через голосування або усереднення.

**Основні принципи:**
- **Bagging:** bootstrap sampling для різноманітності даних
- **Random feature selection:** випадкові ознаки для декореляції дерев
- **Voting/Averaging:** об'єднання передбачень зменшує variance

**Формула variance reduction:**
$$\text{Var}(\text{average}) = \rho \sigma^2 + \frac{1-\rho}{N}\sigma^2$$

**Коли використовувати:**
- Висока точність + robust + табличні дані = Random Forest ✓
- Більше точності потрібно → XGBoost/LightGBM
- Інтерпретованість важлива → Decision Tree

**Налаштування:**
- n_estimators: 100-500
- max_depth: 10-30
- max_features: 'sqrt' (classification), 'log2' або p/3 (regression)
- n_jobs: -1 (паралелізація)

---

#ml #supervised-learning #ensemble #random-forest #bagging #classification #regression
