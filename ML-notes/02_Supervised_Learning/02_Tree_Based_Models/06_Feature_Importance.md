# Feature Importance (Важливість ознак)

## Що це?

**Feature Importance** — це метрика, яка показує, наскільки кожна ознака корисна для передбачень моделі. Допомагає зрозуміти, які ознаки найбільше впливають на рішення моделі.

**Головна ідея:** визначити, які ознаки найбільше впливають на якість передбачень, щоб краще зрозуміти дані та модель.

## Навіщо потрібно?

- 🔍 **Розуміння даних** — які фактори найважливіші?
- 🎯 **Feature selection** — видалити непотрібні ознаки
- 💡 **Інтерпретація моделі** — пояснити передбачення
- 📊 **Покращення моделі** — фокус на важливих ознаках
- 🔧 **Debugging** — знайти проблеми з даними
- 💼 **Business insights** — практичні рекомендації

## Коли використовувати?

**Потрібно:**
- Розуміти, **чому** модель робить передбачення
- **Feature selection** перед навчанням
- Бізнес потребує **інтерпретації**
- Багато ознак (feature engineering)
- **Debugging** моделі

**Не потрібно:**
- Модель вже ідеально працює і не потребує пояснень
- Deep Learning (складно інтерпретувати)
- Дуже мало ознак (всі очевидні)

---

## Типи Feature Importance

```
Feature Importance Methods
       |
       |--- Model-Specific
       |     |
       |     |--- Mean Decrease Impurity (MDI)
       |     |     └── Decision Trees, Random Forest, Gradient Boosting
       |     |
       |     |--- Coefficients
       |     |     └── Linear/Logistic Regression
       |     |
       |     |--- SHAP Values
       |           └── Any model
       |
       |--- Model-Agnostic
             |
             |--- Permutation Importance
             |--- Drop-Column Importance
             |--- LIME
             |--- Partial Dependence Plots
```

---

## 1. Mean Decrease Impurity (MDI)

### Що це?

**Для tree-based моделей:** сума зменшень impurity (Gini/Entropy) по всіх деревах, зважена на кількість зразків.

### Формула

$$\text{Importance}(f) = \frac{\sum_{t \in \text{trees}} \sum_{s \in \text{splits using } f} n_s \Delta I_s}{\sum_{t \in \text{trees}} \sum_{s \in \text{all splits}} n_s \Delta I_s}$$

де:
- $n_s$ — кількість зразків у вузлі $s$
- $\Delta I_s$ — зменшення impurity після розбиття

### Властивості

- ✅ Швидко обчислюється
- ✅ Вже є в моделі (`.feature_importances_`)
- ❌ Bias в сторону числових ознак
- ❌ Bias в сторону high-cardinality категоріальних ознак
- ❌ Може показувати importance навіть для irrelevant features

### Код (scikit-learn)

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Модель
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance (MDI)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Виведення
print("=== Feature Importance (MDI) ===")
for i in range(len(importances)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), 
           [feature_names[i] for i in indices], 
           rotation=45, ha='right')
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importances (MDI)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Приклад результату

```
=== Feature Importance (MDI) ===
1. age: 0.2845
2. income: 0.2134
3. credit_score: 0.1876
4. employment_length: 0.1245
5. debt_to_income: 0.0987
6. num_accounts: 0.0456
7. region: 0.0321
8. education: 0.0136
```

---

## 2. Permutation Importance

### Що це?

**Model-agnostic метод:** перемішуємо одну ознаку і дивимось, наскільки погіршується модель.

### Алгоритм

1. Обчислити baseline метрику (accuracy, R²) на валідаційних даних
2. Для кожної ознаки:
   - Перемішати (shuffle) значення ознаки
   - Обчислити метрику на перемішаних даних
   - Importance = baseline - permuted metric
3. Повторити кілька разів для стабільності

### Формула

$$\text{Importance}(f) = \text{Score}_{\text{original}} - \text{Score}_{\text{permuted } f}$$

### Властивості

- ✅ Model-agnostic (працює з будь-якою моделлю)
- ✅ Не має bias до типів ознак
- ✅ Відображає реальний вплив на передбачення
- ❌ Повільніше обчислюється
- ❌ Потребує валідаційний set

### Код

```python
from sklearn.inspection import permutation_importance

# Обчислити permutation importance
perm_importance = permutation_importance(
    rf,              # Модель
    X_test,          # Валідаційні дані
    y_test,          # Labels
    n_repeats=10,    # Повторення для стабільності
    random_state=42,
    n_jobs=-1
)

# Результати
print("\n=== Permutation Importance ===")
for i in perm_importance.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]}: "
          f"{perm_importance.importances_mean[i]:.4f} "
          f"+/- {perm_importance.importances_std[i]:.4f}")

# Візуалізація з error bars
sorted_idx = perm_importance.importances_mean.argsort()[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(sorted_idx)), 
        perm_importance.importances_mean[sorted_idx])
ax.errorbar(perm_importance.importances_mean[sorted_idx],
            range(len(sorted_idx)),
            xerr=perm_importance.importances_std[sorted_idx],
            fmt='none', ecolor='black', capsize=3)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels([feature_names[i] for i in sorted_idx])
ax.set_xlabel('Permutation Importance', fontsize=12)
ax.set_title('Permutation Importance with Error Bars', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

---

## 3. Drop-Column Importance

### Що це?

**Найінтуїтивніший метод:** видаляємо ознаку і дивимось, наскільки погіршується модель.

### Алгоритм

1. Навчити модель на всіх ознаках → baseline score
2. Для кожної ознаки:
   - Видалити ознаку з даних
   - Навчити модель без цієї ознаки
   - Обчислити score
   - Importance = baseline score - score without feature
3. Чим більше падіння → важливіша ознака

### Властивості

- ✅ Дуже інтуїтивно
- ✅ Реально відображає вплив видалення ознаки
- ❌ Дуже повільно (потрібно $p$ разів навчати модель)
- ❌ Не враховує взаємодії між ознаками

### Код

```python
from sklearn.base import clone
from sklearn.metrics import accuracy_score

# Baseline score (всі ознаки)
baseline_score = rf.score(X_test, y_test)

# Drop-column importance
drop_importances = {}

for i, feature in enumerate(feature_names):
    # Видалити ознаку
    X_train_dropped = np.delete(X_train, i, axis=1)
    X_test_dropped = np.delete(X_test, i, axis=1)
    
    # Навчити модель без ознаки
    model_dropped = clone(rf)
    model_dropped.fit(X_train_dropped, y_train)
    
    # Score без ознаки
    score_dropped = model_dropped.score(X_test_dropped, y_test)
    
    # Importance = падіння score
    importance = baseline_score - score_dropped
    drop_importances[feature] = importance
    
    print(f"{feature}: {importance:.4f}")

# Сортування
sorted_features = sorted(drop_importances.items(), 
                        key=lambda x: x[1], 
                        reverse=True)

print("\n=== Top 5 Most Important Features (Drop-Column) ===")
for feature, importance in sorted_features[:5]:
    print(f"{feature}: {importance:.4f}")
```

---

## 4. Coefficients (Linear Models)

### Що це?

**Для лінійних моделей:** коефіцієнти показують вплив кожної ознаки.

### Формула

Для Linear Regression:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p$$

**Інтерпретація $\beta_j$:**
- Збільшення $x_j$ на 1 одиницю → зміна $y$ на $\beta_j$

### Важливо: Normalization!

**Проблема:** Якщо ознаки в різних масштабах, коефіцієнти не порівнянні.

**Рішення:** Нормалізувати дані перед навчанням.

### Код

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Нормалізація
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Логістична регресія
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Coefficients
coefficients = lr.coef_[0]
abs_coefficients = np.abs(coefficients)

# Сортування за абсолютним значенням
indices = np.argsort(abs_coefficients)[::-1]

print("=== Feature Importance (Coefficients) ===")
for i in indices:
    print(f"{feature_names[i]}: {coefficients[i]:.4f}")

# Візуалізація
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['red' if c < 0 else 'blue' for c in coefficients[indices]]
ax.barh(range(len(coefficients)), coefficients[indices], color=colors)
ax.set_yticks(range(len(coefficients)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Feature Coefficients (Logistic Regression)', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

---

## 5. SHAP Values

### Що це?

**SHapley Additive exPlanations** — метод з теорії ігор для справедливого розподілу "заслуг" між ознаками.

### Концепція

Відповідь на питання: **"Який вклад кожної ознаки у передбачення для конкретного зразка?"**

### Властивості Shapley Values

1. **Local Accuracy:** сума SHAP values = передбачення - baseline
2. **Missingness:** якщо ознака відсутня, її SHAP = 0
3. **Consistency:** якщо ознака стає кориснішою, SHAP не зменшується

### Візуалізації SHAP

#### Summary Plot

Показує важливість та напрямок впливу кожної ознаки.

#### Force Plot

Показує, як кожна ознака впливає на конкретне передбачення.

#### Dependence Plot

Показує залежність між значенням ознаки та її SHAP value.

### Код

```python
import shap

# Створити SHAP explainer
explainer = shap.TreeExplainer(rf)

# Обчислити SHAP values
shap_values = explainer.shap_values(X_test)

# Для бінарної класифікації беремо клас 1
if len(shap_values) == 2:
    shap_values = shap_values[1]

# 1. Summary Plot (Feature Importance)
shap.summary_plot(shap_values, X_test, 
                  feature_names=feature_names,
                  plot_type="bar")

# 2. Summary Plot (Detailed)
shap.summary_plot(shap_values, X_test, 
                  feature_names=feature_names)

# 3. Force Plot (для конкретного зразка)
shap.initjs()
shap.force_plot(explainer.expected_value[1], 
                shap_values[0], 
                X_test[0],
                feature_names=feature_names)

# 4. Dependence Plot (для конкретної ознаки)
shap.dependence_plot("age", shap_values, X_test,
                     feature_names=feature_names)
```

**Переваги SHAP:**
- ✅ Теоретично обґрунтовано
- ✅ Працює з будь-яким моделлю
- ✅ Local explanations (для кожного зразка)
- ✅ Візуалізації дуже інформативні

**Недоліки SHAP:**
- ❌ Обчислювально дорого (особливо для великих даних)
- ❌ Складніше інтерпретувати

---

## 6. LIME (Local Interpretable Model-agnostic Explanations)

### Що це?

**LIME** пояснює передбачення моделі для **конкретного зразка**, апроксимуючи складну модель простою (linear) локально.

### Алгоритм

1. Взяти зразок для пояснення
2. Згенерувати "сусідів" (perturbations)
3. Отримати передбачення для сусідів
4. Навчити просту модель (linear) на сусідах
5. Коефіцієнти простої моделі = пояснення

### Код

```python
import lime
import lime.lime_tabular

# Створити LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)

# Пояснити передбачення для зразка
i = 0  # Індекс зразка
exp = explainer.explain_instance(
    X_test[i], 
    rf.predict_proba,
    num_features=10
)

# Показати пояснення
exp.show_in_notebook(show_table=True)

# Або як текст
print(exp.as_list())

# Візуалізація
fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
```

---

## Порівняння методів

| Метод | Швидкість | Model-Agnostic | Global/Local | Bias | Інтуїтивність |
|-------|-----------|----------------|--------------|------|---------------|
| **MDI** | ⭐⭐⭐⭐⭐ | ❌ (Tree-based) | Global | ⚠️ Є | ⭐⭐⭐⭐ |
| **Permutation** | ⭐⭐⭐ | ✅ | Global | ❌ Немає | ⭐⭐⭐⭐⭐ |
| **Drop-Column** | ⭐ | ✅ | Global | ❌ Немає | ⭐⭐⭐⭐⭐ |
| **Coefficients** | ⭐⭐⭐⭐⭐ | ❌ (Linear) | Global | ❌ Немає | ⭐⭐⭐⭐⭐ |
| **SHAP** | ⭐⭐ | ✅ | Both | ❌ Немає | ⭐⭐⭐ |
| **LIME** | ⭐⭐ | ✅ | Local | ❌ Немає | ⭐⭐⭐⭐ |

---

## Feature Selection на основі Importance

### 1. Threshold-based Selection

```python
from sklearn.feature_selection import SelectFromModel

# Вибрати ознаки з importance > threshold
selector = SelectFromModel(rf, threshold='median')  # або 'mean', 0.1, etc.
selector.fit(X_train, y_train)

# Трансформувати дані
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Які ознаки вибрані?
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
print(f"Selected {len(selected_features)} features:")
print(selected_features)

# Навчити модель на вибраних ознаках
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

print(f"\nOriginal features: {rf.score(X_test, y_test):.4f}")
print(f"Selected features: {rf_selected.score(X_test_selected, y_test):.4f}")
```

### 2. Iterative Feature Selection

```python
# Послідовно видаляти найменш важливі ознаки
importances = rf.feature_importances_
threshold_values = np.linspace(0, importances.max(), 20)

scores = []
n_features_list = []

for threshold in threshold_values:
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    if X_train_sel.shape[1] == 0:
        continue
    
    rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_temp.fit(X_train_sel, y_train)
    score = rf_temp.score(X_test_sel, y_test)
    
    scores.append(score)
    n_features_list.append(X_train_sel.shape[1])

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(n_features_list, scores, 'o-', linewidth=2)
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Accuracy vs Number of Features', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Оптимальна кількість ознак
optimal_idx = np.argmax(scores)
print(f"Optimal number of features: {n_features_list[optimal_idx]}")
print(f"Best accuracy: {scores[optimal_idx]:.4f}")
```

---

## Практичні поради 💡

### 1. Використовуй кілька методів

```python
# Порівняй різні методи
print("=== MDI ===")
print_top_features(rf.feature_importances_, feature_names)

print("\n=== Permutation ===")
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10)
print_top_features(perm_imp.importances_mean, feature_names)
```

### 2. Нормалізуй для коефіцієнтів

```python
# ЗАВЖДИ нормалізуй перед лінійними моделями
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 3. Візуалізуй з error bars

```python
# Для permutation importance
plt.errorbar(x, importances_mean, yerr=importances_std)
```

### 4. Перевіряй stability

```python
# Навчи модель кілька разів
importances_list = []
for i in range(10):
    rf = RandomForestClassifier(random_state=i)
    rf.fit(X_train, y_train)
    importances_list.append(rf.feature_importances_)

# Середнє та std
mean_importance = np.mean(importances_list, axis=0)
std_importance = np.std(importances_list, axis=0)
```

### 5. Domain knowledge

**Не покладайся тільки на числа!**
- Перевір, чи мають сенс важливі ознаки
- Консультуйся з експертами
- Подумай про причинно-наслідкові зв'язки

---

## Повний приклад: Аналіз важливості ознак

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Завантаження даних
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.unique(y)}")

# 1. Random Forest с MDI
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

mdi_importances = rf.feature_importances_
mdi_indices = np.argsort(mdi_importances)[::-1]

print("\n" + "="*70)
print("1. MEAN DECREASE IMPURITY (MDI)")
print("="*70)
for i in range(5):
    idx = mdi_indices[i]
    print(f"{i+1}. {feature_names[idx]}: {mdi_importances[idx]:.4f}")

# 2. Permutation Importance
perm_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)

perm_indices = perm_importance.importances_mean.argsort()[::-1]

print("\n" + "="*70)
print("2. PERMUTATION IMPORTANCE")
print("="*70)
for i in range(5):
    idx = perm_indices[i]
    print(f"{i+1}. {feature_names[idx]}: "
          f"{perm_importance.importances_mean[idx]:.4f} "
          f"+/- {perm_importance.importances_std[idx]:.4f}")

# 3. Linear Model Coefficients
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=10000, random_state=42)
lr.fit(X_train_scaled, y_train)

coefficients = np.abs(lr.coef_[0])
coef_indices = np.argsort(coefficients)[::-1]

print("\n" + "="*70)
print("3. LOGISTIC REGRESSION COEFFICIENTS")
print("="*70)
for i in range(5):
    idx = coef_indices[i]
    print(f"{i+1}. {feature_names[idx]}: {coefficients[idx]:.4f}")

# Візуалізації
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. MDI
top_n = 15
axes[0, 0].barh(range(top_n), mdi_importances[mdi_indices[:top_n]][::-1])
axes[0, 0].set_yticks(range(top_n))
axes[0, 0].set_yticklabels([feature_names[i] for i in mdi_indices[:top_n]][::-1])
axes[0, 0].set_xlabel('Importance', fontsize=11)
axes[0, 0].set_title('Mean Decrease Impurity (MDI)', 
                     fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# 2. Permutation with error bars
axes[0, 1].barh(range(top_n), 
                perm_importance.importances_mean[perm_indices[:top_n]][::-1])
axes[0, 1].errorbar(
    perm_importance.importances_mean[perm_indices[:top_n]][::-1],
    range(top_n),
    xerr=perm_importance.importances_std[perm_indices[:top_n]][::-1],
    fmt='none', ecolor='black', capsize=3
)
axes[0, 1].set_yticks(range(top_n))
axes[0, 1].set_yticklabels([feature_names[i] for i in perm_indices[:top_n]][::-1])
axes[0, 1].set_xlabel('Importance', fontsize=11)
axes[0, 1].set_title('Permutation Importance', 
                     fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Coefficients
axes[1, 0].barh(range(top_n), coefficients[coef_indices[:top_n]][::-1])
axes[1, 0].set_yticks(range(top_n))
axes[1, 0].set_yticklabels([feature_names[i] for i in coef_indices[:top_n]][::-1])
axes[1, 0].set_xlabel('Absolute Coefficient', fontsize=11)
axes[1, 0].set_title('Logistic Regression Coefficients', 
                     fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Comparison of methods
comparison_df = pd.DataFrame({
    'Feature': feature_names,
    'MDI': mdi_importances / mdi_importances.max(),  # Normalize
    'Permutation': perm_importance.importances_mean / 
                   perm_importance.importances_mean.max(),
    'Coefficients': coefficients / coefficients.max()
})

top_features_union = list(set(
    list(mdi_indices[:10]) + 
    list(perm_indices[:10]) + 
    list(coef_indices[:10])
))

comparison_subset = comparison_df.iloc[top_features_union]
comparison_subset = comparison_subset.set_index('Feature')

comparison_subset.plot(kind='barh', ax=axes[1, 1], width=0.8)
axes[1, 1].set_xlabel('Normalized Importance', fontsize=11)
axes[1, 1].set_title('Comparison of Methods (Top Features)', 
                     fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=10, loc='lower right')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Консенсус: які ознаки важливі за всіма методами?
print("\n" + "="*70)
print("CONSENSUS: TOP FEATURES BY ALL METHODS")
print("="*70)

top_k = 10
consensus = set(mdi_indices[:top_k]) & \
            set(perm_indices[:top_k]) & \
            set(coef_indices[:top_k])

print(f"Features in top-{top_k} of all three methods:")
for idx in consensus:
    print(f"- {feature_names[idx]}")

if len(consensus) == 0:
    print("No features in top-10 of all methods!")
    print("\nFeatures in top-10 of at least 2 methods:")
    for idx in top_features_union:
        count = 0
        if idx in mdi_indices[:top_k]: count += 1
        if idx in perm_indices[:top_k]: count += 1
        if idx in coef_indices[:top_k]: count += 1
        if count >= 2:
            print(f"- {feature_names[idx]} (in {count} methods)")
```

---

## Поширені помилки ❌

### 1. Порівнювати коефіцієнти без нормалізації

```python
# ❌ НЕПРАВИЛЬНО
lr.fit(X_train, y_train)  # Без scaling
coefficients = lr.coef_

# ✅ ПРАВИЛЬНО
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
lr.fit(X_scaled, y_train)
coefficients = lr.coef_
```

### 2. Використовувати MDI для категоріальних high-cardinality

```python
# MDI має bias для high-cardinality ознак!
# ✅ Використовуй Permutation Importance замість
```

### 3. Довіряти тільки одному методу

```python
# ❌ НЕ покладайся тільки на MDI
# ✅ Порівняй MDI, Permutation, та інші методи
```

### 4. Ігнорувати domain knowledge

```python
# Якщо "вік пацієнта" найважливіша ознака для діагнозу раку,
# але domain expert каже, що це дивно → перевір дані!
```

---

## Пов'язані теми

- [[01_Decision_Trees]] — MDI importance
- [[02_Random_Forest]] — feature importance
- [[03_Gradient_Boosting]] — feature importance
- [[Feature_Selection]] — вибір ознак
- [[Feature_Engineering]] — створення ознак

## Ресурси

- [Scikit-learn: Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://github.com/marcotcr/lime)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

---

## Ключові висновки

> Feature Importance визначає, які ознаки найбільше впливають на передбачення моделі, допомагаючи зрозуміти дані та покращити модель.

**Основні методи:**
- **MDI** — швидко, tree-based, має bias
- **Permutation** — model-agnostic, без bias, повільніше
- **Drop-Column** — інтуїтивно, дуже повільно
- **Coefficients** — для linear models, потребує scaling
- **SHAP** — теоретично обґрунтовано, local + global
- **LIME** — local explanations, model-agnostic

**Практичні рекомендації:**
- Використовуй **кілька методів** для перевірки
- **Нормалізуй** дані для linear models
- **Permutation Importance** — найнадійніший model-agnostic метод
- **SHAP** — для детального аналізу
- **Domain knowledge** важливіша за числа

**Коли використовувати:**
- Feature selection = Permutation Importance ✓
- Quick check = MDI (tree-based) ✓
- Deep analysis = SHAP ✓
- Local explanations = LIME ✓

**Найважливіше:**
- Не довіряй одному методу
- Візуалізуй результати
- Перевіряй за допомогою domain knowledge
- Стабільність > абсолютні значення

---

#ml #feature-importance #interpretability #feature-selection #explainability #shap #lime #permutation #tree-based
