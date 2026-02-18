# SVR (Support Vector Regression)

## Що це?

**Support Vector Regression (SVR)** — це адаптація SVM для **регресії**, яка замість максимізації margin для класифікації, **мінімізує помилку в межах ε-tube** (epsilon-insensitive loss).

**Головна ідея:** знайти функцію, яка має максимум ε відхилення від фактичних значень, використовуючи якомога менше точок (support vectors).

## Навіщо потрібен?

- 📈 **Robust regression** — стійкий до outliers
- 🎯 **ε-insensitive** — ігнорує малі помилки
- 🔧 **Kernel trick** — нелінійні залежності
- 💡 **High-dimensional** — працює при d > n
- ⚡ **Support vectors** — компактна модель

## Коли використовувати?

**Потрібно:**
- **Robust regression** потрібна
- **Outliers** в даних
- **Нелінійні залежності** (з kernels)
- Середні датасети (n = 1k-50k)
- **High-dimensional** дані

**Не потрібно:**
- **Дуже великі датасети** (n > 50k) → повільно
- Простa **лінійна регресія** достатня → Linear Regression
- Потрібна **інтерпретованість** → Linear/Polynomial Regression
- **Швидкість** критична → Linear models

---

## Концепція: ε-insensitive Loss

### Візуалізація ε-tube

\`\`\`
        y
        |     ε-tube
        |  ─────────
    ×   | •  ───────  ← Predicted function
  ×   × |•  ─────────
×   •   •
      × |
        |____________ x

Точки всередині ε-tube: penalty = 0
Точки за межами ε-tube: penalty пропорційно відстані
\`\`\`

### Чому ε-tube?

**Порівняння з іншими loss functions:**

\`\`\`
Squared Error (MSE):        ε-insensitive:
     Loss                        Loss
      |                            |
      |    /                       |    ___/
      |   /                        |___/    
      |  /                         |
      |_/______ Error              |_______ Error
      0                           -ε  0  ε

MSE: всі помилки мають penalty    SVR: малі помилки ігноруються
\`\`\`

**Переваги ε-insensitive:**
- ✅ **Sparse solution** — багато точок всередині tube → не є SV
- ✅ **Robust до noise** — малі помилки не штрафуються
- ✅ **Outlier resistance** — великі помилки штрафуються лінійно (не квадратично)

---

## Математика SVR

### Оптимізаційна задача

**Мета:** Знайти функцію $f(x) = w^T x + b$ яка апроксимує дані з максимум ε відхиленням.

$$\min_{w, b, \xi, \xi^*} \frac{1}{2}||w||^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)$$

subject to:

$$y_i - (w^T x_i + b) \leq \varepsilon + \xi_i$$
$$(w^T x_i + b) - y_i \leq \varepsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

де:
- $\varepsilon$ — ширина tube (tolerance)
- $\xi_i$ — slack variable для верхнього порушення
- $\xi_i^*$ — slack variable для нижнього порушення
- $C$ — regularization parameter

### Інтерпретація

**Компоненти:**
1. $\frac{1}{2}||w||^2$ — regularization (smooth function)
2. $C \sum (\xi_i + \xi_i^*)$ — penalty за помилки > ε

**Slack variables:**
- $\xi_i = 0$ та $\xi_i^* = 0$ → точка всередині ε-tube ✓
- $\xi_i > 0$ → точка вище верхньої межі tube
- $\xi_i^* > 0$ → точка нижче нижньої межі tube

---

## Kernel SVR

### Нелінійна регресія через kernels

**Так само як Kernel SVM, SVR підтримує kernel trick:**

$$f(x) = \sum_{i \in SV} (\alpha_i - \alpha_i^*) K(x_i, x) + b$$

де $K(x_i, x)$ — kernel function.

### Популярні kernels для SVR

1. **Linear:** $K(x, z) = x^T z$
   - Лінійна залежність
   
2. **Polynomial:** $K(x, z) = (x^T z + c)^d$
   - Поліноміальна залежність
   
3. **RBF (Gaussian):** $K(x, z) = \exp(-\gamma ||x - z||^2)$
   - Нелінійна залежність (універсальний)

---

## Код (Python + scikit-learn)

### Базовий приклад: Linear SVR

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Генерація даних
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=10,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Нормалізація (РЕКОМЕНДОВАНО)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# 3. Linear SVR
svr_linear = SVR(
    kernel='linear',
    C=1.0,              # Regularization
    epsilon=0.1         # ε-tube width
)

# 4. Навчання
svr_linear.fit(X_train_scaled, y_train_scaled)

# 5. Передбачення
y_pred_scaled = svr_linear.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 6. Метрики
print("=== Linear SVR ===")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"Support vectors: {len(svr_linear.support_)}")
print(f"SV percentage: {len(svr_linear.support_)/len(X_train)*100:.1f}%")
\`\`\`

### RBF SVR для нелінійних даних

\`\`\`python
# Нелінійні дані
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(200) * 0.1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RBF SVR
svr_rbf = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=0.1
)

svr_rbf.fit(X_train, y_train)

# Передбачення
X_plot = np.linspace(0, 5, 300).reshape(-1, 1)
y_plot = svr_rbf.predict(X_plot)

# Візуалізація
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.5, s=30, label='Train', color='blue')
plt.scatter(X_test, y_test, alpha=0.5, s=50, label='Test', color='green')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='SVR Prediction')

# ε-tube
epsilon = 0.1
plt.fill_between(X_plot.ravel(),
                 y_plot - epsilon,
                 y_plot + epsilon,
                 alpha=0.2, color='red', label=f'ε-tube (ε={epsilon})')

# Support vectors
plt.scatter(X_train[svr_rbf.support_], y_train[svr_rbf.support_],
           s=200, facecolors='none', edgecolors='red',
           linewidth=2, label='Support Vectors')

plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Support Vector Regression (RBF)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nR²: {r2_score(y_test, svr_rbf.predict(X_test)):.4f}")
print(f"Support vectors: {len(svr_rbf.support_)} / {len(X_train)}")
\`\`\`

---

## Гіперпараметри SVR

### Три головні параметри

1. **C** — regularization strength
   - Малий C → гладка функція, більше помилок
   - Великий C → точніша апроксимація, ризик overfitting

2. **ε (epsilon)** — tube width
   - Малий ε → вужчий tube, більше SV
   - Великий ε → ширший tube, менше SV

3. **γ (gamma)** — для RBF kernel
   - Малий γ → smooth prediction
   - Великий γ → wiggly prediction

### Вплив параметрів

\`\`\`
ε малий (0.01):              ε оптимальний (0.1):        ε великий (1.0):
═══════════                  ═══════════                 ═══════════
Вузький tube                 Баланс                      Широкий tube
Багато SV                    ✓ Найкраще                  Мало SV
Точна апроксимація           Robust                      Груба апроксимація

    •  •  •                      •  •  •                     •  •  •
   ────────                     ─────────                   ═══════════
    •  •  •                      •  •  •                     •  •  •
\`\`\`

### Grid Search для SVR

\`\`\`python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1.0],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale']
}

grid_search = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

# Тест
best_svr = grid_search.best_estimator_
test_r2 = best_svr.score(X_test, y_test)
print(f"Test R²: {test_r2:.4f}")
\`\`\`

---

## Порівняння SVR з іншими моделями

\`\`\`python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Нелінійні дані
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Моделі
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR (Linear)': SVR(kernel='linear', C=1.0),
    'SVR (RBF)': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
}

print("="*60)
print("MODEL COMPARISON")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    print(f"{name:20} Train R²: {train_r2:.4f}  Test R²: {test_r2:.4f}")
\`\`\`

---

## Переваги та недоліки SVR

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Robust до outliers** | ε-insensitive loss |
| **Kernel trick** | Нелінійні залежності |
| **Sparse solution** | Тільки support vectors |
| **High-dimensional** | Працює при d > n |
| **Regularization** | Параметр C |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Повільне training** | O(n²) до O(n³) |
| **Гіперпараметри** | C, ε, γ потрібно підбирати |
| **Великі датасети** | n > 50k дуже повільно |
| **Інтерпретованість** | Чорна скринька |
| **Sensitivity до scaling** | Потребує нормалізації |

---

## Порівняння SVR з іншими

### SVR vs Linear Regression

| Критерій | SVR | Linear Regression |
|----------|-----|-------------------|
| **Outliers** | Robust | Чутливий |
| **Loss function** | ε-insensitive | MSE |
| **Швидкість** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Нелінійні** | ✅ З kernels | ❌ Тільки linear |
| **Інтерпретованість** | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### SVR vs Random Forest

| Критерій | SVR | Random Forest |
|----------|-----|---------------|
| **Точність** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Training** | Повільний | Швидкий |
| **Tuning** | Складний | Простий |
| **High-dimensional** | ✅ Працює | ⚠️ Може погано |

---

## Практичні поради 💡

1. **Нормалізуй X та y** — критично для SVR
2. **Почни з RBF kernel** — універсальний
3. **Grid Search** — підбирай C, ε, γ
4. **ε за замовчуванням** — почни з 0.1
5. **Малі датасети** — SVR працює добре
6. **Візуалізуй SV** — якщо > 50% → зменш ε або збільш C
7. **Порівняй з baseline** — Linear Regression, Random Forest
8. **Linear для interpretability** — якщо потрібна простота
9. **Outliers** — SVR robust, але перевір візуально
10. **Cross-validation** — завжди використовуй CV

---

## Приклад: Real Estate Price Prediction

\`\`\`python
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Синтетичні дані
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'size_sqm': np.random.randint(50, 200, n),
    'rooms': np.random.randint(1, 6, n),
    'age_years': np.random.randint(0, 50, n),
    'distance_center_km': np.random.uniform(1, 20, n)
})

# Ціна з нелінійною залежністю
data['price'] = (
    5000 * data['size_sqm'] +
    20000 * data['rooms'] -
    1000 * data['age_years'] -
    5000 * np.log(data['distance_center_km']) +
    np.random.normal(0, 50000, n)
)

X = data.drop('price', axis=1)
y = data['price']

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Нормалізація
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# SVR
svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr.fit(X_train_scaled, y_train_scaled)

# Передбачення
y_pred_scaled = svr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Оцінка
print("=== SVR for Real Estate ===")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
print(f"Support vectors: {len(svr.support_)} / {len(X_train)}")

# CV score
cv_scores = cross_val_score(svr, X_train_scaled, y_train_scaled, cv=5, scoring='r2')
print(f"\nCV R² mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
\`\`\`

---

## Коли використовувати SVR

### Ідеально підходить ✓

- **Robust regression** потрібна
- **Outliers** в даних
- **Нелінійні залежності**
- **High-dimensional** дані
- Середні датасети (n = 1k-50k)
- **Regularization** важлива

### Краще використати інше ✗

- **Дуже великі дані** (n > 50k) → Ridge, Lasso, SGDRegressor
- **Лінійна регресія достатня** → Linear Regression
- **Потрібна швидкість** → Linear models
- **Interpretability** критична → Linear/Polynomial Regression
- **Structured tabular** → Tree-based часто краще

---

## Ключові висновки

> SVR використовує ε-insensitive loss для robust regression з kernel trick для нелінійних залежностей.

**ε-insensitive loss:**
- Точки всередині ε-tube: penalty = 0
- Точки за межами: penalty пропорційно відстані

**Гіперпараметри:**
- **C:** regularization (малий → smooth, великий → accurate)
- **ε:** tube width (малий → вузький, великий → широкий)
- **γ:** для RBF (малий → smooth, великий → wiggly)

**КРИТИЧНО:**
- Нормалізуй X та y
- Grid Search для параметрів
- RBF kernel для нелінійних
- Візуалізуй support vectors

---

#ml #svr #regression #support-vector-regression #epsilon-insensitive #robust-regression
