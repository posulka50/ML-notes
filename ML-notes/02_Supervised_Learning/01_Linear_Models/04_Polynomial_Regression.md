# Polynomial Regression (Поліноміальна регресія)

## Що це?

**Поліноміальна регресія** — це розширення лінійної регресії, яке дозволяє моделювати **нелінійні залежності** шляхом додавання поліноміальних членів (степенів) вхідних ознак.

**Головна ідея:** перетворити нелінійну задачу в лінійну через feature engineering — створення нових ознак як степенів оригінальних.

## Навіщо потрібна?

- 📈 **Нелінійні залежності** — моделювання кривих, парабол, S-подібних залежностей
- 🔄 **Використання лінійної регресії** — ті ж алгоритми, інша підготовка даних
- 🎯 **Кращі передбачення** — коли лінійна модель занадто проста
- 📊 **Інтерпретованість** — зрозуміліша за нейронні мережі
- 🚀 **Швидкість** — швидше за складні нелінійні моделі

## Коли використовувати?

**Потрібно:**
- Залежність **явно нелінійна** (крива, парабола)
- **Scatter plot** показує нелінійний патерн
- Лінійна регресія дає **низький R²**
- **Residual plot** показує систематичний патерн
- Потрібна **інтерпретованість** (на відміну від нейронних мереж)

**Не потрібно:**
- Залежність **лінійна** → Linear Regression
- Дуже складні нелінійні залежності → Random Forest, XGBoost, Neural Networks
- Багато ознак + високі степені → curse of dimensionality
- Категоріальна змінна → Logistic Regression

---

## Математика

### Проста поліноміальна регресія (1 ознака)

**Степінь 2 (квадратична):**
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon$$

**Степінь 3 (кубічна):**
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \varepsilon$$

**Степінь d (загальний випадок):**
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_d x^d + \varepsilon$$

### Множинна поліноміальна регресія (багато ознак)

**Для двох ознак, степінь 2:**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2 + \varepsilon$$

**Загальний випадок:**
- Включає всі можливі комбінації степенів до d
- **Interaction terms:** $x_1 x_2, x_1 x_2^2$, тощо

### Кількість ознак після трансформації

Для $p$ оригінальних ознак та степеня $d$:

$$\text{Кількість нових ознак} = \binom{p + d}{d} - 1$$

**Приклад:**
- $p=2$ ознаки, $d=2$ степінь: $\binom{4}{2} - 1 = 5$ нових ознак
  - $x_1, x_2, x_1^2, x_2^2, x_1 x_2$

- $p=3$ ознаки, $d=3$ степінь: $\binom{6}{3} - 1 = 19$ нових ознак

⚠️ **Curse of dimensionality:** кількість ознак зростає експоненціально!

---

## Візуалізація різних степенів

```
Degree 1 (Лінійна):          Degree 2 (Квадратична):
    y                            y
    |      ╱                     |        ╱╲
    |    ╱                       |      ╱    ╲
    |  ╱                         |    ╱        ╲
    |╱                           |  ╱            ╲
    |________ x                  |╱________________ x

Degree 3 (Кубічна):          Degree 5 (Високий):
    y                            y
    |    ╱╲                      |   ╱╲  ╱╲
    |  ╱    ╲╲                   | ╱    ╲╱  ╲
    |╱        ╲                  |╱          ╲╲
    |          ╲╲                |            ╲╲
    |____________ x              |______________ x
```

**Спостереження:**
- **Степінь 1:** пряма лінія
- **Степінь 2:** парабола (одна крива)
- **Степінь 3:** S-подібна крива (дві кривини)
- **Високі степені:** багато хвиль (OVERFITTING! ⚠️)

---

## Простий приклад: Зарплата vs Досвід

### Дані

| Досвід (років) | Зарплата (тис. $) |
|----------------|-------------------|
| 1 | 40 |
| 2 | 45 |
| 3 | 55 |
| 4 | 70 |
| 5 | 90 |
| 6 | 115 |
| 7 | 145 |
| 8 | 180 |

### Спробуємо лінійну регресію

```python
# Лінійна модель
β₀ = 10, β₁ = 20
y_pred = 10 + 20 * x

# Передбачення для досвіду 8 років:
y_pred = 10 + 20 * 8 = 170 тис. $
# Реальне: 180 тис. $
# Помилка: 10 тис. $ (5.6%)
```

**Проблема:** Лінійна модель **недооцінює** зарплату на вищих рівнях досвіду.

**Scatter plot показує:** зростання **прискорюється** → нелінійна залежність!

### Поліноміальна регресія (степінь 2)

$$y = \beta_0 + \beta_1 x + \beta_2 x^2$$

Після навчання:
$$y = 35 + 5x + 2.5x^2$$

**Передбачення для досвіду 8 років:**
$$y = 35 + 5(8) + 2.5(64) = 35 + 40 + 160 = 235 \text{ тис. \$}$$

**Інтерпретація:**
- Зарплата зростає квадратично з досвідом
- Більший досвід → ще більше зростання зарплати

---

## Складний приклад: Температура vs Продажі морозива

### Дані

200 днів з температурою та продажами:

| Температура (°C) | Продажі (тис. шт) |
|------------------|-------------------|
| 0 | 5 |
| 10 | 20 |
| 15 | 35 |
| 20 | 55 |
| 25 | 80 |
| 30 | 100 |
| 35 | 95 |
| 40 | 70 |

**Спостереження:**
- При низьких температурах: мало продажів
- При середніх (25-30°C): пік продажів
- При дуже високих (>35°C): продажі знижуються (занадто спекотно!)

**Форма залежності:** перевернута парабола (∩)

### Модель (степінь 2)

$$\text{Продажі} = \beta_0 + \beta_1 \times \text{Temp} + \beta_2 \times \text{Temp}^2$$

Після навчання:
$$\text{Продажі} = -50 + 8 \times \text{Temp} - 0.12 \times \text{Temp}^2$$

**Аналіз коефіцієнтів:**
- $\beta_0 = -50$: базова константа (негативна через параболу)
- $\beta_1 = 8$: позитивний ефект температури
- $\beta_2 = -0.12$: **негативний** → парабола вниз (∩)

### Знаходження оптимальної температури

Максимум параболи при:
$$x_{max} = -\frac{\beta_1}{2\beta_2} = -\frac{8}{2 \times (-0.12)} = \frac{8}{0.24} = 33.3°C$$

**Висновок:** Пік продажів при **33.3°C** ✓

### Передбачення

**Температура 28°C:**
$$\text{Продажі} = -50 + 8(28) - 0.12(784) = -50 + 224 - 94 = 80 \text{ тис. шт}$$

**Температура 38°C (дуже спекотно):**
$$\text{Продажі} = -50 + 8(38) - 0.12(1444) = -50 + 304 - 173 = 81 \text{ тис. шт}$$

Хоча 38°C > 28°C, продажі майже однакові через параболу!

---

## Код (Python + scikit-learn)

### Простий приклад (1 ознака)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Дані
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([40, 45, 55, 70, 90, 115, 145, 180])

# 1. Створення поліноміальних ознак
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("Original features:")
print(X[:3])
# [[1]
#  [2]
#  [3]]

print("\nPolynomial features (degree=2):")
print(X_poly[:3])
# [[ 1.  1.]    ← [x, x²]
#  [ 2.  4.]
#  [ 3.  9.]]

# 2. Навчання лінійної регресії на поліноміальних ознаках
model = LinearRegression()
model.fit(X_poly, y)

# 3. Коефіцієнти
print(f"\nIntercept (β₀): {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}")
# [β₁, β₂] для [x, x²]

# 4. Передбачення
y_pred = model.predict(X_poly)

# 5. Метрики
print(f"\nR²: {r2_score(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")

# 6. Візуалізація
X_plot = np.linspace(0, 9, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=100, label='Дані')
plt.plot(X_plot, y_plot, color='red', linewidth=2, 
         label='Polynomial Regression (degree=2)')
plt.xlabel('Досвід (роки)', fontsize=12)
plt.ylabel('Зарплата (тис. $)', fontsize=12)
plt.title('Поліноміальна регресія: Зарплата vs Досвід', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Порівняння різних степенів

```python
from sklearn.model_selection import train_test_split

# Генерація більшого датасету
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 + 3*X.ravel() + 0.5*X.ravel()**2 + np.random.randn(100) * 5

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Тестуємо різні степені
degrees = [1, 2, 3, 5, 10, 15]
results = []

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

X_plot = np.linspace(0, 10, 300).reshape(-1, 1)

for idx, degree in enumerate(degrees):
    # Трансформація
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    X_plot_poly = poly.transform(X_plot)
    
    # Модель
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Передбачення
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    y_plot = model.predict(X_plot_poly)
    
    # Метрики
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results.append({
        'Degree': degree,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Overfitting': train_r2 - test_r2,
        'Features': X_train_poly.shape[1]
    })
    
    # Візуалізація
    axes[idx].scatter(X_train, y_train, color='blue', 
                     alpha=0.5, label='Train')
    axes[idx].scatter(X_test, y_test, color='green', 
                     alpha=0.5, label='Test')
    axes[idx].plot(X_plot, y_plot, color='red', linewidth=2, 
                  label=f'Degree {degree}')
    axes[idx].set_xlabel('X', fontsize=10)
    axes[idx].set_ylabel('y', fontsize=10)
    axes[idx].set_title(
        f'Degree {degree}\nTrain R²={train_r2:.3f}, Test R²={test_r2:.3f}',
        fontsize=11, fontweight='bold'
    )
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Результати
import pandas as pd
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print(results_df.to_string(index=False))
print("="*60)
```

**Типовий вихід:**

```
Degree  Train R²  Test R²  Overfitting  Features
     1    0.925    0.920        0.005         1
     2    0.985    0.980        0.005         2
     3    0.990    0.975        0.015         3
     5    0.995    0.960        0.035         5
    10    0.999    0.850        0.149        10  ← OVERFITTING!
    15    1.000    0.600        0.400        15  ← СИЛЬНИЙ OVERFITTING!
```

**Висновок:**
- **Degree 2-3:** оптимальний баланс
- **Degree 10+:** сильний overfitting (train R²=1, test R² падає)

---

## Багато ознак з interaction terms

```python
from sklearn.datasets import make_regression

# Дані з 2 ознаками
X, y = make_regression(
    n_samples=200,
    n_features=2,
    noise=10,
    random_state=42
)

# Додаємо нелінійність вручну
y = y + 0.5 * X[:, 0]**2 + 0.3 * X[:, 1]**2 + 0.2 * X[:, 0] * X[:, 1]

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Поліноміальні ознаки з interaction terms
poly = PolynomialFeatures(
    degree=2,
    include_bias=False,
    interaction_only=False  # Включає x₁², x₂², x₁x₂
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print("Original features: 2")
print(f"Polynomial features: {X_train_poly.shape[1]}")
print("Feature names:")
print(poly.get_feature_names_out(['x1', 'x2']))
# ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']

# Модель
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Оцінка
print(f"\nTrain R²: {model.score(X_train_poly, y_train):.4f}")
print(f"Test R²: {model.score(X_test_poly, y_test):.4f}")

# Коефіцієнти
print("\nCoefficients:")
for name, coef in zip(poly.get_feature_names_out(['x1', 'x2']), 
                      model.coef_):
    print(f"  {name}: {coef:.4f}")
```

---

## Вибір оптимального степеня

### 1. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

degrees = range(1, 11)
train_scores = []
cv_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    
    model = LinearRegression()
    
    # Train score
    model.fit(X_poly, y_train)
    train_score = model.score(X_poly, y_train)
    train_scores.append(train_score)
    
    # Cross-validation score
    cv_score = cross_val_score(
        model, X_poly, y_train, cv=5, 
        scoring='r2'
    ).mean()
    cv_scores.append(cv_score)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'o-', linewidth=2, 
         label='Train Score')
plt.plot(degrees, cv_scores, 's-', linewidth=2, 
         label='CV Score')
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Train vs CV Score by Polynomial Degree', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Оптимальний степінь
optimal_degree = degrees[np.argmax(cv_scores)]
print(f"Optimal degree: {optimal_degree}")
```

### 2. Validation Curve

```python
from sklearn.model_selection import validation_curve

# Потрібно створити custom estimator
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('linear', LinearRegression())
])

degrees = np.arange(1, 16)

train_scores, val_scores = validation_curve(
    pipe, X_train, y_train,
    param_name='poly__degree',
    param_range=degrees,
    cv=5,
    scoring='r2'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_mean, 'o-', linewidth=2, label='Train')
plt.fill_between(degrees, train_mean - train_std, 
                 train_mean + train_std, alpha=0.2)
plt.plot(degrees, val_mean, 's-', linewidth=2, label='Validation')
plt.fill_between(degrees, val_mean - val_std, 
                 val_mean + val_std, alpha=0.2)
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Validation Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Pipeline для поліноміальної регресії

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline: Polynomial → Scaling → Linear Regression
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

# Навчання
pipe.fit(X_train, y_train)

# Передбачення
y_pred = pipe.predict(X_test)

# Оцінка
print(f"R² Score: {pipe.score(X_test, y_test):.4f}")

# Доступ до компонентів
poly_features = pipe.named_steps['poly']
scaler = pipe.named_steps['scaler']
model = pipe.named_steps['linear']

print(f"\nFeature names: {poly_features.get_feature_names_out()}")
print(f"Coefficients: {model.coef_}")
```

---

## Regularization для поліноміальної регресії

### Проблема: Overfitting при високих степенях

**Рішення:** Ridge або Lasso регуляризація

```python
from sklearn.linear_model import Ridge, Lasso

# Поліноміальні ознаки високого степеня
poly = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Без регуляризації (OVERFITTING)
lr = LinearRegression()
lr.fit(X_train_poly, y_train)
print("Linear Regression (no regularization):")
print(f"  Train R²: {lr.score(X_train_poly, y_train):.4f}")
print(f"  Test R²: {lr.score(X_test_poly, y_test):.4f}")

# З Ridge регуляризацією
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)
print("\nRidge Regression (alpha=1.0):")
print(f"  Train R²: {ridge.score(X_train_poly, y_train):.4f}")
print(f"  Test R²: {ridge.score(X_test_poly, y_test):.4f}")

# З Lasso регуляризацією
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_poly, y_train)
print("\nLasso Regression (alpha=0.1):")
print(f"  Train R²: {lasso.score(X_train_poly, y_train):.4f}")
print(f"  Test R²: {lasso.score(X_test_poly, y_test):.4f}")
print(f"  Non-zero coefs: {np.sum(lasso.coef_ != 0)} / {len(lasso.coef_)}")
```

**Типовий результат:**

```
Linear Regression:
  Train R²: 0.9999  ← Майже ідеально на train
  Test R²: 0.6500   ← Катастрофа на test! OVERFITTING

Ridge Regression:
  Train R²: 0.9800
  Test R²: 0.9600   ← Набагато краще! ✓

Lasso Regression:
  Train R²: 0.9750
  Test R²: 0.9580
  Non-zero coefs: 15 / 55  ← Feature selection
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Нелінійні залежності** | Може моделювати криві, параболи |
| **Використання лінійної регресії** | Ті ж алгоритми та інтерпретація |
| **Простота** | Легко реалізувати через feature engineering |
| **Інтерпретованість** | Коефіцієнти мають математичний зміст |
| **Швидкість** | Швидше за складні нелінійні моделі |
| **Flexibility** | Легко контролювати складність (degree) |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Curse of dimensionality** | Кількість ознак зростає експоненційно |
| **Overfitting** | Високі степені легко перенавчаються |
| **Екстраполяція** | Погано передбачає за межами тренувальних даних |
| **Вибір степеня** | Потрібен підбір оптимального degree |
| **Мультиколінеарність** | Поліноміальні ознаки сильно корелюють |
| **Обмежена гнучкість** | Тільки поліноміальні залежності |

---

## Екстраполяція: небезпека поліноміальної регресії

### Проблема

Поліноміальні моделі **дуже погано екстраполюють** за межі тренувальних даних.

```python
# Тренувальні дані: X від 0 до 10
X_train = np.linspace(0, 10, 50).reshape(-1, 1)
y_train = 2 + 3*X_train.ravel() + 0.5*X_train.ravel()**2

# Поліноміальна регресія (степінь 3)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Екстраполяція: X від 0 до 20 (за межі!)
X_extrapolate = np.linspace(0, 20, 100).reshape(-1, 1)
X_extrapolate_poly = poly.transform(X_extrapolate)
y_extrapolate = model.predict(X_extrapolate_poly)

# Візуалізація
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', s=50, label='Train data')
plt.plot(X_extrapolate, y_extrapolate, color='red', linewidth=2, 
         label='Polynomial prediction')
plt.axvline(x=10, color='green', linestyle='--', linewidth=2, 
            label='End of training range')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Polynomial Regression: Extrapolation Problem', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)
plt.tight_layout()
plt.show()
```

**Результат:**
- В межах [0, 10]: точні передбачення ✓
- За межами [10, 20]: передбачення **летять у небо або падають** ✗

**Чому?**
- Поліноми високих степенів мають **екстремальну поведінку** на краях
- $x^3$ при великих x → дуже великі значення

---

## Коли використовувати Polynomial Regression

### Ідеально підходить ✓

- Scatter plot показує **явну криву** (парабола, S-крива)
- **Residual plot** лінійної регресії має систематичний патерн
- Потрібна **інтерпретованість** (медицина, природничі науки)
- Невелика кількість ознак ($p < 5$)
- Низькі/середні степені ($d \leq 3-4$)
- Передбачення **в межах** тренувальних даних

### Краще використати інше ✗

- Дуже складні нелінійні залежності → **Random Forest, XGBoost**
- Багато ознак + високі степені → **curse of dimensionality**
- Потрібна **екстраполяція** → обережно або інші методи
- Взаємодії між ознаками невідомі → **Tree-based models**
- Зображення, текст → **Deep Learning**

---

## Практичні поради 💡

1. **Почни з scatter plot** — подивись форму залежності
2. **Спочатку linear regression** — можливо, достатньо
3. **Residual plot** — перевір систематичні патерни
4. **Не переборщуй зі степенем** — зазвичай 2-3 достатньо
5. **Cross-validation** для вибору оптимального degree
6. **Regularization** при високих степенях (Ridge/Lasso)
7. **Нормалізуй дані** — особливо при високих степенях
8. **Interaction terms** — `interaction_only=True` якщо тільки взаємодії
9. **Не екстраполюй** — передбачай тільки в межах train діапазону
10. **Pipeline** — зручно комбінувати poly → scaling → model

---

## Поширені помилки ❌

### 1. Занадто високий степінь

```python
# ❌ НЕПРАВИЛЬНО
poly = PolynomialFeatures(degree=20)  # Overfitting гарантований!

# ✅ ПРАВИЛЬНО
# Підбери через CV або почни з 2-3
poly = PolynomialFeatures(degree=2)
```

### 2. Забути нормалізувати при високих степенях

```python
# ❌ НЕПРАВИЛЬНО (x^10 може бути дуже великим)
X_poly = poly.fit_transform(X)
model.fit(X_poly, y)

# ✅ ПРАВИЛЬНО
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)
model.fit(X_poly_scaled, y)
```

### 3. Екстраполяція без перевірки

```python
# ❌ НЕБЕЗПЕЧНО
X_new = [[100]]  # Далеко за межами train даних
y_pred = model.predict(poly.transform(X_new))

# ✅ ПРАВИЛЬНО
if X_new[0][0] > X_train.max() or X_new[0][0] < X_train.min():
    print("WARNING: Extrapolation! Predictions may be unreliable.")
```

### 4. Не перевірити overfitting

```python
# ❌ НЕПРАВИЛЬНО
# Дивитись тільки на train R²

# ✅ ПРАВИЛЬНО
print(f"Train R²: {model.score(X_train_poly, y_train):.4f}")
print(f"Test R²: {model.score(X_test_poly, y_test):.4f}")
# Якщо Train >> Test → overfitting!
```

---

## Альтернативи поліноміальній регресії

### 1. Spline Regression

**Кусково-поліноміальні функції** — різні поліноми на різних ділянках.

```python
from sklearn.preprocessing import SplineTransformer

spline = SplineTransformer(n_knots=4, degree=3)
X_spline = spline.fit_transform(X)
model = LinearRegression()
model.fit(X_spline, y)
```

**Переваги:**
- Більш гнучкі за прості поліноми
- Краща локальна апроксимація

### 2. Kernel Ridge Regression

```python
from sklearn.kernel_ridge import KernelRidge

model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
model.fit(X_train, y_train)
```

### 3. Tree-based Models

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

**Переваги:**
- Автоматично виявляють нелінійності
- Не потрібен вибір степеня
- Робастні до overfitting

---

## Пов'язані теми

- [[01_Linear_Regression]] — базова модель
- [[03_Regularization]] — Ridge/Lasso для запобігання overfitting
- [[Feature_Engineering]] — створення нових ознак
- [[Cross_Validation]] — вибір оптимального степеня
- [[Bias_Variance_Tradeoff]] — баланс складності моделі

## Ресурси

- [Scikit-learn: Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [StatQuest: Polynomial Regression](https://www.youtube.com/watch?v=Ja5jH9FOpqQ)

---

## Ключові висновки

> Поліноміальна регресія розширює лінійну регресію для нелінійних залежностей через створення поліноміальних ознак.

**Основні принципи:**
- Створює нові ознаки: $x, x^2, x^3, ..., x^d$
- Використовує звичайну лінійну регресію на нових ознаках
- Степінь $d$ контролює складність моделі
- Потребує regularization при високих степенях

**Формула (степінь 2):**
$$y = \beta_0 + \beta_1 x + \beta_2 x^2$$

**Коли використовувати:**
- Нелінійна залежність + інтерпретованість + малий $p$ = Polynomial Regression ✓

**Важливо:**
- Не екстраполюй!
- Підбирай degree через CV
- Використовуй regularization при $d > 3$

---

#ml #supervised-learning #regression #polynomial-regression #nonlinear #feature-engineering
