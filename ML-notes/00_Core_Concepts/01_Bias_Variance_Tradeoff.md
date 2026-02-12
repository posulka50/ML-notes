# Bias-Variance Tradeoff (Компроміс між зміщенням та дисперсією)

## Що це?

**Bias-Variance Tradeoff** — це фундаментальна концепція в машинному навчанні, яка описує **компроміс між помилкою через недостатню складність моделі (bias) та помилкою через надмірну чутливість до тренувальних даних (variance)**.

**Головна ідея:** будь-яка модель має знайти баланс між тим, щоб бути достатньо складною для вловлювання паттернів (низький bias) та достатньо простою, щоб не перенавчатися на шумі (низький variance).

## Навіщо потрібно?

- 🎯 **Розуміння помилок моделі** — чому модель помиляється
- 📊 **Вибір складності моделі** — проста чи складна модель
- 🔧 **Діагностика проблем** — underfitting vs overfitting
- 💡 **Вибір алгоритму** — який метод використати
- 🎚️ **Tuning гіперпараметрів** — як налаштовувати модель
- 🔍 **Покращення моделі** — де фокусувати зусилля

## Коли важливо?

**Потрібно розуміти:**

- Модель погано працює і треба зрозуміти чому
- Вибираєш між простою та складною моделлю
- **Debugging** — train score vs test score дуже різні
- Вибір між різними алгоритмами
- **Feature engineering** — додавати чи ні нові ознаки

---

## Розкладання помилки (Error Decomposition)

### Загальна формула

Для будь-якої моделі, очікувана помилка на новому зразку складається з трьох компонентів:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

де:
- **Bias²** — помилка через неправильні припущення моделі
- **Variance** — помилка через чутливість до тренувальних даних
- **Irreducible Error** — шум у даних (непередбачувана частина)

### Детальна формула

Для регресії (MSE):

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

де:
- $y$ — справжнє значення
- $\hat{f}(x)$ — передбачення моделі
- $\sigma^2$ — irreducible error (variance шуму)

---

## Bias (Зміщення)

### Що це?

**Bias** — це помилка через **надто спрощені припущення** моделі про залежність між $X$ та $y$.

### Формула

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$$

де:
- $\mathbb{E}[\hat{f}(x)]$ — очікуване передбачення моделі (усереднене по всіх можливих тренувальних наборах)
- $f(x)$ — справжня функція

### Інтуїція

**Високий bias** означає, що модель:

- ✗ Робить **сильні спрощення**
- ✗ **Недостатньо гнучка** для вловлювання паттернів
- ✗ Дає **систематично неправильні** передбачення
- ✗ **Underfits** (недонавчається)

**Приклад високого bias:**

```
Справжня залежність: y = x² (парабола)
Модель: y = ax + b (пряма лінія)

     y
     |     •
     |   •   •
     | •       •    ← Справжні дані (парабола)
     |•─────────•
     |           •
     |____________ x
          ↑
      Лінійна модель не може вловити криву!
```

### Характеристики високого bias

| Ознака | Опис |
|--------|------|
| **Train Error** | Високий ❌ |
| **Test Error** | Високий ❌ |
| **Gap** | Малий (train ≈ test) |
| **Проблема** | Underfitting |
| **Модель** | Занадто проста |

### Моделі з високим bias

- Linear Regression (для нелінійних даних)
- Logistic Regression (для складних границь)
- Decision Tree з `max_depth=1` (decision stump)
- Naive Bayes (сильні припущення про незалежність)

### Як зменшити bias?

✅ Збільшити складність моделі:

- Polynomial features
- Більша глибина дерев (`max_depth`)
- Більше layers у нейронних мережах
- Більше ознак (feature engineering)

✅ Використати складніший алгоритм:

- Linear → Polynomial → Neural Network
- Decision Stump → Deep Tree

✅ Зменшити регуляризацію:

- Менший $\lambda$ (Ridge/Lasso)
- Менший `min_samples_leaf`

---

## Variance (Дисперсія)

### Що це?

**Variance** — це помилка через **надмірну чутливість** моделі до конкретного тренувального набору.

### Формула

$$\text{Variance}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$

### Інтуїція

**Висока variance** означає, що модель:
- ✗ Занадто **гнучка**
- ✗ **Запам'ятовує шум** у тренувальних даних
- ✗ Дуже **різні передбачення** на різних train sets
- ✗ **Overfits** (перенавчається)

**Приклад високої variance:**
```
Справжня залежність: пряма лінія з шумом
Модель: поліном 10-го степеня

     y
     |    ╱╲╱╲
     |   ╱    ╲╱╲    ← Модель (проходить через кожну точку!)
     | •╱   •    ╲•
     |╱   •    •   ╲
     |  •    •   •  ╲
     |________________ x
     
Модель запам'ятала шум!
```

### Характеристики високої variance

| Ознака | Опис |
|--------|------|
| **Train Error** | Дуже низький ✓ |
| **Test Error** | Високий ❌ |
| **Gap** | Великий (train << test) |
| **Проблема** | Overfitting |
| **Модель** | Занадто складна |

### Моделі з високою variance

- Polynomial Regression (високі степені)
- Decision Tree (без обмежень глибини)
- KNN з малим K (K=1)
- Neural Networks (без regularization)

### Як зменшити variance?

✅ Зменшити складність моделі:
- Менша глибина дерев (`max_depth`)
- Менший degree у поліномах
- Менше parameters у NN

✅ Додати регуляризацію:
- Ridge/Lasso (більший $\lambda$)
- Dropout у нейронних мережах
- Early stopping

✅ Більше тренувальних даних:
- Збільшити розмір dataset
- Data augmentation

✅ Використати ансамблі:
- Random Forest (зменшує variance через bagging)
- Gradient Boosting (з regularization)

---

## Irreducible Error (Непередбачувана помилка)

### Що це?

**Irreducible Error** — це помилка через **шум у даних**, яку **неможливо усунути** жодною моделлю.

### Формула

$$\sigma^2 = \text{Var}[\epsilon]$$

де $\epsilon$ — випадковий шум у даних: $y = f(x) + \epsilon$

### Інтуїція

**Джерела irreducible error:**

- 📏 Помилки вимірювань
- 🎲 Справжня випадковість у процесі
- 🔍 Відсутні важливі ознаки
- 🌊 Noise у даних

**Приклад:**

```
Передбачення ціни акції через рік:
- Модель може вловити тренди ✓
- Але раптові події (війна, пандемія) непередбачувані ✗
- Це irreducible error
```

### Важливо!

- ❌ **Неможливо зменшити** жодною моделлю
- ✅ Можна тільки **оцінити** (нижня межа помилки)
- 🎯 Мета ML: мінімізувати Bias² + Variance, приймаючи $\sigma^2$

---

## Tradeoff (Компроміс)

### Візуальна інтуїція

```
Total Error
    |
    |    ╱‾‾‾╲        ← Total Error = Bias² + Variance + σ²
    |   ╱     ╲
    |  ╱   ╱‾‾‾‾╲     ← Variance
    | ╱___╱      ╲
    |╱             ╲__ ← Bias²
    |
    |─────────────────╲__ ← Irreducible Error (σ²)
    |_________________ Model Complexity
    Simple          Complex
    
    High Bias       Optimal        High Variance
    Low Variance    Balance        Low Bias
```

### Таблиця компромісу

| Складність моделі | Bias | Variance | Train Error | Test Error | Проблема |
|-------------------|------|----------|-------------|------------|----------|
| **Дуже проста** | ⬆️ Високий | ⬇️ Низька | Високий | Високий | Underfitting |
| **Оптимальна** | ⬇️ Помірний | ⬇️ Помірна | Низький | Низький | **Ідеально** ✓ |
| **Дуже складна** | ⬇️ Низький | ⬆️ Висока | Дуже низький | Високий | Overfitting |

### Ключова ідея

> **Не можна одночасно мінімізувати і bias, і variance!**

- Зменшення bias → збільшення variance
- Зменшення variance → збільшення bias
- **Мета:** знайти оптимальний баланс

---

## Приклади на різних моделях

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

# Справжня функція: y = sin(x) + шум
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = np.sin(X).ravel()
y = y_true + np.random.normal(0, 0.1, 100)

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Різні степені поліномів
degrees = [1, 3, 15]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, degree in enumerate(degrees):
    # Polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Модель
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Scores
    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    
    # Передбачення для візуалізації
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    
    # Візуалізація
    axes[idx].scatter(X_train, y_train, alpha=0.5, s=30, label='Train')
    axes[idx].scatter(X_test, y_test, alpha=0.5, s=30, label='Test')
    axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
    axes[idx].plot(X_plot, np.sin(X_plot), 'g--', linewidth=2, 
                   label='True function', alpha=0.7)
    
    # Діагноз
    if degree == 1:
        diagnosis = "HIGH BIAS\nUnderfitting"
        color = 'red'
    elif degree == 3:
        diagnosis = "OPTIMAL\nGood Balance"
        color = 'green'
    else:
        diagnosis = "HIGH VARIANCE\nOverfitting"
        color = 'red'
    
    axes[idx].set_title(
        f'Degree={degree}\n'
        f'Train R²={train_score:.3f}, Test R²={test_score:.3f}\n'
        f'{diagnosis}',
        fontsize=11, fontweight='bold', color=color
    )
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Очікуваний результат:**
- **Degree=1:** Train R²=0.4, Test R²=0.4 (HIGH BIAS)
- **Degree=3:** Train R²=0.9, Test R²=0.88 (OPTIMAL)
- **Degree=15:** Train R²=0.99, Test R²=0.2 (HIGH VARIANCE)

### Decision Trees

```python
from sklearn.tree import DecisionTreeRegressor

# Різні max_depth
depths = [1, 5, None]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, depth in enumerate(depths):
    # Модель
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    # Scores
    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)
    
    # Передбачення
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    y_plot = dt.predict(X_plot)
    
    # Візуалізація
    axes[idx].scatter(X_train, y_train, alpha=0.5, s=30, label='Train')
    axes[idx].scatter(X_test, y_test, alpha=0.5, s=30, label='Test')
    axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
    axes[idx].plot(X_plot, np.sin(X_plot), 'g--', linewidth=2, 
                   label='True function', alpha=0.7)
    
    # Діагноз
    if depth == 1:
        diagnosis = "HIGH BIAS"
    elif depth == 5:
        diagnosis = "OPTIMAL"
    else:
        diagnosis = "HIGH VARIANCE"
    
    axes[idx].set_title(
        f'max_depth={depth}\n'
        f'Train R²={train_score:.3f}, Test R²={test_score:.3f}\n'
        f'{diagnosis}',
        fontsize=11, fontweight='bold'
    )
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Learning Curves (Криві навчання)

### Що це?

**Learning Curves** показують, як train та test error змінюються зі зміною:

- Кількості тренувальних зразків
- Складності моделі

### High Bias (Underfitting)

```
Error
    |
    |  Test Error ────────
    |                      ← Gap малий
    |  Train Error ───────
    |
    |__________________ Training Set Size
    
Ознаки:
- Train error високий і стабільний
- Test error високий і стабільний
- Gap між ними малий
- Більше даних НЕ допоможе!
```

### High Variance (Overfitting)

```
Error
    |
    |  Test Error ────────
    |                      ← Gap великий
    |            
    |  Train Error ╲
    |               ╲_____ ← Дуже низький
    |__________________ Training Set Size
    
Ознаки:
- Train error дуже низький
- Test error високий
- Великий gap
- Більше даних ДОПОМОЖЕ!
```

### Optimal Model

```
Error
    |
    |  Test Error ╲
    |              ╲______
    |               ╲      ← Gap малий
    |  Train Error  ╲_____
    |__________________ Training Set Size
    
Ознаки:
- Train error помірний
- Test error помірний
- Gap малий
- Обидві криві збігаються
```

### Код

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, title):
    """Побудова learning curves"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        n_jobs=-1,
        scoring='r2'
    )
    
    # Усереднення по folds
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Візуалізація
    plt.figure(figsize=(10, 6))
    
    # Train scores
    plt.plot(train_sizes, train_scores_mean, 'o-', linewidth=2,
             label='Train Score', color='blue')
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color='blue')
    
    # Test scores
    plt.plot(train_sizes, test_scores_mean, 's-', linewidth=2,
             label='Validation Score', color='red')
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title(f'Learning Curves: {title}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Діагноз
    final_gap = train_scores_mean[-1] - test_scores_mean[-1]
    final_test_score = test_scores_mean[-1]
    
    print(f"\n=== Diagnosis for {title} ===")
    print(f"Final Train Score: {train_scores_mean[-1]:.3f}")
    print(f"Final Test Score: {test_scores_mean[-1]:.3f}")
    print(f"Gap (Train - Test): {final_gap:.3f}")
    
    if final_test_score < 0.6 and final_gap < 0.1:
        print("⚠️ HIGH BIAS (Underfitting)")
        print("   → Use more complex model or add features")
    elif final_gap > 0.2:
        print("⚠️ HIGH VARIANCE (Overfitting)")
        print("   → Add regularization or more data")
    else:
        print("✓ Good balance!")

# Приклади
# High Bias
poly_1 = Pipeline([
    ('poly', PolynomialFeatures(degree=1)),
    ('linear', LinearRegression())
])
plot_learning_curves(poly_1, X, y, "Linear (High Bias)")

# Optimal
poly_3 = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])
plot_learning_curves(poly_3, X, y, "Degree 3 (Optimal)")

# High Variance
poly_15 = Pipeline([
    ('poly', PolynomialFeatures(degree=15)),
    ('linear', LinearRegression())
])
plot_learning_curves(poly_15, X, y, "Degree 15 (High Variance)")
```

---

## Validation Curves

### Що це?

**Validation Curves** показують, як train та test error залежать від **гіперпараметра** моделі.

### Код

```python
from sklearn.model_selection import validation_curve

# Для Decision Tree: max_depth
param_range = range(1, 21)

train_scores, test_scores = validation_curve(
    DecisionTreeRegressor(random_state=42),
    X, y,
    param_name='max_depth',
    param_range=param_range,
    cv=5,
    scoring='r2'
)

# Усереднення
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Візуалізація
plt.figure(figsize=(10, 6))

plt.plot(param_range, train_mean, 'o-', linewidth=2,
         label='Train Score', color='blue')
plt.fill_between(param_range, train_mean - train_std,
                 train_mean + train_std, alpha=0.1, color='blue')

plt.plot(param_range, test_mean, 's-', linewidth=2,
         label='Validation Score', color='red')
plt.fill_between(param_range, test_mean - test_std,
                 test_mean + test_std, alpha=0.1, color='red')

# Оптимальне значення
optimal_depth = param_range[np.argmax(test_mean)]
plt.axvline(x=optimal_depth, color='green', linestyle='--',
            linewidth=2, label=f'Optimal (depth={optimal_depth})')

plt.xlabel('max_depth', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Validation Curve: Decision Tree', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nOptimal max_depth: {optimal_depth}")
print(f"Best Validation Score: {test_mean[optimal_depth-1]:.3f}")
```

**Інтерпретація:**
```
max_depth=1-3:   High Bias (обидві криві низько)
max_depth=5-8:   Optimal (найкраща test score)
max_depth=15+:   High Variance (train високо, test низько)
```

---

## Діагностика: Bias vs Variance

### Таблиця діагностики

| Показник | High Bias | Optimal | High Variance |
|----------|-----------|---------|---------------|
| **Train Error** | Високий ❌ | Низький ✓ | Дуже низький ✓ |
| **Test Error** | Високий ❌ | Низький ✓ | Високий ❌ |
| **Gap (Train - Test)** | Малий (~0) | Малий (~0-0.05) | Великий (>0.1) |
| **Learning Curve** | Плато рано | Збіжність | Gap не зменшується |
| **Проблема** | Underfitting | - | Overfitting |
| **Модель** | Занадто проста | Ідеальна | Занадто складна |

### Швидка діагностика (код)

```python
def diagnose_model(model, X_train, X_test, y_train, y_test):
    """Швидка діагностика bias vs variance"""
    
    # Scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score
    
    print("="*60)
    print("MODEL DIAGNOSIS")
    print("="*60)
    print(f"Train Score: {train_score:.4f}")
    print(f"Test Score:  {test_score:.4f}")
    print(f"Gap:         {gap:.4f}")
    print()
    
    # Діагноз
    if test_score < 0.7 and gap < 0.1:
        print("🔴 HIGH BIAS (Underfitting)")
        print("   Problem: Model is too simple")
        print("   Solutions:")
        print("   → Increase model complexity")
        print("   → Add more features (polynomial, interactions)")
        print("   → Use more complex algorithm")
        print("   → Reduce regularization (smaller λ)")
        
    elif gap > 0.15:
        print("🔴 HIGH VARIANCE (Overfitting)")
        print("   Problem: Model is too complex")
        print("   Solutions:")
        print("   → Add more training data")
        print("   → Add regularization (Ridge, Lasso)")
        print("   → Reduce model complexity")
        print("   → Use ensemble methods (Random Forest)")
        print("   → Feature selection (remove irrelevant features)")
        
    elif test_score >= 0.7 and gap <= 0.15:
        print("✅ GOOD BALANCE")
        print("   Model appears to be well-tuned!")
        if gap > 0.05:
            print("   Minor overfitting - consider slight regularization")
    
    else:
        print("⚠️ UNUSUAL PATTERN")
        print("   Check for data leakage or other issues")
    
    print("="*60)
    
    return {
        'train_score': train_score,
        'test_score': test_score,
        'gap': gap
    }

# Використання
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

print("\n### Linear Model (likely HIGH BIAS) ###")
lr = LinearRegression()
lr.fit(X_train, y_train)
diagnose_model(lr, X_train, X_test, y_train, y_test)

print("\n### Deep Tree (likely HIGH VARIANCE) ###")
dt_deep = DecisionTreeRegressor(max_depth=20, random_state=42)
dt_deep.fit(X_train, y_train)
diagnose_model(dt_deep, X_train, X_test, y_train, y_test)

print("\n### Random Forest (likely OPTIMAL) ###")
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
diagnose_model(rf, X_train, X_test, y_train, y_test)
```

---

## Стратегії виправлення

### Якщо HIGH BIAS (Underfitting)

```python
# ❌ Проблема: модель занадто проста

# ✅ Рішення 1: Більша складність
# Було:
model = DecisionTreeRegressor(max_depth=1)

# Стало:
model = DecisionTreeRegressor(max_depth=10)

# ✅ Рішення 2: Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# ✅ Рішення 3: Більше ознак
# Feature engineering: створити взаємодії, нелінійності

# ✅ Рішення 4: Зменшити регуляризацію
# Було:
model = Ridge(alpha=10.0)

# Стало:
model = Ridge(alpha=0.1)

# ✅ Рішення 5: Складніший алгоритм
# Було: Linear Regression
# Стало: Random Forest або Neural Network
```

### Якщо HIGH VARIANCE (Overfitting)

```python
# ❌ Проблема: модель занадто складна

# ✅ Рішення 1: Більше даних
# - Збільшити train set
# - Data augmentation (для зображень)
# - Synthetic data generation

# ✅ Рішення 2: Regularization
# Ridge (L2)
model = Ridge(alpha=1.0)

# Lasso (L1)
model = Lasso(alpha=0.1)

# ✅ Рішення 3: Зменшити складність
# Було:
model = DecisionTreeRegressor(max_depth=None)

# Стало:
model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)

# ✅ Рішення 4: Early Stopping
# Для Gradient Boosting
model = GradientBoostingRegressor(
    n_estimators=1000,
    validation_fraction=0.1,
    n_iter_no_change=50
)

# ✅ Рішення 5: Dropout (Neural Networks)
# Keras/TensorFlow
model.add(Dropout(0.5))

# ✅ Рішення 6: Ensemble методи
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()  # Bagging зменшує variance

# ✅ Рішення 7: Feature Selection
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)
```

---

## Зв'язок з іншими концепціями

### Bias-Variance і Регуляризація

```python
# Ridge Regression: контроль variance через λ
from sklearn.linear_model import Ridge

lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
train_scores = []
test_scores = []

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    ridge.fit(X_train_poly, y_train)
    
    train_scores.append(ridge.score(X_train_poly, y_train))
    test_scores.append(ridge.score(X_test_poly, y_test))

plt.figure(figsize=(10, 6))
plt.plot(lambdas, train_scores, 'o-', label='Train', linewidth=2)
plt.plot(lambdas, test_scores, 's-', label='Test', linewidth=2)
plt.xlabel('λ (Regularization Strength)', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Regularization Effect on Bias-Variance', 
          fontsize=14, fontweight='bold')
plt.xscale('log')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Спостереження:
# λ малий → high variance (overfitting)
# λ великий → high bias (underfitting)
```

### Bias-Variance і Ensemble Methods

**Bagging (Random Forest):**
- ✅ Зменшує **Variance** (через усереднення)
- ❌ Майже не впливає на **Bias**
- **Використання:** high-variance моделі (deep trees)

**Boosting (Gradient Boosting):**
- ✅ Зменшує **Bias** (послідовне покращення)
- ✅ Може зменшити **Variance** (з regularization)
- **Використання:** high-bias моделі (shallow trees)

```python
# Баланс через ансамблі
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# High Variance → Bagging
rf = RandomForestRegressor(n_estimators=100, max_depth=None)
# Variance зменшується через усереднення дерев

# High Bias → Boosting
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3)
# Bias зменшується через послідовне навчання
```

---

## Практичні поради 💡

### 1. Завжди будуй Learning Curves

```python
# Візуалізуй навчання перед production
plot_learning_curves(model, X, y, "My Model")
```

### 2. Розділяй дані правильно

```python
# Обов'язково: Train / Validation / Test
from sklearn.model_selection import train_test_split

# 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)
```

### 3. Використовуй Cross-Validation

```python
# Більш надійна оцінка
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

### 4. Почни просто, ускладнюй поступово

```python
# Крок 1: Baseline (проста модель)
baseline = LinearRegression()

# Крок 2: Додай складність
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

# Крок 3: Регуляризація при потребі
ridge_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge(alpha=1.0))
])
```

### 5. Моніторинг під час навчання

```python
# Для моделей з ітеративним навчанням
import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=1000)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True
)

# Візуалізація
results = model.evals_result()
plt.plot(results['validation_0']['rmse'], label='Train')
plt.plot(results['validation_1']['rmse'], label='Validation')
plt.legend()
plt.show()
```

### 6. Документуй експерименти

```python
# Зберігай результати
experiments = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    experiments.append({
        'model': model_name,
        'train_score': model.score(X_train, y_train),
        'val_score': model.score(X_val, y_val),
        'gap': model.score(X_train, y_train) - model.score(X_val, y_val)
    })

import pandas as pd
df_results = pd.DataFrame(experiments)
print(df_results.sort_values('val_score', ascending=False))
```

### 7. Розумій свої дані

```python
# EDA перед моделюванням
import seaborn as sns

# Розподіл target
sns.histplot(y)

# Кореляції
sns.heatmap(df.corr(), annot=True)

# Scatter plots
sns.pairplot(df)
```

### 8. Feature Engineering відповідально

```python
# Не додавай features сліпо!
# Кожна нова ознака → більша складність → ризик variance

# ✅ Добре: обґрунтовані ознаки
df['price_per_sqft'] = df['price'] / df['sqft']

# ❌ Погано: безглузді комбінації
df['random_feature'] = df['feature1'] * df['feature2'] * df['feature3']
```

### 9. Використовуй ансамблі розумно

```python
# Random Forest: коли маєш high variance
rf = RandomForestRegressor()

# Gradient Boosting: коли маєш high bias
gb = GradientBoostingRegressor()

# Stacking: коли потрібна максимальна точність
from sklearn.ensemble import StackingRegressor
stack = StackingRegressor(
    estimators=[('rf', rf), ('gb', gb)],
    final_estimator=Ridge()
)
```

### 10. Не забувай про domain knowledge

```python
# ML metrics важливі, але не все!
# Перевіряй, чи має сенс модель з точки зору бізнесу/науки

# Приклад: якщо модель каже, що вік = -5 років → проблема!
```

---

## Поширені помилки ❌

### 1. Оцінювати тільки на train set

```python
# ❌ ПОГАНО
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_train, y_train)}")  # Може бути overfitting!

# ✅ ДОБРЕ
print(f"Train: {model.score(X_train, y_train)}")
print(f"Test: {model.score(X_test, y_test)}")
print(f"Gap: {model.score(X_train, y_train) - model.score(X_test, y_test)}")
```

### 2. Не використовувати validation set

```python
# ❌ Tuning на test set → data leakage
for alpha in [0.1, 1, 10]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ❌ Витік інформації!

# ✅ Використовуй validation або cross-validation
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(Ridge(), {'alpha': [0.1, 1, 10]}, cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
final_score = best_model.score(X_test, y_test)  # ✅ Чесна оцінка
```

### 3. Додавати регуляризацію до простої моделі

```python
# ❌ Якщо вже є high bias, регуляризація погіршить!
# Linear model на нелінійних даних + Ridge = ще гірше

# ✅ Спочатку збільш складність, потім додай regularization
```

### 4. Ігнорувати розподіл даних

```python
# ❌ Якщо train/test з різних розподілів
# Модель може здаватися хорошою на train, але погано на test

# ✅ Переконайся, що розподіли схожі
import matplotlib.pyplot as plt
plt.hist(y_train, alpha=0.5, label='Train')
plt.hist(y_test, alpha=0.5, label='Test')
plt.legend()
plt.show()
```

### 5. Занадто рано зупинятися

```python
# ❌ "Train accuracy = 70%, погано, відмовляюсь від моделі"
# Можливо, це optimal для цих даних через irreducible error!

# ✅ Порівняй з baseline та іншими моделями
```

---

## Реальний приклад: Передбачення ціни будинків

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження даних
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Розділення: 60% train, 20% val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print("="*70)
print("BIAS-VARIANCE ANALYSIS: California Housing")
print("="*70)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Нормалізація
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Функція для діагностики
def evaluate_model(name, model, X_tr, X_v, X_te, y_tr, y_v, y_te):
    """Навчити та оцінити модель"""
    model.fit(X_tr, y_tr)
    
    train_score = model.score(X_tr, y_tr)
    val_score = model.score(X_v, y_v)
    test_score = model.score(X_te, y_te)
    gap = train_score - val_score
    
    # Діагноз
    if val_score < 0.6 and gap < 0.1:
        diagnosis = "HIGH BIAS"
        color = "🔴"
    elif gap > 0.15:
        diagnosis = "HIGH VARIANCE"
        color = "🔴"
    else:
        diagnosis = "BALANCED"
        color = "✅"
    
    return {
        'Model': name,
        'Train R²': f"{train_score:.3f}",
        'Val R²': f"{val_score:.3f}",
        'Test R²': f"{test_score:.3f}",
        'Gap': f"{gap:.3f}",
        'Diagnosis': f"{color} {diagnosis}"
    }

# Тестуємо різні моделі
results = []

# 1. Linear Regression (likely high bias на складних даних)
print("\n1. Linear Regression...")
lr = LinearRegression()
results.append(evaluate_model(
    "Linear Regression", lr,
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train, y_val, y_test
))

# 2. Polynomial Features degree=2 (збільшення складності)
print("2. Polynomial Regression (degree=2)...")
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_val_poly = poly_features.transform(X_val_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

lr_poly = LinearRegression()
results.append(evaluate_model(
    "Polynomial (deg=2)", lr_poly,
    X_train_poly, X_val_poly, X_test_poly,
    y_train, y_val, y_test
))

# 3. Ridge (polynomial + regularization)
print("3. Ridge Regression...")
ridge = Ridge(alpha=1.0)
results.append(evaluate_model(
    "Ridge (α=1.0)", ridge,
    X_train_poly, X_val_poly, X_test_poly,
    y_train, y_val, y_test
))

# 4. Decision Tree без обмежень (likely high variance)
print("4. Deep Decision Tree...")
dt_deep = DecisionTreeRegressor(random_state=42)
results.append(evaluate_model(
    "Deep Tree (no limit)", dt_deep,
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train, y_val, y_test
))

# 5. Decision Tree з обмеженням
print("5. Shallow Decision Tree...")
dt_shallow = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=42)
results.append(evaluate_model(
    "Shallow Tree (depth=5)", dt_shallow,
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train, y_val, y_test
))

# 6. Random Forest (зменшує variance)
print("6. Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
results.append(evaluate_model(
    "Random Forest", rf,
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train, y_val, y_test
))

# 7. Gradient Boosting (зменшує bias)
print("7. Gradient Boosting...")
gb = GradientBoostingRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
)
results.append(evaluate_model(
    "Gradient Boosting", gb,
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train, y_val, y_test
))

# Результати
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Візуалізація
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Comparison of scores
model_names = df_results['Model'].values
train_scores = [float(x) for x in df_results['Train R²'].values]
val_scores = [float(x) for x in df_results['Val R²'].values]
test_scores = [float(x) for x in df_results['Test R²'].values]

x = np.arange(len(model_names))
width = 0.25

axes[0, 0].bar(x - width, train_scores, width, label='Train', alpha=0.8)
axes[0, 0].bar(x, val_scores, width, label='Validation', alpha=0.8)
axes[0, 0].bar(x + width, test_scores, width, label='Test', alpha=0.8)
axes[0, 0].set_ylabel('R² Score', fontsize=11)
axes[0, 0].set_title('Model Comparison', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Gap visualization
gaps = [float(x) for x in df_results['Gap'].values]
colors = ['red' if g > 0.15 else 'green' if g < 0.1 else 'orange' for g in gaps]

axes[0, 1].barh(model_names, gaps, color=colors, alpha=0.7)
axes[0, 1].axvline(x=0.1, color='green', linestyle='--', 
                   linewidth=2, label='Good (<0.1)')
axes[0, 1].axvline(x=0.15, color='orange', linestyle='--', 
                   linewidth=2, label='Warning (>0.15)')
axes[0, 1].set_xlabel('Gap (Train - Val)', fontsize=11)
axes[0, 1].set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Learning Curves для найкращої моделі
best_model_idx = np.argmax(val_scores)
best_model_name = model_names[best_model_idx]

print(f"\nGenerating learning curves for best model: {best_model_name}")

# Для прикладу візьмемо Gradient Boosting
train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='r2',
    n_jobs=-1
)

train_mean = np.mean(train_scores_lc, axis=1)
train_std = np.std(train_scores_lc, axis=1)
val_mean = np.mean(val_scores_lc, axis=1)
val_std = np.std(val_scores_lc, axis=1)

axes[1, 0].plot(train_sizes, train_mean, 'o-', linewidth=2, label='Train')
axes[1, 0].fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
axes[1, 0].plot(train_sizes, val_mean, 's-', linewidth=2, label='Validation')
axes[1, 0].fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1)
axes[1, 0].set_xlabel('Training Set Size', fontsize=11)
axes[1, 0].set_ylabel('R² Score', fontsize=11)
axes[1, 0].set_title(f'Learning Curves: {best_model_name}', 
                     fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 4. Bias-Variance visualization
axes[1, 1].scatter(gaps, val_scores, s=200, alpha=0.6, c=range(len(gaps)), 
                   cmap='viridis')

for i, name in enumerate(model_names):
    axes[1, 1].annotate(name, (gaps[i], val_scores[i]), 
                       fontsize=8, ha='center')

# Зони
axes[1, 1].axvline(x=0.1, color='green', linestyle='--', alpha=0.5)
axes[1, 1].axhline(y=0.7, color='blue', linestyle='--', alpha=0.5)

axes[1, 1].text(0.05, 0.75, 'Good Balance\n(Low Bias, Low Variance)', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[1, 1].text(0.25, 0.65, 'High Variance\n(Overfitting)', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
axes[1, 1].text(0.05, 0.55, 'High Bias\n(Underfitting)', 
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

axes[1, 1].set_xlabel('Gap (Train - Val)', fontsize=11)
axes[1, 1].set_ylabel('Validation R²', fontsize=11)
axes[1, 1].set_title('Bias-Variance Map', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

# Рекомендації
for i, row in df_results.iterrows():
    gap = float(row['Gap'])
    val = float(row['Val R²'])
    
    if 'HIGH BIAS' in row['Diagnosis']:
        print(f"\n{row['Model']}:")
        print("  → Increase model complexity")
        print("  → Add more features or polynomial terms")
    elif 'HIGH VARIANCE' in row['Diagnosis']:
        print(f"\n{row['Model']}:")
        print("  → Add regularization")
        print("  → Get more training data")
        print("  → Reduce model complexity")
    else:
        print(f"\n{row['Model']}: ✅ Well balanced!")

print("\n" + "="*70)
```

---

## Пов'язані теми

- [[02_Overfitting_Underfitting]] — практичні прояви bias-variance
- [[03_Train_Test_Split]] — правильне розділення даних
- [[04_Cross_Validation]] — надійна оцінка моделі
- [[03_Regularization]] — контроль variance
- [[02_Random_Forest]] — зменшення variance через bagging
- [[03_Gradient_Boosting]] — зменшення bias через boosting

## Ресурси

- [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)
- [Elements of Statistical Learning (ESL)](https://hastie.su.domains/ElemStatLearn/)
- [Andrew Ng: Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)

---

## Ключові висновки

> Bias-Variance Tradeoff — це фундаментальний компроміс в ML між недостатньою складністю моделі (high bias) та надмірною чутливістю до тренувальних даних (high variance).

**Формула помилки:**
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Ключові принципи:**
- **High Bias** → модель занадто проста → underfitting
- **High Variance** → модель занадто складна → overfitting
- **Optimal** → баланс між bias та variance

**Діагностика:**
| Проблема | Train Error | Test Error | Gap | Рішення |
|----------|-------------|------------|-----|---------|
| High Bias | Високий | Високий | Малий | ↑ Складність |
| High Variance | Низький | Високий | Великий | ↑ Regularization / Дані |
| Optimal | Низький | Низький | Малий | ✅ |

**Практичні інструменти:**
- Learning Curves — візуалізація навчання
- Validation Curves — підбір гіперпараметрів
- Cross-Validation — надійна оцінка
- Train/Val/Test split — правильна валідація

**Найважливіше:**
- Завжди порівнюй train та test scores
- Будуй learning curves
- Починай просто, ускладнюй поступово
- Використовуй regularization розумно
- Баланс > надмірна оптимізація

---

#ml #core-concepts #bias-variance #tradeoff #underfitting #overfitting #model-complexity #diagnostics
