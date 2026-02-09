# Regularization (Регуляризація)

## Що це?

**Регуляризація** — це техніка додавання штрафу (penalty) до функції втрат моделі для **запобігання overfitting** (перенавчанню) шляхом обмеження складності моделі.

**Головна ідея:** знайти баланс між точністю на тренувальних даних і здатністю моделі узагальнювати на нових даних.

## Навіщо потрібна?

- 🎯 **Боротьба з overfitting** — модель не запам'ятовує тренувальні дані
- 📉 **Зменшення variance** — стабільніші передбачення
- 🔍 **Feature selection** — автоматичний відбір важливих ознак (L1)
- ⚖️ **Мультиколінеарність** — робота з корельованими ознаками (L2)
- 🎚️ **Контроль складності** — простіші, більш інтерпритовані моделі

## Коли використовувати?

**Потрібно:**

- **Overfitting** — train score >> test score
- Багато ознак порівняно з кількістю спостережень ($p > n$ або $p \approx n$)
- **Мультиколінеарність** — сильна кореляція між ознаками
- Потрібна **автоматична feature selection**
- Коефіцієнти моделі занадто великі

**Не потрібно:**

- **Underfitting** — модель занадто проста
- Дуже мало ознак ($p << n$)
- Tree-based моделі (вони мають власну регуляризацію)

---

## Проблема Overfitting

### Bias-Variance Trade-off

```
High Bias               Balanced            High Variance
(Underfitting)                             (Overfitting)

    y                      y                    y
    |  ─────               | ╱──╲              | ╱╲╱╲╱╲
    | /                    |╱    ╲             |╱    ╲ ╲
    |/                     /      ╲            ╱      ╲ ╲
    |________              |_______╲           |________╲╲
         x                      x                   x

Train MSE: Високий      Середній            Низький
Test MSE:  Високий      Низький             Високий
```

**Overfitting виникає коли:**
- Модель занадто складна (багато ознак)
- Мало тренувальних даних
- Noise в даних інтерпретується як сигнал

---

## Типи регуляризації

## 1. Ridge Regression (L2 Regularization)

### Функція втрат

$$J(\beta) = \text{MSE} + \lambda \sum_{j=1}^{p} \beta_j^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

де:
- $\lambda$ (lambda) — **регуляризаційний параметр** ($\lambda \geq 0$)
- $\sum \beta_j^2$ — **L2 norm** (сума квадратів коефіцієнтів)

### Як працює?

**Додаємо штраф за великі коефіцієнти:**
я
- Модель намагається мінімізувати MSE
- Але також намагається зменшити коефіцієнти
- **Trade-off:** точність vs простота

### Ефект на коефіцієнти

```
λ = 0:      β може бути дуже великим
λ = 0.01:   β трохи зменшується
λ = 1:      β значно зменшується
λ = 100:    β ≈ 0 (майже всі коефіцієнти близькі до 0)
```

**Ridge НЕ зануляє коефіцієнти!** Вони стають малими, але не 0.

### Властивості

| Властивість | Опис |
|-------------|------|
| **Не зануляє β** | Всі коефіцієнти залишаються в моделі |
| **Група кореляцій** | Корельовані ознаки отримують схожі коефіцієнти |
| **Мультиколінеарність** | Дуже добре справляється |
| **Інтерпретованість** | Менша, ніж Lasso |

### Коли використовувати?

✅ **Мультиколінеарність** — сильна кореляція між ознаками
✅ Всі ознаки потенційно корисні
✅ Потрібна стабільність моделі
✅ $p > n$ (більше ознак, ніж спостережень)

---

## 2. Lasso Regression (L1 Regularization)

### Функція втрат

$$J(\beta) = \text{MSE} + \lambda \sum_{j=1}^{p} |\beta_j| = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

де:
- $\sum |\beta_j|$ — **L1 norm** (сума абсолютних значень)

### Як працює?

**Додаємо штраф за абсолютні значення коефіцієнтів:**
- Lasso **зануляє** деякі коефіцієнти
- **Автоматична feature selection**
- Залишає тільки найважливіші ознаки

### Ефект на коефіцієнти

```
λ = 0:      β₁=5, β₂=3, β₃=2, β₄=1
λ = 0.5:    β₁=4.5, β₂=2.5, β₃=1, β₄=0  ← β₄ занулився!
λ = 1:      β₁=4, β₂=1.5, β₃=0, β₄=0
λ = 5:      β₁=2, β₂=0, β₃=0, β₄=0
λ = 10:     β₁=0, β₂=0, β₃=0, β₄=0      ← Всі нулі
```

**Lasso зануляє коефіцієнти = видаляє ознаки!**

### Властивості

| Властивість | Опис |
|-------------|------|
| **Зануляє β** | Деякі коефіцієнти стають рівно 0 |
| **Feature selection** | Автоматично відбирає важливі ознаки |
| **Sparse models** | Модель з малою кількістю ознак |
| **Корельовані ознаки** | Вибирає одну, ігнорує інші |

### Коли використовувати?

✅ **Feature selection** — потрібно відібрати найважливіші ознаки
✅ Багато непотрібних ознак
✅ Потрібна **інтерпретованість** — проста модель
✅ Sparse data (багато нулів)

---

## 3. Elastic Net (L1 + L2)

### Функція втрат

$$J(\beta) = \text{MSE} + \lambda \left[ \alpha \sum_{j=1}^{p} |\beta_j| + (1-\alpha) \sum_{j=1}^{p} \beta_j^2 \right]$$

де:
- $\alpha \in [0, 1]$ — **mixing parameter**
  - $\alpha = 0$ → Ridge
  - $\alpha = 1$ → Lasso
  - $\alpha = 0.5$ → рівний баланс

### Як працює?

**Комбінує переваги Ridge та Lasso:**
- L1 (Lasso): feature selection
- L2 (Ridge): стабільність при кореляціях

### Коли використовувати?

✅ Корельовані ознаки + потрібна feature selection
✅ $p > n$ з групами корельованих ознак
✅ Невпевненість між Ridge та Lasso

---

## Порівняння методів

| Критерій | Ridge (L2) | Lasso (L1) | Elastic Net |
|----------|------------|------------|-------------|
| **Penalty** | $\sum \beta_j^2$ | $\sum \|\beta_j\|$ | $\alpha L1 + (1-\alpha)L2$ |
| **Зануляє β** | ❌ Ні | ✅ Так | ✅ Так (частково) |
| **Feature selection** | ❌ Ні | ✅ Автоматично | ✅ Автоматично |
| **Мультиколінеарність** | ✅ Дуже добре | ⚠️ Вибирає одну ознаку | ✅ Добре |
| **Корельовані ознаки** | Усереднює | Вибирає одну | Групує схожі |
| **Інтерпретованість** | Середня | Висока | Висока |
| **Обчислення** | Швидко | Повільніше | Повільніше |

### Візуальне порівняння

```
Constraint regions (для 2 ознак):

Ridge (L2):              Lasso (L1):           Elastic Net:
    β₂                      β₂                    β₂
     |                       |                     |
     ●                      ╱│╲                   ╱●╲
    ╱ ╲                   ╱  │  ╲               ╱  |  ╲
   ╱   ╲                 ●───●───●             ●───●───●
  |     |                  ╲ | ╱                ╲  |  ╱
  |     |                   ╲│╱                  ╲ │ ╱
  ●─────●─── β₁              ●──── β₁             ●──── β₁
   ╲   ╱
    ╲ ╱
     ●

Коло/Еліпс           Ромб                   Проміжна форма
(smooth)            (гострі кути)          (smooth + кути)
β рідко = 0         β часто = 0            β іноді = 0
```

**Чому Lasso зануляє:**
- Оптимум часто попадає на кут ромба (осі координат)
- На осях одна з координат = 0

---

## Вибір λ (регуляризаційного параметра)

### Ефект λ

```
λ = 0:      Немає регуляризації (стандартна регресія)
            → може бути overfitting

λ малий:    Легка регуляризація
            → трохи зменшує overfitting

λ середній: Баланс між bias та variance
            → ОПТИМАЛЬНО ✓

λ великий:  Сильна регуляризація
            → може бути underfitting
```

### Підбір λ через Cross-Validation

**Найкращий метод:** Grid Search CV або RandomizedSearchCV

```
Test Error
    |
    |        ╱─────
    |       ╱
    |      ╱
    |     ╱
    |    ╱
    |___╱____________
    0   optimal λ    λ
```

**Оптимальний λ:** мінімум test error на CV.

---

## Код (Python + scikit-learn)

### Ridge Regression

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Дані
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ridge модель
ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X_train, y_train)

# Передбачення
y_pred = ridge.predict(X_test)

# Метрики
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Коефіцієнти
print(f"\nIntercept: {ridge.intercept_:.4f}")
print("Coefficients:")
for i, coef in enumerate(ridge.coef_):
    print(f"  β{i}: {coef:.4f}")
```

### Lasso Regression

```python
from sklearn.linear_model import Lasso

# Lasso модель
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Передбачення
y_pred = lasso.predict(X_test)

# Метрики
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Коефіцієнти (деякі будуть = 0)
print("\nCoefficients:")
for i, coef in enumerate(lasso.coef_):
    if coef != 0:  # Показуємо тільки ненульові
        print(f"  β{i}: {coef:.4f}")
    else:
        print(f"  β{i}: 0.0000 (ELIMINATED)")

# Скільки ознак залишилось
n_features_selected = np.sum(lasso.coef_ != 0)
print(f"\nFeatures selected: {n_features_selected} / {len(lasso.coef_)}")
```

### Elastic Net

```python
from sklearn.linear_model import ElasticNet

# Elastic Net
elastic = ElasticNet(
    alpha=1.0,      # λ
    l1_ratio=0.5    # α (mixing: 0=Ridge, 1=Lasso)
)
elastic.fit(X_train, y_train)

# Передбачення
y_pred = elastic.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Підбір оптимального λ (Cross-Validation)

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score

# Ridge з автоматичним підбором alpha
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Best alpha (Ridge): {ridge_cv.alpha_}")
print(f"R² on test: {ridge_cv.score(X_test, y_test):.4f}")

# Lasso CV
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

print(f"\nBest alpha (Lasso): {lasso_cv.alpha_}")
print(f"R² on test: {lasso_cv.score(X_test, y_test):.4f}")

# Elastic Net CV
elastic_cv = ElasticNetCV(
    alphas=alphas,
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
    cv=5,
    random_state=42
)
elastic_cv.fit(X_train, y_train)

print(f"\nBest alpha (Elastic): {elastic_cv.alpha_}")
print(f"Best l1_ratio: {elastic_cv.l1_ratio_}")
print(f"R² on test: {elastic_cv.score(X_test, y_test):.4f}")
```

### Візуалізація ефекту регуляризації

```python
import matplotlib.pyplot as plt

# Різні значення alpha
alphas = np.logspace(-3, 3, 100)
coefs_ridge = []
coefs_lasso = []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs_ridge.append(ridge.coef_)
    
    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefs_lasso.append(lasso.coef_)

coefs_ridge = np.array(coefs_ridge)
coefs_lasso = np.array(coefs_lasso)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Ridge
for i in range(coefs_ridge.shape[1]):
    axes[0].plot(alphas, coefs_ridge[:, i], label=f'Feature {i}')
axes[0].set_xscale('log')
axes[0].set_xlabel('Alpha (λ)', fontsize=12)
axes[0].set_ylabel('Coefficients', fontsize=12)
axes[0].set_title('Ridge: Coefficients vs Regularization', 
                  fontsize=14, fontweight='bold')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].grid(True, alpha=0.3)

# Lasso
for i in range(coefs_lasso.shape[1]):
    axes[1].plot(alphas, coefs_lasso[:, i], label=f'Feature {i}')
axes[1].set_xscale('log')
axes[1].set_xlabel('Alpha (λ)', fontsize=12)
axes[1].set_ylabel('Coefficients', fontsize=12)
axes[1].set_title('Lasso: Coefficients vs Regularization', 
                  fontsize=14, fontweight='bold')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Порівняння моделей

```python
from sklearn.preprocessing import StandardScaler

# Нормалізація (ВАЖЛИВО для регуляризації!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Моделі
models = {
    'No Regularization': Ridge(alpha=0),
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=1)': Ridge(alpha=1.0),
    'Ridge (α=10)': Ridge(alpha=10),
    'Lasso (α=0.1)': Lasso(alpha=0.1),
    'Lasso (α=1)': Lasso(alpha=1.0),
    'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Тренування та оцінка
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    n_nonzero = np.sum(model.coef_ != 0)
    
    results.append({
        'Model': name,
        'Train R²': train_score,
        'Test R²': test_score,
        'Overfitting': train_score - test_score,
        'Features': n_nonzero
    })

# Результати
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

---

## Регуляризація для Logistic Regression

### Ridge (L2)

```python
from sklearn.linear_model import LogisticRegression

# C = 1/λ (inverse regularization strength)
# Менше C → більша регуляризація
log_reg_ridge = LogisticRegression(
    penalty='l2',
    C=1.0,          # C=1 → λ=1
    solver='lbfgs',
    max_iter=1000
)

log_reg_ridge.fit(X_train, y_train)
```

### Lasso (L1)

```python
log_reg_lasso = LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear'  # або 'saga'
)

log_reg_lasso.fit(X_train, y_train)
```

### Elastic Net

```python
log_reg_elastic = LogisticRegression(
    penalty='elasticnet',
    C=1.0,
    l1_ratio=0.5,
    solver='saga'
)

log_reg_elastic.fit(X_train, y_train)
```

---

## Важливість нормалізації для регуляризації

### Чому потрібна?

**Регуляризація карає великі коефіцієнти:**
- Якщо ознаки в різних шкалах → коефіцієнти теж різні
- Ознака з великими значеннями → малий коефіцієнт
- Ознака з малими значеннями → великий коефіцієнт
- Регуляризація непропорційно карає другу ознаку!

### Приклад

```python
# БЕЗ нормалізації (ПОГАНО!)
X = [[1000, 1],      # Ознака 1: тисячі, Ознака 2: одиниці
     [2000, 2],
     [3000, 3]]

# Ridge без нормалізації
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
# β₁ ≈ 0.001 (мала, бо x₁ велика)
# β₂ ≈ 10 (велика, бо x₂ мала)
# Регуляризація більше карає β₂!

# З нормалізацією (ДОБРЕ!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge.fit(X_scaled, y)
# β₁, β₂ порівнянні за величиною
# Регуляризація справедлива!
```

### Правило

> **ЗАВЖДИ нормалізуй дані перед регуляризацією!**

```python
# ✅ ПРАВИЛЬНО
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge.fit(X_train_scaled, y_train)

# ❌ НЕПРАВИЛЬНО
ridge.fit(X_train, y_train)  # Без нормалізації
```

---

## Regularization Path

### Що це?

Графік зміни коефіцієнтів залежно від λ.

```python
from sklearn.linear_model import lasso_path, ridge_path

# Lasso path
alphas_lasso, coefs_lasso, _ = lasso_path(
    X_train_scaled, y_train, alphas=alphas
)

# Візуалізація
plt.figure(figsize=(12, 6))

for i in range(coefs_lasso.shape[0]):
    plt.plot(alphas_lasso, coefs_lasso[i], label=f'Feature {i}')

plt.xscale('log')
plt.xlabel('Alpha (λ)', fontsize=12)
plt.ylabel('Coefficients', fontsize=12)
plt.title('Lasso Regularization Path', fontsize=14, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Спостереження:**
- При малих λ: всі коефіцієнти ненульові
- При збільшенні λ: коефіцієнти по черзі зануляються
- Порядок зануляння → важливість ознак (останні важливіші)

---

## Переваги та недоліки

### Ridge (L2)

**Переваги ✓**
- ✅ Добре при мультиколінеарності
- ✅ Стабільні коефіцієнти
- ✅ Швидкі обчислення
- ✅ Працює при $p > n$

**Недоліки ✗**
- ❌ Не робить feature selection
- ❌ Всі ознаки залишаються в моделі
- ❌ Менша інтерпретованість

### Lasso (L1)

**Переваги ✓**
- ✅ Автоматична feature selection
- ✅ Sparse моделі (мало ознак)
- ✅ Висока інтерпретованість
- ✅ Працює як regularization + feature engineering

**Недоліки ✗**
- ❌ Нестабільний при корельованих ознаках
- ❌ Вибирає тільки одну з групи схожих ознак
- ❌ Повільніші обчислення

### Elastic Net

**Переваги ✓**
- ✅ Комбінує переваги Ridge та Lasso
- ✅ Feature selection + стабільність
- ✅ Добре при корельованих ознаках

**Недоліки ✗**
- ❌ Два гіперпараметри для підбору
- ❌ Складніша інтерпретація

---

## Коли використовувати який метод?

### Decision Tree 🌳

```
                Корельовані ознаки?
                /                  \
             Так                    Ні
              |                      |
       Feature selection?       Feature selection?
         /           \              /           \
       Так           Ні           Так           Ні
        |             |            |             |
  Elastic Net     Ridge        Lasso         Ridge
                               або
                          Elastic Net
```

### Практичні рекомендації

| Ситуація | Вибір |
|----------|-------|
| Мультиколінеарність + всі ознаки важливі | **Ridge** |
| Багато непотрібних ознак | **Lasso** |
| Корельовані ознаки + feature selection | **Elastic Net** |
| $p > n$ (більше ознак, ніж даних) | **Ridge** або **Elastic Net** |
| Потрібна інтерпретованість | **Lasso** |
| Невпевненість | **Elastic Net** (універсальний) |

---

## Практичні поради 💡

1. **ЗАВЖДИ нормалізуй** дані перед регуляризацією (StandardScaler)
2. **Cross-validation** для підбору λ — не вгадуй!
3. **Почни з Ridge** — простіший, швидший baseline
4. **Lasso для feature selection** — коли багато ознак
5. **Elastic Net при сумнівах** — універсальний вибір
6. **Regularization path** — подивись, як зникають ознаки
7. **Не регуляризуй intercept** — тільки коефіцієнти ознак
8. **Grid Search** для Elastic Net — підбирай α та λ разом
9. **Порівнюй train vs test** — регуляризація зменшує overfitting
10. **Feature importance** з Lasso — які ознаки залишились?

---

## Поширені помилки ❌

### 1. Забути нормалізувати

```python
# ❌ НЕПРАВИЛЬНО
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# ✅ ПРАВИЛЬНО
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
ridge.fit(X_train_scaled, y_train)
```

### 2. Підбирати λ на train без CV

```python
# ❌ НЕПРАВИЛЬНО
best_alpha = None
best_score = -np.inf
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    score = ridge.score(X_train, y_train)  # Overfitting!
    if score > best_score:
        best_score = score
        best_alpha = alpha

# ✅ ПРАВИЛЬНО
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.alpha_
```

### 3. Використовувати занадто великий λ

```python
# Якщо λ → ∞, всі коефіцієнти → 0
# Модель стає константою (underfitting)
ridge = Ridge(alpha=1e10)  # ❌ Занадто багато!
```

### 4. Не врахувати різницю між α (sklearn) та λ

У scikit-learn для Logistic Regression:
- **C = 1/λ**
- Більше C → менша регуляризація
- Менше C → більша регуляризація

```python
# Ridge Regression
Ridge(alpha=1.0)  # alpha = λ

# Logistic Regression
LogisticRegression(C=1.0)  # C = 1/λ
```

---

## Математичне пояснення

### Чому L1 зануляє коефіцієнти?

**Градієнт L1:**
$$\frac{\partial}{\partial \beta_j} |\beta_j| = \text{sign}(\beta_j) = \begin{cases}
+1 & \text{якщо } \beta_j > 0 \\
-1 & \text{якщо } \beta_j < 0 \\
\text{undefined} & \text{якщо } \beta_j = 0
\end{cases}$$

**Градієнт L2:**
$$\frac{\partial}{\partial \beta_j} \beta_j^2 = 2\beta_j$$

**Різниця:**
- L1: константний градієнт → може досягти 0
- L2: градієнт пропорційний β → ніколи не досягає 0

### Геометрична інтуїція

**L1 constraint region:** ромб (гострі кути на осях)
**L2 constraint region:** коло (гладке)

Оптимум часто попадає на кут ромба = координата дорівнює 0.

---

## Пов'язані теми

- [[01_Linear_Regression]] — базова регресія
- [[02_Logistic_Regression]] — класифікація з регуляризацією
- [[Cross_Validation]] — підбір гіперпараметрів
- [[Feature_Selection]] — відбір ознак
- [[Bias_Variance_Tradeoff]] — теорія overfitting

## Ресурси

- [Scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [StatQuest: Regularization](https://www.youtube.com/watch?v=Q81RR3yKn30)
- [Andrew Ng: Regularization](https://www.coursera.org/learn/machine-learning)
- [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

---

## Ключові висновки

> Регуляризація додає штраф до функції втрат для запобігання overfitting та контролю складності моделі.

**Основні типи:**
- **Ridge (L2):** $\text{MSE} + \lambda \sum \beta_j^2$ — зменшує коефіцієнти
- **Lasso (L1):** $\text{MSE} + \lambda \sum |\beta_j|$ — зануляє коефіцієнти
- **Elastic Net:** комбінація L1 + L2

**Ключові принципи:**
- Завжди нормалізуй дані
- Підбирай λ через cross-validation
- Ridge для стабільності, Lasso для feature selection
- Elastic Net — універсальний вибір

**Коли використовувати:**
- Overfitting + багато ознак + мультиколінеарність = Regularization ✓

---

#ml #supervised-learning #regularization #ridge #lasso #elastic-net #overfitting
