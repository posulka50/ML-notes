# Evaluation Metrics (Метрики оцінки)

## Що це?

**Evaluation Metrics** — це **кількісні показники**, які вимірюють **якість роботи** ML моделі. Вони дозволяють об'єктивно оцінити, наскільки добре модель виконує свою задачу.

**Головна ідея:** різні задачі потребують різних метрик. Те, що добре для регресії, не підходить для класифікації. Вибір правильної метрики критично важливий для успіху проекту.

## Навіщо потрібно?

- 🎯 **Оцінка якості** — чи добре працює модель
- 📊 **Порівняння моделей** — яка краща
- 🔍 **Виявлення проблем** — де модель помиляється
- ⚙️ **Оптимізація** — що покращувати
- 💼 **Бізнес-рішення** — чи готова модель до production
- 📈 **Моніторинг** — відстеження деградації в production

## Коли використовувати?

**Завжди!** Після навчання будь-якої моделі потрібно оцінити її якість.

**Важливо:**
- Різні метрики для регресії та класифікації
- Враховувати business context (не тільки accuracy!)
- Використовувати кілька метрик одночасно
- Розуміти trade-offs між метриками

---

## Класифікація метрик

```
Evaluation Metrics
│
├── Regression Metrics
│   ├── MAE (Mean Absolute Error)
│   ├── MSE (Mean Squared Error)
│   ├── RMSE (Root Mean Squared Error)
│   ├── R² (R-squared / Coefficient of Determination)
│   ├── MAPE (Mean Absolute Percentage Error)
│   └── Adjusted R²
│
└── Classification Metrics
    ├── Accuracy
    ├── Precision
    ├── Recall (Sensitivity)
    ├── F1-Score
    ├── Specificity
    ├── ROC-AUC
    ├── Confusion Matrix
    ├── Cohen's Kappa
    └── Log Loss
```

---

# REGRESSION METRICS

## 1. MAE (Mean Absolute Error)

### Формула

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

де:
- $y_i$ — справжнє значення
- $\hat{y}_i$ — передбачення
- $n$ — кількість зразків

### Інтуїція

**MAE** — це **середня абсолютна різниця** між передбаченням та реальністю.

```
Реальні ціни:        [100, 200, 150, 180]
Передбачення:        [110, 190, 160, 170]
Абсолютні помилки:   [ 10,  10,  10,  10]

MAE = (10 + 10 + 10 + 10) / 4 = 10

Інтерпретація: в середньому модель помиляється на 10 грн
```

### Код

```python
from sklearn.metrics import mean_absolute_error
import numpy as np

# Дані
y_true = np.array([100, 200, 150, 180])
y_pred = np.array([110, 190, 160, 170])

# MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.2f}")

# Або вручну
mae_manual = np.mean(np.abs(y_true - y_pred))
print(f"MAE (manual): {mae_manual:.2f}")
```

### Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Проста інтерпретація | ❌ Не карає великі помилки сильніше |
| ✅ Стійка до викидів | ❌ Не диференційовна в 0 |
| ✅ В тих самих одиницях, що y | ❌ Всі помилки рівноцінні |

### Коли використовувати?

- ✅ Коли всі помилки однаково важливі
- ✅ Коли є викиди в даних
- ✅ Коли потрібна проста інтерпретація
- ❌ Коли великі помилки критичніші (використовуй MSE)

---

## 2. MSE (Mean Squared Error)

### Формула

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Інтуїція

**MSE** — це **середній квадрат помилки**. Піднесення до квадрата **карає великі помилки сильніше**.

```
Реальні ціни:        [100, 200, 150, 180]
Передбачення:        [110, 190, 160, 170]
Помилки:             [ 10,  10,  10,  10]
Квадрати помилок:    [100, 100, 100, 100]

MSE = (100 + 100 + 100 + 100) / 4 = 100

Але якщо одна помилка велика:
Помилки:             [ 5,  5,  5,  35]
Квадрати:            [25, 25, 25, 1225]

MSE = (25 + 25 + 25 + 1225) / 4 = 325  ← Набагато більше!
```

### Код

```python
from sklearn.metrics import mean_squared_error

y_true = np.array([100, 200, 150, 180])
y_pred = np.array([110, 190, 160, 170])

# MSE
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.2f}")

# Вручну
mse_manual = np.mean((y_true - y_pred) ** 2)
print(f"MSE (manual): {mse_manual:.2f}")
```

### Візуалізація: MSE vs MAE

```python
import matplotlib.pyplot as plt

errors = np.linspace(-10, 10, 100)

mae_values = np.abs(errors)
mse_values = errors ** 2

plt.figure(figsize=(10, 6))
plt.plot(errors, mae_values, label='MAE = |error|', linewidth=2)
plt.plot(errors, mse_values, label='MSE = error²', linewidth=2)
plt.xlabel('Error', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('MAE vs MSE: How they penalize errors', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# MSE сильно карає великі помилки!
```

### Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Карає великі помилки сильніше | ❌ Чутлива до викидів |
| ✅ Диференційовна (для градієнтів) | ❌ Одиниці — квадрат y (складніше інтерпретувати) |
| ✅ Популярна loss function | ❌ Може бути занадто песимістичною |

### Коли використовувати?

- ✅ Коли великі помилки **неприйнятні**
- ✅ Для оптимізації (loss function)
- ✅ Коли немає значних викидів
- ❌ Коли є викиди (використовуй MAE)

---

## 3. RMSE (Root Mean Squared Error)

### Формула

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}$$

### Інтуїція

**RMSE** — це **корінь з MSE**, повертає метрику в **оригінальні одиниці** y.

```
MSE = 100 (у квадраті грн)
RMSE = √100 = 10 грн  ← В тих самих одиницях!

Інтерпретація: модель в середньому помиляється на ~10 грн
```

### Код

```python
from sklearn.metrics import mean_squared_error
import numpy as np

y_true = np.array([100, 200, 150, 180])
y_pred = np.array([110, 190, 160, 170])

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.2f}")

# Або через параметр
rmse = mean_squared_error(y_true, y_pred, squared=False)
print(f"RMSE: {rmse:.2f}")
```

### Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ В оригінальних одиницях | ❌ Чутлива до викидів |
| ✅ Карає великі помилки | ❌ Складніше оптимізувати (через корінь) |
| ✅ Інтуїтивно зрозуміла | |

### MAE vs RMSE: Коли що?

```python
# Приклад з викидом
y_true = np.array([100, 100, 100, 100, 100])

# Модель A: всі помилки маленькі
y_pred_A = np.array([105, 95, 102, 98, 103])

# Модель B: один великий викид
y_pred_B = np.array([100, 100, 100, 100, 150])

mae_A = mean_absolute_error(y_true, y_pred_A)
mae_B = mean_absolute_error(y_true, y_pred_B)
rmse_A = mean_squared_error(y_true, y_pred_A, squared=False)
rmse_B = mean_squared_error(y_true, y_pred_B, squared=False)

print("Model A (consistent small errors):")
print(f"  MAE:  {mae_A:.2f}")
print(f"  RMSE: {rmse_A:.2f}")

print("\nModel B (one large outlier):")
print(f"  MAE:  {mae_B:.2f}")
print(f"  RMSE: {rmse_B:.2f}")

# Вивід:
# Model A: MAE ≈ 3.4, RMSE ≈ 3.7
# Model B: MAE = 10,  RMSE ≈ 22.4  ← RMSE карає сильніше!
```

---

## 4. R² (R-squared / Coefficient of Determination)

### Формула

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

де:
- $\text{SS}_{\text{res}}$ — сума квадратів залишків (residual sum of squares)
- $\text{SS}_{\text{tot}}$ — загальна сума квадратів (total sum of squares)
- $\bar{y}$ — середнє значення y

### Інтуїція

**R²** показує, **яка частка варіації y пояснюється моделлю**.

```
R² = 1.0  → Модель ідеально передбачає (100% варіації пояснено)
R² = 0.8  → Модель пояснює 80% варіації (добре!)
R² = 0.5  → Модель пояснює 50% варіації (середньо)
R² = 0.0  → Модель не краща за просте середнє
R² < 0    → Модель ГІРШЕ за середнє! ❌
```

### Візуальне пояснення

```
Дані: y = [10, 20, 30, 40, 50]
Середнє: ȳ = 30

Baseline (просто середнє):
Передбачення: [30, 30, 30, 30, 30]
SSₜₒₜ = (10-30)² + (20-30)² + (30-30)² + (40-30)² + (50-30)²
      = 400 + 100 + 0 + 100 + 400 = 1000

Наша модель:
Передбачення: [12, 21, 29, 39, 48]
SSᵣₑₛ = (10-12)² + (20-21)² + (30-29)² + (40-39)² + (50-48)²
      = 4 + 1 + 1 + 1 + 4 = 11

R² = 1 - (11 / 1000) = 1 - 0.011 = 0.989  ← Чудово! 98.9%
```

### Код

```python
from sklearn.metrics import r2_score

y_true = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 21, 29, 39, 48])

# R²
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")

# Вручну
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - y_true.mean()) ** 2)
r2_manual = 1 - (ss_res / ss_tot)
print(f"R² (manual): {r2_manual:.4f}")
```

### Візуалізація R²

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Генерація даних
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X.ravel() + 1 + np.random.normal(0, 2, 50)

# Модель
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# R²
r2 = r2_score(y, y_pred)

# Візуалізація
plt.figure(figsize=(12, 5))

# Subplot 1: Модель
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.6, s=50, label='Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label='Model')
plt.axhline(y=y.mean(), color='green', linestyle='--', 
            linewidth=2, label=f'Baseline (mean={y.mean():.2f})')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Linear Regression (R² = {r2:.3f})', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Subplot 2: Residuals
plt.subplot(1, 2, 2)
residuals = y - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, s=50)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Інтуїтивна (від 0 до 1) | ❌ Може бути < 0 (погана модель) |
| ✅ Незалежна від одиниць | ❌ Завжди зростає з додаванням ознак |
| ✅ Популярна метрика | ❌ Не підходить для нелінійних залежностей |

### Коли використовувати?

- ✅ Основна метрика для регресії
- ✅ Порівняння моделей
- ✅ Пояснення якості моделі стейкхолдерам
- ❌ Для моделей з великою кількістю ознак (використовуй Adjusted R²)

---

## 5. Adjusted R²

### Формула

$$R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

де:
- $n$ — кількість зразків
- $p$ — кількість ознак (предикторів)

### Навіщо?

**Проблема R²:** завжди зростає при додаванні нових ознак, навіть якщо вони безкорисні.

**Adjusted R²** **карає** за додавання непотрібних ознак.

```python
from sklearn.linear_model import LinearRegression

# Генерація даних
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1)  # 1 корисна ознака
y = 2 * X.ravel() + 1 + np.random.normal(0, 0.5, n_samples)

# Додамо 10 випадкових (безкорисних) ознак
X_noise = np.random.randn(n_samples, 10)
X_with_noise = np.hstack([X, X_noise])

# Модель з 1 ознакою
model_simple = LinearRegression()
model_simple.fit(X, y)
r2_simple = model_simple.score(X, y)

# Модель з 11 ознаками
model_complex = LinearRegression()
model_complex.fit(X_with_noise, y)
r2_complex = model_complex.score(X_with_noise, y)

# Adjusted R²
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

adj_r2_simple = adjusted_r2(r2_simple, n_samples, 1)
adj_r2_complex = adjusted_r2(r2_complex, n_samples, 11)

print(f"Simple model (1 feature):")
print(f"  R²:          {r2_simple:.4f}")
print(f"  Adjusted R²: {adj_r2_simple:.4f}")

print(f"\nComplex model (11 features):")
print(f"  R²:          {r2_complex:.4f}")  # Трохи вище!
print(f"  Adjusted R²: {adj_r2_complex:.4f}")  # Але Adj R² нижче!

# R² зросло через додавання ознак
# Але Adjusted R² впало → модель не покращилась насправді!
```

---

## 6. MAPE (Mean Absolute Percentage Error)

### Формула

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

### Інтуїція

**MAPE** — це **середня відсоткова помилка**.

```
Реальна ціна: 100 грн
Передбачення: 110 грн
Помилка: |100 - 110| / 100 = 10 / 100 = 10%

MAPE = 10% → модель в середньому помиляється на 10%
```

### Код

```python
from sklearn.metrics import mean_absolute_percentage_error

y_true = np.array([100, 200, 150, 180])
y_pred = np.array([110, 190, 160, 170])

# MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")

# Вручну
mape_manual = np.mean(np.abs((y_true - y_pred) / y_true))
print(f"MAPE (manual): {mape_manual*100:.2f}%")
```

### Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Інтуїтивна (відсотки) | ❌ Не визначена для y=0 |
| ✅ Незалежна від масштабу | ❌ Асиметрична (карає недооцінку сильніше) |
| ✅ Легко пояснити бізнесу | ❌ Чутлива до малих значень y |

### Проблема MAPE з нулями

```python
y_true = np.array([100, 200, 0, 180])  # Один нуль!
y_pred = np.array([110, 190, 10, 170])

# MAPE
try:
    mape = mean_absolute_percentage_error(y_true, y_pred)
except:
    print("Error: Division by zero!")

# Альтернатива: sMAPE (symmetric MAPE)
def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )

smape_value = smape(y_true, y_pred)
print(f"sMAPE: {smape_value:.2f}%")
```

---

## Порівняння Regression Metrics

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             r2_score, mean_absolute_percentage_error)
import pandas as pd

# Дані
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Модель
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Всі метрики
metrics = {
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': mean_squared_error(y_test, y_pred, squared=False),
    'R²': r2_score(y_test, y_pred),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
}

print("="*50)
print("REGRESSION METRICS COMPARISON")
print("="*50)
for metric, value in metrics.items():
    if metric == 'MAPE':
        print(f"{metric:10s}: {value:.2f}%")
    elif metric == 'R²':
        print(f"{metric:10s}: {value:.4f}")
    else:
        print(f"{metric:10s}: {value:.4f}")

# Інтерпретація
print("\n" + "="*50)
print("INTERPRETATION")
print("="*50)
print(f"MAE:  Model is off by ±{metrics['MAE']:.2f} on average")
print(f"RMSE: Root mean squared error is {metrics['RMSE']:.2f}")
print(f"R²:   Model explains {metrics['R²']*100:.1f}% of variance")
print(f"MAPE: Average error is {metrics['MAPE']:.1f}%")
```

### Коли яку метрику використовувати?

| Ситуація | Рекомендована метрика |
|----------|----------------------|
| **Загальна оцінка** | R² |
| **Є викиди** | MAE |
| **Великі помилки критичні** | RMSE, MSE |
| **Потрібні відсотки** | MAPE |
| **Багато ознак** | Adjusted R² |
| **Бізнес-комунікація** | MAPE, R² |
| **Loss function** | MSE |

---

# CLASSIFICATION METRICS

## Confusion Matrix (Матриця плутанини)

### Що це?

**Confusion Matrix** — це **таблиця**, яка показує, як модель класифікує зразки.

```
                 Predicted
                 Positive  Negative
Actual  Positive    TP        FN
        Negative    FP        TN

TP (True Positive)  — правильно передбачили Positive
TN (True Negative)  — правильно передбачили Negative
FP (False Positive) — неправильно передбачили Positive (Type I error)
FN (False Negative) — неправильно передбачили Negative (Type II error)
```

### Приклад

```
Задача: діагностика хвороби

100 пацієнтів:
- 60 хворих
- 40 здорових

Модель передбачила:
- 55 хворих (з них 50 справді хворі, 5 помилково)
- 45 здорових (з них 35 справді здорові, 10 помилково)

Confusion Matrix:
                Predicted
                Sick  Healthy
Actual  Sick     50      10     ← 50 TP, 10 FN
        Healthy   5      35     ← 5 FP, 35 TN

TP = 50  (правильно знайшли хворих)
TN = 35  (правильно знайшли здорових)
FP = 5   (помилково діагностували хворобу)
FN = 10  (пропустили хворих!) ← КРИТИЧНО!
```

### Код

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Дані
y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Візуалізація
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Negative', 'Positive']
)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Елементи матриці
tn, fp, fn, tp = cm.ravel()
print(f"\nTP (True Positive):  {tp}")
print(f"TN (True Negative):  {tn}")
print(f"FP (False Positive): {fp}")
print(f"FN (False Negative): {fn}")
```

---

## 7. Accuracy (Точність)

### Формула

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Правильні}}{\text{Всього}}$$

### Інтуїція

**Accuracy** — це **частка правильних передбачень**.

```
TP = 50, TN = 35, FP = 5, FN = 10

Accuracy = (50 + 35) / (50 + 35 + 5 + 10)
         = 85 / 100
         = 0.85 = 85%

Модель правильна в 85% випадків
```

### Код

```python
from sklearn.metrics import accuracy_score

y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f} ({accuracy*100:.0f}%)")

# Вручну
correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
total = len(y_true)
accuracy_manual = correct / total
print(f"Accuracy (manual): {accuracy_manual:.2f}")
```

### Проблема Accuracy: Несбалансовані класи

```python
# 95% здорових, 5% хворих
y_true = [0]*95 + [1]*5

# "Модель", яка ЗАВЖДИ каже "здоровий"
y_pred = [0]*100

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # 95%! ← Але модель безглузда!

# Модель не знайшла ЖОДНОГО хворого, але accuracy висока!
```

**Висновок:** Accuracy **не підходить** для несбалансованих класів!

---

## 8. Precision (Точність позитивних передбачень)

### Формула

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{Правильні позитивні}}{\text{Всі позитивні передбачення}}$$

### Інтуїція

**Precision** відповідає на питання: **"Коли модель каже Positive, наскільки часто вона права?"**

```
Модель передбачила 55 "хворих"
З них 50 справді хворі, 5 — помилково

Precision = 50 / (50 + 5) = 50 / 55 ≈ 0.91 = 91%

Коли модель каже "хворий", вона права в 91% випадків
```

### Приклад: Email Spam Filter

```
100 emails:
- 20 spam
- 80 not spam

Модель позначила 25 як spam:
- 18 справді spam (TP)
- 7 хороших листів (FP) ← Помилково в спам!

Precision = 18 / (18 + 7) = 18 / 25 = 0.72 = 72%

72% листів у спамі — справді спам
28% — хороші листи, які ми втратили! ❌
```

### Код

```python
from sklearn.metrics import precision_score

y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]

precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")

# Вручну через confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
precision_manual = tp / (tp + fp)
print(f"Precision (manual): {precision_manual:.4f}")
```

### Коли важлива висока Precision?

- 🚫 **Spam фільтр** — не хочемо втрачати важливі листи
- 💊 **Рекомендація ліків** — помилкова рекомендація небезпечна
- 📺 **Рекламні кампанії** — показувати рекламу тільки зацікавленим
- 🏦 **Виявлення шахрайства** — не блокувати чесних клієнтів

---

## 9. Recall (Повнота / Sensitivity / True Positive Rate)

### Формула

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{Правильні позитивні}}{\text{Всі справжні позитивні}}$$

### Інтуїція

**Recall** відповідає на питання: **"Скільки справжніх Positive ми знайшли?"**

```
60 справді хворих пацієнтів
Модель знайшла 50 з них
10 пропустила (FN)

Recall = 50 / (50 + 10) = 50 / 60 ≈ 0.83 = 83%

Знайшли 83% хворих
Пропустили 17% ← Це може бути критично!
```

### Приклад: Діагностика раку

```
100 пацієнтів:
- 10 з раком
- 90 без раку

Модель знайшла рак у 8 пацієнтів:
- 7 справді з раком (TP)
- 1 помилково (FP)
- 3 з раком пропущені! (FN) ← КРИТИЧНО!

Recall = 7 / (7 + 3) = 7 / 10 = 0.7 = 70%

Знайшли тільки 70% хворих на рак!
30% пропустили — це неприйнятно! ❌
```

### Код

```python
from sklearn.metrics import recall_score

y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]

recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")

# Вручну
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
recall_manual = tp / (tp + fn)
print(f"Recall (manual): {recall_manual:.4f}")
```

### Коли важлива висока Recall?

- 🏥 **Діагностика хвороб** — не можна пропускати хворих
- 🔒 **Виявлення вторгнень** — краще хибна тривога, ніж пропущена атака
- 🔍 **Пошук зниклих людей** — потрібно знайти ВСІХ
- ⚠️ **Виявлення дефектів** — не можна пропускати бракованi вироби

---

## Precision vs Recall Trade-off

### Конфлікт

```
Precision ↑  →  Recall ↓
Recall ↑     →  Precision ↓

Не можна максимізувати обидві одночасно!
```

### Візуалізація

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Генерація даних
X, y = make_classification(n_samples=1000, n_classes=2, n_features=20,
                          n_informative=15, random_state=42)

# Модель
model = LogisticRegression()
model.fit(X, y)

# Ймовірності
y_proba = model.predict_proba(X)[:, 1]

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y, y_proba)

# Візуалізація
plt.figure(figsize=(12, 5))

# Subplot 1: Precision-Recall Curve
plt.subplot(1, 2, 1)
plt.plot(recall, precision, linewidth=2, label='PR Curve')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Thresholds
plt.subplot(1, 2, 2)
plt.plot(thresholds, precision[:-1], label='Precision', linewidth=2)
plt.plot(thresholds, recall[:-1], label='Recall', linewidth=2)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision & Recall vs Threshold', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Приклад: Різні пороги

```python
from sklearn.metrics import precision_score, recall_score

# Ймовірності
y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
y_proba = np.array([0.9, 0.8, 0.6, 0.4, 0.5, 0.3, 0.2, 0.1])

# Різні пороги
thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"\nThreshold = {threshold}:")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall:    {recall:.2f}")
    print(f"  Predictions: {y_pred}")

# Вивід:
# Threshold = 0.3:  Precision: 0.67, Recall: 1.00  ← Високий Recall
# Threshold = 0.5:  Precision: 0.75, Recall: 0.75  ← Баланс
# Threshold = 0.7:  Precision: 1.00, Recall: 0.50  ← Висока Precision
```

---

## 10. F1-Score

### Формула

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### Інтуїція

**F1-Score** — це **гармонічне середнє** Precision та Recall. Балансує між ними.

```
Precision = 0.8
Recall = 0.6

Arithmetic mean: (0.8 + 0.6) / 2 = 0.7
Harmonic mean (F1): 2 * (0.8 * 0.6) / (0.8 + 0.6) ≈ 0.69

F1 ближче до меншого значення!
```

### Чому гармонічне середнє?

```python
# Порівняння середніх
precision = 0.9
recall = 0.1  # Дуже низький!

arithmetic_mean = (precision + recall) / 2
harmonic_mean = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Arithmetic mean: {arithmetic_mean:.2f}")  # 0.50
print(f"F1 (harmonic):   {harmonic_mean:.2f}")    # 0.18

# F1 карає дисбаланс між Precision і Recall!
# Arithmetic mean була б занадто оптимістичною
```

### Код

```python
from sklearn.metrics import f1_score

y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]

f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.4f}")

# Вручну
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1_manual = 2 * (precision * recall) / (precision + recall)
print(f"F1-Score (manual): {f1_manual:.4f}")
```

### F-beta Score (узагальнення)

```python
from sklearn.metrics import fbeta_score

# β контролює баланс між Precision і Recall

# F0.5: Precision важливіша
f05 = fbeta_score(y_true, y_pred, beta=0.5)

# F1: Рівний баланс
f1 = fbeta_score(y_true, y_pred, beta=1.0)

# F2: Recall важливіший
f2 = fbeta_score(y_true, y_pred, beta=2.0)

print(f"F0.5: {f05:.4f}  (favor Precision)")
print(f"F1:   {f1:.4f}   (balanced)")
print(f"F2:   {f2:.4f}   (favor Recall)")
```

---

## 11. Specificity (True Negative Rate)

### Формула

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{\text{Правильні негативні}}{\text{Всі справжні негативні}}$$

### Інтуїція

**Specificity** — це **"Recall для негативного класу"**. Скільки справжніх Negative ми правильно визначили.

```
90 здорових пацієнтів
Модель правильно визначила 85 як здорових
5 помилково діагностувала як хворих (FP)

Specificity = 85 / (85 + 5) = 85 / 90 ≈ 0.94 = 94%

94% здорових людей правильно визначені як здорові
```

### Код

```python
# Specificity немає в sklearn, рахуємо вручну
from sklearn.metrics import confusion_matrix

y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")

# Також можна через recall_score з позитивним класом = 0
from sklearn.metrics import recall_score
specificity_alt = recall_score(y_true, y_pred, pos_label=0)
print(f"Specificity (alternative): {specificity_alt:.4f}")
```

### Коли важлива Specificity?

- 🩺 **Скринінг** — не хочемо панікувати здорових людей
- 💳 **Антифрод** — не блокувати нормальні транзакції
- 📧 **Email фільтрація** — не видаляти важливі листи

---

## 12. ROC Curve & AUC

### ROC Curve (Receiver Operating Characteristic)

**ROC Curve** показує **trade-off між True Positive Rate (Recall) та False Positive Rate** при різних порогах.

```
True Positive Rate (TPR) = Recall = TP / (TP + FN)
False Positive Rate (FPR) = FP / (FP + TN) = 1 - Specificity
```

### Візуалізація

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Дані
X, y = make_classification(n_samples=1000, n_classes=2, n_features=20,
                          n_informative=15, random_state=42)

# Модель
model = LogisticRegression()
model.fit(X, y)

# Ймовірності
y_proba = model.predict_proba(X)[:, 1]

# ROC Curve
fpr, tpr, thresholds = roc_curve(y, y_proba)
auc = roc_auc_score(y, y_proba)

# Візуалізація
plt.figure(figsize=(8, 7))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

# Оптимальна точка (найближча до (0, 1))
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
         label=f'Optimal Threshold = {optimal_threshold:.2f}')

plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"AUC: {auc:.4f}")
print(f"Optimal threshold: {optimal_threshold:.4f}")
```

### Інтерпретація AUC

```
AUC = 1.0  → Ідеальний класифікатор ⭐
AUC = 0.9  → Відмінний
AUC = 0.8  → Добрий
AUC = 0.7  → Середній
AUC = 0.5  → Випадковий (як підкидання монети)
AUC < 0.5  → Гірше випадкового (щось не так!) ❌
```

### Код

```python
from sklearn.metrics import roc_auc_score

# Binary classification
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
y_proba = np.array([0.1, 0.3, 0.6, 0.8, 0.9, 0.2, 0.7, 0.4])

auc = roc_auc_score(y_true, y_proba)
print(f"AUC: {auc:.4f}")

# Multiclass (one-vs-rest)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_proba_multi = model.predict_proba(X_test)

auc_multi = roc_auc_score(y_test, y_proba_multi, multi_class='ovr')
print(f"AUC (multiclass): {auc_multi:.4f}")
```

### Переваги AUC

| Переваги | Недоліки |
|----------|----------|
| ✅ Не залежить від порогу | ❌ Складніше інтерпретувати |
| ✅ Працює з несбалансованими класами | ❌ Може бути оптимістичною |
| ✅ Одне число для порівняння | ❌ Не показує конкретні Precision/Recall |

---

## 13. Log Loss (Cross-Entropy Loss)

### Формула

$$\text{Log Loss} = -\frac{1}{n}\sum_{i=1}^{n} [y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)]$$

де:
- $y_i$ — справжній клас (0 або 1)
- $\hat{p}_i$ — передбачена ймовірність класу 1

### Інтуїція

**Log Loss** карає **впевнені неправильні передбачення** дуже сильно.

```
Справжній клас: 1 (Positive)

Передбачення: p = 0.9  → Log Loss = -log(0.9) ≈ 0.10   ✓ Мала помилка
Передбачення: p = 0.5  → Log Loss = -log(0.5) ≈ 0.69   ⚠️ Середня
Передбачення: p = 0.1  → Log Loss = -log(0.1) ≈ 2.30   ❌ Велика помилка!
Передбачення: p = 0.01 → Log Loss = -log(0.01) ≈ 4.61  ❌❌ Дуже велика!
```

### Код

```python
from sklearn.metrics import log_loss

y_true = [1, 0, 1, 1, 0]

# Гарні передбачення
y_proba_good = [
    [0.1, 0.9],   # Клас 1, ймовірність 0.9 ✓
    [0.8, 0.2],   # Клас 0, ймовірність 0.8 ✓
    [0.2, 0.8],   # Клас 1, ймовірність 0.8 ✓
    [0.1, 0.9],   # Клас 1, ймовірність 0.9 ✓
    [0.9, 0.1]    # Клас 0, ймовірність 0.9 ✓
]

# Погані передбачення
y_proba_bad = [
    [0.9, 0.1],   # Клас 1, ймовірність 0.1 ❌
    [0.2, 0.8],   # Клас 0, ймовірність 0.2 ❌
    [0.8, 0.2],   # Клас 1, ймовірність 0.2 ❌
    [0.9, 0.1],   # Клас 1, ймовірність 0.1 ❌
    [0.1, 0.9]    # Клас 0, ймовірність 0.1 ❌
]

loss_good = log_loss(y_true, y_proba_good)
loss_bad = log_loss(y_true, y_proba_bad)

print(f"Log Loss (good predictions): {loss_good:.4f}")
print(f"Log Loss (bad predictions):  {loss_bad:.4f}")
```

### Коли використовувати?

- ✅ Коли важливі **калібровані ймовірності**
- ✅ Для **loss function** в neural networks
- ✅ Kaggle competitions
- ❌ Коли потрібна проста інтерпретація (використовуй Accuracy/F1)

---

## 14. Cohen's Kappa

### Формула

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

де:
- $p_o$ — observed agreement (accuracy)
- $p_e$ — expected agreement by chance

### Інтуїція

**Cohen's Kappa** враховує **випадкові збіги**. Корисна для несбалансованих класів.

```
100 зразків: 90 негативних, 10 позитивних

Модель завжди каже "негативний":
Accuracy = 90% ← Здається добре!

Але випадкова модель також дасть ~90% (просто через дисбаланс)

Kappa = (0.9 - 0.9) / (1 - 0.9) = 0 / 0.1 = 0
← Модель не краща за випадкову!
```

### Інтерпретація

```
κ = 1.0   → Ідеальна згода
κ = 0.8   → Майже ідеальна
κ = 0.6   → Суттєва
κ = 0.4   → Помірна
κ = 0.2   → Слабка
κ = 0.0   → Випадкова (не краща за chance)
κ < 0     → Гірше випадкової
```

### Код

```python
from sklearn.metrics import cohen_kappa_score

y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

kappa = cohen_kappa_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Cohen's Kappa: {kappa:.2f}")

# Модель, яка завжди каже "0"
y_pred_always_0 = [0] * 10

kappa_bad = cohen_kappa_score(y_true, y_pred_always_0)
accuracy_bad = accuracy_score(y_true, y_pred_always_0)

print(f"\nAlways-0 model:")
print(f"Accuracy: {accuracy_bad:.2f}")  # 0.60
print(f"Cohen's Kappa: {kappa_bad:.2f}")  # Близько до 0
```

---

## Multiclass Classification Metrics

### Macro vs Micro vs Weighted Average

```python
from sklearn.metrics import classification_report

# 3 класи
y_true = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
y_pred = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0]

report = classification_report(y_true, y_pred, 
                               target_names=['Class 0', 'Class 1', 'Class 2'])
print(report)
```

**Вивід:**
```
              precision    recall  f1-score   support

     Class 0       0.75      0.75      0.75         4
     Class 1       0.75      0.75      0.75         4
     Class 2       1.00      0.80      0.89         5

    accuracy                           0.83        13
   macro avg       0.83      0.77      0.80        13
weighted avg       0.86      0.77      0.81        13
```

**Пояснення:**
- **Macro average:** Проста середня по всіх класах (не враховує дисбаланс)
- **Weighted average:** Зважена середня (враховує кількість зразків у кожному класі)
- **Micro average:** Загальна метрика (як accuracy для multiclass)

### Приклад

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
y_pred = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0]

# Macro: середнє по класах
f1_macro = f1_score(y_true, y_pred, average='macro')

# Micro: загальне
f1_micro = f1_score(y_true, y_pred, average='micro')

# Weighted: зважене
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"F1 Macro:    {f1_macro:.4f}")
print(f"F1 Micro:    {f1_micro:.4f}")  # = Accuracy для multiclass
print(f"F1 Weighted: {f1_weighted:.4f}")
```

---

## Порівняння Classification Metrics

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, cohen_kappa_score,
                             log_loss, confusion_matrix)
import pandas as pd

# Дані
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Передбачення
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Всі метрики
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Specificity': tn / (tn + fp),
    'F1-Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_proba),
    'Cohen\'s Kappa': cohen_kappa_score(y_test, y_pred),
    'Log Loss': log_loss(y_test, y_proba)
}

print("="*60)
print("CLASSIFICATION METRICS SUMMARY")
print("="*60)

for metric, value in metrics.items():
    print(f"{metric:20s}: {value:.4f}")

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
print(f"TP (True Positive):  {tp}")
print(f"TN (True Negative):  {tn}")
print(f"FP (False Positive): {fp}")
print(f"FN (False Negative): {fn}")
```

---

## Вибір метрики: Decision Tree

```
                    START
                      |
         ┌────────────┴────────────┐
         │                         │
    Regression?              Classification?
         │                         │
         ▼                         ▼
    ┌────────┐            ┌─────────────┐
    │        │            │             │
    │  Yes   │            │     Yes     │
    │        │            │             │
    └────┬───┘            └──────┬──────┘
         │                       │
         ▼                       ▼
   ┌─────────────┐      ┌────────────────┐
   │ General     │      │ Balanced       │
   │ evaluation? │      │ classes?       │
   └──────┬──────┘      └────────┬───────┘
          │                      │
     ┌────┴────┐           ┌─────┴─────┐
     │         │           │           │
    Yes       No          Yes         No
     │         │           │           │
     ▼         ▼           ▼           ▼
   R²      ┌───────┐   Accuracy   ┌────────┐
          │ Outliers?│            │ What's  │
          └─┬─────┬─┘            │important│
            │     │              └────┬───┬┘
           Yes   No                  │   │
            │     │              ┌───┘   └───┐
            ▼     ▼              │           │
          MAE   RMSE         Minimize    Minimize
                              FP?         FN?
                               │           │
                               ▼           ▼
                          Precision    Recall
                               │           │
                               └─────┬─────┘
                                     │
                                     ▼
                                Both equally?
                                     │
                                     ▼
                                 F1-Score
```

---

## Практичні поради 💡

### 1. Використовуй кілька метрик одночасно

```python
from sklearn.metrics import classification_report

# Не покладайся на одну метрику!
print(classification_report(y_test, y_pred))

# Дивись на:
# - Accuracy (загальна картина)
# - Precision/Recall для кожного класу
# - F1-Score (баланс)
# - Support (скільки зразків)
```

### 2. Завжди дивись на Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

# Візуалізуй де модель помиляється
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# Аналізуй:
# - Які класи плутаються між собою?
# - Чи є асиметричні помилки?
```

### 3. Враховуй business context

```python
# Медична діагностика
# FN (пропустили хвору людину) >> FP (хибна тривога)
# → Максимізуй Recall!

# Spam фільтр
# FP (важливий лист у спам) >> FN (спам у inbox)
# → Максимізуй Precision!

# Fraud detection
# Баланс між Precision і Recall
# → F1-Score або custom threshold
```

### 4. Використовуй ROC-AUC для model selection

```python
from sklearn.model_selection import cross_val_score

# Порівняння моделей
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"{name}: AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 5. Оптимізуй поріг для специфічних потреб

```python
from sklearn.metrics import precision_recall_curve

# Знайти оптимальний поріг
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Максимізувати F1
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold for F1: {optimal_threshold:.3f}")

# Або custom вимоги
# Знайти поріг де Recall >= 0.9
high_recall_idx = np.where(recall >= 0.9)[0]
if len(high_recall_idx) > 0:
    threshold_90_recall = thresholds[high_recall_idx[0]]
    print(f"Threshold for Recall >= 0.9: {threshold_90_recall:.3f}")
```

---

## Реальний приклад: Comprehensive Evaluation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, precision_recall_curve)

print("="*70)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*70)

# Дані
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, stratify=cancer.target, random_state=42
)

print(f"\nDataset: {cancer.data.shape[0]} samples")
print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")
print(f"Class distribution: {np.bincount(cancer.target)}")

# Модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Передбачення
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ============================================
# 1. БАЗОВІ МЕТРИКИ
# ============================================
print("\n" + "="*70)
print("1. BASIC METRICS")
print("="*70)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  TP: {tp:3d}  |  FN: {fn:3d}")
print(f"  FP: {fp:3d}  |  TN: {tn:3d}")

metrics_basic = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Specificity': tn / (tn + fp),
    'F1-Score': f1_score(y_test, y_pred)
}

print("\nScores:")
for metric, value in metrics_basic.items():
    print(f"  {metric:15s}: {value:.4f}")

# ============================================
# 2. ADVANCED METRICS
# ============================================
print("\n" + "="*70)
print("2. ADVANCED METRICS")
print("="*70)

auc = roc_auc_score(y_test, y_proba)
from sklearn.metrics import log_loss, cohen_kappa_score

metrics_advanced = {
    'ROC-AUC': auc,
    'Log Loss': log_loss(y_test, y_proba),
    "Cohen's Kappa": cohen_kappa_score(y_test, y_pred)
}

for metric, value in metrics_advanced.items():
    print(f"  {metric:15s}: {value:.4f}")

# ============================================
# 3. CLASSIFICATION REPORT
# ============================================
print("\n" + "="*70)
print("3. CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, 
                           target_names=['Malignant', 'Benign']))

# ============================================
# 4. CROSS-VALIDATION
# ============================================
print("="*70)
print("4. CROSS-VALIDATION (5-fold)")
print("="*70)

cv_scores = {}
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for metric in scoring_metrics:
    scores = cross_val_score(model, cancer.data, cancer.target, 
                            cv=5, scoring=metric)
    cv_scores[metric] = scores
    print(f"  {metric:10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ============================================
# 5. ВІЗУАЛІЗАЦІЯ
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, 
                       display_labels=['Malignant', 'Benign']).plot(
    ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')

# Subplot 2: ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
axes[0, 1].set_xlabel('False Positive Rate', fontsize=11)
axes[0, 1].set_ylabel('True Positive Rate', fontsize=11)
axes[0, 1].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
axes[1, 0].plot(recall, precision, linewidth=2)
axes[1, 0].set_xlabel('Recall', fontsize=11)
axes[1, 0].set_ylabel('Precision', fontsize=11)
axes[1, 0].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Threshold Analysis
axes[1, 1].plot(thresholds_pr, precision[:-1], label='Precision', linewidth=2)
axes[1, 1].plot(thresholds_pr, recall[:-1], label='Recall', linewidth=2)

# F1 scores
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
axes[1, 1].plot(thresholds_pr, f1_scores, label='F1-Score', linewidth=2, linestyle='--')

# Optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds_pr[optimal_idx]
axes[1, 1].axvline(x=optimal_threshold, color='red', linestyle=':', 
                   linewidth=2, label=f'Optimal ({optimal_threshold:.2f})')

axes[1, 1].set_xlabel('Threshold', fontsize=11)
axes[1, 1].set_ylabel('Score', fontsize=11)
axes[1, 1].set_title('Metrics vs Threshold', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 6. RECOMMENDATIONS
# ============================================
print("\n" + "="*70)
print("6. RECOMMENDATIONS")
print("="*70)

if metrics_basic['Accuracy'] > 0.95:
    print("✅ Excellent performance!")
elif metrics_basic['Accuracy'] > 0.9:
    print("✅ Good performance")
else:
    print("⚠️  Room for improvement")

if abs(metrics_basic['Precision'] - metrics_basic['Recall']) > 0.1:
    print("⚠️  Imbalance between Precision and Recall")
    if metrics_basic['Precision'] > metrics_basic['Recall']:
        print("   → Consider lowering threshold to improve Recall")
    else:
        print("   → Consider raising threshold to improve Precision")

if auc > 0.95:
    print("✅ Excellent discrimination ability (AUC > 0.95)")
elif auc > 0.9:
    print("✅ Good discrimination ability (AUC > 0.9)")
else:
    print("⚠️  Consider feature engineering or different model")

print(f"\n💡 Suggested optimal threshold: {optimal_threshold:.3f}")
print(f"   (maximizes F1-Score = {f1_scores[optimal_idx]:.4f})")

print("="*70)
```

---

## Підсумкова таблиця метрик

### Regression

| Метрика | Формула | Коли використовувати | Одиниці |
|---------|---------|---------------------|---------|
| **MAE** | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Викиди, проста інтерпретація | Ті ж що y |
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | Великі помилки критичні | Квадрат y |
| **RMSE** | $\sqrt{\text{MSE}}$ | MSE + інтерпретація | Ті ж що y |
| **R²** | $1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}$ | Загальна оцінка | 0 to 1 |
| **Adjusted R²** | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ | Багато ознак | 0 to 1 |
| **MAPE** | $\frac{100\%}{n}\sum\|\frac{y-\hat{y}}{y}\|$ | Відсотки, бізнес | % |

### Classification

| Метрика | Формула | Коли використовувати |
|---------|---------|---------------------|
| **Accuracy** | $\frac{TP+TN}{\text{All}}$ | Збалансовані класи |
| **Precision** | $\frac{TP}{TP+FP}$ | Мінімізувати FP |
| **Recall** | $\frac{TP}{TP+FN}$ | Мінімізувати FN |
| **Specificity** | $\frac{TN}{TN+FP}$ | Правильно визначити негативні |
| **F1-Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Баланс Precision/Recall |
| **ROC-AUC** | Area under ROC curve | Порівняння моделей |
| **Log Loss** | $-\frac{1}{n}\sum[y\log(\hat{p})+(1-y)\log(1-\hat{p})]$ | Калібровані ймовірності |
| **Cohen's κ** | $\frac{p_o - p_e}{1 - p_e}$ | Несбалансовані класи |

---

## Пов'язані теми

- [[01_Bias_Variance_Tradeoff]] — як метрики пов'язані з bias/variance
- [[02_Overfitting_Underfitting]] — діагностика через метрики
- [[03_Train_Test_Split]] — де оцінювати метрики
- [[04_Cross_Validation]] — надійна оцінка метрик
- [[Confusion_Matrix]] — основа для classification metrics
- [[Model_Selection]] — вибір моделі на основі метрик

## Ресурси

- [Scikit-learn: Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Precision vs Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [ROC Curves Explained](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

---

## Ключові висновки

> Evaluation Metrics — це кількісні показники якості моделі. Різні задачі потребують різних метрик. Вибір правильної метрики критично важливий для успіху проекту.

**Regression:**
- **R²** — основна метрика (0-1, % поясненої варіації)
- **MAE** — проста, стійка до викидів (ті ж одиниці що y)
- **RMSE** — карає великі помилки (ті ж одиниці що y)

**Classification:**
- **Accuracy** — тільки для збалансованих класів
- **Precision** — мінімізувати FP (не помилково позитивні)
- **Recall** — мінімізувати FN (знайти всі позитивні)
- **F1** — баланс Precision/Recall
- **ROC-AUC** — порівняння моделей, незалежно від порогу

**Золоті правила:**
1. Використовуй **кілька метрик** одночасно
2. Завжди дивись на **Confusion Matrix**
3. Враховуй **business context**
4. Не покладайся тільки на **Accuracy**
5. Візуалізуй результати (ROC, PR curves)

---

#ml #metrics #evaluation #regression-metrics #classification-metrics #accuracy #precision #recall #f1-score #roc-auc
