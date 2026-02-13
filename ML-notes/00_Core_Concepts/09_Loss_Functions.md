# Loss Functions (Функції втрат)

## Що це?

**Loss Function** — це функція, яка **вимірює помилку** моделі. Вона показує, **наскільки погано** модель передбачає на даних. Мета навчання — **мінімізувати** loss function.

**Головна ідея:** різні задачі потребують різних loss functions. Вибір правильної loss function критично важливий для успіху моделі.

## Навіщо потрібно?

- 🎯 **Оптимізація моделі** — що мінімізуємо під час навчання
- 📊 **Gradient descent** — обчислення градієнтів
- 🔍 **Model training** — як модель вчиться
- ⚙️ **Custom objectives** — специфічні бізнес-цілі
- 💡 **Regularization** — додавання penalties
- 🎚️ **Fine-tuning** — налаштування навчання

## Коли важливо?

**Завжди!** Loss function — це серце ML алгоритму.

**Ключові питання:**
- Regression чи classification?
- Binary чи multiclass?
- Чи є outliers?
- Чи важливі всі помилки однаково?

---

## Класифікація Loss Functions

```
Loss Functions
│
├── Regression Losses
│   ├── Mean Squared Error (MSE / L2 Loss)
│   ├── Mean Absolute Error (MAE / L1 Loss)
│   ├── Huber Loss
│   ├── Log-Cosh Loss
│   └── Quantile Loss
│
├── Classification Losses
│   ├── Binary Cross-Entropy (Log Loss)
│   ├── Categorical Cross-Entropy
│   ├── Sparse Categorical Cross-Entropy
│   ├── Hinge Loss (SVM)
│   └── Focal Loss
│
└── Advanced Losses
    ├── Contrastive Loss
    ├── Triplet Loss
    └── Custom Business Losses
```

---

# REGRESSION LOSSES

## 1. Mean Squared Error (MSE / L2 Loss)

### Формула

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Інтуїція

**MSE** — **квадратична** помилка. **Сильно карає великі помилки**.

```
Помилки:     [1, 1, 1, 1, 1]  vs  [0, 0, 0, 0, 5]
Квадрати:    [1, 1, 1, 1, 1]  vs  [0, 0, 0, 0, 25]

MSE:         5/5 = 1.0        vs  25/5 = 5.0

Другий варіант має 1 велику помилку → MSE у 5 разів більше!
```

### Код

```python
import numpy as np

def mse_loss(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

# Приклад
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

loss = mse_loss(y_true, y_pred)
print(f"MSE Loss: {loss:.4f}")

# Градієнт MSE
def mse_gradient(y_true, y_pred):
    """Градієнт MSE по predictions"""
    return 2 * (y_pred - y_true) / len(y_true)

gradient = mse_gradient(y_true, y_pred)
print(f"Gradient: {gradient}")
```

### Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Помилки
errors = np.linspace(-5, 5, 100)
mse_values = errors ** 2
mae_values = np.abs(errors)

plt.figure(figsize=(10, 6))
plt.plot(errors, mse_values, linewidth=2, label='MSE (L2)', color='blue')
plt.plot(errors, mae_values, linewidth=2, label='MAE (L1)', color='red')

plt.xlabel('Error (y_true - y_pred)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('MSE vs MAE Loss Functions', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Анотація
plt.annotate('MSE карає великі помилки\nсильніше (квадратично)', 
            xy=(3, 9), xytext=(1, 15),
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))

plt.tight_layout()
plt.show()
```

### Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Диференційовна (smooth) | ❌ Дуже чутлива до outliers |
| ✅ Добре працює з gradient descent | ❌ Не в оригінальних одиницях |
| ✅ Популярна в ML | ❌ Може давати великі градієнти |
| ✅ Probabilistic interpretation (Gaussian) | |

### Коли використовувати?

- ✅ **Default** для regression
- ✅ Коли великі помилки **критичніші** за малі
- ✅ Коли немає значних outliers
- ✅ Linear regression, neural networks
- ❌ Коли є outliers (використовуй MAE або Huber)

---

## 2. Mean Absolute Error (MAE / L1 Loss)

### Формула

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

### Інтуїція

**MAE** — **лінійна** помилка. **Всі помилки рівноцінні**.

```
Помилки:     [1, 1, 1, 1, 1]  vs  [0, 0, 0, 0, 5]
Абсолютні:   [1, 1, 1, 1, 1]  vs  [0, 0, 0, 0, 5]

MAE:         5/5 = 1.0        vs  5/5 = 1.0

Обидва варіанти мають однакову MAE!
```

### Код

```python
def mae_loss(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

# Приклад
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

loss = mae_loss(y_true, y_pred)
print(f"MAE Loss: {loss:.4f}")

# Градієнт MAE (subgradient в 0)
def mae_gradient(y_true, y_pred):
    """Градієнт MAE"""
    return np.sign(y_pred - y_true) / len(y_true)

gradient = mae_gradient(y_true, y_pred)
print(f"Gradient: {gradient}")
```

### Порівняння MSE vs MAE

```python
import numpy as np

# Дані з outlier
y_true = np.array([1, 2, 3, 4, 100])  # 100 — outlier!
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5])

mse = mse_loss(y_true, y_pred)
mae = mae_loss(y_true, y_pred)

print("З outlier:")
print(f"  MSE: {mse:.2f}")  # Дуже велика через outlier!
print(f"  MAE: {mae:.2f}")  # Менш чутлива

# Без outlier
y_true_clean = np.array([1, 2, 3, 4, 5])
y_pred_clean = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

mse_clean = mse_loss(y_true_clean, y_pred_clean)
mae_clean = mae_loss(y_true_clean, y_pred_clean)

print("\nБез outlier:")
print(f"  MSE: {mse_clean:.2f}")
print(f"  MAE: {mae_clean:.2f}")
```

### Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Стійка до outliers | ❌ Не диференційовна в 0 |
| ✅ В оригінальних одиницях | ❌ Повільніша конвергенція |
| ✅ Проста інтерпретація | ❌ Gradient постійний (може бути проблема) |

### Коли використовувати?

- ✅ Коли є **outliers**
- ✅ Коли всі помилки **рівноцінні**
- ✅ Потрібна **інтерпретація** (ті ж одиниці)
- ❌ Коли великі помилки критичніші (використовуй MSE)

---

## 3. Huber Loss

### Формула

$$L_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

### Інтуїція

**Huber Loss** — це **комбінація MSE і MAE**:
- Для малих помилок → MSE (квадратична)
- Для великих помилок → MAE (лінійна)

**Параметр $\delta$** контролює перехід між MSE і MAE.

```
|error| < δ  → MSE (smooth, швидка конвергенція)
|error| ≥ δ  → MAE (robust до outliers)

Найкраще з обох світів!
```

### Код

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss"""
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    
    return np.where(is_small_error, squared_loss, linear_loss).mean()

# Приклад
y_true = np.array([1, 2, 3, 4, 100])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5])

mse = mse_loss(y_true, y_pred)
mae = mae_loss(y_true, y_pred)
huber = huber_loss(y_true, y_pred, delta=1.0)

print(f"MSE:   {mse:.2f}")
print(f"MAE:   {mae:.2f}")
print(f"Huber: {huber:.2f}")
```

### Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Помилки
errors = np.linspace(-5, 5, 1000)
delta = 1.5

# Loss values
mse_vals = errors ** 2
mae_vals = np.abs(errors)

huber_vals = np.where(
    np.abs(errors) <= delta,
    0.5 * errors ** 2,
    delta * (np.abs(errors) - 0.5 * delta)
)

# Візуалізація
plt.figure(figsize=(12, 6))

plt.plot(errors, mse_vals, linewidth=2, label='MSE', alpha=0.7)
plt.plot(errors, mae_vals, linewidth=2, label='MAE', alpha=0.7)
plt.plot(errors, huber_vals, linewidth=3, label=f'Huber (δ={delta})', color='green')

# Vertical lines at ±δ
plt.axvline(x=delta, color='red', linestyle='--', alpha=0.5, label=f'±δ')
plt.axvline(x=-delta, color='red', linestyle='--', alpha=0.5)

plt.xlabel('Error', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Huber Loss: Best of Both Worlds', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 10)
plt.tight_layout()
plt.show()
```

### Вибір δ

```python
# Різні δ
deltas = [0.5, 1.0, 2.0, 5.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, delta in enumerate(deltas):
    errors = np.linspace(-5, 5, 1000)
    
    huber_vals = np.where(
        np.abs(errors) <= delta,
        0.5 * errors ** 2,
        delta * (np.abs(errors) - 0.5 * delta)
    )
    
    axes[idx].plot(errors, huber_vals, linewidth=2, color='green')
    axes[idx].axvline(x=delta, color='red', linestyle='--', alpha=0.5)
    axes[idx].axvline(x=-delta, color='red', linestyle='--', alpha=0.5)
    axes[idx].set_title(f'δ = {delta}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Error')
    axes[idx].set_ylabel('Loss')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim(0, 10)

plt.suptitle('Huber Loss with Different δ', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Коли використовувати?

- ✅ **Outliers** присутні, але не дуже багато
- ✅ Потрібен баланс між MSE і MAE
- ✅ Regression з шумними даними
- ✅ Reinforcement learning

---

## 4. Log-Cosh Loss

### Формула

$$L(y, \hat{y}) = \sum_{i=1}^{n} \log(\cosh(\hat{y}_i - y_i))$$

де $\cosh(x) = \frac{e^x + e^{-x}}{2}$

### Інтуїція

**Log-Cosh** — це **smoother версія MAE**, яка працює як MSE для малих помилок.

### Код

```python
def log_cosh_loss(y_true, y_pred):
    """Log-Cosh Loss"""
    error = y_pred - y_true
    return np.mean(np.log(np.cosh(error)))

# Порівняння
y_true = np.array([1, 2, 3, 4, 100])
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5])

losses = {
    'MSE': mse_loss(y_true, y_pred),
    'MAE': mae_loss(y_true, y_pred),
    'Huber': huber_loss(y_true, y_pred, delta=1.0),
    'Log-Cosh': log_cosh_loss(y_true, y_pred)
}

for name, loss in losses.items():
    print(f"{name:10s}: {loss:.4f}")
```

---

# CLASSIFICATION LOSSES

## 5. Binary Cross-Entropy (Log Loss)

### Формула

$$\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

де:
- $y_i \in \{0, 1\}$ — справжній клас
- $\hat{y}_i \in [0, 1]$ — передбачена ймовірність класу 1

### Інтуїція

**Binary Cross-Entropy** — це **логарифмічна** loss для бінарної класифікації. **Сильно карає впевнені неправильні передбачення**.

```
Справжній клас: y = 1

Передбачення:  ŷ = 0.9  → Loss = -log(0.9) ≈ 0.10   ✓ Мала
Передбачення:  ŷ = 0.5  → Loss = -log(0.5) ≈ 0.69   ⚠️ Середня
Передбачення:  ŷ = 0.1  → Loss = -log(0.1) ≈ 2.30   ❌ Велика
Передбачення:  ŷ = 0.01 → Loss = -log(0.01) ≈ 4.61  ❌❌ Дуже велика!
```

### Код

```python
def binary_cross_entropy(y_true, y_pred):
    """Binary Cross-Entropy Loss"""
    # Clip для уникнення log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Приклад
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.6, 0.2])

loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy: {loss:.4f}")

# Gradient
def bce_gradient(y_true, y_pred):
    """Градієнт BCE"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)

gradient = bce_gradient(y_true, y_pred)
print(f"Gradient: {gradient}")
```

### Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Передбачені ймовірності
y_pred = np.linspace(0.01, 0.99, 100)

# Loss для y_true = 1
loss_y1 = -np.log(y_pred)

# Loss для y_true = 0
loss_y0 = -np.log(1 - y_pred)

# Візуалізація
plt.figure(figsize=(12, 6))

plt.plot(y_pred, loss_y1, linewidth=3, label='y_true = 1', color='blue')
plt.plot(y_pred, loss_y0, linewidth=3, label='y_true = 0', color='red')

plt.xlabel('Predicted Probability ŷ', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Binary Cross-Entropy Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 5)

# Анотації
plt.annotate('Впевнене правильне\nпередбачення\n(низька loss)', 
            xy=(0.95, 0.05), xytext=(0.7, 1.5),
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.annotate('Впевнене неправильне\nпередбачення\n(висока loss)', 
            xy=(0.05, 3), xytext=(0.2, 4),
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.show()
```

### Коли використовувати?

- ✅ **Binary classification** (основна loss)
- ✅ Logistic regression
- ✅ Binary output layer в neural networks
- ✅ Коли потрібні калібровані ймовірності

---

## 6. Categorical Cross-Entropy

### Формула

$$\text{CCE} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{C}y_{ij}\log(\hat{y}_{ij})$$

де:
- $C$ — кількість класів
- $y_{ij}$ — one-hot encoded true label
- $\hat{y}_{ij}$ — передбачена ймовірність класу $j$ для зразка $i$

### Інтуїція

**Categorical Cross-Entropy** — це узагальнення BCE для **multiclass classification**.

```
Приклад: 3 класи

True label: [0, 1, 0]  (клас 1)
Predicted:  [0.1, 0.7, 0.2]

Loss = -(0*log(0.1) + 1*log(0.7) + 0*log(0.2))
     = -log(0.7)
     ≈ 0.357

Тільки ймовірність справжнього класу впливає!
```

### Код

```python
def categorical_cross_entropy(y_true, y_pred):
    """Categorical Cross-Entropy Loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Приклад
y_true = np.array([
    [0, 1, 0],  # Клас 1
    [1, 0, 0],  # Клас 0
    [0, 0, 1]   # Клас 2
])

y_pred = np.array([
    [0.1, 0.7, 0.2],
    [0.8, 0.1, 0.1],
    [0.2, 0.2, 0.6]
])

loss = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross-Entropy: {loss:.4f}")

# Через Keras
import tensorflow as tf

cce = tf.keras.losses.CategoricalCrossentropy()
loss_keras = cce(y_true, y_pred).numpy()
print(f"Keras CCE: {loss_keras:.4f}")
```

### Sparse Categorical Cross-Entropy

```python
# Якщо labels не one-hot encoded
y_true_sparse = np.array([1, 0, 2])  # Class indices

y_pred = np.array([
    [0.1, 0.7, 0.2],
    [0.8, 0.1, 0.1],
    [0.2, 0.2, 0.6]
])

def sparse_categorical_cross_entropy(y_true, y_pred):
    """Sparse CCE (labels as integers)"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Extract predicted probability for true class
    n_samples = len(y_true)
    log_probs = -np.log(y_pred[range(n_samples), y_true])
    
    return np.mean(log_probs)

loss_sparse = sparse_categorical_cross_entropy(y_true_sparse, y_pred)
print(f"Sparse CCE: {loss_sparse:.4f}")
```

---

## 7. Hinge Loss (SVM)

### Формула

$$L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$$

де:
- $y \in \{-1, +1\}$ — справжній клас
- $\hat{y} \in \mathbb{R}$ — raw prediction (не ймовірність!)

### Інтуїція

**Hinge Loss** використовується в **SVM**. Вимагає не просто правильної класифікації, а **margin** (відступ).

```
y = +1 (positive class)

ŷ > +1  → Loss = 0              ✓ Правильно з margin
ŷ = +1  → Loss = 0              ✓ На межі
ŷ = 0   → Loss = max(0, 1-0) = 1   ❌ Неправильно
ŷ = -1  → Loss = max(0, 1-(-1)) = 2 ❌ Дуже неправильно
```

### Код

```python
def hinge_loss(y_true, y_pred):
    """
    Hinge Loss (for SVM)
    
    y_true in {-1, +1}
    y_pred in R (raw scores)
    """
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# Приклад
y_true = np.array([1, -1, 1, -1, 1])
y_pred = np.array([2.0, -1.5, 0.5, -0.3, -0.1])

loss = hinge_loss(y_true, y_pred)
print(f"Hinge Loss: {loss:.4f}")

# Візуалізація
import matplotlib.pyplot as plt

y_pred_range = np.linspace(-3, 3, 100)

# Loss для y = +1
loss_pos = np.maximum(0, 1 - y_pred_range)

# Loss для y = -1
loss_neg = np.maximum(0, 1 + y_pred_range)

plt.figure(figsize=(10, 6))
plt.plot(y_pred_range, loss_pos, linewidth=2, label='y = +1')
plt.plot(y_pred_range, loss_neg, linewidth=2, label='y = -1')
plt.xlabel('Predicted Score ŷ', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Hinge Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

### Squared Hinge Loss

```python
def squared_hinge_loss(y_true, y_pred):
    """Squared Hinge Loss (smoother)"""
    return np.mean(np.maximum(0, 1 - y_true * y_pred) ** 2)
```

---

## 8. Focal Loss

### Формула

$$\text{FL}(p_t) = -\alpha_t(1-p_t)^\gamma\log(p_t)$$

де:
- $p_t$ — predicted probability для true class
- $\gamma$ — focusing parameter (зазвичай 2)
- $\alpha_t$ — balancing parameter

### Інтуїція

**Focal Loss** — це модифікація cross-entropy для **class imbalance**. **Зменшує вагу easy examples** (де модель впевнена).

```
Cross-Entropy:  -log(p_t)
Focal Loss:     -(1-p_t)^γ * log(p_t)

Якщо p_t = 0.9 (впевнене правильне передбачення):
  CE:  -log(0.9) ≈ 0.10
  FL:  -(1-0.9)^2 * log(0.9) = -0.01 * 0.10 ≈ 0.001  ← У 100 разів менше!

Фокус на важких прикладах!
```

### Код

```python
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal Loss для binary classification
    
    Parameters:
    -----------
    gamma : float
        Focusing parameter (зазвичай 2)
    alpha : float
        Balancing parameter
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Cross entropy
    ce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Focal term
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    focal_term = (1 - p_t) ** gamma
    
    # Alpha balancing
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    
    return np.mean(alpha_t * focal_term * ce)

# Приклад: imbalanced data
y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 80% class 0
y_pred = np.array([0.9, 0.1, 0.2, 0.1, 0.15, 0.05, 0.2, 0.1, 0.05, 0.85])

bce = binary_cross_entropy(y_true, y_pred)
fl = focal_loss(y_true, y_pred, gamma=2.0)

print(f"BCE:        {bce:.4f}")
print(f"Focal Loss: {fl:.4f}")
```

### Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Predicted probabilities
p_t = np.linspace(0.01, 0.99, 100)

# Cross-Entropy
ce = -np.log(p_t)

# Focal Loss з різними gamma
gammas = [0, 0.5, 1, 2, 5]

plt.figure(figsize=(12, 6))

plt.plot(p_t, ce, linewidth=3, label='CE (γ=0)', linestyle='--')

for gamma in gammas[1:]:
    fl = -(1 - p_t) ** gamma * np.log(p_t)
    plt.plot(p_t, fl, linewidth=2, label=f'Focal (γ={gamma})')

plt.xlabel('Predicted Probability p_t (for true class)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Focal Loss: Down-weighting Easy Examples', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 5)

# Анотація
plt.annotate('Easy examples\n(high p_t)\nget low weight', 
            xy=(0.9, 0.1), xytext=(0.7, 2),
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.tight_layout()
plt.show()
```

### Коли використовувати?

- ✅ **Class imbalance** (rare objects detection)
- ✅ **Object detection** (RetinaNet)
- ✅ Коли багато easy examples
- ✅ Segmentation з imbalanced classes

---

# ADVANCED LOSSES

## 9. Contrastive Loss

### Формула

$$L = (1-y) \cdot \frac{1}{2}D^2 + y \cdot \frac{1}{2}\max(0, m - D)^2$$

де:
- $D$ — distance між embeddings
- $y \in \{0, 1\}$ — similar (0) чи dissimilar (1)
- $m$ — margin

### Інтуїція

**Contrastive Loss** для **siamese networks**:
- Similar pairs → мінімізувати distance
- Dissimilar pairs → максимізувати distance (до margin)

### Код

```python
def contrastive_loss(embeddings1, embeddings2, labels, margin=1.0):
    """
    Contrastive Loss
    
    Parameters:
    -----------
    embeddings1, embeddings2 : array-like
        Embeddings пари
    labels : array-like
        0 = similar, 1 = dissimilar
    margin : float
        Minimum distance for dissimilar pairs
    """
    # Euclidean distance
    distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
    
    # Similar pairs: minimize distance
    similar_loss = (1 - labels) * distances ** 2
    
    # Dissimilar pairs: maximize distance up to margin
    dissimilar_loss = labels * np.maximum(0, margin - distances) ** 2
    
    return np.mean(0.5 * (similar_loss + dissimilar_loss))

# Приклад
embeddings1 = np.array([[1, 2], [3, 4], [5, 6]])
embeddings2 = np.array([[1.1, 2.1], [8, 9], [5.2, 6.1]])
labels = np.array([0, 1, 0])  # similar, dissimilar, similar

loss = contrastive_loss(embeddings1, embeddings2, labels, margin=2.0)
print(f"Contrastive Loss: {loss:.4f}")
```

---

## 10. Triplet Loss

### Формула

$$L = \max(0, D(a, p) - D(a, n) + \text{margin})$$

де:
- $a$ — anchor
- $p$ — positive (similar to anchor)
- $n$ — negative (dissimilar to anchor)
- $D$ — distance function

### Інтуїція

**Triplet Loss** для **face recognition**, **metric learning**:
- Distance(anchor, positive) має бути **менше** за
- Distance(anchor, negative) + margin

### Код

```python
def triplet_loss(anchor, positive, negative, margin=0.5):
    """Triplet Loss"""
    # Distances
    pos_dist = np.linalg.norm(anchor - positive, axis=1)
    neg_dist = np.linalg.norm(anchor - negative, axis=1)
    
    # Loss
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    
    return np.mean(loss)

# Приклад
anchor = np.array([[1, 2]])
positive = np.array([[1.1, 2.1]])  # Схожий
negative = np.array([[5, 6]])      # Різний

loss = triplet_loss(anchor, positive, negative, margin=0.5)
print(f"Triplet Loss: {loss:.4f}")
```

---

## Порівняльна таблиця

| Loss Function | Задача | Robust до outliers? | Smooth? | Коли використовувати |
|---------------|--------|---------------------|---------|---------------------|
| **MSE** | Regression | ❌ | ✅ | Default regression, no outliers |
| **MAE** | Regression | ✅ | ❌ (в 0) | Outliers present |
| **Huber** | Regression | ✅ | ✅ | Balance MSE/MAE |
| **BCE** | Binary Class | N/A | ✅ | Binary classification |
| **CCE** | Multiclass | N/A | ✅ | Multiclass classification |
| **Hinge** | Binary Class | ✅ | ❌ | SVM, margin important |
| **Focal** | Class (imb) | N/A | ✅ | Class imbalance |
| **Contrastive** | Metric Learning | N/A | ✅ | Siamese networks |
| **Triplet** | Metric Learning | N/A | ✅ | Face recognition |

---

## Практичні поради 💡

### 1. Вибір loss function

```python
# Regression:
# - Default: MSE
# - Outliers: MAE або Huber
# - Specific quantiles: Quantile Loss

# Binary Classification:
# - Default: Binary Cross-Entropy
# - Class imbalance: Focal Loss
# - Margin important: Hinge Loss

# Multiclass:
# - Default: Categorical Cross-Entropy
# - Imbalanced: Weighted CCE або Focal
```

### 2. Custom Loss Functions

```python
import tensorflow as tf

# Custom loss у Keras
def custom_mse_with_penalty(y_true, y_pred):
    """MSE з додатковим penalty для великих передбачень"""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    penalty = tf.reduce_mean(tf.square(y_pred))  # Penalize large predictions
    return mse + 0.1 * penalty

# Використання
model.compile(optimizer='adam', loss=custom_mse_with_penalty)
```

### 3. Loss Weighting для Imbalanced Classes

```python
from sklearn.utils.class_weight import compute_class_weight

# Обчислити ваги
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train),
                                     y=y_train)

# У Keras
model.fit(X_train, y_train, class_weight=dict(enumerate(class_weights)))
```

### 4. Combine Multiple Losses

```python
def combined_loss(y_true, y_pred, alpha=0.5):
    """Комбінація MSE і MAE"""
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return alpha * mse + (1 - alpha) * mae
```

---

## Реальний приклад: Порівняння Loss Functions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("="*70)
print("COMPARING LOSS FUNCTIONS FOR REGRESSION")
print("="*70)

# Генерація даних з outliers
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Додати outliers
outlier_indices = [10, 25, 50, 75, 90]
y[outlier_indices] += np.random.choice([-100, 100], size=len(outlier_indices))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Models (різні loss functions)
models = {
    'Linear (MSE)': LinearRegression(),
    'Ridge (MSE + L2)': Ridge(alpha=1.0),
    'Lasso (MSE + L1)': Lasso(alpha=1.0)
}

# Train and evaluate
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MSE': mse,
        'MAE': mae
    })
    
    print(f"\n{name}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")

# Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    # Plot
    axes[idx].scatter(X_train, y_train, alpha=0.6, s=50, label='Train')
    axes[idx].scatter(X_test, y_test, alpha=0.6, s=50, label='Test', color='orange')
    
    # Outliers
    axes[idx].scatter(X[outlier_indices], y[outlier_indices], 
                     s=200, marker='x', color='red', linewidths=3, 
                     label='Outliers', zorder=5)
    
    axes[idx].plot(X_plot, y_plot, 'g-', linewidth=2, label='Model')
    
    axes[idx].set_title(f'{name}\nMSE: {results[idx]["MSE"]:.1f} | MAE: {results[idx]["MAE"]:.1f}',
                       fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("Висновок:")
print("MSE (Linear) сильно страждає від outliers")
print("L1 regularization (Lasso) більш robust")
print("="*70)
```

---

## Ключові висновки

> Loss Function — це серце ML алгоритму. Вона визначає, що модель оптимізує під час навчання.

**Regression:**
- **MSE** — default, карає великі помилки
- **MAE** — robust до outliers
- **Huber** — баланс MSE/MAE

**Classification:**
- **Binary CE** — binary classification
- **Categorical CE** — multiclass
- **Focal Loss** — class imbalance

**Вибір:**
```
Outliers?          → MAE або Huber
Class imbalance?   → Focal Loss або weighted CE
Margin important?  → Hinge Loss
Metric learning?   → Triplet/Contrastive
```

**Важливо:**
- Різні losses → різна поведінка моделі
- MSE чутлива до outliers
- Cross-Entropy для classification
- Можна комбінувати losses

---

#ml #loss-functions #mse #mae #cross-entropy #huber-loss #focal-loss #optimization
