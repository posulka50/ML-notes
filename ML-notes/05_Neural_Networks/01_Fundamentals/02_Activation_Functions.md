# Activation Functions (Функції активації)

## Що це?

**Activation function** — це функція, яка застосовується до виходу нейрона після зваженої суми. Вона визначає, який сигнал нейрон передає далі.

**Головна ідея:** без activation function нейронна мережа — це просто лінійна регресія, скільки б шарів ти не додав. Нелінійна активація дає мережі здатність вивчати складні паттерни.

## Навіщо потрібні?

- 🔥 **Нелінійність** — дозволяє мережі апроксимувати будь-яку функцію
- 📡 **Контроль сигналу** — обмежує або нормалізує вихід нейрона
- 🎯 **Градієнти** — забезпечує можливість backpropagation
- 🏗️ **Різні задачі** — вихідний шар потребує іншої активації ніж прихований

---

## Чому нелінійність критична?

Якщо кожен нейрон просто рахує `w·x + b`, то два шари разом — це:

```
Шар 1: y₁ = W₁x + b₁
Шар 2: y₂ = W₂y₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
```

Це знову лінійна функція. Хоч 2 шари, хоч 100 — результат той самий. Нелінійна активація ламає цю закономірність.

---

## Огляд функцій

| Функція | Діапазон | Де використовується |
|---------|----------|---------------------|
| **Step** | {0, 1} | Оригінальний perceptron (застаріло) |
| **Sigmoid** | (0, 1) | Вихідний шар, бінарна класифікація |
| **Tanh** | (-1, 1) | Приховані шари RNN (менш популярно) |
| **ReLU** | [0, ∞) | Приховані шари — стандарт de facto |
| **Leaky ReLU** | (-∞, ∞) | Альтернатива ReLU при dying neurons |
| **ELU** | (-1, ∞) | Альтернатива ReLU, плавніша |
| **Softmax** | (0, 1), сума=1 | Вихідний шар, багатокласова класифікація |
| **Linear** | (-∞, ∞) | Вихідний шар, регресія |

---

## 1. Step Function

Це оригінальна активація perceptron. Жорстке 0 або 1.

**Проблема:** похідна = 0 майже скрізь → backpropagation не працює. Тому не використовується в навчанні мереж.

```python
import numpy as np
import matplotlib.pyplot as plt

def step(z):
    return np.where(z >= 0, 1, 0)

z = np.linspace(-5, 5, 300)

plt.figure(figsize=(8, 4))
plt.plot(z, step(z), linewidth=2)
plt.title("Step Function")
plt.xlabel("z"); plt.ylabel("output")
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 2. Sigmoid (Logistic)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Плавно стискає будь-яке число в діапазон (0, 1). Ідеально інтерпретується як ймовірність.

**Переваги:**
- Гладка, диференційовна скрізь
- Вихід як ймовірність

**Проблеми:**
- **Vanishing gradient** — при великих |z| похідна ≈ 0, градієнти "зникають" у глибоких мережах
- **Not zero-centered** — вихід завжди позитивний, що сповільнює навчання
- Повільне обчислення (exponent)

**Де використовувати:** вихідний шар для бінарної класифікації. У прихованих шарах — уникати.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

z = np.linspace(-6, 6, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(z, sigmoid(z), linewidth=2, color='blue')
axes[0].set_title("Sigmoid"); axes[0].set_xlabel("z"); axes[0].set_ylabel("σ(z)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(z, sigmoid_derivative(z), linewidth=2, color='red')
axes[1].set_title("Sigmoid — Похідна"); axes[1].set_xlabel("z"); axes[1].set_ylabel("σ'(z)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Максимальне значення похідної
print(f"Макс. похідна sigmoid: {sigmoid_derivative(0):.4f}")  # 0.25
```

**Зверни увагу:** максимальна похідна sigmoid = 0.25. У 10-шаровій мережі градієнт зменшується як 0.25¹⁰ ≈ 0.000001. Це і є vanishing gradient.

---

## 3. Tanh (Hyperbolic Tangent)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

Схожа на sigmoid, але zero-centered — вихід від -1 до 1.

**Переваги над sigmoid:**
- Zero-centered → градієнти можуть бути і позитивні, і негативні → навчання стабільніше

**Та сама проблема:**
- Vanishing gradient при насиченні (великих |z|)

**Де використовувати:** приховані шари в RNN/LSTM — там tanh ще актуальна. У звичайних мережах замінена ReLU.

```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

z = np.linspace(-4, 4, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(z, tanh(z), linewidth=2, color='green')
axes[0].set_title("Tanh"); axes[0].set_xlabel("z"); axes[0].set_ylabel("tanh(z)")
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].grid(True, alpha=0.3)

axes[1].plot(z, tanh_derivative(z), linewidth=2, color='orange')
axes[1].set_title("Tanh — Похідна"); axes[1].set_xlabel("z"); axes[1].set_ylabel("tanh'(z)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4. ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

Найпростіша і найпопулярніша активація для прихованих шарів. Від'ємне → 0, позитивне → без змін.

**Переваги:**
- Обчислення: просто `max(0, z)` — дуже швидко
- **Немає vanishing gradient** для позитивних значень (похідна = 1)
- Sparse activation — багато нейронів = 0, мережа більш ефективна

**Проблема: Dying ReLU**
Якщо нейрон потрапляє в зону z < 0 і там залишається (ваги оновлюються так, що z завжди від'ємне), нейрон "помирає" — завжди видає 0, похідна = 0, ваги більше не оновлюються.

**Де використовувати:** приховані шари у MLP, CNN — майже завжди ReLU за замовчуванням.

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

z = np.linspace(-4, 4, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(z, relu(z), linewidth=2, color='purple')
axes[0].set_title("ReLU"); axes[0].set_xlabel("z"); axes[0].set_ylabel("ReLU(z)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(z, relu_derivative(z), linewidth=2, color='brown')
axes[1].set_title("ReLU — Похідна"); axes[1].set_xlabel("z"); axes[1].set_ylabel("ReLU'(z)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 5. Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & \text{якщо } z > 0 \\ \alpha z & \text{якщо } z \leq 0 \end{cases}$$

де α — малий коефіцієнт (зазвичай 0.01).

Вирішує проблему dying ReLU: замість нуля для від'ємних значень дає маленький негативний нахил.

**Де використовувати:** якщо маєш проблему dying neurons — спробуй Leaky ReLU замість ReLU.

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

z = np.linspace(-4, 4, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(z, relu(z), linewidth=2, label='ReLU', color='purple')
axes[0].plot(z, leaky_relu(z), linewidth=2, label='Leaky ReLU (α=0.01)', 
             color='magenta', linestyle='--')
axes[0].set_title("ReLU vs Leaky ReLU"); axes[0].set_xlabel("z")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(z, leaky_relu_derivative(z), linewidth=2, color='magenta')
axes[1].set_title("Leaky ReLU — Похідна"); axes[1].set_xlabel("z")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 6. ELU (Exponential Linear Unit)

$$\text{ELU}(z) = \begin{cases} z & \text{якщо } z > 0 \\ \alpha(e^z - 1) & \text{якщо } z \leq 0 \end{cases}$$

Плавна версія Leaky ReLU. Від'ємна частина — плавна крива, а не лінія.

**Переваги над ReLU:**
- Плавніша, що іноді прискорює збіжність
- Zero-centered у середньому

**Недолік:** повільніше обчислення (exponent).

```python
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

z = np.linspace(-4, 4, 300)

plt.figure(figsize=(8, 4))
plt.plot(z, relu(z), linewidth=2, label='ReLU', color='purple')
plt.plot(z, elu(z), linewidth=2, label='ELU (α=1)', color='teal', linestyle='--')
plt.title("ReLU vs ELU")
plt.xlabel("z"); plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

---

## 7. Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

Перетворює вектор чисел у вектор ймовірностей — всі значення між 0 і 1, сума = 1.

**Це не функція для одного нейрона — вона застосовується до цілого шару.**

**Де використовувати:** виключно вихідний шар для багатокласової класифікації.

```python
def softmax(z):
    # Стабільна версія (відніманням максимуму)
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()

# Приклад
z = np.array([2.0, 1.0, 0.1])
probs = softmax(z)

print("Вхід (logits):", z)
print("Вихід (probabilities):", np.round(probs, 4))
print(f"Сума: {probs.sum():.4f}")  # завжди = 1.0

# Клас з найвищою ймовірністю
print(f"Передбачений клас: {np.argmax(probs)}")
```

**Вихід:**
```
Вхід (logits): [2.  1.  0.1]
Вихід (probabilities): [0.6590 0.2424 0.0986]
Сума: 1.0000
Передбачений клас: 0
```

---

## 8. Linear (без активації)

$$f(z) = z$$

Тобто — нічого не робити. Нейрон просто передає зважену суму далі.

**Де використовувати:** вихідний шар для **регресії**, коли потрібно передбачити довільне число.

---

## Яку активацію вибрати?

### Для прихованих шарів

```python
# Стандартний вибір — ReLU
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
])
```

Якщо маєш dying neurons → спробуй `leaky_relu` або `elu`.

### Для вихідного шару

```python
# Бінарна класифікація (0 або 1)
keras.layers.Dense(1, activation='sigmoid')

# Багатокласова класифікація (3+ класи)
keras.layers.Dense(10, activation='softmax')

# Регресія (довільне число)
keras.layers.Dense(1, activation='linear')  # або просто без activation
```

### Таблиця вибору

| Задача | Вихідний шар | Loss function |
|--------|-------------|---------------|
| Бінарна класифікація | Sigmoid (1 нейрон) | Binary Crossentropy |
| Багатокласова класифікація | Softmax (K нейронів) | Categorical Crossentropy |
| Регресія | Linear (1 нейрон) | MSE / MAE |
| Приховані шари | ReLU | — |

---

## Vanishing Gradient — головна проблема

Це настільки важливо, що варто зрозуміти окремо.

**Що відбувається під час backpropagation:**
- Градієнти перемножуються шар за шаром в напрямку від виходу до входу
- Якщо кожна похідна < 1, градієнт зменшується експоненційно
- Ранні шари практично не навчаються

**Наприклад — sigmoid у 10-шаровій мережі:**

```python
# Максимальна похідна sigmoid = 0.25
max_grad_sigmoid = 0.25
layers = 10

gradient_after_backprop = max_grad_sigmoid ** layers
print(f"Градієнт після {layers} шарів: {gradient_after_backprop:.10f}")
# → 0.0000000954 — практично нуль
```

**Чому ReLU вирішує проблему:**

```python
# Похідна ReLU для позитивних значень = 1
# Градієнт не "зникає" при проходженні через ReLU-нейрони
max_grad_relu = 1.0
gradient_relu = max_grad_relu ** layers
print(f"Градієнт ReLU після {layers} шарів: {gradient_relu:.4f}")
# → 1.0 — без деградації
```

---

## Порівняння всіх функцій на одному графіку

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-4, 4, 300)

functions = {
    'Sigmoid': 1 / (1 + np.exp(-z)),
    'Tanh': np.tanh(z),
    'ReLU': np.maximum(0, z),
    'Leaky ReLU': np.where(z > 0, z, 0.01 * z),
    'ELU': np.where(z > 0, z, 1.0 * (np.exp(z) - 1)),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for i, (name, values) in enumerate(functions.items()):
    axes[i].plot(z, values, linewidth=2)
    axes[i].set_title(name, fontsize=13, fontweight='bold')
    axes[i].set_xlabel("z"); axes[i].set_ylabel("f(z)")
    axes[i].axhline(0, color='k', linewidth=0.5)
    axes[i].axvline(0, color='k', linewidth=0.5)
    axes[i].grid(True, alpha=0.3)

# Всі на одному для порівняння
for name, values in functions.items():
    axes[5].plot(z, values, linewidth=2, label=name)
axes[5].set_title("Порівняння", fontsize=13, fontweight='bold')
axes[5].set_xlabel("z"); axes[5].legend(fontsize=9)
axes[5].axhline(0, color='k', linewidth=0.5)
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Код (PyTorch / Keras)

### PyTorch

```python
import torch
import torch.nn as nn

# Вбудовані activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
elu = nn.ELU(alpha=1.0)
softmax = nn.Softmax(dim=1)

# Або функціонально
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(F.relu(x))
print(F.sigmoid(x))
print(F.tanh(x))
```

### Keras / TensorFlow

```python
import tensorflow as tf
from tensorflow import keras

# У шарах
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),  # 10 класів
])

# Або як окремі шари
model = keras.Sequential([
    keras.layers.Dense(128),
    keras.layers.ReLU(),
    keras.layers.Dense(64),
    keras.layers.ReLU(),
    keras.layers.Dense(10),
    keras.layers.Softmax(),
])

# Leaky ReLU
keras.layers.Dense(64),
keras.layers.LeakyReLU(alpha=0.01),
```

---

## Практичні поради 💡

1. **Починай з ReLU** для прихованих шарів — це safe default
2. **Sigmoid тільки на виході** для бінарної класифікації, не в прихованих шарах
3. **Softmax тільки на виході** для багатокласової класифікації
4. **Dying neurons?** — спробуй Leaky ReLU або зменш learning rate
5. **RNN/LSTM** — там tanh і sigmoid вбудовані архітектурно, не змінюй
6. **Нормалізуй входи** — activation functions чутливі до масштабу вхідних даних
7. **Не використовуй sigmoid/tanh у глибоких мережах** без batch normalization

---

## Пов'язані теми

- [[01_Perceptron]] — step function як перша активація
- [[03_Backpropagation]] — як похідна активації впливає на навчання
- [[01_Batch_Normalization]] — допомагає при vanishing/exploding gradients
- [[01_MLP]] — де і як використовуються ці функції в архітектурі

## Ресурси

- [CS231n: Activation Functions](https://cs231n.github.io/neural-networks-1/#actfun)
- [PyTorch: Non-linear activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [Keras: Activation Functions](https://keras.io/api/layers/activation_layers/)

---

## Ключові висновки

> Activation function додає нелінійність, без якої нейронна мережа — просто лінійна регресія. Вибір активації суттєво впливає на швидкість і якість навчання.

**Практичне правило:**
- Приховані шари → **ReLU**
- Вихід, бінарна класифікація → **Sigmoid**
- Вихід, багатокласова класифікація → **Softmax**
- Вихід, регресія → **Linear**

**Головна проблема sigmoid/tanh:** vanishing gradient у глибоких мережах — саме тому ReLU став стандартом.

---

#ml #neural-networks #activation-functions #relu #sigmoid #softmax #deep-learning #fundamentals
