# Gradient Descent (Градієнтний спуск)

## Що це?

**Gradient Descent** — це **ітеративний алгоритм оптимізації**, який використовується для знаходження мінімуму функції шляхом руху в напрямку найшвидшого спадання (протилежному до градієнта).

**Головна ідея:** починаємо з випадкової точки та крок за кроком рухаємося "вниз по схилу" до мінімуму функції втрат.

## Навіщо потрібен?

- 🎯 **Навчання ML-моделей** — мінімізація функції втрат
- 📊 **Великі дані** — швидше за аналітичні рішення (Normal Equation)
- 🧠 **Neural Networks** — єдиний практичний спосіб навчання
- ⚡ **Масштабованість** — працює з мільйонами параметрів
- 🔧 **Універсальність** — працює для будь-якої диференційовної функції

## Коли використовувати?

**Потрібно:**
- **Великі датасети** — $n > 100,000$ спостережень
- **Багато параметрів** — $p > 10,000$ ознак
- Немає аналітичного рішення (Neural Networks, Logistic Regression)
- **Online learning** — модель оновлюється на нових даних
- Обмеження по пам'яті — не можна завантажити всі дані

**Не потрібно:**
- **Малі дані** — Normal Equation швидше
- Мало параметрів — аналітичне рішення простіше
- Потрібна **гарантія глобального мінімуму** (опуклі функції)

---

## Візуальна інтуїція

### 1D випадок

```
Loss J(θ)
    |
    |    ╱╲
    |   ╱  ╲
    |  ╱    ╲
    | ╱  •   ╲
    |╱    ↓   ╲
    |      •   ╲
    |       ↓   ╲
    |        •→  ╲
    |         ★   ╲
    |______________╲_____ θ
                   min
                   
• — поточна позиція
↓/→ — напрямок руху (протилежний градієнту)
★ — мінімум
```

### 2D випадок (поверхня)

```
Вид зверху (контурні лінії):

        Високі втрати
             ╱──╲
            ╱    ╲
           ╱  •   ╲
          │    ↘   │
          │     •  │
          │      ↘ │
          │       •│
           ╲      ★╱  ← Мінімум
            ╲____╱
            
• — кроки градієнтного спуску
★ — оптимальні параметри
```

**Градієнт вказує напрямок найшвидшого ЗРОСТАННЯ, тому ми йдемо в ПРОТИЛЕЖНОМУ напрямку.**

---

## Математика

### Функція втрат

Для лінійної регресії:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

де:
- $J(\theta)$ — функція втрат (MSE)
- $\theta = [\theta_0, \theta_1, ..., \theta_n]$ — вектор параметрів
- $h_\theta(x) = \theta^T x$ — гіпотеза (передбачення)
- $m$ — кількість прикладів

### Градієнт

**Градієнт** — це вектор часткових похідних:

$$\nabla J(\theta) = \begin{bmatrix}
\frac{\partial J}{\partial \theta_0} \\
\frac{\partial J}{\partial \theta_1} \\
\vdots \\
\frac{\partial J}{\partial \theta_n}
\end{bmatrix}$$

Для лінійної регресії:
$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

### Правило оновлення

**Одночасно** для всіх $j$:
$$\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

де $\alpha$ (alpha) — **learning rate** (швидкість навчання).

### Векторна форма

$$\theta := \theta - \alpha \nabla J(\theta)$$

**Покроковий алгоритм:**
1. Ініціалізувати $\theta$ випадковими значеннями
2. Обчислити градієнт $\nabla J(\theta)$
3. Оновити параметри: $\theta := \theta - \alpha \nabla J(\theta)$
4. Повторювати кроки 2-3 до збіжності

---

## Типи Gradient Descent

## 1. Batch Gradient Descent (Пакетний)

### Як працює?

Використовує **всі** тренувальні дані на кожній ітерації.

$$\theta := \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

### Код

```python
def batch_gradient_descent(X, y, theta, alpha, iterations):
    """
    Batch Gradient Descent
    
    X: матриця ознак (m x n)
    y: вектор цільових значень (m,)
    theta: початкові параметри (n,)
    alpha: learning rate
    iterations: кількість ітерацій
    """
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        # Передбачення
        predictions = X.dot(theta)
        
        # Помилки
        errors = predictions - y
        
        # Градієнт (використовує ВСІ дані)
        gradient = (1/m) * X.T.dot(errors)
        
        # Оновлення параметрів
        theta = theta - alpha * gradient
        
        # Збереження функції втрат
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return theta, cost_history
```

### Переваги ✓

- ✅ **Стабільна збіжність** — плавний спуск до мінімуму
- ✅ **Точний градієнт** — використовує всі дані
- ✅ **Теоретичні гарантії** — збігається для опуклих функцій

### Недоліки ✗

- ❌ **Повільний** — обчислює градієнт на всіх даних
- ❌ **Багато пам'яті** — потрібно завантажити весь датасет
- ❌ **Не працює для online learning**

### Коли використовувати?

- Малі/середні датасети ($m < 10,000$)
- Достатньо пам'яті
- Потрібна стабільна збіжність

---

## 2. Stochastic Gradient Descent (SGD, Стохастичний)

### Як працює?

Використовує **один випадковий** приклад на кожній ітерації.

$$\theta := \theta - \alpha (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

### Код

```python
def stochastic_gradient_descent(X, y, theta, alpha, epochs):
    """
    Stochastic Gradient Descent
    
    epochs: кількість проходів через весь датасет
    """
    m = len(y)
    cost_history = []
    
    for epoch in range(epochs):
        # Перемішати дані
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(m):
            # Один приклад
            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]
            
            # Передбачення
            prediction = xi.dot(theta)
            
            # Помилка
            error = prediction - yi
            
            # Градієнт (тільки для ОДНОГО прикладу)
            gradient = xi.T.dot(error)
            
            # Оновлення
            theta = theta - alpha * gradient.flatten()
        
        # Cost на всьому датасеті (для моніторингу)
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}")
    
    return theta, cost_history
```

### Переваги ✓

- ✅ **Дуже швидкий** — оновлення після кожного прикладу
- ✅ **Мало пам'яті** — по одному прикладу
- ✅ **Online learning** — може оновлюватись на нових даних
- ✅ **Може втікати з локальних мінімумів** — через шум

### Недоліки ✗

- ❌ **Нестабільна збіжність** — "шум" в оновленнях
- ❌ **Не збігається точно** — коливається навколо мінімуму
- ❌ **Потрібен learning rate decay** — зменшувати α

### Коли використовувати?

- Дуже великі датасети ($m > 1,000,000$)
- Online learning
- Обмеження по пам'яті
- Швидкі наближені результати

---

## 3. Mini-Batch Gradient Descent (Міні-пакетний)

### Як працює?

Використовує **невелику підмножину** даних (batch) на кожній ітерації.

$$\theta := \theta - \alpha \frac{1}{b} \sum_{i \in \text{batch}} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

де $b$ — batch size (зазвичай 32, 64, 128, 256).

### Код

```python
def mini_batch_gradient_descent(X, y, theta, alpha, epochs, batch_size=32):
    """
    Mini-Batch Gradient Descent
    """
    m = len(y)
    cost_history = []
    
    for epoch in range(epochs):
        # Перемішати дані
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Ітерація по міні-батчам
        for i in range(0, m, batch_size):
            # Витягти batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            batch_m = len(y_batch)
            
            # Передбачення
            predictions = X_batch.dot(theta)
            
            # Помилки
            errors = predictions - y_batch
            
            # Градієнт (на batch)
            gradient = (1/batch_m) * X_batch.T.dot(errors)
            
            # Оновлення
            theta = theta - alpha * gradient
        
        # Cost на всьому датасеті
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}")
    
    return theta, cost_history
```

### Переваги ✓

- ✅ **Баланс швидкості та стабільності**
- ✅ **Ефективне використання GPU** — векторизовані операції
- ✅ **Менше шуму** ніж SGD
- ✅ **Швидше** ніж Batch GD
- ✅ **Найпопулярніший** на практиці

### Недоліки ✗

- ❌ Потрібен вибір batch_size
- ❌ Все ще має деякий шум

### Коли використовувати?

- **Майже завжди!** (стандарт для Deep Learning)
- Великі датасети
- Використання GPU/TPU
- Баланс між швидкістю та точністю

---

## Порівняння типів

| Характеристика | Batch GD | Stochastic GD | Mini-Batch GD |
|----------------|----------|---------------|---------------|
| **Прикладів за ітерацію** | Всі ($m$) | 1 | $b$ (32-256) |
| **Швидкість ітерації** | Повільна | Дуже швидка | Середня |
| **Збіжність** | Стабільна | Нестабільна | Помірна |
| **Пам'ять** | Багато | Мало | Середньо |
| **GPU прискорення** | Так | Ні | **Так ✓** |
| **Online learning** | ❌ | ✅ | ✅ |
| **Використання** | Рідко | Іноді | **Найчастіше** |

### Візуалізація шляху до мінімуму

```
Batch GD:                SGD:                Mini-Batch GD:
Плавний шлях            Шумний шлях         Помірний шум

     •                      •                    •
      ↘                    ↙↘                   ↘
       •                  ↗  ↘                   ↘•
        ↘                ↙    ↘                   ↘
         •              ↗      ↘                   •
          ↘            ↙        ↘                   ↘
           ★          ↗          ★                   ★
```

---

## Learning Rate (α)

### Що це?

**Learning rate** контролює **розмір кроку** при оновленні параметрів.

$$\theta := \theta - \alpha \nabla J(\theta)$$

### Ефект різних α

```
α занадто мале (0.001):     α оптимальне (0.1):      α занадто велике (10):

Cost                        Cost                      Cost
 |  •                        |  •                      |  • ↘ ↗
 | •                         | •                       |      • ↘ ↗
 |•                          |•                        |          •
 |•                          | ↘                       |  Не збігається!
 | ↘                         |  ★                      |  
 |  •                        |                         |
 |_____ iterations           |_____ iterations         |_____ iterations

Дуже повільно              Швидко та ефективно       Розбіжність
```

### Як вибрати α?

#### 1. Grid Search

```python
learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

for alpha in learning_rates:
    theta, cost_history = gradient_descent(X, y, theta_init, alpha, 1000)
    
    print(f"α = {alpha}: Final cost = {cost_history[-1]:.4f}")
    
    plt.plot(cost_history, label=f'α={alpha}')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend()
plt.show()
```

#### 2. Learning Rate Decay (зменшення α)

**Стратегії:**

**Step Decay:**
$$\alpha_t = \alpha_0 \times \gamma^{\lfloor t / k \rfloor}$$

де $\gamma = 0.1-0.5$, $k$ — кількість епох до зменшення.

```python
def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
    return initial_lr * (drop_rate ** (epoch // epochs_drop))
```

**Exponential Decay:**
$$\alpha_t = \alpha_0 \times e^{-kt}$$

```python
def exponential_decay(initial_lr, epoch, k=0.1):
    return initial_lr * np.exp(-k * epoch)
```

**1/t Decay:**
$$\alpha_t = \frac{\alpha_0}{1 + kt}$$

```python
def inverse_time_decay(initial_lr, epoch, k=0.01):
    return initial_lr / (1 + k * epoch)
```

**Cosine Annealing:**
$$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})(1 + \cos(\frac{t\pi}{T}))$$

---

## Критерії зупинки

### Коли зупинятися?

1. **Фіксована кількість ітерацій**
   ```python
   for i in range(max_iterations):
       ...
   ```

2. **Зміна cost мала**
   ```python
   if abs(cost_history[-1] - cost_history[-2]) < epsilon:
       break
   ```

3. **Gradient малий**
   ```python
   if np.linalg.norm(gradient) < epsilon:
       break
   ```

4. **Validation loss зростає** (early stopping)
   ```python
   if val_cost > best_val_cost:
       patience_counter += 1
       if patience_counter > patience:
           break
   ```

---

## Практичний приклад

### Linear Regression з різними методами GD

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Генерація даних
X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)

# Додаємо bias term (стовпець одиниць)
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X_bias, y, test_size=0.2, random_state=42
)

# Нормалізація (ВАЖЛИВО для GD!)
scaler = StandardScaler()
X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
X_test[:, 1:] = scaler.transform(X_test[:, 1:])

# Ініціалізація параметрів
theta_init = np.random.randn(X_train.shape[1])

# 2. Batch GD
print("=== Batch Gradient Descent ===")
theta_batch, cost_batch = batch_gradient_descent(
    X_train, y_train, theta_init.copy(), alpha=0.01, iterations=1000
)

# 3. Stochastic GD
print("\n=== Stochastic Gradient Descent ===")
theta_sgd, cost_sgd = stochastic_gradient_descent(
    X_train, y_train, theta_init.copy(), alpha=0.01, epochs=100
)

# 4. Mini-Batch GD
print("\n=== Mini-Batch Gradient Descent ===")
theta_minibatch, cost_minibatch = mini_batch_gradient_descent(
    X_train, y_train, theta_init.copy(), alpha=0.01, epochs=100, batch_size=32
)

# 5. Порівняння результатів
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Cost history
axes[0].plot(cost_batch, label='Batch GD', linewidth=2)
axes[0].plot(cost_sgd, label='Stochastic GD', linewidth=2, alpha=0.7)
axes[0].plot(cost_minibatch, label='Mini-Batch GD', linewidth=2)
axes[0].set_xlabel('Iterations/Epochs', fontsize=12)
axes[0].set_ylabel('Cost', fontsize=12)
axes[0].set_title('Cost History Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Predictions
X_plot = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 100).reshape(-1, 1)
X_plot_bias = np.c_[np.ones((X_plot.shape[0], 1)), X_plot]

y_pred_batch = X_plot_bias.dot(theta_batch)
y_pred_sgd = X_plot_bias.dot(theta_sgd)
y_pred_minibatch = X_plot_bias.dot(theta_minibatch)

axes[1].scatter(X_test[:, 1], y_test, alpha=0.5, s=30, label='Test Data')
axes[1].plot(X_plot, y_pred_batch, label='Batch GD', linewidth=2)
axes[1].plot(X_plot, y_pred_sgd, label='Stochastic GD', linewidth=2, alpha=0.7)
axes[1].plot(X_plot, y_pred_minibatch, label='Mini-Batch GD', linewidth=2)
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title('Predictions Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Фінальний cost
print("\n=== Final Costs ===")
print(f"Batch GD: {cost_batch[-1]:.4f}")
print(f"Stochastic GD: {cost_sgd[-1]:.4f}")
print(f"Mini-Batch GD: {cost_minibatch[-1]:.4f}")
```

---

## Оптимізації Gradient Descent

### 1. Momentum

**Ідея:** Додаємо "інерцію" — враховуємо попередній напрямок руху.

$$v_t = \beta v_{t-1} + \alpha \nabla J(\theta)$$
$$\theta := \theta - v_t$$

де $\beta \in [0, 1]$ (зазвичай 0.9).

```python
def gradient_descent_momentum(X, y, theta, alpha, iterations, beta=0.9):
    m = len(y)
    v = np.zeros_like(theta)  # Velocity
    cost_history = []
    
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        
        # Momentum update
        v = beta * v + alpha * gradient
        theta = theta - v
        
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
    
    return theta, cost_history
```

**Переваги:**
- ✅ Швидша збіжність
- ✅ Менше коливань
- ✅ Може подолати плато

### 2. Nesterov Accelerated Gradient (NAG)

**Ідея:** "Дивимось вперед" перед обчисленням градієнта.

$$v_t = \beta v_{t-1} + \alpha \nabla J(\theta - \beta v_{t-1})$$
$$\theta := \theta - v_t$$

### 3. AdaGrad

**Ідея:** Адаптивний learning rate для кожного параметра.

$$G_t = G_{t-1} + (\nabla J(\theta))^2$$
$$\theta := \theta - \frac{\alpha}{\sqrt{G_t + \epsilon}} \nabla J(\theta)$$

### 4. RMSProp

**Ідея:** Експоненціально згладжена версія AdaGrad.

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) (\nabla J(\theta))^2$$
$$\theta := \theta - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \nabla J(\theta)$$

### 5. Adam (Adaptive Moment Estimation)

**Найпопулярніший оптимізатор для Deep Learning!**

Комбінує Momentum + RMSProp:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla J(\theta)$$ (перший момент)
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla J(\theta))^2$$ (другий момент)

Bias correction:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update:
$$\theta := \theta - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Гіперпараметри за замовчуванням:**
- $\alpha = 0.001$
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

```python
def adam_optimizer(X, y, theta, alpha=0.001, iterations=1000, 
                   beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = len(y)
    mt = np.zeros_like(theta)  # First moment
    vt = np.zeros_like(theta)  # Second moment
    cost_history = []
    
    for t in range(1, iterations + 1):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        
        # Update biased first moment
        mt = beta1 * mt + (1 - beta1) * gradient
        
        # Update biased second moment
        vt = beta2 * vt + (1 - beta2) * (gradient ** 2)
        
        # Bias correction
        mt_hat = mt / (1 - beta1 ** t)
        vt_hat = vt / (1 - beta2 ** t)
        
        # Update parameters
        theta = theta - alpha * mt_hat / (np.sqrt(vt_hat) + epsilon)
        
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
    
    return theta, cost_history
```

---

## Scikit-learn реалізація

### SGDRegressor (Linear Regression)

```python
from sklearn.linear_model import SGDRegressor

# SGD для регресії
sgd_reg = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    penalty='l2',        # Ridge regularization
    alpha=0.0001,        # Regularization strength
    learning_rate='invscaling',  # Learning rate strategy
    eta0=0.01,           # Initial learning rate
    random_state=42
)

sgd_reg.fit(X_train, y_train)
y_pred = sgd_reg.predict(X_test)

print(f"Coefficients: {sgd_reg.coef_}")
print(f"Intercept: {sgd_reg.intercept_}")
```

### SGDClassifier (Logistic Regression)

```python
from sklearn.linear_model import SGDClassifier

# SGD для класифікації
sgd_clf = SGDClassifier(
    loss='log',          # Logistic loss
    max_iter=1000,
    tol=1e-3,
    learning_rate='adaptive',
    eta0=0.01,
    random_state=42
)

sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Масштабованість** | Працює з мільйонами прикладів |
| **Ефективність пам'яті** | Не потрібно завантажувати всі дані |
| **Online learning** | Оновлення на нових даних |
| **Універсальність** | Працює для будь-якої диференційовної функції |
| **Parallelization** | Mini-batch добре паралелиться |
| **GPU acceleration** | Швидке обчислення на GPU |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Вибір гіперпараметрів** | α, batch_size потрібно налаштовувати |
| **Локальні мінімуми** | Може застрягти (для неопуклих функцій) |
| **Потребує нормалізації** | Чутливий до масштабу ознак |
| **Стохастичність** | SGD нестабільний |
| **Повільніше за аналітичні методи** | Для малих даних |

---

## Практичні поради 💡

1. **ЗАВЖДИ нормалізуй дані** — StandardScaler перед GD
2. **Почни з Adam** — найкращий універсальний оптимізатор
3. **Візуалізуй cost history** — перевір збіжність
4. **Grid search для α** — знайди оптимальний learning rate
5. **Mini-batch = 32-256** — стандартний вибір
6. **Використовуй momentum** — швидша збіжність
7. **Learning rate decay** — для фінального tuning
8. **Early stopping** — зупиняйся при збільшенні val loss
9. **Checkpointing** — зберігай кращі параметри
10. **Перемішуй дані** — shuffle на кожній епосі

---

## Коли використовувати Gradient Descent

### Ідеально підходить ✓

- **Великі датасети** — n > 100,000
- **Neural Networks** — єдиний практичний метод
- **Online learning** — постійне оновлення
- **Багато параметрів** — p > 10,000
- **GPU доступний** — прискорення обчислень

### Краще використати інше ✗

- **Малі дані** — Normal Equation швидше та точніше
- **Опуклі функції + мало параметрів** — аналітичне рішення
- **Потрібна гарантія глобального мінімуму** — спеціалізовані методи

---

## Пов'язані теми

- [[01_Linear_Regression]] — Normal Equation vs Gradient Descent
- [[02_Logistic_Regression]] — навчання через GD
- [[Neural_Networks]] — основа backpropagation
- [[Optimization_Algorithms]] — Adam, RMSProp, etc.
- [[01_Feature_Scaling]] — критично для GD

## Ресурси

- [Andrew Ng: Gradient Descent](https://www.coursera.org/learn/machine-learning)
- [CS231n: Optimization](http://cs231n.stanford.edu/)
- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
- [Scikit-learn: SGD](https://scikit-learn.org/stable/modules/sgd.html)

---

## Ключові висновки

> Gradient Descent — ітеративний алгоритм оптимізації, який рухається в напрямку протилежному до градієнта для мінімізації функції втрат.

**Основні типи:**
- **Batch GD:** використовує всі дані — стабільний, повільний
- **Stochastic GD:** один приклад — швидкий, нестабільний
- **Mini-Batch GD:** компроміс — **найпопулярніший** ✓

**Правило оновлення:**
$$\theta := \theta - \alpha \nabla J(\theta)$$

**Оптимізатори:**
- **SGD:** базовий
- **Momentum:** швидша збіжність
- **Adam:** **найкращий за замовчуванням** ✓

**Ключові принципи:**
- Завжди нормалізуй дані
- Підбирай learning rate через grid search
- Використовуй mini-batch для балансу
- Adam — найбезпечніший вибір

**Коли використовувати:**
- Великі дані + Neural Networks + GPU = Gradient Descent ✓

---

#ml #optimization #gradient-descent #sgd #adam #deep-learning #supervised-learning
