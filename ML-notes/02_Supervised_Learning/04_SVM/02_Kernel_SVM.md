# Kernel SVM (Kernel Trick)

## Що це?

**Kernel SVM** використовує **kernel trick** для обробки **нелінійно розділимих даних**, переносячи їх у вищий вимір, де вони стають лінійно розділимими, **без явного обчислення** цієї трансформації.

**Головна ідея:** Замість явної трансформації в високий вимір, використовуємо kernel function для обчислення dot products у transformed space.

## Навіщо потрібен?

- 🔧 **Нелінійні boundaries** — обробка складних patterns
- ⚡ **Ефективність** — без явної трансформації
- 🎯 **Універсальність** — RBF може апроксимувати будь-яку функцію
- 📊 **Висока точність** — один з найточніших алгоритмів
- 💡 **Flexibility** — різні kernels для різних задач

## Коли використовувати?

**Потрібно:**

- **Нелінійно розділимі** дані
- Складні **нелінійні patterns**
- Середні датасети (n = 1k-100k)
- Потрібна **висока точність**

**Не потрібно:**

- **Дуже великі датасети** (n > 100k) → повільно
- Дані **лінійно розділимі** → Linear SVM
- Потрібна **інтерпретованість** → Tree-based
- **Швидкість** критична → Linear models

---

## Проблема лінійної розділимості

### Приклад: XOR Problem

```
Original 2D space:

    y
    |  × • ×
    |× • • • ×
    | × • • ×
    |  × • ×
    |________ x

Неможливо розділити прямою лінією! ✗
```

**Linear SVM не спрацює!**

---

## Рішення: Kernel Trick

### Ідея трансформації

**Переносимо дані в вищий вимір, де вони стають лінійно розділимі:**

```
After transformation to 3D:
         z
         |
         |    • • •
         |  •  •  •
    _____|_________
         |
    × × ×|× × ×
         |
         
Тепер можна розділити площиною! ✓
```

### Проблема явної трансформації

**Приклад:** Polynomial transformation degree 2 для 2D → 3D:

$$\phi(x) = [x_1, x_2] \rightarrow [x_1^2, \sqrt{2}x_1x_2, x_2^2]$$

**Проблеми:**

- Потрібно **явно обчислювати** $\phi(x)$ для кожної точки
- **Високий вимір** → багато пам'яті
- **Повільно** для великих трансформацій

### Kernel Trick: Магія!

**Замість обчислення $\phi(x)$, використовуємо kernel function:**

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

**Kernel обчислює dot product у transformed space БЕЗ явного обчислення $\phi(x)$!**

---

## Математика Kernel Trick

### Dual Formulation

**Передбачення Linear SVM через dual:**

$$f(x) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i y_i (x_i^T x) + b\right)$$

**З kernel trick:**

$$f(x) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b\right)$$

**Замінили dot product $x_i^T x$ на kernel $K(x_i, x)$!**

### Kernel Function

**Kernel function** обчислює similarity між двома точками:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

де $\phi$ — трансформація в вищий вимір.

**Приклад (Polynomial kernel d=2):**

$$K(x, z) = (x^T z)^2$$

Це еквівалентно:

$$\phi(x)^T \phi(z)$$

де $\phi([x_1, x_2]) = [x_1^2, \sqrt{2}x_1x_2, x_2^2]$

**Але ми НЕ обчислюємо $\phi$ явно!**

---

## Популярні Kernels

### 1. Linear Kernel

$$K(x_i, x_j) = x_i^T x_j$$

**Коли:**

- ✅ Лінійно розділимі дані
- ✅ High-dimensional (text)
- ✅ Швидкість важлива

**Еквівалентно Linear SVM.**

---

### 2. Polynomial Kernel

$$K(x_i, x_j) = (x_i^T x_j + c)^d$$

де:

- $d$ — degree (зазвичай 2-4)
- $c$ — coef0 (константа, зазвичай 0 або 1)

#### Приклад: degree=2

Для 2D: $x = [x_1, x_2]$

$$K(x, z) = (x^T z + 1)^2$$

Трансформація у 6D:

$$\phi(x) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, x_2^2, \sqrt{2}x_1x_2]$$

#### Коли використовувати

- ✅ Поліноміальні залежності
- ✅ Image processing
- ⚠️ Обмежений degree (d ≤ 4)
- ❌ Numerical instability при великому d

#### Код

```python
from sklearn.svm import SVC

svm_poly = SVC(
    kernel='poly',
    degree=3,           # Polynomial degree
    C=1.0,
    coef0=1,           # Константа
    gamma='scale',
    random_state=42
)

svm_poly.fit(X_train, y_train)
print(f"Polynomial SVM Accuracy: {svm_poly.score(X_test, y_test):.4f}")
```

---

### 3. RBF Kernel (Gaussian)

$$K(x_i, x_j) = \exp\left(-\gamma ||x_i - x_j||^2\right)$$

де $\gamma = \frac{1}{2\sigma^2}$ (параметр ширини).

**Найпопулярніший kernel! 🌟**

#### Властивості

- Відображає в **безкінечновимірний** простір
- Схожість зменшується **експоненційно** з відстанню
- **Універсальний** kernel (може апроксимувати будь-яку функцію)
- **Локальний** kernel — точки далеко майже не впливають

#### Інтуїція

RBF kernel вимірює **схожість** між точками:

$$K(x_i, x_j) = \begin{cases} 1 & \text{якщо } x_i = x_j \ \approx 0 & \text{якщо } x_i \text{ далеко від } x_j \end{cases}$$

#### Параметр γ (gamma)

```
γ малий (0.01):              γ оптимальний (1.0):        γ великий (10):
════════════                 ═══════                     ═══
Широкий Gaussian             Баланс                      Вузький Gaussian
Гладка boundary              ✓ Найкраще                  Дуже локальна
High bias                                                High variance
Underfitting                                             Overfitting

  ×××××× |••••••              ×××× |••••                 ×××|•••
  ×××××× |••••••              ×××× |••••                 ×××|•••
════════════                 ═══════                     ═══
```

**Trade-off γ:**

- **Малий γ:** точки далеко впливають → smooth boundary
- **Великий γ:** тільки близькі точки → wiggly boundary

#### Код

```python
from sklearn.svm import SVC

svm_rbf = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',      # або 'auto', або конкретне значення
    random_state=42
)

svm_rbf.fit(X_train, y_train)
print(f"RBF SVM Accuracy: {svm_rbf.score(X_test, y_test):.4f}")
```

#### Значення gamma

**За замовчуванням:**

- `gamma='scale'` (рекомендовано): $\gamma = \frac{1}{n_{features} \cdot \text{Var}(X)}$
- `gamma='auto'`: $\gamma = \frac{1}{n_{features}}$

**Кастомні значення:**

- Типово: [0.001, 0.01, 0.1, 1, 10]

---

### 4. Sigmoid Kernel (Tanh)

$$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + c)$$

**Коли:**

- ⚠️ Рідко використовується
- Схожий на neural networks activation
- Може бути не positive semi-definite

**Рекомендація:** Краще RBF.

---

## Вибір Kernel

### Decision Tree

```
                Дані лінійно розділимі?
                /                    \
             Так                      Ні
              |                        |
       Linear Kernel            Форма boundary?
                                /            \
                           Polynomial      Unknown
                                |              |
                         Polynomial     RBF Kernel ✓
                          Kernel        (universal)
```

### Практичні рекомендації

1. **Почни з RBF** — найбезпечніший вибір
2. **Linear для high-d** — text з d > 1000
3. **Polynomial** — якщо domain knowledge про поліноміальні залежності
4. **Sigmoid** — майже ніколи

---

## Гіперпараметри для Kernel SVM

### Для RBF Kernel (найважливіше!)

**Два головні параметри:**

1. **C** — regularization strength
    
    - Контролює trade-off між margin та помилками
    - Типові значення: [0.1, 1, 10, 100, 1000]
    
2. **γ (gamma)** — kernel coefficient
    
    - Контролює ширину Gaussian
    - Типові значення: [0.001, 0.01, 0.1, 1, 10]

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale']
}

grid_search = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

---

## Вплив C та γ на Decision Boundary

```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# Нелінійні дані
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

params = [
    {'C': 0.1, 'gamma': 0.1, 'title': 'C=0.1, γ=0.1 (Underfitting)'},
    {'C': 1, 'gamma': 1, 'title': 'C=1, γ=1 (Balanced)'},
    {'C': 100, 'gamma': 10, 'title': 'C=100, γ=10 (Overfitting)'},
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, param in enumerate(params):
    svm = SVC(kernel='rbf', C=param['C'], gamma=param['gamma'])
    svm.fit(X, y)
    
    # Decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                     edgecolors='k', s=50)
    axes[idx].scatter(svm.support_vectors_[:, 0],
                     svm.support_vectors_[:, 1],
                     s=200, linewidth=1.5, facecolors='none',
                     edgecolors='red')
    axes[idx].set_title(f"{param['title']}\nSV: {len(svm.support_vectors_)}",
                       fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Візуалізація Grid Search

```python
import pandas as pd
import seaborn as sns

# Grid Search results
results = grid_search.cv_results_

# Витягти C та gamma
C_values = []
gamma_values = []
scores = []

for params, score in zip(results['params'], results['mean_test_score']):
    if isinstance(params['gamma'], float):  # Тільки числові
        C_values.append(params['C'])
        gamma_values.append(params['gamma'])
        scores.append(score)

# Pivot table для heatmap
df = pd.DataFrame({
    'C': C_values,
    'gamma': gamma_values,
    'score': scores
})
pivot = df.pivot_table(values='score', index='gamma', columns='C')

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
plt.title('Grid Search CV Scores (C vs gamma)', 
          fontsize=14, fontweight='bold')
plt.xlabel('C', fontsize=12)
plt.ylabel('gamma', fontsize=12)
plt.tight_layout()
plt.show()
```

---

## Приклад: Нелінійна класифікація

```python
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Різні нелінійні датасети
datasets = {
    'Circles': make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42),
    'Moons': make_moons(n_samples=200, noise=0.15, random_state=42),
}

for name, (X, y) in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    # Розділення
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Нормалізація
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear SVM (baseline)
    svm_linear = SVC(kernel='linear', C=1.0)
    svm_linear.fit(X_train_scaled, y_train)
    linear_score = svm_linear.score(X_test_scaled, y_test)
    
    # RBF SVM
    svm_rbf = SVC(kernel='rbf', C=10, gamma=1)
    svm_rbf.fit(X_train_scaled, y_train)
    rbf_score = svm_rbf.score(X_test_scaled, y_test)
    
    print(f"Linear SVM: {linear_score:.4f}")
    print(f"RBF SVM:    {rbf_score:.4f}")
    print(f"Improvement: {(rbf_score - linear_score)*100:.2f}%")
```

---

## Переваги та недоліки Kernel SVM

### Переваги ✓

|Перевага|Пояснення|
|---|---|
|**Нелінійні boundaries**|Обробка складних patterns|
|**Kernel trick**|Ефективно без явної трансформації|
|**Універсальність**|RBF може апроксимувати все|
|**Висока точність**|Один з найточніших|
|**Flexibility**|Різні kernels|

### Недоліки ✗

|Недолік|Пояснення|
|---|---|
|**Повільне training**|O(n²) до O(n³)|
|**Вибір kernel**|Domain knowledge|
|**Гіперпараметри**|C, γ потрібно підбирати|
|**Великі дані**|n > 100k дуже повільно|
|**Чорна скринька**|Складно інтерпретувати|

---

## Практичні поради 💡

1. **Почни з RBF kernel** — універсальний вибір
2. **Grid Search для C та γ** — обов'язково!
3. **Нормалізація даних** — критично важлива
4. **Linear для high-d** — швидше для text
5. **Візуалізуй boundaries** — розумій поведінку
6. **Polynomial обережно** — degree > 4 рідко потрібний
7. **Перевір support vectors** — якщо > 50% → overfitting
8. **Початкові значення** — C=1, gamma='scale'
9. **Cross-validation** — завжди використовуй CV
10. **Порівняй kernels** — спробуй кілька варіантів

---

## Ключові висновки

> Kernel SVM використовує kernel trick для нелінійних boundaries без явної трансформації в вищий вимір.

**Kernel Trick:**

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

**Популярні kernels:**

- **Linear:** $K(x, z) = x^T z$
- **Polynomial:** $K(x, z) = (x^T z + c)^d$
- **RBF:** $K(x, z) = \exp(-\gamma ||x - z||^2)$ ⭐

**Гіперпараметри RBF:**

- **C:** малий → wide margin, великий → narrow margin
- **γ:** малий → smooth, великий → wiggly

**КРИТИЧНО:**

- Почни з RBF kernel
- Grid Search для C та γ
- Завжди нормалізуй дані

---

#ml #svm #kernel-svm #rbf #kernel-trick #nonlinear