# Linear SVM (Лінійний SVM)

## Що це?

**Linear SVM** — це алгоритм класифікації, який знаходить **оптимальну гіперплощину**, що максимально розділяє класи, максимізуючи **margin** (відстань до найближчих точок).

**Головна ідея:** знайти таку лінію (в 2D) або гіперплощину (в n-D), яка не просто розділяє класи, але робить це з **максимальним запасом безпеки**.

---

## Навіщо потрібен?

- Оптимальна boundary через margin maximization
- Robust до overfitting — залежить тільки від support vectors
- Працює з high-dimensional даними (навіть при d > n)
- Швидкість — Linear SVM швидший за kernel варіанти
- Стандарт для text classification
- Міцна теоретична база (statistical learning theory)

---

## Коли використовувати?

### Потрібно:

- Лінійно розділимі дані (або майже)
- High-dimensional дані (d > 50, особливо text)
- Потрібна висока точність
- Бінарна класифікація
- Середні/великі датасети (n = 10k-1M)

### Не потрібно:

- Дані нелінійно розділимі → Kernel SVM
- Дуже великі датасети (n > 1M) → SGDClassifier with hinge loss
- Потрібна інтерпретованість → Logistic Regression
- Probabilistic predictions критичні → Logistic Regression

---

## Концепція: Margin Maximization

### Ключові компоненти:

1. **Decision boundary** (гіперплощина) — розділяє класи
2. **Margin** — відстань між boundary та найближчими точками
3. **Support vectors** — найближчі точки, що визначають margin
4. **Goal:** максимізувати margin для кращої узагальнювальної здатності

### Чому максимізувати margin?

- Більший margin → краще узагальнення на нових даних
- Менше ризику помилитися через noise або outliers
- Теоретичні гарантії з VC dimension theory

---

## Математика Linear SVM

### Гіперплощина

**Рівняння гіперплощини:**

$$w^T x + b = 0$$

де:

- $w \in \mathbb{R}^d$ — вектор нормалі до гіперплощини (weights)
- $b \in \mathbb{R}$ — bias (зміщення)
- $x \in \mathbb{R}^d$ — точка у просторі ознак

### Класифікація

**Decision function:**

$$f(x) = \text{sign}(w^T x + b)$$

- $f(x) = +1$ якщо $w^T x + b > 0$ → Class +1
- $f(x) = -1$ якщо $w^T x + b < 0$ → Class -1

### Відстань до гіперплощини

Відстань від точки $x_i$ до гіперплощини:

$$\text{distance}(x_i) = \frac{|w^T x_i + b|}{||w||}$$

### Margin

**Geometric margin** (для правильно класифікованих точок):

$$\gamma_i = \frac{y_i(w^T x_i + b)}{||w||}$$

де $y_i \in {-1, +1}$ — справжній клас.

**Margin моделі:**

$$\gamma = \min_{i=1,...,n} \gamma_i$$

Це відстань від гіперплощини до **найближчої** точки.

---

## Hard Margin SVM

### Оптимізаційна задача

**Мета:** Максимізувати margin

$$\max_{w, b} \frac{1}{||w||}$$

subject to:

$$y_i(w^T x_i + b) \geq 1 \quad \forall i = 1, ..., n$$

**Еквівалентно (мінімізація):**

$$\min_{w, b} \frac{1}{2}||w||^2$$

subject to:

$$y_i(w^T x_i + b) \geq 1 \quad \forall i$$

**Чому $||w||^2$?**

- Максимізувати $\frac{1}{||w||}$ = мінімізувати $||w||$
- Використовуємо $\frac{1}{2}||w||^2$ для математичної зручності (квадратична функція)

### Обмеження

**Геометрична інтерпретація $y_i(w^T x_i + b) \geq 1$:**

- Для класу +1: $w^T x_i + b \geq +1$
- Для класу -1: $w^T x_i + b \leq -1$

Це означає, що всі точки знаходяться **за межами** margin.

### Проблеми Hard Margin

- **Вимагає лінійної розділимості** — не працює, якщо класи перетинаються
- **Дуже чутливий до outliers** — один outlier може зруйнувати все
- **Може не мати рішення** — якщо дані не лінійно розділимі

```
Outlier →  •
             •
               •
         ×  ×  × ×  ×
           ×  ×  ×
             ×

Hard Margin не може знайти рішення!
```

---

## Soft Margin SVM

### Мотивація

Реальні дані майже **ніколи не є ідеально лінійно розділимими**:

- Noise в даних
- Outliers
- Перетинання класів

**Рішення:** Дозволити **деякі помилки** через **slack variables** $\xi_i \geq 0$.

### Slack Variables (ξ)

```
Правильно класифіковані:
                •  ξ=0
              ───────
            ×    ξ=0

Всередині margin:
              •  ξ=0.3
            ───────
          ×      ξ=0

Неправильно класифіковані:
          ×      ξ=1.5
        ───────
      •          ξ=0
```

**Інтерпретація $\xi_i$:**

- $\xi_i = 0$ → точка правильно класифікована з margin ≥ 1 ✓
- $0 < \xi_i < 1$ → точка всередині margin, але правильний клас ⚠️
- $\xi_i \geq 1$ → точка неправильно класифікована ✗

### Оптимізаційна задача (Soft Margin)

$$\min_{w, b, \xi} \frac{1}{2}||w||^2 + C \sum_{i=1}^{n} \xi_i$$

subject to:

$$y_i(w^T x_i + b) \geq 1 - \xi_i \quad \forall i$$

$$\xi_i \geq 0 \quad \forall i$$

**Компоненти:**

- $\frac{1}{2}||w||^2$ — максимізація margin (мінімізація $||w||$)
- $C \sum \xi_i$ — мінімізація помилок
- **C** — regularization parameter (trade-off)

---

## Параметр C (Regularization)

### Роль C

**C контролює trade-off між:**

1. Максимізацією margin (широкий margin)
2. Мінімізацією помилок класифікації (малі $\xi_i$)

### Ефект різних C

```
C малий (0.01):              C оптимальний (1.0):        C великий (100):
════════════════            ═════════                   ═══════
Широкий margin              Баланс                      Вузький margin
Багато помилок              Мало помилок                Майже немає помилок
High bias                   ✓ Найкраще                  High variance
Underfitting                                            Overfitting

  ×  ×  ×  •  •               × × × | • •                ××× |••
  ×  ×  •  •  •               × × × | • •                ××× |••
  ×  •  •  •  •               × × × | • •                ××× |••
────────────────            ─────────                   ───────
```

### Trade-off формула

$$\text{Objective} = \underbrace{\frac{1}{2}||w||^2}_{\text{margin term}} + \underbrace{C \sum \xi_i}_{\text{error term}}$$

**Інтерпретація:**

- **Малий C (C → 0):** більше уваги margin → широкий margin, більше помилок
- **Великий C (C → ∞):** більше уваги помилкам → вузький margin, менше помилок (ризик overfitting)

### Як вибрати C?

**Grid Search через Cross-Validation:**

```python
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
best_C = None
best_score = 0

for C in C_values:
    svm = SVC(kernel='linear', C=C)
    score = cross_val_score(svm, X, y, cv=5).mean()
    if score > best_score:
        best_score = score
        best_C = C

print(f"Optimal C: {best_C}")
```

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Генерація даних
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Нормалізація (РЕКОМЕНДОВАНО для SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Linear SVM (два способи)

# Спосіб 1: SVC з linear kernel
svm_linear = SVC(
    kernel='linear',
    C=1.0,              # Regularization parameter
    random_state=42
)

# Спосіб 2: LinearSVC (ШВИДШЕ для linear!)
linear_svc = LinearSVC(
    C=1.0,
    max_iter=10000,
    random_state=42,
    dual=True           # dual='auto' в новіших версіях
)

# 4. Навчання
svm_linear.fit(X_train_scaled, y_train)

# 5. Передбачення
y_pred = svm_linear.predict(X_test_scaled)

# 6. Оцінка
print("=== Linear SVM ===")
print(f"Train Accuracy: {svm_linear.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Support Vectors
print(f"\nNumber of support vectors: {len(svm_linear.support_vectors_)}")
print(f"Support vectors per class: {svm_linear.n_support_}")
print(f"Percentage: {len(svm_linear.support_vectors_)/len(X_train_scaled)*100:.1f}%")

# 8. Коефіцієнти (для linear)
print(f"\nWeights (w): {svm_linear.coef_}")
print(f"Bias (b): {svm_linear.intercept_}")
```

### Візуалізація Decision Boundary

```python
def plot_svm_decision_boundary(X, y, model, title):
    """Візуалізація decision boundary та support vectors"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                edgecolors='k', s=50)
    
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=200, linewidth=2, facecolors='none',
                   edgecolors='red', label='Support Vectors')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

plot_svm_decision_boundary(X_test_scaled, y_test, svm_linear,
                           'Linear SVM Decision Boundary')
```

---

## Переваги та недоліки

### Переваги

- **Оптимальна boundary** — margin maximization
- **High-dimensional дані** — працює при d > n
- **Robust** — залежить від support vectors, не від усіх даних
- **Text classification** — стандарт для high-dimensional text
- **Швидкість** — LinearSVC дуже швидкий
- **Теоретична база** — міцна математична основа

### Недоліки

- **Тільки лінійні залежності** — не працює для нелінійних даних
- **Feature scaling критичне** — обов'язкова нормалізація
- **Гіперпараметр C** — потрібно підбирати через CV
- **Не probabilistic** — тільки decision function, без ймовірностей
- **Чутливість до дисбалансу класів** — потрібно враховувати

---

## Ключові висновки

Linear SVM знаходить оптимальну лінійну гіперплощину через margin maximization.

**Формула оптимізації:**

$$\min_{w, b} \frac{1}{2}||w||^2 + C \sum \xi_i$$

**Критично важливо:**

- ЗАВЖДИ нормалізуй дані (StandardScaler або MinMaxScaler)
- Підбирай параметр C через Cross-Validation
- Використовуй LinearSVC для великих датасетів (швидше)
- Враховуй дисбаланс класів (параметр class_weight)

---

**Теги:** #ml #svm #linear-svm #classification #margin-maximization #machine-learning