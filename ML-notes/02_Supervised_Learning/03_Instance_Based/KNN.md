# K-Nearest Neighbors (KNN)

## Що це?

**K-Nearest Neighbors (KNN)** — це простий алгоритм supervised learning, який класифікує новий об'єкт на основі **k найближчих сусідів** у просторі ознак.

**Головна ідея:** "Скажи мені, хто твій друг, і я скажу, хто ти" — об'єкти, які знаходяться поруч у просторі ознак, найімовірніше належать до одного класу.

## Навіщо потрібен?

- 🎯 **Простота** — один з найпростіших ML алгоритмів
- 📊 **Універсальність** — класифікація та регресія
- 🔧 **Нелінійні boundaries** — складні decision boundaries
- 💡 **Інтуїтивність** — легко зрозуміти та пояснити
- ⚡ **Lazy learning** — не потребує тренування (instance-based)
- 🎨 **Baseline** — швидкий старт для розуміння даних

## Коли використовувати?

**Потрібно:**

- **Малі/середні датасети** — швидкі обчислення
- Нелінійні decision boundaries
- Потрібен **простий baseline**
- **Інтерпретованість** важлива
- Дані добре масштабовані
- Немає багато ознак (curse of dimensionality)

**Не потрібно:**

- **Великі датасети** (>100k зразків) → дуже повільно
- **Високорозмірні дані** (>50 ознак) → curse of dimensionality
- Потрібна **швидкість inference** → tree-based, linear models
- Дані не нормалізовані
- Багато irrelevant features

---

## Як працює KNN?

### Алгоритм (Класифікація)

**Training phase:** НЕ РОБИТЬ НІЧОГО! Просто зберігає всі тренувальні дані.

**Prediction phase:**

1. Обчислити **відстань** від нового зразка до всіх тренувальних зразків
2. Вибрати **k найближчих сусідів**
3. **Голосування більшості** (majority vote) серед k сусідів
4. Повернути найпопулярніший клас

### Приклад з числами

**Дані:**
```
Train:
  Point 1: [1, 1] → Class A
  Point 2: [2, 2] → Class A
  Point 3: [3, 1] → Class B
  Point 4: [6, 5] → Class B
  Point 5: [7, 7] → Class B

New point: [3, 3] → ?
```

**k=3, Euclidean distance:**

1. Відстані від [3, 3]:
   - Point 1: √((3-1)² + (3-1)²) = √8 = 2.83
   - Point 2: √((3-2)² + (3-2)²) = √2 = 1.41 ← 1st
   - Point 3: √((3-3)² + (3-1)²) = √4 = 2.00 ← 2nd
   - Point 4: √((3-6)² + (3-5)²) = √13 = 3.61
   - Point 5: √((3-7)² + (3-7)²) = √32 = 5.66

2. 3 найближчі: Point 2 (A), Point 3 (B), Point 1 (A)

3. Голосування: A=2, B=1

4. **Prediction: Class A** ✓

---

## Distance Metrics (Метрики відстані)

### 1. Euclidean Distance (Евклідова)

**Формула:**
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

**Коли використовувати:**
- ✅ За замовчуванням (найпопулярніша)
- ✅ Continuous features
- ✅ Рівномірний масштаб ознак

**Приклад:**
```python
x = [1, 2, 3]
y = [4, 5, 6]
d = sqrt((1-4)² + (2-5)² + (3-6)²) = sqrt(27) = 5.20
```

### 2. Manhattan Distance (Манхеттенська)

**Формула:**
$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

**Коли використовувати:**
- ✅ Grid-like структура даних
- ✅ Менш чутлива до outliers
- ✅ High-dimensional data

**Приклад:**
```python
x = [1, 2, 3]
y = [4, 5, 6]
d = |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
```

### 3. Minkowski Distance (Загальна форма)

**Формула:**
$$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

де:
- $p=1$ → Manhattan
- $p=2$ → Euclidean
- $p=\infty$ → Chebyshev

### 4. Cosine Distance (для текстів, sparse data)

**Формула:**
$$\text{similarity}(x, y) = \frac{x \cdot y}{||x|| \cdot ||y||}$$

$$\text{distance}(x, y) = 1 - \text{similarity}(x, y)$$

**Коли використовувати:**
- ✅ Text data (TF-IDF vectors)
- ✅ High-dimensional sparse data
- ✅ Напрямок важливіший за magnitude

### 5. Hamming Distance (для категоріальних)

**Формула:**
$$d(x, y) = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[x_i \neq y_i]$$

**Коли використовувати:**
- ✅ Categorical features
- ✅ Binary data

---

## Вибір k (кількість сусідів)

### Ефект різних k

```
k=1 (занадто малий):        k=optimal:               k=n (занадто великий):
    High variance               Balanced                  High bias
    Overfitting                 ✓ Найкраще                Underfitting
    
Decision boundary:          Decision boundary:        Decision boundary:
Дуже нерівна крива         Згладжена крива           Майже пряма лінія
```

### Як вибрати k?

#### 1. Правило великого пальця

$$k \approx \sqrt{n}$$

де $n$ — кількість тренувальних зразків.

**Приклад:**
- n=100 → k ≈ 10
- n=1000 → k ≈ 32

#### 2. Непарне k для бінарної класифікації

**Чому?** Уникнути нічиї (tie) при голосуванні.

```python
# ✅ ПРАВИЛЬНО
k = 3, 5, 7, 9, ...

# ❌ МОЖЕ БУТИ TIE
k = 2, 4, 6, 8, ...
# При k=2: A=1, B=1 → як обрати?
```

#### 3. Cross-validation

**Найкращий метод:**

```python
from sklearn.model_selection import cross_val_score

k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
```

### Візуалізація впливу k

```
CV Score
    |
0.9 |      ╱────╲
    |     ╱      ╲
0.8 |    ╱        ╲
    |   ╱          ╲___
0.7 |  ╱
    |_╱_____________________ k
      1  5  10  15  20  25

Overfitting ← Optimal → Underfitting
```

---

## KNN для Регресії

### Алгоритм

Замість голосування — **усереднення** значень k сусідів:

$$\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i$$

### Weighted KNN Regression

Сусіди, які ближче, мають більшу вагу:

$$\hat{y} = \frac{\sum_{i=1}^{k} w_i \cdot y_i}{\sum_{i=1}^{k} w_i}$$

де $w_i = \frac{1}{d_i}$ або $w_i = \frac{1}{d_i^2}$

### Приклад

**Дані:**
```
Point 1: [1, 1] → y=10
Point 2: [2, 2] → y=20
Point 3: [3, 1] → y=15
Point 4: [6, 5] → y=50

New point: [2.5, 2.5] → ?
```

**k=3:**
1. Відстані: Point 2 (d=0.71), Point 3 (d=1.58), Point 1 (d=2.12)
2. k=3 сусіди: y=[20, 15, 10]
3. Передбачення: $\hat{y} = \frac{20+15+10}{3} = 15$

**Weighted (w = 1/d):**
$$\hat{y} = \frac{20 \cdot \frac{1}{0.71} + 15 \cdot \frac{1}{1.58} + 10 \cdot \frac{1}{2.12}}{\frac{1}{0.71} + \frac{1}{1.58} + \frac{1}{2.12}} \approx 17.5$$

Weighted дає більшу вагу ближчим сусідам ✓

---

## Код (Python + scikit-learn)

### Класифікація

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Генерація даних
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Нормалізація (КРИТИЧНО для KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. KNN Classifier
knn = KNeighborsClassifier(
    n_neighbors=5,           # k=5
    weights='uniform',       # або 'distance' (weighted)
    metric='euclidean',      # або 'manhattan', 'minkowski'
    algorithm='auto',        # 'ball_tree', 'kd_tree', 'brute'
    n_jobs=-1               # Паралелізація
)

# 4. Навчання (просто зберігає дані!)
knn.fit(X_train_scaled, y_train)

# 5. Передбачення
y_pred = knn.predict(X_test_scaled)
y_pred_proba = knn.predict_proba(X_test_scaled)

# 6. Оцінка
print("=== KNN Classifier ===")
print(f"Train Accuracy: {knn.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 7. Візуалізація decision boundary
def plot_decision_boundary(X, y, model, title):
    h = 0.02  # step size
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
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

plot_decision_boundary(X_test_scaled, y_test, knn, 
                      'KNN Decision Boundary (k=5)')
```

### Регресія

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Дані
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

# KNN Regressor
knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',      # Weighted averaging
    metric='euclidean'
)

# Навчання
knn_reg.fit(X_train, y_train)

# Передбачення
y_pred = knn_reg.predict(X_test)

# Метрики
print("=== KNN Regressor ===")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# Візуалізація
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_plot = knn_reg.predict(X_plot)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.5, s=30, label='Train', color='blue')
plt.scatter(X_test, y_test, alpha=0.5, s=50, label='Test', color='green')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='KNN Prediction')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('KNN Regression (k=5)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Вибір оптимального k через Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Тестуємо різні k
k_values = range(1, 51)
train_scores = []
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train score
    knn.fit(X_train_scaled, y_train)
    train_scores.append(knn.score(X_train_scaled, y_train))
    
    # Cross-validation score
    cv_score = cross_val_score(knn, X_train_scaled, y_train, 
                               cv=5, scoring='accuracy').mean()
    cv_scores.append(cv_score)

# Візуалізація
plt.figure(figsize=(12, 6))
plt.plot(k_values, train_scores, 'o-', linewidth=2, label='Train Score')
plt.plot(k_values, cv_scores, 's-', linewidth=2, label='CV Score')
plt.xlabel('k (Number of Neighbors)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN: Train vs CV Score for Different k', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axvline(x=k_values[np.argmax(cv_scores)], 
            color='red', linestyle='--', 
            label=f'Optimal k={k_values[np.argmax(cv_scores)]}')
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Оптимальний k
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")
print(f"Best CV Score: {max(cv_scores):.4f}")

# Фінальна модель
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train_scaled, y_train)
print(f"Test Score: {knn_final.score(X_test_scaled, y_test):.4f}")
```

---

## Weighted vs Uniform KNN

### Uniform (за замовчуванням)

Всі k сусідів мають **однакову вагу**:

$$P(\text{class } c) = \frac{\text{count}(c \text{ in k neighbors})}{k}$$

### Distance-weighted

Ближчі сусіди мають **більшу вагу**:

$$w_i = \frac{1}{d_i}$$

$$P(\text{class } c) = \frac{\sum_{i \in c} w_i}{\sum_{i=1}^{k} w_i}$$

### Порівняння

```python
# Uniform
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_uniform.fit(X_train_scaled, y_train)
score_uniform = knn_uniform.score(X_test_scaled, y_test)

# Weighted
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_weighted.fit(X_train_scaled, y_train)
score_weighted = knn_weighted.score(X_test_scaled, y_test)

print(f"Uniform weights:   {score_uniform:.4f}")
print(f"Distance weights:  {score_weighted:.4f}")
```

**Коли weighted краще:**
- ✅ Нерівномірний розподіл даних
- ✅ Clusters різних розмірів
- ✅ k досить великий

**Коли uniform достатньо:**
- ✅ Рівномірний розподіл
- ✅ Малий k
- ✅ Простіша модель

---

## Алгоритми пошуку сусідів

### 1. Brute Force

**Як працює:** Обчислює відстань до **всіх** тренувальних зразків.

**Складність:**
- Training: O(1) — просто зберігає дані
- Prediction: O(n·d) — n зразків, d ознак

**Коли використовувати:**
- Малі датасети (n < 1000)
- Гарантовано точний результат

### 2. KD-Tree

**Як працює:** Будує дерево для швидкого пошуку в просторі.

**Складність:**
- Training: O(n·log(n)·d)
- Prediction: O(log(n)·d)

**Коли використовувати:**
- ✅ Середні датасети (1k-100k)
- ✅ Низьковимірні дані (d < 20)
- ❌ НЕ працює добре при d > 50

### 3. Ball Tree

**Як працює:** Інша структура дерева, краще для high-dimensional.

**Складність:**
- Схожа на KD-Tree

**Коли використовувати:**
- ✅ High-dimensional data (d > 20)
- ✅ Non-Euclidean metrics

### Порівняння

```python
import time

algorithms = ['brute', 'kd_tree', 'ball_tree', 'auto']

for algo in algorithms:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
    
    # Training time
    start = time.time()
    knn.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    
    # Prediction time
    start = time.time()
    knn.predict(X_test_scaled)
    pred_time = time.time() - start
    
    print(f"{algo:10s}: Train={train_time:.4f}s, Pred={pred_time:.4f}s")
```

**Рекомендація:** Використовуй **'auto'** — sklearn сам обере найкращий!

---

## Curse of Dimensionality

### Проблема

**В high-dimensional просторі всі точки стають "далекими" одна від одної.**

### Математика

Для одиничного кубу в $d$ вимірах, об'єм кулі радіусом 0.5:

$$V_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)} \cdot 0.5^d$$

**Результат:**
- d=2: V ≈ 0.785 (78.5% кубу)
- d=10: V ≈ 0.0025 (0.25% кубу!)
- d=100: V ≈ 0 (майже нічого!)

**Висновок:** В high dimensions дані стають **sparse** (розрідженими).

### Ефект на KNN

```
Low dimensions (d=2):        High dimensions (d=100):
    
Точки згруповані           Всі точки далеко одна
Чіткі clusters             від одної
Nearest neighbor           "Nearest" не має сенсу
має сенс ✓                 ✗
```

### Візуалізація

```python
# Експеримент: відстань до найближчого сусіда
dimensions = [2, 5, 10, 20, 50, 100]
nearest_distances = []

for d in dimensions:
    X_random = np.random.rand(1000, d)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_random, np.zeros(1000))
    
    # Відстань до найближчого сусіда
    distances, _ = knn.kneighbors(X_random[:100])
    nearest_distances.append(distances.mean())

plt.figure(figsize=(10, 6))
plt.plot(dimensions, nearest_distances, 'o-', linewidth=2)
plt.xlabel('Number of Dimensions', fontsize=12)
plt.ylabel('Average Distance to Nearest Neighbor', fontsize=12)
plt.title('Curse of Dimensionality Effect on KNN', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Як боротися?

1. **Dimensionality reduction:**
   - PCA
   - Feature selection
   - Feature engineering

2. **Feature scaling:**
   - StandardScaler
   - MinMaxScaler

3. **Distance metric:**
   - Експериментуй з різними metrics
   - Cosine для high-dimensional sparse

4. **Використай інші алгоритми:**
   - Tree-based models менш чутливі

---

## Feature Scaling для KNN

### Чому критично важливо?

KNN базується на **відстанях** → ознаки з великими значеннями домінують!

### Приклад проблеми

```
Дані БЕЗ scaling:
  Age: [25, 30, 35] (range: 10)
  Salary: [30000, 50000, 70000] (range: 40000)

Відстань між [25, 30000] та [30, 50000]:
d = sqrt((30-25)² + (50000-30000)²)
  = sqrt(25 + 400000000)
  = sqrt(400000025)
  ≈ 20000

Age майже не впливає! Salary домінує! ✗
```

### Рішення: Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (найпопулярніший)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (альтернатива)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### Порівняння з/без scaling

```python
# БЕЗ scaling
knn_no_scale = KNeighborsClassifier(n_neighbors=5)
knn_no_scale.fit(X_train, y_train)
score_no_scale = knn_no_scale.score(X_test, y_test)

# З scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
score_scaled = knn_scaled.score(X_test_scaled, y_test)

print(f"Without scaling: {score_no_scale:.4f}")
print(f"With scaling:    {score_scaled:.4f}")
print(f"Improvement:     {(score_scaled - score_no_scale)*100:.2f}%")
```

**Типовий результат:**
```
Without scaling: 0.7200
With scaling:    0.8900
Improvement:     17.00%  ← Величезна різниця!
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Простота** | Один з найпростіших алгоритмів |
| **Інтуїтивність** | Легко зрозуміти та пояснити |
| **Універсальність** | Класифікація + регресія |
| **Нелінійні boundaries** | Складні decision boundaries |
| **No training phase** | Instant "навчання" |
| **Online learning** | Легко додавати нові дані |
| **Не потребує assumptions** | Працює з будь-якими даними |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Повільний prediction** | Обчислює відстань до всіх точок |
| **Curse of dimensionality** | Погано працює при d > 20-50 |
| **Memory-intensive** | Зберігає всі тренувальні дані |
| **Чутливість до scaling** | Критично потребує нормалізації |
| **Irrelevant features** | Всі ознаки впливають однаково |
| **Imbalanced data** | Bias до majority class |
| **Не працює з великими даними** | n > 100k дуже повільно |
| **Не інтерпретовані weights** | Немає feature importance |

---

## Порівняння з іншими алгоритмами

### KNN vs Decision Tree

| Критерій | KNN | Decision Tree |
|----------|-----|---------------|
| **Training** | O(1) | O(n·log(n)·d) |
| **Prediction** | O(n·d) | O(log(n)) |
| **Інтерпретованість** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Feature scaling** | Критично | Не потрібна |
| **Overfitting** | k контролює | Легко overfits |
| **Високі розмірності** | ❌ Погано | ✅ Працює |

### KNN vs Logistic Regression

| Критерій | KNN | Logistic Regression |
|----------|-----|---------------------|
| **Assumptions** | Немає | Лінійна розділимість |
| **Boundaries** | Нелінійні ✓ | Лінійні |
| **Training** | O(1) | O(n·d) |
| **Prediction** | O(n·d) | O(d) |
| **Великі дані** | ❌ Повільно | ✅ Швидко |
| **Інтерпретованість** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### KNN vs SVM

| Критерій | KNN | SVM |
|----------|-----|-----|
| **Training** | O(1) | O(n²) to O(n³) |
| **Prediction** | O(n·d) | O(n_sv·d) |
| **Kernel trick** | ❌ Немає | ✅ Так |
| **Margin optimization** | ❌ Немає | ✅ Так |
| **Високі розмірності** | ❌ Погано | ✅ Працює |
| **Простота** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## Коли використовувати KNN

### Ідеально підходить ✓

- **Малі/середні датасети** (n < 10,000)
- **Низьковимірні дані** (d < 20)
- Нелінійні decision boundaries
- Потрібен **швидкий baseline**
- **Інтерпретованість** важлива
- Дані добре масштабовані
- Online learning (додавання нових даних)

### Краще використати інше ✗

- **Великі датасети** (n > 100k) → Random Forest, XGBoost, Linear Models
- **High-dimensional** (d > 50) → Tree-based, SVM with kernel
- Потрібна **швидкість inference** → Linear Models, Tree-based
- Багато irrelevant features → Tree-based (автоматична feature selection)
- Дані не нормалізовані і не можна нормалізувати → Tree-based

---

## Практичні поради 💡

1. **ЗАВЖДИ нормалізуй дані** — StandardScaler перед KNN
2. **Почни з k=√n** — потім tuning через CV
3. **Непарне k** для бінарної класифікації
4. **weights='distance'** часто краще за 'uniform'
5. **algorithm='auto'** — sklearn обере найкращий
6. **Cross-validation** для вибору k — не гадай!
7. **Feature selection** — видали irrelevant features
8. **Візуалізуй decision boundary** — зрозумій модель
9. **Обмеж d < 20** — інакше curse of dimensionality
10. **Порівняй з іншими** — KNN часто baseline

---

## Реальний приклад: Діагностика діабету

```python
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Синтетичні дані
np.random.seed(42)
n_samples = 1000

data = {
    'glucose': np.random.randint(70, 200, n_samples),
    'bmi': np.random.uniform(18, 45, n_samples),
    'age': np.random.randint(21, 81, n_samples),
    'blood_pressure': np.random.randint(60, 120, n_samples),
}

# Симулюємо діабет
diabetes_prob = (
    (data['glucose'] > 140) * 0.4 +
    (data['bmi'] > 30) * 0.3 +
    (data['age'] > 50) * 0.2 +
    np.random.uniform(0, 0.1, n_samples)
)
data['diabetes'] = (diabetes_prob > 0.5).astype(int)

df = pd.DataFrame(data)

X = df.drop('diabetes', axis=1)
y = df['diabetes']

print("="*70)
print("KNN FOR DIABETES PREDICTION")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Diabetes rate: {y.mean():.2%}")

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (КРИТИЧНО!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Базова модель (k=5)
print("\n" + "="*70)
print("1. BASELINE KNN (k=5)")
print("="*70)

knn_base = KNeighborsClassifier(n_neighbors=5)
knn_base.fit(X_train_scaled, y_train)

y_pred = knn_base.predict(X_test_scaled)
y_pred_proba = knn_base.predict_proba(X_test_scaled)[:, 1]

print(f"Train Accuracy: {knn_base.score(X_train_scaled, y_train):.4f}")
print(f"Test Accuracy: {knn_base.score(X_test_scaled, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 2. Підбір оптимального k
print("\n" + "="*70)
print("2. FINDING OPTIMAL k")
print("="*70)

k_values = range(1, 51, 2)  # Непарні значення
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}")
print(f"Best CV Score: {max(cv_scores):.4f}")

# 3. Порівняння uniform vs weighted
print("\n" + "="*70)
print("3. UNIFORM vs WEIGHTED")
print("="*70)

knn_uniform = KNeighborsClassifier(n_neighbors=optimal_k, weights='uniform')
knn_uniform.fit(X_train_scaled, y_train)
score_uniform = knn_uniform.score(X_test_scaled, y_test)

knn_weighted = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
knn_weighted.fit(X_train_scaled, y_train)
score_weighted = knn_weighted.score(X_test_scaled, y_test)

print(f"Uniform weights:   {score_uniform:.4f}")
print(f"Distance weights:  {score_weighted:.4f}")
print(f"Better: {'Weighted' if score_weighted > score_uniform else 'Uniform'}")

# 4. Grid Search для всіх параметрів
print("\n" + "="*70)
print("4. GRID SEARCH CV")
print("="*70)

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print("Best parameters:")
print(grid_search.best_params_)
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# 5. Фінальна модель
print("\n" + "="*70)
print("5. FINAL MODEL EVALUATION")
print("="*70)

best_knn = grid_search.best_estimator_
y_pred_final = best_knn.predict(X_test_scaled)
y_pred_proba_final = best_knn.predict_proba(X_test_scaled)[:, 1]

print(f"Test Accuracy: {best_knn.score(X_test_scaled, y_test):.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba_final):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=['No Diabetes', 'Diabetes']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

# Візуалізації
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. k vs CV Score
axes[0, 0].plot(k_values, cv_scores, 'o-', linewidth=2)
axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k={optimal_k}')
axes[0, 0].set_xlabel('k (Number of Neighbors)', fontsize=12)
axes[0, 0].set_ylabel('Cross-Validation Score', fontsize=12)
axes[0, 0].set_title('CV Score vs k', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# 2. Feature Importance (через perturbation)
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    best_knn, X_test_scaled, y_test, n_repeats=10, random_state=42
)

sorted_idx = perm_importance.importances_mean.argsort()[::-1]
axes[0, 1].barh(range(len(sorted_idx)), 
                perm_importance.importances_mean[sorted_idx])
axes[0, 1].set_yticks(range(len(sorted_idx)))
axes[0, 1].set_yticklabels([X.columns[i] for i in sorted_idx])
axes[0, 1].set_xlabel('Permutation Importance', fontsize=12)
axes[0, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred_proba_final)
auc = roc_auc_score(y_test, y_pred_proba_final)

axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'KNN (AUC={auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
axes[1, 0].set_xlabel('False Positive Rate', fontsize=12)
axes[1, 0].set_ylabel('True Positive Rate', fontsize=12)
axes[1, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# 4. Prediction Distribution
axes[1, 1].hist(y_pred_proba_final[y_test == 0], bins=20, alpha=0.6, 
                label='No Diabetes', color='blue', edgecolor='black')
axes[1, 1].hist(y_pred_proba_final[y_test == 1], bins=20, alpha=0.6, 
                label='Diabetes', color='red', edgecolor='black')
axes[1, 1].set_xlabel('Predicted Probability', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
```

---

## Поширені помилки ❌

### 1. Забути нормалізувати дані

```python
# ❌ КРИТИЧНА ПОМИЛКА
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # БЕЗ scaling!

# ✅ ПРАВИЛЬНО
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn.fit(X_train_scaled, y_train)
```

### 2. Використовувати парне k для бінарної класифікації

```python
# ❌ МОЖЕ БУТИ TIE
knn = KNeighborsClassifier(n_neighbors=4)

# ✅ ПРАВИЛЬНО (непарне)
knn = KNeighborsClassifier(n_neighbors=5)
```

### 3. Не підбирати k через CV

```python
# ❌ ПРОСТО ВЗЯТИ k=5
knn = KNeighborsClassifier(n_neighbors=5)

# ✅ ПРАВИЛЬНО (підібрати через CV)
k_values = range(1, 31, 2)
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), 
                              X_train, y_train, cv=5).mean() 
             for k in k_values]
optimal_k = k_values[np.argmax(cv_scores)]
```

### 4. Використовувати KNN на великих даних

```python
# ❌ ДУЖЕ ПОВІЛЬНО
# X_train має 1,000,000 зразків
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # Training швидко (O(1))
knn.predict(X_test)         # Prediction ДУЖЕ ПОВІЛЬНО (O(n·d))

# ✅ ВИКОРИСТАЙ ІНШИЙ АЛГОРИТМ
# Random Forest, XGBoost, Logistic Regression
```

### 5. Не видаляти irrelevant features

```python
# KNN використовує ВСІ ознаки однаково
# Irrelevant features додають noise!

# ✅ ЗРОБИ FEATURE SELECTION
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

---

## Пов'язані теми

- [[04_SVM]] — інший instance-based підхід
- [[01_Decision_Trees]] — альтернатива для нелінійних boundaries
- [[Feature_Scaling]] — критично для KNN
- [[Cross_Validation]] — вибір k
- [[Distance_Metrics]] — різні способи вимірювання відстані

## Ресурси

- [Scikit-learn: KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [KNN Algorithm Explained](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
- [StatQuest: KNN](https://www.youtube.com/watch?v=HVXime0nQeI)

---

## Ключові висновки

> K-Nearest Neighbors класифікує об'єкти на основі k найближчих сусідів у просторі ознак через голосування більшості.

**Основні принципи:**
- **Lazy learning** — не тренується, просто зберігає дані
- **Instance-based** — рішення базується на схожості
- **Distance-based** — використовує метрики відстані
- **Non-parametric** — не робить assumptions про розподіл даних

**Формула (класифікація):**
$$\hat{y} = \text{mode}\{y_1, y_2, ..., y_k\} \text{ для k найближчих сусідів}$$

**Формула (регресія):**
$$\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i$$

**Коли використовувати:**
- Малі дані + нелінійні boundaries + швидкий baseline = KNN ✓
- Великі дані або high-dimensional → інші алгоритми ✓

**КРИТИЧНО важливо:**
- Завжди нормалізуй дані (StandardScaler)
- Підбирай k через cross-validation
- Обмеж кількість ознак (d < 20 ідеально)
- Використовуй на малих/середніх датасетах

**Trade-offs:**
- Простота vs Швидкість prediction
- Гнучкість vs Curse of dimensionality
- No training vs Повільний inference

---

#ml #supervised-learning #classification #regression #knn #k-nearest-neighbors #instance-based #lazy-learning #distance-metrics
