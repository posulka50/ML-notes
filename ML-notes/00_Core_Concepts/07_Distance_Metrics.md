# Distance Metrics (Метрики відстані)

## Що це?

**Distance Metrics** — це функції, які **вимірюють відстань** (несхожість) між двома точками в просторі. Чим менша відстань, тим більш схожі об'єкти.

**Головна ідея:** різні метрики підходять для різних типів даних та задач. Вибір правильної метрики критично важливий для успіху алгоритмів, що базуються на відстані.

## Навіщо потрібно?

- 🎯 **K-Nearest Neighbors (KNN)** — пошук найближчих сусідів
- 🔍 **K-Means Clustering** — групування схожих об'єктів
- 📊 **Anomaly Detection** — виявлення викидів
- 🗺️ **Dimensionality Reduction** — MDS, t-SNE
- 🔎 **Information Retrieval** — пошук схожих документів
- 📸 **Image Recognition** — порівняння зображень

## Коли використовувати?

**Потрібно:**
- Будь-які distance-based алгоритми (KNN, K-Means)
- Порівняння векторів/об'єктів
- Пошук найближчих сусідів
- Кластеризація

**Не потрібно:**
- Linear models (використовують інші підходи)
- Tree-based models (не базуються на відстані)

---

## Класифікація метрик

```
Distance Metrics
│
├── Minkowski Distance Family
│   ├── Euclidean Distance (p=2)
│   ├── Manhattan Distance (p=1)
│   └── Chebyshev Distance (p=∞)
│
├── Specialized Metrics
│   ├── Cosine Distance
│   ├── Hamming Distance
│   └── Mahalanobis Distance
│
└── Other Metrics
    ├── Jaccard Distance
    ├── Haversine Distance
    └── Edit Distance (Levenshtein)
```

---

# 1. Euclidean Distance (Евклідова відстань)

## Формула

$$d_{\text{Euclidean}}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

Або у векторній формі:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2$$

## Інтуїція

**Euclidean Distance** — це **пряма лінія** між двома точками. Найінтуїтивніша метрика відстані.

```
2D приклад:

    y
    |
  4 |     B(3,4)
    |      *
  3 |     /
    |    /
  2 |   /
    |  /
  1 | * A(1,1)
    |
    +------------- x
      1 2 3 4

A = (1, 1)
B = (3, 4)

d = √[(3-1)² + (4-1)²]
  = √[4 + 9]
  = √13
  ≈ 3.606
```

## Код

```python
import numpy as np

def euclidean_distance(x, y):
    """
    Обчислити Euclidean distance між двома векторами
    
    Parameters:
    -----------
    x, y : array-like
        Вектори
        
    Returns:
    --------
    float : відстань
    """
    return np.sqrt(np.sum((x - y) ** 2))

# Приклад
x = np.array([1, 1])
y = np.array([3, 4])

dist = euclidean_distance(x, y)
print(f"Euclidean distance: {dist:.4f}")

# Через scipy
from scipy.spatial.distance import euclidean
dist_scipy = euclidean(x, y)
print(f"Scipy distance: {dist_scipy:.4f}")

# Через sklearn
from sklearn.metrics.pairwise import euclidean_distances
dist_sklearn = euclidean_distances([x], [y])[0, 0]
print(f"Sklearn distance: {dist_sklearn:.4f}")
```

## Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Точки
A = np.array([1, 1])
B = np.array([3, 4])

# Візуалізація
plt.figure(figsize=(8, 8))

# Точки
plt.scatter(*A, s=200, c='blue', marker='o', label='A(1,1)', zorder=3)
plt.scatter(*B, s=200, c='red', marker='o', label='B(3,4)', zorder=3)

# Пряма лінія (Euclidean)
plt.plot([A[0], B[0]], [A[1], B[1]], 'g-', linewidth=3, 
         label=f'Euclidean = {euclidean_distance(A, B):.2f}')

# Сітка
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Euclidean Distance', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)

# Анотації
mid_x = (A[0] + B[0]) / 2
mid_y = (A[1] + B[1]) / 2
plt.annotate(f'd = {euclidean_distance(A, B):.2f}', 
            xy=(mid_x, mid_y), xytext=(mid_x + 0.5, mid_y + 0.5),
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.tight_layout()
plt.show()
```

## Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Інтуїтивно зрозуміла | ❌ Чутлива до масштабу ознак |
| ✅ Геометрична інтерпретація | ❌ Curse of dimensionality |
| ✅ Підходить для continuous features | ❌ Не підходить для categorical |
| ✅ Стандарт для багатьох задач | ❌ Припускає ізотропну структуру |

## Коли використовувати?

- ✅ Geographical data (координати)
- ✅ Computer vision (pixel values)
- ✅ Continuous numeric features
- ✅ Коли ознаки мають однаковий масштаб
- ❌ Text data (використовуй cosine)
- ❌ High-dimensional sparse data

---

# 2. Manhattan Distance (Манхеттенська відстань)

## Формула

$$d_{\text{Manhattan}}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i|$$

Або:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1$$

## Інтуїція

**Manhattan Distance** (також **L1 distance**, **Taxicab distance**) — це сума **абсолютних різниць**. Назва від вулиць Манхеттена (можна рухатися тільки вздовж вулиць).

```
2D приклад:

    y
    |
  4 |     B(3,4)
    |     *
  3 |     ↑
    |     ↑
  2 |     ↑
    |     ↑
  1 | *→→→* A(1,1)
    |
    +------------- x
      1 2 3 4

A = (1, 1)
B = (3, 4)

d = |3-1| + |4-1|
  = 2 + 3
  = 5

Не можна йти по діагоналі — тільки вздовж осей!
```

## Код

```python
import numpy as np

def manhattan_distance(x, y):
    """Обчислити Manhattan distance"""
    return np.sum(np.abs(x - y))

# Приклад
x = np.array([1, 1])
y = np.array([3, 4])

dist = manhattan_distance(x, y)
print(f"Manhattan distance: {dist:.4f}")

# Через scipy
from scipy.spatial.distance import cityblock
dist_scipy = cityblock(x, y)
print(f"Scipy distance: {dist_scipy:.4f}")

# Через sklearn
from sklearn.metrics.pairwise import manhattan_distances
dist_sklearn = manhattan_distances([x], [y])[0, 0]
print(f"Sklearn distance: {dist_sklearn:.4f}")
```

## Порівняння Euclidean vs Manhattan

```python
import matplotlib.pyplot as plt
import numpy as np

# Точки
A = np.array([1, 1])
B = np.array([3, 4])

# Візуалізація
plt.figure(figsize=(10, 8))

# Точки
plt.scatter(*A, s=200, c='blue', marker='o', label='A(1,1)', zorder=3)
plt.scatter(*B, s=200, c='red', marker='o', label='B(3,4)', zorder=3)

# Euclidean (пряма лінія)
plt.plot([A[0], B[0]], [A[1], B[1]], 'g-', linewidth=3, alpha=0.7,
         label=f'Euclidean = {euclidean_distance(A, B):.2f}')

# Manhattan (ламана лінія)
plt.plot([A[0], B[0], B[0]], [A[1], A[1], B[1]], 'r-', linewidth=3, alpha=0.7,
         label=f'Manhattan = {manhattan_distance(A, B):.2f}')

# Альтернативний Manhattan шлях
plt.plot([A[0], A[0], B[0]], [A[1], B[1], B[1]], 'orange', linewidth=2, 
         alpha=0.5, linestyle='--', label='Alternative path')

plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Euclidean vs Manhattan Distance', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Порівняння в різних вимірах
print("Порівняння відстаней:")
print("="*50)
dimensions = [2, 5, 10, 50, 100]

for dim in dimensions:
    x = np.random.randn(dim)
    y = np.random.randn(dim)
    
    euc = euclidean_distance(x, y)
    man = manhattan_distance(x, y)
    ratio = man / euc
    
    print(f"Dim={dim:3d}: Euclidean={euc:6.2f}, Manhattan={man:6.2f}, "
          f"Ratio={ratio:.2f}")
```

## Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Менш чутлива до викидів | ❌ Не має геометричної прямої |
| ✅ Підходить для grid-based проблем | ❌ Може давати багато однакових відстаней |
| ✅ Швидша в обчисленні (без sqrt) | ❌ Менш інтуїтивна |
| ✅ Працює в high dimensions | |

## Коли використовувати?

- ✅ Grid-based problems (шахи, лабіринти)
- ✅ Коли рух можливий тільки вздовж осей
- ✅ High-dimensional data
- ✅ Коли є викиди

---

# 3. Minkowski Distance (Узагальнення)

## Формула

$$d_{\text{Minkowski}}(\mathbf{x}, \mathbf{y}) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

Або:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_p$$

## Інтуїція

**Minkowski Distance** — це **узагальнення** через параметр $p$:

```
p = 1  → Manhattan Distance
p = 2  → Euclidean Distance
p = ∞  → Chebyshev Distance
```

## Код

```python
def minkowski_distance(x, y, p):
    """Обчислити Minkowski distance"""
    return np.sum(np.abs(x - y) ** p) ** (1/p)

# Приклади
x = np.array([1, 1])
y = np.array([3, 4])

for p in [1, 2, 3, 5, 10]:
    dist = minkowski_distance(x, y, p)
    print(f"p={p:2d}: {dist:.4f}")

# Через scipy
from scipy.spatial.distance import minkowski
for p in [1, 2, 3]:
    dist = minkowski(x, y, p)
    print(f"Scipy p={p}: {dist:.4f}")
```

## Візуалізація Unit Circles

```python
import matplotlib.pyplot as plt
import numpy as np

# Unit circles для різних p
theta = np.linspace(0, 2*np.pi, 1000)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

p_values = [0.5, 1, 1.5, 2, 3, 10]

for idx, p in enumerate(p_values):
    # Обчислити unit circle для Minkowski з параметром p
    if p == np.inf:
        # Chebyshev
        x = np.sign(np.cos(theta))
        y = np.sign(np.sin(theta))
    else:
        # Для інших p
        t = np.linspace(0, 2*np.pi, 1000)
        x = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2/p)
        y = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2/p)
    
    axes[idx].plot(x, y, linewidth=2)
    axes[idx].set_xlim(-1.5, 1.5)
    axes[idx].set_ylim(-1.5, 1.5)
    axes[idx].set_aspect('equal')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_title(f'p = {p}', fontsize=12, fontweight='bold')
    axes[idx].axhline(y=0, color='k', linewidth=0.5)
    axes[idx].axvline(x=0, color='k', linewidth=0.5)

plt.suptitle('Unit Circles for Minkowski Distance (Different p)', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

# 4. Chebyshev Distance

## Формула

$$d_{\text{Chebyshev}}(\mathbf{x}, \mathbf{y}) = \max_{i} |x_i - y_i|$$

Це Minkowski distance з $p = \infty$:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_\infty$$

## Інтуїція

**Chebyshev Distance** — це **максимальна різниця** серед всіх координат.

```
A = (1, 1)
B = (3, 4)

Різниці:
  x: |3 - 1| = 2
  y: |4 - 1| = 3

d = max(2, 3) = 3

Візуально — це найбільший "крок" вздовж однієї осі.
```

## Код

```python
def chebyshev_distance(x, y):
    """Обчислити Chebyshev distance"""
    return np.max(np.abs(x - y))

# Приклад
x = np.array([1, 1])
y = np.array([3, 4])

dist = chebyshev_distance(x, y)
print(f"Chebyshev distance: {dist:.4f}")

# Через scipy
from scipy.spatial.distance import chebyshev
dist_scipy = chebyshev(x, y)
print(f"Scipy distance: {dist_scipy:.4f}")
```

## Порівняння всіх Minkowski метрик

```python
import matplotlib.pyplot as plt
import numpy as np

# Точки
A = np.array([1, 1])
B = np.array([3, 4])

# Обчислити відстані
distances = {
    'Manhattan (p=1)': manhattan_distance(A, B),
    'Euclidean (p=2)': euclidean_distance(A, B),
    'Minkowski (p=3)': minkowski_distance(A, B, 3),
    'Chebyshev (p=∞)': chebyshev_distance(A, B)
}

# Візуалізація
plt.figure(figsize=(12, 6))

names = list(distances.keys())
values = list(distances.values())
colors = ['red', 'green', 'blue', 'purple']

bars = plt.bar(names, values, color=colors, alpha=0.7)

for i, (name, value) in enumerate(zip(names, values)):
    plt.text(i, value + 0.1, f'{value:.2f}', 
            ha='center', fontsize=11, fontweight='bold')

plt.ylabel('Distance', fontsize=12)
plt.title('Comparison of Minkowski Family Distances', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, max(values) * 1.2)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()

print("Distance Comparison:")
print("="*50)
for name, value in distances.items():
    print(f"{name:20s}: {value:.4f}")
```

## Коли використовувати?

- ✅ Chess/checkers (король рухається)
- ✅ Коли важлива тільки найбільша різниця
- ✅ Warehouse logistics
- ❌ Більшість ML задач (рідко використовується)

---

# 5. Cosine Distance / Similarity

## Формула

**Cosine Similarity:**

$$\text{cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}}$$

**Cosine Distance:**

$$d_{\text{cosine}}(\mathbf{x}, \mathbf{y}) = 1 - \text{cos}(\mathbf{x}, \mathbf{y})$$

## Інтуїція

**Cosine Similarity** вимірює **кут між векторами**, не враховуючи їх довжину.

```
2D приклад:

    y
    |
    |    B(2,4)
    |     *
    |    /
    |   / θ
    |  /
    | *-----> A(4,2)
    |
    +------------- x

cos(θ) = A·B / (|A| |B|)

Якщо θ = 0°   → cos = 1  (ідентичний напрямок)
Якщо θ = 90°  → cos = 0  (ортогональні)
Якщо θ = 180° → cos = -1 (протилежний напрямок)
```

## Код

```python
def cosine_similarity(x, y):
    """Обчислити Cosine Similarity"""
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)

def cosine_distance(x, y):
    """Обчислити Cosine Distance"""
    return 1 - cosine_similarity(x, y)

# Приклад
x = np.array([1, 2, 3])
y = np.array([2, 4, 6])  # У 2 рази більше — той самий напрямок!

sim = cosine_similarity(x, y)
dist = cosine_distance(x, y)

print(f"Cosine Similarity: {sim:.4f}")
print(f"Cosine Distance: {dist:.4f}")

# Через sklearn
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
sim_sklearn = sklearn_cosine([x], [y])[0, 0]
print(f"Sklearn Similarity: {sim_sklearn:.4f}")

# Через scipy
from scipy.spatial.distance import cosine
dist_scipy = cosine(x, y)
print(f"Scipy Distance: {dist_scipy:.4f}")
```

## Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Вектори
A = np.array([4, 2])
B = np.array([2, 4])
C = np.array([8, 4])  # 2*A (той самий напрямок)

origin = np.array([0, 0])

# Обчислити кути
def angle_between(v1, v2):
    cos_angle = cosine_similarity(v1, v2)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)

angle_AB = angle_between(A, B)
angle_AC = angle_between(A, C)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Різні напрямки
axes[0].quiver(*origin, *A, angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.015, label='A(4,2)')
axes[0].quiver(*origin, *B, angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.015, label='B(2,4)')

# Кут
from matplotlib.patches import Arc
arc = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=angle_AB, 
         color='green', linewidth=2)
axes[0].add_patch(arc)
axes[0].text(1, 0.5, f'θ={angle_AB:.1f}°', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

axes[0].set_xlim(-1, 5)
axes[0].set_ylim(-1, 5)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].set_title(f'Different Directions\ncos(θ)={cosine_similarity(A, B):.3f}', 
                 fontsize=12, fontweight='bold')

# Subplot 2: Той самий напрямок
axes[1].quiver(*origin, *A, angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.015, label='A(4,2)')
axes[1].quiver(*origin, *C, angles='xy', scale_units='xy', scale=1, 
              color='purple', width=0.015, label='C(8,4) = 2*A')

axes[1].text(6, 3, f'θ={angle_AC:.1f}°', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

axes[1].set_xlim(-1, 9)
axes[1].set_ylim(-1, 5)
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)
axes[1].set_title(f'Same Direction\ncos(θ)={cosine_similarity(A, C):.3f}', 
                 fontsize=12, fontweight='bold')

plt.suptitle('Cosine Similarity: Measures Angle, Not Magnitude', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Ключова відмінність від Euclidean

```python
# Приклад
doc1 = np.array([1, 1, 0])
doc2 = np.array([10, 10, 0])  # Той самий напрямок, але у 10 разів більше
doc3 = np.array([0, 1, 1])    # Інший напрямок

print("Порівняння метрик:")
print("="*60)

# Euclidean
euc_12 = euclidean_distance(doc1, doc2)
euc_13 = euclidean_distance(doc1, doc3)

print(f"Euclidean:")
print(f"  doc1 vs doc2: {euc_12:.4f}")
print(f"  doc1 vs doc3: {euc_13:.4f}")
print(f"  → doc3 ближче за Euclidean!")

# Cosine
cos_12 = cosine_similarity(doc1, doc2)
cos_13 = cosine_similarity(doc1, doc3)

print(f"\nCosine Similarity:")
print(f"  doc1 vs doc2: {cos_12:.4f}")
print(f"  doc1 vs doc3: {cos_13:.4f}")
print(f"  → doc2 більш схожий за Cosine!")

print("\nВисновок:")
print("Euclidean чутлива до magnitude (довжини)")
print("Cosine чутлива тільки до direction (напрямку)")
```

## Переваги та недоліки

| Переваги | Недоліки |
|----------|----------|
| ✅ Не залежить від magnitude | ❌ Втрачає інформацію про magnitude |
| ✅ Підходить для text/sparse data | ❌ Не метрика (не задовольняє triangle inequality) |
| ✅ Нормалізована [-1, 1] | ❌ Не підходить для negative values інтерпретації |
| ✅ High-dimensional data | |

## Коли використовувати?

- ✅ **Text mining** (TF-IDF vectors)
- ✅ **Recommender systems** (user preferences)
- ✅ **Document similarity**
- ✅ **Image retrieval** (feature vectors)
- ✅ Коли важливий напрямок, не magnitude
- ❌ Geographical data (використовуй Euclidean)

---

# 6. Hamming Distance

## Формула

$$d_{\text{Hamming}}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} \mathbb{1}(x_i \neq y_i)$$

де $\mathbb{1}$ — індикаторна функція (1 якщо true, 0 якщо false).

## Інтуїція

**Hamming Distance** — це **кількість позицій**, де символи різні. Для **binary** або **categorical** даних.

```
Приклад:

x = [1, 0, 1, 1, 0]
y = [1, 1, 1, 0, 0]
     ✓  ✗  ✓  ✗  ✓

Hamming distance = 2 (різні в 2 позиціях)
```

## Код

```python
def hamming_distance(x, y):
    """Обчислити Hamming distance"""
    return np.sum(x != y)

# Приклад 1: Binary
x = np.array([1, 0, 1, 1, 0])
y = np.array([1, 1, 1, 0, 0])

dist = hamming_distance(x, y)
print(f"Hamming distance: {dist}")

# Через scipy
from scipy.spatial.distance import hamming
# Scipy повертає normalized (частку різних)
dist_scipy = hamming(x, y)
print(f"Scipy (normalized): {dist_scipy:.4f}")
print(f"Scipy (count): {int(dist_scipy * len(x))}")

# Приклад 2: Strings
s1 = "karolin"
s2 = "kathrin"

# Перетворити в числа
x = np.array([ord(c) for c in s1])
y = np.array([ord(c) for c in s2])

dist = hamming_distance(x, y)
print(f"\nHamming distance ('{s1}' vs '{s2}'): {dist}")
```

## Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Binary vectors
x = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y = np.array([1, 1, 1, 0, 0, 1, 1, 0])

# Візуалізація
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Vector x
axes[0].imshow([x], cmap='gray_r', aspect='auto')
axes[0].set_yticks([0])
axes[0].set_yticklabels(['x'])
axes[0].set_xticks(range(len(x)))
axes[0].set_title('Vector x', fontsize=12, fontweight='bold')

# Додати значення
for i, val in enumerate(x):
    axes[0].text(i, 0, str(val), ha='center', va='center', 
                fontsize=14, fontweight='bold', color='red' if val == 1 else 'blue')

# Vector y
axes[1].imshow([y], cmap='gray_r', aspect='auto')
axes[1].set_yticks([0])
axes[1].set_yticklabels(['y'])
axes[1].set_xticks(range(len(y)))
axes[1].set_title('Vector y', fontsize=12, fontweight='bold')

for i, val in enumerate(y):
    axes[1].text(i, 0, str(val), ha='center', va='center', 
                fontsize=14, fontweight='bold', color='red' if val == 1 else 'blue')

# Differences
diff = x != y
axes[2].imshow([diff], cmap='RdYlGn_r', aspect='auto')
axes[2].set_yticks([0])
axes[2].set_yticklabels(['Diff'])
axes[2].set_xticks(range(len(diff)))
axes[2].set_title(f'Differences (Hamming Distance = {diff.sum()})', 
                 fontsize=12, fontweight='bold')

for i, d in enumerate(diff):
    symbol = '✗' if d else '✓'
    color = 'red' if d else 'green'
    axes[2].text(i, 0, symbol, ha='center', va='center', 
                fontsize=16, fontweight='bold', color=color)

plt.tight_layout()
plt.show()
```

## Коли використовувати?

- ✅ **Binary data** (0/1 features)
- ✅ **Error detection** (коди Хеммінга)
- ✅ **Categorical variables**
- ✅ **DNA sequences** (ATCG)
- ✅ **Image hashing**
- ❌ Continuous data (використовуй Euclidean)

---

# 7. Mahalanobis Distance

## Формула

$$d_{\text{Mahalanobis}}(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^T \mathbf{S}^{-1} (\mathbf{x} - \mathbf{y})}$$

де $\mathbf{S}$ — covariance matrix.

## Інтуїція

**Mahalanobis Distance** враховує **кореляції між ознаками** та **різні масштаби**. Це Euclidean distance у трансформованому просторі.

```
Якщо ознаки некорельовані та мають однаковий масштаб:
  Mahalanobis = Euclidean

Якщо є кореляції або різні масштаби:
  Mahalanobis враховує це!
```

## Код

```python
def mahalanobis_distance(x, y, cov_matrix):
    """
    Обчислити Mahalanobis distance
    
    Parameters:
    -----------
    x, y : array-like
        Вектори
    cov_matrix : array-like
        Covariance matrix
        
    Returns:
    --------
    float : відстань
    """
    diff = x - y
    inv_cov = np.linalg.inv(cov_matrix)
    return np.sqrt(diff @ inv_cov @ diff)

# Приклад
np.random.seed(42)

# Згенерувати дані з кореляціями
mean = [0, 0]
cov = [[2, 1],   # Кореляція між ознаками
       [1, 2]]

data = np.random.multivariate_normal(mean, cov, 1000)

# Дві точки
x = np.array([2, 2])
y = np.array([0, 0])

# Covariance matrix
cov_matrix = np.cov(data.T)

# Відстані
euclidean_dist = euclidean_distance(x, y)
mahal_dist = mahalanobis_distance(x, y, cov_matrix)

print(f"Euclidean distance: {euclidean_dist:.4f}")
print(f"Mahalanobis distance: {mahal_dist:.4f}")

# Через scipy
from scipy.spatial.distance import mahalanobis as scipy_mahal
mahal_scipy = scipy_mahal(x, y, np.linalg.inv(cov_matrix))
print(f"Scipy Mahalanobis: {mahal_scipy:.4f}")
```

## Візуалізація

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# Дані
np.random.seed(42)
mean = [0, 0]
cov = [[2, 1.5],
       [1.5, 2]]

data = np.random.multivariate_normal(mean, cov, 1000)

# Точки для порівняння
origin = np.array([0, 0])
point1 = np.array([2, 0])
point2 = np.array([1.5, 1.5])

# Відстані
euc1 = euclidean_distance(origin, point1)
euc2 = euclidean_distance(origin, point2)
mahal1 = mahalanobis_distance(origin, point1, cov)
mahal2 = mahalanobis_distance(origin, point2, cov)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Euclidean
axes[0].scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
axes[0].scatter(*origin, c='red', s=200, marker='*', zorder=5, label='Origin')
axes[0].scatter(*point1, c='blue', s=200, marker='o', zorder=5, 
               label=f'P1: Euc={euc1:.2f}')
axes[0].scatter(*point2, c='green', s=200, marker='s', zorder=5, 
               label=f'P2: Euc={euc2:.2f}')

# Euclidean circles
circle1 = plt.Circle(origin, euc1, fill=False, color='blue', linewidth=2, linestyle='--')
circle2 = plt.Circle(origin, euc2, fill=False, color='green', linewidth=2, linestyle='--')
axes[0].add_patch(circle1)
axes[0].add_patch(circle2)

axes[0].set_xlim(-4, 4)
axes[0].set_ylim(-4, 4)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)
axes[0].set_title('Euclidean Distance\n(Circles)', fontsize=12, fontweight='bold')

# Subplot 2: Mahalanobis
axes[1].scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
axes[1].scatter(*origin, c='red', s=200, marker='*', zorder=5, label='Origin')
axes[1].scatter(*point1, c='blue', s=200, marker='o', zorder=5, 
               label=f'P1: Mahal={mahal1:.2f}')
axes[1].scatter(*point2, c='green', s=200, marker='s', zorder=5, 
               label=f'P2: Mahal={mahal2:.2f}')

# Mahalanobis ellipses
eigenvalues, eigenvectors = np.linalg.eig(cov)

for distance in [mahal1, mahal2]:
    width = 2 * distance * np.sqrt(eigenvalues[0])
    height = 2 * distance * np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    ellipse = Ellipse(origin, width, height, angle=angle, 
                     fill=False, linewidth=2, linestyle='--',
                     color='blue' if distance == mahal1 else 'green')
    axes[1].add_patch(ellipse)

axes[1].set_xlim(-4, 4)
axes[1].set_ylim(-4, 4)
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].set_title('Mahalanobis Distance\n(Ellipses - accounts for correlation)', 
                 fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nPoint 1: Euclidean={euc1:.2f}, Mahalanobis={mahal1:.2f}")
print(f"Point 2: Euclidean={euc2:.2f}, Mahalanobis={mahal2:.2f}")
print(f"\nЗа Euclidean: точки на однаковій відстані")
print(f"За Mahalanobis: враховується кореляція даних")
```

## Коли використовувати?

- ✅ **Anomaly detection** (викиди)
- ✅ Коли ознаки корельовані
- ✅ Коли ознаки мають різні масштаби
- ✅ **Multivariate statistics**
- ❌ Малі датасети (covariance нестабільна)
- ❌ High dimensions (invertible covariance)

---

# Інші метрики

## Jaccard Distance

Для **множин**.

$$J(\mathbf{A}, \mathbf{B}) = \frac{|\mathbf{A} \cap \mathbf{B}|}{|\mathbf{A} \cup \mathbf{B}|}$$

$$d_{\text{Jaccard}} = 1 - J$$

```python
from scipy.spatial.distance import jaccard

# Binary vectors
x = np.array([1, 1, 0, 0, 1])
y = np.array([1, 0, 0, 1, 1])

dist = jaccard(x, y)
print(f"Jaccard distance: {dist:.4f}")

# Для множин
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

intersection = len(set1 & set2)
union = len(set1 | set2)
jaccard_sim = intersection / union
jaccard_dist = 1 - jaccard_sim

print(f"Jaccard similarity: {jaccard_sim:.4f}")
print(f"Jaccard distance: {jaccard_dist:.4f}")
```

## Haversine Distance

Для **географічних координат** (lat/lon).

```python
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Обчислити відстань між двома точками на Землі
    
    Returns: відстань в кілометрах
    """
    R = 6371  # Радіус Землі в км
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

# Приклад: Київ → Львів
kiev = (50.4501, 30.5234)
lviv = (49.8397, 24.0297)

dist = haversine_distance(*kiev, *lviv)
print(f"Відстань Київ-Львів: {dist:.2f} км")
```

## Edit Distance (Levenshtein)

Для **рядків** (strings).

```python
def levenshtein_distance(s1, s2):
    """Обчислити Edit Distance між двома рядками"""
    m, n = len(s1), len(s2)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Ініціалізація
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Заповнити таблицю
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # Deletion
                    dp[i][j-1],     # Insertion
                    dp[i-1][j-1]    # Substitution
                )
    
    return dp[m][n]

# Приклад
s1 = "kitten"
s2 = "sitting"

dist = levenshtein_distance(s1, s2)
print(f"Edit distance ('{s1}' → '{s2}'): {dist}")

# Через python-Levenshtein
try:
    import Levenshtein
    dist_lib = Levenshtein.distance(s1, s2)
    print(f"Library distance: {dist_lib}")
except ImportError:
    print("Install: pip install python-Levenshtein")
```

---

## Порівняльна таблиця

| Метрика | Формула | Дані | Complexity | Коли використовувати |
|---------|---------|------|------------|---------------------|
| **Euclidean** | $\|\|\mathbf{x}-\mathbf{y}\|\|_2$ | Continuous | O(n) | Geographical, general |
| **Manhattan** | $\|\|\mathbf{x}-\mathbf{y}\|\|_1$ | Continuous | O(n) | Grid problems, high-dim |
| **Cosine** | $1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\|\mathbf{x}\|\| \|\|\mathbf{y}\|\|}$ | Continuous | O(n) | Text, sparse data |
| **Hamming** | $\sum \mathbb{1}(x_i \neq y_i)$ | Binary/Categorical | O(n) | Binary data, DNA |
| **Mahalanobis** | $\sqrt{(\mathbf{x}-\mathbf{y})^T\mathbf{S}^{-1}(\mathbf{x}-\mathbf{y})}$ | Continuous | O(n³) | Correlated features |
| **Jaccard** | $1 - \frac{\|A \cap B\|}{\|A \cup B\|}$ | Sets | O(n) | Sets, recommenders |
| **Levenshtein** | Edit operations | Strings | O(mn) | Text similarity |

---

## Практичні поради 💡

### 1. Завжди нормалізуй дані

```python
from sklearn.preprocessing import StandardScaler

# ❌ ПОГАНО: різні масштаби
X = np.array([[1, 1000],
              [2, 2000],
              [3, 1500]])

# Euclidean буде dominated другою ознакою!
dist = euclidean_distance(X[0], X[1])
print(f"Without scaling: {dist:.2f}")  # ~1000 (dominated by 2nd feature)

# ✅ ДОБРЕ: нормалізація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dist_scaled = euclidean_distance(X_scaled[0], X_scaled[1])
print(f"With scaling: {dist_scaled:.2f}")  # Збалансовано
```

### 2. Вибір метрики залежить від даних

```python
# Continuous numeric → Euclidean або Manhattan
# Text/Sparse → Cosine
# Binary → Hamming
# Sets → Jaccard
# Geographic → Haversine
# Strings → Levenshtein
```

### 3. Використовуй scipy/sklearn

```python
from scipy.spatial.distance import pdist, squareform

# Матриця відстаней для багатьох точок
X = np.random.randn(5, 3)

# Euclidean
dist_matrix = squareform(pdist(X, metric='euclidean'))
print("Distance matrix:")
print(dist_matrix)

# Або через sklearn
from sklearn.metrics.pairwise import pairwise_distances

dist_matrix_sklearn = pairwise_distances(X, metric='euclidean')
```

### 4. KNN: вибір метрики

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Тестуємо різні метрики
metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine']

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    scores = cross_val_score(knn, X, y, cv=5)
    print(f"{metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## Реальний приклад: KNN з різними метриками

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

print("="*70)
print("KNN WITH DIFFERENT DISTANCE METRICS")
print("="*70)

# Генерація даних
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                          n_redundant=0, n_clusters_per_class=1,
                          random_state=42)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Різні метрики
metrics = {
    'Euclidean': 'euclidean',
    'Manhattan': 'manhattan',
    'Chebyshev': 'chebyshev',
    'Cosine': 'cosine'
}

results = []

for name, metric in metrics.items():
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        'Metric': name,
        'Accuracy': accuracy
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")

# Візуалізація
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (name, metric) in enumerate(metrics.items()):
    # Модель
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_scaled, y_train)
    
    # Створити mesh
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Передбачення
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Візуалізація decision boundary
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[idx].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                     c=y_train, cmap='RdYlBu', edgecolors='k', s=50, alpha=0.7)
    axes[idx].set_title(f'{name}\nAccuracy: {results[idx]["Accuracy"]:.4f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Порівняння
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

import pandas as pd
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Accuracy', ascending=False)
print(df_results.to_string(index=False))

print("="*70)
```

---

## Пов'язані теми

- [[08_Similarity_Measures]] — метрики схожості (complement)
- [[KNN]] — основне застосування distance metrics
- [[K_Means]] — використання Euclidean distance
- [[Anomaly_Detection]] — Mahalanobis distance
- [[Text_Mining]] — Cosine distance для TF-IDF

## Ресурси

- [Scipy Distance Metrics](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
- [Sklearn Metrics](https://scikit-learn.org/stable/modules/metrics.html)
- [Distance Metrics Tutorial](https://machinelearningmastery.com/distance-measures-for-machine-learning/)

---

## Ключові висновки

> Distance Metrics вимірюють несхожість між об'єктами. Вибір правильної метрики залежить від типу даних та задачі.

**Основні метрики:**
- **Euclidean** — пряма лінія (default для continuous)
- **Manhattan** — рух вздовж осей (grid problems)
- **Cosine** — кут між векторами (text, direction важливіший за magnitude)
- **Hamming** — кількість різних позицій (binary/categorical)
- **Mahalanobis** — враховує кореляції (anomaly detection)

**Вибір метрики:**
```
Continuous numeric    → Euclidean, Manhattan
Text / Sparse data    → Cosine
Binary data           → Hamming, Jaccard
Geographic coords     → Haversine
Strings               → Levenshtein
Correlated features   → Mahalanobis
```

**Важливо:**
- Завжди **нормалізуй** дані перед обчисленням відстані
- Різні метрики для різних **типів даних**
- Euclidean чутлива до **масштабу**
- Cosine **не залежить** від magnitude

**Формули для запам'ятовування:**
```
Euclidean:  √Σ(xᵢ - yᵢ)²
Manhattan:  Σ|xᵢ - yᵢ|
Cosine:     1 - (x·y)/(||x|| ||y||)
Hamming:    Σ 𝟙(xᵢ ≠ yᵢ)
```

---

#ml #distance-metrics #euclidean #manhattan #cosine #hamming #mahalanobis #knn #clustering
