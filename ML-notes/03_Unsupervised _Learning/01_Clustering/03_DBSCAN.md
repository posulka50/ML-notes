# DBSCAN (Density-Based Spatial Clustering)

## Що це?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** — це алгоритм unsupervised learning для кластеризації на основі **щільності**, який може знаходити кластери **довільної форми** та автоматично визначати **outliers**.

**Головна ідея:** кластер — це область з високою щільністю точок, оточена областями з низькою щільністю. Точки, які не належать жодному щільному региону, вважаються шумом (outliers).

## Навіщо потрібен?

- 🎯 **Складна форма кластерів** — не тільки круглі, а будь-які
- 🔍 **Автоматичне виявлення outliers** — шум виявляється природно
- 📊 **Не потрібно знати K** — кількість кластерів визначається автоматично
- 🗺️ **Географічні дані** — групування локацій з неоднорідною щільністю
- 🛡️ **Робастність до шуму** — добре працює з зашумленими даними
- 🌌 **Кластери різного розміру** — не вимагає однакової величини

## Коли використовувати?

**Потрібно:**
- **Не знаємо кількість кластерів** — DBSCAN знаходить сам
- Кластери **складної форми** — S-подібні, кільця, довгі
- Кластери **різного розміру/щільності**
- Багато **outliers** у даних
- **Географічні/просторові дані**
- Потрібна **робастність до шуму**

**Не потрібно:**
- Кластери **дуже різної щільності** → OPTICS, HDBSCAN
- **Високорозмірні дані** (>10-20 features) → curse of dimensionality
- Потрібна **швидкість** на великих даних → K-Means
- **Сферичні кластери однакового розміру** → K-Means простіший

---

## Основні концепції

### 1. Параметри

**DBSCAN має 2 ключові параметри:**

**ε (epsilon)** — **радіус околиці**
- Максимальна відстань між двома точками, щоб вони вважалися сусідами
- Визначає "наскільки далеко дивитися"

**MinPts (min_samples)** — **мінімальна кількість точок**
- Мінімальна кількість точок в околиці ε для формування кластера
- Зазвичай MinPts ≥ dimensions + 1

### 2. Типи точок

**Core Point (ядрова точка):**

- Має ≥ MinPts сусідів в радіусі ε (включаючи себе)
- Формує "ядро" кластера

**Border Point (прикордонна точка):**

- Має < MinPts сусідів в радіусі ε
- Але знаходиться в околиці якоїсь core point
- Належить кластеру, але не може розширювати його

**Noise Point (шум/outlier):**

- Має < MinPts сусідів в радіусі ε
- НЕ знаходиться в околиці жодної core point
- Не належить жодному кластеру

### 3. Density Reachability

**Точка q досяжна з точки p** (density-reachable), якщо існує ланцюжок core points від p до q.

```
p → p₁ → p₂ → ... → q
(core) (core) (core)  (може бути border)
```

**Два типи досяжності:**

**Directly density-reachable:**
- q в ε-околиці core point p

**Density-reachable:**
- Існує ланцюжок directly density-reachable core points від p до q

---

## Як працює DBSCAN?

### Алгоритм

**Вхід:** дані $X$, параметри $\varepsilon$ (epsilon), MinPts

**1. Початок:**
   - Всі точки позначити як невідвідані
   - Лічильник кластерів = 0

**2. Для кожної невідвіданої точки $p$:**

   **a) Знайти всіх сусідів** в радіусі $\varepsilon$:
   $$N_\varepsilon(p) = \{q \in X : \text{dist}(p, q) \leq \varepsilon\}$$

   **b) Якщо $|N_\varepsilon(p)| < \text{MinPts}$:**
   - Поки що позначити як **noise**
   - (Може пізніше стати border point)

   **c) Якщо $|N_\varepsilon(p)| \geq \text{MinPts}$:**
   - $p$ — **core point**
   - Створити новий кластер $C$
   - Додати $p$ до $C$
   - **Розширити кластер:**
     - Для кожного сусіда $q$ з $N_\varepsilon(p)$:
       - Якщо $q$ noise → змінити на border point кластера $C$
       - Якщо $q$ невідвіданий:
         - Додати до $C$
         - Знайти $N_\varepsilon(q)$
         - Якщо $|N_\varepsilon(q)| \geq \text{MinPts}$ → додати сусідів до черги

**3. Повторювати до відвідування всіх точок**

### Псевдокод

```
DBSCAN(X, ε, MinPts):
    C = 0  # Лічильник кластерів
    for each point p in X:
        if p.visited:
            continue
        p.visited = True
        
        NeighborPts = regionQuery(p, ε)
        
        if |NeighborPts| < MinPts:
            p.label = NOISE
        else:
            C = C + 1
            expandCluster(p, NeighborPts, C, ε, MinPts)
    
    return labels

expandCluster(p, NeighborPts, C, ε, MinPts):
    p.label = C
    
    for each point q in NeighborPts:
        if not q.visited:
            q.visited = True
            NeighborPts' = regionQuery(q, ε)
            
            if |NeighborPts'| >= MinPts:
                NeighborPts = NeighborPts ∪ NeighborPts'
        
        if q.label == UNDEFINED or q.label == NOISE:
            q.label = C

regionQuery(p, ε):
    return {q ∈ X : dist(p, q) ≤ ε}
```

---

## Візуалізація роботи алгоритму

### Крок 1: Початкові дані

```
    y
    |  •   • •
    |    •   •  •
    | •  • •  •
    |•  •  •
    |     ◊      •
    |_________ x
    
Всі точки невідвідані
```

### Крок 2: Вибираємо першу точку

```
ε = 0.5, MinPts = 3

    y
    |  •   • •
    |    •   •  •
    | •  ⊕ •  •   ← поточна точка p
    |•  •  •
    |     ◊      •
    |_________ x
    
Околиця p (в радіусі ε):
• • ⊕ • • → 5 точок ≥ MinPts=3
⇒ p — CORE POINT ✓
⇒ Створити кластер 1
```

### Крок 3: Розширюємо кластер

```
    y
    |  •   • •
    |    •   •  •
    | 🔴  🔴 🔴  🔴   ← кластер 1
    |🔴  🔴  🔴
    |     ◊      •
    |_________ x
    
Рекурсивно додаємо всі density-reachable точки
```

### Крок 4: Наступна невідвідана точка

```
    y
    |  ⊕   • •      ← нова поточна точка
    |    •   •  •
    | 🔴  🔴 🔴  🔴
    |🔴  🔴  🔴
    |     ◊      •
    |_________ x
    
Околиця: 3 точки ≥ MinPts=3
⇒ CORE POINT
⇒ Створити кластер 2
```

### Крок 5: Фінальний результат

```
    y
    |  🔵   🔵 🔵    ← кластер 2
    |    🔵   🔵  🔵
    | 🔴  🔴 🔴  🔴   ← кластер 1
    |🔴  🔴  🔴
    |     ⚫      🟢   ← noise    ← кластер 3
    |_________ x
    
🔴 Кластер 1
🔵 Кластер 2  
🟢 Кластер 3
⚫ Noise (outlier)
```

---

## Простий приклад: Географічне групування

### Дані

Координати ресторанів у місті:

| ID | Latitude | Longitude | Район |
|----|----------|-----------|-------|
| 1 | 50.45 | 30.52 | Центр |
| 2 | 50.46 | 30.53 | Центр |
| 3 | 50.44 | 30.51 | Центр |
| 4 | 50.40 | 30.60 | Схід |
| 5 | 50.41 | 30.61 | Схід |
| 6 | 50.50 | 30.45 | Захід |
| 7 | 50.30 | 30.70 | Outlier |

### Параметри

- **ε = 0.05** (≈5 км в lat/lon)
- **MinPts = 2**

### Результат DBSCAN

**Кластер 1 (Центр):** {1, 2, 3}
- Щільна група ресторанів у центрі

**Кластер 2 (Схід):** {4, 5}
- Група на сході міста

**Noise:** {6, 7}
- Окремі ресторани, не формують кластер

### Інтерпретація

- **Центр:** густонаселена зона, багато ресторанів поряд
- **Схід:** менша група
- **Захід/Outlier:** поодинокі локації

---

## Математика

### Відстань (за замовчуванням — Euclidean)

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

**Для 2D:**
$$d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$

### ε-околиця точки p

$$N_\varepsilon(p) = \{q \in X : d(p, q) \leq \varepsilon\}$$

**Приклад:**

Точка $p = [2, 3]$, $\varepsilon = 1.5$

Точки:
- $q_1 = [2.5, 3.5]$ → $d = \sqrt{0.25 + 0.25} = 0.71 \leq 1.5$ ✓
- $q_2 = [4, 5]$ → $d = \sqrt{4 + 4} = 2.83 > 1.5$ ✗

Сусіди: $N_\varepsilon(p) = \{q_1\}$

### Умова Core Point

$$|N_\varepsilon(p)| \geq \text{MinPts}$$

Якщо в околиці ≥ MinPts точок → core point.

---

## Складний приклад: Аномалії в транзакціях

### Задача

Банк має транзакції клієнтів:
- **Amount** — сума транзакції
- **Time** — час доби (години)

**Мета:** знайти групи нормальних транзакцій та виявити аномалії.

### Дані (спрощено)

```python
Транзакції:
- Ранок (7-9): $20-50 (кава, сніданок)
- Обід (12-14): $50-100 (обід)
- Вечір (18-20): $100-200 (вечеря, покупки)
- АНОМАЛІЇ: $5000 о 3:00 (підозріла транзакція)
```

### DBSCAN

**Параметри:**
- ε = 2.0 (нормалізовані одиниці)
- MinPts = 5

**Результат:**

**Кластер 1:** Ранкові транзакції ($20-50, 7-9 год)
**Кластер 2:** Обідні транзакції ($50-100, 12-14 год)
**Кластер 3:** Вечірні транзакції ($100-200, 18-20 год)
**Noise:** Транзакція $5000 о 3:00 → **FRAUD ALERT!** 🚨

### Переваги підходу

- ✅ Автоматично виявляє аномалії
- ✅ Не потребує заздалегідь знати кількість типів транзакцій
- ✅ Робастний до нових типів шахрайства

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 1. Генерація даних (2 "півмісяці")
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# 2. DBSCAN
dbscan = DBSCAN(
    eps=0.3,           # Радіус ε
    min_samples=5,     # Мінімальна кількість точок
    metric='euclidean' # Метрика відстані
)

# 3. Навчання (fit_predict)
labels = dbscan.fit_predict(X)

# 4. Результати
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Кількість кластерів: {n_clusters}")
print(f"Кількість noise точок: {n_noise}")
print(f"Унікальні мітки: {set(labels)}")

# -1 означає NOISE!

# 5. Додаткова інформація
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

print(f"\nCore points: {np.sum(core_samples_mask)}")
print(f"Border points: {len(labels) - n_noise - np.sum(core_samples_mask)}")

# 6. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# До DBSCAN
axes[0].scatter(X[:, 0], X[:, 1], s=50, alpha=0.6)
axes[0].set_title('Before DBSCAN', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# Після DBSCAN
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noise: чорний колір з x-маркером
        col = [0, 0, 0, 1]
        marker = 'x'
        label = 'Noise'
    else:
        marker = 'o'
        label = f'Cluster {k}'
    
    class_member_mask = (labels == k)
    
    xy = X[class_member_mask & core_samples_mask]
    axes[1].scatter(xy[:, 0], xy[:, 1], s=100, marker=marker, 
                   c=[col], edgecolors='black', linewidths=1.5,
                   label=f'{label} (core)')
    
    xy = X[class_member_mask & ~core_samples_mask]
    axes[1].scatter(xy[:, 0], xy[:, 1], s=50, marker=marker,
                   c=[col], alpha=0.3, edgecolors='black', linewidths=0.5,
                   label=f'{label} (border)' if k != -1 else label)

axes[1].set_title(f'After DBSCAN (eps={dbscan.eps}, min_samples={dbscan.min_samples})',
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Повний приклад: Географічна кластеризація

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Генерація даних про ресторани
np.random.seed(42)

# Кластер 1: Центр міста
center1 = np.random.normal([50.45, 30.52], [0.01, 0.01], (50, 2))

# Кластер 2: Район на півночі
center2 = np.random.normal([50.50, 30.45], [0.008, 0.008], (30, 2))

# Кластер 3: Район на сході
center3 = np.random.normal([50.40, 30.60], [0.012, 0.012], (40, 2))

# Outliers
outliers = np.array([
    [50.35, 30.70],
    [50.55, 30.40],
    [50.38, 30.55],
    [50.48, 30.58]
])

# Об'єднати всі дані
X = np.vstack([center1, center2, center3, outliers])

df = pd.DataFrame(X, columns=['Latitude', 'Longitude'])
df['ID'] = range(len(df))

print("=== Dataset Info ===")
print(f"Total points: {len(df)}")
print(df.head())

# Візуалізація до кластеризації
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.scatter(df['Longitude'], df['Latitude'], s=50, alpha=0.6)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Restaurant Locations (Before DBSCAN)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# DBSCAN
# Не потрібен scaling для географічних даних в однакових одиницях,
# але можна для узагальненості
dbscan = DBSCAN(
    eps=0.02,          # ≈ 2км в lat/lon координатах
    min_samples=5,
    metric='euclidean'
)

labels = dbscan.fit_predict(df[['Latitude', 'Longitude']])
df['Cluster'] = labels

# Статистика
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("\n" + "="*60)
print("=== DBSCAN Results ===")
print("="*60)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Labels: {set(labels)}")

# Аналіз кластерів
for cluster in sorted(set(labels)):
    cluster_data = df[df['Cluster'] == cluster]
    if cluster == -1:
        print(f"\nNoise points: {len(cluster_data)}")
        print(cluster_data[['Latitude', 'Longitude']])
    else:
        print(f"\nCluster {cluster}: {len(cluster_data)} points")
        print(f"  Center: Lat={cluster_data['Latitude'].mean():.4f}, "
              f"Lon={cluster_data['Longitude'].mean():.4f}")
        print(f"  Spread: Lat±{cluster_data['Latitude'].std():.4f}, "
              f"Lon±{cluster_data['Longitude'].std():.4f}")

# Візуалізація після DBSCAN
plt.subplot(2, 1, 2)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Чорний для noise
        marker = 'x'
        label = 'Noise'
    else:
        marker = 'o'
        label = f'Cluster {k}'
    
    class_member_mask = (labels == k)
    
    # Core points
    xy = df.loc[class_member_mask & core_samples_mask, ['Longitude', 'Latitude']].values
    if len(xy) > 0:
        plt.scatter(xy[:, 0], xy[:, 1], s=100, marker=marker,
                   c=[col], edgecolors='black', linewidths=1.5,
                   label=f'{label} (core)')
    
    # Border points
    xy = df.loc[class_member_mask & ~core_samples_mask, ['Longitude', 'Latitude']].values
    if len(xy) > 0:
        plt.scatter(xy[:, 0], xy[:, 1], s=50, marker=marker,
                   c=[col], alpha=0.5, edgecolors='black', linewidths=0.5,
                   label=f'{label} (border)' if k != -1 else '')

plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title(f'After DBSCAN (eps={dbscan.eps}, min_samples={dbscan.min_samples})',
         fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Експорт результатів
df_clusters = df.groupby('Cluster').agg({
    'Latitude': ['mean', 'std', 'count'],
    'Longitude': ['mean', 'std']
}).round(4)

print("\n" + "="*60)
print("=== Cluster Summary ===")
print("="*60)
print(df_clusters)
```

### Порівняння з K-Means

```python
from sklearn.cluster import KMeans

# Дані з складною формою (2 півмісяці)
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Візуалізація порівняння
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
axes[0].scatter(kmeans.cluster_centers_[:, 0], 
               kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[0].set_title('K-Means (K=2)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# DBSCAN
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]
    
    class_member_mask = (dbscan_labels == k)
    xy = X[class_member_mask]
    axes[1].scatter(xy[:, 0], xy[:, 1], c=[col], s=50,
                   marker='x' if k == -1 else 'o')

axes[1].set_title('DBSCAN (eps=0.3, min_samples=5)', 
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nK-Means: Погано справляється з непрямокутними кластерами")
print("DBSCAN: Ідеально знаходить півмісяці! ✓")
```

---

## Вибір параметрів ε та MinPts

### Проблема

**DBSCAN дуже чутливий до параметрів!**

- ε занадто мале → багато noise, мало кластерів
- ε занадто велике → все один кластер
- MinPts занадто мале → багато маленьких кластерів
- MinPts занадто велике → багато noise

### 1. K-Distance Graph (рекомендовано ✓)

**Метод для вибору ε:**

1. Для кожної точки знайти відстань до k-го найближчого сусіда
2. Відсортувати ці відстані
3. Побудувати графік
4. Знайти "лікоть" (різке зростання) → це ε

```python
from sklearn.neighbors import NearestNeighbors

# MinPts = 5 (приклад)
min_samples = 5

# Знайти відстані до k-го сусіда
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

# Взяти відстань до останнього (k-го) сусіда
k_distances = distances[:, -1]
k_distances = np.sort(k_distances)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.xlabel('Points sorted by distance', fontsize=12)
plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph', fontsize=14, fontweight='bold')
plt.axhline(y=0.3, color='red', linestyle='--', 
            label='Suggested ε=0.3 (elbow)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Знайти "лікоть" автоматично (наближено)
# Точка найбільшої кривизни
diff = np.diff(k_distances)
diff2 = np.diff(diff)
elbow_idx = np.argmax(diff2) + 1
suggested_eps = k_distances[elbow_idx]

print(f"Suggested ε: {suggested_eps:.4f}")
```

**Інтерпретація графіка:**

```
Distance
    |            ╱╱╱╱  ← різке зростання (outliers)
    |          ╱
    |        ╱  ← "лікоть" = оптимальний ε
    |      ╱
    |    ╱
    |  ╱
    |╱_____________ Point index
```

### 2. MinPts (емпіричне правило)

**Рекомендації:**

$$\text{MinPts} \geq \text{dimensions} + 1$$

**Практичні значення:**

| Розмірність | MinPts |
|-------------|--------|
| 2D | 4-5 |
| 3D | 5-6 |
| Високорозмірні | 2 × dimensions |

**Загальне правило:**
- **Більше шуму** → збільшити MinPts
- **Маленькі кластери** → зменшити MinPts

### 3. Grid Search (якщо є ground truth)

```python
from sklearn.metrics import silhouette_score

# Сітка параметрів
eps_values = np.arange(0.1, 1.0, 0.1)
min_samples_values = [3, 5, 7, 10]

best_score = -1
best_params = {}

results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Пропустити, якщо всі точки noise або один кластер
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            continue
        
        # Silhouette score (тільки для non-noise точок)
        mask = labels != -1
        if np.sum(mask) > 0:
            score = silhouette_score(X[mask], labels[mask])
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': np.sum(~mask),
                'silhouette': score
            })
            
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

# Результати
results_df = pd.DataFrame(results)
print("=== Top 5 Parameter Combinations ===")
print(results_df.nlargest(5, 'silhouette'))

print(f"\nBest parameters: {best_params}")
print(f"Best silhouette score: {best_score:.4f}")
```

### 4. Domain Knowledge

**Географічні дані:**
- ε = відстань, на якій точки вважаються "близькими"
- Наприклад: 500м для пішоходів, 5км для авто

**Часові дані:**
- ε = часовий інтервал (години, дні)

**Бізнес-правила:**
- MinPts = мінімальний розмір значущої групи
- Наприклад: мінімум 10 клієнтів для сегмента

---

## Метрики відстані

### За замовчуванням: Euclidean

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

### Інші метрики

```python
from sklearn.cluster import DBSCAN

# Manhattan distance
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='manhattan')

# Cosine distance
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')

# Власна метрика
def custom_distance(x, y):
    return np.sum(np.abs(x - y))

dbscan = DBSCAN(eps=0.5, min_samples=5, metric=custom_distance)
```

**Коли використовувати різні метрики:**

| Метрика | Використання |
|---------|--------------|
| **Euclidean** | За замовчуванням, географічні дані |
| **Manhattan** | Grid-based дані (міські квартали) |
| **Cosine** | Текстові дані (TF-IDF vectors) |
| **Haversine** | Географічні координати (lat/lon на сфері) |

### Haversine для географічних даних

```python
from sklearn.metrics.pairwise import haversine_distances
import math

# Конвертувати в радіани
X_radians = np.radians(X)

# DBSCAN з Haversine
dbscan = DBSCAN(
    eps=0.01,  # в радіанах (≈111km)
    min_samples=5,
    metric='haversine'
)

labels = dbscan.fit_predict(X_radians)

# Або через precomputed distance matrix
distance_matrix = haversine_distances(X_radians) * 6371  # радіус Землі в км
dbscan = DBSCAN(eps=5, min_samples=5, metric='precomputed')
labels = dbscan.fit_predict(distance_matrix)
```

---

## Preprocessing для DBSCAN

### 1. Scaling (ВАЖЛИВО!)

**Якщо ознаки в різних масштабах:**

```python
from sklearn.preprocessing import StandardScaler

# Приклад: вік (0-100) та дохід (0-150000)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
```

**Коли НЕ потрібен scaling:**
- Всі ознаки в однакових одиницях (lat/lon, всі в метрах)
- Використовуєш cosine distance

### 2. Dimensionality Reduction

**PCA перед DBSCAN для високорозмірних даних:**

```python
from sklearn.decomposition import PCA

# Зменшити розмірність
pca = PCA(n_components=0.95)  # Залишити 95% variance
X_pca = pca.fit_transform(X_scaled)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_pca)
```

**Чому:** Curse of dimensionality — у високих розмірностях всі точки однаково далекі.

### 3. Outlier Removal (опціонально)

**Якщо дуже багато outliers:**

```python
# Попередня очистка екстремальних outliers
from scipy import stats

z_scores = np.abs(stats.zscore(X))
mask = (z_scores < 3).all(axis=1)
X_clean = X[mask]

# Потім DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_clean)
```

---

## Оцінка кластеризації DBSCAN

### 1. Silhouette Score

**Тільки для non-noise точок!**

```python
from sklearn.metrics import silhouette_score

# Видалити noise точки
mask = labels != -1
if np.sum(mask) > 0 and len(set(labels[mask])) > 1:
    score = silhouette_score(X[mask], labels[mask])
    print(f"Silhouette Score: {score:.4f}")
```

### 2. Davies-Bouldin Index

```python
from sklearn.metrics import davies_bouldin_score

mask = labels != -1
if np.sum(mask) > 0:
    score = davies_bouldin_score(X[mask], labels[mask])
    print(f"Davies-Bouldin Index: {score:.4f}")  # Менше = краще
```

### 3. Статистика кластерів

```python
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = list(labels).count(-1)

print(f"Кількість кластерів: {n_clusters}")
print(f"Кількість noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

# Розміри кластерів
for label in sorted(unique_labels):
    if label == -1:
        continue
    cluster_size = np.sum(labels == label)
    print(f"Кластер {label}: {cluster_size} точок")

# Core vs Border points
core_mask = np.zeros_like(labels, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True

n_core = np.sum(core_mask)
n_border = np.sum((labels != -1) & ~core_mask)

print(f"\nCore points: {n_core}")
print(f"Border points: {n_border}")
print(f"Noise points: {n_noise}")
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Не потрібно знати K** | Кількість кластерів визначається автоматично |
| **Складна форма кластерів** | Знаходить кластери будь-якої форми (не тільки круглі) |
| **Виявлення outliers** | Автоматично ідентифікує noise |
| **Робастність до шуму** | Не чутливий до outliers як K-Means |
| **Різні розміри кластерів** | Не вимагає однакових розмірів |
| **Один прохід даних** | Ефективний алгоритм |
| **Детерміністичний** | Однакові результати (при однакових параметрах) |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Чутливість до параметрів** | ε та MinPts потрібно підбирати |
| **Різна щільність** | Погано з кластерами дуже різної щільності |
| **Високорозмірні дані** | Curse of dimensionality (distance становиться безглуздою) |
| **Складність** | O(n²) без оптимізацій, O(n log n) з індексами |
| **Не працює з категоріальними** | Тільки числові дані (потрібна метрика відстані) |
| **Складно інтерпретувати параметри** | Не інтуїтивно зрозуміло, що таке ε |

---

## Порівняння з іншими методами

| Метод | Потрібно K? | Форма | Outliers | Різна щільність | Складність |
|-------|-------------|-------|----------|------------------|------------|
| **DBSCAN** | ❌ Ні | Будь-яка | ✅ Виявляє | ⚠️ Погано | O(n log n) |
| **K-Means** | ✅ Так | Сферичні | ❌ Чутливий | ❌ Погано | O(nKdi) |
| **Hierarchical** | ❌ Ні | Будь-яка | ⚠️ Середньо | ✅ Добре | O(n²) |
| **OPTICS** | ❌ Ні | Будь-яка | ✅ Виявляє | ✅ Добре | O(n log n) |
| **HDBSCAN** | ❌ Ні | Будь-яка | ✅ Виявляє | ✅ Добре | O(n log n) |

---

## DBSCAN vs Варіації

### OPTICS

**Ordering Points To Identify the Clustering Structure**

**Відмінність:** Створює reachability plot замість прямої кластеризації.

**Переваги:**
- ✅ Працює з різною щільністю
- ✅ Не потрібно задавати ε (тільки MinPts)

```python
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=5, max_eps=2.0)
labels = optics.fit_predict(X)
```

### HDBSCAN

**Hierarchical DBSCAN**

**Відмінність:** Ієрархічний підхід + автоматичний вибір ε.

**Переваги:**
- ✅ Працює з різною щільністю
- ✅ Більш робастний до параметрів
- ✅ Краще з різними розмірами кластерів

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
labels = clusterer.fit_predict(X)
```

---

## Коли використовувати DBSCAN

### Ідеально підходить ✓

- **Не знаємо кількість кластерів** — автоматично визначає
- **Складна форма** — S-криві, кільця, довгі кластери
- **Outliers важливі** — потрібно їх знайти та обробити окремо
- **Географічні дані** — групування локацій
- **Різні розміри кластерів**
- Розмірність **2-10** features
- **Аномалії/fraud detection**

### Краще використати інше ✗

- **Дуже різна щільність** → OPTICS, HDBSCAN
- **Високорозмірні дані** (>20D) → dimensionality reduction → DBSCAN
- **Потрібна швидкість** на великих даних → K-Means, Mini-Batch K-Means
- **Сферичні кластери однакового розміру** → K-Means (простіший)
- **Ієрархія** кластерів потрібна → Hierarchical Clustering

---

## Практичні поради 💡

### 1. Використовуй K-Distance Graph для ε

```python
# ЗАВЖДИ будуй K-distance graph перед DBSCAN
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X)
distances, _ = neighbors.kneighbors(X)
distances = np.sort(distances[:, -1])

plt.plot(distances)
plt.ylabel('5-th Nearest Neighbor Distance')
plt.show()

# Візуально знайди "лікоть"
```

### 2. MinPts = 2 × dimensions (мінімум)

```python
# Емпіричне правило
min_samples = max(4, 2 * X.shape[1])

dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
```

### 3. Scaling якщо різні одиниці

```python
# Якщо вік (0-100) та дохід (0-150K) → SCALING!
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
```

### 4. Експериментуй з параметрами

```python
# Спробуй кілька комбінацій
for eps in [0.3, 0.5, 0.7]:
    for min_samples in [3, 5, 7]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"eps={eps}, min_samples={min_samples}: "
              f"{n_clusters} clusters, {n_noise} noise")
```

### 5. Візуалізуй результати

```python
# ЗАВЖДИ візуалізуй (навіть якщо >2D, використай PCA)
from sklearn.decomposition import PCA

if X.shape[1] > 2:
    pca = PCA(n_components=2)
    X_plot = pca.fit_transform(X)
else:
    X_plot = X

plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis')
plt.scatter(X_plot[labels == -1, 0], X_plot[labels == -1, 1],
           c='black', marker='x', s=100, label='Noise')
plt.legend()
plt.show()
```

### 6. Аналізуй noise points окремо

```python
# Noise може містити цінну інформацію!
noise_points = X[labels == -1]

print(f"Noise points: {len(noise_points)}")
if len(noise_points) > 0:
    print("Характеристики noise:")
    print(pd.DataFrame(noise_points).describe())
    
    # Можливо, це окремий кластер з іншою щільністю?
```

### 7. Grid Search якщо невпевнений

```python
# Використовуй Silhouette для оцінки
best_score = -1
best_params = {}

for eps in np.arange(0.1, 2.0, 0.1):
    for min_samples in [3, 5, 7, 10]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        mask = labels != -1
        if np.sum(mask) > 0 and len(set(labels[mask])) > 1:
            score = silhouette_score(X[mask], labels[mask])
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

print(f"Best: {best_params}, Score: {best_score:.4f}")
```

### 8. Domain knowledge > автоматика

```python
# Географічні дані: 500м = 0.005 градусів (приблизно)
# MinPts = 5 для "району" (мінімум 5 точок)

dbscan = DBSCAN(eps=0.005, min_samples=5, metric='haversine')
```

### 9. Розглянь HDBSCAN для складних випадків

```python
# Якщо різна щільність або невпевнений у параметрах
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(X)
```

### 10. Зберігай параметри

```python
import joblib

# Зберегти модель та параметри
params = {
    'eps': dbscan.eps,
    'min_samples': dbscan.min_samples,
    'metric': dbscan.metric
}

joblib.dump(params, 'dbscan_params.pkl')
joblib.dump(dbscan, 'dbscan_model.pkl')

# Для нових даних
dbscan_loaded = joblib.load('dbscan_model.pkl')
new_labels = dbscan_loaded.fit_predict(X_new)
```

---

## Реальні застосування

### 1. Fraud Detection (Виявлення шахрайства)

**Задача:** Знайти аномальні транзакції.

**Підхід:**
- Ознаки: сума, час, локація, тип
- DBSCAN групує нормальні транзакції
- Noise = потенційне шахрайство

**Переваги:**
- Автоматично виявляє нові типи fraud
- Не потрібно знати типи шахрайства наперед

### 2. Географічна кластеризація

**Задача:** Знайти райони міста з високою концентрацією ресторанів.

**Підхід:**
- Координати ресторанів
- DBSCAN з Haversine metric
- Кластери = райони
- Noise = поодинокі ресторани

### 3. Network Traffic Analysis

**Задача:** Виявити DDoS атаки.

**Підхід:**
- Ознаки: IP, час, розмір пакета
- DBSCAN групує нормальний трафік
- Noise = аномальний трафік (потенційна атака)

### 4. Customer Segmentation (з outliers)

**Задача:** Сегментувати клієнтів, окремо виділити VIP/аномалії.

**Підхід:**
- RFM features
- DBSCAN знаходить основні сегменти
- Noise може бути VIP або проблемні клієнти

### 5. Image Segmentation

**Задача:** Розділити зображення на регіони.

**Підхід:**
- Кожен піксель = точка в кольоровому просторі + позиція
- DBSCAN групує схожі регіони
- Працює з об'єктами складної форми

---

## Поширені помилки ❌

### 1. Не підбирати ε

```python
# ❌ Просто вгадати
dbscan = DBSCAN(eps=0.5, min_samples=5)

# ✅ Використовуй K-distance graph
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X)
distances, _ = neighbors.kneighbors(X)
plt.plot(np.sort(distances[:, -1]))
plt.show()
# Знайди лікоть візуально
```

### 2. Ігнорувати scaling

```python
# ❌ Якщо вік (0-100) та дохід (0-150K)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)  # Дохід домінує!

# ✅ Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan.fit(X_scaled)
```

### 3. MinPts = 2 (занадто мало)

```python
# ❌ Багато маленьких кластерів
dbscan = DBSCAN(eps=0.5, min_samples=2)

# ✅ MinPts ≥ dimensions + 1
min_samples = max(4, X.shape[1] + 1)
dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
```

### 4. Використовувати на високорозмірних даних

```python
# ❌ 50 ознак → всі відстані однакові (curse of dimensionality)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_50d)

# ✅ Спочатку PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_50d)
dbscan.fit(X_reduced)
```

### 5. Не аналізувати noise

```python
# ❌ Ігнорувати точки з міткою -1
labels = dbscan.fit_predict(X)

# ✅ Аналізувати noise окремо
noise_points = X[labels == -1]
print(f"Noise: {len(noise_points)} points")
# Можливо, це важливі аномалії або окремий кластер!
```

### 6. Неправильна метрика відстані

```python
# ❌ Euclidean для географічних координат
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan.fit(lat_lon_data)

# ✅ Haversine для lat/lon
dbscan = DBSCAN(eps=0.01, min_samples=5, metric='haversine')
X_radians = np.radians(lat_lon_data)
dbscan.fit(X_radians)
```

---

## Пов'язані теми

- [[01_KMeans]] — альтернатива для сферичних кластерів
- [[02_Hierarchical_Clustering]] — ієрархічна кластеризація
- [[04_Gaussian_Mixture_Models]] — probabilistic clustering
- [[05_Clustering_Evaluation]] — метрики оцінки
- [[OPTICS]] — покращена версія DBSCAN
- [[HDBSCAN]] — hierarchical DBSCAN

## Ресурси

- [Scikit-learn: DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [Original Paper: Ester et al. (1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
- [DBSCAN Revisited: Why and How You Should (Still) Use DBSCAN](https://dl.acm.org/doi/10.1145/3068335)
- [StatQuest: DBSCAN](https://www.youtube.com/watch?v=RDZUdRSDOok)

---

## Ключові висновки

> DBSCAN — це density-based алгоритм кластеризації, який знаходить кластери довільної форми та автоматично виявляє outliers.

**Основні принципи:**
- **Density-based:** кластер = щільна область точок
- **Два параметри:** ε (радіус) та MinPts (мін. точок)
- **Три типи точок:** Core, Border, Noise
- **Не потрібно знати K** — кількість кластерів визначається автоматично

**Ключові поняття:**
- **Core point:** ≥ MinPts сусідів в радіусі ε
- **Density-reachable:** існує ланцюжок core points
- **Noise (outliers):** точки, які не належать жодному кластеру

**Коли використовувати:**
- Складна форма + не знаємо K + outliers = DBSCAN ✓
- Різна щільність → OPTICS або HDBSCAN ✓
- Сферичні кластери + знаємо K → K-Means ✓

**Найважливіше:**
- Використовуй K-distance graph для вибору ε
- MinPts ≥ dimensions + 1
- Scaling якщо різні одиниці вимірювання
- PCA для високорозмірних даних
- Аналізуй noise points окремо!

---

#ml #unsupervised-learning #clustering #dbscan #density-based #outlier-detection #anomaly-detection
