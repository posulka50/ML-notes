# Hierarchical Clustering (Ієрархічна кластеризація)

## Що це?

**Hierarchical Clustering** — це алгоритм unsupervised learning, який будує **ієрархію кластерів** у вигляді дерева (дендрограми), де кожен рівень представляє різний ступінь деталізації групування.

**Головна ідея:** послідовно об'єднувати (agglomerative) або розділяти (divisive) кластери, створюючи деревоподібну структуру, яка показує як об'єкти групуються на різних рівнях схожості.

## Навіщо потрібна?

- 🌳 **Візуалізація ієрархії** — дендрограма показує структуру даних
- 🔍 **Не потрібно знати K** — можна вибрати кількість кластерів пізніше
- 📊 **Розуміння відношень** — як об'єкти пов'язані між собою
- 🧬 **Таксономія** — створення ієрархічних систем класифікації
- 🎯 **Багаторівнева кластеризація** — різні рівні деталізації
- 📈 **Стабільність** — детермінований результат (без random initialization)

## Коли використовувати?

**Потрібно:**
- Потрібна **ієрархія/таксономія** — дерево відношень
- **Візуалізація структури** даних важлива
- **Не знаємо K** — хочемо побачити всі рівні
- **Малі/середні дані** (< 10,000 точок)
- Потрібна **інтерпретованість** структури
- **Біологічні дані** — філогенетичні дерева, таксономія

**Не потрібно:**
- **Великі дані** (> 10,000 точок) → K-Means, DBSCAN
- Тільки фінальна кластеризація без ієрархії → K-Means
- Потрібна **швидкість** → K-Means
- Кластери **складної форми** → DBSCAN

---

## Типи Hierarchical Clustering

### 1. Agglomerative (Об'єднувальний) ⬆️

**Bottom-up підхід:**

```
Крок 0: Кожна точка = окремий кластер
    A    B    C    D    E

Крок 1: Об'єднати найближчі
    AB   C    D    E

Крок 2: Об'єднати найближчі
    AB   CD   E

Крок 3: Об'єднати найближчі
    ABCD E

Крок 4: Фінальний кластер
    ABCDE
```

**Найпопулярніший!** (використовується в scikit-learn)

### 2. Divisive (Розділювальний) ⬇️

**Top-down підхід:**

```
Крок 0: Всі точки в одному кластері
    ABCDE

Крок 1: Розділити на 2
    ABC   DE

Крок 2: Розділити ABC
    AB    C    DE

Крок 3: Розділити AB
    A    B    C    DE

Крок 4: Розділити DE
    A    B    C    D    E
```

**Рідше використовується** (складніше обчислення)

---

## Agglomerative Clustering (детально)

### Алгоритм

**Вхід:** дані $X$, метрика відстані, метод linkage

**1. Ініціалізація:**
   - Кожна точка = окремий кластер
   - Обчислити матрицю відстаней між усіма точками

**2. Повторювати до одного кластера:**
   
   **a) Знайти пару найближчих кластерів:**
   $$C_i, C_j = \arg\min_{i \neq j} d(C_i, C_j)$$
   
   **b) Об'єднати їх в новий кластер:**
   $$C_{new} = C_i \cup C_j$$
   
   **c) Оновити матрицю відстаней:**
   - Видалити $C_i$ та $C_j$
   - Додати $C_{new}$
   - Обчислити відстані від $C_{new}$ до інших кластерів (залежить від linkage)
   
   **d) Зберегти інформацію про об'єднання** (для дендрограми)

**3. Вихід:** Дендрограма (дерево об'єднань)

### Псевдокод

```
AgglomerativeClustering(X, linkage):
    # Ініціалізація
    clusters = {{x₁}, {x₂}, ..., {xₙ}}
    distances = compute_pairwise_distances(X)
    dendrogram = []
    
    while len(clusters) > 1:
        # Знайти найближчі кластери
        (C_i, C_j, dist) = find_closest_clusters(clusters, distances)
        
        # Об'єднати
        C_new = C_i ∪ C_j
        
        # Зберегти для дендрограми
        dendrogram.append((C_i, C_j, dist))
        
        # Оновити
        clusters.remove(C_i)
        clusters.remove(C_j)
        clusters.add(C_new)
        
        # Перерахувати відстані
        update_distances(distances, C_new, linkage)
    
    return dendrogram
```

---

## Linkage Methods (Методи зв'язування)

**Ключове питання:** Як обчислити відстань між кластерами?

### 1. Single Linkage (Найближчий сусід)

**Відстань = мінімум відстаней між точками з різних кластерів**

$$d_{\text{single}}(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)$$

```
Кластер A: {•  •}     Кластер B: {•  •}
              |______________|
                   ^
            Найкоротша відстань
```

**Переваги:**
- ✅ Може знаходити кластери **неправильної форми**
- ✅ Може знаходити **довгі/витягнуті** кластери

**Недоліки:**
- ❌ **Chaining effect** — схильність до створення довгих ланцюгів
- ❌ Чутливий до **outliers** та шуму

**Використання:** Коли кластери можуть бути витягнутими.

### 2. Complete Linkage (Найдальший сусід)

**Відстань = максимум відстаней між точками**

$$d_{\text{complete}}(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)$$

```
Кластер A: {•  •}     Кластер B: {•  •}
 |______________________________|
               ^
        Найдовша відстань
```

**Переваги:**
- ✅ Створює **компактні** кластери
- ✅ Менш чутливий до **outliers**
- ✅ Уникає chaining effect

**Недоліки:**
- ❌ Може **розбивати** великі кластери
- ❌ Не знаходить витягнуті кластери

**Використання:** Коли потрібні компактні, приблизно сферичні кластери.

### 3. Average Linkage (Середнє)

**Відстань = середня відстань між усіма парами точок**

$$d_{\text{average}}(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)$$

**Переваги:**
- ✅ **Баланс** між single та complete
- ✅ Менш чутливий до outliers
- ✅ Зазвичай **найкращі** результати

**Недоліки:**
- ❌ Обчислювально дорожче

**Використання:** **Рекомендовано за замовчуванням** ✓

### 4. Ward Linkage (Мінімум variance)

**Мінімізує приріст sum of squared errors (SSE) при об'єднанні**

$$d_{\text{ward}}(C_i, C_j) = \frac{|C_i| \cdot |C_j|}{|C_i| + |C_j|} \|\mu_i - \mu_j\|^2$$

де $\mu_i$ — центроїд кластера $C_i$.

**Ідея:** Об'єднувати кластери так, щоб мінімізувати збільшення дисперсії.

**Переваги:**
- ✅ Створює **рівні за розміром** кластери
- ✅ **Компактні** кластери
- ✅ Зазвичай **найкращі** результати

**Недоліки:**
- ❌ Працює тільки з **Euclidean distance**
- ❌ Схильний до створення однакових за розміром кластерів

**Використання:** **Найпопулярніший метод** (за замовчуванням у scikit-learn) ✓

### Візуальне порівняння

```
Single Linkage:          Complete Linkage:        Average Linkage:
  ●   ●●●                  ●●●  ●●●                 ●●● ●●●
   ● ●  ●                  ● ●  ● ●                 ● ●  ●●
    ●   ●                  ●●   ●●                  ●●  ●●

Витягнуті ланцюги        Компактні групи         Збалансовані

Ward Linkage:
   ●●●  ●●●
   ●●   ●●
   ●●●  ●●●

Рівні за розміром, компактні
```

---

## Дендрограма

### Що це?

**Дендрограма** — це деревоподібна діаграма, яка показує ієрархію об'єднань кластерів.

```
Height (відстань)
    |
  6 |         ┌─────────────┐
    |         │             │
  4 |     ┌───┴───┐     ┌───┴───┐
    |     │       │     │       │
  2 |   ┌─┴─┐   ┌─┴─┐ ┌─┴─┐   ┌─┴─┐
    |   │   │   │   │ │   │   │   │
  0 |   A   B   C   D E   F   G   H
    |________________________________
                Samples
```

### Інтерпретація

**Вісь Y (height):**
- Відстань/відмінність при об'єднанні
- Чим вище об'єднання → тим більша різниця між кластерами

**Горизонтальна лінія (cut):**
- Розрізаючи дендрограму на певній висоті → отримуємо K кластерів
- Нижче cut → більше кластерів (детальніше)
- Вище cut → менше кластерів (загальніше)

**Приклад інтерпретації:**

```
    |
  6 |         ┌─────────────┐  ← cut на висоті 5 → 2 кластери
    |   ------│-------------│------
  4 |     ┌───┴───┐     ┌───┴───┐  ← cut на висоті 3 → 4 кластери
    |   --│-------│-----│-------│--
  2 |   ┌─┴─┐   ┌─┴─┐ ┌─┴─┐   ┌─┴─┐
    |   │   │   │   │ │   │   │   │
  0 |   A   B   C   D E   F   G   H
```

**Висновки:**
- A та B дуже схожі (об'єднуються на height ≈ 2)
- {AB} та {CD} помірно схожі (об'єднуються на height ≈ 4)
- {ABCD} та {EFGH} дуже різні (об'єднуються на height ≈ 6)

---

## Простий приклад: Групування тварин

### Дані

| Тварина | Ссавець | Літає | Живе у воді |
|---------|---------|-------|-------------|
| Кіт | 1 | 0 | 0 |
| Собака | 1 | 0 | 0 |
| Кажан | 1 | 1 | 0 |
| Орел | 0 | 1 | 0 |
| Дельфін | 1 | 0 | 1 |
| Акула | 0 | 0 | 1 |

### Крок 1: Матриця відстаней

Використаємо Euclidean distance:

```
        Кіт  Собака Кажан Орел  Дельфін Акула
Кіт     0    0      1.41  1.73  1.41    2.24
Собака  0    0      1.41  1.73  1.41    2.24
Кажан   1.41 1.41   0     1.41  2.00    2.45
Орел    1.73 1.73   1.41  0     2.24    2.00
Дельфін 1.41 1.41   2.00  2.24  0       1.41
Акула   2.24 2.24   2.45  2.00  1.41    0
```

### Крок 2: Agglomerative процес (Average Linkage)

**Ітерація 1:**
- Найближчі: Кіт і Собака (d = 0)
- Об'єднати: {Кіт, Собака}

**Ітерація 2:**
- Найближчі: Дельфін і Акула (d = 1.41)
- Об'єднати: {Дельфін, Акула}

**Ітерація 3:**
- Найближчі: Кажан і Орел (d = 1.41)
- Об'єднати: {Кажан, Орел}

**Ітерація 4:**
- Найближчі: {Кіт, Собака} і {Кажан, Орел}
- Об'єднати: {Кіт, Собака, Кажан, Орел}

**Ітерація 5:**
- Об'єднати всі: {Всі тварини}

### Дендрограма

```
Height
  3 |              ┌──────────────┐
    |              │              │
  2 |     ┌────────┴────┐    ┌────┴────┐
    |     │             │    │         │
  1 |   ┌─┴─┐      ┌────┴┐ ┌─┴─┐
    |   │   │      │     │ │   │
  0 | Кіт Собака Кажан Орел Дельфін Акула
```

**Інтерпретація:**
- **Кластер 1:** {Кіт, Собака} — домашні ссавці
- **Кластер 2:** {Кажан, Орел} — літаючі
- **Кластер 3:** {Дельфін, Акула} — водні

**Вибір K:**
- K=2: {Кіт, Собака, Кажан, Орел} vs {Дельфін, Акула} → наземні vs водні
- K=3: {Кіт, Собака}, {Кажан, Орел}, {Дельфін, Акула}
- K=6: кожна тварина окремо

---

## Складний приклад: Сегментація клієнтів

### Задача

Інтернет-магазин має 200 клієнтів. Потрібно створити ієрархію сегментів.

**Ознаки:**
- Recency (дні від останньої покупки)
- Frequency (кількість покупок)
- Monetary (середній чек)
- Age (вік)

### Результат

**Дендрограма показує:**

```
Level 0 (глибока деталізація):
├─ VIP покупці (високий Monetary, високий Frequency)
├─ Активні молоді (низький Recency, молоді)
├─ Occasional shoppers (середні показники)
├─ Неактивні (високий Recency)
└─ Нові користувачі (низький Frequency)

Level 1 (середня деталізація):
├─ Цінні клієнти {VIP, Активні}
├─ Звичайні клієнти {Occasional}
└─ Ризикові клієнти {Неактивні, Нові}

Level 2 (високий рівень):
├─ Engaged {Цінні, Звичайні}
└─ At Risk {Ризикові}
```

**Бізнес-цінність:**
- Різні маркетингові стратегії для різних рівнів
- Розуміння еволюції клієнтів (як переходять між групами)

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Генерація даних
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=50, centers=3, random_state=42)

# 2. Hierarchical Clustering
hc = AgglomerativeClustering(
    n_clusters=3,           # Кількість кластерів (якщо задано)
    linkage='ward',         # Метод linkage
    metric='euclidean'      # Метрика відстані
)

# 3. Навчання
labels = hc.fit_predict(X)

print(f"Кластери: {set(labels)}")
print(f"Розміри кластерів: {np.bincount(labels)}")

# 4. Візуалізація результату
plt.figure(figsize=(12, 5))

# До кластеризації
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6)
plt.title('Before Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

# Після кластеризації
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title('After Hierarchical Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Побудова дендрограми

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Обчислити linkage matrix
# Це матриця, яка описує ієрархію об'єднань
linkage_matrix = linkage(X, method='ward')

# Структура linkage_matrix:
# [cluster_1, cluster_2, distance, sample_count]

print("Перші 5 об'єднань:")
print(linkage_matrix[:5])

# 2. Побудувати дендрограму
plt.figure(figsize=(14, 7))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # Показати тільки останні p об'єднань
    p=12,                    # Кількість об'єднань для показу
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.title('Dendrogram (Ward Linkage)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.axhline(y=50, color='red', linestyle='--', label='Cut for 3 clusters')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Вибір оптимальної кількості кластерів

```python
# Метод 1: Візуальний аналіз дендрограми
# Шукаємо найдовшу вертикальну лінію без горизонтальних перетинів

# Метод 2: Elbow на відстанях об'єднань
distances = linkage_matrix[:, 2]
last_distances = distances[-10:]  # Останні 10 об'єднань

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), last_distances[::-1], 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Distance at Merge', fontsize=12)
plt.title('Distance vs Number of Clusters', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Оптимальна кількість кластерів — де найбільший стрибок

# Метод 3: Автоматичне визначення через maximum distance gap
max_gap_idx = np.argmax(np.diff(distances[-10:][::-1]))
optimal_clusters = max_gap_idx + 2
print(f"Suggested optimal clusters: {optimal_clusters}")
```

### Повний приклад: Customer Segmentation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Генерація даних про клієнтів
np.random.seed(42)

data = {
    'Recency': np.concatenate([
        np.random.normal(10, 3, 50),    # Активні
        np.random.normal(30, 5, 50),    # Середні
        np.random.normal(90, 15, 50),   # Неактивні
        np.random.normal(5, 2, 50)      # VIP
    ]),
    'Frequency': np.concatenate([
        np.random.normal(15, 3, 50),    # Активні
        np.random.normal(8, 2, 50),     # Середні
        np.random.normal(2, 1, 50),     # Неактивні
        np.random.normal(25, 5, 50)     # VIP
    ]),
    'Monetary': np.concatenate([
        np.random.normal(500, 100, 50),  # Активні
        np.random.normal(300, 50, 50),   # Середні
        np.random.normal(150, 30, 50),   # Неактивні
        np.random.normal(1500, 300, 50)  # VIP
    ])
}

df = pd.DataFrame(data)
df['CustomerID'] = range(len(df))

# Переконатися в позитивних значеннях
df['Recency'] = df['Recency'].clip(1, 365)
df['Frequency'] = df['Frequency'].clip(1, 100)
df['Monetary'] = df['Monetary'].clip(50, 5000)

print("=== Dataset Info ===")
print(df.describe())

# Візуалізація розподілів
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Count', fontsize=11)
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Нормалізація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

# Обчислити linkage для дендрограми
linkage_matrix = linkage(X_scaled, method='ward')

# Візуалізація дендрограми
plt.figure(figsize=(16, 8))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.title('Customer Segmentation Dendrogram (Ward Linkage)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
plt.ylabel('Distance', fontsize=12)

# Лінії для різної кількості кластерів
plt.axhline(y=15, color='red', linestyle='--', label='4 clusters', linewidth=2)
plt.axhline(y=10, color='orange', linestyle='--', label='6 clusters', linewidth=2)

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Визначити оптимальну кількість кластерів
last_merges = linkage_matrix[-10:, 2]
gaps = np.diff(last_merges[::-1])
optimal_k = np.argmax(gaps) + 2

print(f"\nSuggested optimal clusters: {optimal_k}")

# Hierarchical Clustering з оптимальним K
optimal_k = 4  # Базуючись на дендрограмі
hc = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
df['Cluster'] = hc.fit_predict(X_scaled)

# Аналіз кластерів
print("\n" + "="*70)
print("=== Cluster Analysis ===")
print("="*70)

for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster} (n={len(cluster_data)}):")
    print(cluster_data[['Recency', 'Frequency', 'Monetary']].describe().loc[['mean', 'std']])

# Назви сегментів (базуючись на RFM)
def name_segment(row):
    if row['Monetary'] > 1000:
        return 'VIP Customers'
    elif row['Recency'] < 20 and row['Frequency'] > 10:
        return 'Active Buyers'
    elif row['Recency'] > 60:
        return 'Inactive/At Risk'
    else:
        return 'Regular Customers'

df['Segment_Name'] = df.apply(name_segment, axis=1)

# Візуалізація кластерів
fig = plt.figure(figsize=(16, 12))

# 3D scatter
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
scatter = ax1.scatter(df['Recency'], df['Frequency'], df['Monetary'],
                     c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
ax1.set_xlabel('Recency (days)', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_zlabel('Monetary', fontsize=10)
ax1.set_title('3D Cluster Visualization', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Cluster')

# Recency vs Frequency
ax2 = fig.add_subplot(2, 2, 2)
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    ax2.scatter(cluster_data['Recency'], cluster_data['Frequency'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax2.set_xlabel('Recency (days)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Recency vs Frequency', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Frequency vs Monetary
ax3 = fig.add_subplot(2, 2, 3)
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    ax3.scatter(cluster_data['Frequency'], cluster_data['Monetary'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax3.set_xlabel('Frequency', fontsize=11)
ax3.set_ylabel('Monetary', fontsize=11)
ax3.set_title('Frequency vs Monetary', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Cluster profiles (heatmap)
ax4 = fig.add_subplot(2, 2, 4)
cluster_profiles = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_profiles_normalized = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()

sns.heatmap(cluster_profiles_normalized.T, annot=True, fmt='.2f', 
            cmap='RdYlGn_r', center=0, ax=ax4, cbar_kws={'label': 'Standardized Value'})
ax4.set_title('Cluster Profiles (Normalized)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Cluster', fontsize=11)
ax4.set_ylabel('Feature', fontsize=11)

plt.tight_layout()
plt.show()

# Сегменти summary
print("\n" + "="*70)
print("=== Segment Summary ===")
print("="*70)

segment_summary = df.groupby('Cluster').agg({
    'Recency': ['mean', 'std'],
    'Frequency': ['mean', 'std'],
    'Monetary': ['mean', 'std'],
    'CustomerID': 'count'
}).round(2)

segment_summary.columns = ['_'.join(col) for col in segment_summary.columns]
segment_summary = segment_summary.rename(columns={'CustomerID_count': 'Size'})

print(segment_summary)
```

### Порівняння різних linkage методів

```python
# Порівняти Single, Complete, Average, Ward
linkage_methods = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    # Hierarchical Clustering
    hc = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = hc.fit_predict(X_scaled)
    
    # Візуалізація
    axes[idx].scatter(df['Recency'], df['Frequency'], 
                     c=labels, cmap='viridis', s=50, alpha=0.6)
    axes[idx].set_xlabel('Recency', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'{method.capitalize()} Linkage (K=3)', 
                       fontsize=13, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nLinkage Method Comparison:")
print("Single: Can find elongated clusters, prone to chaining")
print("Complete: Creates compact clusters, breaks large clusters")
print("Average: Balanced, generally good results")
print("Ward: Most popular, creates equal-sized compact clusters")
```

---

## Вибір кількості кластерів

### 1. Візуальний аналіз дендрограми

**Правило:** Шукаємо найдовшу вертикальну лінію без горизонтальних перетинів.

```
    |
  8 |         ┌─────────────┐  ← Довга вертикаль
    |   ------│-------------│------ Cut тут → 2 кластери
  6 |         │             │
    |     ┌───┴───┐     ┌───┴───┐
  4 |     │       │     │       │
    |   ┌─┴─┐   ┌─┴─┐ ┌─┴─┐   ┌─┴─┐
  2 |   │   │   │   │ │   │   │   │
  0 |   A   B   C   D E   F   G   H
```

**Інтерпретація:**
- Різка зміна висоти → природний поділ на кластери
- Cut на height=7 → 2 кластери {ABCD}, {EFGH}

### 2. Elbow Method на відстанях

```python
# Відстані останніх об'єднань
distances = linkage_matrix[:, 2]
last_distances = distances[-10:]  # Останні 10 об'єднань

# Шукаємо максимальний gap
gaps = np.diff(last_distances[::-1])
optimal_k = np.argmax(gaps) + 2

print(f"Optimal K: {optimal_k}")

# Візуалізація
plt.plot(range(1, 11), last_distances[::-1], 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Distance')
plt.title('Elbow Method for Hierarchical Clustering')
plt.show()
```

### 3. Silhouette Score

```python
from sklearn.metrics import silhouette_score

# Перевірити різні K
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hc.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette={score:.4f}")

# Оптимальний K
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K (Silhouette): {optimal_k}")

# Візуалізація
plt.plot(K_range, silhouette_scores, 'o-', linewidth=2)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Метрики відстані

### За замовчуванням: Euclidean

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

### Інші метрики

```python
# Manhattan
hc = AgglomerativeClustering(
    n_clusters=3,
    metric='manhattan',
    linkage='average'  # Ward працює тільки з Euclidean!
)

# Cosine
hc = AgglomerativeClustering(
    n_clusters=3,
    metric='cosine',
    linkage='average'
)
```

**Обмеження Ward:**
- Ward працює **тільки з Euclidean distance**
- Для інших метрик використовуй average/complete/single linkage

---

## Connectivity Constraints (обмеження зв'язності)

### Що це?

**Дозволити об'єднувати тільки "сусідні" об'єкти** (за певною структурою).

**Застосування:**
- **Зображення** — пікселі можуть об'єднуватись тільки з сусідніми
- **Часові ряди** — послідовні точки
- **Географічні дані** — сусідні регіони

### Код

```python
from sklearn.neighbors import kneighbors_graph

# Створити connectivity matrix (k-nearest neighbors)
connectivity = kneighbors_graph(X, n_neighbors=5, include_self=False)

# Hierarchical Clustering з обмеженнями
hc_constrained = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward',
    connectivity=connectivity
)

labels_constrained = hc_constrained.fit_predict(X)

# Порівняння
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Без обмежень
hc_free = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_free = hc_free.fit_predict(X)

axes[0].scatter(X[:, 0], X[:, 1], c=labels_free, cmap='viridis')
axes[0].set_title('Without Connectivity Constraints', fontsize=13, fontweight='bold')

# З обмеженнями
axes[1].scatter(X[:, 0], X[:, 1], c=labels_constrained, cmap='viridis')
axes[1].set_title('With Connectivity Constraints (k=5)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Не потрібно знати K** | Можна вибрати пізніше з дендрограми |
| **Ієрархічна структура** | Показує відношення на різних рівнях |
| **Візуалізація** | Дендрограма легко інтерпретується |
| **Детермінізм** | Однакові результати (без random init) |
| **Гнучкість** | Різні linkage методи для різних задач |
| **Багаторівнева кластеризація** | Одна модель → багато розбиттів |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Складність O(n²) - O(n³)** | Дуже повільно на великих даних |
| **Пам'ять O(n²)** | Потрібно зберігати матрицю відстаней |
| **Не масштабується** | Проблеми при n > 10,000 |
| **Не можна "відкрутити"** | Неправильне об'єднання неможливо виправити |
| **Чутливість до outliers** | Особливо single linkage |
| **Chaining effect** | Single linkage може створювати довгі ланцюги |

---

## Порівняння з іншими методами

| Метод | Ієрархія? | Потрібно K? | Складність | Розмір даних | Візуалізація |
|-------|-----------|-------------|------------|--------------|--------------|
| **Hierarchical** | ✅ Так | ❌ Ні | O(n² - n³) | < 10K | ⭐⭐⭐⭐⭐ |
| **K-Means** | ❌ Ні | ✅ Так | O(nKdi) | Будь-який | ⭐⭐ |
| **DBSCAN** | ❌ Ні | ❌ Ні | O(n log n) | Будь-який | ⭐⭐⭐ |
| **GMM** | ❌ Ні | ✅ Так | O(nKd²) | Будь-який | ⭐⭐ |

---

## Коли використовувати Hierarchical Clustering

### Ідеально підходить ✓

- **Малі/середні дані** (< 10,000 точок)
- Потрібна **ієрархія/таксономія** — дерево відношень
- **Візуалізація структури** важлива для інтерпретації
- **Не знаємо K** — хочемо побачити всі можливі рівні
- **Біологічні дані** — філогенетичні дерева, класифікація видів
- **Багаторівнева кластеризація** — різні рівні деталізації
- **Connectivity constraints** — географічні/просторові обмеження

### Краще використати інше ✗

- **Великі дані** (> 10,000) → K-Means, Mini-Batch K-Means
- Потрібна **швидкість** → K-Means
- **Складна форма** кластерів → DBSCAN
- Тільки **фінальна кластеризація** без ієрархії → K-Means
- **Дуже великі дані** (> 100,000) → Sampling + Hierarchical

---

## Практичні поради 💡

### 1. Ward linkage за замовчуванням

```python
# Для більшості випадків
hc = AgglomerativeClustering(linkage='ward')
```

### 2. Завжди будуй дендрограму спочатку

```python
# ОБОВ'ЯЗКОВО візуалізуй дендрограму перед вибором K
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.show()
```

### 3. Використовуй truncated dendrogram для великих даних

```python
# Показати тільки верхні рівні
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # Останні p об'єднань
    p=20
)
```

### 4. Scaling КРИТИЧНИЙ

```python
# ЗАВЖДИ нормалізуй дані
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hc = AgglomerativeClustering(n_clusters=3)
labels = hc.fit_predict(X_scaled)
```

### 5. Sampling для дуже великих даних

```python
# Якщо n > 10,000 → використовуй sampling
sample_size = 5000
indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[indices]

# Кластеризувати sample
hc = AgglomerativeClustering(n_clusters=5)
hc.fit(X_sample)

# Потім predict для всіх даних (якщо потрібно)
# Можна використати KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_sample, hc.labels_)
all_labels = knn.predict(X)
```

### 6. Порівняй різні linkage методи

```python
# Експериментуй
for method in ['ward', 'average', 'complete', 'single']:
    hc = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = hc.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, labels)
    print(f"{method}: Silhouette={score:.4f}")
```

### 7. Інтерпретуй дендрограму правильно

```python
# Шукай довгі вертикальні лінії
# Різкі зміни висоти = природні кластери

# Приклад:
# Height 0-2: багато об'єднань (деталі)
# Height 2-5: помірно (підгрупи)
# Height 5-10: мало об'єднань (основні групи)
# Height >10: РІЗКИЙ стрибок → cut тут!
```

### 8. Domain knowledge для вибору K

```python
# Біологія: види → роди → родини
# Бізнес: індивідуальні → сегменти → групи
# Не завжди оптимальний K за метрикою = правильний для бізнесу
```

### 9. Connectivity constraints для просторових даних

```python
# Якщо дані мають просторову структуру
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(X, n_neighbors=10)
hc = AgglomerativeClustering(n_clusters=5, connectivity=connectivity)
```

### 10. Зберігай linkage matrix

```python
import joblib

# Зберегти linkage matrix для дендрограми
linkage_matrix = linkage(X_scaled, method='ward')
joblib.dump(linkage_matrix, 'linkage_matrix.pkl')

# Завантажити
linkage_matrix = joblib.load('linkage_matrix.pkl')
dendrogram(linkage_matrix)
```

---

## Реальні застосування

### 1. Біологічна таксономія

**Задача:** Створити філогенетичне дерево видів.

**Підхід:**
- Ознаки: ДНК послідовності, морфологічні характеристики
- Hierarchical Clustering для побудови дерева еволюції
- Дендрограма = філогенетичне дерево

### 2. Document Clustering

**Задача:** Організувати документи в тематичну ієрархію.

**Підхід:**
- TF-IDF векторизація текстів
- Hierarchical Clustering (average linkage, cosine distance)
- Дендрограма показує теми → підтеми → документи

### 3. Географічна ієрархія

**Задача:** Групувати міста в регіони, регіони в країни.

**Підхід:**
- Координати міст
- Hierarchical Clustering з connectivity constraints
- Різні рівні cut → різні адміністративні рівні

### 4. Image Segmentation

**Задача:** Розділити зображення на регіони.

**Підхід:**
- Кожен піксель = точка в кольоровому просторі + позиція
- Hierarchical Clustering з spatial connectivity
- Різні рівні = різна деталізація сегментації

### 5. Customer Journey Analysis

**Задача:** Зрозуміти шляхи клієнтів через рівні лояльності.

**Підхід:**
- RFM features для клієнтів
- Hierarchical Clustering для багаторівневої сегментації
- Дендрограма показує еволюцію від нових → активних → VIP

---

## Поширені помилки ❌

### 1. Використовувати на великих даних

```python
# ❌ n = 100,000 → ДУЖЕ повільно
hc = AgglomerativeClustering(n_clusters=5)
hc.fit(X_large)  # Може працювати години!

# ✅ Sampling спочатку
X_sample = X_large[np.random.choice(len(X_large), 5000)]
hc.fit(X_sample)
```

### 2. Не робити scaling

```python
# ❌ Вік (0-100) + Дохід (0-150K)
hc = AgglomerativeClustering()
hc.fit(X)  # Дохід домінує!

# ✅ Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
hc.fit(X_scaled)
```

### 3. Використовувати Ward з non-Euclidean

```python
# ❌ Ward працює тільки з Euclidean
hc = AgglomerativeClustering(
    linkage='ward',
    metric='cosine'  # ПОМИЛКА!
)

# ✅ Average linkage з cosine
hc = AgglomerativeClustering(
    linkage='average',
    metric='cosine'
)
```

### 4. Не дивитись на дендрограму

```python
# ❌ Просто вибрати K=3 без аналізу
hc = AgglomerativeClustering(n_clusters=3)

# ✅ Спочатку дендрограма
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.show()
# Потім вибрати K на основі візуального аналізу
```

### 5. Ігнорувати linkage method

```python
# ❌ Використовувати random linkage
hc = AgglomerativeClustering(linkage='single')  # Може дати chaining

# ✅ Почни з ward або average
hc = AgglomerativeClustering(linkage='ward')
```

### 6. Забути про connectivity для просторових даних

```python
# Для географічних/просторових даних
# ✅ Використовуй connectivity constraints
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(X, n_neighbors=10)
hc = AgglomerativeClustering(
    n_clusters=5,
    connectivity=connectivity
)
```

---

## Пов'язані теми

- [[01_KMeans]] — альтернатива для великих даних
- [[03_DBSCAN]] — density-based clustering
- [[04_Gaussian_Mixture_Models]] — probabilistic clustering
- [[05_Clustering_Evaluation]] — метрики оцінки
- [[06_Clustering_Comparison]] — порівняння методів

## Ресурси

- [Scikit-learn: Hierarchical Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [SciPy: Hierarchical Clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Original Paper: Ward (1963)](https://academic.oup.com/jss/article/58/2/259/2381506)
- [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo)

---

## Ключові висновки

> Hierarchical Clustering будує деревоподібну ієрархію кластерів (дендрограму) через послідовне об'єднання або розділення, дозволяючи аналізувати структуру даних на різних рівнях деталізації.

**Основні принципи:**
- **Agglomerative (bottom-up):** кожна точка → об'єднання → один кластер
- **Divisive (top-down):** один кластер → розділення → окремі точки
- **Linkage methods:** визначають відстань між кластерами
- **Дендрограма:** візуалізація ієрархії

**Linkage методи:**
- **Single:** мінімум відстаней (знаходить витягнуті, chaining)
- **Complete:** максимум відстаней (компактні кластери)
- **Average:** середнє (збалансований) ✓
- **Ward:** мінімум variance (популярний, компактні) ✓

**Коли використовувати:**
- Малі дані + ієрархія + візуалізація = Hierarchical ✓
- Великі дані + швидкість → K-Means ✓
- Складна форма → DBSCAN ✓

**Найважливіше:**
- Обмеження: O(n²) складність, тільки для малих даних
- ЗАВЖДИ будуй дендрограму перед вибором K
- Ward або Average linkage рекомендовані
- Scaling критичний
- Sampling для великих даних

---

#ml #unsupervised-learning #clustering #hierarchical-clustering #dendrogram #agglomerative #linkage #taxonomy
