	
Повний практичний гайд по використанню DBSCAN в scikit-learn з прикладами коду.

---

## 📦 Основні імпорти

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DBSCAN
from sklearn.cluster import DBSCAN, OPTICS

# Metrics
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Neighbors (для вибору eps)
from sklearn.neighbors import NearestNeighbors

# Data
from sklearn.datasets import make_moons, make_circles, make_blobs
```

---

## 1️⃣ DBSCAN — основний клас

### Параметри

```python
DBSCAN(
    eps=0.5,                   # Радіус околиці (epsilon)
    min_samples=5,             # Мінімальна кількість точок для core point
    metric='euclidean',        # Метрика відстані: 'euclidean', 'manhattan', 'cosine', etc.
    metric_params=None,        # Додаткові параметри для метрики
    algorithm='auto',          # 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,              # Розмір листа для ball_tree/kd_tree
    p=None,                    # Параметр для Minkowski metric (p=2 → euclidean)
    n_jobs=None                # Кількість ядер (-1 = всі)
)
```

### Атрибути після fit

```python
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

# Доступні атрибути:
dbscan.labels_              # Мітки кластерів (n_samples,) [-1 = noise]
dbscan.core_sample_indices_ # Індекси core points
dbscan.components_          # Координати core points (n_core_samples, n_features)
```

### Методи

```python
# Навчання
dbscan.fit(X)

# Навчання + передбачення
labels = dbscan.fit_predict(X)

# ВАЖЛИВО: DBSCAN не має методу predict()!
# Не можна передбачити для нових точок
```

---

## 2️⃣ Базовий приклад

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 1. Генерація даних (два "півмісяці")
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# 2. DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# 3. Результати
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Кластерів знайдено: {n_clusters}")
print(f"Noise точок: {n_noise}")
print(f"Core points: {len(dbscan.core_sample_indices_)}")

# 4. Візуалізація
plt.figure(figsize=(10, 6))

# Кластери
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noise — чорні точки
        col = 'black'
        marker = 'x'
        label = 'Noise'
    else:
        marker = 'o'
        label = f'Cluster {k}'
    
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                s=50, alpha=0.6, label=label)

# Core points (більші точки)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

plt.scatter(X[core_samples_mask, 0], X[core_samples_mask, 1],
            s=100, facecolors='none', edgecolors='red',
            linewidths=2, label='Core points')

plt.title(f'DBSCAN: {n_clusters} clusters, {n_noise} noise points')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3️⃣ Вибір параметрів eps і min_samples

### Метод 1: k-distance графік (найкращий!)

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

def plot_k_distance(X, k=5):
    """
    Будує k-distance графік для вибору eps
    
    k — зазвичай дорівнює min_samples
    """
    # Знайти k найближчих сусідів
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    
    distances, indices = neighbors.kneighbors(X)
    
    # Відстань до k-го сусіда (індекс k-1, бо включає саму точку)
    k_distances = distances[:, k-1]
    
    # Відсортувати за спаданням
    k_distances = np.sort(k_distances)[::-1]
    
    # Графік
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.ylabel(f'{k}-distance', fontsize=12)
    plt.xlabel('Data points sorted by distance', fontsize=12)
    plt.title(f'k-distance Graph (k={k})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Додати горизонтальну лінію для візуального пошуку "коліна"
    plt.axhline(y=np.median(k_distances), color='r', linestyle='--', 
                alpha=0.5, label=f'Median: {np.median(k_distances):.3f}')
    plt.legend()
    plt.show()
    
    # Виведення рекомендацій
    print(f"Рекомендований eps (візуально визначити 'коліно' на графіку)")
    print(f"Орієнтовно:")
    print(f"  - 25% percentile: {np.percentile(k_distances, 25):.3f}")
    print(f"  - 50% percentile: {np.percentile(k_distances, 50):.3f}")
    print(f"  - 75% percentile: {np.percentile(k_distances, 75):.3f}")

# Приклад
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

plot_k_distance(X, k=5)
```

**Як читати графік:**

```
k-distance
  ↑
  │●
  │ ●
  │  ●
  │   ●●
  │     ●●
  │       ●●●  ← "коліно" (різке сповільнення)
  │          ●●●●●●●●──────────
  │
  └────────────────────────────→ Points

eps ≈ значення y в точці коліна
```

---

### Метод 2: Автоматичний пошук коліна

```python
from kneed import KneeLocator  # pip install kneed

def find_optimal_eps(X, k=5):
    """Автоматично знаходить eps через пошук коліна"""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    
    k_distances = np.sort(distances[:, k-1])
    
    # Пошук коліна
    kneedle = KneeLocator(
        range(len(k_distances)), 
        k_distances,
        curve='convex',
        direction='increasing'
    )
    
    optimal_eps = k_distances[kneedle.knee] if kneedle.knee else None
    
    return optimal_eps

# Приклад
optimal_eps = find_optimal_eps(X, k=5)
print(f"Оптимальний eps: {optimal_eps:.3f}")
```

---

### Метод 3: Grid Search з метриками

```python
from sklearn.metrics import silhouette_score

def grid_search_dbscan(X, eps_range, min_samples_range):
    """
    Grid search для DBSCAN з оцінкою Silhouette Score
    """
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Пропустити, якщо всі точки noise або один кластер
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters < 2 or n_clusters == len(X):
                continue
            
            # Silhouette (тільки для не-noise точок)
            mask = labels != -1
            if sum(mask) > 1:
                try:
                    score = silhouette_score(X[mask], labels[mask])
                except:
                    score = -1
            else:
                score = -1
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_pct': n_noise / len(X) * 100,
                'silhouette': score
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('silhouette', ascending=False)
    
    return df

# Приклад
eps_range = np.arange(0.1, 1.0, 0.05)
min_samples_range = range(3, 15)

results_df = grid_search_dbscan(X, eps_range, min_samples_range)

print("Top 5 конфігурацій:")
print(results_df.head(10))

# Візуалізація
best_params = results_df.iloc[0]
print(f"\nНайкраща конфігурація:")
print(f"  eps: {best_params['eps']:.3f}")
print(f"  min_samples: {int(best_params['min_samples'])}")
print(f"  Silhouette: {best_params['silhouette']:.3f}")
print(f"  Кластерів: {int(best_params['n_clusters'])}")
print(f"  Noise: {best_params['noise_pct']:.1f}%")
```

---

### Метод 4: Heatmap для Grid Search

```python
def plot_dbscan_heatmap(X, eps_range, min_samples_range, metric='silhouette'):
    """
    Будує heatmap для різних комбінацій параметрів
    """
    results = np.zeros((len(min_samples_range), len(eps_range)))
    
    for i, min_samples in enumerate(min_samples_range):
        for j, eps in enumerate(eps_range):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            mask = labels != -1
            
            if n_clusters < 2 or sum(mask) < 2:
                results[i, j] = -1
            else:
                try:
                    if metric == 'silhouette':
                        results[i, j] = silhouette_score(X[mask], labels[mask])
                    elif metric == 'n_clusters':
                        results[i, j] = n_clusters
                    elif metric == 'noise_pct':
                        results[i, j] = list(labels).count(-1) / len(X) * 100
                except:
                    results[i, j] = -1
    
    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        results,
        xticklabels=[f'{e:.2f}' for e in eps_range],
        yticklabels=min_samples_range,
        cmap='viridis',
        annot=False,
        fmt='.2f'
    )
    plt.xlabel('eps', fontsize=12)
    plt.ylabel('min_samples', fontsize=12)
    plt.title(f'DBSCAN Grid Search - {metric}', fontsize=14)
    plt.tight_layout()
    plt.show()

# Приклад
eps_range = np.arange(0.1, 1.0, 0.05)
min_samples_range = range(3, 15)

plot_dbscan_heatmap(X, eps_range, min_samples_range, metric='silhouette')
plot_dbscan_heatmap(X, eps_range, min_samples_range, metric='n_clusters')
```

---

## 4️⃣ Порівняння з K-Means

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Дані з нелінійною структурою
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# K-Means
axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=50)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[0].set_title('K-Means (K=2)\n✗ Неправильно розділяє півмісяці', fontsize=12)
axes[0].grid(True, alpha=0.3)

# DBSCAN
for k in set(labels_dbscan):
    if k == -1:
        col = 'black'
        marker = 'x'
    else:
        col = plt.cm.viridis(k / max(labels_dbscan))
        marker = 'o'
    
    mask = labels_dbscan == k
    axes[1].scatter(X[mask, 0], X[mask, 1], c=[col], marker=marker, s=50)

axes[1].set_title(f'DBSCAN (eps=0.3, min_samples=5)\n✓ Правильно знаходить півмісяці', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 5️⃣ Практичні приклади

### Приклад 1: Два кільця

```python
from sklearn.datasets import make_circles

# Генерація даних
X, _ = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

# Масштабування
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
labels = dbscan.fit_predict(X_scaled)

# Результати
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Кластерів: {n_clusters}")
print(f"Noise: {n_noise} ({n_noise/len(X)*100:.1f}%)")

# Візуалізація
plt.figure(figsize=(10, 6))

for k in set(labels):
    if k == -1:
        plt.scatter(X[labels == k, 0], X[labels == k, 1], 
                   c='black', marker='x', s=50, label='Noise')
    else:
        plt.scatter(X[labels == k, 0], X[labels == k, 1], 
                   s=50, label=f'Cluster {k}')

plt.title('DBSCAN на концентричних колах')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Приклад 2: Виявлення аномалій у транзакціях

```python
import numpy as np
import pandas as pd

# Симуляція даних транзакцій
np.random.seed(42)

# Нормальні транзакції (2 групи: дрібні та середні покупки)
normal_small = np.random.normal(loc=50, scale=10, size=(400, 2))
normal_medium = np.random.normal(loc=200, scale=30, size=(300, 2))

# Аномалії (дуже великі або незвичні транзакції)
anomalies = np.random.uniform(low=500, high=1000, size=(20, 2))

# Об'єднання
X = np.vstack([normal_small, normal_medium, anomalies])

df = pd.DataFrame(X, columns=['Amount', 'Frequency'])
df['Type'] = ['Normal']*700 + ['Anomaly']*20  # Справжні мітки (для перевірки)

# Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN для виявлення аномалій
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X_scaled)

df['Cluster'] = labels
df['Predicted'] = ['Anomaly' if l == -1 else 'Normal' for l in labels]

# Результати
print("=== Confusion Matrix ===")
print(pd.crosstab(df['Type'], df['Predicted']))

print(f"\nАномалій виявлено: {list(labels).count(-1)}")
print(f"Справжніх аномалій: {sum(df['Type'] == 'Anomaly')}")

# Візуалізація
plt.figure(figsize=(12, 5))

# Справжні мітки
plt.subplot(1, 2, 1)
scatter = plt.scatter(df['Amount'], df['Frequency'], 
                     c=(df['Type'] == 'Anomaly'), cmap='coolwarm', s=50)
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.title('Справжні мітки')
plt.colorbar(scatter, label='Anomaly')

# DBSCAN результати
plt.subplot(1, 2, 2)
for cluster in set(labels):
    if cluster == -1:
        plt.scatter(df[df['Cluster'] == cluster]['Amount'],
                   df[df['Cluster'] == cluster]['Frequency'],
                   c='red', marker='x', s=100, label='Anomaly (DBSCAN)')
    else:
        plt.scatter(df[df['Cluster'] == cluster]['Amount'],
                   df[df['Cluster'] == cluster]['Frequency'],
                   s=50, label=f'Cluster {cluster}')

plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.title('DBSCAN результати')
plt.legend()

plt.tight_layout()
plt.show()
```

---

### Приклад 3: Сегментація клієнтів за геолокацією

```python
import numpy as np

# Симуляція GPS координат клієнтів у місті
np.random.seed(42)

# 3 райони міста
downtown = np.random.normal(loc=[40.7128, -74.0060], scale=0.01, size=(100, 2))
suburb_north = np.random.normal(loc=[40.7589, -73.9851], scale=0.015, size=(80, 2))
suburb_south = np.random.normal(loc=[40.6782, -73.9442], scale=0.012, size=(70, 2))

# Одиночні клієнти (викиди)
outliers = np.random.uniform(low=[40.65, -74.05], high=[40.80, -73.90], size=(10, 2))

X = np.vstack([downtown, suburb_north, suburb_south, outliers])

# DBSCAN (eps в градусах, ~0.01° ≈ 1.1 км)
dbscan = DBSCAN(eps=0.02, min_samples=10)
labels = dbscan.fit_predict(X)

# Результати
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Районів знайдено: {n_clusters}")
print(f"Одиночних клієнтів: {n_noise}")

# Візуалізація
plt.figure(figsize=(10, 8))

for k in set(labels):
    if k == -1:
        plt.scatter(X[labels == k, 1], X[labels == k, 0],
                   c='black', marker='x', s=100, label='Outliers', zorder=5)
    else:
        plt.scatter(X[labels == k, 1], X[labels == k, 0],
                   s=50, alpha=0.6, label=f'Район {k+1}')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Сегментація клієнтів за геолокацією\n{n_clusters} районів, {n_noise} outliers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6️⃣ Візуалізація для високовимірних даних

### PCA проекція

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Високовимірні дані (4D)
iris = load_iris()
X = iris.data

# Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# PCA для візуалізації
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Візуалізація
plt.figure(figsize=(12, 5))

# PCA проекція
plt.subplot(1, 2, 1)
for k in set(labels):
    if k == -1:
        plt.scatter(X_pca[labels == k, 0], X_pca[labels == k, 1],
                   c='black', marker='x', s=100, label='Noise')
    else:
        plt.scatter(X_pca[labels == k, 0], X_pca[labels == k, 1],
                   s=50, alpha=0.6, label=f'Cluster {k}')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('DBSCAN (PCA проекція)')
plt.legend()
plt.grid(True, alpha=0.3)

# Справжні мітки (для порівняння)
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis', s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Справжні мітки')
plt.colorbar(scatter, label='Species')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 7️⃣ OPTICS — альтернатива DBSCAN

```python
from sklearn.cluster import OPTICS

# OPTICS не потребує eps!
optics = OPTICS(
    min_samples=5,
    xi=0.05,          # Крутизна для виявлення кластерів (0-1)
    min_cluster_size=10
)

labels = optics.fit_predict(X)

# Reachability plot
plt.figure(figsize=(12, 5))

# Кластери
plt.subplot(1, 2, 1)
for k in set(labels):
    if k == -1:
        plt.scatter(X[labels == k, 0], X[labels == k, 1],
                   c='black', marker='x', s=50, label='Noise')
    else:
        plt.scatter(X[labels == k, 0], X[labels == k, 1],
                   s=50, label=f'Cluster {k}')
plt.title('OPTICS Clustering')
plt.legend()
plt.grid(True, alpha=0.3)

# Reachability plot
plt.subplot(1, 2, 2)
space = np.arange(len(X))
reachability = optics.reachability_[optics.ordering_]
plt.plot(space, reachability, 'k-', alpha=0.5)
plt.fill_between(space, 0, reachability, alpha=0.3)
plt.ylabel('Reachability Distance')
plt.xlabel('Sample Index (ordered)')
plt.title('Reachability Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 8️⃣ Оцінка якості

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_dbscan(X, labels):
    """
    Оцінити якість DBSCAN кластеризації
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_samples = len(X)
    
    print("=== DBSCAN Results ===")
    print(f"Кластерів: {n_clusters}")
    print(f"Noise точок: {n_noise} ({n_noise/n_samples*100:.1f}%)")
    print(f"Core points: {sum(labels != -1)}")
    
    # Метрики (без noise)
    mask = labels != -1
    
    if sum(mask) > 1 and len(set(labels[mask])) > 1:
        sil_score = silhouette_score(X[mask], labels[mask])
        db_score = davies_bouldin_score(X[mask], labels[mask])
        
        print(f"\n=== Metrics (excluding noise) ===")
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Davies-Bouldin Index: {db_score:.3f}")
        
        # Інтерпретація
        print(f"\nІнтерпретація:")
        if sil_score > 0.7:
            print("  ✓ Відмінна кластеризація")
        elif sil_score > 0.5:
            print("  ✓ Добра кластеризація")
        elif sil_score > 0.25:
            print("  ⚠ Слабка кластеризація")
        else:
            print("  ✗ Погана кластеризація")
    else:
        print("\n⚠ Недостатньо кластерів для метрик")
    
    # Розподіл по кластерах
    print(f"\n=== Cluster Sizes ===")
    for k in sorted(set(labels)):
        if k == -1:
            continue
        count = list(labels).count(k)
        print(f"Cluster {k}: {count} points ({count/n_samples*100:.1f}%)")

# Приклад
evaluate_dbscan(X_scaled, labels)
```

---

## 9️⃣ Збереження та завантаження

```python
import joblib

# Навчання
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Збереження
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(dbscan, 'dbscan_model.pkl')

# ВАЖЛИВО: Зберегти також core_sample_indices_ для можливості інтерпретації
model_data = {
    'dbscan': dbscan,
    'scaler': scaler,
    'labels_train': labels,
    'X_train_scaled': X_scaled
}
joblib.dump(model_data, 'dbscan_full.pkl')

# Завантаження
loaded_data = joblib.load('dbscan_full.pkl')
loaded_dbscan = loaded_data['dbscan']
loaded_scaler = loaded_data['scaler']

print(f"Core points: {len(loaded_dbscan.core_sample_indices_)}")
```

**ВАЖЛИВО:** DBSCAN не має методу `predict()` для нових точок!

### Як передбачити для нових точок?

```python
def predict_dbscan(new_X, dbscan, X_train, scaler, eps):
    """
    Спосіб передбачення для нових точок через найближчого сусіда
    
    ОБМЕЖЕННЯ: Це наближення, не офіційний метод
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Масштабувати нові дані
    new_X_scaled = scaler.transform(new_X)
    
    # Знайти найближчого сусіда з тренувальних даних
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_train)
    
    distances, indices = nn.kneighbors(new_X_scaled)
    
    # Якщо відстань < eps → присвоїти мітку найближчого
    # Інакше → noise (-1)
    predictions = []
    for dist, idx in zip(distances.flatten(), indices.flatten()):
        if dist <= eps:
            predictions.append(dbscan.labels_[idx])
        else:
            predictions.append(-1)  # Noise
    
    return np.array(predictions)

# Приклад
new_data = np.array([[1.5, 2.5], [5.0, 5.0]])
new_labels = predict_dbscan(
    new_data, 
    loaded_dbscan, 
    loaded_data['X_train_scaled'],
    loaded_scaler,
    eps=0.5
)

print(f"Нові точки: {new_labels}")
```

---

## 🔟 Поради та best practices

### 1. Завжди масштабуй дані

```python
# ПОГАНО
dbscan = DBSCAN(eps=0.5)
dbscan.fit(X)

# ДОБРЕ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.5)
dbscan.fit(X_scaled)
```

---

### 2. Використовуй k-distance графік

```python
# Завжди будуй k-distance перед DBSCAN
plot_k_distance(X_scaled, k=5)
# → Визнач eps візуально
```

---

### 3. MinPts евристика

```python
# Правило: MinPts = 2 × dimensionality
n_features = X.shape[1]
min_samples = 2 * n_features

# Але не менше 4
min_samples = max(min_samples, 4)
```

---

### 4. Перевіряй % noise

```python
n_noise = list(labels).count(-1)
noise_pct = n_noise / len(X) * 100

if noise_pct > 20:
    print("⚠️ Занадто багато noise! Спробуй:")
    print("  - Збільшити eps")
    print("  - Зменшити min_samples")
```

---

### 5. Візуалізуй core/border/noise

```python
# Core points
core_mask = np.zeros_like(labels, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True

# Border points (не core, але не noise)
border_mask = (labels != -1) & (~core_mask)

# Noise
noise_mask = (labels == -1)

plt.scatter(X[core_mask, 0], X[core_mask, 1], c='blue', label='Core', s=50)
plt.scatter(X[border_mask, 0], X[border_mask, 1], c='cyan', label='Border', s=50)
plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', marker='x', label='Noise', s=100)
plt.legend()
```

---

### 6. Для великих даних — оптимізації

```python
# Використовуй ball_tree або kd_tree для прискорення
dbscan = DBSCAN(
    eps=0.5,
    min_samples=5,
    algorithm='ball_tree',  # або 'kd_tree'
    leaf_size=30,
    n_jobs=-1  # Паралелізація
)
```

---

## Чек-лист для DBSCAN

```python
# ✅ 1. Завантажити дані
X = load_data()

# ✅ 2. EDA
print(X.shape)
print(pd.DataFrame(X).describe())

# ✅ 3. Масштабування (ОБОВ'ЯЗКОВО!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Вибір eps через k-distance графік
plot_k_distance(X_scaled, k=5)
# Візуально визначити eps

# ✅ 5. Вибір min_samples
min_samples = 2 * X.shape[1]  # або 4-5 для 2D

# ✅ 6. Навчання DBSCAN
dbscan = DBSCAN(eps=chosen_eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

# ✅ 7. Оцінка результатів
evaluate_dbscan(X_scaled, labels)

# ✅ 8. Візуалізація
visualize_clusters(X, labels)

# ✅ 9. Якщо потрібно — Grid Search
results = grid_search_dbscan(X_scaled, eps_range, min_samples_range)

# ✅ 10. Збереження
joblib.dump({'dbscan': dbscan, 'scaler': scaler}, 'model.pkl')
```

---

## Порівняльна таблиця параметрів

|Ситуація|eps|min_samples|Очікуваний результат|
|---|---|---|---|
|Щільні кластери|Малий (0.1-0.3)|Середній (5-10)|Багато малих кластерів|
|Розріджені кластери|Великий (0.5-1.0)|Малий (3-5)|Мало великих кластерів|
|Багато шуму|Середній|Великий (10-20)|Менше noise|
|Високі розмірності|Великий|Великий|Складно ☹|
|Географічні дані|~0.01-0.05°|10-20|Райони міста|

---

## Корисні посилання

- [sklearn DBSCAN docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [sklearn OPTICS docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)
- [sklearn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html#dbscan)

---

**Створено для практичного використання DBSCAN у проєктах** 🚀