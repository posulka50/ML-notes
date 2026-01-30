Повний практичний гайд по використанню K-Means в scikit-learn з прикладами коду.

---

## 📦 Основні імпорти

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# K-Means
from sklearn.cluster import KMeans, MiniBatchKMeans

# Метрики
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples
)

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Дані
from sklearn.datasets import make_blobs, load_iris
```

---

## 1️⃣ KMeans — основний клас

### Параметри

```python
KMeans(
    n_clusters=8,              # Кількість кластерів K
    init='k-means++',          # Метод ініціалізації: 'k-means++', 'random',                                                                         або array
    n_init=10,                 # Кількість запусків з різними ініціалізаціями
    max_iter=300,              # Максимум ітерацій на один запуск
    tol=1e-4,                  # Толерантність для збіжності
    verbose=0,                 # Виводити прогрес (0, 1, 2)
    random_state=None,         # Seed для відтворюваності
    copy_x=True,               # Копіювати дані
    algorithm='lloyd'          # 'lloyd', 'elkan' (швидший для багатьох кластерів)
)
```

### Атрибути після fit

```python
model = KMeans(n_clusters=3)
model.fit(X)

# Доступні атрибути:
model.cluster_centers_     # Координати центрів кластерів (K, n_features)
model.labels_              # Мітки кластерів для тренувальних даних (n_samples,)
model.inertia_             # Сума квадратів відстаней до центрів
model.n_iter_              # Кількість ітерацій до збіжності
model.n_features_in_       # Кількість ознак
```

### Методи

```python
# Навчання
model.fit(X)

# Передбачення міток
labels = model.predict(X_new)

# Навчання + передбачення
labels = model.fit_predict(X)

# Відстань до центрів
distances = model.transform(X)  # shape: (n_samples, n_clusters)

# Оцінка (negative inertia)
score = model.score(X)  # Повертає -inertia
```

---

## 2️⃣ Базовий приклад

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Генерація даних
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=0
)

# 2. Створення та навчання моделі
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 3. Отримання результатів
labels = kmeans.labels_
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"Inertia: {inertia:.2f}")
print(f"Iterations: {kmeans.n_iter_}")

# 4. Візуалізація
plt.figure(figsize=(10, 6))

# Точки з кольорами за кластерами
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)

# Центри кластерів
plt.scatter(centers[:, 0], centers[:, 1], 
            c='red', marker='X', s=200, 
            edgecolors='black', linewidths=2,
            label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.colorbar(label='Cluster')
plt.show()
```

---

## 3️⃣ Вибір оптимального K

### Метод 1: Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Тестуємо різні K
K_range = range(1, 11)
inertias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(True, alpha=0.3)
plt.show()

# Знаходимо "лікоть" автоматично (приблизно)
from kneed import KneeLocator  # pip install kneed

kl = KneeLocator(K_range, inertias, curve='convex', direction='decreasing')
optimal_k = kl.elbow
print(f"Optimal K by Elbow: {optimal_k}")
```

### Метод 2: Silhouette Score

```python
from sklearn.metrics import silhouette_score

K_range = range(2, 11)
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette={score:.3f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Different K')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.legend()
plt.show()

# Оптимальний K
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal K by Silhouette: {optimal_k}")
```

### Метод 3: Комбінований підхід

```python
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score
)

def evaluate_kmeans(X, k_range):
    results = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        results.append({
            'K': k,
            'Inertia': kmeans.inertia_,
            'Silhouette': silhouette_score(X, labels),
            'Davies-Bouldin': davies_bouldin_score(X, labels),
            'Calinski-Harabasz': calinski_harabasz_score(X, labels)
        })
    
    return pd.DataFrame(results)

# Оцінка
results_df = evaluate_kmeans(X, range(2, 11))
print(results_df)

# Візуалізація всіх метрик
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Inertia (Elbow)
axes[0, 0].plot(results_df['K'], results_df['Inertia'], 'o-')
axes[0, 0].set_title('Elbow Method (Inertia)')
axes[0, 0].set_xlabel('K')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].grid(True, alpha=0.3)

# Silhouette (максимум)
axes[0, 1].plot(results_df['K'], results_df['Silhouette'], 'o-', color='green')
axes[0, 1].set_title('Silhouette Score (max)')
axes[0, 1].set_xlabel('K')
axes[0, 1].set_ylabel('Silhouette')
axes[0, 1].grid(True, alpha=0.3)

# Davies-Bouldin (мінімум)
axes[1, 0].plot(results_df['K'], results_df['Davies-Bouldin'], 'o-', color='red')
axes[1, 0].set_title('Davies-Bouldin Index (min)')
axes[1, 0].set_xlabel('K')
axes[1, 0].set_ylabel('Davies-Bouldin')
axes[1, 0].grid(True, alpha=0.3)

# Calinski-Harabasz (максимум)
axes[1, 1].plot(results_df['K'], results_df['Calinski-Harabasz'], 'o-', color='purple')
axes[1, 1].set_title('Calinski-Harabasz Score (max)')
axes[1, 1].set_xlabel('K')
axes[1, 1].set_ylabel('Calinski-Harabasz')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4️⃣ Silhouette Plot

```python
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_silhouette(X, n_clusters):
    """Візуалізація Silhouette Plot для кластеризації"""
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Silhouette scores
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Візуалізація
    fig, ax = plt.subplots(figsize=(10, 7))
    
    y_lower = 10
    for i in range(n_clusters):
        # Silhouette scores для кластера i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        
        # Label кластера
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        y_lower = y_upper + 10
    
    ax.set_title(f'Silhouette Plot for K={n_clusters}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    
    # Середня лінія
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Average: {silhouette_avg:.3f}')
    
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.legend()
    plt.show()

# Приклад використання
plot_silhouette(X, n_clusters=4)
```

---

## 5️⃣ Preprocessing — масштабування

```python
from sklearn.preprocessing import StandardScaler

# K-Means чутливий до масштабу ознак!
# Завжди масштабуй дані перед кластеризацією

# Без масштабування
kmeans_no_scale = KMeans(n_clusters=3, random_state=42)
labels_no_scale = kmeans_no_scale.fit_predict(X)

# З масштабуванням
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=3, random_state=42)
labels_scaled = kmeans_scaled.fit_predict(X_scaled)

# Порівняння
from sklearn.metrics import silhouette_score

print(f"Silhouette без масштабування: {silhouette_score(X, labels_no_scale):.3f}")
print(f"Silhouette з масштабуванням: {silhouette_score(X_scaled, labels_scaled):.3f}")
```

---

## 6️⃣ MiniBatchKMeans — для великих даних

```python
from sklearn.cluster import MiniBatchKMeans
import time

# Генерація великого датасету
X_large, _ = make_blobs(n_samples=100000, centers=5, random_state=42)

# KMeans (стандартний)
start = time.time()
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_large)
time_kmeans = time.time() - start

# MiniBatchKMeans
start = time.time()
mbkmeans = MiniBatchKMeans(
    n_clusters=5,
    batch_size=1000,      # Розмір батчу
    max_iter=100,
    random_state=42
)
mbkmeans.fit(X_large)
time_mbkmeans = time.time() - start

print(f"KMeans:         {time_kmeans:.2f} сек")
print(f"MiniBatchKMeans: {time_mbkmeans:.2f} сек")
print(f"Прискорення:    {time_kmeans/time_mbkmeans:.1f}x")

# Порівняння якості
from sklearn.metrics import silhouette_score

labels_km = kmeans.predict(X_large)
labels_mbkm = mbkmeans.predict(X_large)

print(f"\nSilhouette KMeans:     {silhouette_score(X_large, labels_km):.3f}")
print(f"Silhouette MiniBatch:  {silhouette_score(X_large, labels_mbkm):.3f}")
```

### Параметри MiniBatchKMeans

```python
MiniBatchKMeans(
    n_clusters=8,
    init='k-means++',
    max_iter=100,
    batch_size=1024,           # Розмір батчу (більше = точніше, повільніше)
    verbose=0,
    compute_labels=True,
    random_state=None,
    tol=0.0,
    max_no_improvement=10,     # Зупинка, якщо немає покращення
    init_size=None,            # Розмір вибірки для ініціалізації
    n_init=3,                  # Кількість ініціалізацій
    reassignment_ratio=0.01    # Поріг для переприсвоєння
)
```

---

## 7️⃣ Візуалізація результатів

### 2D візуалізація

```python
def visualize_clusters(X, labels, centers=None, title='K-Means Clustering'):
    """Візуалізація кластерів для 2D даних"""
    plt.figure(figsize=(10, 6))
    
    # Кластери
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, 
                         cmap='viridis', alpha=0.6, s=50)
    
    # Центри
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1],
                   c='red', marker='X', s=200,
                   edgecolors='black', linewidths=2,
                   label='Centroids')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label='Cluster')
    if centers is not None:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Приклад
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
visualize_clusters(X, labels, kmeans.cluster_centers_)
```

### Візуалізація через PCA (для багатовимірних даних)

```python
from sklearn.decomposition import PCA

def visualize_high_dim_clusters(X, labels, centers=None):
    """Візуалізація багатовимірних даних через PCA"""
    
    # PCA до 2 компонент
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Центри також трансформуємо
    centers_pca = None
    if centers is not None:
        centers_pca = pca.transform(centers)
    
    # Візуалізація
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                         cmap='viridis', alpha=0.6, s=50)
    
    if centers_pca is not None:
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                   c='red', marker='X', s=200,
                   edgecolors='black', linewidths=2,
                   label='Centroids')
    
    plt.title(f'K-Means (PCA projection, explained var: {pca.explained_variance_ratio_.sum():.2%})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, label='Cluster')
    if centers_pca is not None:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Приклад з Iris (4D → 2D)
from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_iris)

visualize_high_dim_clusters(X_iris, labels, kmeans.cluster_centers_)
```

---

## 8️⃣ Повний приклад: сегментація клієнтів

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Генерація даних клієнтів
np.random.seed(42)
n_customers = 500

data = {
    'Age': np.random.randint(18, 70, n_customers),
    'Income': np.random.randint(20000, 150000, n_customers),
    'SpendingScore': np.random.randint(1, 100, n_customers),
    'Frequency': np.random.randint(1, 50, n_customers)
}

df = pd.DataFrame(data)
print(df.head())
print(df.describe())

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Вибір K
K_range = range(2, 11)
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette={score:.3f}")

# Оптимальний K
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K: {optimal_k}")

# 4. Фінальна кластеризація
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Аналіз кластерів
print("\n=== Cluster Analysis ===")
cluster_summary = df.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Income': ['mean', 'std'],
    'SpendingScore': ['mean', 'std'],
    'Frequency': ['mean', 'std']
}).round(2)
print(cluster_summary)

# Кількість клієнтів у кожному кластері
print("\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())

# 6. Візуалізація
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Income vs Spending Score
axes[0, 0].scatter(df['Income'], df['SpendingScore'], 
                   c=df['Cluster'], cmap='viridis', alpha=0.6)
axes[0, 0].set_xlabel('Income')
axes[0, 0].set_ylabel('Spending Score')
axes[0, 0].set_title('Income vs Spending Score')

# Age vs Spending Score
axes[0, 1].scatter(df['Age'], df['SpendingScore'],
                   c=df['Cluster'], cmap='viridis', alpha=0.6)
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Spending Score')
axes[0, 1].set_title('Age vs Spending Score')

# Income vs Frequency
axes[1, 0].scatter(df['Income'], df['Frequency'],
                   c=df['Cluster'], cmap='viridis', alpha=0.6)
axes[1, 0].set_xlabel('Income')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Income vs Frequency')

# Cluster distribution
df['Cluster'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Cluster Distribution')

plt.tight_layout()
plt.show()

# 7. Інтерпретація кластерів
print("\n=== Cluster Interpretation ===")
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_data)} customers):")
    print(f"  Age: {cluster_data['Age'].mean():.1f} years")
    print(f"  Income: ${cluster_data['Income'].mean():.0f}")
    print(f"  Spending Score: {cluster_data['SpendingScore'].mean():.1f}")
    print(f"  Frequency: {cluster_data['Frequency'].mean():.1f} times")
```

---

## 9️⃣ Збереження та завантаження моделі

```python
import joblib

# Збереження
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Завантаження
loaded_kmeans = joblib.load('kmeans_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Використання
new_data = np.array([[25, 50000, 75, 12]])
new_data_scaled = loaded_scaler.transform(new_data)
cluster = loaded_kmeans.predict(new_data_scaled)

print(f"New customer belongs to cluster: {cluster[0]}")
```

---

## 🔟 Поради та best practices

### 1. Завжди масштабуй дані

```python
# ПОГАНО
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# ДОБРЕ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
```

### 2. Використовуй n_init > 1

```python
# ПОГАНО
kmeans = KMeans(n_clusters=3, n_init=1)

# ДОБРЕ (10 різних ініціалізацій, вибирає найкращу)
kmeans = KMeans(n_clusters=3, n_init=10)
```

### 3. Фіксуй random_state

```python
kmeans = KMeans(n_clusters=3, random_state=42)
```

### 4. Перевіряй метрики

```python
from sklearn.metrics import silhouette_score

labels = kmeans.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels)

if score < 0.25:
    print("⚠️ Слабка структура кластерів!")
```

### 5. Візуалізуй результати

```python
# Завжди дивись на кластери очима
visualize_clusters(X_scaled, labels, kmeans.cluster_centers_)
```

### 6. Для великих даних → MiniBatchKMeans

```python
if len(X) > 10000:
    kmeans = MiniBatchKMeans(n_clusters=3, batch_size=1000)
else:
    kmeans = KMeans(n_clusters=3)
```

---

## Чек-лист для K-Means

```python
# ✅ 1. Завантажити дані
X = load_data()

# ✅ 2. EDA
print(X.shape)
print(pd.DataFrame(X).describe())

# ✅ 3. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Вибір K (Elbow + Silhouette)
evaluate_kmeans(X_scaled, range(2, 11))

# ✅ 5. Навчання моделі
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# ✅ 6. Оцінка якості
silhouette = silhouette_score(X_scaled, labels)
print(f"Silhouette: {silhouette:.3f}")

# ✅ 7. Візуалізація
visualize_clusters(X_scaled, labels, kmeans.cluster_centers_)

# ✅ 8. Інтерпретація
analyze_clusters(X, labels)

# ✅ 9. Збереження
joblib.dump(kmeans, 'model.pkl')
```

---

## Корисні посилання

- [sklearn KMeans docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [sklearn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html)
- [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

---

**Створено для практичного використання K-Means в проєктах** 🚀