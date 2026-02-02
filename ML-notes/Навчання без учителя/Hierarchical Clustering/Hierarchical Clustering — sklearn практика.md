
Повний практичний гайд по використанню Hierarchical Clustering в scikit-learn та scipy з прикладами коду.

---

## 📦 Основні імпорти

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# Metrics
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Data
from sklearn.datasets import make_blobs, load_iris
```

---

## 1️⃣ AgglomerativeClustering — основний клас (sklearn)

### Параметри

```python
AgglomerativeClustering(
    n_clusters=2,              # Кількість кластерів (або None для distance_threshold)
    affinity='euclidean',      # Метрика відстані: 'euclidean', 'manhattan', 'cosine', etc.
    memory=None,               # Кешування для прискорення
    connectivity=None,         # Матриця зв'язаності (обмеження на об'єднання)
    compute_full_tree='auto',  # Будувати повне дерево
    linkage='ward',            # Метод linkage: 'ward', 'complete', 'average', 'single'
    distance_threshold=None,   # Поріг відстані (якщо None, використовуємо n_clusters)
    compute_distances=False    # Зберігати відстані між кластерами
)
```

### Атрибути після fit

```python
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
model.fit(X)

# Доступні атрибути:
model.labels_              # Мітки кластерів (n_samples,)
model.n_clusters_          # Кількість кластерів
model.n_leaves_            # Кількість листків у дереві
model.n_connected_components_  # Кількість зв'язаних компонент
model.children_            # Історія об'єднань (n_samples-1, 2)
```

### Методи

```python
# Навчання
model.fit(X)

# Навчання + передбачення
labels = model.fit_predict(X)

# ВАЖЛИВО: AgglomerativeClustering НЕ має методу predict()!
# Не можна передбачити для нових точок
```

---

## 2️⃣ Базовий приклад

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Генерація даних
X, y_true = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)

# 2. Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X)

# 3. Результати
print(f"Кількість кластерів: {hc.n_clusters_}")
print(f"Кількість листків: {hc.n_leaves_}")

# 4. Візуалізація
plt.figure(figsize=(10, 6))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('Hierarchical Clustering (Ward linkage)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3️⃣ Дендрограма з scipy

### Базова дендрограма

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 1. Побудувати дерево (linkage matrix)
Z = linkage(X, method='ward')  # method: 'ward', 'single', 'complete', 'average'

# 2. Візуалізація дендрограми
plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.title('Dendrogram (Ward linkage)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.axhline(y=10, color='r', linestyle='--', label='Cut at height=10')
plt.legend()
plt.show()

# Linkage matrix Z:
# Z[i] = [cluster_1, cluster_2, distance, sample_count]
# - перші 2 колонки: які кластери об'єдналися
# - 3-я колонка: відстань об'єднання
# - 4-а колонка: кількість точок у новому кластері
```

---

### Покращена дендрограма

```python
def plot_dendrogram(X, method='ward', truncate_mode=None, p=30):
    """
    Красива дендрограма з додатковою інформацією
    """
    # Linkage matrix
    Z = linkage(X, method=method)
    
    # Графік
    plt.figure(figsize=(15, 7))
    
    # Дендрограма
    dendrogram(
        Z,
        truncate_mode=truncate_mode,  # None, 'lastp', 'level'
        p=p,                          # кількість кластерів для показу (якщо truncate)
        leaf_font_size=10,
        show_contracted=True          # показати скорочені кластери
    )
    
    plt.title(f'Dendrogram ({method} linkage)', fontsize=16)
    plt.xlabel('Sample index (or cluster size)', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return Z

# Приклад
Z = plot_dendrogram(X, method='ward')
```

---

### Truncated dendrogram (для великих даних)

```python
# Для великих даних: показати тільки top 30 кластерів
plt.figure(figsize=(15, 7))

dendrogram(
    Z,
    truncate_mode='lastp',  # показати тільки останні p об'єднань
    p=30,                   # кількість кластерів
    show_leaf_counts=True,  # показати кількість точок
    leaf_font_size=10
)

plt.title('Truncated Dendrogram (last 30 merges)')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
plt.show()
```

---

## 4️⃣ Вибір кількості кластерів

### Метод 1: Візуально з дендрограми

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Побудувати дендрограму
Z = linkage(X, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.title('Find the largest vertical gap')
plt.xlabel('Sample')
plt.ylabel('Distance')

# Додати можливі розрізи
for height in [5, 10, 15]:
    plt.axhline(y=height, color='r', linestyle='--', alpha=0.5, 
                label=f'Cut at {height}')
plt.legend()
plt.show()

# Шукай найбільший вертикальний розрив!
```

---

### Метод 2: Elbow на відстанях об'єднання

```python
# Відстані між об'єднаннями
distances = Z[:, 2]

# Графік
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(distances)+1), distances, 'o-')
plt.xlabel('Merge step')
plt.ylabel('Distance')
plt.title('Distances in hierarchical clustering')
plt.grid(True, alpha=0.3)
plt.show()

# Шукай різке збільшення відстані (elbow)
```

---

### Метод 3: Silhouette Score для різних K

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hc.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette={score:.3f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.grid(True, alpha=0.3)
plt.show()

# Оптимальний K
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K: {optimal_k}")
```

---

### Метод 4: Distance threshold (автоматичний вибір)

```python
from scipy.cluster.hierarchy import fcluster

# Розрізати на певній висоті
height_threshold = 10
clusters = fcluster(Z, height_threshold, criterion='distance')

print(f"Кількість кластерів при height={height_threshold}: {len(set(clusters))}")

# Різні висоти
for height in [5, 10, 15, 20]:
    clusters = fcluster(Z, height, criterion='distance')
    n_clusters = len(set(clusters))
    print(f"Height {height}: {n_clusters} clusters")
```

---

## 5️⃣ Порівняння linkage методів

```python
from sklearn.datasets import make_moons

# Дані з нелінійною структурою (два "півмісяці")
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Масштабування
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Різні linkage методи
linkage_methods = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    # Hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=2, linkage=method)
    labels = hc.fit_predict(X_scaled)
    
    # Silhouette
    sil_score = silhouette_score(X_scaled, labels)
    
    # Візуалізація
    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    axes[idx].set_title(f'{method.capitalize()} Linkage\nSilhouette: {sil_score:.3f}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Очікувані результати:**

- **Single:** Знайде півмісяці ✓ (але chaining на складних даних)
- **Complete:** Розділить вертикально ✗
- **Average:** Баланс
- **Ward:** Розділить вертикально ✗ (передбачає сферичні кластери)

---

## 6️⃣ Практичні приклади

### Приклад 1: Сегментація клієнтів

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 1. Генерація даних клієнтів
np.random.seed(42)
n_customers = 200

data = {
    'Age': np.random.randint(18, 70, n_customers),
    'Income': np.random.randint(20000, 150000, n_customers),
    'SpendingScore': np.random.randint(1, 100, n_customers)
}

df = pd.DataFrame(data)
print(df.head())

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Побудова дендрограми
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(15, 7))
dendrogram(Z, truncate_mode='lastp', p=20)
plt.title('Customer Segmentation Dendrogram')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
plt.axhline(y=8, color='r', linestyle='--', label='Cut at height=8')
plt.legend()
plt.show()

# 4. Вибір кількості кластерів
optimal_k = 4

# 5. Кластеризація
hc = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
df['Cluster'] = hc.fit_predict(X_scaled)

# 6. Аналіз кластерів
print("\n=== Cluster Analysis ===")
cluster_summary = df.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Income': ['mean', 'std'],
    'SpendingScore': ['mean', 'std']
}).round(2)
print(cluster_summary)

# Кількість клієнтів у кожному кластері
print("\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())

# 7. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Income vs Spending Score
axes[0].scatter(df['Income'], df['SpendingScore'], 
                c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
axes[0].set_xlabel('Income')
axes[0].set_ylabel('Spending Score')
axes[0].set_title('Income vs Spending Score')
axes[0].grid(True, alpha=0.3)

# Age vs Income
axes[1].scatter(df['Age'], df['Income'], 
                c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Income')
axes[1].set_title('Age vs Income')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Приклад 2: Ієрархія документів (text clustering)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Дані (документи)
documents = [
    "Machine learning is great for data science",
    "Deep learning neural networks are powerful",
    "Python is perfect for machine learning",
    "The cat sat on the mat",
    "Dogs are friendly animals",
    "My cat loves playing with toys",
    "Data science requires statistics knowledge",
    "Neural networks learn from data",
]

# 2. TF-IDF векторизація
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(documents).toarray()

# 3. Hierarchical clustering
Z = linkage(X_tfidf, method='average')

# 4. Дендрограма
plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=[f"Doc {i+1}" for i in range(len(documents))],
    leaf_font_size=10
)
plt.title('Document Hierarchy')
plt.xlabel('Document')
plt.ylabel('Distance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Кластеризація
n_clusters = 3
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
labels = hc.fit_predict(X_tfidf)

# 6. Результати
for cluster_id in range(n_clusters):
    print(f"\n=== Cluster {cluster_id} ===")
    cluster_docs = [doc for doc, label in zip(documents, labels) if label == cluster_id]
    for doc in cluster_docs:
        print(f"  - {doc}")
```

**Очікувані кластери:**

- Cluster 0: ML/Data Science документи
- Cluster 1: Тварини (кішки/собаки)
- Cluster 2: Інше

---

### Приклад 3: Ієрархічна сегментація зображень (колір)

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 1. Генерація кольорових точок (RGB)
np.random.seed(42)

# 3 групи кольорів: червоні, сині, зелені
colors_red = np.random.rand(50, 3) * [1, 0.3, 0.3] + [0, 0, 0]
colors_blue = np.random.rand(50, 3) * [0.3, 0.3, 1] + [0, 0, 0]
colors_green = np.random.rand(50, 3) * [0.3, 1, 0.3] + [0, 0, 0]

X_colors = np.vstack([colors_red, colors_blue, colors_green])
X_colors = np.clip(X_colors, 0, 1)  # обмежити [0, 1]

# 2. Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X_colors)

# 3. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Оригінальні кольори
axes[0].scatter(range(len(X_colors)), [0]*len(X_colors), 
                c=X_colors, s=100, marker='s')
axes[0].set_title('Original Colors')
axes[0].set_yticks([])
axes[0].set_xlabel('Color index')

# Кластеризовані (відсортовані по кластерах)
sorted_idx = np.argsort(labels)
axes[1].scatter(range(len(X_colors)), [0]*len(X_colors),
                c=X_colors[sorted_idx], s=100, marker='s')
axes[1].set_title('Clustered Colors (sorted by cluster)')
axes[1].set_yticks([])
axes[1].set_xlabel('Color index')

plt.tight_layout()
plt.show()

# 4. Дендрограма
Z = linkage(X_colors, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=20, color_threshold=0)
plt.title('Color Hierarchy')
plt.xlabel('Sample')
plt.ylabel('Distance')
plt.show()
```

---

## 7️⃣ Оцінка якості

```python
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score
)

def evaluate_hierarchical(X, labels):
    """
    Оцінити якість Hierarchical Clustering
    """
    n_clusters = len(set(labels))
    n_samples = len(X)
    
    print("=== Hierarchical Clustering Results ===")
    print(f"Кількість кластерів: {n_clusters}")
    
    # Метрики
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    
    print(f"\n=== Metrics ===")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Davies-Bouldin Index: {db_score:.3f}")
    print(f"Calinski-Harabasz Score: {ch_score:.1f}")
    
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
    
    # Розподіл по кластерах
    print(f"\n=== Cluster Sizes ===")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"Cluster {cluster_id}: {count} points ({count/n_samples*100:.1f}%)")

# Приклад
evaluate_hierarchical(X_scaled, labels)
```

---

## 8️⃣ Distance threshold (без фіксованого K)

```python
from sklearn.cluster import AgglomerativeClustering

# Замість n_clusters використовуємо distance_threshold
hc = AgglomerativeClustering(
    n_clusters=None,           # None!
    distance_threshold=10,     # поріг відстані
    linkage='ward'
)

labels = hc.fit_predict(X_scaled)

print(f"Знайдено кластерів: {hc.n_clusters_}")
print(f"Кількість листків: {hc.n_leaves_}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title(f'HC with distance_threshold=10 → {hc.n_clusters_} clusters')
plt.colorbar(label='Cluster')
plt.show()
```

---

## 9️⃣ Connectivity constraints (обмеження на сусідів)

```python
from sklearn.neighbors import kneighbors_graph

# Побудувати граф k найближчих сусідів
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# Hierarchical clustering з обмеженням:
# можна об'єднувати тільки сусідів по графу
hc_constrained = AgglomerativeClustering(
    n_clusters=3,
    connectivity=connectivity,
    linkage='ward'
)

labels_constrained = hc_constrained.fit_predict(X)

# Порівняння
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Без обмежень
hc_normal = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_normal = hc_normal.fit_predict(X)

axes[0].scatter(X[:, 0], X[:, 1], c=labels_normal, cmap='viridis', s=50)
axes[0].set_title('Normal HC')
axes[0].grid(True, alpha=0.3)

# З обмеженнями
axes[1].scatter(X[:, 0], X[:, 1], c=labels_constrained, cmap='viridis', s=50)
axes[1].set_title('HC with connectivity constraints')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Використання:** географічні дані, де можна об'єднувати тільки сусідні регіони.

---

## 🔟 Збереження та завантаження

```python
import joblib
from scipy.cluster.hierarchy import linkage

# 1. Навчання
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X_scaled)

# Також зберігаємо linkage matrix для дендрограми
Z = linkage(X_scaled, method='ward')

# 2. Збереження
model_data = {
    'hc': hc,
    'scaler': scaler,
    'linkage_matrix': Z,
    'labels_train': labels,
    'X_train_scaled': X_scaled
}

joblib.dump(model_data, 'hierarchical_model.pkl')

# 3. Завантаження
loaded_data = joblib.load('hierarchical_model.pkl')
loaded_hc = loaded_data['hc']
loaded_scaler = loaded_data['scaler']
loaded_Z = loaded_data['linkage_matrix']

print(f"Кластерів: {loaded_hc.n_clusters_}")

# 4. Дендрограма зі збережених даних
plt.figure(figsize=(12, 5))
dendrogram(loaded_Z)
plt.title('Loaded Dendrogram')
plt.show()
```

**ВАЖЛИВО:** Hierarchical Clustering **НЕ має** методу `predict()` для нових точок!

### Як класифікувати нові точки?

```python
from sklearn.neighbors import KNeighborsClassifier

def predict_hierarchical(new_X, hc, X_train, scaler):
    """
    Передбачення для нових точок через KNN
    
    Логіка: знайти найближчих сусідів з тренувальних даних
            і присвоїти їх кластер
    """
    # Масштабувати нові дані
    new_X_scaled = scaler.transform(new_X)
    
    # KNN на тренувальних даних
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, hc.labels_)
    
    # Передбачення
    predictions = knn.predict(new_X_scaled)
    
    return predictions

# Приклад
new_data = np.array([[1.5, 2.5], [8.0, 8.0]])
new_labels = predict_hierarchical(
    new_data, 
    loaded_hc, 
    loaded_data['X_train_scaled'],
    loaded_scaler
)

print(f"Нові точки належать до кластерів: {new_labels}")
```

---

## 1️⃣1️⃣ BIRCH — для великих даних

```python
from sklearn.cluster import Birch

# BIRCH: Balanced Iterative Reducing and Clustering using Hierarchies
# Добре масштабується на великі дані

birch = Birch(
    n_clusters=3,              # або None для автоматичного вибору
    threshold=0.5,             # поріг для CFTree
    branching_factor=50        # кількість підкластерів у вузлі
)

labels = birch.fit_predict(X_scaled)

print(f"Кластерів: {birch.n_features_in_}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title('BIRCH Clustering')
plt.colorbar(label='Cluster')
plt.show()
```

**Коли використовувати BIRCH:**

- Дані > 10,000 точок
- Потрібна ієрархія, але звичайний HC занадто повільний
- Можна пожертвувати трохи точності заради швидкості

---

## 1️⃣2️⃣ Візуалізація для високовимірних даних (PCA)

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 1. Дані (4D)
iris = load_iris()
X_iris = iris.data

# 2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# 3. Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X_scaled)

# 4. PCA для візуалізації
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# HC результати
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('Hierarchical Clustering (PCA projection)')
axes[0].grid(True, alpha=0.3)

# Справжні мітки
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis', s=50)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[1].set_title('True Labels (PCA projection)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Silhouette
from sklearn.metrics import silhouette_score
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.3f}")
```

---

## 1️⃣3️⃣ Поради та best practices

### 1. Завжди масштабуй дані

```python
# ПОГАНО
hc = AgglomerativeClustering(n_clusters=3)
hc.fit(X)

# ДОБРЕ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
hc = AgglomerativeClustering(n_clusters=3)
hc.fit(X_scaled)
```

---

### 2. Використовуй Ward для загальних випадків

```python
# За замовчуванням
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
```

Якщо не працює, спробуй `'average'` або `'complete'`.

---

### 3. Візуалізуй дендрограму перед вибором K

```python
# ЗАВЖДИ будуй дендрограму перед фіксацією K!
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.show()

# Тільки потім вибирай K
```

---

### 4. Для великих даних: sampling або BIRCH

```python
# Якщо n > 10,000
if len(X) > 10000:
    # Варіант 1: Sampling
    sample_idx = np.random.choice(len(X), size=5000, replace=False)
    X_sample = X[sample_idx]
    Z = linkage(X_sample, method='ward')
    
    # Варіант 2: BIRCH
    from sklearn.cluster import Birch
    birch = Birch(n_clusters=3)
    labels = birch.fit_predict(X)
```

---

### 5. Перевіряй різні linkage

```python
for method in ['ward', 'average', 'complete']:
    hc = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = hc.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"{method}: Silhouette={score:.3f}")
```

---

### 6. Використовуй distance_threshold для автоматичного K

```python
# Замість гадання K, використай distance_threshold
hc = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=10,
    linkage='ward'
)
labels = hc.fit_predict(X_scaled)
print(f"Автоматично знайдено {hc.n_clusters_} кластерів")
```

---

## Чек-лист для Hierarchical Clustering

```python
# ✅ 1. Завантажити дані
X = load_data()

# ✅ 2. EDA
print(X.shape)
print(pd.DataFrame(X).describe())

# ✅ 3. Масштабування (ОБОВ'ЯЗКОВО!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Побудувати дендрограму
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.show()

# ✅ 5. Вибрати K (візуально або через метрики)
optimal_k = 3  # з дендрограми або silhouette

# ✅ 6. Навчання
hc = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
labels = hc.fit_predict(X_scaled)

# ✅ 7. Оцінка якості
evaluate_hierarchical(X_scaled, labels)

# ✅ 8. Візуалізація
visualize_clusters(X, labels)

# ✅ 9. Збереження
joblib.dump({'hc': hc, 'scaler': scaler, 'Z': Z}, 'model.pkl')
```

---

## Порівняння з K-Means і DBSCAN

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_moons
import time

# Дані з нелінійною структурою
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
start = time.time()
kmeans = KMeans(n_clusters=2, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)
time_km = time.time() - start

# DBSCAN
start = time.time()
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_db = dbscan.fit_predict(X_scaled)
time_db = time.time() - start

# Hierarchical
start = time.time()
hc = AgglomerativeClustering(n_clusters=2, linkage='single')
labels_hc = hc.fit_predict(X_scaled)
time_hc = time.time() - start

# Результати
from sklearn.metrics import silhouette_score

print("=== Comparison ===")
print(f"K-Means:      Silhouette={silhouette_score(X_scaled, labels_km):.3f}, Time={time_km:.4f}s")
print(f"DBSCAN:       Silhouette={silhouette_score(X_scaled[labels_db!=-1], labels_db[labels_db!=-1]):.3f}, Time={time_db:.4f}s")
print(f"Hierarchical: Silhouette={silhouette_score(X_scaled, labels_hc):.3f}, Time={time_hc:.4f}s")

# Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, cmap='viridis', s=50)
axes[0].set_title(f'K-Means (Sil={silhouette_score(X_scaled, labels_km):.2f})')

axes[1].scatter(X[:, 0], X[:, 1], c=labels_db, cmap='viridis', s=50)
axes[1].set_title(f'DBSCAN (Sil={silhouette_score(X_scaled[labels_db!=-1], labels_db[labels_db!=-1]):.2f})')

axes[2].scatter(X[:, 0], X[:, 1], c=labels_hc, cmap='viridis', s=50)
axes[2].set_title(f'Hierarchical-Single (Sil={silhouette_score(X_scaled, labels_hc):.2f})')

plt.tight_layout()
plt.show()
```

**Висновок:**

- **K-Means:** швидкий, але погано на складних формах
- **DBSCAN:** добре на складних формах, може знайти noise
- **Hierarchical (single):** добре на складних формах, показує ієрархію

---

## Корисні посилання

- [sklearn AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [sklearn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

---

**Створено для практичного використання Hierarchical Clustering в проєктах** 🚀