# Gaussian Mixture Models — sklearn практика

Повний практичний гайд по використанню GMM в scikit-learn з прикладами коду.

---

## 📦 Основні імпорти

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

# GMM
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Metrics
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Data
from sklearn.datasets import make_blobs, load_iris, make_moons
```

---

## 1️⃣ GaussianMixture — основний клас

### Параметри

```python
GaussianMixture(
    n_components=1,            # Кількість компонент (кластерів)
    covariance_type='full',    # 'full', 'tied', 'diag', 'spherical'
    tol=1e-3,                  # Толерантність для збіжності EM
    reg_covar=1e-6,            # Regularization для стабільності
    max_iter=100,              # Максимум ітерацій EM
    n_init=1,                  # Кількість ініціалізацій
    init_params='kmeans',      # 'kmeans', 'k-means++', 'random', 'random_from_data'
    weights_init=None,         # Початкові ваги (πₖ)
    means_init=None,           # Початкові центри (μₖ)
    precisions_init=None,      # Початкові точності (Σₖ⁻¹)
    random_state=None,         # Seed для відтворюваності
    warm_start=False,          # Використовувати попередні параметри
    verbose=0,                 # Виводити прогрес
    verbose_interval=10        # Частота виводу
)
```

---

### Атрибути після fit

```python
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# Доступні атрибути:
gmm.weights_              # Ваги компонент πₖ (n_components,)
gmm.means_                # Центри μₖ (n_components, n_features)
gmm.covariances_          # Коваріаційні матриці Σₖ (залежить від типу)
gmm.precisions_           # Зворотні коваріації Σₖ⁻¹
gmm.precisions_cholesky_  # Cholesky decomposition для швидкості
gmm.converged_            # Чи зійшовся EM алгоритм
gmm.n_iter_               # Кількість ітерацій до збіжності
gmm.lower_bound_          # Log-likelihood на останній ітерації
```

---

### Методи

```python
# Навчання
gmm.fit(X)

# Передбачення міток (hard clustering)
labels = gmm.predict(X)

# Навчання + передбачення
labels = gmm.fit_predict(X)

# Ймовірності належності (soft clustering)
probs = gmm.predict_proba(X)  # shape: (n_samples, n_components)

# Log-probability для кожної точки
log_probs = gmm.score_samples(X)  # shape: (n_samples,)

# Середня log-likelihood (для порівняння моделей)
avg_log_likelihood = gmm.score(X)

# BIC
bic = gmm.bic(X)

# AIC
aic = gmm.aic(X)

# Семплювання нових точок
new_samples, new_labels = gmm.sample(n_samples=100)
```

---

## 2️⃣ Базовий приклад

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Генерація даних
X, y_true = make_blobs(
    n_samples=300, 
    centers=3, 
    cluster_std=[1.0, 1.5, 0.5],  # різна дисперсія
    random_state=42
)

# 2. GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 3. Передбачення
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

# 4. Результати
print(f"Зійшовся: {gmm.converged_}")
print(f"Ітерацій: {gmm.n_iter_}")
print(f"Log-likelihood: {gmm.lower_bound_:.2f}")
print(f"BIC: {gmm.bic(X):.2f}")
print(f"AIC: {gmm.aic(X):.2f}")

print(f"\nВаги компонент: {gmm.weights_}")
print(f"\nЦентри:\n{gmm.means_}")

# 5. Візуалізація
plt.figure(figsize=(12, 5))

# Hard clustering
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', linewidths=2,
            label='Centers')
plt.title('GMM - Hard Clustering')
plt.legend()
plt.grid(True, alpha=0.3)

# Soft clustering (показати ймовірності для першого компонента)
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=probs[:, 0], cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='P(component 0)')
plt.title('GMM - Soft Clustering (Component 0 probability)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 3️⃣ Вибір кількості компонент (K)

### Метод 1: BIC (рекомендовано)

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Тестуємо різні K
K_range = range(1, 11)
bic_scores = []
aic_scores = []

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

# Візуалізація
plt.figure(figsize=(12, 5))

# BIC
plt.subplot(1, 2, 1)
plt.plot(K_range, bic_scores, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of components (K)')
plt.ylabel('BIC')
plt.title('BIC Score (lower is better)')
plt.grid(True, alpha=0.3)

# Оптимальний K
optimal_k_bic = K_range[np.argmin(bic_scores)]
plt.axvline(x=optimal_k_bic, color='r', linestyle='--', 
            label=f'Optimal K={optimal_k_bic}')
plt.legend()

# AIC
plt.subplot(1, 2, 2)
plt.plot(K_range, aic_scores, 'o-', linewidth=2, markersize=8, color='orange')
plt.xlabel('Number of components (K)')
plt.ylabel('AIC')
plt.title('AIC Score (lower is better)')
plt.grid(True, alpha=0.3)

optimal_k_aic = K_range[np.argmin(aic_scores)]
plt.axvline(x=optimal_k_aic, color='r', linestyle='--',
            label=f'Optimal K={optimal_k_aic}')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Optimal K (BIC): {optimal_k_bic}")
print(f"Optimal K (AIC): {optimal_k_aic}")
```

---

### Метод 2: Комбінований (BIC + Silhouette)

```python
from sklearn.metrics import silhouette_score

K_range = range(2, 11)
results = []

for k in K_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X)
    
    results.append({
        'K': k,
        'BIC': gmm.bic(X),
        'AIC': gmm.aic(X),
        'Silhouette': silhouette_score(X, labels),
        'Log-likelihood': gmm.score(X)
    })

df_results = pd.DataFrame(results)
print(df_results)

# Візуалізація всіх метрик
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# BIC
axes[0, 0].plot(df_results['K'], df_results['BIC'], 'o-')
axes[0, 0].set_title('BIC (min)')
axes[0, 0].set_xlabel('K')
axes[0, 0].grid(True, alpha=0.3)

# AIC
axes[0, 1].plot(df_results['K'], df_results['AIC'], 'o-', color='orange')
axes[0, 1].set_title('AIC (min)')
axes[0, 1].set_xlabel('K')
axes[0, 1].grid(True, alpha=0.3)

# Silhouette
axes[1, 0].plot(df_results['K'], df_results['Silhouette'], 'o-', color='green')
axes[1, 0].set_title('Silhouette Score (max)')
axes[1, 0].set_xlabel('K')
axes[1, 0].grid(True, alpha=0.3)

# Log-likelihood
axes[1, 1].plot(df_results['K'], df_results['Log-likelihood'], 'o-', color='purple')
axes[1, 1].set_title('Log-likelihood (max)')
axes[1, 1].set_xlabel('K')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4️⃣ Типи коваріаційних матриць

### Порівняння всіх типів

```python
from sklearn.datasets import make_blobs

# Дані з різними формами кластерів
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Додамо кореляцію (розтягнемо дані)
transformation = np.array([[2, 1], [0, 1]])
X_transformed = X @ transformation.T

covariance_types = ['spherical', 'diag', 'tied', 'full']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, cov_type in enumerate(covariance_types):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
    labels = gmm.fit_predict(X_transformed)
    
    # Візуалізація
    axes[idx].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                     c=labels, cmap='viridis', s=50, alpha=0.6)
    axes[idx].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                     c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    
    # BIC
    bic = gmm.bic(X_transformed)
    axes[idx].set_title(f'{cov_type.capitalize()} (BIC={bic:.1f})')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Візуалізація еліпсів (форма кластерів)

```python
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def plot_gmm_ellipses(gmm, X, ax=None):
    """Візуалізує GMM кластери з еліпсами коваріації"""
    if ax is None:
        ax = plt.gca()
    
    labels = gmm.predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', alpha=0.6)
    
    # Еліпси для кожного компонента
    for i in range(gmm.n_components):
        if gmm.covariance_type == 'full':
            covariance = gmm.covariances_[i][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariance = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariance = np.diag(gmm.covariances_[i][:2])
        elif gmm.covariance_type == 'spherical':
            covariance = np.eye(2) * gmm.covariances_[i]
        
        # Власні значення та вектори
        v, w = np.linalg.eigh(covariance)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 2 std dev
        u = w[0] / np.linalg.norm(w[0])
        
        # Кут повороту
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        
        # Еліпс
        ell = Ellipse(gmm.means_[i, :2], v[0], v[1], angle=180.0 + angle,
                     edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(ell)
        
        ax.scatter(gmm.means_[i, 0], gmm.means_[i, 1], 
                  c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)

# Приклад
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_transformed)

plt.figure(figsize=(10, 8))
plot_gmm_ellipses(gmm, X_transformed)
plt.title('GMM with Covariance Ellipses')
plt.show()
```

---

## 5️⃣ Soft vs Hard Clustering

```python
from sklearn.mixture import GaussianMixture

# GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Hard clustering (як K-Means)
hard_labels = gmm.predict(X)

# Soft clustering (ймовірності)
soft_probs = gmm.predict_proba(X)

# Приклад для першої точки
print(f"Точка x₁ = {X[0]}")
print(f"Hard label: {hard_labels[0]}")
print(f"Soft probabilities: {soft_probs[0]}")
print(f"  - Component 0: {soft_probs[0, 0]:.2%}")
print(f"  - Component 1: {soft_probs[0, 1]:.2%}")
print(f"  - Component 2: {soft_probs[0, 2]:.2%}")

# Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Hard clustering
axes[0].scatter(X[:, 0], X[:, 1], c=hard_labels, cmap='viridis', s=50)
axes[0].set_title('Hard Clustering (predict)')

# Soft clustering - Component 0
axes[1].scatter(X[:, 0], X[:, 1], c=soft_probs[:, 0], 
               cmap='viridis', s=50, vmin=0, vmax=1)
axes[1].set_title('Soft Clustering - P(Component 0)')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Probability')

# Soft clustering - uncertainty (entropy)
from scipy.stats import entropy
uncertainties = entropy(soft_probs.T)
axes[2].scatter(X[:, 0], X[:, 1], c=uncertainties, cmap='Reds', s=50)
axes[2].set_title('Uncertainty (higher = more uncertain)')
plt.colorbar(axes[2].collections[0], ax=axes[2], label='Entropy')

plt.tight_layout()
plt.show()
```

---

## 6️⃣ Практичні приклади

### Приклад 1: Сегментація клієнтів з ймовірностями

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# 1. Генерація даних клієнтів
np.random.seed(42)
n_customers = 300

data = {
    'Age': np.random.randint(18, 70, n_customers),
    'Income': np.random.randint(20000, 150000, n_customers),
    'SpendingScore': np.random.randint(1, 100, n_customers),
}

df = pd.DataFrame(data)
print(df.head())

# 2. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Вибір K через BIC
bic_scores = []
K_range = range(1, 10)

for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))

optimal_k = K_range[np.argmin(bic_scores)]
print(f"\nOptimal K (BIC): {optimal_k}")

# 4. Фінальна модель
gmm = GaussianMixture(n_components=optimal_k, covariance_type='full', random_state=42)
df['Cluster'] = gmm.fit_predict(X_scaled)

# 5. Додати ймовірності
probs = gmm.predict_proba(X_scaled)
for i in range(optimal_k):
    df[f'Prob_Cluster_{i}'] = probs[:, i]

# 6. Аналіз кластерів
print("\n=== Cluster Analysis ===")
cluster_summary = df.groupby('Cluster').agg({
    'Age': ['mean', 'std'],
    'Income': ['mean', 'std'],
    'SpendingScore': ['mean', 'std']
}).round(2)
print(cluster_summary)

print("\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())

# 7. Знайти "невпевнені" точки (high entropy)
from scipy.stats import entropy
df['Uncertainty'] = entropy(probs.T)

print("\n=== Top 5 most uncertain customers ===")
uncertain_customers = df.nlargest(5, 'Uncertainty')[['Age', 'Income', 'SpendingScore', 'Cluster', 'Uncertainty']]
print(uncertain_customers)

# 8. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Income vs Spending Score
axes[0].scatter(df['Income'], df['SpendingScore'], 
                c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
axes[0].set_xlabel('Income')
axes[0].set_ylabel('Spending Score')
axes[0].set_title('Customer Segmentation')
axes[0].grid(True, alpha=0.3)

# Uncertainty
axes[1].scatter(df['Income'], df['SpendingScore'],
                c=df['Uncertainty'], cmap='Reds', s=50, alpha=0.6)
axes[1].set_xlabel('Income')
axes[1].set_ylabel('Spending Score')
axes[1].set_title('Uncertainty (red = uncertain)')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Entropy')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Приклад 2: Виявлення аномалій через GMM

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# 1. Дані: нормальні точки + аномалії
np.random.seed(42)

# Нормальні точки (3 кластери)
X_normal_1 = np.random.randn(100, 2) * 0.5 + [0, 0]
X_normal_2 = np.random.randn(100, 2) * 0.5 + [3, 3]
X_normal_3 = np.random.randn(80, 2) * 0.5 + [0, 3]

# Аномалії
X_anomalies = np.random.uniform(low=-2, high=5, size=(20, 2))

X = np.vstack([X_normal_1, X_normal_2, X_normal_3, X_anomalies])
y_true = np.array([0]*100 + [1]*100 + [2]*80 + [3]*20)  # 3 = anomaly

# 2. GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 3. Обчислити log-probability для кожної точки
log_probs = gmm.score_samples(X)

# 4. Виявлення аномалій (низька ймовірність)
threshold = np.percentile(log_probs, 5)  # нижні 5%
anomalies_pred = log_probs < threshold

print(f"Threshold: {threshold:.2f}")
print(f"Виявлено аномалій: {anomalies_pred.sum()}")
print(f"Справжніх аномалій: {(y_true == 3).sum()}")

# 5. Метрики
from sklearn.metrics import classification_report

y_pred = (log_probs < threshold).astype(int)
y_true_binary = (y_true == 3).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_true_binary, y_pred, 
                           target_names=['Normal', 'Anomaly']))

# 6. Візуалізація
plt.figure(figsize=(15, 5))

# Справжні мітки
plt.subplot(1, 3, 1)
plt.scatter(X[y_true != 3, 0], X[y_true != 3, 1], c='blue', s=50, alpha=0.6, label='Normal')
plt.scatter(X[y_true == 3, 0], X[y_true == 3, 1], c='red', marker='x', s=100, label='True Anomalies')
plt.title('True Labels')
plt.legend()
plt.grid(True, alpha=0.3)

# GMM кластери
plt.subplot(1, 3, 2)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('GMM Clusters')
plt.grid(True, alpha=0.3)

# Виявлені аномалії
plt.subplot(1, 3, 3)
plt.scatter(X[~anomalies_pred, 0], X[~anomalies_pred, 1], 
           c='blue', s=50, alpha=0.6, label='Normal')
plt.scatter(X[anomalies_pred, 0], X[anomalies_pred, 1],
           c='red', marker='x', s=100, label='Detected Anomalies')
plt.title('GMM Anomaly Detection')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Log-probability розподіл
plt.figure(figsize=(10, 5))
plt.hist(log_probs[y_true != 3], bins=50, alpha=0.7, label='Normal', density=True)
plt.hist(log_probs[y_true == 3], bins=20, alpha=0.7, label='Anomalies', density=True)
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold:.2f}')
plt.xlabel('Log-probability')
plt.ylabel('Density')
plt.title('Log-probability Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Приклад 3: Генерація нових даних

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 1. Навчити GMM на реальних даних
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 2. Згенерувати нові точки
n_new_samples = 300
X_new, labels_new = gmm.sample(n_samples=n_new_samples)

# 3. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Оригінальні дані
axes[0].scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.6)
axes[0].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[0].set_title('Original Data')
axes[0].grid(True, alpha=0.3)

# Згеноровані дані
axes[1].scatter(X_new[:, 0], X_new[:, 1], c=labels_new, cmap='viridis', s=50, alpha=0.6)
axes[1].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[1].set_title('Generated Data (GMM samples)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Порівняння розподілів
print("=== Component distribution ===")
print(f"Original: {np.bincount(gmm.predict(X)) / len(X)}")
print(f"Generated: {np.bincount(labels_new) / len(labels_new)}")
```

---

## 7️⃣ GMM vs K-Means

```python
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Дані з еліптичними кластерами
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
transformation = np.array([[2, 0.5], [0.5, 1]])
X_elliptical = X @ transformation.T

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_elliptical)

# GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X_elliptical)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# K-Means
axes[0].scatter(X_elliptical[:, 0], X_elliptical[:, 1], 
               c=labels_kmeans, cmap='viridis', s=50, alpha=0.6)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[0].set_title(f'K-Means\nSilhouette: {silhouette_score(X_elliptical, labels_kmeans):.3f}')
axes[0].grid(True, alpha=0.3)

# GMM з еліпсами
axes[1].scatter(X_elliptical[:, 0], X_elliptical[:, 1],
               c=labels_gmm, cmap='viridis', s=50, alpha=0.6)
plot_gmm_ellipses(gmm, X_elliptical, ax=axes[1])
axes[1].set_title(f'GMM (full covariance)\nSilhouette: {silhouette_score(X_elliptical, labels_gmm):.3f}')

plt.tight_layout()
plt.show()

print(f"\nK-Means Silhouette: {silhouette_score(X_elliptical, labels_kmeans):.3f}")
print(f"GMM Silhouette: {silhouette_score(X_elliptical, labels_gmm):.3f}")
```

---

## 8️⃣ Оцінка якості

```python
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

def evaluate_gmm(X, gmm, y_true=None):
    """
    Оцінити якість GMM кластеризації
    """
    labels = gmm.predict(X)
    
    print("=== GMM Results ===")
    print(f"Компонент: {gmm.n_components}")
    print(f"Зійшовся: {gmm.converged_}")
    print(f"Ітерацій: {gmm.n_iter_}")
    
    print(f"\n=== Model Selection ===")
    print(f"BIC: {gmm.bic(X):.2f}")
    print(f"AIC: {gmm.aic(X):.2f}")
    print(f"Log-likelihood: {gmm.score(X):.2f}")
    
    print(f"\n=== Clustering Metrics ===")
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Davies-Bouldin Index: {db_score:.3f}")
    print(f"Calinski-Harabasz Score: {ch_score:.1f}")
    
    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        print(f"\nAdjusted Rand Index: {ari:.3f}")
    
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
        print(f"Cluster {cluster_id}: {count} points ({count/len(X)*100:.1f}%)")
    
    # Ваги компонент
    print(f"\n=== Component Weights ===")
    for i, weight in enumerate(gmm.weights_):
        print(f"Component {i}: {weight:.3f}")

# Приклад
evaluate_gmm(X_scaled, gmm, y_true=y_true)
```

---

## 9️⃣ Візуалізація для високовимірних даних (PCA)

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 1. Дані (4D)
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# 2. Масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# 3. GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)
probs_gmm = gmm.predict_proba(X_scaled)

# 4. PCA для візуалізації
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# GMM hard clustering
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_gmm, cmap='viridis', s=50, alpha=0.6)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('GMM - Hard Clustering')
axes[0].grid(True, alpha=0.3)

# GMM soft clustering (uncertainty)
from scipy.stats import entropy
uncertainty = entropy(probs_gmm.T)
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=uncertainty, cmap='Reds', s=50, alpha=0.6)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[1].set_title('GMM - Uncertainty')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Entropy')
axes[1].grid(True, alpha=0.3)

# Справжні мітки
axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_iris, cmap='viridis', s=50, alpha=0.6)
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[2].set_title('True Labels')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Метрики
from sklearn.metrics import adjusted_rand_score
print(f"Adjusted Rand Index: {adjusted_rand_score(y_iris, labels_gmm):.3f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, labels_gmm):.3f}")
```

---

## 🔟 Збереження та завантаження

```python
import joblib

# 1. Навчання
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# 2. Збереження
model_data = {
    'gmm': gmm,
    'scaler': scaler,
    'optimal_k': 3,
    'bic': gmm.bic(X_scaled),
    'aic': gmm.aic(X_scaled)
}

joblib.dump(model_data, 'gmm_model.pkl')

# 3. Завантаження
loaded_data = joblib.load('gmm_model.pkl')
loaded_gmm = loaded_data['gmm']
loaded_scaler = loaded_data['scaler']

print(f"Оптимальний K: {loaded_data['optimal_k']}")
print(f"BIC: {loaded_data['bic']:.2f}")

# 4. Використання на нових даних
new_data = np.array([[25, 50000, 75]])
new_data_scaled = loaded_scaler.transform(new_data)

# Hard clustering
new_label = loaded_gmm.predict(new_data_scaled)
print(f"\nНова точка належить кластеру: {new_label[0]}")

# Soft clustering
new_probs = loaded_gmm.predict_proba(new_data_scaled)
print(f"Ймовірності: {new_probs[0]}")
for i, prob in enumerate(new_probs[0]):
    print(f"  Component {i}: {prob:.2%}")
```

---

## 1️⃣1️⃣ Bayesian GMM (автоматичний вибір K)

```python
from sklearn.mixture import BayesianGaussianMixture

# BayesianGMM автоматично "вимикає" зайві компоненти
# через Dirichlet Process Prior

# Задаємо МАКСИМАЛЬНУ кількість компонент
bgmm = BayesianGaussianMixture(
    n_components=10,           # максимум компонент
    covariance_type='full',
    weight_concentration_prior=0.01,  # менше = більше регуляризації
    random_state=42
)

bgmm.fit(X_scaled)

# Ефективна кількість компонент
effective_components = (bgmm.weights_ > 0.01).sum()
print(f"Ефективних компонент: {effective_components}")

# Ваги
print(f"\nВаги компонент:")
for i, weight in enumerate(bgmm.weights_):
    if weight > 0.01:
        print(f"  Component {i}: {weight:.3f}")

# Візуалізація
labels_bgmm = bgmm.predict(X_scaled)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_bgmm, cmap='viridis', s=50, alpha=0.6)
plt.title(f'Bayesian GMM (effective components: {effective_components})')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 1️⃣2️⃣ Поради та best practices

### 1. Завжди масштабуй дані

```python
# ПОГАНО
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# ДОБРЕ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3)
gmm.fit(X_scaled)
```

---

### 2. Використовуй BIC для вибору K

```python
# ЗАВЖДИ перевіряй BIC для різних K
bic_scores = [GaussianMixture(n_components=k).fit(X).bic(X) 
              for k in range(1, 11)]
optimal_k = np.argmin(bic_scores) + 1
```

---

### 3. Починай з 'full' covariance

```python
# За замовчуванням використовуй 'full' (найгнучкіше)
gmm = GaussianMixture(n_components=3, covariance_type='full')

# Якщо повільно або переобучення, спробуй 'tied' або 'diag'
```

---

### 4. Використовуй n_init > 1 для стабільності

```python
# Запустити EM кілька разів
gmm = GaussianMixture(n_components=3, n_init=10, random_state=42)
```

---

### 5. Regularization для уникнення сингулярностей

```python
# Додати reg_covar для стабільності
gmm = GaussianMixture(
    n_components=3,
    reg_covar=1e-6  # невелика константа
)
```

---

### 6. Перевіряй збіжність

```python
gmm.fit(X)
if not gmm.converged_:
    print("⚠️ EM не зійшовся! Збільш max_iter")
```

---

### 7. Використовуй soft clustering для uncertainty

```python
# Замість hard labels
labels = gmm.predict(X)

# Використовуй ймовірності
probs = gmm.predict_proba(X)

# Знайди невпевнені точки
from scipy.stats import entropy
uncertainty = entropy(probs.T)
uncertain_points = X[uncertainty > 1.0]  # високий ентропія
```

---

## Чек-лист для GMM

```python
# ✅ 1. Завантажити дані
X = load_data()

# ✅ 2. EDA
print(X.shape)
print(pd.DataFrame(X).describe())

# ✅ 3. Масштабування (ОБОВ'ЯЗКОВО!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 4. Вибір K через BIC
bic_scores = []
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))

optimal_k = np.argmin(bic_scores) + 1
print(f"Optimal K: {optimal_k}")

# ✅ 5. Навчання GMM
gmm = GaussianMixture(
    n_components=optimal_k,
    covariance_type='full',
    n_init=10,
    random_state=42
)
gmm.fit(X_scaled)

# ✅ 6. Перевірка збіжності
if not gmm.converged_:
    print("⚠️ Не зійшовся!")

# ✅ 7. Hard та soft clustering
labels = gmm.predict(X_scaled)
probs = gmm.predict_proba(X_scaled)

# ✅ 8. Оцінка якості
evaluate_gmm(X_scaled, gmm)

# ✅ 9. Візуалізація
visualize_gmm(X, labels, probs, gmm)

# ✅ 10. Збереження
joblib.dump({'gmm': gmm, 'scaler': scaler}, 'gmm_model.pkl')
```

---

## Порівняння GMM з іншими методами

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import time

# Дані
X, y_true = make_blobs(n_samples=500, centers=3, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

methods = {
    'K-Means': KMeans(n_clusters=3, random_state=42),
    'GMM': GaussianMixture(n_components=3, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Hierarchical': AgglomerativeClustering(n_clusters=3)
}

results = []

for name, model in methods.items():
    start = time.time()
    labels = model.fit_predict(X_scaled)
    elapsed = time.time() - start
    
    # Метрики
    if name == 'DBSCAN':
        mask = labels != -1
        if sum(mask) > 0 and len(set(labels[mask])) > 1:
            sil = silhouette_score(X_scaled[mask], labels[mask])
        else:
            sil = -1
    else:
        sil = silhouette_score(X_scaled, labels)
    
    ari = adjusted_rand_score(y_true, labels)
    
    results.append({
        'Method': name,
        'Silhouette': sil,
        'ARI': ari,
        'Time': elapsed
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
```

---

## Корисні посилання

- [sklearn GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [sklearn BayesianGaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)
- [GMM Tutorial](https://scikit-learn.org/stable/modules/mixture.html)

---

**Створено для практичного використання GMM в проєктах** 🚀
