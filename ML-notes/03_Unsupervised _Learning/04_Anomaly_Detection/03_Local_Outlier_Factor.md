# Local Outlier Factor (LOF)

## Що це?

**LOF (Local Outlier Factor)** — це **density-based** алгоритм для виявлення **локальних аномалій** (outliers), що порівнює **локальну щільність** точки з щільністю її сусідів. Точки в регіонах з низькою щільністю відносно сусідів вважаються аномаліями.

**Головна ідея:** аномалія — це точка, чия локальна щільність значно нижча за щільність її сусідів. LOF вимірює наскільки "ізольованою" є точка порівняно з оточенням.

## Навіщо потрібен?

- 🎯 **Локальні outliers** — аномалії в локальному контексті
- 📊 **Density-based** — враховує структуру даних
- 🔍 **Cluster outliers** — точки далеко від кластерів
- 🌐 **Variable density** — працює з різною щільністю
- 📈 **Degree of outlierness** — не binary, а score
- 🏥 **Medical diagnosis** — локальні аномалії в аналізах
- 🏭 **Sensor data** — несправності обладнання

## Коли використовувати?

**Потрібно:**
- **Локальні outliers** — важлива локальна структура
- **Variable density** — різні щільності в даних
- **Clusters** — outliers між/навколо кластерів
- **Interpretable scores** — LOF score має значення
- **Середні дані** (100-50,000 зразків)
- **Numerical features** — неперервні дані

**Не потрібно:**
- **Дуже великі дані** (> 100K) → Isolation Forest
- **High-dimensional** (> 50 features) → Isolation Forest
- **Global outliers** тільки → Isolation Forest OK
- **Real-time streaming** → простіші методи
- **Швидкість критична** → Isolation Forest

---

## Інтуїція

### Локальні vs Глобальні outliers

**Проблема з глобальними методами:**

```
Cluster 1 (dense):        Cluster 2 (sparse):
    ●●●●●                     ○  ○
    ●●●●●                     ○  ○
    ●●●●●                   ○  ★  ○
    ●●●●●                     ○  ○
                              ○  ○

★ — локальний outlier в Cluster 2
Глобально: нормальна відстань від інших
Локально: далеко від своїх сусідів (Cluster 2)

Звичайні методи: ✗ пропустять
LOF: ✓ знайде!
```

### Приклад

```
Dense region:              Sparse region:
   ●●●●●                     ○    ○
   ●●●●●                   ○  ★    ○
   ●●●●●                     ○    ○
   ●●●●●

Точка в dense:             Точка ★ в sparse:
- Близько до багатьох      - Далеко від сусідів
- LOF ≈ 1 (normal)         - LOF > 1 (outlier!)
```

---

## Математика

### 1. k-distance

**k-distance(A)** — відстань від точки A до k-го найближчого сусіда.

```python
k = 5  # parameter

Point A:
    ●₅
  ●₄  ●₃
 A ●₂ ●₁

k-distance(A) = distance to 5th neighbor
```

### 2. Reachability Distance

**Реальна відстань з "порогом":**

$$\text{reach-dist}_k(A, B) = \max\{\text{k-distance}(B), d(A, B)\}$$

**Інтуїція:**
- Якщо A далеко від B → використовуй реальну відстань
- Якщо A близько → використовуй k-distance(B) як мінімум

**Чому?** Згладжує статистичні флуктуації для близьких точок.

```
B's k-neighborhood:
    ●●●●●
    ● B ●   A далеко
    ●●●●●     ↓
              ●A

reach-dist(A, B) = real distance (A далеко)

    ●●●●●
    ● B ●
    ●●A●●   A близько

reach-dist(A, B) = k-distance(B) (використовуй поріг)
```

### 3. Local Reachability Density (LRD)

**Локальна щільність точки A:**

$$\text{LRD}_k(A) = \frac{1}{\frac{\sum_{B \in N_k(A)} \text{reach-dist}_k(A, B)}{|N_k(A)|}}$$

де $N_k(A)$ — k найближчих сусідів A.

**Інтуїція:**
- LRD = 1 / (середня reachability distance до сусідів)
- **Висока щільність** → малі відстані → **високий LRD**
- **Низька щільність** → великі відстані → **низький LRD**

### 4. Local Outlier Factor (LOF)

**Порівняння щільності A з сусідами:**

$$\text{LOF}_k(A) = \frac{\sum_{B \in N_k(A)} \frac{\text{LRD}_k(B)}{\text{LRD}_k(A)}}{|N_k(A)|}$$

**Спрощено:**

$$\text{LOF}_k(A) = \frac{\text{Average LRD of neighbors}}{\text{LRD of A}}$$

**Інтерпретація:**

```
LOF ≈ 1     → Normal (схожа щільність з сусідами)
LOF > 1     → Outlier (нижча щільність)
LOF >> 1    → Strong outlier (набагато нижча)

Typically:
LOF < 1.5   → Normal
LOF > 2.0   → Outlier
```

---

## Алгоритм

### Покроковий процес

```
Дано: Dataset X, parameter k

FOR кожної точки A в X:
    
    1. Знайти k-distance(A)
       - Відстань до k-го найближчого сусіда
    
    2. Знайти k-nearest neighbors N_k(A)
    
    3. Для кожного сусіда B в N_k(A):
       Обчислити reach-dist_k(A, B) = max{k-distance(B), d(A, B)}
    
    4. Обчислити LRD_k(A):
       LRD_k(A) = k / Σ reach-dist_k(A, B)
    
    5. Обчислити LOF_k(A):
       LOF_k(A) = (Σ LRD_k(B) / LRD_k(A)) / k
       де B в N_k(A)

RETURN LOF scores для всіх точок
```

### Псевдокод

```python
def LOF(X, k):
    n = len(X)
    lof_scores = []
    
    # Обчислити всі відстані
    distances = compute_distances(X)
    
    FOR i in range(n):
        # 1. k-distance та k-neighbors
        k_dist = k_distance(distances[i], k)
        neighbors = k_neighbors(distances[i], k)
        
        # 2. Reachability distances
        reach_dists = []
        FOR j in neighbors:
            rd = max(k_distance(distances[j], k), distances[i][j])
            reach_dists.append(rd)
        
        # 3. Local Reachability Density
        lrd_i = k / sum(reach_dists)
        
        # 4. LOF score
        lrd_neighbors = []
        FOR j in neighbors:
            lrd_j = compute_lrd(j, k)
            lrd_neighbors.append(lrd_j)
        
        lof_i = mean(lrd_neighbors) / lrd_i
        lof_scores.append(lof_i)
    
    RETURN lof_scores
```

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs

# 1. Генерувати дані з різною щільністю
np.random.seed(42)

# Dense cluster
X_dense, _ = make_blobs(
    n_samples=200,
    centers=[[0, 0]],
    cluster_std=0.3,
    random_state=42
)

# Sparse cluster
X_sparse, _ = make_blobs(
    n_samples=50,
    centers=[[4, 4]],
    cluster_std=0.8,
    random_state=42
)

# Combine
X_normal = np.vstack([X_dense, X_sparse])

# Додати outliers
X_outliers = np.array([
    [0, 3],      # Between clusters
    [-2, -2],    # Far from dense
    [6, 6],      # Far from sparse (локальний outlier!)
    [2, 2]       # Between clusters
])

X = np.vstack([X_normal, X_outliers])
y_true = np.array([0]*250 + [1]*4)  # 0=normal, 1=outlier

print(f"Total points: {len(X)}")
print(f"Normal: {(y_true == 0).sum()}")
print(f"Outliers: {(y_true == 1).sum()}")

# 2. Fit LOF
clf = LocalOutlierFactor(
    n_neighbors=20,        # k parameter
    contamination=0.05,    # Expected fraction of outliers
    novelty=False          # False = fit_predict, True = fit + predict new
)

# fit_predict на training data
y_pred = clf.fit_predict(X)  # 1=inlier, -1=outlier

# Negative outlier factor (чим менше, тим більша аномалія)
lof_scores = -clf.negative_outlier_factor_

print(f"\n=== Predictions ===")
print(f"Predicted outliers: {(y_pred == -1).sum()}")

# 3. Metrics
from sklearn.metrics import classification_report, confusion_matrix

y_pred_binary = (y_pred == -1).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_binary,
                           target_names=['Normal', 'Outlier']))

# 4. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: True labels
axes[0].scatter(X[y_true == 0, 0], X[y_true == 0, 1],
               c='blue', s=20, alpha=0.6, label='Normal')
axes[0].scatter(X[y_true == 1, 0], X[y_true == 1, 1],
               c='red', s=100, marker='x', linewidths=2, label='True Outlier')
axes[0].set_title('True Labels', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: LOF scores
scatter = axes[1].scatter(X[:, 0], X[:, 1],
                         c=lof_scores, cmap='RdYlBu_r',
                         s=30, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.colorbar(scatter, ax=axes[1], label='LOF Score\n(higher = more anomalous)')

# Mark predicted outliers
outlier_mask = y_pred == -1
axes[1].scatter(X[outlier_mask, 0], X[outlier_mask, 1],
               facecolors='none', edgecolors='red',
               s=150, linewidths=2, label='Predicted Outliers')

axes[1].set_title('LOF Scores and Predictions', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Top outliers
top_outlier_indices = np.argsort(lof_scores)[-10:][::-1]

print("\n=== Top 10 Outliers ===")
for idx in top_outlier_indices:
    print(f"Index {idx}: LOF = {lof_scores[idx]:.3f}, "
          f"Point = {X[idx]}, "
          f"True label = {'Outlier' if y_true[idx] == 1 else 'Normal'}")
```

### Novelty Detection mode

```python
# LOF для novelty detection (train на normal, predict на new)

# Train тільки на normal data
X_train = X_normal.copy()

# LOF з novelty=True
clf_novelty = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=True  # ← Enable predict для нових даних
)

clf_novelty.fit(X_train)

# Test data (нові точки)
X_test = np.vstack([
    X_sparse[:10],     # Normal from sparse cluster
    X_outliers         # Outliers
])

y_test_true = np.array([0]*10 + [1]*4)

# Predict на нових даних
y_test_pred = clf_novelty.predict(X_test)  # Тепер працює!
scores_test = clf_novelty.score_samples(X_test)

print("\n=== Novelty Detection ===")
print(f"Test set: {len(X_test)} samples")
print(f"Predicted outliers: {(y_test_pred == -1).sum()}")

# Візуалізація
plt.figure(figsize=(10, 7))

# Training data
plt.scatter(X_train[:, 0], X_train[:, 1],
           c='blue', s=20, alpha=0.3, label='Training (Normal)')

# Test normal
plt.scatter(X_test[y_test_true == 0, 0], X_test[y_test_true == 0, 1],
           c='green', s=50, marker='s', alpha=0.7, label='Test Normal')

# Test outliers
plt.scatter(X_test[y_test_true == 1, 0], X_test[y_test_true == 1, 1],
           c='red', s=100, marker='x', linewidths=2, label='Test Outlier')

# Predicted outliers
pred_outliers = y_test_pred == -1
plt.scatter(X_test[pred_outliers, 0], X_test[pred_outliers, 1],
           facecolors='none', edgecolors='orange',
           s=200, linewidths=2, label='Predicted Outlier')

plt.title('LOF Novelty Detection', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Порівняння різних k

```python
# Вплив параметра k (n_neighbors)

k_values = [5, 10, 20, 50]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    print(f"\nk = {k}")
    
    clf = LocalOutlierFactor(n_neighbors=k, contamination=0.05)
    y_pred = clf.fit_predict(X)
    lof_scores = -clf.negative_outlier_factor_
    
    # Plot
    scatter = axes[idx].scatter(X[:, 0], X[:, 1],
                               c=lof_scores, cmap='RdYlBu_r',
                               s=30, alpha=0.7, edgecolors='black',
                               linewidths=0.5)
    
    plt.colorbar(scatter, ax=axes[idx], label='LOF Score')
    
    # Mark predicted outliers
    outliers = y_pred == -1
    axes[idx].scatter(X[outliers, 0], X[outliers, 1],
                     facecolors='none', edgecolors='red',
                     s=150, linewidths=2)
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, (y_pred == -1).astype(int))
    
    axes[idx].set_title(f'k = {k} (Accuracy: {acc:.2f})',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Real example: Credit Card Fraud

```python
# Симулюємо credit card transactions

np.random.seed(42)

# Normal transactions (2 типи поведінки)
# Type 1: Regular small purchases
n_regular = 4000
amount_regular = np.random.lognormal(mean=3, sigma=0.5, size=n_regular)
time_regular = np.random.uniform(0, 24, size=n_regular)

# Type 2: Occasional large purchases
n_large = 1000
amount_large = np.random.lognormal(mean=5, sigma=0.3, size=n_large)
time_large = np.random.uniform(0, 24, size=n_large)

X_normal = np.vstack([
    np.column_stack([amount_regular, time_regular]),
    np.column_stack([amount_large, time_large])
])

# Fraudulent transactions (різні patterns)
n_fraud = 100

# Type 1: Very high amounts at night
amount_fraud1 = np.random.uniform(1000, 5000, size=n_fraud//2)
time_fraud1 = np.random.uniform(2, 5, size=n_fraud//2)  # 2-5 AM

# Type 2: Rapid succession (локальний outlier за часом)
amount_fraud2 = np.random.uniform(300, 800, size=n_fraud//2)
time_fraud2 = np.random.choice([3.1, 3.2, 3.3, 3.4], size=n_fraud//2)  # Clustered in time

X_fraud = np.vstack([
    np.column_stack([amount_fraud1, time_fraud1]),
    np.column_stack([amount_fraud2, time_fraud2])
])

# Combine
X_transactions = np.vstack([X_normal, X_fraud])
y_true = np.array([0]*len(X_normal) + [1]*n_fraud)

print(f"Total transactions: {len(X_transactions)}")
print(f"Fraud rate: {(y_true == 1).sum() / len(y_true):.2%}")

# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transactions)

# LOF
clf = LocalOutlierFactor(
    n_neighbors=30,
    contamination=0.02,  # Expect 2% fraud
    novelty=False
)

y_pred = clf.fit_predict(X_scaled)
lof_scores = -clf.negative_outlier_factor_

# Metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

y_pred_binary = (y_pred == -1).astype(int)

print("\n=== Fraud Detection Results ===")
print(f"Precision: {precision_score(y_true, y_pred_binary):.3f}")
print(f"Recall: {recall_score(y_true, y_pred_binary):.3f}")
print(f"F1-Score: {f1_score(y_true, y_pred_binary):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_true, lof_scores):.3f}")

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# True labels
axes[0].scatter(X_transactions[y_true == 0, 0],
               X_transactions[y_true == 0, 1],
               c='green', s=5, alpha=0.3, label='Legitimate')
axes[0].scatter(X_transactions[y_true == 1, 0],
               X_transactions[y_true == 1, 1],
               c='red', s=30, marker='x', linewidths=1.5, label='Fraud')
axes[0].set_xlabel('Transaction Amount ($)')
axes[0].set_ylabel('Time of Day (hour)')
axes[0].set_title('True Labels', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LOF predictions
scatter = axes[1].scatter(X_transactions[:, 0], X_transactions[:, 1],
                         c=lof_scores, cmap='RdYlBu_r',
                         s=5, alpha=0.5)
plt.colorbar(scatter, ax=axes[1], label='LOF Score')

# Predicted fraud
fraud_mask = y_pred == -1
axes[1].scatter(X_transactions[fraud_mask, 0],
               X_transactions[fraud_mask, 1],
               facecolors='none', edgecolors='red',
               s=50, linewidths=1.5, label='Predicted Fraud')

axes[1].set_xlabel('Transaction Amount ($)')
axes[1].set_ylabel('Time of Day (hour)')
axes[1].set_title('LOF Predictions', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Параметри

### Основні

```python
LocalOutlierFactor(
    n_neighbors=20,         # k parameter (критичний!)
    contamination=0.1,      # Expected fraction outliers
    novelty=False,          # False=fit_predict, True=fit+predict
    algorithm='auto',       # 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski',     # Distance metric
    p=2                     # Minkowski parameter (2=euclidean)
)
```

| Параметр | Опис | Типові значення | Рекомендації |
|----------|------|-----------------|--------------|
| **n_neighbors** | k (розмір околу) | 10-50 | 20 (default) добре для початку |
| **contamination** | % outliers | 0.01-0.2 | Базується на domain knowledge |
| **novelty** | Mode | False/True | False для fit_predict, True для predict new |
| **metric** | Distance | 'euclidean', 'manhattan' | 'euclidean' для більшості |

### n_neighbors (k) — найважливіший!

**Вплив:**

```python
k = 5     # Малий k → чутливий до локальних флуктуацій
k = 20    # Середній → баланс
k = 50    # Великий k → більш глобальна перспектива
```

**Візуально:**

```
Small k (5):
    ●●●
    ● ● ← може визначити як outlier (мало сусідів)
    ●●●

Large k (50):
    ●●●●●●●●●
    ●       ●
    ●   ●   ● ← нормальний (багато сусідів в радіусі)
    ●       ●
    ●●●●●●●●●
```

**Рекомендації:**
- **Щільні дані:** k = 10-20
- **Розріджені дані:** k = 30-50
- **Невизначеність:** спробувати кілька k та порівняти

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Локальні outliers** | Знаходить outliers в локальному контексті |
| **Variable density** | Працює з різною щільністю |
| **Interpretable score** | LOF має чітке значення |
| **No assumptions** | Не припускає розподілів |
| **Cluster outliers** | Добре для outliers між кластерами |
| **Theory** | Solid mathematical foundation |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Повільний** | O(n²) — обчислення всіх відстаней |
| **Не масштабується** | Погано на > 50K |
| **High-dimensional** | Curse of dimensionality |
| **Parameter sensitive** | k сильно впливає |
| **Memory** | Зберігає distance matrix |
| **No global view** | Може пропустити глобальні patterns |

---

## Порівняння з іншими методами

| Метод | Локальні outliers | Швидкість | Variable density | Масштабованість |
|-------|-------------------|-----------|------------------|-----------------|
| **LOF** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Isolation Forest** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **One-Class SVM** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **DBSCAN** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**Коли що:**
- **Локальні outliers + variable density** → LOF ✓
- **Великі дані + швидкість** → Isolation Forest ✓
- **Global outliers** → Isolation Forest ✓
- **Clustering + outliers** → DBSCAN ✓

---

## Практичні поради 💡

### 1. Експериментуй з n_neighbors

```python
# Спробуй різні k
for k in [10, 20, 30, 50]:
    clf = LocalOutlierFactor(n_neighbors=k)
    y_pred = clf.fit_predict(X)
    
    # Evaluate
    score = evaluate(y_pred, y_true)
    print(f"k={k}: score={score:.3f}")
```

### 2. Feature scaling важливий

```python
# ✅ LOF чутливий до scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LocalOutlierFactor()
clf.fit_predict(X_scaled)
```

### 3. Використовуй scores для ranking

```python
# Не тільки binary predictions
clf = LocalOutlierFactor(novelty=False)
y_pred = clf.fit_predict(X)

# LOF scores (negative_outlier_factor_)
lof_scores = -clf.negative_outlier_factor_

# Top-N найбільш аномальних
top_n = 10
top_outliers = np.argsort(lof_scores)[-top_n:][::-1]

for idx in top_outliers:
    print(f"Rank {idx}: LOF = {lof_scores[idx]:.3f}")
```

### 4. novelty mode для production

```python
# Train на clean normal data
X_train_normal = clean_data

clf = LocalOutlierFactor(novelty=True)
clf.fit(X_train_normal)

# Predict на новий stream
for new_sample in data_stream:
    is_anomaly = clf.predict([new_sample])[0] == -1
    
    if is_anomaly:
        alert()
```

### 5. Combine з іншими методами

```python
# Ensemble: LOF + Isolation Forest
from sklearn.ensemble import IsolationForest

# LOF
lof = LocalOutlierFactor(novelty=True)
lof.fit(X_train)
lof_scores = lof.score_samples(X_test)

# Isolation Forest
iforest = IsolationForest()
iforest.fit(X_train)
if_scores = iforest.score_samples(X_test)

# Normalize та combine
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
lof_norm = scaler.fit_transform(lof_scores.reshape(-1, 1)).ravel()
if_norm = scaler.fit_transform(if_scores.reshape(-1, 1)).ravel()

# Voting або averaging
ensemble_scores = (lof_norm + if_norm) / 2
```

### 6. Dimensionality reduction спочатку

```python
# Для high-dimensional data
from sklearn.decomposition import PCA

# PCA спочатку
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# LOF на reduced
clf = LocalOutlierFactor()
y_pred = clf.fit_predict(X_pca)
```

### 7. Cross-validation для k

```python
# Якщо є validation set
k_values = [10, 20, 30, 40, 50]
best_f1 = 0
best_k = None

for k in k_values:
    clf = LocalOutlierFactor(n_neighbors=k, novelty=False)
    y_pred = clf.fit_predict(X_val)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val_true, (y_pred == -1).astype(int))
    
    if f1 > best_f1:
        best_f1 = f1
        best_k = k

print(f"Best k: {best_k}")
```

### 8. Visualize neighborhoods

```python
# Для debugging — подивитись на neighborhoods
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=20)
nbrs.fit(X)

# Для конкретної точки
point_idx = 0
distances, indices = nbrs.kneighbors([X[point_idx]])

plt.scatter(X[:, 0], X[:, 1], c='lightgray', s=20, alpha=0.5)
plt.scatter(X[point_idx, 0], X[point_idx, 1], 
           c='red', s=100, marker='*', label='Query point')
plt.scatter(X[indices[0], 0], X[indices[0], 1],
           c='blue', s=50, alpha=0.7, label='Neighbors')
plt.legend()
plt.show()
```

### 9. Incremental updates

```python
# LOF не має incremental learning
# Для streaming: періодично refit

class StreamingLOF:
    def __init__(self, window_size=1000, k=20):
        self.window_size = window_size
        self.k = k
        self.clf = LocalOutlierFactor(n_neighbors=k, novelty=True)
        self.buffer = []
        self.scaler = StandardScaler()
    
    def add_sample(self, x):
        self.buffer.append(x)
        
        if len(self.buffer) >= self.window_size:
            # Refit
            X_train = np.array(self.buffer[-self.window_size:])
            X_scaled = self.scaler.fit_transform(X_train)
            self.clf.fit(X_scaled)
            
            # Keep half
            self.buffer = self.buffer[-self.window_size//2:]
    
    def predict(self, x):
        x_scaled = self.scaler.transform([x])
        return self.clf.predict(x_scaled)[0]
```

### 10. Domain-specific distance metrics

```python
# Для специфічних даних — custom metric
def custom_distance(x, y):
    # Domain-specific logic
    return np.sum((x - y) ** 2)

clf = LocalOutlierFactor(
    n_neighbors=20,
    metric=custom_distance
)
```

---

## Поширені помилки ❌

### 1. Не scale features

```python
# ❌ Features в різних scales
clf = LocalOutlierFactor()
clf.fit_predict(X_raw)

# ✅ Scale спочатку
X_scaled = StandardScaler().fit_transform(X_raw)
clf.fit_predict(X_scaled)
```

### 2. Неправильний k

```python
# ❌ k = 5 (занадто малий для 10,000 points)
# ❌ k = 100 (занадто великий для 200 points)

# ✅ Розумний k відносно dataset size
k = min(50, len(X) // 10)
```

### 3. novelty=False для production

```python
# ❌ novelty=False не може predict нові дані
clf = LocalOutlierFactor(novelty=False)
clf.fit(X_train)
# clf.predict(X_test)  # Error!

# ✅ novelty=True для predict
clf = LocalOutlierFactor(novelty=True)
clf.fit(X_train)
clf.predict(X_test)  # OK!
```

### 4. Використовувати на дуже великих даних

```python
# ❌ 100,000+ points
# Дуже повільно!

# ✅ Sample або використай Isolation Forest
if len(X) > 50000:
    # Use Isolation Forest
    from sklearn.ensemble import IsolationForest
```

### 5. Ігнорувати high-dimensional curse

```python
# ❌ 100+ features
# LOF погано працює через curse of dimensionality

# ✅ Dimensionality reduction
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X)
clf.fit_predict(X_reduced)
```

---

## Реальні застосування

### 1. Network Intrusion Detection

```python
# Локальні аномалії в network traffic
# Train на normal traffic, detect attacks

clf = LocalOutlierFactor(n_neighbors=30, novelty=True)
clf.fit(normal_traffic_features)

# Real-time
for packet in stream:
    is_attack = clf.predict([packet_features])[0] == -1
    
    if is_attack:
        block_and_alert()
```

### 2. Medical Diagnosis

```python
# Локальні аномалії в test results
# Враховує що норма різна для різних груп

clf = LocalOutlierFactor(n_neighbors=20)
lof_scores = -clf.fit_predict(patient_data)

# Patients з high LOF → додаткові обстеження
high_risk = lof_scores > 2.0
flag_for_review(high_risk_patients)
```

### 3. Manufacturing Quality Control

```python
# Дефекти можуть бути локальними
# (різні типи products мають різні норми)

clf = LocalOutlierFactor(n_neighbors=25)
y_pred = clf.fit_predict(sensor_readings)

defective = y_pred == -1
reject_items(defective_indices)
```

---

## Пов'язані теми

- [[01_Isolation_Forest]] — швидша альтернатива
- [[02_One_Class_SVM]] — kernel-based approach
- [[04_Anomaly_Detection_Methods]] — порівняння всіх
- [[DBSCAN]] — clustering з outlier detection

## Ресурси

- [Original Paper (Breunig et al., 2000)](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf)
- [Scikit-learn: LOF](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [LOF Tutorial](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html)

---

## Ключові висновки

> LOF (Local Outlier Factor) — density-based алгоритм для виявлення локальних аномалій, що порівнює локальну щільність точки з щільністю її k-найближчих сусідів. Точка з LOF >> 1 вважається outlier.

**Основна ідея:**
- Аномалія = низька локальна щільність відносно сусідів
- Порівняння Local Reachability Density
- LOF score = ratio щільностей

**Алгоритм:**
1. Для кожної точки знайти k-nearest neighbors
2. Обчислити reachability distance до сусідів
3. Обчислити Local Reachability Density (LRD)
4. Обчислити LOF = avg(LRD_neighbors) / LRD_point

**LOF Score інтерпретація:**
- LOF ≈ 1 → Normal (схожа щільність)
- LOF > 1.5 → Potential outlier
- LOF > 2.0 → Strong outlier

**Переваги:**
- ✅ Знаходить локальні outliers
- ✅ Працює з variable density
- ✅ Interpretable score
- ✅ No assumptions про розподіли

**Недоліки:**
- ❌ Повільний (O(n²))
- ❌ Не масштабується (>50K)
- ❌ Чутливий до k
- ❌ High-dimensional проблеми

**Ключові параметри:**
- **n_neighbors (k):** розмір околу (10-50)
- **contamination:** expected % outliers
- **novelty:** False=fit_predict, True=fit+predict

**Коли використовувати:**
- Локальні outliers + variable density = LOF ✓
- Великі дані + швидкість → Isolation Forest ✓
- Global outliers → Isolation Forest ✓
- High-dimensional → PCA + LOF ✓

**Практичні поради:**
- ЗАВЖДИ scale features
- Експериментуй з k (10-50)
- novelty=True для production
- Combine з Isolation Forest
- PCA для high-dimensional
- Incremental refit для streaming

---

#ml #unsupervised-learning #anomaly-detection #outlier-detection #lof #local-outlier-factor #density-based #nearest-neighbors
