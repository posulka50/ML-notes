# Isolation Forest

## Що це?

**Isolation Forest** — це **tree-based** алгоритм для виявлення аномалій (outlier detection), що базується на простій ідеї: **аномалії легше ізолювати** ніж нормальні точки. Замість моделювати нормальні дані, він явно ізолює аномалії.

**Головна ідея:** аномалії — це "few and different". Їх можна швидко відокремити випадковими розбиттями, тоді як нормальні точки потребують багато розбиттів щоб бути ізольованими.

## Навіщо потрібен?

- 🔍 **Anomaly Detection** — виявлення незвичайних паттернів
- 🛡️ **Fraud Detection** — шахрайські транзакції
- 🏭 **Industrial Monitoring** — несправності обладнання
- 🌐 **Network Security** — кібератаки, intrusions
- 💳 **Credit Card Fraud** — підозрілі покупки
- 📊 **Data Quality** — виявлення помилок в даних
- 🏥 **Medical Diagnosis** — рідкісні захворювання

## Коли використовувати?

**Потрібно:**

- **Unsupervised detection** — немає labeled anomalies
- **Великі дані** — швидкий та ефективний
- **High-dimensional data** — працює добре
- **Contamination відомий** — знаємо приблизно % аномалій
- **Real-time** — швидкі predictions
- **Числові features** — неперервні дані

**Не потрібно:**

- **Labeled anomalies** є → Supervised methods
- **Categorical data** → інші методи (LOF, HDBSCAN)
- **Дуже малі дані** (< 100) → statistical methods
- **Складні patterns** → Deep Learning (Autoencoders)

---

## Інтуїція

### Чому аномалії легше ізолювати?

**Приклад 1D:**

```
Нормальні точки (густо):
|●●●●●●●●●●●●●●|           |○|
0              10          20

Anomaly (○) далеко від кластера

Скільки random splits потрібно щоб ізолювати?
- Anomaly: 1 split (|15)
- Normal point: багато splits (треба розділити dense cluster)
```

**Приклад 2D:**

```
        y
    10  |              ○ anomaly
        |
     5  |  ●●●●●
        |  ●●●●●
        |  ●●●●●  normal cluster
     0  |_____________ x
        0    5    10

Ізолювати аномалію:
Split 1: x > 8  → anomaly відокремлена! (1 split)

Ізолювати normal point:
Split 1: x > 3
Split 2: y > 2
Split 3: x < 6
Split 4: y < 7
... (багато splits)
```

**Висновок:** Аномалії мають **shorter average path length** в ізоляційному дереві.

---

## Математика

### Isolation Tree (iTree)

**Побудова:**

1. Випадково вибрати feature
2. Випадково вибрати split value між min та max
3. Розділити дані
4. Повторювати рекурсивно до:
   - Точка ізольована (одна в node), або
   - Досягнуто max depth

**Path length** $h(x)$ — кількість edges від root до leaf для точки $x$.

### Anomaly Score

**Для точки $x$:**

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

де:
- $E(h(x))$ — average path length по всіх деревах
- $c(n)$ — average path length для BST з $n$ точок (нормалізація)

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

де $H(i)$ — harmonic number ≈ $\ln(i) + 0.5772$

**Інтерпретація:**

```
s(x) ≈ 1   → Anomaly (короткий path)
s(x) ≈ 0.5 → Normal (середній path)
s(x) < 0.5 → Точно normal (довгий path)
```

### Чому це працює?

**Average path length для:**

- **Anomaly:** $E(h(x)) \ll c(n)$ → $s(x) \rightarrow 1$
- **Normal:** $E(h(x)) \approx c(n)$ → $s(x) \rightarrow 0.5$

---

### Інтуїція

Алгоритм багаторазово випадково "розрізає" простір.  
Точки, які швидко відокремлюються від інших, вважаються підозрілими.

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# 1. Генерувати дані з аномаліями
np.random.seed(42)

# Normal points (dense cluster)
X_normal, _ = make_blobs(
    n_samples=300,
    centers=[[0, 0]],
    cluster_std=0.5,
    random_state=42
)

# Anomalies (scattered outliers)
X_anomalies = np.random.uniform(low=-4, high=4, size=(20, 2))

# Combine
X = np.vstack([X_normal, X_anomalies])
y_true = np.array([0]*300 + [1]*20)  # 0=normal, 1=anomaly

print(f"Total points: {len(X)}")
print(f"Normal: {(y_true == 0).sum()}")
print(f"Anomalies: {(y_true == 1).sum()}")
print(f"Contamination: {(y_true == 1).sum() / len(X):.2%}")

# 2. Fit Isolation Forest
clf = IsolationForest(
    n_estimators=100,           # Кількість дерев
    max_samples=256,            # Розмір sample для кожного дерева
    contamination=0.1,          # Очікуваний % аномалій
    random_state=42
)

clf.fit(X)

# 3. Predict
y_pred = clf.predict(X)  # 1=normal, -1=anomaly
scores = clf.score_samples(X)  # Anomaly scores (чим менше, тим більша аномалія)

print(f"\n=== Predictions ===")
print(f"Predicted anomalies: {(y_pred == -1).sum()}")
print(f"Predicted normal: {(y_pred == 1).sum()}")

# 4. Metrics
from sklearn.metrics import classification_report, confusion_matrix

# Convert: -1 → 1 (anomaly), 1 → 0 (normal)
y_pred_binary = (y_pred == -1).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_binary, 
                           target_names=['Normal', 'Anomaly']))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred_binary)
print(cm)
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# 5. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: True labels
axes[0].scatter(X[y_true == 0, 0], X[y_true == 0, 1],
               c='blue', s=20, alpha=0.6, label='Normal')
axes[0].scatter(X[y_true == 1, 0], X[y_true == 1, 1],
               c='red', s=50, alpha=0.8, marker='x', label='Anomaly')
axes[0].set_title('True Labels', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Predictions (color by anomaly score)
scatter = axes[1].scatter(X[:, 0], X[:, 1],
                         c=scores, cmap='RdYlBu_r',
                         s=30, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.colorbar(scatter, ax=axes[1], label='Anomaly Score\n(lower = more anomalous)')

# Mark predicted anomalies
anomaly_mask = y_pred == -1
axes[1].scatter(X[anomaly_mask, 0], X[anomaly_mask, 1],
               facecolors='none', edgecolors='red',
               s=100, linewidths=2, label='Predicted Anomalies')

axes[1].set_title('Isolation Forest Predictions', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Decision boundary visualization

```python
# Візуалізувати decision boundary

# Create mesh
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
    np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200)
)

# Predict на mesh
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))

# Contour plot
plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
plt.colorbar(label='Decision Function')

# Threshold contour
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

# Data points
plt.scatter(X[y_true == 0, 0], X[y_true == 0, 1],
           c='blue', s=20, alpha=0.6, edgecolors='black', 
           linewidths=0.5, label='Normal')
plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1],
           c='red', s=50, alpha=0.8, marker='x', 
           linewidths=2, label='Anomaly')

plt.title('Isolation Forest Decision Boundary', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Real example: Credit Card Fraud

```python
# Симулюємо credit card transactions

np.random.seed(42)

# Normal transactions
n_normal = 9000
amount_normal = np.random.lognormal(mean=3, sigma=1, size=n_normal)
time_normal = np.random.uniform(0, 24, size=n_normal)

X_normal = np.column_stack([amount_normal, time_normal])

# Fraudulent transactions (different patterns)
n_fraud = 100

# Type 1: Unusually high amounts
amount_fraud1 = np.random.uniform(1000, 5000, size=n_fraud//2)
time_fraud1 = np.random.uniform(0, 24, size=n_fraud//2)

# Type 2: Night transactions with medium amounts
amount_fraud2 = np.random.uniform(200, 500, size=n_fraud//2)
time_fraud2 = np.random.uniform(2, 5, size=n_fraud//2)  # 2-5 AM

X_fraud = np.vstack([
    np.column_stack([amount_fraud1, time_fraud1]),
    np.column_stack([amount_fraud2, time_fraud2])
])

# Combine
X_transactions = np.vstack([X_normal, X_fraud])
y_true = np.array([0]*n_normal + [1]*n_fraud)

print(f"Total transactions: {len(X_transactions)}")
print(f"Fraud rate: {(y_true == 1).sum() / len(y_true):.2%}")

# Fit Isolation Forest
clf = IsolationForest(
    n_estimators=100,
    contamination=0.01,  # Expect 1% fraud
    random_state=42
)

clf.fit(X_transactions)

# Predictions
y_pred = clf.predict(X_transactions)
y_pred_binary = (y_pred == -1).astype(int)

# Metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

print("\n=== Fraud Detection Results ===")
print(f"Precision: {precision_score(y_true, y_pred_binary):.3f}")
print(f"Recall: {recall_score(y_true, y_pred_binary):.3f}")
print(f"F1-Score: {f1_score(y_true, y_pred_binary):.3f}")

# ROC-AUC (використовуємо scores замість binary predictions)
scores = -clf.score_samples(X_transactions)  # Invert (higher = more anomalous)
print(f"ROC-AUC: {roc_auc_score(y_true, scores):.3f}")

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# True labels
axes[0].scatter(X_transactions[y_true == 0, 0], 
               X_transactions[y_true == 0, 1],
               c='green', s=5, alpha=0.3, label='Legitimate')
axes[0].scatter(X_transactions[y_true == 1, 0],
               X_transactions[y_true == 1, 1],
               c='red', s=20, alpha=0.8, marker='x', label='Fraud')
axes[0].set_xlabel('Transaction Amount ($)')
axes[0].set_ylabel('Time of Day (hour)')
axes[0].set_title('True Labels', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predictions
axes[1].scatter(X_transactions[y_pred == 1, 0],
               X_transactions[y_pred == 1, 1],
               c='green', s=5, alpha=0.3, label='Predicted Legitimate')
axes[1].scatter(X_transactions[y_pred == -1, 0],
               X_transactions[y_pred == -1, 1],
               c='red', s=20, alpha=0.8, marker='x', label='Predicted Fraud')
axes[1].set_xlabel('Transaction Amount ($)')
axes[1].set_ylabel('Time of Day (hour)')
axes[1].set_title('Isolation Forest Predictions', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Параметри

### Основні

```python
IsolationForest(
    n_estimators=100,       # Кількість дерев
    max_samples='auto',     # Розмір sample (default: min(256, n_samples))
    contamination=0.1,      # Очікуваний % аномалій
    max_features=1.0,       # Features для split
    bootstrap=False,        # Sample з/без replacement
    random_state=None       # Відтворюваність
)
```

| Параметр | Опис | Типові значення | Рекомендації |
|----------|------|-----------------|--------------|
| **n_estimators** | Кількість дерев | 50-200 | 100 (default) добре |
| **max_samples** | Розмір sample | 256 (default), 'auto' | 256 достатньо для більшості |
| **contamination** | % аномалій | 0.01-0.1 | Базується на domain knowledge |
| **max_features** | Features per split | 1.0 (all) | 1.0 для повної randomness |

### contamination

**Критичний параметр!**

```python
# Якщо знаємо приблизно скільки аномалій
contamination = n_anomalies / n_total

# Приклади:
# Fraud detection: 0.01 (1%)
# Network intrusion: 0.05 (5%)
# Manufacturing defects: 0.001 (0.1%)

# Якщо НЕ знаємо → початок з 0.1 та adjust
```

**Вплив:**
```python
contamination = 0.01  # Строгий (тільки top 1%)
contamination = 0.1   # М'який (top 10%)
contamination = 0.5   # Дуже м'який (половина!)
```

### n_estimators

**Trade-off: accuracy vs speed**

```python
# Мало дерев → швидше, але менш стабільно
n_estimators = 50

# Багато дерев → повільніше, але стабільніше
n_estimators = 200

# Default 100 — добрий баланс
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Швидкість** | O(n log n) — дуже швидкий |
| **Масштабованість** | Працює на великих даних |
| **High-dimensional** | Добре на багатьох features |
| **Unsupervised** | Не потребує labels |
| **Інтерпретованість** | Path length має сенс |
| **Memory efficient** | Subsampling зменшує пам'ять |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Contamination** потрібен | Треба знати приблизно % |
| **Categorical features** | Погано працює |
| **Локальні аномалії** | Може пропустити (LOF краще) |
| **Imbalanced splits** | Може давати біаси |
| **Не для clusters** | Не детектує cluster-based outliers |

---

## Порівняння з іншими методами

| Метод | Швидкість | High-dim | Локальні аномалії | Supervised |
|-------|-----------|----------|-------------------|------------|
| **Isolation Forest** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ |
| **LOF** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| **One-Class SVM** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⚠️ |
| **Autoencoder** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |

**Коли що:**
- **Великі дані + швидкість** → Isolation Forest ✓
- **Локальні outliers** → LOF ✓
- **Складні patterns** → Autoencoder ✓
- **Labeled anomalies** → Supervised methods ✓

---

## Практичні поради 💡

### 1. Налаштуй contamination під дані

```python
# Початок: estimate з EDA
anomaly_rate_estimate = 0.05  # 5%

clf = IsolationForest(contamination=anomaly_rate_estimate)

# Adjust based на results
# Якщо занадто багато false positives → зменш contamination
# Якщо пропускає очевидні outliers → збільш
```

### 2. Feature scaling НЕ обов'язковий

```python
# ✅ Isolation Forest працює без scaling
# (random splits не залежать від scale)

clf = IsolationForest()
clf.fit(X)  # Raw features OK

# Але якщо є features з різними ranges
# scaling може трохи допомогти
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

### 3. Використовуй score_samples для ranking

```python
# Замість binary predictions (anomaly/normal)
# Використовуй scores для ranking

scores = clf.score_samples(X)
sorted_indices = np.argsort(scores)  # Ascending (most anomalous first)

# Top-N найбільш аномальних
top_n_anomalies = sorted_indices[:10]

print("Most anomalous samples:")
for idx in top_n_anomalies:
    print(f"  Index {idx}: score = {scores[idx]:.4f}")
```

### 4. Cross-validation для параметрів

```python
# Якщо є невелика кількість labeled anomalies
# Можна optimize contamination

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Wrapper для compatibility
class IFWrapper:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.clf = IsolationForest(contamination=contamination)
    
    def fit(self, X, y=None):
        self.clf.fit(X)
        return self
    
    def predict(self, X):
        return (self.clf.predict(X) == -1).astype(int)

# Grid search (якщо є labels)
param_grid = {'contamination': [0.01, 0.05, 0.1, 0.15]}
grid = GridSearchCV(IFWrapper(), param_grid, scoring='f1', cv=3)
grid.fit(X, y_true)

print(f"Best contamination: {grid.best_params_['contamination']}")
```

### 5. Ensemble з іншими методами

```python
# Комбінувати Isolation Forest з LOF

from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest
if_scores = IsolationForest().fit(X).score_samples(X)

# LOF
lof_scores = LocalOutlierFactor(novelty=True).fit(X).score_samples(X)

# Combine (average)
combined_scores = (if_scores + lof_scores) / 2

# Або voting
if_pred = (if_scores < np.percentile(if_scores, 10))
lof_pred = (lof_scores < np.percentile(lof_scores, 10))
ensemble_pred = if_pred | lof_pred  # Union
```

### 6. Feature importance

```python
# Хоча IF не має прямого feature_importances_
# Можна approximate через permutation

from sklearn.inspection import permutation_importance

# Wrapper для scoring
def anomaly_score(X):
    return clf.score_samples(X)

# Permutation importance потребує sklearn estimator
# Альтернативно: ручний підрахунок

def calculate_feature_importance(clf, X, n_permutations=10):
    """Manual feature importance"""
    base_scores = clf.score_samples(X)
    importances = []
    
    for feature_idx in range(X.shape[1]):
        perm_scores = []
        
        for _ in range(n_permutations):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, feature_idx])
            
            scores_perm = clf.score_samples(X_perm)
            diff = np.mean(np.abs(base_scores - scores_perm))
            perm_scores.append(diff)
        
        importances.append(np.mean(perm_scores))
    
    return np.array(importances)

importances = calculate_feature_importance(clf, X)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances)
plt.yticks(range(len(importances)), [f'Feature {i}' for i in range(len(importances))])
plt.xlabel('Importance (permutation effect)')
plt.title('Feature Importance for Anomaly Detection')
plt.tight_layout()
plt.show()
```

### 7. Incremental detection (streaming)

```python
# Для streaming data: періодично refit

class IncrementalIF:
    def __init__(self, window_size=1000, contamination=0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.clf = IsolationForest(contamination=contamination)
        self.buffer = []
    
    def add_sample(self, x):
        """Додати новий зразок"""
        self.buffer.append(x)
        
        # Якщо buffer повний → refit
        if len(self.buffer) >= self.window_size:
            X_train = np.array(self.buffer[-self.window_size:])
            self.clf.fit(X_train)
            self.buffer = self.buffer[-self.window_size//2:]  # Keep half
    
    def predict(self, x):
        """Predict для нового зразка"""
        return self.clf.predict([x])[0]

# Usage
detector = IncrementalIF()

for new_sample in data_stream:
    detector.add_sample(new_sample)
    is_anomaly = detector.predict(new_sample)
    
    if is_anomaly == -1:
        alert("Anomaly detected!")
```

### 8. Threshold tuning

```python
# Замість contamination, можна задати custom threshold

scores = clf.score_samples(X)

# Метод 1: Percentile
threshold = np.percentile(scores, 5)  # Bottom 5%

# Метод 2: Standard deviations
mean_score = np.mean(scores)
std_score = np.std(scores)
threshold = mean_score - 2 * std_score  # 2 std below

# Метод 3: Visual inspection (histogram)
plt.hist(scores, bins=50)
plt.axvline(threshold, color='red', linestyle='--')
plt.show()

# Predict з custom threshold
y_pred_custom = (scores < threshold).astype(int)
```

### 9. Multivariate vs Univariate

```python
# Univariate: окремо для кожної feature
from scipy.stats import zscore

z_scores = np.abs(zscore(X, axis=0))
univariate_outliers = (z_scores > 3).any(axis=1)

# Multivariate: Isolation Forest
multivariate_outliers = (clf.predict(X) == -1)

# Порівняння
print(f"Univariate outliers: {univariate_outliers.sum()}")
print(f"Multivariate outliers: {multivariate_outliers.sum()}")
print(f"Both: {(univariate_outliers & multivariate_outliers).sum()}")
```

### 10. Visualize anomaly regions

```python
# Для 2D даних — показати decision regions

if X.shape[1] == 2:
    # Mesh
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
        np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100)
    )
    
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c='black', s=20)
    plt.colorbar(label='Anomaly Score')
    plt.title('Anomaly Regions')
    plt.show()
```

---

## Поширені помилки ❌

### 1. Неправильний contamination

```python
# ❌ Contamination = 0.5 (50%!)
# Половина даних — аномалії? Nonsense!

# ✅ Розумний contamination на основі domain
contamination = 0.01  # 1% для fraud
contamination = 0.05  # 5% для sensor errors
```

### 2. Не перевіряти результати

```python
# ❌ Сліпо довіряти predictions

# ✅ Завжди inspect
scores = clf.score_samples(X)

plt.hist(scores, bins=50)
plt.title('Distribution of Anomaly Scores')
plt.show()

# Check найбільш аномальні manually
```

### 3. Використовувати на categorical data

```python
# ❌ Isolation Forest погано на categorical
X_cat = ['Red', 'Blue', 'Green', ...]

# ✅ One-hot encode спочатку, АБО використай LOF
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_cat)
```

### 4. Забути про data preprocessing

```python
# ❌ Missing values, duplicates
clf.fit(X_with_nans)  # Error!

# ✅ Clean data спочатку
X_clean = handle_missing_values(X)
X_clean = remove_duplicates(X_clean)
clf.fit(X_clean)
```

### 5. Не валідувати (якщо є labels)

```python
# Якщо є labels — USE THEM для validation!

from sklearn.metrics import classification_report

y_pred = (clf.predict(X) == -1).astype(int)

print(classification_report(y_true, y_pred))

# Check precision/recall
```

---

## Реальні застосування

### 1. Credit Card Fraud Detection

```python
# Features: amount, time, merchant, location
# Anomaly: fraudulent transactions

clf = IsolationForest(contamination=0.001)  # 0.1% fraud rate
clf.fit(transactions[features])

# Real-time detection
new_transaction = [amount, time, merchant_id, ...]
is_fraud = clf.predict([new_transaction])[0] == -1

if is_fraud:
    block_transaction()
    send_alert()
```

### 2. Network Intrusion Detection

```python
# Features: packet size, duration, protocol, ports
# Anomaly: attacks, malware

clf = IsolationForest(contamination=0.05)
clf.fit(network_traffic[features])

# Monitor
for packet in stream:
    score = clf.score_samples([packet])[0]
    
    if score < threshold:
        log_alert(packet)
```

### 3. Manufacturing Quality Control

```python
# Features: temperature, pressure, speed, vibration
# Anomaly: defective products

clf = IsolationForest(contamination=0.01)
clf.fit(sensor_readings[features])

# Detect defects
predictions = clf.predict(new_batch)
defect_indices = np.where(predictions == -1)[0]

reject_items(defect_indices)
```

### 4. Health Monitoring (IoT sensors)

```python
# Features: heart rate, blood pressure, activity
# Anomaly: health issues

clf = IsolationForest(contamination=0.05)
clf.fit(patient_data[features])

# Real-time monitoring
current_readings = get_sensor_data()
is_abnormal = clf.predict([current_readings])[0] == -1

if is_abnormal:
    notify_doctor()
```

---

## Пов'язані теми

- [[02_One_Class_SVM]] — інший unsupervised метод
- [[03_Local_Outlier_Factor]] — density-based detection
- [[04_Anomaly_Detection_Methods]] — порівняння методів
- [[Autoencoders]] — deep learning approach

## Ресурси

- [Original Paper (Liu et al., 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Scikit-learn: Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Anomaly Detection with Isolation Forest](https://towardsdatascience.com/anomaly-detection-with-isolation-forest-e41f1f55cc6)

---

## Ключові висновки

> Isolation Forest — tree-based алгоритм для anomaly detection, що базується на принципі: аномалії легше ізолювати випадковими splits ніж нормальні точки, тому мають shorter average path length в isolation trees.

**Основна ідея:**
- Аномалії = "few and different"
- Легко ізолювати random splits
- Short path → high anomaly score

**Алгоритм:**
1. Побудувати ensemble isolation trees (random features + splits)
2. Для кожної точки обчислити average path length
3. Нормалізувати: $s(x) = 2^{-h(x)/c(n)}$
4. Threshold для classification

**Anomaly score:**
- s ≈ 1 → Anomaly
- s ≈ 0.5 → Normal
- s < 0.5 → Definitely normal

**Переваги:**
- ⚡ Дуже швидкий (O(n log n))
- 📈 Масштабується на великі дані
- 🔢 Добре на high-dimensional
- 💾 Memory efficient (subsampling)

**Недоліки:**
- ❌ Потребує contamination parameter
- ❌ Погано на categorical features
- ❌ Може пропускати локальні outliers

**Ключові параметри:**
- **contamination:** % очікуваних аномалій (CRITICAL!)
- **n_estimators:** кількість дерев (100 default)
- **max_samples:** розмір sample (256 default)

**Коли використовувати:**
- Великі дані + швидкість = Isolation Forest ✓
- Локальні outliers → LOF ✓
- Складні patterns → Autoencoder ✓
- Labeled data → Supervised ✓

**Практичні поради:**
- Налаштуй contamination під domain
- Використовуй scores для ranking
- Валідуй якщо є labels
- Ensemble з LOF для кращих results
- Incremental refit для streaming

---

#ml #unsupervised-learning #anomaly-detection #outlier-detection #isolation-forest #fraud-detection #security #tree-based
