# One-Class SVM

## Що це?

**One-Class SVM (Support Vector Machine)** — це алгоритм для **novelty/anomaly detection**, що навчається на тільки **нормальних даних** і будує границю навколо них. Все що за межами цієї границі вважається аномалією.

**Головна ідея:** знайти гіперплощину, що відокремлює нормальні дані від початку координат у високорозмірному feature space, максимізуючи margin (відстань від границі до точок).

## Навіщо потрібен?

- 🛡️ **Novelty Detection** — виявлення нових, незвичайних зразків
- 🔍 **Outlier Detection** — знаходження викидів
- 🏭 **Quality Control** — дефекти в виробництві
- 🌐 **Network Security** — intrusion detection
- 🖼️ **Image Anomaly** — дефекти на зображеннях
- 📊 **One-class classification** — тільки positive examples
- 🧪 **Rare event detection** — рідкісні події

## Коли використовувати?

**Потрібно:**
- **Тільки нормальні дані** для навчання
- **Novelty detection** — нові типи аномалій
- **Smooth decision boundary** — гладка границя
- **Kernel trick** — нелінійні patterns
- **Theoretical foundation** — математично обґрунтовано
- **Середні дані** (100-10,000 зразків)

**Не потрібно:**
- **Дуже великі дані** (> 100K) → Isolation Forest швидше
- **High-dimensional + sparse** → Isolation Forest
- **Real-time streaming** → простіші методи
- **Interpretability важлива** → Isolation Forest
- **Дуже малі дані** (< 50) → statistical methods

---

## Математика

### Основна ідея

**Мета:** знайти smallest hypersphere (або hyperplane) що містить більшість нормальних точок.

**В feature space:**

```
Original space:           Feature space (після kernel):
    ●●●●                      ●●●●●
    ●  ●                      ●   ●
    ●●●●                      ●●●●●
                              ↓
                       Find boundary around data
```

### Optimization Problem

**Primal формулювання:**

$$\min_{w, \rho, \xi} \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum_{i=1}^{n}\xi_i$$

**Subject to:**
$$w^T\phi(x_i) \geq \rho - \xi_i, \quad \xi_i \geq 0$$

де:
- $w$ — нормаль до hyperplane
- $\rho$ — offset (відстань від origin)
- $\xi_i$ — slack variables (дозволяють помилки)
- $\nu \in (0,1]$ — upper bound на fraction outliers
- $\phi(x)$ — kernel mapping

**Інтуїція:**
- Максимізувати margin (мінімізувати $\|w\|$)
- Максимізувати $\rho$ (відштовхнути від origin)
- Мінімізувати порушення (slack $\xi_i$)

### Decision Function

**Для нової точки $x$:**

$$f(x) = \text{sign}(w^T\phi(x) - \rho) = \text{sign}\left(\sum_{i}\alpha_i K(x_i, x) - \rho\right)$$

де:
- $f(x) = +1$ → Normal
- $f(x) = -1$ → Anomaly
- $K(x_i, x)$ — kernel function

### Kernel Functions

**Linear:**
$$K(x, x') = x^T x'$$

**RBF (Gaussian) — найпопулярніший:**
$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$

**Polynomial:**
$$K(x, x') = (x^T x' + c)^d$$

**Sigmoid:**
$$K(x, x') = \tanh(\gamma x^T x' + c)$$

---

## Візуальна інтуїція

### 2D приклад

```
Linear kernel:
    y
    |  ●●●●●
    |  ●   ●
    |  ●●●●●
    |_________x
    
Decision boundary = straight line

RBF kernel:
    y
    |  ●●●●●
    |  ● ○ ●  ← може захопити тут
    |  ●●●●●
    |_________x
    
Decision boundary = curved (гнучкіша)
```

### Margin visualization

```
        ●●●●●
      ●●     ●●
     ●         ●
    ●    ○○○   ●  ← Boundary
     ●         ●
      ●●     ●●
        ●●●●●
        
    ↑ margin ↑
```

---

## Алгоритм

### Training

```
1. Дано: тільки NORMAL data X = {x₁, x₂, ..., xₙ}

2. Вибрати kernel K(·,·) та параметри (γ для RBF, ν)

3. Розв'язати optimization problem:
   min 1/2||w||² - ρ + (1/νn)Σξᵢ
   subject to: wᵀφ(xᵢ) ≥ ρ - ξᵢ, ξᵢ ≥ 0

4. Знайти support vectors (αᵢ > 0)

5. Обчислити ρ

6. Decision function: f(x) = sign(Σαᵢ K(xᵢ, x) - ρ)
```

### Prediction

```
Для нової точки x:

1. Обчислити kernel values з support vectors:
   s(x) = Σ αᵢ K(xᵢ, x) - ρ

2. Classify:
   IF s(x) ≥ 0:
       x is NORMAL
   ELSE:
       x is ANOMALY
```

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

# 1. Генерувати ТІЛЬКИ нормальні дані для training
np.random.seed(42)

X_train, _ = make_blobs(
    n_samples=200,
    centers=[[0, 0]],
    cluster_std=0.5,
    random_state=42
)

print(f"Training data: {X_train.shape}")
print("Training on NORMAL data only!")

# 2. Створити test set з нормальними + аномаліями
X_test_normal, _ = make_blobs(
    n_samples=50,
    centers=[[0, 0]],
    cluster_std=0.5,
    random_state=100
)

X_test_anomalies = np.random.uniform(low=-4, high=4, size=(10, 2))

X_test = np.vstack([X_test_normal, X_test_anomalies])
y_test_true = np.array([1]*50 + [-1]*10)  # 1=normal, -1=anomaly

# 3. Fit One-Class SVM
clf = OneClassSVM(
    kernel='rbf',       # RBF kernel (default)
    gamma='auto',       # 1 / (n_features * X.var())
    nu=0.1              # Upper bound на fraction outliers
)

clf.fit(X_train)

# 4. Predict
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print(f"\n=== Training Set ===")
print(f"Predicted outliers: {(y_train_pred == -1).sum()} / {len(y_train_pred)}")
print(f"Expected: ~{int(0.1 * len(y_train_pred))} (nu=0.1)")

print(f"\n=== Test Set ===")
print(f"Predicted anomalies: {(y_test_pred == -1).sum()}")
print(f"True anomalies: {(y_test_true == -1).sum()}")

# 5. Metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Classification Report ===")
print(classification_report(y_test_true, y_test_pred,
                           target_names=['Anomaly', 'Normal']))

# 6. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training data + decision boundary
xx, yy = np.meshgrid(
    np.linspace(-4, 4, 200),
    np.linspace(-4, 4, 200)
)

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot 1: Training
axes[0].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
axes[0].contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
axes[0].scatter(X_train[:, 0], X_train[:, 1],
               c='blue', s=20, alpha=0.6, edgecolors='black',
               linewidths=0.5, label='Training (Normal)')

# Support vectors
support_vectors = clf.support_vectors_
axes[0].scatter(support_vectors[:, 0], support_vectors[:, 1],
               s=100, facecolors='none', edgecolors='red',
               linewidths=2, label='Support Vectors')

axes[0].set_title('Training: One-Class SVM', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Test
axes[1].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
axes[1].contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

# Normal test points
axes[1].scatter(X_test[y_test_true == 1, 0], X_test[y_test_true == 1, 1],
               c='blue', s=30, alpha=0.7, label='Normal')

# Anomalies
axes[1].scatter(X_test[y_test_true == -1, 0], X_test[y_test_true == -1, 1],
               c='red', s=50, marker='x', linewidths=2, label='Anomaly')

axes[1].set_title('Test: Predictions', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Порівняння різних kernels

```python
# Різні kernels для порівняння
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, kernel in enumerate(kernels):
    print(f"\nTraining {kernel} kernel...")
    
    clf = OneClassSVM(kernel=kernel, nu=0.1, gamma='auto')
    clf.fit(X_train)
    
    y_pred = clf.predict(X_test)
    
    # Decision function
    xx, yy = np.meshgrid(
        np.linspace(-4, 4, 200),
        np.linspace(-4, 4, 200)
    )
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    axes[idx].contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
    axes[idx].contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    
    axes[idx].scatter(X_test[y_test_true == 1, 0], X_test[y_test_true == 1, 1],
                     c='blue', s=20, alpha=0.6, label='Normal')
    axes[idx].scatter(X_test[y_test_true == -1, 0], X_test[y_test_true == -1, 1],
                     c='red', s=40, marker='x', linewidths=2, label='Anomaly')
    
    # Support vectors
    sv = clf.support_vectors_
    axes[idx].scatter(sv[:, 0], sv[:, 1],
                     s=80, facecolors='none', edgecolors='red',
                     linewidths=2, label='Support Vectors')
    
    # Metrics
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test_true, y_pred)
    
    axes[idx].set_title(f'{kernel.upper()} kernel (Acc: {acc:.2f})',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Параметр tuning (gamma та nu)

```python
# Grid search для найкращих параметрів

from sklearn.model_selection import ParameterGrid

# Параметри для пошуку
param_grid = {
    'gamma': [0.001, 0.01, 0.1, 1.0],
    'nu': [0.01, 0.05, 0.1, 0.2]
}

best_score = -np.inf
best_params = None
results = []

for params in ParameterGrid(param_grid):
    clf = OneClassSVM(kernel='rbf', **params)
    clf.fit(X_train)
    
    y_pred = clf.predict(X_test)
    
    # Score (F1 для anomaly class)
    from sklearn.metrics import f1_score
    score = f1_score(y_test_true, y_pred, pos_label=-1)
    
    results.append({
        'gamma': params['gamma'],
        'nu': params['nu'],
        'f1_score': score
    })
    
    if score > best_score:
        best_score = score
        best_params = params

print("\n=== Parameter Tuning Results ===")
print(f"Best params: {best_params}")
print(f"Best F1-score: {best_score:.4f}")

# Візуалізація results
import pandas as pd

df_results = pd.DataFrame(results)
pivot = df_results.pivot(index='nu', columns='gamma', values='f1_score')

plt.figure(figsize=(10, 6))
import seaborn as sns
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
plt.title('F1-Score Heatmap (gamma vs nu)', fontsize=14, fontweight='bold')
plt.ylabel('nu')
plt.xlabel('gamma')
plt.tight_layout()
plt.show()
```

### Real example: Network Intrusion Detection

```python
# Симулюємо network traffic data

np.random.seed(42)

# Normal traffic (training)
n_normal_train = 1000

packet_size_normal = np.random.normal(500, 100, n_normal_train)
duration_normal = np.random.exponential(2, n_normal_train)
packets_per_sec_normal = np.random.poisson(10, n_normal_train)

X_train_network = np.column_stack([
    packet_size_normal,
    duration_normal,
    packets_per_sec_normal
])

print(f"Training set: {X_train_network.shape}")

# Test data (normal + attacks)
n_normal_test = 200
n_attacks = 50

# Normal
packet_size_test = np.random.normal(500, 100, n_normal_test)
duration_test = np.random.exponential(2, n_normal_test)
packets_test = np.random.poisson(10, n_normal_test)

X_test_normal = np.column_stack([
    packet_size_test,
    duration_test,
    packets_test
])

# Attacks (different patterns)
# Type 1: DDoS (high packets per sec)
packet_size_attack1 = np.random.normal(100, 20, n_attacks//2)
duration_attack1 = np.random.uniform(0.1, 0.5, n_attacks//2)
packets_attack1 = np.random.poisson(100, n_attacks//2)  # Very high!

# Type 2: Large file transfer (suspicious)
packet_size_attack2 = np.random.normal(5000, 500, n_attacks//2)
duration_attack2 = np.random.uniform(10, 20, n_attacks//2)
packets_attack2 = np.random.poisson(5, n_attacks//2)

X_test_attacks = np.vstack([
    np.column_stack([packet_size_attack1, duration_attack1, packets_attack1]),
    np.column_stack([packet_size_attack2, duration_attack2, packets_attack2])
])

# Combine test
X_test_network = np.vstack([X_test_normal, X_test_attacks])
y_test_network = np.array([1]*n_normal_test + [-1]*n_attacks)

print(f"Test set: {X_test_network.shape}")
print(f"  Normal: {n_normal_test}")
print(f"  Attacks: {n_attacks}")

# Feature scaling (ВАЖЛИВО!)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_network)
X_test_scaled = scaler.transform(X_test_network)

# Train One-Class SVM
clf = OneClassSVM(
    kernel='rbf',
    gamma=0.1,
    nu=0.05  # Expect ~5% outliers in training
)

clf.fit(X_train_scaled)

# Predict
y_pred = clf.predict(X_test_scaled)

print("\n=== Intrusion Detection Results ===")
print(classification_report(y_test_network, y_pred,
                           target_names=['Attack', 'Normal']))

# ROC curve
from sklearn.metrics import roc_curve, auc

decision_scores = clf.decision_function(X_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test_network, decision_scores, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve: Network Intrusion Detection', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Feature importance (через perturbation)
def feature_importance_ocsvm(clf, X, feature_names):
    """Approximate feature importance"""
    base_scores = clf.decision_function(X)
    importances = []
    
    for feature_idx in range(X.shape[1]):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, feature_idx])
        
        scores_perm = clf.decision_function(X_perm)
        diff = np.mean(np.abs(base_scores - scores_perm))
        importances.append(diff)
    
    return np.array(importances)

feature_names = ['Packet Size', 'Duration', 'Packets/sec']
importances = feature_importance_ocsvm(clf, X_test_scaled, feature_names)

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Importance (perturbation effect)', fontsize=12)
plt.title('Feature Importance for Intrusion Detection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Параметри

### Основні

```python
OneClassSVM(
    kernel='rbf',           # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',          # Kernel coefficient
    nu=0.5,                 # Upper bound на fraction outliers
    degree=3,               # Degree for poly kernel
    coef0=0.0,              # Independent term for poly/sigmoid
    tol=1e-3,               # Tolerance for stopping
    max_iter=-1             # Max iterations (-1 = no limit)
)
```

| Параметр | Опис | Типові значення | Рекомендації |
|----------|------|-----------------|--------------|
| **kernel** | Тип kernel | 'rbf', 'linear', 'poly' | 'rbf' (default) найкращий |
| **gamma** | RBF kernel width | 'scale', 'auto', 0.001-1.0 | 'scale' = 1/(n_features * X.var()) |
| **nu** | Upper bound outliers | 0.01-0.5 | Lower = строгіше (fewer outliers) |
| **degree** | Poly degree | 2-5 | Тільки для poly kernel |

### nu parameter (критичний!)

**Що це:** Upper bound на:
1. Fraction training errors (outliers в training set)
2. Lower bound на fraction support vectors

```python
nu = 0.01  # Дуже строго (1% outliers max)
nu = 0.1   # Середньо (10%)
nu = 0.5   # М'яко (50%)
```

**Вплив:**

```
nu = 0.01:  Вузька boundary
    ●●●
    ●●●   → Tight fit

nu = 0.5:   Широка boundary
  ●●●●●
 ●     ●  → Loose fit
  ●●●●●
```

### gamma parameter (для RBF)

**Що це:** Контролює "радіус впливу" support vectors.

$$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$

**Вплив:**

```python
gamma = 0.001  # Wide influence (smooth boundary)
gamma = 0.1    # Medium
gamma = 1.0    # Narrow influence (complex boundary)
```

**Візуально:**

```
Low gamma (0.01):
    ●●●●●●●
    ●     ●
    ●●●●●●●
    Smooth boundary

High gamma (1.0):
    ●●●●●
    ● ○ ●  ← Може захопити індивідуальні points
    ●●●●●
    Complex boundary (може overfit!)
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Kernel trick** | Нелінійні boundaries |
| **Theoretical foundation** | Solid math theory |
| **Novelty detection** | Добре для нових типів аномалій |
| **Smooth boundary** | Не такий чутливий до noise |
| **Few parameters** | nu, gamma (просто налаштувати) |
| **Effective** | Працює добре на малих-середніх даних |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Повільний** | O(n²) до O(n³) |
| **Не масштабується** | Погано на >10K зразків |
| **Memory intensive** | Потребує багато пам'яті |
| **Sensitive to scaling** | Потребує normalization |
| **Parameter tuning** | nu та gamma треба підбирати |
| **Binary output** | Тільки anomaly/normal (не score) |

---

## Порівняння з іншими методами

| Метод | Швидкість | Масштабованість | Kernel trick | Interpretability |
|-------|-----------|-----------------|--------------|------------------|
| **One-Class SVM** | ⭐⭐ | ⭐⭐ | ✅ | ⭐⭐ |
| **Isolation Forest** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ |
| **LOF** | ⭐⭐ | ⭐⭐ | ❌ | ⭐⭐⭐ |
| **Autoencoder** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | ⭐ |

**Коли що:**
- **Малі дані + нелінійні patterns** → One-Class SVM ✓
- **Великі дані + швидкість** → Isolation Forest ✓
- **Локальні outliers** → LOF ✓
- **Складні patterns** → Autoencoder ✓

---

## Практичні поради 💡

### 1. ЗАВЖДИ scale features

```python
# ❌ Без scaling
clf = OneClassSVM()
clf.fit(X)  # Погано!

# ✅ Зі scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = OneClassSVM()
clf.fit(X_scaled)

# Для нових даних
X_new_scaled = scaler.transform(X_new)
```

### 2. Почни з RBF kernel та default параметрів

```python
# ✅ Добрий початок
clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)

# Потім tune якщо потрібно
```

### 3. Grid search для nu та gamma

```python
from sklearn.model_selection import GridSearchCV

# Wrapper для cross-validation
param_grid = {
    'gamma': [0.001, 0.01, 0.1, 1.0],
    'nu': [0.01, 0.05, 0.1, 0.2]
}

# Якщо є validation set з labels
best_f1 = 0
best_params = None

for gamma in param_grid['gamma']:
    for nu in param_grid['nu']:
        clf = OneClassSVM(gamma=gamma, nu=nu)
        clf.fit(X_train)
        y_pred = clf.predict(X_val)
        
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val_true, y_pred, pos_label=-1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = {'gamma': gamma, 'nu': nu}

print(f"Best params: {best_params}")
```

### 4. Використовуй decision_function для ranking

```python
# Замість binary predict
scores = clf.decision_function(X_test)

# Negative scores = more anomalous
sorted_indices = np.argsort(scores)

print("Most anomalous samples:")
for idx in sorted_indices[:10]:
    print(f"  Index {idx}: score = {scores[idx]:.4f}")
```

### 5. Ensemble з іншими методами

```python
# Combine з Isolation Forest
from sklearn.ensemble import IsolationForest

# One-Class SVM
ocsvm = OneClassSVM(nu=0.1)
ocsvm.fit(X_train)
ocsvm_scores = ocsvm.decision_function(X_test)

# Isolation Forest
iforest = IsolationForest(contamination=0.1)
iforest.fit(X_train)
if_scores = iforest.score_samples(X_test)

# Normalize scores
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

ocsvm_scores_norm = scaler.fit_transform(ocsvm_scores.reshape(-1, 1)).ravel()
if_scores_norm = scaler.fit_transform(if_scores.reshape(-1, 1)).ravel()

# Combine (average)
combined_scores = (ocsvm_scores_norm + if_scores_norm) / 2

# Threshold
threshold = np.percentile(combined_scores, 10)  # Bottom 10%
y_pred_ensemble = (combined_scores < threshold).astype(int)
```

### 6. Cross-validation для model selection

```python
# Якщо є labeled validation set
from sklearn.model_selection import cross_val_score

# Wrapper для scoring
class OCSVMWrapper:
    def __init__(self, nu=0.1, gamma='scale'):
        self.clf = OneClassSVM(nu=nu, gamma=gamma)
    
    def fit(self, X, y=None):
        self.clf.fit(X)
        return self
    
    def score(self, X, y):
        y_pred = self.clf.predict(X)
        from sklearn.metrics import f1_score
        return f1_score(y, y_pred, pos_label=-1)

# Cross-validation
wrapper = OCSVMWrapper(nu=0.1)
scores = cross_val_score(wrapper, X_train, y_train, cv=3)
print(f"CV F1-scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 7. Incremental learning (streaming)

```python
# One-Class SVM не має incremental learning
# Для streaming: періодично refit

class StreamingOCSVM:
    def __init__(self, window_size=1000, nu=0.1):
        self.window_size = window_size
        self.clf = OneClassSVM(nu=nu)
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

# Usage
detector = StreamingOCSVM()

for sample in data_stream:
    detector.add_sample(sample)
    
    if len(detector.buffer) >= detector.window_size:
        is_anomaly = detector.predict(sample)
        
        if is_anomaly == -1:
            alert("Anomaly detected!")
```

### 8. Visualize decision boundary (2D)

```python
if X.shape[1] == 2:
    # Mesh
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
        np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200)
    )
    
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.scatter(X[:, 0], X[:, 1], c='blue', s=20)
    plt.colorbar(label='Decision Function')
    plt.title('One-Class SVM Decision Boundary')
    plt.show()
```

### 9. Handle imbalanced data

```python
# Якщо training set містить трохи outliers
# Adjust nu відповідно

estimated_outlier_fraction = 0.05  # 5% outliers в train
clf = OneClassSVM(nu=estimated_outlier_fraction)
```

### 10. Combine з dimensionality reduction

```python
# Для high-dimensional data
from sklearn.decomposition import PCA

# PCA спочатку
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_train)

# One-Class SVM на reduced space
clf = OneClassSVM(nu=0.1)
clf.fit(X_pca)

# For new data
X_new_pca = pca.transform(X_new)
y_pred = clf.predict(X_new_pca)
```

---

## Поширені помилки ❌

### 1. Не робити scaling

```python
# ❌ КРИТИЧНА помилка
clf = OneClassSVM()
clf.fit(X_raw)  # Features в різних scales!

# ✅ ЗАВЖДИ scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
clf.fit(X_scaled)
```

### 2. Неправильний nu

```python
# ❌ nu = 0.5 (50% outliers?)
# Занадто м'яко!

# ✅ Розумний nu базуючись на domain
nu = 0.01  # 1% для fraud
nu = 0.05  # 5% для defects
```

### 3. Використовувати на великих даних

```python
# ❌ 100,000 зразків
# Дуже повільно! Години!

# ✅ Sample спочатку або використай Isolation Forest
if len(X) > 10000:
    # Use Isolation Forest instead
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest()
```

### 4. Забути про kernel choice

```python
# ❌ Linear kernel для нелінійних patterns

# ✅ RBF для більшості випадків
clf = OneClassSVM(kernel='rbf')
```

### 5. Не tune gamma

```python
# ❌ Default gamma може бути поганим

# ✅ Grid search
for gamma in [0.001, 0.01, 0.1, 1.0]:
    clf = OneClassSVM(gamma=gamma)
    # Test and compare
```

---

## Реальні застосування

### 1. Manufacturing Quality Control

```python
# Train на GOOD products only
# Detect defects

clf = OneClassSVM(nu=0.01)  # 1% defect rate
clf.fit(sensor_readings_good)

# New product
is_defect = clf.predict([new_product_readings])[0] == -1

if is_defect:
    reject_product()
```

### 2. Medical Anomaly Detection

```python
# Train на healthy patients
# Detect diseases

clf = OneClassSVM(kernel='rbf', nu=0.05)
clf.fit(healthy_patient_data)

# New patient
is_abnormal = clf.predict([patient_readings])[0] == -1

if is_abnormal:
    flag_for_doctor_review()
```

### 3. Video Surveillance

```python
# Train на normal behavior
# Detect suspicious activity

clf = OneClassSVM(nu=0.1)
clf.fit(normal_activity_features)

# Real-time
for frame in video_stream:
    features = extract_features(frame)
    is_suspicious = clf.predict([features])[0] == -1
    
    if is_suspicious:
        alert_security()
```

---

## Пов'язані теми

- [[01_Isolation_Forest]] — швидша альтернатива
- [[03_Local_Outlier_Factor]] — density-based
- [[04_Anomaly_Detection_Methods]] — порівняння всіх
- [[SVM_Classification]] — two-class SVM

## Ресурси

- [Original Paper (Schölkopf et al., 2001)](http://users.cecs.anu.edu.au/~williams/papers/P132.pdf)
- [Scikit-learn: One-Class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [One-Class SVM Tutorial](https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html)

---

## Ключові висновки

> One-Class SVM — це kernel-based алгоритм для novelty/anomaly detection, що навчається тільки на нормальних даних і будує decision boundary, яка відокремлює нормальні точки від origin у feature space, максимізуючи margin.

**Основна ідея:**
- Train тільки на NORMAL data
- Знайти hyperplane/hypersphere навколо даних
- Kernel trick для нелінійних boundaries

**Математика:**
- Optimization: min ||w||² - ρ + penalties
- Decision: f(x) = sign(Σαᵢ K(xᵢ, x) - ρ)
- Kernel: RBF найпопулярніший

**Ключові параметри:**
- **nu:** upper bound на outliers (0.01-0.2)
- **gamma:** RBF width (0.001-1.0)
- **kernel:** 'rbf' (default), 'linear', 'poly'

**Переваги:**
- ✅ Kernel trick (нелінійні boundaries)
- ✅ Solid theory
- ✅ Novelty detection
- ✅ Smooth boundaries

**Недоліки:**
- ❌ Повільний (O(n²-n³))
- ❌ Не масштабується (>10K)
- ❌ Потребує scaling
- ❌ Parameter tuning

**Коли використовувати:**
- Малі-середні дані + нелінійні = One-Class SVM ✓
- Великі дані → Isolation Forest ✓
- Локальні outliers → LOF ✓
- Швидкість критична → Isolation Forest ✓

**Практичні поради:**
- **ЗАВЖДИ scale features** (StandardScaler)
- Почни з RBF kernel та default
- Grid search для nu та gamma
- Використовуй decision_function для ranking
- Ensemble з Isolation Forest
- Sample якщо >10K зразків

---

#ml #unsupervised-learning #anomaly-detection #one-class-svm #novelty-detection #kernel-methods #support-vector-machines #outlier-detection
