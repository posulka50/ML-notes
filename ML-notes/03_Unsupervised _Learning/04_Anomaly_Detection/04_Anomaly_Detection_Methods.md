# Anomaly Detection Methods: Повний Огляд

## Що це?

**Anomaly Detection (Outlier Detection)** — це задача знаходження незвичайних, рідкісних зразків, що значно відрізняються від більшості даних. Це **unsupervised learning** задача (зазвичай немає labeled anomalies).

## Типи аномалій

### 1. Point Anomalies (Global)

**Що:** Окрема точка далеко від всіх інших.

```
        ●●●●●
        ●●●●●
        ●●●●●
                    ○ ← Point anomaly

Глобально відрізняється від всіх
```

**Приклади:**
- Транзакція на $10,000 коли зазвичай $50
- Температура сенсора 500°C коли норма 20-30°C

### 2. Contextual Anomalies (Conditional)

**Що:** Нормальна в одному контексті, аномальна в іншому.

```
Temperature over time:

Summer:  30°C ← Normal
Winter:  30°C ← ANOMALY! (зимою холодно)

Context matters!
```

**Приклади:**
- Покупка зимової куртки влітку
- High traffic о 3 AM (зазвичай низький)

### 3. Collective Anomalies

**Що:** Група точок разом аномальна.

```
Heartbeat:
Normal:   ●-●-●-●-●-●-●
Anomaly:  ●-●-●●●●●-●-●  ← Rapid sequence

Кожна точка OK, але sequence abnormal
```

**Приклади:**
- DDoS attack (багато requests разом)
- Credit card: багато транзакцій за 5 хвилин

---

## Основні методи

### Швидке порівняння

| Метод | Тип | Швидкість | Масштабованість | Локальні outliers | Interpretability |
|-------|-----|-----------|-----------------|-------------------|------------------|
| **Isolation Forest** | Tree-based | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **One-Class SVM** | Kernel | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **LOF** | Density | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Autoencoder** | Neural Net | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **DBSCAN** | Clustering | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Statistical** | Statistics | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

---

## 1. Isolation Forest

### Принцип

**Ізолювати аномалії через random splits.**

```
Аномалія:         Normal point:
○                 ●●●●●
|                 ●●●●●
1 split!          ●●●●●
                  |||||||
                  Many splits
```

### Переваги ✓
- ⚡ Дуже швидкий (O(n log n))
- 📈 Масштабується на мільйони
- 🔢 Добре на high-dimensional
- 💾 Memory efficient

### Недоліки ✗
- ❌ Потребує contamination parameter
- ❌ Може пропускати локальні outliers
- ❌ Погано на categorical features

### Коли використовувати
- ✅ Великі дані (> 10K points)
- ✅ High-dimensional (> 50 features)
- ✅ Швидкість критична
- ✅ Global anomalies

### Код
```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

clf.fit(X_train)
y_pred = clf.predict(X_test)  # 1=normal, -1=anomaly
scores = clf.score_samples(X_test)  # Lower = more anomalous
```

---

## 2. One-Class SVM

### Принцип

**Побудувати boundary навколо normal data.**

```
Feature space:
    ●●●●●●●
    ●     ●
    ●  ○  ●  ← Decision boundary
    ●     ●
    ●●●●●●●

Outside boundary = anomaly
```

### Переваги ✓
- 🎯 Kernel trick (нелінійні boundaries)
- 📐 Solid theory (SVM math)
- 🎨 Smooth boundaries
- ✅ Novelty detection

### Недоліки ✗
- ❌ Повільний (O(n²-n³))
- ❌ Не масштабується (< 10K)
- ❌ Потребує parameter tuning (nu, gamma)
- ❌ ОБОВ'ЯЗКОВИЙ scaling

### Коли використовувати
- ✅ Малі-середні дані (< 10K)
- ✅ Нелінійні patterns
- ✅ Smooth decision boundary важлива
- ✅ Novelty detection

### Код
```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ЗАВЖДИ scale!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

clf = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.1
)

clf.fit(X_scaled)
y_pred = clf.predict(scaler.transform(X_test))
```

---

## 3. Local Outlier Factor (LOF)

### Принцип

**Порівняти локальну щільність з сусідами.**

```
Dense cluster:        Sparse area:
  ●●●●●                 ○    ○
  ●●●●●               ○  ★    ○
  ●●●●●                 ○    ○

Point in dense:       Point ★:
LOF ≈ 1 (normal)      LOF >> 1 (outlier!)
```

### Переваги ✓
- 🎯 Знаходить локальні outliers
- 📊 Variable density (різна щільність)
- 💯 Interpretable score (LOF value)
- 🔍 Cluster outliers

### Недоліки ✗
- ❌ Повільний (O(n²))
- ❌ Не масштабується (< 50K)
- ❌ Чутливий до n_neighbors
- ❌ High-dimensional проблеми

### Коли використовувати
- ✅ Локальні outliers важливі
- ✅ Variable density в даних
- ✅ Середні дані (< 50K)
- ✅ Outliers між кластерами

### Код
```python
from sklearn.neighbors import LocalOutlierFactor

# For fit_predict
clf = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=False
)

y_pred = clf.fit_predict(X_train)
lof_scores = -clf.negative_outlier_factor_

# For novelty detection
clf_novelty = LocalOutlierFactor(novelty=True)
clf_novelty.fit(X_train)
y_pred = clf_novelty.predict(X_test)
```

---

## 4. Autoencoders

### Принцип

**Навчитись стискати та відновлювати. Аномалії = високий reconstruction error.**

```
Input → Encoder → Bottleneck → Decoder → Output
 ●●●     ↓          ●●           ↓        ●●●

Normal:  Input ≈ Output (low error)
Anomaly: Input ≠ Output (high error)
```

### Переваги ✓
- 🧠 Складні нелінійні patterns
- 📈 Масштабується добре
- 🎨 Unsupervised feature learning
- 🔧 Гнучкість (різні архітектури)

### Недоліки ✗
- ❌ Потребує багато даних для навчання
- ❌ Складно налаштувати
- ❌ Повільне навчання
- ❌ Не інтерпретовні

### Коли використовувати
- ✅ Великі дані (> 10K)
- ✅ Складні нелінійні patterns
- ✅ Image/sequence data
- ✅ Deep learning infrastructure є

### Код
```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# Train
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ... training loop ...

# Detect
with torch.no_grad():
    recon = model(X_test)
    errors = torch.mean((X_test - recon)**2, dim=1)
    
threshold = np.percentile(errors, 95)
anomalies = errors > threshold
```

---

## 5. Statistical Methods

### Z-Score

**Відстань від mean в standard deviations.**

$$z = \frac{x - \mu}{\sigma}$$

```python
from scipy.stats import zscore

z_scores = np.abs(zscore(X, axis=0))
outliers = (z_scores > 3).any(axis=1)  # 3-sigma rule
```

**Переваги:** Простий, швидкий, інтерпретований
**Недоліки:** Тільки Gaussian, univariate

### IQR (Interquartile Range)

**Викиди за межами Q1-1.5×IQR та Q3+1.5×IQR.**

```python
Q1 = np.percentile(X, 25)
Q3 = np.percentile(X, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = (X < lower) | (X > upper)
```

**Переваги:** Robust до outliers, не припускає розподілу
**Недоліки:** Univariate, може пропустити multivariate outliers

### Mahalanobis Distance

**Multivariate відстань з урахуванням covariance.**

$$D = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

```python
from scipy.spatial.distance import mahalanobis

mean = np.mean(X, axis=0)
cov = np.cov(X.T)
inv_cov = np.linalg.inv(cov)

distances = [mahalanobis(x, mean, inv_cov) for x in X]
threshold = np.percentile(distances, 95)
outliers = np.array(distances) > threshold
```

**Переваги:** Multivariate, враховує корреляції
**Недоліки:** Припускає Gaussian, потребує invertible covariance

---

## 6. DBSCAN (як outlier detector)

### Принцип

**Points не в жодному кластері = outliers.**

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

outliers = labels == -1  # -1 = noise/outlier
```

**Переваги:** Знаходить outliers автоматично, працює з variable density
**Недоліки:** Чутливий до eps та min_samples

---

## Decision Tree: Який метод вибрати?

```
Скільки даних?
├─ < 1,000
│  └─ Statistical methods (Z-score, IQR)
│
├─ 1,000 - 10,000
│  │
│  Чи важливі локальні outliers?
│  ├─ Так → LOF
│  └─ Ні
│     │
│     Чи нелінійні patterns?
│     ├─ Так → One-Class SVM
│     └─ Ні → Isolation Forest
│
└─ > 10,000
   │
   Чи є GPU та багато даних (>100K)?
   ├─ Так → Autoencoder
   └─ Ні → Isolation Forest
```

### За типом задачі

**Global outliers:**
1. Isolation Forest (best)
2. One-Class SVM
3. Statistical methods

**Local outliers:**
1. LOF (best)
2. DBSCAN
3. Autoencoder

**High-dimensional (>50 features):**
1. Isolation Forest (best)
2. Autoencoder
3. PCA + any method

**Streaming/Real-time:**
1. Statistical methods (fastest)
2. Isolation Forest
3. Incremental refit others

**Interpretability:**
1. Statistical methods (best)
2. Isolation Forest
3. LOF

---

## Порівняльний benchmark

### Synthetic data experiment

```python
import numpy as np
import time
from sklearn.metrics import roc_auc_score

# Generate data
np.random.seed(42)
n_normal = 1000
n_anomalies = 50

X_normal = np.random.randn(n_normal, 10)
X_anomalies = np.random.uniform(-4, 4, (n_anomalies, 10))

X = np.vstack([X_normal, X_anomalies])
y_true = np.array([0]*n_normal + [1]*n_anomalies)

# Methods to compare
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

methods = {
    'Isolation Forest': IsolationForest(contamination=0.05, random_state=42),
    'One-Class SVM': OneClassSVM(nu=0.05),
    'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
}

results = []

for name, clf in methods.items():
    print(f"\nTesting {name}...")
    
    # Time
    start = time.time()
    
    if name == 'LOF':
        y_pred = clf.fit_predict(X)
        scores = -clf.negative_outlier_factor_
    else:
        clf.fit(X)
        y_pred = clf.predict(X)
        scores = -clf.score_samples(X)
    
    elapsed = time.time() - start
    
    # Metrics
    y_pred_binary = (y_pred == -1).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary)
    rec = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, scores)
    
    results.append({
        'Method': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'Time (s)': elapsed
    })

# Display results
import pandas as pd
df_results = pd.DataFrame(results)
print("\n=== Benchmark Results ===")
print(df_results.to_string(index=False))

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Metrics
df_results.set_index('Method')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
    kind='bar', ax=axes[0], rot=45
)
axes[0].set_title('Performance Metrics', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Time
df_results.plot(x='Method', y='Time (s)', kind='bar', ax=axes[1], 
               legend=False, rot=45, color='steelblue')
axes[1].set_title('Execution Time', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Time (seconds)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Ensemble Methods

### Voting

**Комбінувати predictions різних методів.**

```python
# Multiple methods
if_pred = IsolationForest().fit_predict(X)
lof_pred = LocalOutlierFactor().fit_predict(X)
svm_pred = OneClassSVM().fit(X).predict(X)

# Voting: якщо >= 2 методи кажуть outlier
votes = (if_pred == -1).astype(int) + \
        (lof_pred == -1).astype(int) + \
        (svm_pred == -1).astype(int)

ensemble_pred = votes >= 2  # Majority vote
```

### Score averaging

**Комбінувати anomaly scores.**

```python
from sklearn.preprocessing import MinMaxScaler

# Get scores
if_scores = IsolationForest().fit(X).score_samples(X)
lof_scores = LocalOutlierFactor(novelty=True).fit(X).score_samples(X)

# Normalize
scaler = MinMaxScaler()
if_norm = scaler.fit_transform(if_scores.reshape(-1, 1)).ravel()
lof_norm = scaler.fit_transform(lof_scores.reshape(-1, 1)).ravel()

# Average
ensemble_scores = (if_norm + lof_norm) / 2

# Threshold
threshold = np.percentile(ensemble_scores, 5)
outliers = ensemble_scores < threshold
```

---

## Практичні рекомендації 💡

### 1. Почни з exploratory analysis

```python
# Подивись на дані!
import seaborn as sns

# Distributions
for col in X.columns:
    plt.figure()
    sns.histplot(X[col])
    plt.title(f'Distribution: {col}')
    plt.show()

# Correlations
sns.heatmap(X.corr(), annot=True)
plt.show()

# Outliers візуально
for col in X.columns:
    plt.boxplot(X[col])
    plt.title(f'Boxplot: {col}')
    plt.show()
```

### 2. Спробуй кілька методів

```python
# Не обмежуйся одним!
methods = [
    ('IF', IsolationForest()),
    ('LOF', LocalOutlierFactor(novelty=False)),
    ('OCSVM', OneClassSVM())
]

for name, clf in methods:
    if name == 'LOF':
        y_pred = clf.fit_predict(X)
    else:
        clf.fit(X)
        y_pred = clf.predict(X)
    
    print(f"{name}: {(y_pred == -1).sum()} outliers detected")
```

### 3. Validate якщо є labels

```python
# Якщо є навіть трохи labeled data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y_true, test_size=0.2, random_state=42
)

# Test methods
for name, clf in methods:
    clf.fit(X_train)
    y_pred = clf.predict(X_val)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, (y_pred == -1).astype(int))
    print(f"{name}: F1={f1:.3f}")
```

### 4. Feature engineering

```python
# Додай derived features
X['feature_ratio'] = X['feature1'] / (X['feature2'] + 1e-6)
X['feature_diff'] = X['feature1'] - X['feature2']
X['feature_product'] = X['feature1'] * X['feature2']

# Time-based features (якщо є timestamp)
X['hour'] = df['timestamp'].dt.hour
X['day_of_week'] = df['timestamp'].dt.dayofweek
```

### 5. Preprocessing

```python
# Scaling (для SVM та LOF)
from sklearn.preprocessing import StandardScaler, RobustScaler

# StandardScaler (чутливий до outliers)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# RobustScaler (robust до outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# For Isolation Forest: не потрібен scaling
```

### 6. Dimensionality reduction

```python
# Для high-dimensional
from sklearn.decomposition import PCA

# PCA спочатку
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X)

# Потім anomaly detection
clf = IsolationForest()
y_pred = clf.fit_predict(X_reduced)
```

### 7. Cross-validation для параметрів

```python
# Grid search якщо є validation
param_grid = {
    'contamination': [0.01, 0.05, 0.1, 0.15],
    'n_estimators': [50, 100, 200]
}

best_f1 = 0
best_params = None

for contamination in param_grid['contamination']:
    for n_estimators in param_grid['n_estimators']:
        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators
        )
        
        clf.fit(X_train)
        y_pred = clf.predict(X_val)
        
        f1 = f1_score(y_val, (y_pred == -1).astype(int))
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = {
                'contamination': contamination,
                'n_estimators': n_estimators
            }

print(f"Best params: {best_params}")
```

### 8. Monitor в production

```python
# Track metrics over time
class AnomalyMonitor:
    def __init__(self):
        self.history = []
    
    def log(self, timestamp, n_anomalies, avg_score):
        self.history.append({
            'timestamp': timestamp,
            'n_anomalies': n_anomalies,
            'avg_score': avg_score
        })
    
    def plot_trends(self):
        df = pd.DataFrame(self.history)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(df['timestamp'], df['n_anomalies'])
        axes[0].set_title('Anomalies Over Time')
        axes[0].set_ylabel('Count')
        
        axes[1].plot(df['timestamp'], df['avg_score'])
        axes[1].set_title('Average Anomaly Score')
        axes[1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.show()
```

### 9. Explainability

```python
# Для Isolation Forest - feature importance
def get_anomaly_explanation(clf, X, sample_idx):
    """Які features найбільш аномальні?"""
    
    sample = X[sample_idx]
    
    # Permutation importance
    base_score = clf.score_samples([sample])[0]
    
    importances = []
    for feature_idx in range(X.shape[1]):
        X_perm = sample.copy()
        X_perm[feature_idx] = np.median(X[:, feature_idx])
        
        score_perm = clf.score_samples([X_perm])[0]
        importance = abs(base_score - score_perm)
        importances.append(importance)
    
    return np.array(importances)

# Usage
sample_idx = 0  # Аномальна точка
importances = get_anomaly_explanation(clf, X, sample_idx)

# Топ features
top_features = np.argsort(importances)[-5:][::-1]
print("Most anomalous features:")
for idx in top_features:
    print(f"  Feature {idx}: importance={importances[idx]:.4f}")
```

### 10. A/B testing

```python
# Перевірити чи працює detection

# Baseline period (before detection system)
baseline_fraud_rate = 0.05  # 5%
baseline_loss = 100000  # $100K

# Test period (with detection)
detected_fraud = 40  # Caught
missed_fraud = 10    # Missed
total_fraud = 50

detection_rate = detected_fraud / total_fraud
test_loss = missed_fraud * avg_fraud_loss

print(f"Detection rate: {detection_rate:.1%}")
print(f"Loss reduction: ${baseline_loss - test_loss:,.0f}")
```

---

## Поширені помилки ❌

### 1. Не перевіряти результати

```python
# ❌ Сліпо довіряти
y_pred = clf.fit_predict(X)

# ✅ Завжди inspect
outlier_indices = np.where(y_pred == -1)[0]

print(f"Found {len(outlier_indices)} outliers")
print("Sample outliers:")
print(X[outlier_indices[:5]])

# Візуалізуй
if X.shape[1] == 2:
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()
```

### 2. Використовувати неправильний метод

```python
# ❌ LOF на 100,000 points
# Дуже повільно!

# ✅ Вибери метод під розмір даних
if len(X) > 50000:
    clf = IsolationForest()  # Швидко
else:
    clf = LocalOutlierFactor()  # Якість
```

### 3. Забути про scaling

```python
# ❌ Для One-Class SVM та LOF
clf = OneClassSVM()
clf.fit(X_raw)  # Погано!

# ✅ Scale спочатку
X_scaled = StandardScaler().fit_transform(X_raw)
clf.fit(X_scaled)
```

### 4. Неправильний contamination

```python
# ❌ contamination=0.5 (50%!)
# Половина даних аномалії?

# ✅ Realistic estimate
contamination = 0.01  # 1% для fraud
contamination = 0.05  # 5% для defects
```

### 5. Не використовувати ensemble

```python
# ❌ Один метод
# Може пропустити деякі типи аномалій

# ✅ Ensemble
if_pred = IsolationForest().fit_predict(X)
lof_pred = LocalOutlierFactor().fit_predict(X)

ensemble = (if_pred == -1) | (lof_pred == -1)
```

---

## Реальні кейси

### Case 1: Credit Card Fraud

**Вимоги:**
- Real-time (< 100ms)
- High precision (few false positives)
- Millions of transactions/day

**Рішення:**
```
1. Feature engineering: RFM, velocity, geo
2. Isolation Forest (fast, scalable)
3. Rules-based override (high amount + night)
4. Human review queue for borderline cases
```

**Результат:**
- 95% fraud detection rate
- < 50ms latency
- 2% false positive rate

### Case 2: Manufacturing Defects

**Вимоги:**
- Локальні аномалії (різні product types)
- Sensor time series data
- Пояснення чому defect

**Рішення:**
```
1. Segment by product type
2. LOF (handles variable density)
3. Feature importance для пояснень
4. Dashboard для operators
```

**Результат:**
- 85% defect detection (раніше 60%)
- Зменшення false positives на 40%

### Case 3: Network Intrusion

**Вимоги:**
- Novel attack types (zero-day)
- High-dimensional features (100+)
- Streaming data

**Рішення:**
```
1. PCA для dimensionality reduction
2. Autoencoder (learns normal patterns)
3. Incremental retraining
4. Alert prioritization by score
```

**Результат:**
- Виявлення 98% відомих attacks
- 60% novel attacks (не бачених раніше)

---

## Metrics для evaluation

### Якщо є labels

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

y_pred_binary = (y_pred == -1).astype(int)

print("=== Metrics ===")
print(f"Accuracy: {accuracy_score(y_true, y_pred_binary):.3f}")
print(f"Precision: {precision_score(y_true, y_pred_binary):.3f}")
print(f"Recall: {recall_score(y_true, y_pred_binary):.3f}")
print(f"F1-Score: {f1_score(y_true, y_pred_binary):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_true, scores):.3f}")
print(f"PR-AUC: {average_precision_score(y_true, scores):.3f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_binary)
print("\nConfusion Matrix:")
print(cm)
```

### Якщо немає labels

```python
# Unsupervised metrics

# 1. Silhouette score (якщо є кластери)
from sklearn.metrics import silhouette_score
sil = silhouette_score(X, y_pred)

# 2. Visual inspection
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

# 3. Domain expert review
outlier_samples = X[y_pred == -1]
print("Review these samples:")
print(outlier_samples[:10])

# 4. Stability (run multiple times)
predictions = []
for seed in range(10):
    clf = IsolationForest(random_state=seed)
    pred = clf.fit_predict(X)
    predictions.append(pred)

# Якщо стабільно → more confident
stability = np.mean([np.array_equal(predictions[0], p) 
                     for p in predictions[1:]])
print(f"Stability: {stability:.1%}")
```

---

## Пов'язані теми

- [[01_Isolation_Forest]] — tree-based метод
- [[02_One_Class_SVM]] — kernel-based
- [[03_Local_Outlier_Factor]] — density-based
- [[Clustering_Methods]] — DBSCAN для outliers
- [[Autoencoders]] — deep learning approach

## Ресурси

- [Scikit-learn: Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Anomaly Detection Survey](https://arxiv.org/abs/1901.03407)
- [PyOD Library](https://github.com/yzhao062/pyod) — багато методів

---

## Ключові висновки

> Anomaly Detection — це задача знаходження незвичайних зразків. Не існує одного "найкращого" методу — вибір залежить від розміру даних, типу аномалій, вимог до швидкості та інтерпретованості.

**Основні методи:**

| Метод | Best для | Розмір даних |
|-------|---------|--------------|
| **Isolation Forest** | Global outliers, швидкість | > 10K |
| **One-Class SVM** | Нелінійні boundaries | < 10K |
| **LOF** | Локальні outliers | < 50K |
| **Autoencoder** | Складні patterns | > 100K |
| **Statistical** | Простота, швидкість | Будь-який |

**Quick Decision Guide:**

```
START
  ↓
Скільки даних?
  ├─ < 10K → One-Class SVM або LOF
  └─ > 10K → Isolation Forest

Локальні outliers важливі?
  ├─ Так → LOF
  └─ Ні → Isolation Forest

High-dimensional (>50)?
  ├─ Так → Isolation Forest або PCA+method
  └─ Ні → Будь-який метод

Швидкість критична?
  ├─ Так → Isolation Forest
  └─ Ні → Залежить від задачі
```

**Best Practices:**
1. Спробуй кілька методів
2. Ensemble для кращих results
3. Validate якщо є labels
4. Feature engineering критичний
5. Scale features (SVM, LOF)
6. Monitor в production
7. Explain decisions
8. Iterate based на feedback

**Найважливіше:**
- Розумій свої дані та domain
- Не існує silver bullet
- Validate, validate, validate
- Ensemble часто найкраще
- Interpretability vs accuracy trade-off

---

#ml #unsupervised-learning #anomaly-detection #outlier-detection #comparison #ensemble #isolation-forest #one-class-svm #lof #methods-overview
