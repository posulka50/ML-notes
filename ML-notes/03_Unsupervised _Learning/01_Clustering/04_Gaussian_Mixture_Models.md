# Gaussian Mixture Models (GMM)

## Що це?

**Gaussian Mixture Model (GMM)** — це **probabilistic** алгоритм unsupervised learning, який моделює дані як суміш (mixture) кількох **багатовимірних нормальних (Gaussian) розподілів**.

**Головна ідея:** кожен кластер представлений Gaussian розподілом з власним середнім (mean) та коваріацією (covariance). Кожна точка має **ймовірність належності** до кожного кластера, а не жорстку мітку.

## Навіщо потрібна?

- 🎲 **Soft clustering** — ймовірності замість жорстких міток
- 🔔 **Гнучка форма** — еліптичні кластери різних форм
- 📊 **Density estimation** — моделювання розподілу даних
- 🎯 **Uncertainty** — квантифікація впевненості в кластеризації
- 🧬 **Generative model** — можна генерувати нові точки
- 📈 **Перекриття кластерів** — точки можуть частково належати декільком кластерам

## Коли використовувати?

**Потрібно:**
- **Soft clustering** — ймовірності належності важливі
- **Еліптичні кластери** різних форм/орієнтацій
- **Перекриття кластерів** — нечіткі границі
- **Density estimation** — моделювання розподілу
- **Генерація даних** — потрібен generative model
- **Uncertainty quantification** — наскільки впевнені в кластеризації
- **Statistically motivated** — потрібне теоретичне обґрунтування

**Не потрібно:**
- Кластери **дуже складної форми** → DBSCAN
- Потрібна **швидкість** → K-Means
- **Categorical features** → K-Modes
- Дуже **великі дані** → K-Means (швидше)

---

## Основні концепції

### 1. Gaussian (Normal) Distribution

**Одновимірний Gaussian:**

$$p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

де:
- $\mu$ — середнє (mean)
- $\sigma^2$ — дисперсія (variance)

**Багатовимірний Gaussian:**

$$p(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

де:
- $\boldsymbol{\mu}$ — вектор середніх (mean vector)
- $\boldsymbol{\Sigma}$ — матриця коваріації (covariance matrix)
- $d$ — розмірність
- $|\boldsymbol{\Sigma}|$ — детермінант коваріації

### 2. Mixture Model

**GMM = зважена сума K Gaussian розподілів:**

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

де:
- $K$ — кількість компонентів (кластерів)
- $\pi_k$ — **mixing coefficient** (вага компонента), $\sum_{k=1}^{K} \pi_k = 1$
- $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ — k-й Gaussian компонент

### 3. Soft Clustering (Responsibilities)

**Posterior probability (responsibility):** ймовірність що точка $\mathbf{x}_i$ належить кластеру $k$:

$$\gamma_{ik} = p(z_i = k | \mathbf{x}_i) = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

де $z_i$ — прихована змінна (latent variable), що вказує на кластер.

**Приклад:**
```
Точка x має:
- 70% ймовірність належати кластеру 1
- 25% ймовірність належати кластеру 2  
- 5% ймовірність належати кластеру 3
```

### 4. Covariance Types

**Форма матриці коваріації визначає форму кластера:**

#### Full Covariance

$$\boldsymbol{\Sigma}_k = \begin{bmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{21} & \sigma_{22} \end{bmatrix}$$

- ✅ Кожен кластер має **власну форму** та **орієнтацію**
- ✅ Найгнучкіший
- ❌ Багато параметрів: $K \times d \times (d+1)/2$

```
Приклад: еліпси різних форм
    
    ●●●●           ●
   ●    ●        ● ● ●
  ●      ●      ●  ●  ●
   ●    ●        ● ● ●
    ●●●●           ●
Широкий        Вузький
горизонтальний вертикальний
```

#### Tied Covariance

$$\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma} \text{ для всіх } k$$

- Всі кластери мають **однакову форму** та орієнтацію
- Тільки позиція відрізняється
- Менше параметрів

#### Diagonal Covariance

$$\boldsymbol{\Sigma}_k = \begin{bmatrix} \sigma_{11} & 0 \\ 0 & \sigma_{22} \end{bmatrix}$$

- Еліпси **вирівняні по осях** (без обертання)
- Кожна вісь незалежна
- Середня кількість параметрів

```
Приклад:
    ●●●
   ●   ●
  ●     ●
   ●   ●
    ●●●
Еліпс вирівняний по осях
```

#### Spherical Covariance

$$\boldsymbol{\Sigma}_k = \sigma_k^2 \mathbf{I}$$

- **Кругові/сферичні** кластери
- Найменше параметрів
- Аналогічно K-Means

```
Приклад:
   ●●●
  ●   ●
  ●   ●
   ●●●
Коло (сфера)
```

---

## EM Algorithm (Expectation-Maximization)

**GMM навчається через EM algorithm** — ітераційний метод для знаходження максимуму likelihood.

### Алгоритм

**Вхід:** дані $X$, кількість компонентів $K$

**1. Ініціалізація:**
   - Випадкові $\boldsymbol{\mu}_k$, $\boldsymbol{\Sigma}_k$, $\pi_k$
   - Або використати K-Means для початкових центрів

**2. Повторювати до збіжності:**

   **E-step (Expectation):**
   - Обчислити responsibilities (posterior probabilities):
   
   $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

   **M-step (Maximization):**
   - Оновити параметри для максимізації expected log-likelihood:
   
   **Mixing coefficients:**
   $$\pi_k^{new} = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}$$
   
   **Means:**
   $$\boldsymbol{\mu}_k^{new} = \frac{\sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i}{\sum_{i=1}^{N} \gamma_{ik}}$$
   
   **Covariances:**
   $$\boldsymbol{\Sigma}_k^{new} = \frac{\sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{new})(\mathbf{x}_i - \boldsymbol{\mu}_k^{new})^T}{\sum_{i=1}^{N} \gamma_{ik}}$$

**3. Перевірка збіжності:**
   - Якщо зміна log-likelihood < threshold → зупинити

**4. Вихід:** параметри $\{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^{K}$

### Псевдокод

```
EM_GMM(X, K, max_iter=100, tol=1e-3):
    # Ініціалізація
    initialize(μ, Σ, π)  # Випадково або через K-Means
    
    log_likelihood_old = -∞
    
    for iter in range(max_iter):
        # E-step: обчислити responsibilities
        for i in 1..N:
            for k in 1..K:
                γ[i,k] = π[k] * N(x[i] | μ[k], Σ[k])
            γ[i] = γ[i] / sum(γ[i])  # Нормалізація
        
        # M-step: оновити параметри
        for k in 1..K:
            N_k = sum(γ[:, k])
            
            π[k] = N_k / N
            μ[k] = sum(γ[:, k] * X) / N_k
            Σ[k] = sum(γ[:, k] * (X - μ[k])@(X - μ[k]).T) / N_k
        
        # Перевірка збіжності
        log_likelihood = compute_log_likelihood(X, π, μ, Σ)
        
        if abs(log_likelihood - log_likelihood_old) < tol:
            break
        
        log_likelihood_old = log_likelihood
    
    return π, μ, Σ, γ
```

### Візуалізація EM процесу

```
Ітерація 0 (Ініціалізація):
    ●  ●  ●     ●  ●
      ●  ●   ●  ●
    ●     ●  ●
    
Випадкові центри та коваріації

Ітерація 1 (E-step → M-step):
    🔴  ●  🔵     ●  🟢
      ●  🔴   🔵  ●
    🔴     ●  🟢
    
Responsibilities обчислені, параметри оновлені

Ітерація 5:
   🔴🔴🔴     🔵🔵     🟢🟢
    🔴🔴      🔵🔵🔵    🟢🟢🟢
   🔴🔴🔴      🔵🔵     🟢🟢
   
Кластери стабілізувались
```

---

## Простий приклад: Висота людей

### Дані

Висота 100 людей (в см):

```
Жінки: μ=165, σ=6 (40 людей)
Чоловіки: μ=178, σ=7 (60 людей)
```

Ми **не знаємо** стать, тільки висоту. GMM має знайти 2 розподіли.

### GMM з K=2

**Після навчання:**

```
Компонент 1: μ₁=165.2, σ₁=6.1, π₁=0.39
Компонент 2: μ₂=177.8, σ₂=7.2, π₂=0.61
```

**Інтерпретація:**
- Компонент 1 ≈ Жінки (39% популяції)
- Компонент 2 ≈ Чоловіки (61% популяції)

### Soft Clustering

**Людина висотою 170 см:**

$$\gamma_1 = P(\text{Компонент 1} | x=170) = 0.42$$
$$\gamma_2 = P(\text{Компонент 2} | x=170) = 0.58$$

**Інтерпретація:**
- 42% ймовірність жінка
- 58% ймовірність чоловік
- Невпевненість через перекриття розподілів

**Людина висотою 160 см:**

$$\gamma_1 = 0.89, \quad \gamma_2 = 0.11$$

**Людина висотою 185 см:**

$$\gamma_1 = 0.05, \quad \gamma_2 = 0.95$$

---

## Складний приклад: Iris Dataset

### Задача

Iris dataset: 150 квітів, 4 ознаки (довжина/ширина sepal/petal), 3 види.

**Мета:** Кластеризувати без знання виду (unsupervised).

### Результат GMM

**K=3 компоненти:**

| Компонент | Mean Petal Length | Mean Petal Width | π_k | Інтерпретація |
|-----------|-------------------|------------------|-----|---------------|
| 0 | 1.46 | 0.25 | 0.33 | Setosa |
| 1 | 4.26 | 1.33 | 0.33 | Versicolor |
| 2 | 5.60 | 2.03 | 0.34 | Virginica |

### Soft Clustering

**Квітка з petal_length=3.5, petal_width=1.0:**

```
Responsibilities:
- Setosa: 5%
- Versicolor: 85%
- Virginica: 10%

→ Скоріш за все Versicolor, але є невпевненість
```

**Переваги GMM над K-Means:**
- Показує **невпевненість** (перехідні квіти)
- **Еліптичні кластери** (краще відповідають даним)
- **Статистична інтерпретація** (розподіли)

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 1. Генерація даних
X, y_true = make_blobs(n_samples=300, centers=3, 
                       cluster_std=[1.0, 1.5, 0.5],
                       random_state=42)

# 2. GMM
gmm = GaussianMixture(
    n_components=3,           # Кількість компонентів (кластерів)
    covariance_type='full',   # Тип коваріації
    max_iter=100,             # Максимум ітерацій EM
    n_init=10,                # Кількість random ініціалізацій
    random_state=42
)

# 3. Навчання
gmm.fit(X)

# 4. Predict (hard clustering)
labels = gmm.predict(X)

# 5. Predict probabilities (soft clustering)
probs = gmm.predict_proba(X)

print("=== GMM Results ===")
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"\nMeans:\n{gmm.means_}")
print(f"\nCovariances shape: {gmm.covariances_.shape}")
print(f"\nWeights (π): {gmm.weights_}")
print(f"\nLog-likelihood: {gmm.score(X) * len(X):.2f}")

# 6. Приклад soft clustering
print("\n=== Example: Soft Clustering ===")
sample_idx = 0
print(f"Point: {X[sample_idx]}")
print(f"Probabilities: {probs[sample_idx]}")
print(f"Hard label: {labels[sample_idx]}")

# 7. Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Hard clustering
axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
axes[0].scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[0].set_title('Hard Clustering', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# Soft clustering (розміри точок = впевненість)
max_prob = probs.max(axis=1)
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
               s=max_prob*200, alpha=0.6)
axes[1].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[1].set_title('Soft Clustering (size = confidence)', 
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].grid(True, alpha=0.3)

# Density (contours)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[2].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
contours = axes[2].contour(xx, yy, Z, levels=10, linewidths=2, alpha=0.7)
axes[2].clabel(contours, inline=True, fontsize=8)
axes[2].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[2].set_title('Density Contours', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Порівняння covariance types

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Генерація даних з еліптичними кластерами
np.random.seed(42)

# Кластер 1: горизонтальний еліпс
cov1 = [[2.0, 0.5], [0.5, 0.5]]
X1 = np.random.multivariate_normal([0, 0], cov1, 100)

# Кластер 2: вертикальний еліпс
cov2 = [[0.5, 0.3], [0.3, 2.0]]
X2 = np.random.multivariate_normal([5, 5], cov2, 100)

# Кластер 3: діагональний
cov3 = [[1.0, 0.8], [0.8, 1.0]]
X3 = np.random.multivariate_normal([5, 0], cov3, 100)

X = np.vstack([X1, X2, X3])

# Порівняти різні covariance types
covariance_types = ['full', 'tied', 'diag', 'spherical']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, cov_type in enumerate(covariance_types):
    gmm = GaussianMixture(
        n_components=3,
        covariance_type=cov_type,
        random_state=42
    )
    
    labels = gmm.fit_predict(X)
    
    # Візуалізація
    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                     s=30, alpha=0.6)
    axes[idx].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                     c='red', marker='X', s=200, 
                     edgecolors='black', linewidths=2)
    
    # Density contours
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contour(xx, yy, Z, levels=10, alpha=0.5)
    
    axes[idx].set_title(f'{cov_type.capitalize()} Covariance', 
                       fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Feature 1', fontsize=11)
    axes[idx].set_ylabel('Feature 2', fontsize=11)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nCovariance Types Comparison:")
print("Full: Most flexible, captures ellipses at any angle")
print("Tied: All clusters same shape, different positions")
print("Diag: Ellipses aligned with axes")
print("Spherical: Circular clusters (like K-Means)")
```

### Повний приклад: Customer Segmentation з soft clustering

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Генерація даних про клієнтів
np.random.seed(42)

# 4 сегменти з різними характеристиками
segments = {
    'VIP': {'n': 50, 'recency': (5, 2), 'frequency': (25, 5), 'monetary': (1500, 300)},
    'Active': {'n': 100, 'recency': (15, 5), 'frequency': (12, 3), 'monetary': (500, 100)},
    'Regular': {'n': 80, 'recency': (30, 8), 'frequency': (6, 2), 'monetary': (300, 80)},
    'At-Risk': {'n': 70, 'recency': (90, 20), 'frequency': (2, 1), 'monetary': (150, 50)}
}

data_list = []
for seg_name, params in segments.items():
    n = params['n']
    recency = np.random.normal(params['recency'][0], params['recency'][1], n)
    frequency = np.random.normal(params['frequency'][0], params['frequency'][1], n)
    monetary = np.random.normal(params['monetary'][0], params['monetary'][1], n)
    
    for i in range(n):
        data_list.append({
            'Recency': max(1, recency[i]),
            'Frequency': max(1, frequency[i]),
            'Monetary': max(50, monetary[i]),
            'True_Segment': seg_name
        })

df = pd.DataFrame(data_list)

print("=== Dataset Info ===")
print(f"Total customers: {len(df)}")
print(f"\n{df.groupby('True_Segment').size()}")
print(f"\n{df.describe()}")

# Нормалізація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

# GMM з різними K
n_components_range = range(2, 8)
bic_scores = []
aic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, 
                         covariance_type='full',
                         random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

# Візуалізація BIC/AIC
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n_components_range, bic_scores, 'o-', label='BIC', linewidth=2)
ax.plot(n_components_range, aic_scores, 's-', label='AIC', linewidth=2)
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Information Criterion', fontsize=12)
ax.set_title('Model Selection: BIC and AIC', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

optimal_k = n_components_range[np.argmin(bic_scores)]
print(f"\nOptimal K (BIC): {optimal_k}")

# Навчити фінальну модель
gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',
    n_init=20,
    random_state=42
)

gmm.fit(X_scaled)

# Hard clustering
df['Cluster'] = gmm.predict(X_scaled)

# Soft clustering (probabilities)
probs = gmm.predict_proba(X_scaled)
df['Prob_0'] = probs[:, 0]
df['Prob_1'] = probs[:, 1]
df['Prob_2'] = probs[:, 2]
df['Prob_3'] = probs[:, 3]

# Максимальна ймовірність (впевненість)
df['Confidence'] = probs.max(axis=1)

print("\n" + "="*70)
print("=== GMM Results ===")
print("="*70)
print(f"Converged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"\nWeights (π): {gmm.weights_}")
print(f"\nMeans (in scaled space):\n{gmm.means_}")

# Inverse transform means для інтерпретації
means_original = scaler.inverse_transform(gmm.means_)
means_df = pd.DataFrame(means_original, 
                       columns=['Recency', 'Frequency', 'Monetary'])
print(f"\nMeans (original scale):\n{means_df}")

# Аналіз кластерів
print("\n" + "="*70)
print("=== Cluster Analysis ===")
print("="*70)

for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster} (n={len(cluster_data)}):")
    print(cluster_data[['Recency', 'Frequency', 'Monetary', 'Confidence']].describe())
    
    # Найбільш впевнені точки
    most_confident = cluster_data.nlargest(5, 'Confidence')
    print(f"\nMost confident assignments (top 5):")
    print(most_confident[['Recency', 'Frequency', 'Monetary', 'Confidence']])

# Аналіз невпевненості
print("\n" + "="*70)
print("=== Uncertainty Analysis ===")
print("="*70)

# Точки з низькою впевненістю (перехідні)
uncertain_threshold = 0.6
uncertain_points = df[df['Confidence'] < uncertain_threshold]

print(f"Points with confidence < {uncertain_threshold}: {len(uncertain_points)} "
      f"({len(uncertain_points)/len(df)*100:.1f}%)")

if len(uncertain_points) > 0:
    print("\nExample uncertain points:")
    print(uncertain_points.nlargest(5, 'Confidence', keep='last')[
        ['Recency', 'Frequency', 'Monetary', 'Cluster', 
         'Prob_0', 'Prob_1', 'Prob_2', 'Prob_3', 'Confidence']
    ])

# Візуалізація
fig = plt.figure(figsize=(18, 12))

# 3D scatter з confidence
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
scatter = ax1.scatter(df['Recency'], df['Frequency'], df['Monetary'],
                     c=df['Cluster'], cmap='viridis', 
                     s=df['Confidence']*100, alpha=0.6)
ax1.set_xlabel('Recency', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_zlabel('Monetary', fontsize=10)
ax1.set_title('3D Clusters (size = confidence)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Cluster')

# Recency vs Frequency з confidence
ax2 = fig.add_subplot(2, 3, 2)
scatter2 = ax2.scatter(df['Recency'], df['Frequency'], 
                      c=df['Cluster'], cmap='viridis',
                      s=df['Confidence']*100, alpha=0.6,
                      edgecolors='black', linewidths=0.5)
ax2.set_xlabel('Recency', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Recency vs Frequency', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Cluster')

# Frequency vs Monetary
ax3 = fig.add_subplot(2, 3, 3)
scatter3 = ax3.scatter(df['Frequency'], df['Monetary'],
                      c=df['Cluster'], cmap='viridis',
                      s=df['Confidence']*100, alpha=0.6,
                      edgecolors='black', linewidths=0.5)
ax3.set_xlabel('Frequency', fontsize=11)
ax3.set_ylabel('Monetary', fontsize=11)
ax3.set_title('Frequency vs Monetary', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='Cluster')

# Cluster profiles
ax4 = fig.add_subplot(2, 3, 4)
cluster_profiles = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
cluster_profiles_norm = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
sns.heatmap(cluster_profiles_norm.T, annot=True, fmt='.2f',
           cmap='RdYlGn_r', center=0, ax=ax4, cbar_kws={'label': 'Std. Value'})
ax4.set_title('Cluster Profiles (Normalized)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Cluster', fontsize=11)

# Confidence distribution
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(df['Confidence'], bins=30, edgecolor='black', alpha=0.7)
ax5.axvline(uncertain_threshold, color='red', linestyle='--', 
           linewidth=2, label=f'Threshold={uncertain_threshold}')
ax5.set_xlabel('Confidence', fontsize=11)
ax5.set_ylabel('Count', fontsize=11)
ax5.set_title('Distribution of Confidence Scores', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Cluster sizes
ax6 = fig.add_subplot(2, 3, 6)
cluster_sizes = df['Cluster'].value_counts().sort_index()
bars = ax6.bar(cluster_sizes.index, cluster_sizes.values, 
              edgecolor='black', alpha=0.7)
for bar, weight in zip(bars, gmm.weights_):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'π={weight:.2f}',
            ha='center', va='bottom', fontsize=10)
ax6.set_xlabel('Cluster', fontsize=11)
ax6.set_ylabel('Count', fontsize=11)
ax6.set_title('Cluster Sizes (with weights)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Генерація нових даних
print("\n" + "="*70)
print("=== Generative Model: Sample New Customers ===")
print("="*70)

# Згенерувати 5 нових клієнтів
new_samples_scaled, labels = gmm.sample(5)
new_samples = scaler.inverse_transform(new_samples_scaled)

new_df = pd.DataFrame(new_samples, columns=['Recency', 'Frequency', 'Monetary'])
new_df['Generated_Cluster'] = labels

print("\nGenerated customers:")
print(new_df)
```

### Density Estimation

```python
# GMM для density estimation
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X)

# Обчислити log-likelihood для кожної точки
log_likelihood = gmm.score_samples(X)

# Або для нових точок
X_new = np.array([[0, 0], [10, 10]])
log_prob_new = gmm.score_samples(X_new)

print(f"Log probability for new points: {log_prob_new}")
print(f"Probability: {np.exp(log_prob_new)}")

# Визначити outliers (низька ймовірність)
threshold = np.percentile(log_likelihood, 5)  # 5% найнижчих
outliers = X[log_likelihood < threshold]

print(f"Detected {len(outliers)} outliers")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=log_likelihood, cmap='viridis', s=50)
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x', 
           s=100, linewidths=2, label='Outliers')
plt.colorbar(label='Log Likelihood')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Density Estimation with GMM')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Вибір кількості компонентів K

### Проблема

**GMM потребує заздалегідь задати K** (як K-Means).

### 1. BIC (Bayesian Information Criterion) ⭐

**Найпопулярніший метод для GMM.**

$$\text{BIC} = -2 \ln(\mathcal{L}) + k \ln(n)$$

де:
- $\mathcal{L}$ — likelihood
- $k$ — кількість параметрів
- $n$ — кількість точок

**Менше BIC = краще** (компроміс між fit та складністю)

```python
from sklearn.mixture import GaussianMixture

# Перевірити різні K
n_components_range = range(1, 11)
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, 
                         covariance_type='full',
                         random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))

# Оптимальний K
optimal_k = n_components_range[np.argmin(bic_scores)]

print(f"Optimal K (BIC): {optimal_k}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Components (K)', fontsize=12)
plt.ylabel('BIC', fontsize=12)
plt.title('BIC vs Number of Components', fontsize=14, fontweight='bold')
plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. AIC (Akaike Information Criterion)

$$\text{AIC} = -2 \ln(\mathcal{L}) + 2k$$

**Менше AIC = краще**

AIC менш консервативний ніж BIC (схильний до більших K).

```python
aic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(X)
    aic_scores.append(gmm.aic(X))

optimal_k_aic = n_components_range[np.argmin(aic_scores)]
print(f"Optimal K (AIC): {optimal_k_aic}")
```

### 3. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = []

for n_components in range(2, 11):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    # Negative log-likelihood (lower is better)
    score = cross_val_score(gmm, X, cv=5, scoring='neg_log_loss').mean()
    scores.append(-score)

optimal_k = range(2, 11)[np.argmax(scores)]
```

### 4. Silhouette Score

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for n_components in range(2, 11):
    gmm = GaussianMixture(n_components=n_components)
    labels = gmm.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

optimal_k = range(2, 11)[np.argmax(silhouette_scores)]
```

**Рекомендація:** Використовуй **BIC** як основний метод ✓

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Soft clustering** | Ймовірності належності замість жорстких міток |
| **Uncertainty quantification** | Показує невпевненість у кластеризації |
| **Гнучка форма кластерів** | Еліптичні кластери різних форм/орієнтацій |
| **Статистичне обґрунтування** | Probabilistic model з теоретичною базою |
| **Density estimation** | Може моделювати розподіл даних |
| **Generative model** | Можна генерувати нові точки |
| **Перекриття кластерів** | Добре працює з нечіткими границями |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Потрібно знати K** | Кількість компонентів задається заздалегідь |
| **Складність O(nKd³)** | Повільніше за K-Means |
| **Припущення Gaussian** | Працює погано якщо дані не Gaussian |
| **Локальні мінімуми** | EM може застрягти (потрібно n_init) |
| **Чутливість до ініціалізації** | Результат залежить від початкових параметрів |
| **Багато параметрів** | Full covariance: K × d × (d+1)/2 параметрів |
| **Не для складних форм** | Не знайде S-криві, кільця (→ DBSCAN) |

---

## Порівняння з іншими методами

| Метод | Clustering Type | Потрібно K? | Форма кластерів | Outliers | Складність |
|-------|-----------------|-------------|-----------------|----------|------------|
| **GMM** | Soft (probabilistic) | ✅ Так | Еліптичні | ⚠️ Середньо | O(nKd³) |
| **K-Means** | Hard | ✅ Так | Сферичні | ❌ Чутливий | O(nKdi) |
| **DBSCAN** | Hard | ❌ Ні | Будь-які | ✅ Виявляє | O(n log n) |
| **Hierarchical** | Hard | ❌ Ні | Будь-які | ⚠️ Середньо | O(n²-n³) |

### GMM vs K-Means

**GMM:**
- ✅ Soft clustering (ймовірності)
- ✅ Еліптичні кластери
- ✅ Статистична інтерпретація
- ❌ Повільніше
- ❌ Більше параметрів

**K-Means:**
- ✅ Швидше
- ✅ Простіше
- ❌ Тільки сферичні кластери
- ❌ Hard clustering

```python
# Порівняння на одних даних
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

gmm = GaussianMixture(n_components=3, random_state=42)
labels_gmm = gmm.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis')
axes[0].set_title('K-Means (Hard, Spherical)', fontsize=13, fontweight='bold')

axes[1].scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='viridis')
axes[1].set_title('GMM (Soft, Elliptical)', fontsize=13, fontweight='bold')

plt.show()
```

---

## Коли використовувати GMM

### Ідеально підходить ✓

- **Soft clustering** потрібен — ймовірності важливі
- **Uncertainty quantification** — наскільки впевнені
- **Еліптичні кластери** різних форм
- **Density estimation** — моделювання розподілу
- **Generative tasks** — генерація нових даних
- **Перекриття кластерів** — нечіткі границі
- **Статистична інтерпретація** важлива
- Дані **приблизно Gaussian**

### Краще використати інше ✗

- **Дуже складна форма** (S-криві, кільця) → DBSCAN
- Потрібна **швидкість** → K-Means
- **Дуже великі дані** → K-Means, Mini-Batch K-Means
- Дані **не Gaussian** → DBSCAN, Mean Shift
- **Не знаємо K** і важливо знайти автоматично → DBSCAN, HDBSCAN

---

## Практичні поради 💡

### 1. Завжди використовуй BIC для вибору K

```python
# Перевір різні K
bic_scores = []
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, covariance_type='full')
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))

optimal_k = np.argmin(bic_scores) + 1
```

### 2. Почни з full covariance

```python
# Full covariance найгнучкіший
gmm = GaussianMixture(n_components=3, covariance_type='full')

# Якщо overfitting → спробуй tied або diag
```

### 3. Використовуй n_init для стабільності

```python
# EM може застрягти в локальних мінімумах
# Спробуй кілька ініціалізацій
gmm = GaussianMixture(
    n_components=3,
    n_init=20,  # 20 різних ініціалізацій
    random_state=42
)
```

### 4. Scaling КРИТИЧНИЙ

```python
# ЗАВЖДИ нормалізуй
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm.fit(X_scaled)
```

### 5. Аналізуй uncertainty

```python
# Знайди точки з низькою впевненістю
probs = gmm.predict_proba(X)
confidence = probs.max(axis=1)

uncertain = X[confidence < 0.6]
print(f"Uncertain points: {len(uncertain)}")
```

### 6. Використовуй для outlier detection

```python
# Низька ймовірність = outlier
log_prob = gmm.score_samples(X)
threshold = np.percentile(log_prob, 5)
outliers = X[log_prob < threshold]
```

### 7. Візуалізуй density contours

```python
# Показуй density для інтерпретації
x_grid, y_grid = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
)

Z = -gmm.score_samples(np.c_[x_grid.ravel(), y_grid.ravel()])
Z = Z.reshape(x_grid.shape)

plt.contour(x_grid, y_grid, Z, levels=10)
```

### 8. Порівняй з K-Means

```python
# Якщо результати дуже схожі → можна використати K-Means (швидше)
# Якщо дуже відрізняються → еліптичні кластери, GMM краще
```

### 9. Регуляризація для малих даних

```python
# Додай регуляризацію до коваріації
gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',
    reg_covar=1e-6  # Додати до діагоналі
)
```

### 10. Ініціалізація через K-Means

```python
# За замовчуванням GMM використовує K-Means для ініціалізації
gmm = GaussianMixture(
    n_components=3,
    init_params='kmeans'  # або 'random'
)
```

---

## Реальні застосування

### 1. Customer Segmentation з uncertainty

**Задача:** Сегментувати клієнтів, виявити "граничні" випадки.

**Підхід:**
- RFM features
- GMM для soft clustering
- Точки з low confidence → персоналізовані кампанії

**Переваги:**
- Знаємо наскільки впевнені в сегменті
- Можемо виявити клієнтів у перехідних станах

### 2. Image Segmentation

**Задача:** Розділити зображення на регіони.

**Підхід:**
- Кожен піксель = точка в кольоровому просторі (RGB)
- GMM моделює розподіл кольорів
- Кожен компонент = один колір/об'єкт

### 3. Anomaly Detection

**Задача:** Виявити аномалії в даних.

**Підхід:**
- GMM моделює нормальний розподіл
- Low probability points = anomalies
- Density estimation природно для цього

### 4. Speech Recognition

**Задача:** Розпізнавання мовлення.

**Підхід:**
- GMM моделює фонеми (звуки)
- Кожна фонема = Gaussian mixture
- Hidden Markov Models + GMM

### 5. Background Subtraction (Computer Vision)

**Задача:** Виділити рухомі об'єкти на відео.

**Підхід:**
- GMM моделює background кожного пікселя
- Відхилення від моделі = foreground (об'єкт)

---

## Поширені помилки ❌

### 1. Не використовувати scaling

```python
# ❌ Вік (0-100) + Дохід (0-150K)
gmm = GaussianMixture(n_components=3)
gmm.fit(X)  # Дохід домінує!

# ✅ Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm.fit(X_scaled)
```

### 2. Не перевіряти convergence

```python
# ❌ Просто fit
gmm.fit(X)

# ✅ Перевіряй чи зійшлось
gmm.fit(X)
if not gmm.converged_:
    print("WARNING: EM did not converge!")
    print(f"Iterations: {gmm.n_iter_}")
```

### 3. Використовувати тільки 1 ініціалізацію

```python
# ❌ Може застрягти в локальному мінімумі
gmm = GaussianMixture(n_components=3, n_init=1)

# ✅ Кілька ініціалізацій
gmm = GaussianMixture(n_components=3, n_init=10)
```

### 4. Ігнорувати covariance type

```python
# ❌ Завжди full (може бути overkill)
gmm = GaussianMixture(n_components=10, covariance_type='full')
# Дуже багато параметрів!

# ✅ Спробуй tied або diag якщо full перенавчається
gmm = GaussianMixture(n_components=10, covariance_type='tied')
```

### 5. Не аналізувати uncertainty

```python
# ❌ Тільки hard labels
labels = gmm.predict(X)

# ✅ Аналізуй ймовірності
probs = gmm.predict_proba(X)
confidence = probs.max(axis=1)

# Знайди невпевнені точки
uncertain = X[confidence < 0.7]
```

### 6. Використовувати на non-Gaussian даних

```python
# ❌ Дані з heavy tails, multimodal але не Gaussian
# GMM дасть погані результати

# ✅ Спочатку перевір розподіл
import scipy.stats as stats

for feature in range(X.shape[1]):
    stat, p_value = stats.normaltest(X[:, feature])
    print(f"Feature {feature}: p={p_value:.4f}")
    # Якщо p < 0.05 → НЕ Gaussian
```

---

## Пов'язані теми

- [[01_KMeans]] — hard clustering, швидша альтернатива
- [[03_DBSCAN]] — density-based, складні форми
- [[02_Hierarchical_Clustering]] — ієрархічна кластеризація
- [[05_Clustering_Evaluation]] — метрики оцінки
- [[EM_Algorithm]] — алгоритм навчання
- [[Bayesian_GMM]] — Bayesian підхід до GMM

## Ресурси

- [Scikit-learn: Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html)
- [Original Paper: Dempster et al. (1977) - EM Algorithm](https://www.jstor.org/stable/2984875)
- [Bishop: Pattern Recognition (Chapter 9)](https://www.microsoft.com/en-us/research/people/cmbishop/)
- [StatQuest: Gaussian Mixture Models](https://www.youtube.com/watch?v=qMTuMa86NzU)

---

## Ключові висновки

> Gaussian Mixture Models (GMM) моделюють дані як суміш Gaussian розподілів, забезпечуючи soft clustering з ймовірностями належності замість жорстких міток.

**Основні принципи:**
- **Probabilistic model:** кожна точка має ймовірність належності до кожного кластера
- **EM algorithm:** ітеративне навчання (E-step → M-step)
- **Soft clustering:** responsibilities (γᵢₖ) показують ймовірності
- **Covariance types:** full, tied, diag, spherical (форма кластерів)

**Ключові компоненти:**
- **Mixing coefficients (π):** ваги компонентів
- **Means (μ):** центри розподілів
- **Covariances (Σ):** форма та орієнтація еліпсів

**Вибір K:**
- **BIC** (Bayesian Information Criterion) — найкращий метод ✓
- AIC (менш консервативний)
- Cross-validation, Silhouette

**Коли використовувати:**
- Soft clustering + uncertainty + еліптичні кластери = GMM ✓
- Швидкість + сферичні → K-Means ✓
- Складна форма → DBSCAN ✓

**Найважливіше:**
- Scaling критичний
- Використовуй BIC для вибору K
- n_init ≥ 10 для стабільності
- Full covariance найгнучкіший
- Аналізуй uncertainty (confidence scores)
- Працює найкраще на Gaussian даних

---

#ml #unsupervised-learning #clustering #gmm #gaussian-mixture-models #probabilistic #soft-clustering #em-algorithm #density-estimation
