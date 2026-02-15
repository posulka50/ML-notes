# PCA (Principal Component Analysis)

## Що це?

**PCA (Principal Component Analysis)** — це алгоритм **dimensionality reduction** (зменшення розмірності), який перетворює дані в новий простір з меншою кількістю вимірів, зберігаючи **максимум інформації** (варіації).

**Головна ідея:** знайти нові осі (principal components), вздовж яких дані мають найбільшу дисперсію, та спроектувати дані на ці осі.

## Навіщо потрібен?

- 📊 **Зменшення розмірності** — з 100 features до 10 без великої втрати інформації
- 🎨 **Візуалізація** — візуалізувати високо розмірні дані в 2D/3D
- ⚡ **Прискорення навчання** — менше features → швидше ML алгоритми
- 🧹 **Видалення шуму** — компоненти з низькою variance = шум
- 🔍 **Feature extraction** — знайти найважливіші напрямки варіації
- 💾 **Стиснення даних** — зберігати менше даних
- 🎯 **Multicollinearity** — декорелювати сильно корельовані features

## Коли використовувати?

**Потрібно

- **Багато features** (>10-20) — curse of dimensionality
- Features **корельовані** між собою
- Потрібна **візуалізація** високо розмірних даних
- **Числові features** (неперервні)
- Дані **приблизно лінійні** (лінійні залежності)
- Потрібна **інтерпретованість** — що означають компоненти
- **Preprocessing** перед supervised learning

**Не потрібно:**

- **Мало features** (3-5) — не потрібно зменшувати
- **Categorical features** → One-hot encoding спочатку
- **Сильно нелінійні дані** → t-SNE, UMAP, Autoencoders
- Потрібна **розрідженість** (sparsity) → NMF
- **Supervised task** де features важливі → Feature selection

---

### Математичний процес

**Вхід:** матриця даних $X$ розміру $n \times d$ (n точок, d features)

**Крок 1: Центрування даних**

- Відняти середнє від кожної ознаки:
$$X_{centered} = X - \bar{X}$$

**Крок 2: Обчислити коваріаційну матрицю**
$$C = \frac{1}{n-1} X_{centered}^T X_{centered}$$

**Крок 3: Знайти власні вектори та власні значення**
- Розв'язати: $C \mathbf{v} = \lambda \mathbf{v}$
- $\mathbf{v}$ — власний вектор (principal component)
- $\lambda$ — власне значення (variance вздовж компонента)

**Крок 4: Відсортувати за власними значеннями**
- $\lambda_1 > \lambda_2 > \lambda_3 > ...$

**Крок 5: Вибрати top k компонентів**
- Перші $k$ власних векторів → матриця $W$ розміру $d \times k$

**Крок 6: Трансформувати дані**
$$X_{transformed} = X_{centered} \cdot W$$

**Вихід:** нові дані розміру $n \times k$ (зменшена розмірність!)

---

## Математика

### Коваріаційна матриця

**Для 2 features:**

$$C = \begin{bmatrix} 
\text{Var}(X_1) & \text{Cov}(X_1, X_2) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2)
\end{bmatrix}$$

**Variance (дисперсія):**
$$\text{Var}(X) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Covariance (коваріація):**
$$\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

### Власні вектори та власні значення

**Eigenvalue equation:**
$$C \mathbf{v} = \lambda \mathbf{v}$$

де:
- $C$ — коваріаційна матриця
- $\mathbf{v}$ — власний вектор (напрямок principal component)
- $\lambda$ — власне значення (variance вздовж цього напрямку)

### Пояснена варіація (Explained Variance)

**Частка variance поясненої k-м компонентом:**

$$\text{Explained Variance Ratio}_k = \frac{\lambda_k}{\sum_{i=1}^{d} \lambda_i}$$

**Cumulative explained variance:**

$$\text{Cumulative}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

**Приклад:**
```
PC1: 65% variance
PC2: 20% variance
PC3: 10% variance
PC4: 5% variance

Cumulative:
PC1: 65%
PC1+PC2: 85%
PC1+PC2+PC3: 95%  ← зберігає 95% інформації!
```

---

## Простий приклад: 2D → 1D

### Дані

Студенти: оцінки з Math та Physics (сильно корельовані)

| Student | Math | Physics |
|---------|------|---------|
| A | 90 | 85 |
| B | 80 | 78 |
| C | 70 | 72 |
| D | 60 | 58 |
| E | 50 | 52 |

### Візуалізація

```
Physics
  100|
     |
   80|    A •
     |       B •
   60|          C •
     |             D •
   40|                E •
     |
    0|_________________
      0  20  40  60  80 100  Math
```

**Проблема:** 2 features, але вони дуже схожі (якщо Math високий → Physics високий)

### PCA Process

**1. Центрування:**

```
Mean(Math) = 70, Mean(Physics) = 69

Centered data:
Student | Math  | Physics
A       | +20   | +16
B       | +10   | +9
C       | 0     | +3
D       | -10   | -11
E       | -20   | -17
```

**2. Коваріаційна матриця:**

```
C = [100   98]
    [98    96]
```

**3. Власні вектори:**

```
PC1 = [0.71, 0.70]  λ₁ = 196  (98% variance)
PC2 = [-0.70, 0.71] λ₂ = 4    (2% variance)
```

**4. Проекція на PC1:**

```
Student | PC1 (нова координата)
A       | +25.4
B       | +13.4
C       | +2.1
D       | -14.8
E       | -26.1
```

**Результат:** 2D → 1D, зберігаючи 98% інформації!

**PC1 можна інтерпретувати як:** "загальна успішність" (Math + Physics)

---

## Складний приклад: Iris Dataset

### Задача

Iris dataset: 150 квітів, 4 features (sepal/petal length/width), 3 види.

**Мета:** Зменшити з 4D до 2D для візуалізації.

### Результат PCA

**Explained variance:**
```
PC1: 72.96%
PC2: 22.85%
PC3: 3.67%
PC4: 0.52%

PC1+PC2: 95.81% ← 2 компоненти зберігають 96% інформації!
```

**Інтерпретація компонентів:**

**PC1** = 0.52×sepal_length + 0.37×sepal_width + 0.72×petal_length + 0.26×petal_width
- Найбільший вклад: petal_length
- **Інтерпретація:** "розмір квітки загалом"

**PC2** = 0.38×sepal_length - 0.86×sepal_width + 0.17×petal_length + 0.08×petal_width
- Найбільший вклад: sepal_width (негативний)
- **Інтерпретація:** "форма квітки" (широкий vs вузький)

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. Завантажити дані
iris = load_iris()
X = iris.data  # 150 samples, 4 features
y = iris.target

print(f"Original shape: {X.shape}")  # (150, 4)

# 2. ОБОВ'ЯЗКОВО: Стандартизація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=2)  # Зменшити до 2 компонентів
X_pca = pca.fit_transform(X_scaled)

print(f"Transformed shape: {X_pca.shape}")  # (150, 2)

# 4. Explained variance
print("\n=== Explained Variance ===")
print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Total: {pca.explained_variance_ratio_.sum():.4f}")

# 5. Components (loadings)
print("\n=== Principal Components ===")
components_df = pd.DataFrame(
    pca.components_,
    columns=iris.feature_names,
    index=['PC1', 'PC2']
)
print(components_df)

# 6. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# До PCA (2 з 4 features)
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
axes[0].set_xlabel(iris.feature_names[0], fontsize=11)
axes[0].set_ylabel(iris.feature_names[1], fontsize=11)
axes[0].set_title('Original Data (2 of 4 features)', 
                 fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Після PCA (всі 4 features → 2 PC)
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=y, cmap='viridis', s=50, alpha=0.6)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                  fontsize=11)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                  fontsize=11)
axes[1].set_title('After PCA (all 4 features → 2 PCs)', 
                 fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Легенда
plt.colorbar(scatter, ax=axes[1], label='Species', 
            ticks=[0, 1, 2])

plt.tight_layout()
plt.show()
```

### Вибір кількості компонентів

```python
# Метод 1: Explained variance ratio
pca = PCA()  # Всі компоненти
pca.fit(X_scaled)

# Cumulative explained variance
cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(12, 5))

# Scree plot
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_, alpha=0.7, edgecolor='black')
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('Scree Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Cumulative variance
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum) + 1), cumsum, 'o-', linewidth=2, markersize=8)
plt.axhline(y=0.95, color='red', linestyle='--', 
           label='95% threshold', linewidth=2)
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance', fontsize=12)
plt.title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Знайти кількість компонентів для 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"\nКількість компонентів для 95% variance: {n_components_95}")

# Метод 2: Задати бажану variance
pca_95 = PCA(n_components=0.95)  # 95% variance
X_pca_95 = pca_95.fit_transform(X_scaled)
print(f"Вибрано компонентів: {pca_95.n_components_}")
```

### Повний приклад: MNIST

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# Завантажити дані (8x8 = 64 features)
digits = load_digits()
X = digits.data  # (1797, 64)
y = digits.target

print(f"Original shape: {X.shape}")
print(f"Features: 64 pixels (8x8 image)")

# Стандартизація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: всі компоненти для аналізу
pca_full = PCA()
pca_full.fit(X_scaled)

# Аналіз explained variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

print("\n=== Explained Variance Analysis ===")
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_comp = np.argmax(cumsum >= threshold) + 1
    print(f"{threshold:.0%} variance: {n_comp} components "
          f"(compression: {64/n_comp:.1f}x)")

# Візуалізація explained variance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scree plot
axes[0, 0].bar(range(1, 21), 
              pca_full.explained_variance_ratio_[:20],
              alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Principal Component', fontsize=11)
axes[0, 0].set_ylabel('Explained Variance Ratio', fontsize=11)
axes[0, 0].set_title('Scree Plot (first 20 PCs)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Cumulative variance
axes[0, 1].plot(range(1, 65), cumsum, 'o-', linewidth=2, markersize=4)
axes[0, 1].axhline(y=0.95, color='red', linestyle='--', 
                  label='95% threshold', linewidth=2)
axes[0, 1].axhline(y=0.90, color='orange', linestyle='--',
                  label='90% threshold', linewidth=2, alpha=0.7)
axes[0, 1].set_xlabel('Number of Components', fontsize=11)
axes[0, 1].set_ylabel('Cumulative Explained Variance', fontsize=11)
axes[0, 1].set_title('Cumulative Variance', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# PCA 2D візуалізація
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

scatter = axes[1, 0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                            c=y, cmap='tab10', s=20, alpha=0.6)
axes[1, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})', 
                     fontsize=11)
axes[1, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})', 
                     fontsize=11)
axes[1, 0].set_title('2D Projection (64D → 2D)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Digit')

# Перші 10 principal components (як зображення)
axes[1, 1].axis('off')
axes[1, 1].set_title('First 10 Principal Components', 
                    fontsize=12, fontweight='bold')

# Створити sub-grid для компонентів
for i in range(10):
    ax = plt.subplot(4, 5, i + 11)
    component = pca_full.components_[i].reshape(8, 8)
    ax.imshow(component, cmap='RdBu_r', aspect='auto')
    ax.set_title(f'PC{i+1}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Реконструкція зображень з різною кількістю компонентів
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Image Reconstruction with Different Numbers of PCs', 
            fontsize=14, fontweight='bold')

sample_idx = 0  # Перше зображення (цифра)
original = X_scaled[sample_idx]

n_components_list = [1, 2, 5, 10, 20, 30, 40, 50, 60, 64]

for idx, n_comp in enumerate(n_components_list):
    ax = axes[idx // 5, idx % 5]
    
    if n_comp == 64:
        # Оригінал
        reconstructed = original
        mse = 0
    else:
        # PCA реконструкція
        pca_temp = PCA(n_components=n_comp)
        X_temp = pca_temp.fit_transform(X_scaled)
        reconstructed = pca_temp.inverse_transform(X_temp)[sample_idx]
        mse = np.mean((original - reconstructed) ** 2)
    
    # Відобразити
    img = scaler.inverse_transform(reconstructed.reshape(1, -1)).reshape(8, 8)
    ax.imshow(img, cmap='gray', aspect='auto')
    ax.set_title(f'{n_comp} PCs\nMSE: {mse:.4f}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Compression vs Quality
n_comp_range = range(1, 65)
mse_scores = []

for n_comp in n_comp_range:
    pca_temp = PCA(n_components=n_comp)
    X_temp = pca_temp.fit_transform(X_scaled)
    X_reconstructed = pca_temp.inverse_transform(X_temp)
    
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    mse_scores.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(n_comp_range, mse_scores, linewidth=2)
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Reconstruction Error vs Number of Components', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n=== Reconstruction Quality ===")
for n_comp in [5, 10, 20, 30, 40]:
    idx = n_comp - 1
    print(f"{n_comp:2d} components: MSE = {mse_scores[idx]:.6f}, "
          f"Compression = {64/n_comp:.1f}x")
```

### Inverse Transform (реконструкція)

```python
# PCA forward transform
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Inverse transform (реконструкція)
X_reconstructed_scaled = pca.inverse_transform(X_pca)

# Inverse scaling
X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)

# Порівняння
original_sample = X[0]
reconstructed_sample = X_reconstructed[0]

print("Original:", original_sample[:4])
print("Reconstructed:", reconstructed_sample[:4])
print(f"MSE: {np.mean((original_sample - reconstructed_sample)**2):.6f}")
```

---

## Kernel PCA (для нелінійних даних)

### Проблема

**Лінійний PCA** не працює з нелінійними структурами:

```
Before (нелінійна структура):
    y
    |  ●●●
    | ●   ●
    |●     ●
    |●     ●
    | ●   ●
    |  ●●●
    |_______ x
    Коло

After лінійний PCA:
PC2|
   |●●●●●●●
   |
   |_______ PC1
   
Не розділяє! ❌
```

### Рішення: Kernel PCA

**Ідея:** Відобразити дані в вищу розмірність через kernel trick, потім PCA.

```python
from sklearn.decomposition import KernelPCA

# Kernel PCA з RBF kernel
kpca = KernelPCA(
    n_components=2,
    kernel='rbf',      # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma=15,          # Параметр kernel
    fit_inverse_transform=True
)

X_kpca = kpca.fit_transform(X)

# Візуалізація
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Оригінальні дані
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
axes[0].set_title('Original Data', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Лінійний PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
axes[1].set_title('Linear PCA', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Kernel PCA
axes[2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', s=50)
axes[2].set_title('Kernel PCA (RBF)', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Kernel types

| Kernel | Формула | Використання |
|--------|---------|--------------|
| **Linear** | $\mathbf{x}^T \mathbf{y}$ | Лінійні дані (= звичайний PCA) |
| **Polynomial** | $(\gamma \mathbf{x}^T \mathbf{y} + c)^d$ | Поліноміальні залежності |
| **RBF (Gaussian)** | $\exp(-\gamma \|\mathbf{x} - \mathbf{y}\|^2)$ | Складні нелінійні структури |
| **Sigmoid** | $\tanh(\gamma \mathbf{x}^T \mathbf{y} + c)$ | Нейронні мережі |

---

## Incremental PCA (для великих даних)

### Проблема

**Звичайний PCA:** потребує завантажити всі дані в пам'ять.

**Великі дані** (не вміщуються в RAM) → проблема!

### Рішення: Incremental PCA

**Ідея:** Обробляти дані батчами (по частинах).

```python
from sklearn.decomposition import IncrementalPCA

# Incremental PCA
n_batches = 10
inc_pca = IncrementalPCA(n_components=50)

# Навчання по батчам
batch_size = len(X) // n_batches

for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    
    batch = X_scaled[start:end]
    inc_pca.partial_fit(batch)

# Transform
X_inc_pca = inc_pca.transform(X_scaled)

print(f"Incremental PCA shape: {X_inc_pca.shape}")
print(f"Explained variance: {inc_pca.explained_variance_ratio_.sum():.4f}")
```

**Коли використовувати:**
- Дані не вміщуються в RAM
- Streaming data (дані приходять поступово)
- Дуже великі датасети (GB+)

---

## Sparse PCA

### Проблема

**Звичайний PCA:** principal components - це лінійні комбінації **всіх** features.

**Важко інтерпретувати:** кожен PC використовує всі features.

### Рішення: Sparse PCA

**Ідея:** Змусити components бути **розрідженими** (багато нулів).

```python
from sklearn.decomposition import SparsePCA

# Sparse PCA
spca = SparsePCA(
    n_components=5,
    alpha=1.0,        # Regularization (більше = більше sparsity)
    max_iter=100,
    random_state=42
)

X_spca = spca.fit_transform(X_scaled)

# Порівняння з звичайним PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Sparsity comparison
print("=== Sparsity Comparison ===")
print(f"PCA zeros: {np.sum(pca.components_ == 0)}")
print(f"Sparse PCA zeros: {np.sum(spca.components_ == 0)}")

# Візуалізація компонентів
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    # PCA
    axes[0, i].bar(range(len(pca.components_[i])), 
                   np.abs(pca.components_[i]))
    axes[0, i].set_title(f'PCA PC{i+1}', fontsize=10)
    axes[0, i].set_ylim(0, 0.6)
    
    # Sparse PCA
    axes[1, i].bar(range(len(spca.components_[i])), 
                   np.abs(spca.components_[i]))
    axes[1, i].set_title(f'Sparse PC{i+1}', fontsize=10)
    axes[1, i].set_ylim(0, 0.6)

axes[0, 0].set_ylabel('PCA\nAbsolute Weight', fontsize=11)
axes[1, 0].set_ylabel('Sparse PCA\nAbsolute Weight', fontsize=11)

plt.tight_layout()
plt.show()
```

**Переваги:**
- ✅ Легше інтерпретувати (менше features в кожному PC)
- ✅ Feature selection (автоматично вибирає важливі features)

**Недоліки:**
- ❌ Повільніше
- ❌ Менше explained variance

---

## Preprocessing для PCA

### 1. Scaling (КРИТИЧНО! ⚠️)

**PCA дуже чутливий до масштабу!**

```python
# ❌ БЕЗ SCALING - НЕПРАВИЛЬНО!
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Feature з більшим діапазоном домінує!
# Наприклад: вік (0-100) vs зарплата (0-150000)
# PCA буде майже повністю базуватися на зарплаті!

# ✅ ЗІ SCALING - ПРАВИЛЬНО!
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

**Приклад без/зі scaling:**

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Дані: вік (20-80) та зарплата (20000-150000)
np.random.seed(42)
age = np.random.uniform(20, 80, 100)
salary = np.random.uniform(20000, 150000, 100)

X = np.column_stack([age, salary])

# Без scaling
pca_no_scale = PCA(n_components=2)
pca_no_scale.fit(X)

print("=== WITHOUT SCALING ===")
print("PC1 loadings:", pca_no_scale.components_[0])
print("Explained variance:", pca_no_scale.explained_variance_ratio_)
# PC1 майже повністю = salary (бо більший масштаб)

# Зі scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_with_scale = PCA(n_components=2)
pca_with_scale.fit(X_scaled)

print("\n=== WITH SCALING ===")
print("PC1 loadings:", pca_with_scale.components_[0])
print("Explained variance:", pca_with_scale.explained_variance_ratio_)
# Тепер обидві features впливають рівномірно
```

### 2. Missing Values

**PCA не працює з NaN!**

```python
# Обробка missing values
from sklearn.impute import SimpleImputer

# Заповнити середнім
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Потім scaling та PCA
X_scaled = scaler.fit_transform(X_imputed)
X_pca = pca.fit_transform(X_scaled)
```

### 3. Outliers

**Outliers сильно впливають на PCA!**

```python
# Видалити екстремальні outliers
from scipy import stats

z_scores = np.abs(stats.zscore(X))
mask = (z_scores < 3).all(axis=1)
X_clean = X[mask]

# Або robust scaling
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Зменшення розмірності** | 100 features → 10 без великої втрати інформації |
| **Візуалізація** | Високорозмірні дані → 2D/3D |
| **Декорреляція features** | Компоненти незалежні (ортогональні) |
| **Видалення шуму** | Компоненти з низькою variance = шум |
| **Прискорення ML** | Менше features → швидше навчання |
| **Математично обґрунтований** | Оптимальна проекція (максимум variance) |
| **Детерміністичний** | Однакові результати (без random) |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Лінійність** | Працює тільки з лінійними залежностями |
| **Чутливість до масштабу** | ОБОВ'ЯЗКОВИЙ scaling |
| **Важко інтерпретувати PC** | Лінійні комбінації всіх features |
| **Втрата інформації** | Зменшення розмірності = втрата деталей |
| **Чутливість до outliers** | Викиди впливають на PC |
| **Потребує багато пам'яті** | Коваріаційна матриця d×d |
| **Supervised info втрачена** | Не використовує labels (якщо є) |

---

## Порівняння з іншими методами

| Метод | Лінійний? | Supervised? | Інтерпретованість | Використання |
|-------|-----------|-------------|-------------------|--------------|
| **PCA** | ✅ Так | ❌ Ні | ⭐⭐⭐ | Загальне зменшення розмірності |
| **t-SNE** | ❌ Ні | ❌ Ні | ⭐ | Візуалізація (тільки 2D/3D) |
| **UMAP** | ❌ Ні | ❌ Ні | ⭐⭐ | Візуалізація + downstream tasks |
| **LDA** | ✅ Так | ✅ Так | ⭐⭐⭐⭐ | Класифікація (supervised) |
| **Autoencoders** | ❌ Ні | ❌ Ні | ⭐ | Складні нелінійні структури |

### PCA vs t-SNE

**PCA:**
- ✅ Швидкий
- ✅ Детерміністичний
- ✅ Працює з будь-якою розмірністю
- ❌ Тільки лінійний
- ❌ Глобальна структура (може втратити локальні паттерни)

**t-SNE:**
- ✅ Нелінійний (знаходить складні структури)
- ✅ Краща візуалізація кластерів
- ❌ Повільний
- ❌ Тільки для візуалізації (2D/3D)
- ❌ Недетерміністичний

**Рекомендація:** PCA спочатку (100D → 50D), потім t-SNE (50D → 2D) ✓

---

## Коли використовувати PCA

### Ідеально підходить ✓

- **Багато features** (>10-20) — curse of dimensionality
- **Корельовані features** — PCA декорелює
- **Візуалізація** високорозмірних даних
- **Preprocessing** перед ML (прискорення)
- **Видалення шуму** — викинути компоненти з низькою variance
- **Стиснення даних** — зберігати менше
- **Лінійні залежності** між features
- **Числові features** (неперервні)

### Краще використати інше ✗

- **Нелінійні структури** → Kernel PCA, Autoencoders, UMAP
- **Supervised task** де labels важливі → LDA
- **Тільки візуалізація** кластерів → t-SNE, UMAP
- **Categorical features** → MCA (Multiple Correspondence Analysis)
- **Інтерпретованість** критична → Feature selection
- **Розрідженість** (sparsity) важлива → Sparse PCA, NMF

---

## Практичні поради 💡

### 1. ЗАВЖДИ робити scaling!

```python
# ❌ КРИТИЧНА ПОМИЛКА
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# ✅ ПРАВИЛЬНО
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = pca.fit_transform(X_scaled)
```

### 2. Використовуй 95% variance як threshold

```python
# Cumulative variance
pca = PCA()
pca.fit(X_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)

# Знайти n_components для 95%
n_components = np.argmax(cumsum >= 0.95) + 1
print(f"95% variance: {n_components} components")

# Або автоматично
pca = PCA(n_components=0.95)  # Зберегти 95% variance
X_pca = pca.fit_transform(X_scaled)
```

### 3. Візуалізуй scree plot

```python
# Scree plot для вибору n_components
pca = PCA()
pca.fit(X_scaled)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()

# Шукай "лікоть" (elbow)
```

### 4. Інтерпретуй компоненти

```python
# Які features найбільше впливають на кожен PC?
components_df = pd.DataFrame(
    pca.components_,
    columns=feature_names,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

# Візуалізація
plt.figure(figsize=(12, 6))
sns.heatmap(components_df, cmap='RdBu_r', center=0, 
           annot=True, fmt='.2f')
plt.title('Principal Component Loadings')
plt.tight_layout()
plt.show()

# Найважливіші features для PC1
pc1_loadings = np.abs(components_df.iloc[0])
top_features = pc1_loadings.nlargest(5)
print("Top 5 features for PC1:")
print(top_features)
```

### 5. PCA для preprocessing перед ML

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Pipeline: scaling → PCA → classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)

print(f"Accuracy: {score:.4f}")
print(f"Used {pipeline.named_steps['pca'].n_components_} components")
```

### 6. Видали outliers перед PCA

```python
# Outliers сильно впливають на PC
from scipy import stats

z_scores = np.abs(stats.zscore(X))
mask = (z_scores < 3).all(axis=1)
X_clean = X[mask]

print(f"Removed {np.sum(~mask)} outliers")
```

### 7. Використовуй Incremental PCA для великих даних

```python
# Якщо дані не вміщуються в RAM
from sklearn.decomposition import IncrementalPCA

inc_pca = IncrementalPCA(n_components=50, batch_size=1000)

# Навчання по батчам
for batch in data_batches:
    inc_pca.partial_fit(batch)
```

### 8. Kernel PCA для нелінійних даних

```python
# Якщо лінійний PCA не працює
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X_scaled)
```

### 9. Зберігай scaler та PCA разом

```python
import joblib

# Зберегти
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

# Завантажити та застосувати
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

X_new_scaled = scaler.transform(X_new)
X_new_pca = pca.transform(X_new_scaled)
```

### 10. Перевіряй reconstruction error

```python
# Якщо reconstruction error великий → потрібно більше компонентів
X_reconstructed = pca.inverse_transform(X_pca)
mse = np.mean((X_scaled - X_reconstructed) ** 2)

print(f"Reconstruction MSE: {mse:.6f}")

# Або для окремих зразків
errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
worst_samples = np.argsort(errors)[-5:]
print(f"Samples with worst reconstruction: {worst_samples}")
```

---

## Реальні застосування

### 1. Image Compression

**Задача:** Зменшити розмір зображень без великої втрати якості.

**Підхід:**
- Кожен піксель = feature
- PCA на пікселях
- Зберігати тільки top k компонентів

**Результат:**
- 1000×1000 RGB = 3,000,000 values
- PCA (500 компонентів) = 500 values
- Compression: 6000x!

```python
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image

# Завантажити зображення
img = Image.open('image.jpg').convert('RGB')
img_array = np.array(img)  # (height, width, 3)

# Обробити кожен канал окремо
compressed_channels = []

for channel in range(3):  # R, G, B
    channel_data = img_array[:, :, channel]
    
    # PCA
    pca = PCA(n_components=50)  # Зберегти 50 компонентів
    transformed = pca.fit_transform(channel_data)
    
    compressed_channels.append({
        'pca': pca,
        'transformed': transformed
    })

# Реконструкція
reconstructed_img = np.zeros_like(img_array)

for channel in range(3):
    pca = compressed_channels[channel]['pca']
    transformed = compressed_channels[channel]['transformed']
    
    reconstructed = pca.inverse_transform(transformed)
    reconstructed_img[:, :, channel] = np.clip(reconstructed, 0, 255)

reconstructed_img = reconstructed_img.astype(np.uint8)

# Зберегти
Image.fromarray(reconstructed_img).save('compressed.jpg')

# Compression ratio
original_size = img_array.size
compressed_size = sum(
    ch['transformed'].size + ch['pca'].components_.size 
    for ch in compressed_channels
)

print(f"Compression ratio: {original_size / compressed_size:.1f}x")
```

### 2. Face Recognition (Eigenfaces)

**Задача:** Розпізнавання облич.

**Підхід:**
- Кожне зображення обличчя = точка в високорозмірному просторі
- PCA знаходить "eigenfaces" (головні компоненти облич)
- Нове обличчя = лінійна комбінація eigenfaces

**Переваги:**
- Зменшує розмірність (100×100 = 10,000D → 50D)
- Видаляє шум
- Швидке розпізнавання

### 3. Recommender Systems

**Задача:** Рекомендувати фільми користувачам.

**Підхід:**
- Матриця users × movies (дуже розріджена!)
- PCA знаходить латентні фактори (жанри, стилі)
- User preferences = комбінація латентних факторів

**Приклад:**
```
Original: 10,000 users × 5,000 movies = 50M features
PCA: 10,000 users × 20 factors = 200K features
Compression: 250x!
```

### 4. Gene Expression Analysis

**Задача:** Проаналізувати дані експресії генів.

**Дані:**
- Зразки пацієнтів × гени (1000 × 20,000)
- Дуже високорозмірні!

**Підхід:**
- PCA для зменшення до 50-100 компонентів
- Візуалізація в 2D/3D
- Кластеризація пацієнтів

**Результат:**
- Виявлення підтипів захворювань
- Прогностичні маркери

### 5. Financial Portfolio Analysis

**Задача:** Аналіз кореляцій між акціями.

**Підхід:**
- Features = ціни різних акцій
- PCA знаходить "факторні портфелі"
- PC1 часто = "ринковий фактор" (загальний тренд ринку)

**Застосування:**
- Диверсифікація портфеля
- Risk management
- Factor investing

---

## Поширені помилки ❌

### 1. Не робити scaling

```python
# ❌ БЕЗ SCALING
# Features: вік (20-80), зарплата (20K-150K), стаж (0-40)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Зарплата повністю домінує!

# ✅ ЗІ SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = pca.fit_transform(X_scaled)
```

### 2. Використовувати PCA на categorical features

```python
# ❌ PCA на категоріях
# Features: країна (UK, US, FR), стать (M, F)
pca = PCA(n_components=2)
pca.fit(X_categorical)  # Безглуздо!

# ✅ One-hot encoding спочатку
X_encoded = pd.get_dummies(df, drop_first=True)
X_scaled = scaler.fit_transform(X_encoded)
X_pca = pca.fit_transform(X_scaled)
```

### 3. Ігнорувати explained variance

```python
# ❌ Просто вибрати n_components=2 для візуалізації
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Можливо втрачено 95% інформації!

# ✅ Перевірити explained variance
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Якщо < 80% → додати більше компонентів або PCA не підходить
```

### 4. Застосовувати PCA на train+test разом

```python
# ❌ FIT на всіх даних (включаючи test)
X_all = np.vstack([X_train, X_test])
pca = PCA(n_components=10)
pca.fit(X_all)  # DATA LEAKAGE!

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# ✅ FIT тільки на train
pca = PCA(n_components=10)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

### 5. Ігнорувати outliers

```python
# ❌ PCA з outliers
# Викиди сильно впливають на principal components!

# ✅ Видалити або robust scaling
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
```

### 6. Очікувати що PCA покращить модель

```python
# ❌ "PCA завжди покращує accuracy"
# PCA видаляє інформацію! Не гарантує покращення.

# ✅ Порівняй з/без PCA
# З PCA: швидше, менше overfitting
# Без PCA: більше інформації, можливо краща accuracy
```

### 7. Використовувати PCA для feature selection

```python
# ❌ "PCA вибирає найкращі features"
# PCA створює НОВІ features (комбінації старих)!

# ✅ Для feature selection використовуй:
from sklearn.feature_selection import SelectKBest
# або Recursive Feature Elimination
```

### 8. Забути inverse_transform при реконструкції

```python
# ❌ Забути про scaler при реконструкції
X_pca = pca.transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)
# X_reconstructed все ще scaled!

# ✅ Inverse scaling
X_reconstructed_original = scaler.inverse_transform(X_reconstructed)
```

---

## Пов'язані теми

- [[02_t-SNE]] — нелінійна візуалізація
- [[03_UMAP]] — швидша альтернатива t-SNE
- [[04_LDA]] — supervised dimensionality reduction
- [[05_Autoencoders]] — neural network based
- [[06_Manifold_Learning]] — нелінійні методи
- [[Feature_Selection]] — альтернатива до PCA
- [[SVD]] — математична основа PCA

## Ресурси

- [Scikit-learn: PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Original Paper: Pearson (1901)](https://www.tandfonline.com/doi/abs/10.1080/14786440109462720)
- [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [A Tutorial on PCA (Shlens, 2014)](https://arxiv.org/abs/1404.1100)

---

## Ключові висновки

> PCA знаходить нові осі (principal components) вздовж яких дані мають найбільшу variance, дозволяючи зменшити розмірність зберігаючи максимум інформації.

**Основні принципи:**

- **Лінійна трансформація:** нові features = лінійні комбінації старих
- **Максимум variance:** PC1 захоплює найбільшу variance, PC2 — другу найбільшу, і т.д.
- **Ортогональність:** principal components перпендикулярні (декорельовані)
- **Unsupervised:** не використовує labels (якщо є)

**Процес:**

1. Центрування даних (відняти середнє)
2. Обчислити коваріаційну матрицю
3. Знайти власні вектори та значення
4. Відсортувати за власними значеннями
5. Вибрати top k компонентів
6. Трансформувати дані

**Вибір кількості компонентів:**

- **Explained variance:** зберегти 90-95% variance
- **Scree plot:** шукати "лікоть"
- **Domain knowledge:** скільки компонентів має сенс

**Коли використовувати:**

- Багато features + корельовані + візуалізація = PCA ✓
- Нелінійні структури → t-SNE, UMAP ✓
- Supervised + labels важливі → LDA ✓

**Найважливіше:**

- **ЗАВЖДИ робити scaling** (StandardScaler)
- Перевіряти explained variance (≥ 80-95%)
- Візуалізувати scree plot
- FIT тільки на train (уникати data leakage)
- Інтерпретувати компоненти (loadings)
- Видаляти outliers перед PCA

---

#ml #unsupervised-learning #dimensionality-reduction #pca #principal-component-analysis #feature-extraction #visualization #linear-transformation
