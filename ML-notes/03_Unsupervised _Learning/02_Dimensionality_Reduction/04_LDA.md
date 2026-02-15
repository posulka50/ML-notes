# LDA (Linear Discriminant Analysis)

## Що це?

**LDA (Linear Discriminant Analysis)** — це **supervised** алгоритм dimensionality reduction, який знаходить лінійну комбінацію features, що найкраще **розділяє класи**. На відміну від PCA, LDA використовує інформацію про labels.

**Головна ідея:** знайти напрямки (discriminants), вздовж яких різні класи максимально роз'єднані, а точки всередині класу максимально згруповані.

## Навіщо потрібен?

- 🎯 **Supervised reduction** — використовує labels для кращого розділення
- 📊 **Classification preprocessing** — features для класифікатора
- 🔍 **Feature extraction** — знайти найважливіші напрямки для розділення
- 🎨 **Візуалізація** — 2D/3D проекція з розділенням класів
- ⚖️ **Максимальна сепарація** — between-class variance / within-class variance
- 📈 **Інтерпретованість** — компоненти мають чітке значення

## Коли використовувати?

**Потрібно:**
- **Є labels** (supervised task) — ключова вимога!
- **Classification** — preprocessing перед класифікатором
- **Класи збалансовані** — приблизно однакова кількість зразків
- **Числові features** (неперервні)
- Потрібна **розділяюча здатність** між класами
- **< 20 класів** (обмеження: n_components ≤ n_classes - 1)

**Не потрібно:**
- **Unsupervised** (немає labels) → PCA, t-SNE, UMAP
- **Дуже небалансовані класи** → weighted LDA
- **Нелінійне розділення** → Kernel LDA або Neural Networks
- **Багато класів** (>100) → інші методи
- **Categorical features** → preprocessing спочатку

---

## Як працює LDA?

### Інтуїція

**Проблема PCA:** Максимізує variance, але не враховує класи.

**LDA:** Знаходить проекцію, де класи розділені.

### Математична мета

**Максимізувати:**

$$J(w) = \frac{\text{between-class variance}}{\text{within-class variance}} = \frac{w^T S_B w}{w^T S_W w}$$

де:
- $S_B$ — between-class scatter matrix (відмінності між класами)
- $S_W$ — within-class scatter matrix (розкид всередині класів)

**Інтуїція:**
- **Між класами** — хочемо ВЕЛИКІ відстані
- **Всередині класу** — хочемо МАЛЕНЬКИЙ розкид

---

## Математика

### 1. Within-class scatter matrix

**Розкид всередині кожного класу:**

$$S_W = \sum_{c=1}^{C} \sum_{x \in \text{class } c} (x - \mu_c)(x - \mu_c)^T$$

де:
- $C$ — кількість класів
- $\mu_c$ — середнє класу $c$

**Інтуїція:** Наскільки розкидані точки всередині кожного класу.

### 2. Between-class scatter matrix

**Відмінності між класами:**

$$S_B = \sum_{c=1}^{C} n_c (\mu_c - \mu)(\mu_c - \mu)^T$$

де:
- $n_c$ — кількість зразків у класі $c$
- $\mu$ — загальне середнє всіх даних

**Інтуїція:** Наскільки далеко центри класів від загального центру.

### 3. Розв'язання

**Знайти власні вектори:**

$$S_W^{-1} S_B w = \lambda w$$

**Процес:**
1. Обчислити $S_W$ та $S_B$
2. Розв'язати $S_W^{-1} S_B w = \lambda w$
3. Відсортувати власні вектори за власними значеннями
4. Вибрати top k власних векторів

**Кількість компонентів:** максимум $C - 1$ (де $C$ — кількість класів)

---

## Простий приклад: 2 класи в 2D

### Дані

```
Клас A (червоні):    Клас B (сині):
x = [1, 2]           x = [5, 6]
x = [2, 3]           x = [6, 7]
x = [1.5, 2.5]       x = [5.5, 6.5]
```

### Візуалізація

```
    y
  8 |
  7 |        🔵 🔵
  6 |      🔵
  5 |
  4 |
  3 |  🔴
  2 |🔴  🔴
  1 |
  0 |____________ x
    0  2  4  6  8
```

### LDA проекція

**Крок 1:** Обчислити середні класів
```
μ_A = [1.5, 2.5]
μ_B = [5.5, 6.5]
```

**Крок 2:** Within-class scatter
```
S_W = scatter_A + scatter_B
```

**Крок 3:** Between-class scatter
```
S_B = n_A * (μ_A - μ_overall)(μ_A - μ_overall)^T 
    + n_B * (μ_B - μ_overall)(μ_B - μ_overall)^T
```

**Крок 4:** Знайти напрямок розділення
```
LD1 напрямок: [1, 1] (diagonal)

Проекція на LD1:
    
    LD1
     |
  🔴🔴|    🔵🔵
     |
     
Ідеальне розділення!
```

---

## Складний приклад: Iris Dataset

### Задача

Iris: 150 квітів, 4 features, 3 види (setosa, versicolor, virginica).

**Мета:** Зменшити до 2D для візуалізації з максимальним розділенням класів.

### Результат

**LDA знаходить 2 компоненти** (3 класи - 1 = 2):
- **LD1:** Розділяє setosa від (versicolor + virginica)
- **LD2:** Розділяє versicolor від virginica

```
    LD2
     |
   2 |  versicolor
     |    ●●●
   0 |●●●     ●●●  
     |setosa  virginica
  -2 |
     |_____________ LD1
    -8  -4  0  4  8
```

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Завантажити дані
iris = load_iris()
X = iris.data
y = iris.target

print(f"Original shape: {X.shape}")  # (150, 4)
print(f"Classes: {np.unique(y)}")    # [0, 1, 2]

# 2. Scaling (рекомендовано)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. LDA
lda = LDA(n_components=2)  # max = n_classes - 1 = 2
X_lda = lda.fit_transform(X_scaled, y)

print(f"LDA shape: {X_lda.shape}")  # (150, 2)

# 4. Explained variance ratio
print("\n=== Explained Variance ===")
print(f"LD1: {lda.explained_variance_ratio_[0]:.4f}")
print(f"LD2: {lda.explained_variance_ratio_[1]:.4f}")
print(f"Total: {lda.explained_variance_ratio_.sum():.4f}")

# 5. Візуалізація
plt.figure(figsize=(10, 7))

colors = ['red', 'green', 'blue']
target_names = iris.target_names

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_lda[y == i, 0],
        X_lda[y == i, 1],
        alpha=0.8,
        color=color,
        label=target_name,
        s=50
    )

plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.title('LDA Projection of Iris Dataset', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Порівняння PCA vs LDA

```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (unsupervised)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# LDA (supervised)
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Візуалізація порівняння
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    axes[0].scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                   alpha=0.8, color=color, label=target_name, s=50)

axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
axes[0].set_title('PCA (Unsupervised)', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LDA
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    axes[1].scatter(X_lda[y == i, 0], X_lda[y == i, 1],
                   alpha=0.8, color=color, label=target_name, s=50)

axes[1].set_xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.1%})', fontsize=11)
axes[1].set_ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.1%})', fontsize=11)
axes[1].set_title('LDA (Supervised)', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Comparison ===")
print("PCA: Maximizes variance (ignores classes)")
print("LDA: Maximizes class separation (uses labels)")
print("\nLDA shows better class separation!")
```

### LDA для класифікації

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# LDA як classifier (вбудована класифікація!)
lda_classifier = LDA()
lda_classifier.fit(X_train, y_train)

# Predictions
y_pred = lda_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"LDA Classifier Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# LDA може використовуватись і як reducer, і як classifier!
```

### Повний приклад: Wine Dataset

```python
import pandas as pd
from sklearn.datasets import load_wine

# Завантажити Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

print(f"Original shape: {X.shape}")  # (178, 13)
print(f"Classes: {np.unique(y)}")    # [0, 1, 2] - 3 types of wine

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# LDA transformation (dimensionality reduction)
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

print(f"\n=== LDA Transformation ===")
print(f"Original: {X_train.shape[1]} features")
print(f"LDA: {X_train_lda.shape[1]} components")
print(f"Explained variance: {lda.explained_variance_ratio_.sum():.2%}")

# Візуалізація
plt.figure(figsize=(12, 5))

# Train set
plt.subplot(1, 2, 1)
for i in range(3):
    mask = y_train == i
    plt.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1],
               label=f'Class {i}', s=50, alpha=0.7)
plt.xlabel('LD1', fontsize=11)
plt.ylabel('LD2', fontsize=11)
plt.title('Train Set (LDA)', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Test set
plt.subplot(1, 2, 2)
for i in range(3):
    mask = y_test == i
    plt.scatter(X_test_lda[mask, 0], X_test_lda[mask, 1],
               label=f'Class {i}', s=50, alpha=0.7)
plt.xlabel('LD1', fontsize=11)
plt.ylabel('LD2', fontsize=11)
plt.title('Test Set (LDA)', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Classification on LDA features
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# На оригінальних features
knn.fit(X_train, y_train)
acc_original = knn.score(X_test, y_test)

# На LDA features
knn.fit(X_train_lda, y_train)
acc_lda = knn.score(X_test_lda, y_test)

print(f"\n=== KNN Classification ===")
print(f"Original features (13D): {acc_original:.4f}")
print(f"LDA features (2D): {acc_lda:.4f}")
print(f"Dimension reduction: {13/2:.1f}x")
```

### Coefficients (Feature Importance)

```python
# LDA coefficients показують важливість features
lda = LDA(n_components=2)
lda.fit(X_scaled, y)

# Coefficients для кожного компонента
coefficients = lda.coef_

print("\n=== LDA Coefficients ===")
for i, coef in enumerate(coefficients):
    print(f"\nLD{i+1} coefficients:")
    
    # Абсолютні значення для важливості
    abs_coef = np.abs(coef)
    
    # Топ-5 features
    top_indices = np.argsort(abs_coef)[-5:][::-1]
    
    for idx in top_indices:
        print(f"  {wine.feature_names[idx]}: {coef[idx]:.4f}")
```

### Decision Boundaries

```python
from matplotlib.colors import ListedColormap

# Використаємо тільки 2 features для візуалізації
X_2d = X_scaled[:, [0, 1]]  # Alcohol та Malic acid

# LDA на 2D
lda = LDA()
lda.fit(X_2d, y)

# Create mesh
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict на mesh
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red', 'green', 'blue']))

# Scatter points
colors = ['red', 'green', 'blue']
for i in range(3):
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colors[i], label=f'Class {i}',
               s=50, alpha=0.7, edgecolors='black')

plt.xlabel(wine.feature_names[0], fontsize=11)
plt.ylabel(wine.feature_names[1], fontsize=11)
plt.title('LDA Decision Boundaries', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Варіації LDA

### 1. Quadratic Discriminant Analysis (QDA)

**Що це:** Дозволяє різні коваріаційні матриці для кожного класу (квадратичні границі).

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# QDA (більш гнучкі границі)
qda = QDA()
qda.fit(X_train, y_train)
qda_score = qda.score(X_test, y_test)

# LDA (лінійні границі)
lda = LDA()
lda.fit(X_train, y_train)
lda_score = lda.score(X_test, y_test)

print(f"LDA accuracy: {lda_score:.4f}")
print(f"QDA accuracy: {qda_score:.4f}")

# QDA може бути кращим якщо класи мають різні розподіли
```

**Коли використовувати:**
- LDA: Класи мають подібні covariances
- QDA: Класи мають різні covariances

### 2. Shrinkage LDA

**Що це:** Regularization для малих datasets (коли n_samples < n_features).

```python
# LDA з shrinkage
lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
lda_shrinkage.fit(X_train, y_train)

print(f"Shrinkage parameter: {lda_shrinkage.covariance_.shape}")
```

**Коли використовувати:**
- Малий dataset (n < p)
- High-dimensional data
- Regularization потрібна

---

## Припущення LDA

### 1. Нормальність

**Кожен клас має Gaussian розподіл.**

```python
# Перевірка нормальності
from scipy.stats import shapiro

feature_idx = 0  # Перша feature
class_idx = 0    # Перший клас

data = X_scaled[y == class_idx, feature_idx]

stat, p_value = shapiro(data)

if p_value > 0.05:
    print(f"Feature {feature_idx} in class {class_idx} is normally distributed")
else:
    print(f"Not normally distributed (p={p_value:.4f})")
```

### 2. Однакова коваріація

**Всі класи мають однакову коваріаційну матрицю.**

```python
# Візуально перевірити
for i in range(3):
    class_data = X_scaled[y == i]
    cov = np.cov(class_data.T)
    print(f"\nClass {i} covariance (first 2x2):")
    print(cov[:2, :2])

# Якщо дуже різні → розглянути QDA
```

### 3. Лінійна розділюваність

**Класи можна розділити лінійними границями.**

```python
# Якщо ні → Kernel LDA або Neural Networks
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Supervised** | Використовує labels для кращого розділення |
| **Інтерпретованість** | Coefficients показують важливість features |
| **Classifier + Reducer** | Може і класифікувати, і зменшувати розмірність |
| **Швидкість** | Дуже швидкий (closed-form solution) |
| **Детермінізм** | Однакові результати завжди |
| **Максимальна сепарація** | Оптимізує розділення класів |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Потребує labels** | Тільки supervised tasks |
| **Лінійність** | Тільки лінійні границі |
| **Припущення** | Gaussian, однакова covariance |
| **Обмеження компонентів** | Максимум n_classes - 1 |
| **Чутливість до outliers** | Впливають на оцінку covariance |
| **Малі класи** | Погано якщо клас має мало зразків |

---

## Порівняння з іншими методами

| Критерій | PCA | LDA | t-SNE | UMAP |
|----------|-----|-----|-------|------|
| **Supervised** | ❌ | ✅ | ❌ | ⚠️ (опційно) |
| **Лінійний** | ✅ | ✅ | ❌ | ❌ |
| **Швидкість** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| **Класифікація** | ❌ | ✅ | ❌ | ❌ |
| **Max компонентів** | min(n,p) | n_classes-1 | будь-який | будь-який |
| **Інтерпретованість** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |

### LDA vs PCA (детально)

**PCA:**
- ✅ Unsupervised (не потребує labels)
- ✅ Максимізує variance
- ✅ Будь-яка кількість компонентів
- ❌ Ігнорує класи

**LDA:**
- ✅ Supervised (використовує labels)
- ✅ Максимізує розділення класів
- ✅ Можна використати як classifier
- ❌ Максимум n_classes - 1 компонентів

**Коли що:**
- **Classification task** → LDA ✓
- **Exploratory analysis без labels** → PCA ✓
- **Багато features, мало класів** → PCA → LDA ✓

---

## Коли використовувати LDA

### Ідеально підходить ✓

- **Classification task** з labels
- **Preprocessing** перед класифікатором
- **Візуалізація** з розділенням класів
- **Feature extraction** для розділення
- **Малий датасет** (швидше за neural networks)
- **Інтерпретованість** важлива
- **2-10 класів** (оптимально)

### Краще використати інше ✗

- **Unsupervised** (немає labels) → PCA, t-SNE, UMAP
- **Нелінійне розділення** → Kernel LDA, Neural Networks
- **Багато класів** (>100) → інші методи
- **Класи не Gaussian** → інші методи
- **Дуже небалансовані класи** → weighted LDA або інше

---

## Практичні поради 💡

### 1. Завжди використовуй зі scaling

```python
# ✅ ПРАВИЛЬНО
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lda = LDA()
X_lda = lda.fit_transform(X_scaled, y)
```

### 2. Перевіряй баланс класів

```python
# Перевірка
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples")

# Якщо дуже небалансовано → розглянути resampling
```

### 3. PCA перед LDA для high-dimensional

```python
# Якщо n_features >> n_samples
from sklearn.pipeline import Pipeline

# Pipeline: PCA → LDA
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),  # Зменшити спочатку
    ('lda', LDA(n_components=2))    # Потім LDA
])

X_reduced = pipeline.fit_transform(X, y)
```

### 4. Використовуй як classifier

```python
# LDA може класифікувати!
lda = LDA()
lda.fit(X_train, y_train)

# Predictions
y_pred = lda.predict(X_test)

# Probabilities
y_proba = lda.predict_proba(X_test)
print(f"Probabilities for first sample: {y_proba[0]}")
```

### 5. Перевіряй explained variance

```python
lda = LDA()
lda.fit(X_scaled, y)

print("Explained variance ratio:")
for i, ratio in enumerate(lda.explained_variance_ratio_):
    print(f"  LD{i+1}: {ratio:.2%}")

cumsum = np.cumsum(lda.explained_variance_ratio_)
print(f"\nCumulative: {cumsum[-1]:.2%}")
```

### 6. QDA якщо класи мають різні covariances

```python
# Порівняй LDA та QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

lda = LDA()
qda = QDA()

lda_score = lda.fit(X_train, y_train).score(X_test, y_test)
qda_score = qda.fit(X_train, y_train).score(X_test, y_test)

if qda_score > lda_score + 0.05:  # Значно краще
    print("Use QDA (classes have different covariances)")
else:
    print("Use LDA (simpler, similar performance)")
```

### 7. Cross-validation для оцінки

```python
from sklearn.model_selection import cross_val_score

lda = LDA()
scores = cross_val_score(lda, X_scaled, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 8. Feature importance через coefficients

```python
lda = LDA(n_components=1)
lda.fit(X_scaled, y)

# Absolute coefficients
importance = np.abs(lda.coef_[0])

# Sort
sorted_idx = np.argsort(importance)[::-1]

print("Top 5 most important features:")
for i in sorted_idx[:5]:
    print(f"  {feature_names[i]}: {importance[i]:.4f}")
```

### 9. Візуалізуй decision boundaries

```python
# Для 2D даних
from matplotlib.colors import ListedColormap

# Train LDA
lda = LDA()
lda.fit(X_2d, y)

# Create mesh
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.show()
```

### 10. Shrinkage для малих datasets

```python
# Якщо n_samples невелике
lda = LDA(solver='lsqr', shrinkage='auto')
lda.fit(X_train, y_train)

# Або задати shrinkage вручну
lda = LDA(solver='lsqr', shrinkage=0.5)
```

---

## Реальні застосування

### 1. Face Recognition

**Задача:** Розпізнавання облич (Fisherfaces method).

**Підхід:**
```python
# 1. Кожна особа = клас
# 2. LDA знаходить напрямки, що максимально розділяють осіб
# 3. Проектувати нові обличчя на ці напрямки

# Features: pixel values або PCA features
faces_lda = lda.fit_transform(face_features, person_ids)

# Класифікація: nearest neighbor в LDA space
```

### 2. Medical Diagnosis

**Задача:** Діагностика захворювань за симптомами.

**Підхід:**
```python
# Classes: healthy, disease A, disease B
# Features: test results, symptoms

lda = LDA()
lda.fit(patient_features, diagnoses)

# Нові пацієнти
diagnosis_pred = lda.predict(new_patient_features)
probabilities = lda.predict_proba(new_patient_features)
```

### 3. Credit Scoring

**Задача:** Оцінка кредитоспроможності.

**Підхід:**
```python
# Classes: good credit, bad credit
# Features: income, debt, history, etc.

lda = LDA(n_components=1)  # 1 компонент для binary
credit_score = lda.fit_transform(applicant_features, credit_status)

# Використати як credit score
threshold = find_optimal_threshold(credit_score, credit_status)
```

### 4. Document Classification

**Задача:** Класифікація текстів за темами.

**Підхід:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(documents)

# PCA → LDA (для зменшення розмірності)
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_tfidf.toarray())

lda = LDA(n_components=5)  # 6 topics → 5 LDs
X_lda = lda.fit_transform(X_pca, topics)
```

### 5. Biometric Authentication

**Задача:** Аутентифікація за біометрією (fingerprints, voice).

**Підхід:**
```python
# Features: биометричні ознаки
# Classes: різні користувачі

lda = LDA()
lda.fit(biometric_features, user_ids)

# Верифікація: чи новий зразок від claimed user?
def verify(new_sample, claimed_user_id):
    proba = lda.predict_proba([new_sample])[0]
    user_proba = proba[claimed_user_id]
    
    return user_proba > threshold
```

---

## Поширені помилки ❌

### 1. Використовувати без labels

```python
# ❌ LDA потребує labels!
lda = LDA()
lda.fit(X)  # TypeError!

# ✅ Передай y
lda.fit(X, y)
```

### 2. Не робити scaling

```python
# ❌ Без scaling (features в різних масштабах)
lda = LDA()
lda.fit(X, y)

# ✅ Зі scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lda.fit(X_scaled, y)
```

### 3. Більше компонентів ніж класів - 1

```python
# ❌ Занадто багато компонентів
lda = LDA(n_components=5)  # але тільки 3 класи!
# ValueError: n_components > n_classes - 1

# ✅ Максимум n_classes - 1
n_classes = len(np.unique(y))
lda = LDA(n_components=min(2, n_classes - 1))
```

### 4. Дуже небалансовані класи

```python
# ❌ Клас 1: 1000 зразків, Клас 2: 10 зразків
# LDA буде bias до великого класу

# ✅ Resampling або weighted LDA
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

lda.fit(X_resampled, y_resampled)
```

### 5. Очікувати нелінійне розділення

```python
# ❌ LDA тільки лінійний!
# Якщо класи нелінійно розділені → погані результати

# ✅ Kernel LDA або Neural Networks
# Або feature engineering спочатку
```

### 6. Ігнорувати припущення

```python
# ❌ Не перевіряти Gaussian assumption
# Якщо дані дуже не-Gaussian → LDA може погано працювати

# ✅ Перевірити візуально або статистично
import seaborn as sns

for cls in np.unique(y):
    class_data = X_scaled[y == cls, 0]  # Перша feature
    sns.histplot(class_data, kde=True)
plt.show()
```

### 7. Fit на всіх даних (train + test)

```python
# ❌ DATA LEAKAGE
X_all = np.vstack([X_train, X_test])
y_all = np.hstack([y_train, y_test])
lda.fit(X_all, y_all)  # ← Leakage!

# ✅ Fit тільки на train
lda.fit(X_train, y_train)
X_test_lda = lda.transform(X_test)
```

---

## Пов'язані теми

- [[01_PCA]] — unsupervised альтернатива
- [[02_t-SNE]] — нелінійна візуалізація
- [[03_UMAP]] — швидка нелінійна альтернатива
- [[Linear_Regression]] — регресія замість класифікації
- [[Logistic_Regression]] — інший supervised classifier
- [[QDA]] — квадратична версія LDA

## Ресурси

- [Scikit-learn: LDA](https://scikit-learn.org/stable/modules/lda_qda.html)
- [Original Paper: Fisher (1936)](https://digital.library.adelaide.edu.au/dspace/bitstream/2440/15227/1/138.pdf)
- [StatQuest: LDA](https://www.youtube.com/watch?v=azXCzI57Yfc)
- [Comparison: PCA vs LDA](https://sebastianraschka.com/Articles/2014_python_lda.html)

---

## Ключові висновки

> LDA — це supervised алгоритм dimensionality reduction, який знаходит напрямки, що максимально розділяють класи, максимізуючи відношення between-class variance до within-class variance.

**Основні принципи:**
- **Supervised:** використовує labels (ключова відмінність від PCA)
- **Лінійний:** знаходить лінійні комбінації features
- **Оптимізація:** max(between-class var / within-class var)
- **Classifier + Reducer:** може і класифікувати, і зменшувати розмірність

**Математика:**
- **Максимізувати:** $J(w) = \frac{w^T S_B w}{w^T S_W w}$
- **Розв'язок:** Власні вектори $S_W^{-1} S_B$
- **Обмеження:** Максимум n_classes - 1 компонентів

**Припущення:**
- Gaussian розподіл кожного класу
- Однакова коваріаційна матриця
- Лінійна розділюваність

**Переваги над PCA:**
- ✅ Краще розділення класів
- ✅ Supervised (uses labels)
- ✅ Можна класифікувати
- ✅ Інтерпретованість (coefficients)

**Коли використовувати:**
- Classification + є labels = LDA ✓
- Unsupervised → PCA ✓
- Нелінійне → Kernel LDA або NN ✓
- Багато класів (>100) → інше ✓

**Найважливіше:**
- **Потрібні labels** — supervised метод
- **Scaling критичний** — завжди StandardScaler
- **Max n_classes - 1** компонентів
- **PCA → LDA** pipeline для high-dimensional
- **Перевіряй баланс** класів
- **QDA** якщо різні covariances

---

#ml #supervised-learning #dimensionality-reduction #lda #linear-discriminant-analysis #classification #feature-extraction #supervised
