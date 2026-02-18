# LDA (Linear Discriminant Analysis)

## Що це?

**Linear Discriminant Analysis (LDA)** — це алгоритм класифікації, який знаходить **лінійну комбінацію ознак**, що найкраще розділяє класи, максимізуючи відстань між класами та мінімізуючи variance всередині класів.

**Головна ідея:** проектувати дані на простір меншої розмірності таким чином, щоб класи були максимально розділені.

## Навіщо потрібен?

- 📊 **Dimensionality reduction** + класифікація
- 🎯 **Probabilistic** — дає ймовірності класів
- ⚡ **Швидкість** — дуже швидкий
- 💡 **Інтерпретованість** — лінійні decision boundaries
- 🔧 **Assumes Gaussian** — працює добре якщо припущення виконуються
- 📈 **Feature extraction** — LDA як preprocessing

## Коли використовувати?

**Потрібно:**
- Класи **Gaussian розподілені**
- **Shared covariance** між класами (однакова для всіх класів)
- Потрібна **dimensionality reduction**
- **Probabilistic predictions**
- Швидкість важлива
- Малі/середні датасети

**Не потрібно:**
- **Нелінійні boundaries** → QDA, Kernel methods
- Різні covariances між класами → QDA
- Потрібна **максимальна точність** → Tree-based, SVM, Neural Networks
- **Multimodal distributions** → GMM, Tree-based

---

## Математика LDA

### Припущення

1. Ознаки розподілені **нормально** для кожного класу
2. Класи мають **однакову covariance matrix** (shared Σ)
3. Prior probabilities відомі або оцінюються з даних

### Notation

- $\mu_k$ — mean vector для класу $k$
- $\Sigma$ — shared covariance matrix (однакова для всіх класів)
- $\pi_k$ — prior probability класу $k$

### Discriminant Function

**Для класу $k$:**

$$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)$$

**Класифікація:**

$$\hat{y} = \arg\max_k \delta_k(x)$$

Обираємо клас з найвищим discriminant function value.

### Decision Boundary (2 класи)

Decision boundary між класами 1 та 2:

$$\delta_1(x) = \delta_2(x)$$

Це дає **лінійну** decision boundary (гіперплощину).

---

## Візуальна інтуїція

### 2D приклад

\`\`\`
Original 2D space:

    Feature 2
        |
        |    Class A (•)
        |  •  •  •
        |    •  •
        |---------- Feature 1
        |  ×  ×
        |×  ×  ×
        |  Class B (×)

LDA проекція на 1D:

    ─•─•─•────×─×─×─→ LDA axis
    
Класи максимально розділені!
\`\`\`

### Що робить LDA?

1. **Maximize between-class variance** — віддаляє центри класів
2. **Minimize within-class variance** — робить класи компактними
3. **Лінійна проекція** — знаходить оптимальну вісь

---

## LDA як Dimensionality Reduction

### Формула

**Знаходимо напрямок $w$, що максимізує:**

$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$

де:
- $S_B$ — between-class scatter matrix
- $S_W$ — within-class scatter matrix

### Between-class Scatter

$$S_B = \sum_{k=1}^{K} n_k (\mu_k - \mu)(\mu_k - \mu)^T$$

де $\mu$ — загальний mean.

### Within-class Scatter

$$S_W = \sum_{k=1}^{K} \sum_{x \in C_k} (x - \mu_k)(x - \mu_k)^T$$

### Optimal Direction

**Розв'язуємо eigenvalue problem:**

$$S_W^{-1} S_B w = \lambda w$$

**LDA проекція:** top $K-1$ eigenvectors.

---

## Код (Python + scikit-learn)

### Базовий приклад

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Дані
iris = load_iris()
X = iris.data
y = iris.target

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. LDA
lda = LinearDiscriminantAnalysis()

# 3. Навчання
lda.fit(X_train, y_train)

# 4. Передбачення
y_pred = lda.predict(X_test)
y_pred_proba = lda.predict_proba(X_test)

# 5. Оцінка
print("=== Linear Discriminant Analysis ===")
print(f"Train Accuracy: {lda.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Параметри
print("\n=== Model Parameters ===")
print(f"Priors: {lda.priors_}")
print(f"Means shape: {lda.means_.shape}")
print(f"Covariance shape: {lda.covariance_.shape}")
\`\`\`

### LDA для Dimensionality Reduction

\`\`\`python
# LDA з n_components (dimensionality reduction)
lda_2d = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda_2d.fit_transform(X, y)

print(f"\nOriginal shape: {X.shape}")
print(f"LDA shape: {X_lda.shape}")
print(f"Explained variance ratio: {lda_2d.explained_variance_ratio_}")

# Візуалізація
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1],
               alpha=0.7, color=color, label=iris.target_names[i])
plt.xlabel(f'LD1 ({lda_2d.explained_variance_ratio_[0]:.2%})', fontsize=12)
plt.ylabel(f'LD2 ({lda_2d.explained_variance_ratio_[1]:.2%})', fontsize=12)
plt.title('LDA Projection to 2D', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
\`\`\`

---

## Порівняння з PCA

### LDA vs PCA

\`\`\`python
from sklearn.decomposition import PCA

# PCA (unsupervised)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# LDA (supervised)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA
for i, color in enumerate(colors):
    axes[0].scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                   alpha=0.7, color=color, label=iris.target_names[i])
axes[0].set_title('PCA (Unsupervised)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# LDA
for i, color in enumerate(colors):
    axes[1].scatter(X_lda[y == i, 0], X_lda[y == i, 1],
                   alpha=0.7, color=color, label=iris.target_names[i])
axes[1].set_title('LDA (Supervised)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('LD1')
axes[1].set_ylabel('LD2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
\`\`\`

**Різниця:**
- **PCA:** максимізує variance (unsupervised)
- **LDA:** максимізує class separation (supervised)

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Швидкість** | Дуже швидкий |
| **Probabilistic** | Дає ймовірності |
| **Dimensionality reduction** | Preprocessing для інших моделей |
| **Інтерпретованість** | Лінійні boundaries |
| **Малі дані** | Працює добре |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Припущення** | Gaussian + shared covariance |
| **Лінійні boundaries** | Не працює для нелінійних |
| **Різні covariances** | Порушення припущення → QDA |
| **Multimodal** | Не працює добре |

---

## Ключові висновки

> LDA знаходить лінійну проекцію, що максимізує між-класову відстань та мінімізує внутрішньо-класову variance.

**Коли використовувати:**
- Gaussian дані + shared covariance → LDA ✓
- Різні covariances → QDA ✓
- Нелінійні boundaries → Kernel methods ✓

---

#ml #lda #discriminant-analysis #dimensionality-reduction #probabilistic
