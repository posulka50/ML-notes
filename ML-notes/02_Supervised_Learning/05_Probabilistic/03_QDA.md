# QDA (Quadratic Discriminant Analysis)

## Що це?

**Quadratic Discriminant Analysis (QDA)** — це розширення LDA, яке дозволяє кожному класу мати **власну covariance matrix**, що призводить до **квадратичних decision boundaries**.

**Головна ідея:** так само як LDA, але без припущення про shared covariance → більше flexibility, але більше параметрів.

## Навіщо потрібен?

- 🎯 **Quadratic boundaries** — більш гнучкі ніж LDA
- 📊 **Different covariances** — кожен клас має свою форму
- ⚡ **Probabilistic** — дає ймовірності
- 💡 **Flexibility** — менше обмежень ніж LDA
- 🔧 **Gaussian assumption** — працює якщо дані Gaussian

## Коли використовувати?

**Потрібно:**
- Класи мають **різні covariances**
- **Quadratic boundaries** потрібні
- Класи **Gaussian розподілені**
- Достатньо даних (більше ніж для LDA)

**Не потрібно:**
- **Малі датасети** → LDA краще (менше параметрів)
- Shared covariance працює → LDA простіше
- **Нелінійні non-quadratic** → Kernel methods, Tree-based
- **Дуже високорозмірні** → LDA, regularization methods

---

## Математика QDA

### Різниця з LDA

**LDA:**
- Shared covariance: $\Sigma$ (однакова для всіх)
- **Лінійні** decision boundaries

**QDA:**
- Per-class covariance: $\Sigma_k$ (своя для кожного класу)
- **Квадратичні** decision boundaries

### Discriminant Function

**Для класу $k$:**

$$\delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log(\pi_k)$$

**Класифікація:**

$$\hat{y} = \arg\max_k \delta_k(x)$$

### Порівняння формул

**LDA (лінійна):**
$$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)$$

**QDA (квадратична):**
$$\delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log(\pi_k)$$

**Різниця:** $\Sigma_k$ замість $\Sigma$ + додатковий term $\log|\Sigma_k|$.

---

## Візуалізація

### LDA vs QDA

\`\`\`
LDA (linear boundary):        QDA (quadratic boundary):

    Class A (•)                    Class A (•)
      •  •  •                        •  •  •
      •  •  •                        •  •  •
    ──────────                      ╱──────╲
      ×  ×  ×                      ×  ×  ×
      ×  ×  ×                      ×  ×  ×
    Class B (×)                    Class B (×)

Пряма лінія                    Крива (еліптична)
\`\`\`

---

## Код (Python + scikit-learn)

### Базовий приклад

\`\`\`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Дані
iris = load_iris()
X = iris.data
y = iris.target

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# QDA
qda = QuadraticDiscriminantAnalysis()

# Навчання
qda.fit(X_train, y_train)

# Передбачення
y_pred = qda.predict(X_test)
y_pred_proba = qda.predict_proba(X_test)

# Оцінка
print("=== Quadratic Discriminant Analysis ===")
print(f"Train Accuracy: {qda.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Параметри
print("\n=== Model Parameters ===")
print(f"Priors: {qda.priors_}")
print(f"Means shape: {qda.means_.shape}")
print(f"Number of covariance matrices: {len(qda.covariance_)}")
\`\`\`

### LDA vs QDA порівняння

\`\`\`python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Навчання обох моделей
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train, y_train)
qda.fit(X_train, y_train)

print("="*60)
print("LDA vs QDA")
print("="*60)
print(f"LDA Train: {lda.score(X_train, y_train):.4f}  Test: {lda.score(X_test, y_test):.4f}")
print(f"QDA Train: {qda.score(X_train, y_train):.4f}  Test: {qda.score(X_test, y_test):.4f}")
\`\`\`

---

## Коли LDA, коли QDA?

### Decision Tree

\`\`\`
            Класи Gaussian розподілені?
                    /           \
                 Ні             Так
                  |               |
          Інші методи      Covariances однакові?
                            /              \
                         Так                Ні
                          |                  |
                        LDA              QDA (якщо достатньо даних)
                                             |
                                      Малий датасет?
                                        /        \
                                     Так          Ні
                                      |            |
                                    LDA          QDA
\`\`\`

### Практичні рекомендації

**Використовуй LDA коли:**
- ✅ Малі датасети
- ✅ Covariances схожі
- ✅ Потрібна простота
- ✅ Regularization важлива

**Використовуй QDA коли:**
- ✅ Достатньо даних
- ✅ Covariances явно різні
- ✅ Потрібні quadratic boundaries
- ✅ LDA underfitting

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Flexibility** | Різні covariances |
| **Quadratic boundaries** | Більше гнучкості ніж LDA |
| **Probabilistic** | Дає ймовірності |
| **Швидкість** | Швидший за SVM, RF |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Більше параметрів** | Потрібно більше даних |
| **Overfitting** | Якщо мало даних |
| **Gaussian assumption** | Порушення → погано |
| **High-dimensional** | Багато параметрів |

---

## Кількість параметрів

### LDA

**Параметрів:** $K \cdot d + \frac{d(d+1)}{2}$

- $K$ means ($K \cdot d$)
- 1 shared covariance ($\frac{d(d+1)}{2}$)

### QDA

**Параметрів:** $K \cdot d + K \cdot \frac{d(d+1)}{2}$

- $K$ means ($K \cdot d$)
- $K$ covariances ($K \cdot \frac{d(d+1)}{2}$)

**Приклад:** $K=3$, $d=10$

- LDA: $3 \cdot 10 + 55 = 85$ параметрів
- QDA: $3 \cdot 10 + 3 \cdot 55 = 195$ параметрів

**QDA потребує більше даних!**

---

## Практичні поради 💡

1. **Почни з LDA** — простіша baseline
2. **Порівняй LDA vs QDA** — через CV
3. **Достатньо даних?** — QDA потребує більше
4. **Візуалізуй covariances** — чи вони різні?
5. **High-dimensional** — LDA краще (менше параметрів)
6. **Regularization** — якщо QDA overfitting
7. **Gaussian assumption** — перевір візуально

---

## Ключові висновки

> QDA дозволяє різні covariances для класів → квадратичні decision boundaries, але потребує більше даних.

**LDA vs QDA:**
- **LDA:** shared Σ → linear boundaries
- **QDA:** per-class Σ_k → quadratic boundaries

**Trade-off:**
- QDA більш гнучкий, але більше параметрів
- LDA простіший, краще для малих даних

---

#ml #qda #discriminant-analysis #quadratic #probabilistic
