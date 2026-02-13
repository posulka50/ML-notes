# Similarity Measures (Метрики схожості)

## Що це?

**Similarity Measures** — це функції, які **вимірюють схожість** між двома об'єктами. Чим більше значення, тим більш схожі об'єкти. Це **доповнення** до distance metrics.

**Головна ідея:** 
```
Distance ↔ Similarity

Distance велика → об'єкти різні
Similarity велика → об'єкти схожі

Часто: Similarity = 1 / (1 + Distance)
```

## Навіщо потрібно?

- 🔍 **Recommender Systems** — знайти схожих користувачів/товари
- 📄 **Document Similarity** — порівняння текстів
- 🖼️ **Image Retrieval** — пошук схожих зображень
- 🧬 **Bioinformatics** — порівняння ДНК/білків
- 📊 **Collaborative Filtering** — рекомендації
- 🎯 **Clustering** — групування схожих об'єктів

## Коли використовувати?

**Потрібно:**
- Recommender systems
- Пошук схожих об'єктів
- Collaborative filtering
- Feature matching

**Відмінність від Distance:**
- Distance → "наскільки різні?"
- Similarity → "наскільки схожі?"

---

## Класифікація метрик

```
Similarity Measures
│
├── Vector-based
│   ├── Cosine Similarity
│   ├── Dot Product
│   └── Pearson Correlation
│
├── Set-based
│   ├── Jaccard Similarity
│   ├── Dice Coefficient
│   └── Overlap Coefficient
│
├── Probabilistic
│   ├── Kullback-Leibler Divergence
│   └── Jensen-Shannon Divergence
│
└── Other
    ├── Spearman Correlation
    ├── Kendall Tau
    └── Tanimoto Coefficient
```

---

# 1. Cosine Similarity

## Формула

$$\text{cosine\_similarity}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}}$$

Range: $[-1, 1]$
- 1 → ідентичний напрямок
- 0 → ортогональні (незалежні)
- -1 → протилежний напрямок

## Інтуїція

**Cosine Similarity** вимірює **кут між векторами**, ігноруючи їх довжину.

```
Приклад: Уподобання користувачів

User A: [5, 4, 0, 0, 3]  (фільми: Action, Comedy, Drama, Horror, Sci-Fi)
User B: [10, 8, 0, 0, 6]  (у 2 рази більше оцінок, але ті ж жанри!)

Euclidean distance: велика (різні magnitude)
Cosine similarity: 1.0 (той самий напрямок уподобань!)

Висновок: користувачі дуже схожі за смаками
```

## Код

```python
import numpy as np

def cosine_similarity(x, y):
    """Обчислити Cosine Similarity"""
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)

# Приклад: Рейтинги фільмів
user_A = np.array([5, 4, 0, 0, 3])
user_B = np.array([10, 8, 0, 0, 6])
user_C = np.array([0, 0, 5, 5, 0])

sim_AB = cosine_similarity(user_A, user_B)
sim_AC = cosine_similarity(user_A, user_C)

print(f"Similarity(A, B): {sim_AB:.4f}")  # ~1.0 (дуже схожі)
print(f"Similarity(A, C): {sim_AC:.4f}")  # 0.0 (різні жанри)

# Через sklearn
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

users = np.array([user_A, user_B, user_C])
similarity_matrix = sklearn_cosine(users)

print("\nSimilarity Matrix:")
print(similarity_matrix)
```

## Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Вектори документів (TF-IDF для слів)
doc1 = np.array([2, 3, 0, 1])  # "machine learning AI python"
doc2 = np.array([4, 6, 0, 2])  # Той самий контент, більше слів
doc3 = np.array([0, 1, 5, 0])  # Інший контент "data statistics R"

# Візуалізація (використаємо перші 2 виміри)
docs_2d = np.array([
    [doc1[0], doc1[1]],
    [doc2[0], doc2[1]],
    [doc3[0], doc3[1]]
])

# Обчислити схожості
sim_12 = cosine_similarity(doc1, doc2)
sim_13 = cosine_similarity(doc1, doc3)

# Візуалізація
plt.figure(figsize=(10, 8))

origin = [0, 0]

# Вектори
plt.quiver(*origin, docs_2d[0, 0], docs_2d[0, 1], angles='xy', scale_units='xy', 
          scale=1, color='blue', width=0.01, label=f'Doc1')
plt.quiver(*origin, docs_2d[1, 0], docs_2d[1, 1], angles='xy', scale_units='xy', 
          scale=1, color='green', width=0.01, label=f'Doc2')
plt.quiver(*origin, docs_2d[2, 0], docs_2d[2, 1], angles='xy', scale_units='xy', 
          scale=1, color='red', width=0.01, label=f'Doc3')

# Анотації
plt.text(docs_2d[0, 0]/2, docs_2d[0, 1]/2 + 0.5, 'Doc1\n(ML, AI)', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
plt.text(docs_2d[1, 0]/2, docs_2d[1, 1]/2 - 0.5, 'Doc2\n(Same topic)', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
plt.text(docs_2d[2, 0]/2 + 0.5, docs_2d[2, 1]/2, 'Doc3\n(Stats)', 
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.xlim(-1, 7)
plt.ylim(-1, 8)
plt.xlabel('Feature 1 (e.g., "machine")', fontsize=12)
plt.ylabel('Feature 2 (e.g., "learning")', fontsize=12)
plt.title(f'Cosine Similarity\nDoc1-Doc2: {sim_12:.3f} | Doc1-Doc3: {sim_13:.3f}', 
         fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

## Приклад: Document Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Документи
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a type of machine learning",
    "Natural language processing uses machine learning",
    "Statistics is important for data science"
]

# TF-IDF векторизація
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Document Similarity Matrix:")
print(similarity_matrix)

# Знайти найбільш схожі документи
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        sim = similarity_matrix[i, j]
        print(f"\nDoc{i} vs Doc{j}: {sim:.4f}")
        print(f"  Doc{i}: {documents[i][:50]}...")
        print(f"  Doc{j}: {documents[j][:50]}...")

# Візуалізація
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
           xticklabels=[f'Doc{i}' for i in range(len(documents))],
           yticklabels=[f'Doc{i}' for i in range(len(documents))],
           cbar_kws={'label': 'Cosine Similarity'})
plt.title('Document Similarity Matrix (Cosine)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Коли використовувати?

- ✅ **Text mining** (TF-IDF vectors)
- ✅ **Recommender systems** (user-item ratings)
- ✅ **Image retrieval** (feature vectors)
- ✅ Коли важливий **напрямок**, не magnitude
- ✅ **Sparse high-dimensional** data
- ❌ Коли magnitude важлива

---

# 2. Pearson Correlation Coefficient

## Формула

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Range: $[-1, 1]$
- 1 → perfect positive correlation
- 0 → no correlation
- -1 → perfect negative correlation

## Інтуїція

**Pearson Correlation** вимірює **лінійну залежність** між двома змінними. На відміну від Cosine, **центрує** дані (віднімає mean).

```
Різниця Cosine vs Pearson:

User ratings:
A: [5, 5, 5, 1, 1]
B: [4, 4, 4, 2, 2]

Cosine: висока схожість (схожий паттерн)
Pearson: ще вища (враховує, що B систематично нижче)

Pearson краще для recommender systems!
```

## Код

```python
import numpy as np

def pearson_correlation(x, y):
    """Обчислити Pearson Correlation"""
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    
    return numerator / denominator

# Приклад: Рейтинги користувачів
user_A = np.array([5, 5, 5, 1, 1])
user_B = np.array([4, 4, 4, 2, 2])
user_C = np.array([1, 1, 1, 5, 5])

corr_AB = pearson_correlation(user_A, user_B)
corr_AC = pearson_correlation(user_A, user_C)

print(f"Pearson(A, B): {corr_AB:.4f}")  # Позитивна кореляція
print(f"Pearson(A, C): {corr_AC:.4f}")  # Негативна кореляція

# Через numpy
corr_AB_np = np.corrcoef(user_A, user_B)[0, 1]
print(f"NumPy Pearson(A, B): {corr_AB_np:.4f}")

# Через scipy
from scipy.stats import pearsonr
corr, p_value = pearsonr(user_A, user_B)
print(f"Scipy Pearson(A, B): {corr:.4f} (p-value: {p_value:.4f})")
```

## Візуалізація Cosine vs Pearson

```python
import matplotlib.pyplot as plt
import numpy as np

# Дані
user_A = np.array([5, 5, 5, 1, 1])
user_B = np.array([4, 4, 4, 2, 2])  # Систематично нижче
user_C = np.array([1, 1, 1, 5, 5])  # Протилежні вподобання

items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']

# Обчислити метрики
cos_AB = cosine_similarity(user_A, user_B)
cos_AC = cosine_similarity(user_A, user_C)
pearson_AB = pearson_correlation(user_A, user_B)
pearson_AC = pearson_correlation(user_A, user_C)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: A vs B
axes[0].plot(items, user_A, 'o-', linewidth=2, markersize=10, label='User A')
axes[0].plot(items, user_B, 's-', linewidth=2, markersize=10, label='User B')
axes[0].set_ylabel('Rating', fontsize=12)
axes[0].set_title(f'User A vs B\nCosine: {cos_AB:.3f} | Pearson: {pearson_AB:.3f}', 
                 fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 6)

# Subplot 2: A vs C
axes[1].plot(items, user_A, 'o-', linewidth=2, markersize=10, label='User A')
axes[1].plot(items, user_C, '^-', linewidth=2, markersize=10, label='User C', color='red')
axes[1].set_ylabel('Rating', fontsize=12)
axes[1].set_title(f'User A vs C\nCosine: {cos_AC:.3f} | Pearson: {pearson_AC:.3f}', 
                 fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 6)

plt.tight_layout()
plt.show()

print("Висновки:")
print("A vs B: схожі паттерни (B систематично нижче)")
print(f"  Cosine:  {cos_AB:.3f}")
print(f"  Pearson: {pearson_AB:.3f} (майже perfect correlation)")
print("\nA vs C: протилежні вподобання")
print(f"  Cosine:  {cos_AC:.3f}")
print(f"  Pearson: {pearson_AC:.3f} (perfect negative correlation)")
```

## Коли використовувати?

- ✅ **Recommender systems** (centered ratings)
- ✅ Коли є **systematic bias** (один user завжди вище)
- ✅ Виявлення **linear relationships**
- ✅ **Feature selection** (correlation with target)
- ❌ Non-linear relationships (використовуй Spearman)

---

# 3. Jaccard Similarity

## Формула

$$J(\mathbf{A}, \mathbf{B}) = \frac{|\mathbf{A} \cap \mathbf{B}|}{|\mathbf{A} \cup \mathbf{B}|}$$

Range: $[0, 1]$
- 0 → зовсім різні
- 1 → ідентичні

## Інтуїція

**Jaccard Similarity** для **множин** — співвідношення **спільних елементів** до **всіх унікальних**.

```
Приклад: Покупки

Customer A купив: {milk, bread, eggs, butter}
Customer B купив: {milk, bread, cheese}

Спільні: {milk, bread} → 2 елементи
Всі унікальні: {milk, bread, eggs, butter, cheese} → 5 елементів

Jaccard = 2/5 = 0.4
```

## Код

```python
def jaccard_similarity(set_a, set_b):
    """Обчислити Jaccard Similarity для множин"""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0

# Приклад: Покупки
customer_A = {'milk', 'bread', 'eggs', 'butter'}
customer_B = {'milk', 'bread', 'cheese'}
customer_C = {'apple', 'banana', 'orange'}

sim_AB = jaccard_similarity(customer_A, customer_B)
sim_AC = jaccard_similarity(customer_A, customer_C)

print(f"Jaccard(A, B): {sim_AB:.4f}")  # 0.4 (схожі)
print(f"Jaccard(A, C): {sim_AC:.4f}")  # 0.0 (різні)

# Для binary vectors
from sklearn.metrics import jaccard_score

# Binary representation
items = ['milk', 'bread', 'eggs', 'butter', 'cheese', 'apple']
vector_A = [1, 1, 1, 1, 0, 0]  # A купив milk, bread, eggs, butter
vector_B = [1, 1, 0, 0, 1, 0]  # B купив milk, bread, cheese

jaccard_sklearn = jaccard_score(vector_A, vector_B, average='binary')
print(f"Jaccard (binary vectors): {jaccard_sklearn:.4f}")
```

## Візуалізація

```python
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Множини
set_A = {'milk', 'bread', 'eggs', 'butter'}
set_B = {'milk', 'bread', 'cheese'}

# Jaccard
jaccard = jaccard_similarity(set_A, set_B)

# Візуалізація Venn diagram
plt.figure(figsize=(10, 6))

venn = venn2([set_A, set_B], set_labels=('Customer A', 'Customer B'))

# Кольори
venn.get_patch_by_id('10').set_color('lightblue')
venn.get_patch_by_id('01').set_color('lightcoral')
venn.get_patch_by_id('11').set_color('lightgreen')

plt.title(f'Jaccard Similarity = {jaccard:.3f}\n'
         f'Intersection: {len(set_A & set_B)} | Union: {len(set_A | set_B)}',
         fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Set A: {set_A}")
print(f"Set B: {set_B}")
print(f"Intersection: {set_A & set_B}")
print(f"Union: {set_A | set_B}")
print(f"Jaccard: {jaccard:.4f}")
```

## Коли використовувати?

- ✅ **Recommender systems** (item sets)
- ✅ **Market basket analysis**
- ✅ **Document similarity** (word sets)
- ✅ **Genomics** (gene sets)
- ✅ Binary/set data
- ❌ Continuous values (використовуй Cosine/Pearson)

---

# 4. Dice Coefficient (Sørensen-Dice)

## Формула

$$\text{Dice}(\mathbf{A}, \mathbf{B}) = \frac{2|\mathbf{A} \cap \mathbf{B}|}{|\mathbf{A}| + |\mathbf{B}|}$$

Range: $[0, 1]$

## Інтуїція

**Dice Coefficient** схожа на Jaccard, але **дає більшу вагу спільним елементам** (множить на 2).

```
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

Intersection: {3, 4} → 2
Union: {1, 2, 3, 4, 5, 6} → 6

Jaccard = 2/6 ≈ 0.33
Dice = 2*2/(4+4) = 4/8 = 0.5  ← Вище!
```

## Код

```python
def dice_coefficient(set_a, set_b):
    """Обчислити Dice Coefficient"""
    intersection = len(set_a & set_b)
    return 2 * intersection / (len(set_a) + len(set_b)) if (len(set_a) + len(set_b)) > 0 else 0

# Приклад
set_A = {1, 2, 3, 4}
set_B = {3, 4, 5, 6}

jaccard = jaccard_similarity(set_A, set_B)
dice = dice_coefficient(set_A, set_B)

print(f"Jaccard: {jaccard:.4f}")
print(f"Dice:    {dice:.4f}")

# Зв'язок між Jaccard і Dice
# Dice = 2*Jaccard / (1 + Jaccard)
dice_from_jaccard = 2 * jaccard / (1 + jaccard)
print(f"Dice (from Jaccard): {dice_from_jaccard:.4f}")
```

## Порівняння Jaccard vs Dice

```python
import matplotlib.pyplot as plt
import numpy as np

# Різні розміри перетину
sizes = np.linspace(0, 1, 100)  # Частка перетину

jaccard_values = []
dice_values = []

for intersection_ratio in sizes:
    # Припустимо |A| = |B| = 1
    # Intersection = intersection_ratio
    # Union = 2 - intersection_ratio
    
    jaccard = intersection_ratio / (2 - intersection_ratio)
    dice = 2 * intersection_ratio / 2
    
    jaccard_values.append(jaccard)
    dice_values.append(dice)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(sizes, jaccard_values, linewidth=2, label='Jaccard')
plt.plot(sizes, dice_values, linewidth=2, label='Dice', linestyle='--')
plt.xlabel('Intersection Ratio', fontsize=12)
plt.ylabel('Similarity', fontsize=12)
plt.title('Jaccard vs Dice Coefficient', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Спостереження:")
print("Dice завжди >= Jaccard (дає більшу вагу перетину)")
```

---

# 5. Spearman Rank Correlation

## Формула

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

де $d_i$ — різниця рангів.

Range: $[-1, 1]$

## Інтуїція

**Spearman** — це **Pearson для рангів**. Вимірює **монотонну** (не обов'язково лінійну) залежність.

```
Приклад: Salary vs Experience

Experience: [1, 2, 5, 10, 15] years
Salary:     [30, 35, 55, 85, 100] k$

Нелінійна залежність (exponential growth)
Pearson: може бути середня
Spearman: висока (monotonic increase)
```

## Код

```python
from scipy.stats import spearmanr, pearsonr
import numpy as np

# Приклад: Нелінійна залежність
x = np.array([1, 2, 5, 10, 15])
y = np.array([30, 35, 55, 85, 100])  # Приблизно exponential

# Pearson (лінійна)
pearson_corr, _ = pearsonr(x, y)

# Spearman (монотонна)
spearman_corr, _ = spearmanr(x, y)

print(f"Pearson:  {pearson_corr:.4f}")
print(f"Spearman: {spearman_corr:.4f}")

# Для сильно нелінійної
x_nonlinear = np.array([1, 2, 3, 4, 5])
y_nonlinear = np.array([1, 4, 9, 16, 25])  # Квадратична

pearson_nl, _ = pearsonr(x_nonlinear, y_nonlinear)
spearman_nl, _ = spearmanr(x_nonlinear, y_nonlinear)

print(f"\nНелінійна залежність (y=x²):")
print(f"  Pearson:  {pearson_nl:.4f}")
print(f"  Spearman: {spearman_nl:.4f}")  # Perfect monotonic!
```

## Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Різні типи залежностей
x = np.linspace(1, 10, 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Лінійна
y_linear = 2*x + 1 + np.random.normal(0, 1, 50)
pearson_1, _ = pearsonr(x, y_linear)
spearman_1, _ = spearmanr(x, y_linear)

axes[0, 0].scatter(x, y_linear, alpha=0.6)
axes[0, 0].set_title(f'Linear\nPearson: {pearson_1:.3f} | Spearman: {spearman_1:.3f}',
                    fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Квадратична
y_quadratic = x**2 + np.random.normal(0, 5, 50)
pearson_2, _ = pearsonr(x, y_quadratic)
spearman_2, _ = spearmanr(x, y_quadratic)

axes[0, 1].scatter(x, y_quadratic, alpha=0.6, color='orange')
axes[0, 1].set_title(f'Quadratic\nPearson: {pearson_2:.3f} | Spearman: {spearman_2:.3f}',
                    fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Exponential
y_exp = np.exp(x/5) + np.random.normal(0, 5, 50)
pearson_3, _ = pearsonr(x, y_exp)
spearman_3, _ = spearmanr(x, y_exp)

axes[1, 0].scatter(x, y_exp, alpha=0.6, color='green')
axes[1, 0].set_title(f'Exponential\nPearson: {pearson_3:.3f} | Spearman: {spearman_3:.3f}',
                    fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. No correlation
y_random = np.random.normal(50, 10, 50)
pearson_4, _ = pearsonr(x, y_random)
spearman_4, _ = spearmanr(x, y_random)

axes[1, 1].scatter(x, y_random, alpha=0.6, color='red')
axes[1, 1].set_title(f'No Correlation\nPearson: {pearson_4:.3f} | Spearman: {spearman_4:.3f}',
                    fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Коли використовувати?

- ✅ **Non-linear monotonic** relationships
- ✅ **Ordinal data** (rankings)
- ✅ Коли є **outliers** (robust)
- ✅ **Feature selection** (non-linear)
- ❌ Коли потрібна саме лінійна залежність

---

# 6. Kullback-Leibler Divergence (KL Divergence)

## Формула

$$D_{KL}(P \| Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

Range: $[0, \infty)$
- 0 → розподіли ідентичні
- ∞ → зовсім різні

**Важливо:** KL Divergence **не симетрична**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

## Інтуїція

**KL Divergence** вимірює, **скільки інформації втрачається**, якщо використовувати розподіл $Q$ замість справжнього $P$.

```
Приклад: Моделювання кидання кубика

P (справжній):  [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
Q (модель):     [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

KL(P || Q) = скільки інформації втрачаємо
```

## Код

```python
import numpy as np

def kl_divergence(p, q):
    """
    Обчислити KL Divergence
    
    D_KL(P || Q) = Σ P(i) log(P(i)/Q(i))
    """
    # Уникаємо log(0)
    q = np.where(q == 0, 1e-10, q)
    p = np.where(p == 0, 1e-10, p)
    
    return np.sum(p * np.log(p / q))

# Приклад
p_true = np.array([0.2, 0.3, 0.5])  # Справжній розподіл
q_model = np.array([0.25, 0.25, 0.5])  # Модель

kl_pq = kl_divergence(p_true, q_model)
kl_qp = kl_divergence(q_model, p_true)

print(f"KL(P || Q): {kl_pq:.6f}")
print(f"KL(Q || P): {kl_qp:.6f}")
print(f"Asymmetric: {kl_pq != kl_qp}")

# Через scipy
from scipy.stats import entropy

kl_scipy = entropy(p_true, q_model)
print(f"Scipy KL(P || Q): {kl_scipy:.6f}")
```

## Візуалізація

```python
import matplotlib.pyplot as plt
import numpy as np

# Два розподіли
x = np.array([1, 2, 3, 4, 5])
p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # Справжній
q = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform (модель)

# KL
kl = kl_divergence(p, q)

# Візуалізація
plt.figure(figsize=(10, 6))

width = 0.35
plt.bar(x - width/2, p, width, label='P (True)', alpha=0.7, color='blue')
plt.bar(x + width/2, q, width, label='Q (Model)', alpha=0.7, color='red')

plt.xlabel('Category', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title(f'KL Divergence D(P || Q) = {kl:.4f}', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

## Jensen-Shannon Divergence (Symmetric)

```python
def js_divergence(p, q):
    """
    Jensen-Shannon Divergence (symmetric version of KL)
    
    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    де M = (P + Q) / 2
    """
    m = (p + q) / 2
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Приклад
p = np.array([0.2, 0.3, 0.5])
q = np.array([0.25, 0.25, 0.5])

js = js_divergence(p, q)
js_reverse = js_divergence(q, p)

print(f"JS(P, Q): {js:.6f}")
print(f"JS(Q, P): {js_reverse:.6f}")
print(f"Symmetric: {np.isclose(js, js_reverse)}")

# Через scipy
from scipy.spatial.distance import jensenshannon

js_scipy = jensenshannon(p, q) ** 2  # scipy повертає sqrt
print(f"Scipy JS: {js_scipy:.6f}")
```

## Коли використовувати?

- ✅ **Comparing probability distributions**
- ✅ **Information theory**
- ✅ **Model evaluation** (generative models)
- ✅ **NLP** (topic modeling)
- ❌ Коли потрібна симетрична метрика (використовуй JS)

---

# Порівняльна таблиця

| Метрика | Тип даних | Range | Симетрична? | Коли використовувати |
|---------|-----------|-------|-------------|---------------------|
| **Cosine** | Vectors | [-1, 1] | ✅ | Text, sparse, direction важливий |
| **Pearson** | Vectors | [-1, 1] | ✅ | Linear correlation, recommenders |
| **Spearman** | Rankings | [-1, 1] | ✅ | Monotonic, ordinal data |
| **Jaccard** | Sets | [0, 1] | ✅ | Binary, sets, market basket |
| **Dice** | Sets | [0, 1] | ✅ | Sets (більша вага intersection) |
| **KL Divergence** | Distributions | [0, ∞) | ❌ | Probability distributions |
| **JS Divergence** | Distributions | [0, 1] | ✅ | Symmetric KL |

---

## Практичні приклади

### 1. Movie Recommender System

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# User-Item ratings matrix
ratings = pd.DataFrame({
    'Movie1': [5, 4, 0, 0, 3],
    'Movie2': [4, 5, 0, 0, 4],
    'Movie3': [0, 0, 5, 4, 0],
    'Movie4': [0, 0, 4, 5, 0],
    'Movie5': [3, 4, 0, 0, 5]
}, index=['User1', 'User2', 'User3', 'User4', 'User5'])

print("Ratings Matrix:")
print(ratings)

# Cosine similarity між користувачами
user_similarity = cosine_similarity(ratings)
user_similarity_df = pd.DataFrame(user_similarity, 
                                  index=ratings.index,
                                  columns=ratings.index)

print("\nUser Similarity (Cosine):")
print(user_similarity_df)

# Знайти найбільш схожих користувачів для User1
user1_similarities = user_similarity_df.loc['User1'].drop('User1').sort_values(ascending=False)
print(f"\nMost similar to User1:")
print(user1_similarities)

# Рекомендації
def recommend_movies(user, ratings, user_similarity, n_recommendations=2):
    """Рекомендувати фільми на основі схожих користувачів"""
    # Знайти схожих користувачів
    similar_users = user_similarity_df.loc[user].drop(user).sort_values(ascending=False)
    
    # Фільми, які user ще не дивився
    user_movies = ratings.loc[user]
    unwatched = user_movies[user_movies == 0].index
    
    # Оцінки від схожих користувачів
    recommendations = {}
    for movie in unwatched:
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user in similar_users.index[:3]:  # Top 3
            if ratings.loc[similar_user, movie] > 0:
                weight = similar_users[similar_user]
                weighted_sum += weight * ratings.loc[similar_user, movie]
                similarity_sum += weight
        
        if similarity_sum > 0:
            recommendations[movie] = weighted_sum / similarity_sum
    
    # Топ рекомендації
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

recommendations = recommend_movies('User1', ratings, user_similarity_df)
print(f"\nRecommendations for User1:")
for movie, score in recommendations:
    print(f"  {movie}: predicted rating = {score:.2f}")
```

### 2. Document Clustering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Документи
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing is a branch of AI",
    "Statistics is the foundation of data science",
    "Probability theory is used in statistical modeling",
    "Data visualization helps understand patterns in data"
]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Clustering
clustering = AgglomerativeClustering(n_clusters=3, metric='cosine', linkage='average')
labels = clustering.fit_predict(tfidf_matrix.toarray())

# Візуалізація similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
           xticklabels=[f'Doc{i+1}' for i in range(len(documents))],
           yticklabels=[f'Doc{i+1}' for i in range(len(documents))],
           cbar_kws={'label': 'Cosine Similarity'})
plt.title('Document Similarity Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Показати кластери
print("\nDocument Clusters:")
for i, (doc, label) in enumerate(zip(documents, labels)):
    print(f"Cluster {label}: Doc{i+1} - {doc[:50]}...")
```

---

## Поширені помилки ❌

### 1. Плутати Distance і Similarity

```python
# ❌ ПОГАНО
from sklearn.metrics.pairwise import cosine_similarity
distance = 1 - cosine_similarity(x, y)  # Це Cosine DISTANCE, не similarity!

# ✅ ДОБРЕ
similarity = cosine_similarity(x, y)  # Range [0, 1] для normalized
```

### 2. Не нормалізувати для Cosine

```python
# Cosine similarity вже нормалізована (ділить на норми)
# Але якщо використовуєш як Distance в KNN:

from sklearn.neighbors import KNeighborsClassifier

# ✅ ДОБРЕ: cosine metric вбудована
knn = KNeighborsClassifier(metric='cosine')
```

### 3. Використовувати Pearson для sparse data

```python
# ❌ ПОГАНО: Pearson для sparse TF-IDF
# Багато нулів → mean зміщена

# ✅ ДОБРЕ: Cosine для sparse
```

---

## Практичні поради 💡

### 1. Вибір метрики за типом даних

```python
# Dense continuous → Pearson або Cosine
# Sparse continuous → Cosine
# Binary/Sets → Jaccard або Dice
# Rankings → Spearman
# Distributions → KL або JS
```

### 2. Normalization важлива

```python
from sklearn.preprocessing import normalize

# Для Cosine similarity
X_normalized = normalize(X, norm='l2')  # Unit vectors
similarity = X_normalized @ X_normalized.T  # Dot product = Cosine
```

### 3. Обирай symmetric метрику

```python
# KL divergence asymmetric → проблеми
# Використовуй JS divergence (symmetric) або
# Average: (KL(P||Q) + KL(Q||P)) / 2
```

---

## Ключові висновки

> Similarity Measures вимірюють схожість між об'єктами. Вибір метрики залежить від типу даних та задачі.

**Основні метрики:**
- **Cosine** — кут між векторами (text, sparse data)
- **Pearson** — лінійна кореляція (recommenders, centered)
- **Spearman** — монотонна залежність (rankings, non-linear)
- **Jaccard** — для множин (sets, binary)
- **KL/JS** — для розподілів ймовірностей

**Вибір:**
```
Text/Sparse       → Cosine
Recommenders      → Pearson (centered ratings)
Rankings          → Spearman
Binary/Sets       → Jaccard
Distributions     → KL/JS
```

**Distance ↔ Similarity:**
```
Similarity = 1 / (1 + Distance)
або
Distance = 1 - Similarity  (для normalized)
```

---

#ml #similarity-measures #cosine #pearson #jaccard #spearman #kl-divergence #recommender-systems
