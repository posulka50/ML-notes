# K-Means Clustering (K-середніх)

## Що це?

**K-Means** — це алгоритм unsupervised learning для **кластеризації** (групування) даних у $K$ кластерів на основі схожості ознак.

**Головна ідея:** розділити дані на $K$ груп так, щоб об'єкти всередині кластера були схожими, а об'єкти з різних кластерів — відмінними.

## Навіщо потрібен?

- 🎯 **Сегментація клієнтів** — розділити клієнтів на групи для таргетованого маркетингу
- 📊 **Стиснення даних** — представити дані меншою кількістю центроїдів
- 🔍 **Anomaly detection** — знайти викиди (далеко від центроїдів)
- 🎨 **Стиснення зображень** — зменшити кількість кольорів
- 📈 **Feature engineering** — створити нові ознаки на основі кластерів
- 🗺️ **Географічна сегментація** — групування локацій

## Коли використовувати?

**Потрібно:**
- **Знаємо приблизну кількість кластерів** $K$
- Кластери **приблизно сферичної форми**
- Кластери **приблизно однакового розміру**
- Потрібна **швидкість** — K-Means дуже швидкий
- **Числові ознаки** (неперервні)
- Великі дані (масштабується добре)

**Не потрібно:**
- **Не знаємо $K$** → Hierarchical Clustering, DBSCAN
- Кластери **складної форми** (еліпси, довгі) → DBSCAN, Gaussian Mixture
- Кластери **різного розміру/щільності** → DBSCAN
- **Категоріальні дані** → K-Modes
- Потрібна **ієрархія** кластерів → Hierarchical Clustering

---

## Як працює K-Means?

### Алгоритм

**Вхід:** дані $X = \{x_1, x_2, ..., x_n\}$, кількість кластерів $K$

**1. Ініціалізація:**
   - Випадково вибрати $K$ точок як початкові центроїди $\mu_1, \mu_2, ..., \mu_K$

**2. Повторювати до збіжності:**

   **a) Assignment step (призначення):**
   - Для кожної точки $x_i$ знайти найближчий центроїд
   - Призначити точку до кластера $c_i$:
   $$c_i = \arg\min_k ||x_i - \mu_k||^2$$

   **b) Update step (оновлення):**
   - Перерахувати центроїди як середнє точок у кластері:
   $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$
   де $C_k$ — множина точок у кластері $k$

**3. Зупинка:**
   - Центроїди не змінюються
   - Або досягнута максимальна кількість ітерацій

### Візуалізація алгоритму

```
Ітерація 0 (ініціалізація):
    y
    |  •   • •
    |    +     •    + = центроїди
    | •    •  •
    |•   +   •
    |_________ x

Ітерація 1 (assignment):
    y
    |  🔴   🔴 🔴
    |    +     🔵    
    | 🔵    🔵  🔵
    |🟢   +   🟢
    |_________ x
    
    Кожна точка пофарбована в колір найближчого центроїду

Ітерація 1 (update):
    y
    |  🔴   🔴 🔴
    |      +   🔵    ← центроїди зсунулись
    | 🔵    🔵  🔵
    |🟢     +
    |_________ x

Ітерація 2-3... → збіжність
```

### Функція втрат (Inertia / Within-Cluster Sum of Squares)

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

**Інтуїція:** сума квадратів відстаней від точок до їхніх центроїдів.

**Мета:** мінімізувати $J$ → компактні кластери.

---

## Математика

### Евклідова відстань

$$d(x_i, \mu_k) = ||x_i - \mu_k|| = \sqrt{\sum_{j=1}^{p} (x_{ij} - \mu_{kj})^2}$$

де:
- $p$ — кількість ознак
- $x_{ij}$ — значення $j$-ї ознаки точки $i$
- $\mu_{kj}$ — значення $j$-ї ознаки центроїда $k$

**Для 2D:**
$$d = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2}$$

### Приклад обчислення

**Точка:** $x = [3, 4]$
**Центроїд:** $\mu = [1, 2]$

$$d = \sqrt{(3-1)^2 + (4-2)^2} = \sqrt{4 + 4} = \sqrt{8} \approx 2.83$$

### Оновлення центроїда

**Кластер містить точки:** $\{[1, 2], [3, 4], [2, 3]\}$

**Новий центроїд:**
$$\mu = \left[\frac{1+3+2}{3}, \frac{2+4+3}{3}\right] = [2, 3]$$

---

## Простий приклад: Сегментація клієнтів

### Дані

| Клієнт | Вік | Дохід (тис. $) |
|--------|-----|----------------|
| A | 25 | 30 |
| B | 45 | 80 |
| C | 35 | 50 |
| D | 50 | 90 |
| E | 28 | 35 |
| F | 47 | 85 |
| G | 32 | 45 |

**Мета:** розділити на $K=2$ сегменти.

### Ітерація 0: Ініціалізація

**Випадково вибрані центроїди:**
- $\mu_1 = [25, 30]$ (клієнт A)
- $\mu_2 = [50, 90]$ (клієнт D)

### Ітерація 1: Assignment

**Для кожного клієнта обчислюємо відстані:**

**Клієнт B:** $[45, 80]$
- До $\mu_1$: $\sqrt{(45-25)^2 + (80-30)^2} = \sqrt{400+2500} = 53.85$
- До $\mu_2$: $\sqrt{(45-50)^2 + (80-90)^2} = \sqrt{25+100} = 11.18$ ✓

**Клієнт B → Кластер 2**

Аналогічно для інших...

**Результат призначення:**
- **Кластер 1:** A, C, E, G (молоді з низьким доходом)
- **Кластер 2:** B, D, F (старші з високим доходом)

### Ітерація 1: Update

**Кластер 1:** $\{[25,30], [35,50], [28,35], [32,45]\}$
$$\mu_1 = \left[\frac{25+35+28+32}{4}, \frac{30+50+35+45}{4}\right] = [30, 40]$$

**Кластер 2:** $\{[45,80], [50,90], [47,85]\}$
$$\mu_2 = \left[\frac{45+50+47}{3}, \frac{80+90+85}{3}\right] = [47.3, 85]$$

### Ітерація 2, 3...

Повторюємо до збіжності (центроїди перестають змінюватися).

**Фінальні кластери:**
- **Сегмент 1:** "Молоді з середнім доходом"
- **Сегмент 2:** "Старші з високим доходом"

---

## Складний приклад: Стиснення зображення

### Задача

Зображення 100x100 пікселів, RGB (16,777,216 можливих кольорів).

**Мета:** зменшити до $K=16$ кольорів (стиснення).

### Підхід

1. Представити кожен піксель як точку в 3D просторі: $[R, G, B]$
2. Застосувати K-Means з $K=16$
3. Замінити кожен піксель на центроїд його кластера

### Результат

**До:** 10,000 пікселів × 3 байти = 30,000 байт
**Після:** 10,000 індексів (4 біти) + 16 центроїдів × 3 байти = 5,048 байт

**Стиснення:** ~83% 🎉

### Код (приклад)

```python
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

# Завантажити зображення
img = Image.open('image.jpg')
img_array = np.array(img)  # shape: (height, width, 3)

# Reshape до (n_pixels, 3)
pixels = img_array.reshape(-1, 3)

# K-Means з 16 кольорами
kmeans = KMeans(n_clusters=16, random_state=42, n_init=10)
kmeans.fit(pixels)

# Замінити кожен піксель на центроїд
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]

# Reshape назад до зображення
compressed_img = compressed_pixels.reshape(img_array.shape).astype(np.uint8)

# Зберегти
Image.fromarray(compressed_img).save('compressed.jpg')
```

---

## Код (Python + scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Генерація даних
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# 2. K-Means
kmeans = KMeans(
    n_clusters=4,        # Кількість кластерів
    init='k-means++',    # Метод ініціалізації (розумний вибір)
    n_init=10,           # Кількість запусків з різними центроїдами
    max_iter=300,        # Максимум ітерацій
    random_state=42
)

# 3. Навчання
kmeans.fit(X)

# 4. Результати
labels = kmeans.labels_              # Мітки кластерів для кожної точки
centroids = kmeans.cluster_centers_  # Координати центроїдів
inertia = kmeans.inertia_            # Сума квадратів відстаней (WCSS)

print(f"Inertia (WCSS): {inertia:.2f}")
print(f"Number of iterations: {kmeans.n_iter_}")

# 5. Передбачення для нових даних
X_new = np.array([[0, 0], [4, 4]])
predicted_labels = kmeans.predict(X_new)
print(f"Predicted clusters: {predicted_labels}")

# 6. Візуалізація
plt.figure(figsize=(12, 5))

# До кластеризації
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6)
plt.title('Before K-Means', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

# Після кластеризації
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, alpha=0.6, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', s=200, marker='X', edgecolors='black', linewidths=2,
            label='Centroids')
plt.title('After K-Means', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Повний приклад: Сегментація клієнтів

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Генерація даних про клієнтів
np.random.seed(42)
n_customers = 500

data = {
    'Age': np.concatenate([
        np.random.normal(25, 5, 150),   # Молоді
        np.random.normal(45, 8, 200),   # Середній вік
        np.random.normal(65, 7, 150)    # Старші
    ]),
    'Income': np.concatenate([
        np.random.normal(35, 8, 150),   # Низький дохід
        np.random.normal(65, 12, 200),  # Середній дохід
        np.random.normal(95, 15, 150)   # Високий дохід
    ]),
    'Spending_Score': np.concatenate([
        np.random.normal(30, 10, 150),
        np.random.normal(50, 15, 200),
        np.random.normal(75, 12, 150)
    ])
}

df = pd.DataFrame(data)

# Переконатися, що значення в реалістичних межах
df['Age'] = df['Age'].clip(18, 80)
df['Income'] = df['Income'].clip(20, 150)
df['Spending_Score'] = df['Spending_Score'].clip(1, 100)

print("=== Dataset Info ===")
print(df.describe())

# Візуалізація розподілів
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, col in enumerate(['Age', 'Income', 'Spending_Score']):
    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Нормалізація (ВАЖЛИВО для K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Age', 'Income', 'Spending_Score']])

# K-Means з різною кількістю кластерів
K_range = range(2, 11)
inertias = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Elbow Method та Silhouette Score
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(K_range, inertias, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (WCSS)', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Optimal K=4')
axes[0].legend()

# Silhouette score
axes[1].plot(K_range, silhouette_scores, 's-', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Optimal K=4')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\nOptimal K (Elbow): ~4")
print(f"Optimal K (Silhouette): {K_range[np.argmax(silhouette_scores)]}")

# Фінальна модель з K=4
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Аналіз кластерів
print("\n" + "="*70)
print("=== Cluster Analysis ===")
print("="*70)

for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster} (n={len(cluster_data)}):")
    print(cluster_data[['Age', 'Income', 'Spending_Score']].mean())

# Назви сегментів
cluster_names = {
    0: "Young Low Income",
    1: "Middle Age Medium Income",
    2: "Senior High Income",
    3: "Young High Spenders"
}

df['Segment'] = df['Cluster'].map(cluster_names)

# Візуалізація кластерів
fig = plt.figure(figsize=(16, 12))

# 3D scatter
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
scatter = ax1.scatter(df['Age'], df['Income'], df['Spending_Score'],
                     c=df['Cluster'], cmap='viridis', s=50, alpha=0.6)
ax1.set_xlabel('Age', fontsize=11)
ax1.set_ylabel('Income', fontsize=11)
ax1.set_zlabel('Spending Score', fontsize=11)
ax1.set_title('3D Cluster Visualization', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Cluster')

# Age vs Income
ax2 = fig.add_subplot(2, 2, 2)
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    ax2.scatter(cluster_data['Age'], cluster_data['Income'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax2.set_xlabel('Age', fontsize=11)
ax2.set_ylabel('Income', fontsize=11)
ax2.set_title('Age vs Income', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Income vs Spending
ax3 = fig.add_subplot(2, 2, 3)
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    ax3.scatter(cluster_data['Income'], cluster_data['Spending_Score'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax3.set_xlabel('Income', fontsize=11)
ax3.set_ylabel('Spending Score', fontsize=11)
ax3.set_title('Income vs Spending Score', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Cluster sizes
ax4 = fig.add_subplot(2, 2, 4)
cluster_sizes = df['Cluster'].value_counts().sort_index()
ax4.bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', edgecolor='black')
ax4.set_xlabel('Cluster', fontsize=11)
ax4.set_ylabel('Number of Customers', fontsize=11)
ax4.set_title('Cluster Sizes', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Профілі кластерів
print("\n" + "="*70)
print("=== Customer Segments Profiles ===")
print("="*70)

for cluster in range(optimal_k):
    print(f"\n{cluster_names[cluster]}:")
    cluster_profile = df[df['Cluster'] == cluster][['Age', 'Income', 'Spending_Score']].describe()
    print(cluster_profile.loc[['mean', 'std']])
```

---

## Вибір кількості кластерів K

### Проблема

**K-Means потребує заздалегідь задану кількість кластерів $K$!**

Як вибрати оптимальне $K$?

### 1. Elbow Method (Метод ліктя)

**Ідея:** побудувати графік inertia (WCSS) vs $K$ та знайти "лікоть".

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (WCSS)', fontsize=12)
plt.title('Elbow Method', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Графік:**
```
Inertia
    |•
    | •
    |  •
    |   •___  ← "Лікоть" (оптимальний K)
    |       •___
    |           •___
    |_______________  K
     1  2  3  4  5  6
```

**Оптимальний K:** там, де крива "зламується" (формує лікоть).

**Недолік:** не завжди чіткий лікоть.

### 2. Silhouette Score

**Silhouette coefficient** для точки $i$:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

де:
- $a_i$ — середня відстань до точок свого кластера
- $b_i$ — середня відстань до точок найближчого іншого кластера

**Діапазон:** $[-1, 1]$
- $s_i \approx 1$ → добре класифіковано ✓
- $s_i \approx 0$ → на межі кластерів
- $s_i < 0$ → можливо в неправильному кластері ✗

**Середній Silhouette Score для всіх точок:**

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 's-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score vs K', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal K: {optimal_k}")
print(f"Best Silhouette Score: {max(silhouette_scores):.4f}")
```

### 3. Silhouette Diagram

**Детальна візуалізація** для конкретного $K$:

```python
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# K-Means з K=4
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Silhouette values для кожної точки
silhouette_vals = silhouette_samples(X, labels)

fig, ax = plt.subplots(figsize=(10, 6))

y_lower = 10
for i in range(4):
    # Silhouette values для кластера i
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.viridis(float(i) / 4)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    # Label кластера
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient', fontsize=12)
ax.set_ylabel('Cluster', fontsize=12)
ax.set_title('Silhouette Diagram (K=4)', fontsize=14, fontweight='bold')

# Середня лінія
avg_score = silhouette_score(X, labels)
ax.axvline(x=avg_score, color="red", linestyle="--", 
           label=f'Average: {avg_score:.3f}')
ax.legend()

plt.tight_layout()
plt.show()
```

**Інтерпретація:**
- ✅ Товсті однакові "стовпчики" → добре
- ⚠️ Різна товщина → кластери різного розміру
- ❌ Стовпчики не доходять до середньої лінії → погано

### 4. Gap Statistic

**Порівнює inertia з випадковими даними:**

$$\text{Gap}(k) = E[\log(W_k^*)] - \log(W_k)$$

де:
- $W_k$ — inertia для наших даних
- $W_k^*$ — inertia для випадкових даних

**Оптимальний $K$:** максимальний gap.

### 5. Davies-Bouldin Index

$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left(\frac{s_i + s_j}{d_{ij}}\right)$$

де:
- $s_i$ — середня відстань точок до центроїда кластера $i$
- $d_{ij}$ — відстань між центроїдами $i$ та $j$

**Менше значення → краще!**

```python
from sklearn.metrics import davies_bouldin_score

db_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = davies_bouldin_score(X, labels)
    db_scores.append(score)

optimal_k = K_range[np.argmin(db_scores)]
print(f"Optimal K (Davies-Bouldin): {optimal_k}")
```

---

## Ініціалізація центроїдів

### Проблема

**Випадкова ініціалізація** може призвести до:
- Повільної збіжності
- Локальних мінімумів (не глобальний оптимум)

### 1. Random Initialization

**За замовчуванням:** випадково вибрати $K$ точок з даних.

**Недолік:** залежить від luck.

### 2. K-Means++ (рекомендовано ✓)

**Розумний вибір початкових центроїдів:**

1. Вибрати перший центроїд випадково
2. Для кожної наступної позиції:
   - Обчислити відстань $D(x)$ від кожної точки до найближчого центроїда
   - Вибрати нову точку з ймовірністю $\propto D(x)^2$
   - (Точки далі від центроїдів мають вищу ймовірність)

**Переваги:**
- ✅ Швидша збіжність
- ✅ Кращі результати
- ✅ За замовчуванням у scikit-learn

```python
kmeans = KMeans(n_clusters=4, init='k-means++')  # Рекомендовано
# або
kmeans = KMeans(n_clusters=4, init='random')     # Випадково
```

### 3. Multiple Runs (n_init)

**Запустити K-Means кілька разів** з різними ініціалізаціями та вибрати найкращий результат (мінімальна inertia).

```python
kmeans = KMeans(n_clusters=4, n_init=10)  # 10 запусків
# Автоматично вибере найкращий
```

**За замовчуванням:** `n_init=10` у scikit-learn.

---

## Параметри K-Means

### Основні параметри

```python
KMeans(
    n_clusters=8,           # Кількість кластерів
    init='k-means++',       # Метод ініціалізації
    n_init=10,              # Кількість запусків
    max_iter=300,           # Максимум ітерацій
    tol=1e-4,               # Критерій зупинки
    random_state=42,        # Відтворюваність
    algorithm='lloyd'       # Алгоритм ('lloyd', 'elkan')
)
```

| Параметр | Опис | Типові значення |
|----------|------|-----------------|
| **n_clusters** | Кількість кластерів $K$ | 2-10 (визначити через Elbow/Silhouette) |
| **init** | Метод ініціалізації | 'k-means++' (рекомендовано), 'random' |
| **n_init** | Кількість запусків | 10 (за замовчуванням) |
| **max_iter** | Макс. ітерацій | 300 |
| **tol** | Критерій зупинки | 1e-4 |
| **random_state** | Seed для відтворюваності | 42 |

### Алгоритми

**Lloyd (стандартний):**
- Класичний K-Means алгоритм
- $O(nKdi)$ де $d$ — розмірність, $i$ — ітерації

**Elkan:**
- Використовує triangle inequality для оптимізації
- Швидше на великих даних з багатьма кластерами
- Працює тільки з Euclidean distance

```python
# Для великих даних
kmeans = KMeans(algorithm='elkan')
```

---

## Mini-Batch K-Means

### Проблема

**K-Means повільний на дуже великих даних** (мільйони точок).

### Рішення: Mini-Batch K-Means

**Ідея:** на кожній ітерації використовувати випадкову підмножину (mini-batch) замість всіх даних.

### Алгоритм

1. Ініціалізувати центроїди
2. Повторювати:
   - Вибрати випадковий mini-batch розміру $b$
   - Призначити точки batch до кластерів
   - Оновити центроїди (з врахуванням попередніх)
3. До збіжності

### Код

```python
from sklearn.cluster import MiniBatchKMeans

# Для великих даних
mb_kmeans = MiniBatchKMeans(
    n_clusters=4,
    batch_size=100,      # Розмір mini-batch
    max_iter=100,
    random_state=42
)

mb_kmeans.fit(X)
labels = mb_kmeans.labels_

print(f"Mini-Batch K-Means completed in {mb_kmeans.n_iter_} iterations")
```

### Порівняння

| Характеристика | K-Means | Mini-Batch K-Means |
|----------------|---------|---------------------|
| **Швидкість** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Точність** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Пам'ять** | Більше | Менше |
| **Використання** | < 100K точок | > 100K точок |

**Коли використовувати Mini-Batch:**
- ✅ Дані не вміщуються в пам'ять
- ✅ > 100,000 точок
- ✅ Потрібна швидкість > точність

---

## Preprocessing для K-Means

### 1. Scaling (КРИТИЧНО! ⚠️)

**Проблема:** K-Means використовує Euclidean distance → чутливий до масштабу.

**Приклад:**
```
Вік: 20-80 (діапазон: 60)
Дохід: 20,000-150,000 (діапазон: 130,000)

Без scaling: дохід домінує!
```

**Рішення: StandardScaler**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_scaled)  # ✅ Використовуй scaled дані
```

**Альтернативи:**
- MinMaxScaler (якщо потрібен діапазон [0, 1])
- RobustScaler (для даних з outliers)

### 2. Feature Selection

**Видалити непотрібні/зашумлені ознаки:**

```python
from sklearn.feature_selection import VarianceThreshold

# Видалити ознаки з низькою variance
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)
```

### 3. Dimensionality Reduction

**PCA перед K-Means для візуалізації або прискорення:**

```python
from sklearn.decomposition import PCA

# Зменшити до 2D для візуалізації
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_pca)

# Візуалізація
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```

### 4. Outlier Removal

**Викиди можуть зіпсувати кластеризацію:**

```python
from sklearn.ensemble import IsolationForest

# Видалити outliers
iso = IsolationForest(contamination=0.1, random_state=42)
outliers = iso.fit_predict(X_scaled)

X_clean = X_scaled[outliers == 1]  # Тільки normal points
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Простота** | Легко зрозуміти та реалізувати |
| **Швидкість** | Дуже швидкий, $O(nKdi)$ |
| **Масштабованість** | Працює з великими даними (Mini-Batch) |
| **Гарантована збіжність** | Завжди збігається (до локального мінімуму) |
| **Універсальність** | Працює в різних задачах |
| **Легка інтерпретація** | Центроїди мають чіткий зміст |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Потрібно знати $K$** | Не знає скільки кластерів |
| **Чутливість до ініціалізації** | Може застрягти в локальних мінімумах |
| **Сферичні кластери** | Передбачає круглу форму |
| **Однаковий розмір** | Погано з різними розмірами кластерів |
| **Чутливість до outliers** | Викиди зміщують центроїди |
| **Чутливість до масштабу** | Потрібен scaling |
| **Тільки числові дані** | Не працює з категоріальними |

---

## Порівняння з іншими методами кластеризації

| Метод | Потрібно $K$? | Форма кластерів | Outliers | Швидкість | Використання |
|-------|---------------|-----------------|----------|-----------|--------------|
| **K-Means** | ✅ Так | Сферичні | ❌ Чутливий | ⭐⭐⭐⭐⭐ | Загальна кластеризація |
| **Hierarchical** | ❌ Ні | Будь-яка | ⚠️ Чутливий | ⭐⭐ | Ієрархія, дендрограми |
| **DBSCAN** | ❌ Ні | Будь-яка | ✅ Робастний | ⭐⭐⭐ | Складна форма, outliers |
| **GMM** | ✅ Так | Еліптичні | ⚠️ Середньо | ⭐⭐⭐ | Soft clustering, ймовірності |

---

## Коли використовувати K-Means

### Ідеально підходить ✓

- **Знаємо приблизно $K$** — кількість груп зрозуміла
- **Кластери сферичні** — приблизно круглі
- **Великі дані** — швидкість критична
- **Числові ознаки** — неперервні змінні
- Кластери **приблизно однакового розміру**
- **Першим кроком** — quick baseline

### Краще використати інше ✗

- **Не знаємо $K$** → Hierarchical, DBSCAN
- **Складна форма** кластерів → DBSCAN, GMM
- **Різні розміри** кластерів → DBSCAN
- **Багато outliers** → DBSCAN
- **Категоріальні дані** → K-Modes
- Потрібна **ієрархія** → Hierarchical Clustering
- Потрібні **ймовірності** → Gaussian Mixture Models

---

## Практичні поради 💡

### 1. ЗАВЖДИ scaling!

```python
# ❌ НЕПРАВИЛЬНО
kmeans.fit(X)

# ✅ ПРАВИЛЬНО
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
```

### 2. Використовуй Elbow + Silhouette

```python
# Не покладайся тільки на один метод
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    print(f"K={k}: Inertia={kmeans.inertia_:.0f}, "
          f"Silhouette={silhouette_score(X_scaled, kmeans.labels_):.3f}")
```

### 3. Візуалізуй результати

```python
# ЗАВЖДИ візуалізуй (навіть якщо > 2D, використай PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            c='red', marker='X', s=200)
plt.show()
```

### 4. Перевіряй стабільність

```python
# Запусти кілька разів та порівняй результати
labels_list = []
for i in range(10):
    kmeans = KMeans(n_clusters=4, random_state=i)
    labels_list.append(kmeans.fit_predict(X_scaled))

# Якщо labels сильно відрізняються → погана кластеризація
```

### 5. Domain knowledge

**Інтерпретуй кластери з точки зору бізнесу:**

```python
# Профіль кластерів
for cluster in range(K):
    print(f"\nCluster {cluster}:")
    print(df[df['Cluster'] == cluster][features].describe())
    # ЧИ МАЄ СЕНС ця група?
```

### 6. n_init=10 мінімум

```python
# Завжди кілька запусків для надійності
kmeans = KMeans(n_clusters=4, n_init=10)  # Мінімум
# Або більше для критичних задач
kmeans = KMeans(n_clusters=4, n_init=50)
```

### 7. Mini-Batch для великих даних

```python
# Якщо > 100K точок
if len(X) > 100000:
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=1000)
```

### 8. Видали outliers спочатку

```python
# Outliers зіпсують центроїди
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05)
mask = iso.fit_predict(X_scaled) == 1
X_clean = X_scaled[mask]
```

### 9. Зберігай scaler!

```python
import joblib

# Зберегти scaler та model разом
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans.pkl')

# Для нових даних
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

X_new_scaled = scaler.transform(X_new)
labels_new = kmeans.predict(X_new_scaled)
```

### 10. Назви кластерів

```python
# Дай кластерам зрозумілі назви
cluster_names = {
    0: "Budget Shoppers",
    1: "High-Value Customers",
    2: "Occasional Buyers",
    3: "New Users"
}

df['Segment'] = df['Cluster'].map(cluster_names)
```

---

## Реальні застосування

### 1. Customer Segmentation (Сегментація клієнтів)

**Задача:** Розділити клієнтів на групи для таргетованого маркетингу.

**Ознаки:**
- RFM (Recency, Frequency, Monetary)
- Демографія (вік, стать, локація)
- Поведінка (клік-рейт, час на сайті)

**Приклад кластерів:**
- VIP клієнти (високі покупки)
- Активні (часті візити)
- Неактивні (рідко купують)
- Нові користувачі

### 2. Image Compression (Стиснення зображень)

**Задача:** Зменшити кількість кольорів у зображенні.

**Підхід:**
- Кожен піксель = точка в RGB просторі
- K-Means групує схожі кольори
- Замінити кольори на центроїди

**Результат:** Зображення з $K$ кольорів замість мільйонів.

### 3. Document Clustering (Групування документів)

**Задача:** Згрупувати схожі документи/статті.

**Підхід:**
- TF-IDF векторизація тексту
- K-Means на vectors
- Кожен кластер = тема

### 4. Anomaly Detection (Виявлення аномалій)

**Задача:** Знайти незвичайні транзакції/поведінку.

**Підхід:**
- Кластеризувати нормальні дані
- Точки далеко від центроїдів = аномалії
- Threshold на відстань

### 5. Recommendation Systems

**Задача:** Рекомендації на основі схожості.

**Підхід:**
- Кластеризувати користувачів/товари
- Рекомендувати популярні items у кластері

---

## Поширені помилки ❌

### 1. Не робити scaling

```python
# ❌ КРИТИЧНА ПОМИЛКА
kmeans = KMeans(n_clusters=3)
kmeans.fit(df[['Age', 'Income']])  # Age: 0-100, Income: 0-150000

# Income домінує через великий діапазон!
```

### 2. Використовувати на категоріальних даних

```python
# ❌ НЕПРАВИЛЬНО
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})  # Не має сенсу для K-Means
kmeans.fit(df)

# ✅ ПРАВИЛЬНО
# Використовуй One-Hot Encoding + K-Modes або інший метод
```

### 3. Не перевіряти $K$

```python
# ❌ Просто вибрати K=3 без аналізу
kmeans = KMeans(n_clusters=3)

# ✅ Використовуй Elbow/Silhouette
```

### 4. Ігнорувати outliers

```python
# Outliers сильно зміщують центроїди!
# ✅ Видали їх спочатку
```

### 5. Інтерпретувати без domain knowledge

```python
# ❌ "Кластер 0 = група 1, кластер 1 = група 2"
# ✅ Подивись на профілі та дай зрозумілі назви
```

### 6. Один запуск (n_init=1)

```python
# ❌ РИЗИК ЛОКАЛЬНОГО МІНІМУМУ
kmeans = KMeans(n_clusters=4, n_init=1)

# ✅ Кілька запусків
kmeans = KMeans(n_clusters=4, n_init=10)
```

---

## Пов'язані теми

- [[02_Hierarchical_Clustering]] — альтернативний метод
- [[03_DBSCAN]] — для складних форм
- [[04_Gaussian_Mixture_Models]] — probabilistic clustering
- [[05_Clustering_Evaluation]] — метрики оцінки
- [[PCA]] — dimensionality reduction перед кластеризацією

## Ресурси

- [Scikit-learn: K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Original Paper: Lloyd (1982)](https://cs.nyu.edu/~roweis/csc2515-2006/readings/lloyd57.pdf)
- [StatQuest: K-Means](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [K-Means++: Arthur & Vassilvitskii (2007)](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

---

## Ключові висновки

> K-Means — це ітеративний алгоритм кластеризації, який розділяє дані на $K$ кластерів шляхом мінімізації відстані від точок до центроїдів.

**Основні принципи:**
- **Ітеративний процес:** assignment → update → repeat
- **Мінімізує WCSS** (Within-Cluster Sum of Squares)
- **Потребує $K$** — кількість кластерів заздалегідь
- **K-Means++** ініціалізація для кращих результатів

**Формула оновлення центроїда:**
$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

**Коли використовувати:**
- Сферичні кластери + знаємо $K$ + швидкість = K-Means ✓
- Складна форма кластерів → DBSCAN ✓
- Не знаємо $K$ → Hierarchical Clustering ✓

**Найважливіше:**
- ЗАВЖДИ робити scaling (StandardScaler)
- Використовуй Elbow + Silhouette для вибору $K$
- n_init=10 мінімум для надійності
- Візуалізуй та інтерпретуй результати
- Domain knowledge > математика

---

#ml #unsupervised-learning #clustering #k-means #centroid-based #customer-segmentation
