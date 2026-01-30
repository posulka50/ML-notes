# DBSCAN — Density-Based Spatial Clustering

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** — алгоритм кластеризації, який групує точки на основі їх **щільності** (density), а не відстані до центрів.

**Головна ідея:** кластер — це область високої щільності точок, відділена від інших областей зонами низької щільності.

---

## Що це

DBSCAN — це алгоритм кластеризації, який:

- **Знаходить кластери довільної форми** (не тільки сферичні!)
- **Автоматично виявляє викиди** (noise/outliers)
- **Не потребує заздалегідь знати кількість кластерів K**

---

## Навіщо

### Проблеми K-Means

```
K-Means:
  ✗ Тільки сферичні кластери
  ✗ Потрібно знати K
  ✗ Чутливий до викидів
  ✗ Погано працює з кластерами різного розміру

DBSCAN:
  ✓ Кластери будь-якої форми
  ✓ K визначається автоматично
  ✓ Викиди позначаються окремо
  ✓ Працює з різними розмірами
```

### Приклади задач

- Виявлення аномалій у транзакціях
- Сегментація географічних даних
- Аналіз соціальних мереж
- Обробка зображень
- Виявлення патернів у складних даних

---

## Основні поняття

### 1. Параметри алгоритму

**ε (epsilon)** — радіус околиці

- Відстань, в межах якої шукаємо сусідів
- Приклад: ε = 0.5 → всі точки в радіусі 0.5 вважаються сусідами

**MinPts (min_samples)** — мінімальна кількість точок

- Скільки точок потрібно в околиці ε, щоб область була щільною
- Приклад: MinPts = 5 → потрібно мінімум 5 сусідів

---

### 2. Типи точок

#### **Core Point (Ядро)**

Точка, яка має **≥ MinPts** сусідів в радіусі ε (включно з собою).

```
     ε
    ╱ ╲
   │ ● │  ← центральна точка
   │●●●│  ← 5+ сусідів в радіусі
    ╲ ╱
    
Це core point! ✓
```

#### **Border Point (Межа)**

Точка, яка:

- Сама НЕ є core point (< MinPts сусідів)
- АЛЕ знаходиться в околиці якогось core point

```
Core:  ●●●●●
Border:     ● ← в радіусі core, але сама має мало сусідів
```

#### **Noise Point (Шум/Викид)**

Точка, яка:

- НЕ є core point
- НЕ є border point (далеко від усіх core points)

```
Core: ●●●●●
              
              ● ← одинока точка, далеко від усіх
              
Це noise! (outlier)
```

---

## Алгоритм DBSCAN

### Словами (як це працює)

**Крок 1: Позначаємо всі точки як "не відвідані"**

**Крок 2: Для кожної не відвіданої точки P:**

1. Позначаємо P як відвідану
2. Знаходимо всіх сусідів P в радіусі ε
3. **Якщо сусідів < MinPts:**
    - Поки що позначаємо P як noise
    - Переходимо до наступної точки
4. **Якщо сусідів ≥ MinPts:**
    - P — це **core point**!
    - Створюємо новий кластер C
    - Додаємо P та всіх її сусідів до кластера C
    - **Розширюємо кластер:**
        - Для кожного сусіда Q:
            - Якщо Q не відвідана → позначаємо як відвідану
            - Знаходимо сусідів Q
            - Якщо Q теж core point → додаємо її сусідів до кластера
        - Повторюємо, поки не знайдемо всі density-reachable точки

**Крок 3: Точки, що залишились як noise**

- Якщо точка так і не потрапила в жоден кластер → це outlier

---

### Математично (формально)

**Вхід:**

- Дані: $D = {x_1, x_2, ..., x_n}$
- Параметри: $\varepsilon$ (epsilon), $MinPts$

**Визначення:**

1. **ε-околиця** точки $p$: $$N_\varepsilon(p) = {q \in D \mid dist(p, q) \leq \varepsilon}$$
    
2. **Core point:** $$|N_\varepsilon(p)| \geq MinPts$$
    
3. **Directly density-reachable:** Точка $q$ досяжна з $p$, якщо:
    

- $q \in N_\varepsilon(p)$
- $p$ є core point

4. **Density-reachable:** Існує ланцюжок точок $p_1, p_2, ..., p_n$, де:

- $p_1 = p$, $p_n = q$
- $p_{i+1}$ directly density-reachable з $p_i$

5. **Density-connected:** Точки $p$ і $q$ density-connected, якщо існує точка $o$, з якої досяжні обидві.

**Кластер:** Максимальний набір density-connected точок.

---

### Псевдокод

```python
def DBSCAN(D, eps, MinPts):
    C = 0  # Лічильник кластерів
    labels = [-1] * len(D)  # -1 = не класифіковано
    
    for i, point in enumerate(D):
        if labels[i] != -1:  # Вже відвідана
            continue
            
        # Знайти сусідів
        neighbors = get_neighbors(point, D, eps)
        
        if len(neighbors) < MinPts:
            labels[i] = -1  # Noise (поки що)
            continue
        
        # Новий кластер
        C += 1
        labels[i] = C
        
        # Розширити кластер
        seed_set = neighbors.copy()
        
        while seed_set:
            current_point = seed_set.pop()
            current_idx = D.index(current_point)
            
            if labels[current_idx] == -1:  # Був noise
                labels[current_idx] = C  # Тепер border point
                
            if labels[current_idx] != -1:  # Вже в кластері
                continue
                
            labels[current_idx] = C
            
            # Якщо це core point, додати її сусідів
            current_neighbors = get_neighbors(current_point, D, eps)
            if len(current_neighbors) >= MinPts:
                seed_set.extend(current_neighbors)
    
    return labels


def get_neighbors(point, D, eps):
    """Знайти всі точки в радіусі eps"""
    neighbors = []
    for other_point in D:
        if distance(point, other_point) <= eps:
            neighbors.append(other_point)
    return neighbors
```

---

## Простий приклад (вручну)

### Дані

```
x₁ = [1, 2]
x₂ = [2, 2]
x₃ = [2, 3]
x₄ = [8, 7]
x₅ = [8, 8]
x₆ = [7, 9]
x₇ = [9, 9]
x₈ = [5, 5]  ← одинока точка
```

**Параметри:** ε = 1.5, MinPts = 3

### Візуально

```
  10│
   9│        x₆  x₇
   8│    x₅
   7│    x₄
   6│
   5│      x₈  ← далеко від усіх
   4│
   3│  x₃
   2│x₁ x₂
   1│
   0└─────────────
     1 2 3 4 5 6 7 8 9
```

---

### Покрокове виконання

#### **Крок 1: Обробка x₁ = [1, 2]**

Знаходимо сусідів в радіусі ε = 1.5:

```
d(x₁, x₁) = 0     ✓
d(x₁, x₂) = 1.0   ✓ (в радіусі!)
d(x₁, x₃) = 1.41  ✓ (в радіусі!)
d(x₁, x₄) = 8.6   ✗
d(x₁, x₅) = 8.5   ✗
...

Сусіди x₁: {x₁, x₂, x₃} → кількість = 3 ≥ MinPts
```

**Висновок:** x₁ — **core point**!

**Створюємо Кластер 1** = {x₁, x₂, x₃}

---

#### **Крок 2: Розширюємо Кластер 1**

Перевіряємо сусідів:

**x₂ = [2, 2]:**

```
Сусіди: {x₁, x₂, x₃} → 3 точки → core point ✓
Вже в Кластері 1, нових точок не додає
```

**x₃ = [2, 3]:**

```
Сусіди: {x₁, x₂, x₃} → 3 точки → core point ✓
Вже в Кластері 1, нових точок не додає
```

**Кластер 1 завершено:** {x₁, x₂, x₃}

---

#### **Крок 3: Обробка x₄ = [8, 7]**

```
d(x₄, x₅) = 1.0   ✓
d(x₄, x₆) = 2.24  ✗ (> 1.5)
d(x₄, x₇) = 2.24  ✗

Сусіди x₄: {x₄, x₅} → кількість = 2 < MinPts
```

**Висновок:** x₄ поки що **noise**

---

#### **Крок 4: Обробка x₅ = [8, 8]**

```
d(x₅, x₄) = 1.0   ✓
d(x₅, x₆) = 1.41  ✓
d(x₅, x₇) = 1.41  ✓

Сусіди x₅: {x₄, x₅, x₆, x₇} → кількість = 4 ≥ MinPts
```

**Висновок:** x₅ — **core point**!

**Створюємо Кластер 2** = {x₄, x₅, x₆, x₇}

**Важливо:** x₄ була noise, але тепер стає **border point** кластера 2!

---

#### **Крок 5: Розширюємо Кластер 2**

**x₄:** вже перевірили, не core (border point)

**x₆ = [7, 9]:**

```
Сусіди: {x₅, x₆, x₇} → 3 точки → core point ✓
```

**x₇ = [9, 9]:**

```
Сусіди: {x₅, x₆, x₇} → 3 точки → core point ✓
```

**Кластер 2 завершено:** {x₄, x₅, x₆, x₇}

---

#### **Крок 6: Обробка x₈ = [5, 5]**

```
Відстані до всіх інших > 1.5

Сусіди x₈: {x₈} → кількість = 1 < MinPts
```

**Висновок:** x₈ — **noise (outlier)** ✗

---

### Фінальний результат

```
Кластер 1: {x₁, x₂, x₃}
  - Core points: x₁, x₂, x₃

Кластер 2: {x₄, x₅, x₆, x₇}
  - Core points: x₅, x₆, x₇
  - Border points: x₄

Noise: {x₈}
```


---

## Складний приклад: два кільця

### Дані

Два концентричні кола:

```
Внутрішнє коло: 50 точок з радіусом ~2
Зовнішнє коло: 100 точок з радіусом ~5
```

### Візуалізація

```
       ●●●●●
     ●       ●
    ●   ●●●   ●
   ●   ●   ●   ●
   ●  ●     ●  ●
   ●   ●   ●   ●
    ●   ●●●   ●
     ●       ●
       ●●●●●
```

### K-Means vs DBSCAN

**K-Means (K=2):**

```
       ●●●●●  ← Cluster 1
     ●▓▓▓▓▓●  
    ●▓▓▓●●●▓▓●  ← Розділяє вертикально
   ●▓▓●▓▓▓●▓▓●   (не по колах!)
   ●▓▓●▓▓▓▓●▓▓●  
   ●▓▓▓●▓▓▓●▓▓●  
    ●▓▓▓●●●▓▓●  
     ●▓▓▓▓▓●  
       ●●●●●  ← Cluster 2

ПОМИЛКА! ✗
```

**DBSCAN (ε=0.3, MinPts=5):**

```
       ①①①①①  ← Cluster 1 (зовнішнє)
     ①       ①  
    ①   ②②②   ①  ← Cluster 2 (внутрішнє)
   ①   ②   ②   ①
   ①  ②     ②  ①
   ①   ②   ②   ①
    ①   ②②②   ①
     ①       ①
       ①①①①①

ПРАВИЛЬНО! ✓
```

### Чому DBSCAN працює?

1. **Внутрішнє коло:**
    
    - Точки щільно розташовані → багато сусідів
    - Формується один density-connected компонент
2. **Зовнішнє коло:**
    
    - Також щільно розташовані
    - АЛЕ відділені від внутрішнього (низька щільність між ними)
    - Формується окремий компонент
3. **Між колами:**
    
    - Низька щільність → не формуються кластери

---

## Вибір параметрів ε та MinPts

### Проблема

**Занадто малий ε:**

```
ε = 0.1  → більшість точок = noise
Кластерів: 0-1
```

**Занадто великий ε:**

```
ε = 10   → всі точки в одному кластері
Кластерів: 1
```

---

### Метод 1: k-distance график

**Ідея:** Будуємо графік відстані до k-го найближчого сусіда для всіх точок.

**Алгоритм:**

1. Для кожної точки знайти відстань до k-го сусіда (зазвичай k = MinPts)
2. Відсортувати ці відстані за спаданням
3. Побудувати графік
4. Знайти "коліно" (elbow) на графіку → це оптимальний ε

**Приклад:**

```python
from sklearn.neighbors import NearestNeighbors

# MinPts = 4
k = 4
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X)

distances, indices = neighbors.kneighbors(X)

# Відстань до 4-го сусіда (індекс 3, бо включає саму точку)
k_distances = distances[:, k-1]
k_distances = np.sort(k_distances)[::-1]

# Графік
plt.plot(k_distances)
plt.xlabel('Точки відсортовані за відстанню')
plt.ylabel(f'{k}-distance')
plt.title('k-distance Graph')
plt.grid(True)
```

**Інтерпретація:**

```
k-dist
  ↑
  │●
  │ ●
  │  ●
  │   ●
  │    ●────────  ← "коліно" (різкий спад)
  │              
  │─────────────────  ← плато
  └────────────────→ Points

ε ≈ значення в точці коліна
```

---

### Метод 2: Евристики для MinPts

**Загальні правила:**

1. **Мінімальне значення:** MinPts ≥ D + 1
    
    - D — розмірність даних
    - Приклад: для 2D даних → MinPts ≥ 3
2. **Рекомендоване:** MinPts = 2 × D
    
    - Для 2D → MinPts = 4
    - Для 3D → MinPts = 6
3. **Для шумних даних:** більше MinPts
    
    - Більше викидів → MinPts = 5-10

**Практичні значення:**

|Розмірність|MinPts рекомендоване|
|---|---|
|2D|4-5|
|3D|6-8|
|10D|20|
|100D|200|

---

### Метод 3: Grid Search

Перебрати різні комбінації та оцінити метриками:

```python
from sklearn.metrics import silhouette_score

best_score = -1
best_params = None

for eps in np.arange(0.1, 2.0, 0.1):
    for min_samples in range(3, 10):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Пропустити, якщо всі точки noise або один кластер
        if len(set(labels)) <= 1 or len(set(labels)) == len(X):
            continue
        
        # Silhouette (виключаємо noise = -1)
        mask = labels != -1
        if sum(mask) > 0:
            score = silhouette_score(X[mask], labels[mask])
            
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

print(f"Best: {best_params}, Silhouette: {best_score:.3f}")
```

---

## Метрики оцінки DBSCAN

### 1. Silhouette Score

**Як для K-Means**, але:

- Виключаємо точки з міткою -1 (noise)
- Рахуємо тільки для кластеризованих точок

```python
from sklearn.metrics import silhouette_score

labels = dbscan.fit_predict(X)

# Виключити noise
mask = labels != -1
if sum(mask) > 1:
    score = silhouette_score(X[mask], labels[mask])
else:
    score = -1  # Немає кластерів
```

---

### 2. Кількість кластерів

```python
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Кластерів: {n_clusters}")
print(f"Noise точок: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
```

**Що добре:**

- n_clusters > 0 (знайшли кластери)
- n_noise < 5% (мало викидів)

**Що погано:**

- n_clusters = 0 (всі точки noise) → ε занадто малий
- n_clusters = 1 AND n_noise ≈ 0 (всі в одному) → ε занадто великий

---

### 3. Davies-Bouldin Index

Як для K-Means (менше = краще):

```python
from sklearn.metrics import davies_bouldin_score

mask = labels != -1
if sum(mask) > 0 and len(set(labels[mask])) > 1:
    db_score = davies_bouldin_score(X[mask], labels[mask])
```

---

### 4. Calinski-Harabasz Index

Як для K-Means (більше = краще):

```python
from sklearn.metrics import calinski_harabasz_score

mask = labels != -1
if sum(mask) > 0 and len(set(labels[mask])) > 1:
    ch_score = calinski_harabasz_score(X[mask], labels[mask])
```

---

## Порівняння DBSCAN vs K-Means

|Аспект|K-Means|DBSCAN|
|---|---|---|
|**Форма кластерів**|Тільки сферичні|Довільна форма ✓|
|**Кількість кластерів**|Потрібно задати K|Визначається автоматично ✓|
|**Викиди**|Кожна точка в кластері|Виявляє як noise ✓|
|**Різні розміри**|Погано працює|Добре працює ✓|
|**Швидкість**|O(n·K·d·t) — швидко|O(n²) або O(n log n) — повільніше|
|**Параметри**|K (інтуїтивно)|ε, MinPts (складно підібрати) ✗|
|**Детермінованість**|Ні (залежить від ініціалізації)|Так (завжди однаковий результат) ✓|
|**Високі розмірності**|Працює добре|Погано (curse of dimensionality) ✗|

---

## Коли використовувати DBSCAN

### ✅ Ідеально підходить

1. **Кластери складної форми**
    
    - Кільця, спіралі, "банани"
    - Географічні регіони
2. **Невідома кількість кластерів**
    
    - Exploratory data analysis
    - Не знаєш, скільки груп має бути
3. **Наявність викидів**
    
    - Важливо їх виявити
    - Fraud detection
4. **Різна щільність НЕ КРИТИЧНА**
    
    - Якщо кластери приблизно однакової щільності

**Приклади:**

- Сегментація клієнтів за геолокацією
- Виявлення аномалій у мережевому трафіку
- Кластеризація генів за експресією

---

### ❌ Краще використати інше

1. **Кластери різної щільності**
    
    - DBSCAN поганий вибір
    - Альтернатива: HDBSCAN, OPTICS
2. **Високорозмірні дані (D > 20)**
    
    - Всі точки стають "далекими"
    - Альтернатива: K-Means, GMM після PCA
3. **Потрібна швидкість на великих даних**
    
    - DBSCAN повільний (O(n²) без оптимізацій)
    - Альтернатива: MiniBatchKMeans
4. **Потрібні центри кластерів**
    
    - DBSCAN не має поняття центру
    - Альтернатива: K-Means, GMM

---

## Варіанти та розширення DBSCAN

### 1. HDBSCAN (Hierarchical DBSCAN)

**Що покращує:**

- Працює з кластерами **різної щільності** ✓
- Автоматично вибирає ε для кожного кластера
- Стабільніший на різних даних

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(X)
```

**Коли використовувати:** Якщо DBSCAN не працює через різну щільність.

---

### 2. OPTICS

**Що це:**

- Узагальнення DBSCAN
- Будує порядок обробки точок
- Можна візуалізувати "reachability plot"

**Перевага:**

- Не потрібно задавати ε!
- Можна витягти кластери для різних ε з одного запуску

```python
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=5, xi=0.05)
labels = optics.fit_predict(X)
```

---

### 3. ST-DBSCAN (Spatial-Temporal)

**Для просторово-часових даних:**

- Враховує як просторову, так і часову близькість
- Приклад: кластеризація подій (де і коли сталися)

---

## Переваги та недоліки DBSCAN

### ✅ Переваги

|Перевага|Пояснення|
|---|---|
|**Довільна форма**|Знаходить кластери будь-якої форми|
|**Автоматичний K**|Не потрібно заздалегідь знати кількість|
|**Виявлення викидів**|Автоматично позначає noise|
|**Детермінованість**|Завжди однаковий результат (без randomness)|
|**Один прохід**|Не потребує ітерацій (як K-Means)|

---

### ❌ Недоліки

|Недолік|Пояснення|
|---|---|
|**Складний підбір параметрів**|ε і MinPts важко вибрати|
|**Різна щільність**|Погано працює, якщо кластери дуже різної щільності|
|**Висока розмірність**|Curse of dimensionality (D > 20)|
|**Повільний**|O(n²) без оптимізацій (дерева пошуку → O(n log n))|
|**Немає центрів**|Не має поняття "центр кластера"|

---

## Складність алгоритму

### Час виконання

**Наївна реалізація:**

- O(n²) — для кожної точки перевіряємо всі інші

**З spatial indexing (KD-tree, Ball-tree):**

- O(n log n) — для низькорозмірних даних (D < 20)
- O(n²) — для високорозмірних (дерева неефективні)

### Пам'ять

- O(n) — зберігаємо мітки кластерів

---

## Практичні поради

### 1. Масштабуй дані!

DBSCAN чутливий до масштабу ознак:

```python
from sklearn.preprocessing import StandardScaler

# ПОГАНО (якщо ознаки різного масштабу)
dbscan = DBSCAN(eps=0.5)
dbscan.fit(X)

# ДОБРЕ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.5)
dbscan.fit(X_scaled)
```

---

### 2. Почни з k-distance графіка

```python
# 1. Побудувати k-distance график
from sklearn.neighbors import NearestNeighbors

k = 4  # або MinPts
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X)
distances, _ = neighbors.kneighbors(X)
distances = np.sort(distances[:, k-1])[::-1]

plt.plot(distances)
plt.ylabel(f'{k}-distance')
plt.xlabel('Points sorted by distance')
plt.show()

# 2. Знайти "коліно" візуально
# 3. Встановити eps ≈ значення в коліні
```

---

### 3. Експериментуй з MinPts

```python
# Спробувати різні MinPts
for min_pts in [3, 5, 10, 15]:
    dbscan = DBSCAN(eps=0.5, min_samples=min_pts)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"MinPts={min_pts}: Clusters={n_clusters}, Noise={n_noise}")
```

---

### 4. Візуалізуй результати

```python
import matplotlib.pyplot as plt

# Для 2D даних
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title(f'DBSCAN: {n_clusters} clusters, {n_noise} noise')
plt.colorbar(label='Cluster')
plt.show()

# Для багатовимірних — через PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN (PCA projection)')
plt.show()
```

---

### 5. Перевір на переобучення параметрів

```python
# Якщо підбираєш ε і MinPts через grid search на тих самих даних,
# перевір на новому датасеті або через cross-validation

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Підібрати на train
best_params = grid_search_dbscan(X_train)

# Застосувати на test
dbscan = DBSCAN(**best_params)
labels_test = dbscan.fit_predict(X_test)
```

---

## Ключові висновки

> DBSCAN — потужний алгоритм кластеризації на основі щільності, який знаходить кластери довільної форми та автоматично виявляє викиди.

**Основні принципи:**

- **Core points:** точки з ≥ MinPts сусідів в радіусі ε
- **Border points:** точки в околиці core, але самі не core
- **Noise points:** точки далеко від усіх core points
- **Кластер:** максимальний набір density-connected точок

**Формула успіху:**

1. Масштабувати дані (StandardScaler)
2. Побудувати k-distance графік для вибору ε
3. MinPts ≈ 2×D (розмірність)
4. Візуалізувати результати
5. Оцінити метриками (Silhouette, кількість кластерів/noise)

**Коли використовувати:**

- Кластери складної форми ✓
- Невідома кількість кластерів ✓
- Потрібно виявити викиди ✓

**Коли НЕ використовувати:**

- Кластери дуже різної щільності → HDBSCAN
- Високі розмірності (D > 20) → PCA + K-Means
- Потрібна швидкість на мільйонах точок → K-Means

---

## Наступні кроки

1. **HDBSCAN** — для різної щільності
2. **OPTICS** — для автоматичного вибору ε
3. **Mean Shift** — ще один density-based метод
4. **Spectral Clustering** — для складних форм через графи