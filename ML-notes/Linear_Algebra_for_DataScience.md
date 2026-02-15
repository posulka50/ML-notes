# Лінійна Алгебра для Data Science: Повний Гайд

## Зміст

1. [Вектори](#вектори)
2. [Матриці](#матриці)
3. [Матричні операції](#матричні-операції)
4. [Системи лінійних рівнянь](#системи-лінійних-рівнянь)
5. [Векторні простори](#векторні-простори)
6. [Власні значення та власні вектори](#власні-значення-та-власні-вектори)
7. [Сингулярне розкладання (SVD)](#сингулярне-розкладання-svd)
8. [Норми та відстані](#норми-та-відстані)
9. [Застосування в ML](#застосування-в-ml)

---

# Вектори

## Що це?

**Вектор** — це упорядкований набір чисел (координат). У Data Science вектор представляє точку в багатовимірному просторі або набір характеристик (features).

### Геометрична інтуїція

```
2D вектор v = [3, 2]:
    
    y
  4 |
  3 |
  2 |      • (3, 2)
  1 |     /
  0 |____/_______ x
    0  1  2  3  4
    
Стрілка з початку координат до точки (3, 2)
```

### Математичне означення

**Вектор-стовпець:**
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

**Вектор-рядок:**
$$\mathbf{v}^T = \begin{bmatrix} v_1 & v_2 & \cdots & v_n \end{bmatrix}$$

### В Data Science

**Приклад:** Характеристики будинку
```python
house = [120,    # площа (м²)
         3,      # кількість кімнат
         2010,   # рік побудови
         500000] # ціна ($)
```

Це 4-вимірний вектор: $\mathbf{x} = \begin{bmatrix} 120 \\ 3 \\ 2010 \\ 500000 \end{bmatrix}$

---

## Операції з векторами

### 1. Додавання векторів

**Геометрично:** Правило паралелограма

```
a = [2, 1]
b = [1, 2]

    y
  3 |     b+a
  2 |    •
  1 |   /|\
  0 |__/_|_\__ x
    0  1  2  3
    
a + b = [2+1, 1+2] = [3, 3]
```

**Математично:**
$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

**Код:**
```python
import numpy as np

a = np.array([2, 1])
b = np.array([1, 2])
c = a + b

print(f"a + b = {c}")  # [3 3]
```

### 2. Множення на скаляр

**Геометрично:** Розтягування/стиснення

```
v = [2, 1]
2v = [4, 2]

    y
  2 |    2v
  1 | v •
  0 |__•______ x
    0  2  4
    
Множення на 2 → подвоює довжину
```

**Математично:**
$$\alpha \mathbf{v} = \alpha \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = \begin{bmatrix} \alpha v_1 \\ \alpha v_2 \\ \vdots \\ \alpha v_n \end{bmatrix}$$

**Код:**
```python
v = np.array([2, 1])
scaled = 2 * v

print(f"2v = {scaled}")  # [4 2]
```

### 3. Скалярний добуток (Dot Product)

**Що це:** Міра "схожості" напрямків векторів.

**Математично:**
$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n = \sum_{i=1}^{n} a_i b_i$$

**Геометрично:**
$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$$

де $\theta$ — кут між векторами.

**Інтуїція:**
```
Якщо a · b > 0  →  Вектори в одному напрямку
Якщо a · b = 0  →  Вектори перпендикулярні (ортогональні)
Якщо a · b < 0  →  Вектори в протилежних напрямках
```

**Приклад:**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Скалярний добуток
dot_product = np.dot(a, b)
# або
dot_product = a @ b
# або
dot_product = (a * b).sum()

print(f"a · b = {dot_product}")  # 1*4 + 2*5 + 3*6 = 32
```

**Застосування в ML:**
- **Similarity:** Подібність документів (cosine similarity)
- **Predictions:** Лінійна регресія $\hat{y} = \mathbf{w} \cdot \mathbf{x}$
- **Neural Networks:** Weighted sum inputs

### 4. Довжина вектора (Норма)

**Евклідова норма (L2):**
$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}$$

**Геометрично:** Відстань від початку координат до точки.

```python
v = np.array([3, 4])

# Норма
norm = np.linalg.norm(v)
# або
norm = np.sqrt((v ** 2).sum())

print(f"||v|| = {norm}")  # sqrt(9 + 16) = 5
```

**Нормалізація (одиничний вектор):**
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

```python
v = np.array([3, 4])
v_normalized = v / np.linalg.norm(v)

print(f"Normalized: {v_normalized}")  # [0.6, 0.8]
print(f"Norm: {np.linalg.norm(v_normalized)}")  # 1.0
```

**Застосування:**
- **Feature scaling:** Нормалізація features
- **Cosine similarity:** Порівняння векторів незалежно від довжини

### 5. Відстань між векторами

**Евклідова відстань:**
$$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\| = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Відстань
distance = np.linalg.norm(a - b)
# або з scipy
from scipy.spatial.distance import euclidean
distance = euclidean(a, b)

print(f"Distance: {distance}")  # sqrt(27) ≈ 5.196
```

### 6. Cosine Similarity

**Що це:** Косинус кута між векторами (не залежить від довжини).

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \cos(\theta)$$

**Діапазон:** [-1, 1]
- 1 → Однаковий напрямок
- 0 → Перпендикулярні
- -1 → Протилежні

```python
from sklearn.metrics.pairwise import cosine_similarity

a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])

similarity = cosine_similarity(a, b)[0, 0]

# або вручну
similarity = np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Cosine similarity: {similarity}")  # 0.974
```

**Застосування в NLP:**
```python
# Порівняння документів (TF-IDF vectors)
doc1 = np.array([0.2, 0.5, 0.3, 0.0])  # TF-IDF вектор
doc2 = np.array([0.1, 0.6, 0.2, 0.1])

similarity = cosine_similarity([doc1], [doc2])[0, 0]
print(f"Document similarity: {similarity}")
```

---

## Простір векторів (Vector Space)

### Лінійна комбінація

**Що це:** Сума векторів, помножених на скаляри.

$$\mathbf{v} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_n \mathbf{v}_n$$

**Приклад:**
```python
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Будь-який вектор в 2D можна виразити через v1 та v2
v = 3 * v1 + 2 * v2  # [3, 2]
```

### Лінійна незалежність

**Вектори лінійно незалежні**, якщо жоден не можна виразити через інші.

```python
# Лінійно незалежні
v1 = np.array([1, 0])
v2 = np.array([0, 1])
# v2 ≠ α * v1 для будь-якого α

# Лінійно залежні
v1 = np.array([1, 2])
v2 = np.array([2, 4])  # v2 = 2 * v1
```

**Перевірка через визначник (для 2 векторів в 2D):**
```python
v1 = np.array([1, 2])
v2 = np.array([3, 4])

# Побудувати матрицю з векторів-стовпців
A = np.column_stack([v1, v2])

# Визначник
det = np.linalg.det(A)

if det != 0:
    print("Лінійно незалежні")
else:
    print("Лінійно залежні")
```

### Базис

**Базис** — мінімальний набір лінійно незалежних векторів, через які можна виразити будь-який вектор простору.

**Стандартний базис R²:**
```python
e1 = np.array([1, 0])
e2 = np.array([0, 1])

# Будь-який вектор v = [a, b] можна виразити:
# v = a*e1 + b*e2
```

**Розмірність** = кількість векторів у базисі.

---

## Проекції

### Проекція вектора на вектор

**Що це:** "Тінь" одного вектора на іншому.

$$\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} \mathbf{b} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b}$$

**Геометрія:**
```
        a
       /|
      / |
     /  | (projection)
    /   |
   /    |
  /_____|
     b
```

```python
a = np.array([3, 4])
b = np.array([1, 0])

# Проекція a на b
projection = (np.dot(a, b) / np.dot(b, b)) * b

print(f"Projection: {projection}")  # [3, 0]
```

**Застосування:**
- **PCA:** Проекція даних на principal components
- **Regression:** Orthogonal projection

---

# Матриці

## Що це?

**Матриця** — це прямокутна таблиця чисел.

$$A = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

**Розмір:** $m \times n$ (m рядків, n стовпців)

### В Data Science

**Dataset як матриця:**
```python
# Кожен рядок = зразок (sample)
# Кожен стовпець = ознака (feature)

data = np.array([
    [120, 3, 2010],  # Будинок 1
    [80,  2, 2015],  # Будинок 2
    [150, 4, 2005],  # Будинок 3
])

print(f"Shape: {data.shape}")  # (3, 3) = 3 samples × 3 features
```

**Матриця = стек векторів:**
```python
# Вектори-рядки (row vectors)
sample1 = data[0]  # [120, 3, 2010]

# Вектори-стовпці (column vectors)
feature1 = data[:, 0]  # [120, 80, 150] (площа)
```

---

## Типи матриць

### 1. Квадратна матриця

**n × n:**
```python
A = np.array([
    [1, 2],
    [3, 4]
])
# 2×2 квадратна
```

### 2. Діагональна матриця

**Ненульові тільки на діагоналі:**
$$D = \begin{bmatrix}
d_1 & 0 & 0 \\
0 & d_2 & 0 \\
0 & 0 & d_3
\end{bmatrix}$$

```python
D = np.diag([2, 3, 5])
print(D)
# [[2 0 0]
#  [0 3 0]
#  [0 0 5]]
```

### 3. Одинична матриця (Identity)

**Діагональна з одиницями:**
$$I = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}$$

**Властивість:** $A \cdot I = I \cdot A = A$

```python
I = np.eye(3)
print(I)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

A = np.array([[1, 2], [3, 4]])
print(A @ np.eye(2))  # Same as A
```

### 4. Симетрична матриця

**$A = A^T$:**
$$A = \begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 5 \\
3 & 5 & 6
\end{bmatrix}$$

```python
A = np.array([
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 6]
])

print(np.array_equal(A, A.T))  # True
```

**Застосування:**
- **Коваріаційні матриці** завжди симетричні
- **Kernel matrices** в ML

### 5. Ортогональна матриця

**$A^T A = A A^T = I$:**

Стовпці (та рядки) — ортонормовані вектори.

```python
# Приклад: матриця обертання
theta = np.pi / 4  # 45 градусів
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Перевірка
print(R @ R.T)  # ≈ I
```

**Властивості:**
- Зберігають довжину векторів
- Зберігають кути
- Обертання, відображення

---

## Транспонування

**Що це:** Поміняти рядки та стовпці місцями.

$$A^T_{ij} = A_{ji}$$

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

A_T = A.T
print(A_T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

**Властивості:**
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(AB)^T = B^T A^T$

---

# Матричні операції

## Множення матриць

### Математично

**$C = AB$:**

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

**Розміри:**
- $A: m \times n$
- $B: n \times p$
- $C: m \times p$

**Важливо:** Кількість стовпців $A$ = кількість рядків $B$!

### Геометрична інтуїція

**Матриця = лінійне перетворення:**

```python
# Матриця розтягування
S = np.array([
    [2, 0],
    [0, 3]
])

v = np.array([1, 1])
v_transformed = S @ v  # [2, 3]

# Розтягнули по x в 2 рази, по y в 3 рази
```

### Код

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

# Множення матриць
C = A @ B
# або
C = np.dot(A, B)
# або
C = np.matmul(A, B)

print(C)
# [[19 22]
#  [43 50]]
```

### Матриця × Вектор

**Дуже важливо в ML!**

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

x = np.array([1, 2, 3])

# Matrix-vector product
y = A @ x

print(y)  # [14, 32]

# y[0] = 1*1 + 2*2 + 3*3 = 14
# y[1] = 4*1 + 5*2 + 6*3 = 32
```

**Застосування:**
```python
# Лінійна регресія: y = Xw
X = np.array([
    [1, 2],  # sample 1
    [3, 4],  # sample 2
    [5, 6]   # sample 3
])

w = np.array([0.5, 0.3])  # weights

predictions = X @ w
print(predictions)  # [1.1, 2.7, 4.3]
```

### Властивості

**НЕ комутативне:**
$$AB \neq BA$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)
# [[19 22]
#  [43 50]]

print(B @ A)
# [[23 34]
#  [31 46]]

# Different!
```

**Асоціативне:**
$$(AB)C = A(BC)$$

**Дистрибутивне:**
$$A(B + C) = AB + AC$$

---

## Визначник (Determinant)

**Тільки для квадратних матриць!**

### Що це?

**Геометрично:** Масштабування об'єму при лінійному перетворенні.

**Для 2×2:**
$$\det(A) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$$

**Для 3×3:**
$$\det(A) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

### Код

```python
A = np.array([
    [1, 2],
    [3, 4]
])

det = np.linalg.det(A)
print(f"Determinant: {det}")  # -2.0
```

### Інтерпретація

```python
# Якщо det(A) = 0  →  Матриця виродженна (singular)
#                      Не має оберненої
#                      Вектори лінійно залежні

A = np.array([
    [1, 2],
    [2, 4]  # другий рядок = 2 × перший
])

print(np.linalg.det(A))  # 0.0

# Якщо det(A) ≠ 0  →  Матриця невиродженна
#                      Має обернену
#                      Вектори лінійно незалежні
```

**Властивості:**
- $\det(AB) = \det(A) \det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = 1/\det(A)$

---

## Обернена матриця

**Що це:** Матриця $A^{-1}$ така, що:

$$A A^{-1} = A^{-1} A = I$$

**Існує тільки якщо** $\det(A) \neq 0$!

### Для 2×2

$$A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

де $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$

### Код

```python
A = np.array([
    [1, 2],
    [3, 4]
])

# Обернена матриця
A_inv = np.linalg.inv(A)

print("A^-1:")
print(A_inv)

# Перевірка
print("\nA @ A^-1:")
print(A @ A_inv)  # ≈ I
```

### Розв'язання систем рівнянь

**$Ax = b$ → $x = A^{-1}b$**

```python
A = np.array([
    [2, 1],
    [1, 3]
])

b = np.array([5, 6])

# Розв'язок
x = np.linalg.inv(A) @ b
# АБО краще (більш стабільно чисельно):
x = np.linalg.solve(A, b)

print(f"Solution: {x}")  # [1.8, 1.4]

# Перевірка
print(f"A @ x = {A @ x}")  # ≈ b
```

**Застосування:**
- Розв'язання лінійних систем
- Лінійна регресія: $w = (X^T X)^{-1} X^T y$

---

## Ранг матриці

**Що це:** Кількість лінійно незалежних рядків (або стовпців).

**Максимальний ранг = min(m, n)** для матриці m×n.

### Код

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]  # лінійно залежний від перших двох
])

rank = np.linalg.matrix_rank(A)
print(f"Rank: {rank}")  # 2 (а не 3!)

# Повний ранг
B = np.array([
    [1, 2],
    [3, 4]
])

print(np.linalg.matrix_rank(B))  # 2 (повний ранг)
```

**Інтерпретація:**
- **Ранг < min(m,n)** → Лінійна залежність
- **Ранг = min(m,n)** → Повний ранг

**Застосування:**
- Перевірка лінійної незалежності features
- Dimensionality reduction

---

# Системи лінійних рівнянь

## Матричне представлення

**Система:**
$$\begin{cases}
2x + 3y = 8 \\
4x + 5y = 14
\end{cases}$$

**Матрична форма: $Ax = b$**

$$\begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 \\ 14 \end{bmatrix}$$

### Розв'язання

```python
A = np.array([
    [2, 3],
    [4, 5]
])

b = np.array([8, 14])

# Метод 1: Обернена матриця (не рекомендовано!)
x = np.linalg.inv(A) @ b

# Метод 2: np.linalg.solve (краще!)
x = np.linalg.solve(A, b)

print(f"Solution: x = {x[0]}, y = {x[1]}")  # x=1, y=2

# Перевірка
print(f"Check: A @ x = {A @ x}")  # [8, 14]
```

## Переобумовлені системи (Overdetermined)

**Більше рівнянь ніж невідомих** (m > n):

```python
# 3 рівняння, 2 невідомих
A = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])

b = np.array([2, 3, 5])

# Немає точного розв'язку!
# Знайти найкраще наближення (least squares)

x = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"Best fit: {x}")

# Залишок (residual)
residual = A @ x - b
print(f"Residual: {residual}")
```

**Це основа лінійної регресії!**

---

# Векторні простори

## Підпростір

**Підпростір** векторного простору — це підмножина, яка також є векторним простором.

**Приклад в R³:**
- Лінія через початок координат → 1D підпростір
- Площина через початок → 2D підпростір

### Column Space (Простір стовпців)

**Що це:** Всі можливі лінійні комбінації стовпців матриці.

$$\text{Col}(A) = \{\mathbf{y} : \mathbf{y} = A\mathbf{x} \text{ for some } \mathbf{x}\}$$

**Розмірність = ранг матриці.**

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# Простір стовпців має розмірність 2
# (всі лінійні комбінації 2 стовпців в R³)

rank = np.linalg.matrix_rank(A)
print(f"Dimension of Col(A): {rank}")  # 2
```

### Null Space (Нуль-простір)

**Що це:** Всі вектори $\mathbf{x}$ такі, що $A\mathbf{x} = \mathbf{0}$.

```python
from scipy.linalg import null_space

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Знайти null space
null = null_space(A)
print("Null space basis:")
print(null)

# Перевірка
print("\nA @ null ≈ 0:")
print(A @ null)
```

---

# Власні значення та власні вектори

## Що це?

**Власний вектор** $\mathbf{v}$ матриці $A$ — це вектор, що тільки масштабується (не обертається) при множенні на $A$:

$$A\mathbf{v} = \lambda \mathbf{v}$$

де $\lambda$ — **власне значення** (eigenvalue).

### Геометрична інтуїція

```
Звичайний вектор:
    A @ u
     ↗
    u → A обертає та масштабує

Власний вектор:
    A @ v = λv
    ↑
    v → A тільки масштабує (в тому ж напрямку)
```

### Приклад

```python
A = np.array([
    [2, 0],
    [0, 3]
])

# Власні значення та вектори
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)     # [2, 3]
print("Eigenvectors:\n", eigenvectors)

# Перевірка для першого власного вектора
v = eigenvectors[:, 0]
lambda_val = eigenvalues[0]

print(f"\nA @ v = {A @ v}")
print(f"λ * v = {lambda_val * v}")
# Однакові!
```

### Діагональна матриця

**Для діагональної матриці:**
- Власні значення = діагональні елементи
- Власні вектори = стандартний базис

```python
D = np.array([
    [5, 0, 0],
    [0, 3, 0],
    [0, 0, 7]
])

eigenvalues, eigenvectors = np.linalg.eig(D)

print("Eigenvalues:", eigenvalues)  # [5, 3, 7]
print("Eigenvectors:\n", eigenvectors)
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]
```

---

## Власне розкладання (Eigendecomposition)

**Для симетричної матриці:**

$$A = Q \Lambda Q^T$$

де:
- $Q$ — матриця власних векторів (ортогональних)
- $\Lambda$ — діагональна матриця власних значень

```python
# Симетрична матриця
A = np.array([
    [4, 2],
    [2, 3]
])

# Власні значення та вектори
eigenvalues, eigenvectors = np.linalg.eig(A)

# Створити Λ (Lambda)
Lambda = np.diag(eigenvalues)

# Реконструкція: A = Q Λ Q^T
Q = eigenvectors
A_reconstructed = Q @ Lambda @ Q.T

print("Original A:")
print(A)
print("\nReconstructed A:")
print(A_reconstructed)
# Однакові!
```

### Застосування

**1. PCA (Principal Component Analysis):**
```python
from sklearn.decomposition import PCA

# Дані
X = np.random.randn(100, 5)

# PCA знаходить власні вектори коваріаційної матриці
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# Власні вектори = principal components
print("Principal components (eigenvectors):")
print(pca.components_)

# Власні значення = explained variance
print("\nExplained variance (eigenvalues):")
print(pca.explained_variance_)
```

**2. Швидке обчислення степенів матриці:**

$$A^n = Q \Lambda^n Q^T$$

```python
# A^100 через власне розкладання
eigenvalues, eigenvectors = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)
Q = eigenvectors

# A^100
Lambda_100 = np.diag(eigenvalues ** 100)
A_100 = Q @ Lambda_100 @ Q.T

# Порівняти з прямим обчисленням (повільно!)
# A_100_direct = np.linalg.matrix_power(A, 100)
```

---

# Сингулярне розкладання (SVD)

## Що це?

**SVD (Singular Value Decomposition)** — розкладання будь-якої матриці:

$$A = U \Sigma V^T$$

де:
- $A$: $m \times n$ (вихідна матриця)
- $U$: $m \times m$ (ортогональна, left singular vectors)
- $\Sigma$: $m \times n$ (діагональна, singular values)
- $V^T$: $n \times n$ (ортогональна, right singular vectors)

### Візуалізація

```
A (m×n)  =  U (m×m)  ×  Σ (m×n)  ×  V^T (n×n)

[data]  =  [left]   ×  [scale]  ×  [right]
```

### Код

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# SVD
U, S, VT = np.linalg.svd(A, full_matrices=True)

print(f"U shape: {U.shape}")    # (3, 3)
print(f"S shape: {S.shape}")    # (3,) - тільки діагональ
print(f"VT shape: {VT.shape}")  # (3, 3)

# Singular values
print(f"\nSingular values: {S}")

# Реконструкція
Sigma = np.zeros((3, 3))
Sigma[:3, :3] = np.diag(S)

A_reconstructed = U @ Sigma @ VT

print("\nOriginal A:")
print(A)
print("\nReconstructed A:")
print(A_reconstructed)
```

---

## Зв'язок з власним розкладанням

**Для симетричної матриці:**
- SVD та eigendecomposition дають те саме
- Singular values = |eigenvalues|

**Загалом:**
- $A^T A$ має власні значення = $\sigma_i^2$ (квадрати singular values)
- $V$ = власні вектори $A^T A$
- $U$ = власні вектори $A A^T$

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Власні значення A^T A
eigenvalues_ATA = np.linalg.eigvalsh(A.T @ A)

print("Singular values:", S)
print("sqrt(eigenvalues of A^T A):", np.sqrt(eigenvalues_ATA[::-1]))
# Однакові!
```

---

## Застосування SVD

### 1. Dimensionality Reduction

**Truncated SVD** — зберегти тільки top k singular values:

```python
from sklearn.decomposition import TruncatedSVD

# Дані: 100 samples, 50 features
X = np.random.randn(100, 50)

# Зменшити до 10 компонентів
svd = TruncatedSVD(n_components=10)
X_reduced = svd.fit_transform(X)

print(f"Original shape: {X.shape}")      # (100, 50)
print(f"Reduced shape: {X_reduced.shape}")  # (100, 10)

# Пояснена variance
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
```

### 2. Image Compression

```python
from PIL import Image

# Завантажити зображення (grayscale)
img = Image.open('image.jpg').convert('L')
img_array = np.array(img, dtype=float)

print(f"Original shape: {img_array.shape}")

# SVD
U, S, VT = np.linalg.svd(img_array, full_matrices=False)

# Стискання: зберегти тільки top k singular values
k = 50

U_k = U[:, :k]
S_k = S[:k]
VT_k = VT[:k, :]

# Реконструкція
img_compressed = U_k @ np.diag(S_k) @ VT_k

# Compression ratio
original_size = img_array.size
compressed_size = U_k.size + S_k.size + VT_k.size
ratio = original_size / compressed_size

print(f"Compression ratio: {ratio:.2f}x")

# Зберегти
img_compressed_uint8 = np.clip(img_compressed, 0, 255).astype(np.uint8)
Image.fromarray(img_compressed_uint8).save('compressed.jpg')
```

### 3. Pseudo-inverse (Moore-Penrose)

**Для переобумовлених систем:**

$$A^+ = V \Sigma^+ U^T$$

де $\Sigma^+$ — pseudo-inverse діагональної матриці.

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# Pseudo-inverse через SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Σ^+ (обернені ненульові singular values)
S_inv = 1 / S
Sigma_plus = np.diag(S_inv)

A_pinv = VT.T @ Sigma_plus @ U.T

# Або просто:
A_pinv = np.linalg.pinv(A)

print("Pseudo-inverse:")
print(A_pinv)

# Застосування: least squares
b = np.array([1, 2, 3])
x = A_pinv @ b

print(f"\nSolution: {x}")
print(f"A @ x = {A @ x}")  # ≈ b (best approximation)
```

### 4. Recommender Systems

```python
# User-item matrix (ratings)
R = np.array([
    [5, 3, 0, 1],  # User 1
    [4, 0, 0, 1],  # User 2
    [1, 1, 0, 5],  # User 3
    [1, 0, 0, 4],  # User 4
])

# SVD (replace 0 with mean first in practice)
U, S, VT = np.linalg.svd(R, full_matrices=False)

# Зменшити розмірність
k = 2
U_k = U[:, :k]
S_k = S[:k]
VT_k = VT[:k, :]

# Передбачення всіх рейтингів
R_pred = U_k @ np.diag(S_k) @ VT_k

print("Predicted ratings:")
print(R_pred)
# Можна заповнити відсутні рейтинги!
```

---

# Норми та відстані

## Векторні норми

### L1 norm (Manhattan)

$$\|\mathbf{x}\|_1 = |x_1| + |x_2| + \cdots + |x_n| = \sum_{i=1}^{n} |x_i|$$

```python
x = np.array([3, -4, 5])

l1_norm = np.sum(np.abs(x))
# або
l1_norm = np.linalg.norm(x, ord=1)

print(f"L1 norm: {l1_norm}")  # 12
```

**Геометрично:** Відстань "міськими кварталами".

**Застосування:**
- Lasso regression (L1 regularization)
- Sparse models

### L2 norm (Euclidean)

$$\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2} = \sqrt{\sum_{i=1}^{n} x_i^2}$$

```python
x = np.array([3, 4])

l2_norm = np.sqrt(np.sum(x ** 2))
# або
l2_norm = np.linalg.norm(x)  # default ord=2

print(f"L2 norm: {l2_norm}")  # 5.0
```

**Геометрично:** Пряма відстань.

**Застосування:**
- Ridge regression (L2 regularization)
- Euclidean distance

### L∞ norm (Maximum)

$$\|\mathbf{x}\|_\infty = \max(|x_1|, |x_2|, \ldots, |x_n|)$$

```python
x = np.array([3, -7, 2])

linf_norm = np.max(np.abs(x))
# або
linf_norm = np.linalg.norm(x, ord=np.inf)

print(f"L∞ norm: {linf_norm}")  # 7
```

### Порівняння норм

```python
x = np.array([3, 4])

print(f"L1 norm: {np.linalg.norm(x, 1)}")    # 7
print(f"L2 norm: {np.linalg.norm(x, 2)}")    # 5
print(f"L∞ norm: {np.linalg.norm(x, np.inf)}")  # 4

# L1 ≥ L2 ≥ L∞ (для одиничних векторів)
```

---

## Матричні норми

### Frobenius norm

$$\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}^2}$$

**Аналог L2 norm для матриць.**

```python
A = np.array([
    [1, 2],
    [3, 4]
])

frob_norm = np.linalg.norm(A, 'fro')
# або
frob_norm = np.sqrt(np.sum(A ** 2))

print(f"Frobenius norm: {frob_norm}")  # sqrt(30) ≈ 5.48
```

### Spectral norm (2-norm)

**Найбільше singular value:**

$$\|A\|_2 = \sigma_{\max}(A)$$

```python
A = np.array([
    [1, 2],
    [3, 4]
])

spectral_norm = np.linalg.norm(A, 2)

# Перевірка через SVD
_, S, _ = np.linalg.svd(A)
print(f"Spectral norm: {spectral_norm}")  # max singular value
print(f"Max singular value: {S[0]}")
```

---

## Відстані між векторами

### Euclidean distance

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

```python
from scipy.spatial.distance import euclidean

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

dist = euclidean(x, y)
# або
dist = np.linalg.norm(x - y)

print(f"Euclidean distance: {dist}")  # sqrt(27) ≈ 5.196
```

### Manhattan distance

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1 = \sum_{i=1}^{n} |x_i - y_i|$$

```python
from scipy.spatial.distance import cityblock

dist = cityblock(x, y)
# або
dist = np.sum(np.abs(x - y))

print(f"Manhattan distance: {dist}")  # 9
```

### Cosine distance

$$d(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$$

```python
from scipy.spatial.distance import cosine

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

dist = cosine(x, y)

print(f"Cosine distance: {dist}")  # 1 - cosine_similarity
```

---

# Застосування в Machine Learning

## 1. Лінійна регресія

**Модель:** $y = \mathbf{w}^T \mathbf{x} + b$

**Матрична форма:** $\mathbf{y} = X\mathbf{w}$

**Розв'язок (Normal Equation):**
$$\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$

```python
from sklearn.linear_model import LinearRegression

# Дані
X = np.array([
    [1, 1],
    [1, 2],
    [2, 2],
    [2, 3]
])

y = np.array([1, 2, 2, 3])

# Ручний розрахунок
XTX_inv = np.linalg.inv(X.T @ X)
w = XTX_inv @ X.T @ y

print(f"Weights (manual): {w}")

# Через sklearn
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

print(f"Weights (sklearn): {model.coef_}")
```

**Чому працює:** Мінімізує $\|X\mathbf{w} - \mathbf{y}\|^2$ (L2 loss).

---

## 2. PCA (Principal Component Analysis)

**Що робить:**
1. Обчислює коваріаційну матрицю: $C = \frac{1}{n} X^T X$
2. Знаходить власні вектори $C$
3. Проектує дані на top k власних векторів

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Генерація даних
np.random.seed(42)
X = np.random.randn(100, 5)

# Scaling (важливо!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Original shape: {X.shape}")      # (100, 5)
print(f"PCA shape: {X_pca.shape}")       # (100, 2)

# Власні вектори (principal components)
print("\nPrincipal Components:")
print(pca.components_)

# Власні значення (explained variance)
print("\nExplained variance:")
print(pca.explained_variance_)
print(f"Total: {pca.explained_variance_ratio_.sum():.2%}")
```

**Ручний розрахунок:**
```python
# 1. Центрування
X_centered = X_scaled - X_scaled.mean(axis=0)

# 2. Коваріаційна матриця
Cov = (X_centered.T @ X_centered) / (len(X) - 1)

# 3. Власні значення та вектори
eigenvalues, eigenvectors = np.linalg.eigh(Cov)

# Відсортувати за спаданням
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 4. Проекція
X_pca_manual = X_centered @ eigenvectors[:, :2]

print("\nManual PCA:")
print(X_pca_manual[:5])
print("\nsklearn PCA:")
print(X_pca[:5])
# Однакові (з точністю до знаку)!
```

---

## 3. Cosine Similarity (NLP)

**Порівняння документів:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Документи
documents = [
    "machine learning is great",
    "deep learning is amazing",
    "cats and dogs are animals"
]

# TF-IDF векторизація
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Cosine similarity
similarity_matrix = cosine_similarity(X)

print("Similarity matrix:")
print(similarity_matrix)
# [[1.   0.49 0.  ]   Doc 0 vs Doc 0, 1, 2
#  [0.49 1.   0.  ]   Doc 1 vs Doc 0, 1, 2
#  [0.   0.   1.  ]]  Doc 2 vs Doc 0, 1, 2

# Документи 0 та 1 схожі (0.49)
# Документ 2 не схожий на інші (0.0)
```

---

## 4. Regularization

### L1 (Lasso)

**Мінімізує:** $\|X\mathbf{w} - \mathbf{y}\|^2 + \alpha \|\mathbf{w}\|_1$

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# L1 створює sparse вектор w (багато нулів)
print(f"Non-zero weights: {np.sum(model.coef_ != 0)}")
```

### L2 (Ridge)

**Мінімізує:** $\|X\mathbf{w} - \mathbf{y}\|^2 + \alpha \|\mathbf{w}\|_2^2$

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# L2 робить weights маленькими, але не нульовими
```

---

## 5. Distance-based methods

### K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

# KNN використовує Euclidean distance (за замовчуванням)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Для нової точки знаходить 5 найближчих (за L2 norm)
y_pred = knn.predict(X_test)
```

### K-Means Clustering

```python
from sklearn.cluster import KMeans

# K-Means мінімізує відстані до центроїдів
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Центроїди
print("Centroids:")
print(kmeans.cluster_centers_)

# Відстань = Euclidean
```

---

## 6. Neural Networks

**Forward pass:**
$$\mathbf{h} = \sigma(W \mathbf{x} + \mathbf{b})$$

**Матричне множення всюди!**

```python
import torch
import torch.nn as nn

# Простий feedforward layer
layer = nn.Linear(in_features=10, out_features=5)

# Внутрішньо: y = W @ x + b
x = torch.randn(32, 10)  # batch size 32
y = layer(x)             # (32, 5)

print(f"Weight matrix shape: {layer.weight.shape}")  # (5, 10)
print(f"Output shape: {y.shape}")  # (32, 5)
```

---

## Практичні поради 💡

### 1. Завжди перевіряй розміри (shape)

```python
# ❌ Неправильно
A = np.array([[1, 2, 3]])  # (1, 3)
B = np.array([[4, 5, 6]])  # (1, 3)
# C = A @ B  # ValueError! (1,3) @ (1,3) не працює

# ✅ Правильно
C = A @ B.T  # (1, 3) @ (3, 1) = (1, 1)
```

### 2. Використовуй broadcasting розумно

```python
# Broadcasting дозволяє операції з різними shapes
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)

b = np.array([10, 20, 30])  # (3,)

# Додавання: broadcast b до кожного рядка A
C = A + b  # (2, 3)
print(C)
# [[11 22 33]
#  [14 25 36]]
```

### 3. Уникай обернених матриць коли можливо

```python
# ❌ Повільно та нестабільно
x = np.linalg.inv(A) @ b

# ✅ Краще
x = np.linalg.solve(A, b)
```

### 4. Перевіряй умови існування

```python
A = np.array([[1, 2], [2, 4]])

# Перевірити чи можна інвертувати
det = np.linalg.det(A)
if abs(det) < 1e-10:
    print("Matrix is singular! Cannot invert.")
else:
    A_inv = np.linalg.inv(A)
```

### 5. Scaling для numerical stability

```python
# Для великих чисел
X = np.array([[1000, 2000], [3000, 4000]])

# Scale before operations
X_scaled = X / 1000  # [1-4] range
# Compute
# Then scale back
```

---

## Шпаргалка операцій

```python
import numpy as np

# Створення
A = np.array([[1, 2], [3, 4]])
I = np.eye(3)                    # Одинична 3×3
Z = np.zeros((2, 3))             # Нулі 2×3
D = np.diag([1, 2, 3])           # Діагональна

# Операції
A + B                            # Додавання
A - B                            # Віднімання
A * B                            # Element-wise множення
A @ B                            # Матричне множення
A.T                              # Транспонування
np.linalg.inv(A)                 # Обернена
np.linalg.det(A)                 # Визначник
np.linalg.matrix_rank(A)         # Ранг

# Векторні операції
np.dot(a, b)                     # Скалярний добуток
np.linalg.norm(v)                # Норма (L2)
np.linalg.norm(v, 1)             # L1 норма
np.linalg.norm(v, np.inf)        # L∞ норма

# Розкладання
eigenvalues, eigenvectors = np.linalg.eig(A)     # Власні
U, S, VT = np.linalg.svd(A)                      # SVD

# Розв'язання систем
x = np.linalg.solve(A, b)        # Ax = b
x = np.linalg.lstsq(A, b)[0]     # Least squares

# Корисні функції
A.shape                          # Розмір
A.reshape(3, 2)                  # Reshape
A.flatten()                      # В 1D
np.concatenate([A, B])           # Склеїти
np.column_stack([v1, v2])        # Вектори в матрицю
```

---

## Ресурси для поглибленого вивчення

### Книги
- **"Linear Algebra and Its Applications" by Gilbert Strang** — класика
- **"Introduction to Linear Algebra" by Gilbert Strang** — більш доступна
- **"No Bullshit Guide to Linear Algebra" by Ivan Savov** — практична

### Відео курси
- **3Blue1Brown - Essence of Linear Algebra** (YouTube) — найкраща візуалізація!
- **Gilbert Strang's MIT 18.06** — повний курс
- **Khan Academy - Linear Algebra** — step-by-step

### Інтерактивні ресурси
- **Matrix Calculus** (matrixcalculus.org) — обчислення похідних матриць
- **Seeing Theory** — візуалізації

---

## Ключові висновки

> Лінійна алгебра — це мова Data Science. Все в ML зводиться до операцій з векторами та матрицями.

**Фундаментальні концепції:**
1. **Вектори** — точки/напрямки в просторі
2. **Матриці** — лінійні перетворення
3. **Скалярний добуток** — similarity
4. **Норми** — довжина/відстань
5. **Власні вектори** — напрямки, що зберігаються при перетворенні

**Ключові операції:**
- Матричне множення: $C = AB$
- Транспонування: $A^T$
- Обернена: $A^{-1}$ (якщо існує)
- Визначник: $\det(A)$
- Ранг: лінійна незалежність

**Розкладання:**
- **Eigendecomposition:** $A = Q\Lambda Q^T$ (симетричні матриці)
- **SVD:** $A = U\Sigma V^T$ (будь-які матриці)

**В Machine Learning:**
- **Regression:** $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$
- **PCA:** Власні вектори коваріаційної матриці
- **Neural Networks:** Матричні множення всюди
- **Similarity:** Cosine similarity, distances
- **Regularization:** L1/L2 norms

**Найважливіше:**
- Розумій геометричну інтуїцію
- Перевіряй розміри матриць (shape)
- NumPy — твій найкращий друг
- Практикуйся на реальних даних

---

#math #linear-algebra #vectors #matrices #eigenvalues #svd #ml-fundamentals #data-science #numpy
