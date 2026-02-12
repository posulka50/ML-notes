# Train-Test Split (Розділення на тренувальний та тестовий набори)

## Що це?

**Train-Test Split** — це процес **розділення датасету на дві частини**: одна для **навчання** моделі (train set), друга для **оцінки** її продуктивності на нових даних (test set).

**Головна ідея:** модель навчається на train set, а оцінюється на test set, який вона **ніколи не бачила** під час навчання. Це дозволяє об'єктивно оцінити здатність моделі **узагальнювати** на нові дані.

## Навіщо потрібно?

- 🎯 **Об'єктивна оцінка** — як модель працює на нових даних
- 🔍 **Виявлення overfitting** — чи не запам'ятала модель train set
- 📊 **Порівняння моделей** — чесна конкуренція
- ⚠️ **Попередження проблем** — before production
- 💡 **Вибір гіперпараметрів** — разом з validation set
- 🚀 **Production-ready** — впевненість у якості

## Коли використовувати?

**Завжди! 🔥**
- Будь-яка задача ML (регресія, класифікація, кластеризація з метриками)
- Перед навчанням моделі
- При порівнянні алгоритмів
- Коли потрібна чесна оцінка

**Виняток:**
- Time series (використовується time-based split)
- Cross-validation (але навіть тоді потрібен final test set)

---

## Основний принцип

### Чому потрібно розділяти?

```python
# ❌ ПОГАНО: оцінка на тих самих даних, на яких навчалася
model.fit(X, y)
score = model.score(X, y)  # Може бути 100%, але це нічого не означає!

# ✅ ДОБРЕ: оцінка на НОВИХ даних
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Реальна оцінка!
```

### Візуальна інтуїція

```
Весь датасет (100%):
╔════════════════════════════════════════╗
║ • • • • • • • • • • • • • • • • • • • ║
║ • • • • • • • • • • • • • • • • • • • ║
╚════════════════════════════════════════╝

Train-Test Split (70/30):
╔══════════════════════════╦═══════════╗
║ TRAIN SET (70%)          ║ TEST      ║
║ Модель БАЧИТЬ ці дані ✓  ║ Модель    ║
║ Навчається тут            ║ НЕ БАЧИТЬ ║
║ • • • • • • • • • • •    ║ ці дані ✗ ║
║ • • • • • • • • • • •    ║ • • • •   ║
╚══════════════════════════╩═══════════╝
                           │
                           └─> Оцінка узагальнення
```

---

## Базове використання (Scikit-learn)

### Найпростіший приклад

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Дані
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([1, 2, 3, 4, 5])

# Розділення: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% для test
    random_state=42     # Відтворюваність
)

print(f"X shape: {X.shape}")
print(f"X_train shape: {X_train.shape}")  # (4, 2) - 80%
print(f"X_test shape: {X_test.shape}")    # (1, 2) - 20%
```

### З реальними даними

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження даних (приклад)
# boston = load_boston()
# X = boston.data
# y = boston.target

# Альтернатива (boston deprecated):
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = housing.data
y = housing.target

print(f"Dataset size: {X.shape[0]} samples")
print(f"Features: {X.shape[1]}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,     # 30% для тесту
    random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Навчання моделі
model = LinearRegression()
model.fit(X_train, y_train)

# Оцінка на train
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Оцінка на test
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Train R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
print(f"Test R²:  {test_r2:.4f} | RMSE: {test_rmse:.4f}")
print(f"Gap:      {train_r2 - test_r2:.4f}")

if train_r2 - test_r2 > 0.1:
    print("⚠️  Large gap → possible overfitting")
elif test_r2 < 0.6:
    print("⚠️  Low test score → possible underfitting")
else:
    print("✅ Good balance!")
```

---

## Параметри train_test_split

### test_size

**Розмір тестового набору** (частка або абсолютне число).

```python
# Частка (найчастіше використовується)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2  # 20% для test
)

# Абсолютна кількість
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=100  # Рівно 100 зразків для test
)

# Якщо не вказано, за замовчуванням test_size=0.25 (25%)
```

**Рекомендації:**
- **Малий датасет** (< 1000): `test_size=0.2-0.3` (20-30%)
- **Середній датасет** (1000-10000): `test_size=0.2` (20%)
- **Великий датасет** (> 10000): `test_size=0.1-0.2` (10-20%)

### train_size

**Розмір тренувального набору** (альтернатива test_size).

```python
# Можна вказати train_size замість test_size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8  # 80% для train → 20% автоматично для test
)

# Або обидва (мають сумуватися до 1.0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3
)
```

### random_state

**Seed для генератора випадкових чисел** — забезпечує **відтворюваність**.

```python
# З random_state — завжди однаковий розподіл
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# X_train1 == X_train2 ✓ (ідентичні)

# Без random_state — кожен раз різний розподіл
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X, y, test_size=0.3  # random_state=None (default)
)
# X_train3 != X_train1 (різні)
```

**Рекомендація:** завжди використовуй `random_state` для **reproducibility**!

```python
# ✅ ДОБРЕ
random_state = 42  # Будь-яке число, але фіксоване
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
```

### shuffle

**Чи перемішувати дані** перед розділенням.

```python
# За замовчуванням shuffle=True (перемішує)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

# Без перемішування (для time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
```

**Коли shuffle=False?**
- **Time series** — порядок важливий!
- Дані вже впорядковані і це важливо

**Увага:** якщо дані відсортовані (наприклад, всі 0 в початку, всі 1 в кінці), `shuffle=False` призведе до **неправильного розподілу**!

```python
# ❌ ПОГАНО: дані відсортовані
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# З shuffle=False
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, shuffle=False
)
# y_train = [0, 0, 0, 0, 0] — тільки клас 0!
# y_test = [1, 1, 1, 1, 1] — тільки клас 1!

# ✅ ДОБРЕ: з shuffle=True
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, shuffle=True, random_state=42
)
# y_train та y_test містять обидва класи
```

### stratify

**Зберігає пропорції класів** у train та test sets (для класифікації).

```python
# Без stratify — може бути дисбаланс
y = np.array([0]*90 + [1]*10)  # 90% клас 0, 10% клас 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Train: {np.bincount(y_train)}")  # Може бути [64, 6] (91% vs 9%)
print(f"Test: {np.bincount(y_test)}")    # Може бути [26, 4] (87% vs 13%)

# З stratify — пропорції зберігаються
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"Train: {np.bincount(y_train)}")  # [63, 7] (90% vs 10%) ✓
print(f"Test: {np.bincount(y_test)}")    # [27, 3] (90% vs 10%) ✓
```

**Коли використовувати stratify?**
- **Класифікація** з несбалансованими класами
- Малі датасети
- Коли важливо зберегти розподіл

```python
# ✅ Рекомендується для класифікації
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,      # Зберегти пропорції класів
    random_state=42
)
```

---

## Типові пропорції розділення

### 70/30 Split

```python
# 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Використання: середні датасети
```

### 80/20 Split (найпопулярніший)

```python
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Використання: стандарт для більшості задач
```

### 90/10 Split

```python
# 90% train, 10% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Використання: великі датасети (>100k зразків)
```

### Правило вибору

| Розмір датасету | Рекомендація | Приклад |
|-----------------|--------------|---------|
| **Малий** (< 1000) | 70/30 або 60/40 | `test_size=0.3` |
| **Середній** (1k-10k) | 80/20 | `test_size=0.2` |
| **Великий** (10k-100k) | 80/20 або 85/15 | `test_size=0.15` |
| **Дуже великий** (>100k) | 90/10 або 95/5 | `test_size=0.1` |

**Ключовий принцип:**
- Test set має бути **достатньо великим** для надійної оцінки
- Train set має бути **достатньо великим** для навчання

```python
# Мінімальні розміри
# Test: хоча б 100-200 зразків (для надійної статистики)
# Train: залежить від алгоритму та кількості ознак

# Приклад: 500 зразків
# 70/30 → train=350, test=150 ✓
# 80/20 → train=400, test=100 ✓ (але test на межі)
# 90/10 → train=450, test=50 ❌ (test занадто малий)
```

---

## Train/Validation/Test Split

### Навіщо 3 набори?

```
╔═════════════════════╦══════════════╦═══════════╗
║ TRAIN (60%)         ║ VAL (20%)    ║ TEST (20%)║
╠═════════════════════╬══════════════╬═══════════╣
║ Навчання моделі     ║ Tuning       ║ Фінальна  ║
║                     ║ параметрів   ║ оцінка    ║
║ model.fit()         ║ GridSearch   ║ score()   ║
║                     ║ Early Stop   ║ 1 раз!    ║
╚═════════════════════╩══════════════╩═══════════╝
```

### Реалізація

**Метод 1: Два послідовні split**

```python
from sklearn.model_selection import train_test_split

# Крок 1: Відділити test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Крок 2: Розділити temp на train і validation
# 0.25 * 0.8 = 0.2 (20% від оригінального датасету)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"Original: {len(X)}")
print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")
```

**Метод 2: Функція для 3-way split**

```python
def train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, 
                         random_state=None, stratify=None):
    """
    Розділення на train, validation, test
    """
    assert train_size + val_size + test_size == 1.0, \
        "Sizes must sum to 1.0"
    
    # Train + Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )
    
    # Train vs Val
    # val_size / (train_size + val_size) = val частка від temp
    val_ratio = val_size / (train_size + val_size)
    
    stratify_temp = None
    if stratify is not None:
        stratify_temp = y_temp
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# Використання
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, 
    train_size=0.6, 
    val_size=0.2, 
    test_size=0.2,
    random_state=42,
    stratify=y  # Для класифікації
)
```

### Workflow з validation

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 1. Відокремити test set (торкаємось ТІЛЬКИ в кінці!)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Розділити на train і validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# 3. Tuning на train/validation
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None]
}

# Варіант A: Manual validation
best_score = -np.inf
best_params = None

for n_est in param_grid['n_estimators']:
    for max_d in param_grid['max_depth']:
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, 
                                   random_state=42)
        rf.fit(X_train, y_train)
        val_score = rf.score(X_val, y_val)
        
        if val_score > best_score:
            best_score = val_score
            best_params = {'n_estimators': n_est, 'max_depth': max_d}

print(f"Best params: {best_params}")
print(f"Best validation score: {best_score:.4f}")

# Варіант B: GridSearchCV (з CV на train+val)
# Примітка: GridSearchCV сам робить CV, але ми можемо використати
# validation set для early stopping або фінального вибору

# 4. Фінальна модель з найкращими параметрами
final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)  # Або X_temp (train+val)

# 5. ФІНАЛЬНА оцінка на test (тільки ОДИН раз!)
test_score = final_model.score(X_test, y_test)
print(f"\nFinal test score: {test_score:.4f}")
```

---

## Для Time Series

### Чому не використовувати звичайний train_test_split?

**Time series має часову залежність!**

```python
# ❌ ПОГАНО: shuffle=True для time series
dates = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', 
         '2020-06', '2020-07', '2020-08']

# З shuffle=True → порушує часову послідовність
# Train: ['2020-02', '2020-05', '2020-07', '2020-08']
# Test: ['2020-01', '2020-03', '2020-04', '2020-06']
# Модель навчається на майбутніх даних! ❌
```

### Time-based split

```python
# ✅ ДОБРЕ: для time series
# Train: минуле
# Test: майбутнє

# Варіант 1: Manual split
train_size = int(len(X) * 0.8)

X_train = X[:train_size]   # Перші 80%
y_train = y[:train_size]

X_test = X[train_size:]    # Останні 20%
y_test = y[train_size:]

# Варіант 2: з train_test_split (shuffle=False!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    shuffle=False  # КРИТИЧНО для time series!
)
```

### Time Series Split з Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series CV
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"\nFold {fold + 1}:")
    print(f"  Train: index {train_idx[0]} to {train_idx[-1]}")
    print(f"  Test: index {test_idx[0]} to {test_idx[-1]}")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  Score: {score:.4f}")
```

**Візуалізація TimeSeriesSplit:**

```
Fold 1: [Train----] [Test]
Fold 2: [Train--------] [Test]
Fold 3: [Train------------] [Test]
Fold 4: [Train----------------] [Test]
Fold 5: [Train--------------------] [Test]

Кожен fold:
- Train росте (включає всі попередні дані)
- Test завжди ПІСЛЯ train (майбутнє)
```

---

## Поширені помилки ❌

### 1. Оцінка на train set

```python
# ❌ ДУЖЕ ПОГАНО
model.fit(X, y)
score = model.score(X, y)
print(f"Accuracy: {score}")  # Може бути 100%, але неправда!

# ✅ ДОБРЕ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train: {train_score:.3f}, Test: {test_score:.3f}")
```

### 2. Tuning на test set

```python
# ❌ ПОГАНО: витік інформації з test set
for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ❌ Використали test для вибору!

# ✅ ДОБРЕ: використовуй validation або CV
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(Ridge(), {'alpha': [0.1, 1.0, 10.0]}, cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Тепер ОДИН раз оцінюємо на test
final_score = best_model.score(X_test, y_test)
```

### 3. Preprocessing перед split

```python
# ❌ ПОГАНО: витік інформації
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Використав ВЕСЬ датасет!

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
# Test set вплинув на нормалізацію train set!

# ✅ ДОБРЕ: спочатку split, потім preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)    # Тільки train!
X_test_scaled = scaler.transform(X_test)          # Застосувати до test
```

### 4. Не фіксувати random_state

```python
# ❌ ПОГАНО: різні результати кожен раз
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Неможливо відтворити експеримент!

# ✅ ДОБРЕ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)
# Завжди однаковий розподіл
```

### 5. Shuffle для time series

```python
# ❌ ПОГАНО: для часових рядів
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True  # Порушує часову послідовність!
)

# ✅ ДОБРЕ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False  # Зберегти порядок
)
```

### 6. Не використовувати stratify для несбалансованих класів

```python
# Дані: 95% клас 0, 5% клас 1
y = np.array([0]*950 + [1]*50)

# ❌ ПОГАНО: без stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Може бути: y_test містить 0 екземплярів класу 1!

# ✅ ДОБРЕ: зі stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
# y_test містить ~5% класу 1 (пропорційно)
```

### 7. Забувати про random seed у всіх місцях

```python
# ❌ ПОГАНО: тільки в train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

model = RandomForestRegressor()  # random_state=None → різні результати
model.fit(X_train, y_train)

# ✅ ДОБРЕ: скрізь
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
```

---

## Практичні поради 💡

### 1. Завжди фіксуй random_state

```python
# Створи константу
RANDOM_STATE = 42

# Використовуй скрізь
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = RandomForestRegressor(random_state=RANDOM_STATE)
```

### 2. Перевіряй розподіл класів (для класифікації)

```python
import pandas as pd

# Після split
print("Train set class distribution:")
print(pd.Series(y_train).value_counts(normalize=True))

print("\nTest set class distribution:")
print(pd.Series(y_test).value_counts(normalize=True))

# Повинні бути схожі!
```

### 3. Використовуй validation set для tuning

```python
# Схема: Train → Validation → Test
# Train: навчання
# Validation: підбір гіперпараметрів
# Test: ФІНАЛЬНА оцінка (1 раз!)

# НЕ торкайся test set до самого кінця!
```

### 4. Документуй розподіл

```python
# В коді або в коментарях
"""
Data split:
- Train: 60% (12,000 samples)
- Validation: 20% (4,000 samples)
- Test: 20% (4,000 samples)
- Random state: 42
- Stratified: Yes (imbalanced classes)
"""
```

### 5. Зберігай індекси

```python
# Корисно для debugging
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, range(len(X)),  # Передати індекси
    test_size=0.2,
    random_state=42
)

# Тепер можна знайти оригінальні зразки
print(f"Test set indices: {test_idx[:10]}")
```

### 6. Візуалізуй розподіл

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Train
axes[0].hist(y_train, bins=30, alpha=0.7)
axes[0].set_title('Train Set Target Distribution')
axes[0].set_xlabel('Target')
axes[0].set_ylabel('Frequency')

# Test
axes[1].hist(y_test, bins=30, alpha=0.7, color='orange')
axes[1].set_title('Test Set Target Distribution')
axes[1].set_xlabel('Target')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Розподіли повинні бути схожі!
```

### 7. Wrapper function для consistency

```python
def prepare_data(X, y, test_size=0.2, val_size=0.2, random_state=42, 
                 stratify=False, scale=False):
    """
    Комплексна підготовка даних з best practices
    
    Returns:
        dict з keys: X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    from sklearn.preprocessing import StandardScaler
    
    # Stratify
    strat = y if stratify else None
    
    # Split на train+val та test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    
    # Split на train та val
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)
        strat_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, 
            random_state=random_state, stratify=strat_temp
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = None, None
    
    # Scaling
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if X_val is not None:
            X_val = scaler.transform(X_val)
    
    # Print info
    print(f"Data split complete:")
    print(f"  Train: {X_train.shape[0]} samples")
    if X_val is not None:
        print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler
    }

# Використання
data = prepare_data(X, y, test_size=0.2, val_size=0.2, 
                    random_state=42, stratify=True, scale=True)

X_train = data['X_train']
y_train = data['y_train']
# etc.
```

---

## Реальний приклад: Повний workflow

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Константа для reproducibility
RANDOM_STATE = 42

print("="*70)
print("COMPLETE ML WORKFLOW WITH TRAIN-TEST SPLIT")
print("="*70)

# 1. Завантаження даних
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print(f"\n📊 Dataset Info:")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)} (0=malignant, 1=benign)")
print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

# 2. Train/Val/Test Split (60/20/20)
print(f"\n📂 Splitting data (60/20/20)...")

# Крок 1: Відокремити test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # Зберегти пропорції класів
    random_state=RANDOM_STATE
)

# Крок 2: Розділити temp на train/val (60/20 від оригіналу)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,  # 0.25 * 0.8 = 0.2
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# Перевірка stratification
print(f"\nClass distribution:")
print(f"  Train: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
print(f"  Val:   {pd.Series(y_val).value_counts(normalize=True).to_dict()}")
print(f"  Test:  {pd.Series(y_test).value_counts(normalize=True).to_dict()}")

# 3. Preprocessing (ПІСЛЯ split!)
print(f"\n🔧 Scaling features...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # Fit тільки на train!
X_val_scaled = scaler.transform(X_val)          # Transform val
X_test_scaled = scaler.transform(X_test)        # Transform test

# 4. Model Selection & Tuning (на train + validation)
print(f"\n🔍 Hyperparameter tuning...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,  # 5-fold CV на train set
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# 5. Evaluation на validation
best_model = grid_search.best_estimator_
val_score = best_model.score(X_val_scaled, y_val)
y_val_pred = best_model.predict(X_val_scaled)
val_auc = roc_auc_score(y_val, best_model.predict_proba(X_val_scaled)[:, 1])

print(f"\n📊 Validation Results:")
print(f"Accuracy: {val_score:.4f}")
print(f"ROC-AUC: {val_auc:.4f}")

# 6. FINAL Evaluation на test (ТІЛЬКИ ОДИН РАЗ!)
print(f"\n🎯 FINAL Test Results:")
test_score = best_model.score(X_test_scaled, y_test)
y_test_pred = best_model.predict(X_test_scaled)
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])

print(f"Accuracy: {test_score:.4f}")
print(f"ROC-AUC: {test_auc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Malignant', 'Benign']))

# 7. Візуалізація
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test_scaled, y_test,
    cmap='Blues', ax=axes[0, 0]
)
axes[0, 0].set_title('Confusion Matrix (Test Set)', fontweight='bold')

# ROC Curve
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(
    best_model, X_test_scaled, y_test,
    ax=axes[0, 1]
)
axes[0, 1].set_title(f'ROC Curve (AUC={test_auc:.3f})', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Feature Importances
feature_imp = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[1, 0].barh(feature_imp['feature'], feature_imp['importance'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importances', fontweight='bold')
axes[1, 0].invert_yaxis()

# Train/Val/Test Comparison
scores = {
    'Train': best_model.score(X_train_scaled, y_train),
    'Validation': val_score,
    'Test': test_score
}

axes[1, 1].bar(scores.keys(), scores.values(), 
              color=['blue', 'orange', 'green'], alpha=0.7)
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Model Performance Comparison', fontweight='bold')
axes[1, 1].set_ylim([0.9, 1.0])
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (name, val) in enumerate(scores.items()):
    axes[1, 1].text(i, val + 0.005, f'{val:.3f}', 
                   ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# 8. Підсумок
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Model: Random Forest")
print(f"Best params: {grid_search.best_params_}")
print(f"\nPerformance:")
print(f"  Train Accuracy:      {scores['Train']:.4f}")
print(f"  Validation Accuracy: {scores['Validation']:.4f}")
print(f"  Test Accuracy:       {scores['Test']:.4f}")
print(f"  Test ROC-AUC:        {test_auc:.4f}")

# Діагноз
gap = scores['Train'] - scores['Test']
if gap > 0.1:
    print(f"\n⚠️  Large gap ({gap:.3f}) → possible overfitting")
elif scores['Test'] < 0.9:
    print(f"\n⚠️  Low test score → room for improvement")
else:
    print(f"\n✅ Excellent performance with good generalization!")

print(f"{'='*70}")
```

---

## Пов'язані теми

- [[01_Bias_Variance_Tradeoff]] — чому важливо тестувати на нових даних
- [[02_Overfitting_Underfitting]] — діагностика за допомогою train/test
- [[04_Cross_Validation]] — більш надійна альтернатива
- [[Preprocessing]] — важливість порядку операцій
- [[Model_Selection]] — вибір і tuning моделей

## Ресурси

- [Scikit-learn: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Machine Learning Mastery: Train-Test Split](https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/)
- [Towards Data Science: The Right Way to Split Data](https://towardsdatascience.com/how-to-split-data-into-three-sets-train-validation-and-test-and-why-e50d22d3e54c)

---

## Ключові висновки

> Train-Test Split — фундаментальний крок в ML pipeline. Модель навчається на train set, а оцінюється на test set, який вона НІКОЛИ не бачила. Це єдиний спосіб чесно оцінити здатність до узагальнення.

**Best Practices:**
```python
# 1. Завжди розділяй дані
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Preprocessing ПІСЛЯ split
scaler.fit_transform(X_train)  # ✓
scaler.transform(X_test)       # ✓

# 3. Три набори для tuning
# Train → навчання
# Validation → підбір параметрів  
# Test → фінальна оцінка (1 раз!)

# 4. Фіксуй random_state для reproducibility
```

**Типові помилки:**
- ❌ Оцінка на train set
- ❌ Tuning на test set
- ❌ Preprocessing перед split
- ❌ Shuffle для time series
- ❌ Не використовувати stratify для несбалансованих класів

**Правило розміру:**
- Малі дані (< 1000): 70/30 або 60/20/20
- Середні (1k-10k): 80/20 або 60/20/20
- Великі (> 10k): 90/10 або 70/15/15

---

#ml #core-concepts #train-test-split #validation #model-evaluation #best-practices
