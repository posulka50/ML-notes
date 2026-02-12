# Overfitting & Underfitting (Перенавчання та Недонавчання)

## Що це?

**Overfitting** — модель **занадто добре запам'ятовує тренувальні дані**, включно з шумом та випадковими відхиленнями. Результат: відмінні результати на train, погані на test.

**Underfitting** — модель **занадто проста** і не може вловити справжні закономірності в даних. Результат: погані результати і на train, і на test.

**Ідеальна модель** — знаходить баланс між складністю та узагальненням, добре працює на нових даних.

## Навіщо потрібно?

- 🎯 **Діагностика проблем моделі** — чому погано працює
- 🔍 **Вибір правильної складності** — не занадто просто, не занадто складно
- 📊 **Покращення узагальнення** — краща робота на нових даних
- ⚙️ **Налаштування гіперпараметрів** — що крутити
- 🚀 **Production-ready моделі** — стабільні передбачення
- 💰 **Економія ресурсів** — не витрачати час на погані моделі

## Коли важливо розуміти?

**Потрібно:**

- Модель добре працює на train, але погано на test
- Вибираєш між простою та складною архітектурою
- **Debugging ML моделі** — що не так?
- Налаштування регуляризації
- Feature engineering — додавати чи ні ознаки

**Не потрібно:**

- Модель іде в production і добре працює
- Train = Test scores (хороші результати)

---

## Underfitting (Недонавчання)

### Що це?

**Underfitting** — модель **не може вловити** навіть основні паттерни в даних через **надмірну простоту**.

### Візуальна інтуїція

```
Справжні дані (парабола):

     y
     |      •
     |    •   •
     |  •       •
     | •         •
     |•           •
     |_______________ x

Лінійна модель (underfitting):

     y
     |      •
     |    • | •
     |  •   |   •     ← Пряма не може вловити криву!
     | •────┼────•
     |•     |     •
     |______|________ x
     
Train Error: ВИСОКИЙ ❌
Test Error: ВИСОКИЙ ❌
Gap: МАЛИЙ
```

### Характеристики

| Метрика | Значення | Інтерпретація |
|---------|----------|---------------|
| **Train Score** | Низький (< 0.6) | ❌ Погано навіть на train |
| **Test Score** | Низький (< 0.6) | ❌ Погано на test |
| **Gap (Train - Test)** | Малий (< 0.05) | Scores близькі |
| **Діагноз** | Модель занадто проста | HIGH BIAS |

### Ознаки underfitting

#### 1. Погані метрики на train

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Нелінійні дані
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)

# Лінійна модель
lr = LinearRegression()
lr.fit(X, y)

train_score = lr.score(X, y)
print(f"Train R²: {train_score:.3f}")  # Наприклад: 0.001 ❌

# Якщо train score поганий → underfitting!
```

#### 2. Train ≈ Test (обидва погані)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

lr.fit(X_train, y_train)

train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)

print(f"Train: {train_score:.3f}")  # 0.002
print(f"Test:  {test_score:.3f}")   # 0.001
print(f"Gap:   {train_score - test_score:.3f}")  # 0.001 (малий)

# Обидва погані, gap малий → underfitting!
```

#### 3. Learning curve виходить на "плато"

```
R² Score
    |
0.6 |  Train ─────────────
    |  Test  ─────────────  ← Обидві криві низько і плоско
0.4 |
    |
0.2 |
    |_____________________________
        Training Set Size
    
Більше даних НЕ допоможе!
```

#### 4. Візуальний аналіз

```python
import matplotlib.pyplot as plt

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.5, label='Train data')
plt.scatter(X_test, y_test, alpha=0.5, label='Test data')
plt.plot(X, lr.predict(X), 'r-', linewidth=2, label='Model')
plt.plot(X, np.sin(X), 'g--', linewidth=2, label='True function')
plt.legend()
plt.title('Underfitting: Linear model on non-linear data')
plt.show()

# Якщо модель явно не відповідає даним → underfitting!
```

### Причини underfitting

#### 1. Модель занадто проста

```python
# ❌ Лінійна регресія на складних нелінійних даних
lr = LinearRegression()

# ❌ Logistic Regression на даних з складною boundary
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

# ❌ Decision Tree з max_depth=1
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=1)  # Занадто просто!
```

#### 2. Недостатньо ознак

```python
# ❌ Тільки одна ознака для складної задачі
X = df[['sqft']]  # Тільки площа
y = df['price']   # Ціна залежить від багатьох факторів!

# ✅ Більше ознак
X = df[['sqft', 'bedrooms', 'location', 'age', 'condition']]
```

#### 3. Занадто сильна регуляризація

```python
# ❌ Ridge з дуже великим λ
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1000.0)  # Занадто сильно!

# Модель стає майже константною
```

#### 4. Неправильна обробка даних

```python
# ❌ Видалили важливі ознаки
X = df.drop(['important_feature'], axis=1)

# ❌ Погана нормалізація
# Залишили масштаби несумісними
```

### Як виправити underfitting?

#### ✅ 1. Збільшити складність моделі

```python
# Decision Tree: більша глибина
# Було:
dt = DecisionTreeRegressor(max_depth=1)

# Стало:
dt = DecisionTreeRegressor(max_depth=10)

# Neural Network: більше шарів/нейронів
# Було:
model = Sequential([
    Dense(10, activation='relu'),
    Dense(1)
])

# Стало:
model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
```

#### ✅ 2. Додати polynomial features

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Було: y = ax + b
lr = LinearRegression()

# Стало: y = ax² + bx + c
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

poly_model.fit(X_train, y_train)
```

#### ✅ 3. Більше ознак (Feature Engineering)

```python
# Додати взаємодії
df['price_per_sqft'] = df['price'] / df['sqft']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Додати нелінійні трансформації
df['sqft_squared'] = df['sqft'] ** 2
df['log_sqft'] = np.log(df['sqft'])

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 5, 10, 20, 50], 
                         labels=['new', 'recent', 'old', 'very_old'])
```

#### ✅ 4. Зменшити регуляризацію

```python
# Ridge: зменшити alpha
# Було:
ridge = Ridge(alpha=100.0)

# Стало:
ridge = Ridge(alpha=0.1)

# Lasso: менший alpha
lasso = Lasso(alpha=0.01)

# Decision Tree: менше обмежень
dt = DecisionTreeRegressor(
    max_depth=None,           # Без обмеження
    min_samples_split=2,      # Мінімальне значення
    min_samples_leaf=1        # Мінімальне значення
)
```

#### ✅ 5. Використати складніший алгоритм

```python
# Було: Linear Regression
lr = LinearRegression()

# Стало: Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)

# Або: Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100)

# Або: Neural Network
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(hidden_layer_sizes=(100, 50))
```

---

## Overfitting (Перенавчання)

### Що це?

**Overfitting** — модель **занадто добре запам'ятовує** тренувальні дані, включно з **шумом та випадковостями**. Не узагальнює на нові дані.

### Візуальна інтуїція

```
Справжні дані (лінія + шум):

     y
     |    •
     |  •   •
     | •  •  •
     |•  •  •
     |  •  •
     |_____________ x

Поліном 15-го степеня (overfitting):

     y
     |    ╱•╲
     |  •╱   ╲•      ← Проходить через ВСІ точки!
     | •╱  •  ╲•     ← Запам'ятала шум!
     |╱•  •  • ╲
     |  •  •    ╲
     |_____________ x
     
Train Error: ДУЖЕ НИЗЬКИЙ ✓
Test Error: ВИСОКИЙ ❌
Gap: ВЕЛИКИЙ ❌
```

### Характеристики

| Метрика | Значення | Інтерпретація |
|---------|----------|---------------|
| **Train Score** | Дуже високий (> 0.95) | ✓ Ідеально на train |
| **Test Score** | Низький (< 0.7) | ❌ Погано на test |
| **Gap (Train - Test)** | Великий (> 0.15) | ❌ Проблема! |
| **Діагноз** | Модель запам'ятала train set | HIGH VARIANCE |

### Ознаки overfitting

#### 1. Високий train score, низький test score

```python
from sklearn.tree import DecisionTreeRegressor

# Глибоке дерево (без обмежень)
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

train_score = dt.score(X_train, y_train)
test_score = dt.score(X_test, y_test)

print(f"Train: {train_score:.3f}")  # 1.000 ✓ (ідеально!)
print(f"Test:  {test_score:.3f}")   # 0.600 ❌ (погано)
print(f"Gap:   {train_score - test_score:.3f}")  # 0.400 ❌ (великий!)

# Великий gap → overfitting!
```

#### 2. Модель запам'ятала outliers

```python
# Візуалізація
plt.figure(figsize=(12, 5))

# Subplot 1: Train data
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5)
X_plot = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)
plt.plot(X_plot, dt.predict(X_plot), 'r-', linewidth=2)
plt.title(f'Train Set (R² = {train_score:.3f})')
plt.xlabel('X')
plt.ylabel('y')

# Subplot 2: Test data
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.5)
plt.plot(X_plot, dt.predict(X_plot), 'r-', linewidth=2)
plt.title(f'Test Set (R² = {test_score:.3f})')
plt.xlabel('X')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# Якщо на train модель "пробігає" через кожну точку,
# а на test "промахується" → overfitting!
```

#### 3. Learning curves розходяться

```
R² Score
    |
1.0 |  Train ───────────────  ← Train дуже високо
    |           
0.8 |           
    |  Test ─────────────────  ← Test низько
0.6 |
    |    ↑ Великий gap!
0.4 |
    |_____________________________
        Training Set Size
    
Gap НЕ зменшується зі збільшенням даних!
(або зменшується дуже повільно)
```

#### 4. Дуже складна модель

```python
# Дуже багато параметрів
print(f"Number of parameters: {dt.tree_.node_count}")
# Наприклад: 5000 nodes для 1000 train samples → overfitting!

# Правило: якщо parameters >> samples → ризик overfitting
```

### Причини overfitting

#### 1. Модель занадто складна

```python
# ❌ Decision Tree без обмежень
dt = DecisionTreeRegressor()  # max_depth=None → може рости до безкінечності!

# ❌ Polynomial з високим degree
poly = PolynomialFeatures(degree=15)  # Занадто гнучко!

# ❌ KNN з K=1
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=1)  # Запам'ятовує кожну точку!
```

#### 2. Мало тренувальних даних

```python
# ❌ 100 samples, 50 features → overfitting неминучий!
print(f"Samples: {X_train.shape[0]}")    # 100
print(f"Features: {X_train.shape[1]}")   # 50

# Правило: потрібно хоча б 10 samples на кожну ознаку
```

#### 3. Немає регуляризації

```python
# ❌ Linear Regression без penalty
lr = LinearRegression()

# ❌ Neural Network без dropout, без regularization
model = Sequential([
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1)
])
# Дуже багато параметрів, нічого не стримує!
```

#### 4. Тренування занадто довго

```python
# ❌ Neural Network тренується 1000 epochs
model.fit(X_train, y_train, epochs=1000, verbose=0)
# Модель починає запам'ятовувати шум після певної кількості epochs
```

#### 5. Дублікати в даних

```python
# ❌ Train і test overlap
# Якщо випадково той самий зразок потрапив і в train, і в test
# → штучно завищена test score, але насправді overfitting
```

### Як виправити overfitting?

#### ✅ 1. Більше тренувальних даних

```python
# Найкращий спосіб (якщо можливо)!

# Збільшити dataset
# - Зібрати більше даних
# - Data augmentation (для зображень)
# - Synthetic data generation

# Data augmentation (приклад для зображень)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
```

#### ✅ 2. Регуляризація (Regularization)

**L2 Regularization (Ridge):**

```python
from sklearn.linear_model import Ridge

# Додає penalty на великі ваги: λ * ||w||²
ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X_train, y_train)

# Більший alpha → сильніша регуляризація
# alpha = 0 → звичайна Linear Regression
# alpha = ∞ → всі ваги → 0
```

**L1 Regularization (Lasso):**

```python
from sklearn.linear_model import Lasso

# Додає penalty: λ * ||w||₁
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Lasso може зануляти ваги → feature selection
```

**Elastic Net (L1 + L2):**

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
# l1_ratio = 0 → Ridge
# l1_ratio = 1 → Lasso
# l1_ratio = 0.5 → 50/50
```

#### ✅ 3. Зменшити складність моделі

**Decision Trees:**

```python
# Було: необмежене дерево
dt = DecisionTreeRegressor()

# Стало: обмеження
dt = DecisionTreeRegressor(
    max_depth=5,              # Максимальна глибина
    min_samples_split=20,     # Мінімум зразків для split
    min_samples_leaf=10,      # Мінімум зразків в листі
    max_features='sqrt'       # Випадковий підмножина ознак
)
```

**Neural Networks:**

```python
# Було: дуже глибока мережа
model = Sequential([
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1)
])

# Стало: менша мережа
model = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
```

**Polynomial Regression:**

```python
# Було: degree=15
poly = PolynomialFeatures(degree=15)

# Стало: degree=3
poly = PolynomialFeatures(degree=3)
```

#### ✅ 4. Dropout (для Neural Networks)

```python
from keras.layers import Dropout

model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.5),              # Випадково вимикає 50% нейронів
    Dense(64, activation='relu'),
    Dropout(0.3),              # 30%
    Dense(1)
])

# Dropout примушує мережу не покладатися на окремі нейрони
# → краще узагальнення
```

#### ✅ 5. Early Stopping

```python
from keras.callbacks import EarlyStopping

# Зупинка, коли validation loss перестає покращуватися
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,              # Чекати 10 epochs
    restore_best_weights=True # Повернути найкращі ваги
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    callbacks=[early_stop]
)
```

#### ✅ 6. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Замість одного train/test split → 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std: {scores.std():.3f}")

# Якщо std високий → модель нестабільна → можливо overfitting
```

#### ✅ 7. Ensemble Methods

**Bagging (зменшує variance):**

```python
from sklearn.ensemble import RandomForestRegressor

# Random Forest = Bagging of Decision Trees
rf = RandomForestRegressor(
    n_estimators=100,    # 100 дерев
    max_depth=10,        # Кожне дерево обмежене
    max_features='sqrt'  # Випадкові ознаки
)

# Усереднення багатьох моделей → зменшує overfitting
```

**Boosting (з regularization):**

```python
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,           # Shallow trees
    learning_rate=0.1,     # Повільне навчання
    subsample=0.8,         # 80% даних для кожного дерева
    min_samples_leaf=5     # Регуляризація
)
```

#### ✅ 8. Feature Selection

```python
# Видалити непотрібні/шумливі ознаки
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# Менше ознак → менша складність → менше overfitting
```

#### ✅ 9. Data Augmentation (для зображень)

```python
# Збільшити train set через трансформації
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])
```

---

## Оптимальна модель (Sweet Spot)

### Характеристики

| Метрика | Значення |
|---------|----------|
| **Train Score** | Високий (0.8-0.95) ✓ |
| **Test Score** | Високий (0.75-0.9) ✓ |
| **Gap** | Малий (< 0.1) ✓ |
| **Узагальнення** | Добре ✓ |

### Як досягти?

```python
# 1. Почати з простої моделі (baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
baseline_score = lr.score(X_test, y_test)
print(f"Baseline: {baseline_score:.3f}")

# 2. Поступово збільшувати складність
from sklearn.model_selection import GridSearchCV

# Decision Tree: tuning max_depth
param_grid = {'max_depth': [1, 2, 3, 5, 7, 10, 15, 20, None]}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)

grid.fit(X_train, y_train)

print(f"Best max_depth: {grid.best_params_['max_depth']}")
print(f"Best CV score: {grid.best_score_:.3f}")

best_model = grid.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.3f}")

# 3. Якщо gap великий → додати regularization
# 4. Якщо train score низький → збільшити складність
# 5. Повторювати до оптимального балансу
```

---

## Практичний workflow для діагностики

```python
def diagnose_and_fix(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Комплексна діагностика та рекомендації
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    
    print("="*70)
    print(f"DIAGNOSIS: {model_name}")
    print("="*70)
    
    # 1. Базові метрики
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score
    
    print(f"\n📊 Scores:")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²:  {test_score:.4f}")
    print(f"  Gap:      {gap:.4f}")
    
    # 2. Діагноз
    print(f"\n🔍 Diagnosis:")
    
    if test_score < 0.6 and gap < 0.1:
        diagnosis = "UNDERFITTING (HIGH BIAS)"
        color = '🔴'
        print(f"  {color} {diagnosis}")
        print(f"\n💡 Recommendations:")
        print(f"  1. Increase model complexity:")
        print(f"     - Larger max_depth (trees)")
        print(f"     - Higher polynomial degree")
        print(f"     - More layers/neurons (NN)")
        print(f"  2. Add more features")
        print(f"  3. Reduce regularization (smaller λ)")
        print(f"  4. Use a more complex algorithm")
        
    elif gap > 0.15:
        diagnosis = "OVERFITTING (HIGH VARIANCE)"
        color = '🔴'
        print(f"  {color} {diagnosis}")
        print(f"\n💡 Recommendations:")
        print(f"  1. Get more training data")
        print(f"  2. Add regularization:")
        print(f"     - Ridge/Lasso (larger λ)")
        print(f"     - Dropout (NN)")
        print(f"  3. Reduce model complexity:")
        print(f"     - Smaller max_depth")
        print(f"     - Pruning")
        print(f"  4. Use ensemble methods (Random Forest)")
        print(f"  5. Feature selection")
        print(f"  6. Early stopping")
        
    elif test_score >= 0.6 and gap <= 0.15:
        diagnosis = "GOOD BALANCE"
        color = '✅'
        print(f"  {color} {diagnosis}")
        if gap > 0.05:
            print(f"\n💡 Minor recommendations:")
            print(f"  - Slight overfitting detected (gap={gap:.3f})")
            print(f"  - Consider adding light regularization")
        else:
            print(f"\n🎉 Model is well-tuned!")
    
    else:
        diagnosis = "UNUSUAL PATTERN"
        color = '⚠️'
        print(f"  {color} {diagnosis}")
        print(f"\n⚠️  Check for:")
        print(f"  - Data leakage")
        print(f"  - Wrong train/test split")
        print(f"  - Bugs in preprocessing")
    
    # 3. Learning Curves
    print(f"\n📈 Generating learning curves...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Візуалізація
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Learning Curves
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, train_mean, 'o-', linewidth=2, 
             label='Train Score', color='blue')
    plt.fill_between(train_sizes, 
                     train_mean - train_std, 
                     train_mean + train_std,
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 's-', linewidth=2, 
             label='Validation Score', color='red')
    plt.fill_between(train_sizes, 
                     val_mean - val_std, 
                     val_mean + val_std,
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size', fontsize=11)
    plt.ylabel('R² Score', fontsize=11)
    plt.title(f'Learning Curves: {model_name}', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Predictions vs Actual
    plt.subplot(1, 2, 2)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    plt.scatter(y_train, y_pred_train, alpha=0.5, s=30, 
               label=f'Train (R²={train_score:.3f})')
    plt.scatter(y_test, y_pred_test, alpha=0.5, s=30, 
               label=f'Test (R²={test_score:.3f})')
    
    # Perfect prediction line
    all_y = np.concatenate([y_train, y_test])
    plt.plot([all_y.min(), all_y.max()], 
             [all_y.min(), all_y.max()], 
             'k--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual', fontsize=11)
    plt.ylabel('Predicted', fontsize=11)
    plt.title('Predictions vs Actual', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Підсумок
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Diagnosis: {diagnosis}")
    print(f"Train R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")
    print(f"Gap: {gap:.4f}")
    
    # Інтерпретація learning curves
    final_gap_lc = train_mean[-1] - val_mean[-1]
    if final_gap_lc > 0.2:
        print(f"\n⚠️  Learning curves show high variance")
        print(f"   More data would likely help!")
    elif val_mean[-1] < 0.6 and final_gap_lc < 0.1:
        print(f"\n⚠️  Learning curves plateau early")
        print(f"   More data won't help - need more complex model!")
    else:
        print(f"\n✓ Learning curves look reasonable")
    
    print("="*70)
    
    return {
        'train_score': train_score,
        'test_score': test_score,
        'gap': gap,
        'diagnosis': diagnosis
    }


# Приклад використання
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

# Генерація даних
X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Тест різних моделей
print("\n### TEST 1: Shallow Tree (likely UNDERFITTING) ###")
dt_shallow = DecisionTreeRegressor(max_depth=2, random_state=42)
diagnose_and_fix(dt_shallow, X_train, X_test, y_train, y_test, 
                "Shallow Tree (max_depth=2)")

print("\n### TEST 2: Deep Tree (likely OVERFITTING) ###")
dt_deep = DecisionTreeRegressor(random_state=42)
diagnose_and_fix(dt_deep, X_train, X_test, y_train, y_test, 
                "Deep Tree (no limits)")

print("\n### TEST 3: Optimal Tree ###")
dt_optimal = DecisionTreeRegressor(max_depth=7, min_samples_leaf=10, random_state=42)
diagnose_and_fix(dt_optimal, X_train, X_test, y_train, y_test, 
                "Optimal Tree (max_depth=7)")
```

---

## Порівняння Underfitting vs Overfitting

| Аспект | Underfitting | Overfitting | Optimal |
|--------|--------------|-------------|---------|
| **Train Error** | Високий ❌ | Дуже низький ✓ | Низький ✓ |
| **Test Error** | Високий ❌ | Високий ❌ | Низький ✓ |
| **Gap** | Малий | Великий | Малий |
| **Проблема** | Занадто просто | Занадто складно | Баланс |
| **Bias** | Високий ❌ | Низький ✓ | Помірний ✓ |
| **Variance** | Низька ✓ | Висока ❌ | Помірна ✓ |
| **Узагальнення** | Погане | Погане | Добре ✓ |
| **Learning Curve** | Плато рано | Великий gap | Збіжність |
| **Візуально** | Не вловлює паттерн | Проходить через всі точки | Вловлює тренд |

### Рішення

**Underfitting →** Збільшити складність:
- ↑ max_depth
- ↑ polynomial degree
- ↑ features
- ↓ regularization

**Overfitting →** Зменшити variance:
- ↑ training data
- ↑ regularization
- ↓ model complexity
- Use ensemble methods
- Feature selection

---

## Приклади на різних алгоритмах

### Linear Regression

```python
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Нелінійні дані
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * np.sin(X).ravel() + np.random.normal(0, 0.3, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Тестуємо різні степені
degrees = [1, 3, 15]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, degree in enumerate(degrees):
    # Модель
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    poly_model.fit(X_train, y_train)
    
    train_score = poly_model.score(X_train, y_train)
    test_score = poly_model.score(X_test, y_test)
    gap = train_score - test_score
    
    # Діагноз
    if degree == 1:
        diagnosis = "UNDERFITTING"
        color = 'red'
    elif degree == 3:
        diagnosis = "OPTIMAL"
        color = 'green'
    else:
        diagnosis = "OVERFITTING"
        color = 'red'
    
    # Візуалізація
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    y_plot = poly_model.predict(X_plot)
    
    axes[idx].scatter(X_train, y_train, alpha=0.6, s=40, label='Train')
    axes[idx].scatter(X_test, y_test, alpha=0.6, s=40, label='Test')
    axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
    axes[idx].plot(X_plot, 2 * np.sin(X_plot), 'g--', 
                   linewidth=2, alpha=0.7, label='True function')
    
    axes[idx].set_title(
        f'Degree = {degree}\n'
        f'Train R² = {train_score:.3f}, Test R² = {test_score:.3f}\n'
        f'Gap = {gap:.3f}\n'
        f'{diagnosis}',
        fontsize=11, fontweight='bold', color=color
    )
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

# Класифікація: два "місяці"
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Тестуємо різні max_depth
depths = [1, 5, None]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, depth in enumerate(depths):
    # Модель
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)
    gap = train_score - test_score
    
    # Діагноз
    if depth == 1:
        diagnosis = "UNDERFITTING"
        color = 'red'
    elif depth == 5:
        diagnosis = "OPTIMAL"
        color = 'green'
    else:
        diagnosis = "OVERFITTING"
        color = 'red'
    
    # Візуалізація decision boundary
    from matplotlib.colors import ListedColormap
    
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ['red', 'blue']
    
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    
    # Train points
    axes[idx].scatter(X_train[:, 0], X_train[:, 1], 
                     c=y_train, cmap='coolwarm', 
                     edgecolor='k', s=50, alpha=0.7, label='Train')
    
    # Test points
    axes[idx].scatter(X_test[:, 0], X_test[:, 1], 
                     c=y_test, cmap='coolwarm', 
                     edgecolor='k', s=50, marker='s', alpha=0.7, label='Test')
    
    axes[idx].set_title(
        f'max_depth = {depth}\n'
        f'Train Acc = {train_score:.3f}, Test Acc = {test_score:.3f}\n'
        f'Gap = {gap:.3f}\n'
        f'{diagnosis}',
        fontsize=11, fontweight='bold', color=color
    )
    axes[idx].set_xlabel('X₁')
    axes[idx].set_ylabel('X₂')
    axes[idx].legend(fontsize=9)

plt.tight_layout()
plt.show()
```

### Neural Networks

```python
from sklearn.neural_network import MLPRegressor

# Різна кількість нейронів
architectures = [
    (10,),           # Проста
    (100, 50),       # Середня
    (200, 200, 200)  # Складна
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, hidden_layers in enumerate(architectures):
    # Модель
    nn = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=1000,
        random_state=42,
        early_stopping=False  # Вимкнути для чистого тесту
    )
    
    nn.fit(X_train, y_train)
    
    train_score = nn.score(X_train, y_train)
    test_score = nn.score(X_test, y_test)
    gap = train_score - test_score
    
    # Діагноз
    if idx == 0:
        diagnosis = "UNDERFITTING"
        color = 'red'
    elif idx == 1:
        diagnosis = "OPTIMAL"
        color = 'green'
    else:
        diagnosis = "OVERFITTING"
        color = 'red'
    
    # Візуалізація
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    y_plot = nn.predict(X_plot)
    
    axes[idx].scatter(X_train, y_train, alpha=0.6, s=40, label='Train')
    axes[idx].scatter(X_test, y_test, alpha=0.6, s=40, label='Test')
    axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
    axes[idx].plot(X_plot, 2 * np.sin(X_plot), 'g--', 
                   linewidth=2, alpha=0.7, label='True function')
    
    axes[idx].set_title(
        f'Architecture: {hidden_layers}\n'
        f'Train R² = {train_score:.3f}, Test R² = {test_score:.3f}\n'
        f'Gap = {gap:.3f}\n'
        f'{diagnosis}',
        fontsize=11, fontweight='bold', color=color
    )
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Поширені помилки ❌

### 1. Оцінювати тільки train score

```python
# ❌ ПОГАНО
model.fit(X_train, y_train)
print(f"Score: {model.score(X_train, y_train)}")
# Може бути 1.0, але це нічого не означає!

# ✅ ДОБРЕ
print(f"Train: {model.score(X_train, y_train)}")
print(f"Test:  {model.score(X_test, y_test)}")
```

### 2. Не використовувати validation set

```python
# ❌ Tuning на test set
for depth in [1, 5, 10, 20]:
    dt = DecisionTreeRegressor(max_depth=depth)
    dt.fit(X_train, y_train)
    score = dt.score(X_test, y_test)  # Витік інформації!

# ✅ Використовуй validation або CV
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(DecisionTreeRegressor(), 
                    {'max_depth': [1, 5, 10, 20]}, 
                    cv=5)
grid.fit(X_train, y_train)
```

### 3. Збільшувати складність без діагностики

```python
# ❌ "Модель погана, додамо більше features!"
# Якщо вже є overfitting, це тільки погіршить!

# ✅ Спочатку діагностуй
diagnose_and_fix(model, X_train, X_test, y_train, y_test)
# Потім дій відповідно до діагнозу
```

### 4. Ігнорувати early stopping signs

```python
# ❌ Тренувати до кінця epochs незалежно від validation
model.fit(X, y, epochs=1000)

# ✅ Early stopping
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X, y, validation_split=0.2, callbacks=[early_stop])
```

### 5. Не візуалізувати результати

```python
# ❌ Дивитися тільки на числа
print(f"R² = {score}")

# ✅ Будувати графіки
plt.scatter(y_test, model.predict(X_test))
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
```

---

## Практичні поради 💡

### 1. Завжди використовуй Train/Val/Test split

```python
# 60% train, 20% validation, 20% test
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# Train → навчання моделі
# Validation → підбір гіперпараметрів
# Test → фінальна оцінка (торкаємося ОДИН раз!)
```

### 2. Baseline спочатку

```python
# Завжди починай з простої моделі
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
baseline = dummy.score(X_test, y_test)

print(f"Baseline R²: {baseline:.3f}")
# Будь-яка модель має бути краще за baseline!
```

### 3. Регулярна діагностика

```python
# Після кожної зміни моделі
def quick_check(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    train = model.score(X_tr, y_tr)
    test = model.score(X_te, y_te)
    print(f"Train: {train:.3f} | Test: {test:.3f} | Gap: {train-test:.3f}")

quick_check(model, X_train, X_test, y_train, y_test)
```

### 4. Документуй експерименти

```python
import pandas as pd

results = []

for max_depth in [1, 3, 5, 7, 10, None]:
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    
    results.append({
        'max_depth': max_depth,
        'train_r2': dt.score(X_train, y_train),
        'val_r2': dt.score(X_val, y_val),
        'gap': dt.score(X_train, y_train) - dt.score(X_val, y_val)
    })

df = pd.DataFrame(results)
print(df.sort_values('val_r2', ascending=False))
```

### 5. Cross-Validation для надійності

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R²: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# Якщо std великий → нестабільна модель
```

---

## Реальний приклад: Класифікація

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження даних
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print("="*70)
print("OVERFITTING vs UNDERFITTING: Breast Cancer Classification")
print("="*70)
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")

# Розділення
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Моделі для тестування
models = {
    'Shallow Tree (depth=1)': DecisionTreeClassifier(max_depth=1, random_state=42),
    'Medium Tree (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Deep Tree (no limit)': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Model: {name}")
    print(f"{'='*70}")
    
    # Навчання
    model.fit(X_train, y_train)
    
    # Оцінка
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    gap = train_acc - val_acc
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Gap:            {gap:.4f}")
    
    # Діагноз
    if val_acc < 0.85 and gap < 0.05:
        diagnosis = "🔴 UNDERFITTING"
    elif gap > 0.1:
        diagnosis = "🔴 OVERFITTING"
    else:
        diagnosis = "✅ GOOD BALANCE"
    
    print(f"Diagnosis: {diagnosis}")
    
    results.append({
        'Model': name,
        'Train Acc': train_acc,
        'Val Acc': val_acc,
        'Test Acc': test_acc,
        'Gap': gap,
        'Diagnosis': diagnosis
    })

# Підсумкова таблиця
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Візуалізація
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy comparison
model_names = df_results['Model'].values
train_accs = df_results['Train Acc'].values
val_accs = df_results['Val Acc'].values
test_accs = df_results['Test Acc'].values

x = np.arange(len(model_names))
width = 0.25

axes[0, 0].bar(x - width, train_accs, width, label='Train', alpha=0.8)
axes[0, 0].bar(x, val_accs, width, label='Validation', alpha=0.8)
axes[0, 0].bar(x + width, test_accs, width, label='Test', alpha=0.8)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Model Comparison', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Gap analysis
gaps = df_results['Gap'].values
colors = ['red' if g > 0.1 else 'green' if g < 0.05 else 'orange' for g in gaps]

axes[0, 1].barh(model_names, gaps, color=colors, alpha=0.7)
axes[0, 1].axvline(x=0.05, color='green', linestyle='--', label='Good (<0.05)')
axes[0, 1].axvline(x=0.1, color='orange', linestyle='--', label='Warning (>0.1)')
axes[0, 1].set_xlabel('Gap (Train - Val)', fontsize=11)
axes[0, 1].set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Learning curves для Random Forest
print("\nGenerating learning curves for Random Forest...")
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

axes[1, 0].plot(train_sizes, train_mean, 'o-', linewidth=2, label='Train')
axes[1, 0].fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
axes[1, 0].plot(train_sizes, val_mean, 's-', linewidth=2, label='Validation')
axes[1, 0].fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1)
axes[1, 0].set_xlabel('Training Set Size', fontsize=11)
axes[1, 0].set_ylabel('Accuracy', fontsize=11)
axes[1, 0].set_title('Learning Curves: Random Forest', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Bias-Variance map
axes[1, 1].scatter(gaps, val_accs, s=200, alpha=0.6, c=range(len(gaps)), cmap='viridis')

for i, name in enumerate(model_names):
    axes[1, 1].annotate(name.split('(')[0].strip(), 
                       (gaps[i], val_accs[i]), 
                       fontsize=8, ha='center')

axes[1, 1].axvline(x=0.05, color='green', linestyle='--', alpha=0.5)
axes[1, 1].axhline(y=0.9, color='blue', linestyle='--', alpha=0.5)

axes[1, 1].text(0.02, 0.92, 'Excellent', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[1, 1].text(0.15, 0.88, 'High Variance', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

axes[1, 1].set_xlabel('Gap (Train - Val)', fontsize=11)
axes[1, 1].set_ylabel('Validation Accuracy', fontsize=11)
axes[1, 1].set_title('Bias-Variance Map', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
```

---

## Пов'язані теми

- [[01_Bias_Variance_Tradeoff]] — теоретична основа
- [[03_Train_Test_Split]] — правильне розділення даних
- [[04_Cross_Validation]] — надійна валідація
- [[03_Regularization]] — методи боротьби з overfitting
- [[02_Random_Forest]] — зменшення variance
- [[Learning_Curves]] — візуалізація проблем

## Ресурси

- [Machine Learning Mastery: Overfitting and Underfitting](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)
- [Towards Data Science: Understanding Overfitting](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
- [Andrew Ng: Advice for Applying ML](https://www.coursera.org/learn/machine-learning)

---

## Ключові висновки

> **Underfitting** — модель занадто проста, не вловлює паттерни. **Overfitting** — модель занадто складна, запам'ятовує шум. **Оптимальна модель** — знаходить баланс.

**Діагностика:**
| Проблема | Train | Test | Gap | Рішення |
|----------|-------|------|-----|---------|
| Underfitting | Низький | Низький | Малий | ↑ Складність |
| Overfitting | Високий | Низький | Великий | ↑ Регуляризація |
| Optimal | Високий | Високий | Малий | ✅ |

**Ключові інструменти:**
- Train/Val/Test split — правильна валідація
- Learning Curves — візуальна діагностика
- Cross-Validation — надійна оцінка
- Regularization — контроль складності

**Найважливіше:**
- Завжди перевіряй train **І** test scores
- Візуалізуй learning curves
- Починай просто (baseline)
- Додавай складність поступово
- Регулярно діагностуй
- Документуй експерименти

---

#ml #core-concepts #overfitting #underfitting #model-selection #diagnostics #regularization
