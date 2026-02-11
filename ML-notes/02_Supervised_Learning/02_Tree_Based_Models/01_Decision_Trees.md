# Decision Trees (Дерева рішень)

## Що це?

**Decision Trees (Дерева рішень)** — це алгоритм supervised learning, який приймає рішення на основі послідовності запитань (умов) про ознаки даних, формуючи деревоподібну структуру.

**Головна ідея:** розбивати дані на підгрупи за допомогою послідовності простих правил типу "якщо-то" до тих пір, поки не отримаємо достатньо однорідні групи для передбачення.

## Навіщо потрібні?

- 🌳 **Інтерпретованість** — легко пояснити рішення (візуалізація дерева)
- 🎯 **Універсальність** — регресія та класифікація
- 🔧 **Мінімальна підготовка даних** — не потрібна нормалізація
- 📊 **Нелінійні залежності** — автоматично виявляє складні паттерни
- 🚀 **Швидкість** — швидкі передбачення (O(log n))
- 💡 **Feature importance** — показує важливість ознак

## Коли використовувати?

**Потрібно:**
- **Інтерпретованість критична** — медицина, банківська справа, юриспруденція
- Категоріальні та числові ознаки разом
- **Нелінійні залежності** та складні взаємодії
- Потрібна **візуалізація** процесу прийняття рішення
- Дані мають **природну ієрархічну структуру**

**Не потрібно:**
- Потрібна максимальна точність → **Random Forest, Gradient Boosting**
- Лінійна залежність → Linear/Logistic Regression
- Дуже високорозмірні дані → dimensionality reduction + інші методи

---

## Структура дерева

### Термінологія

```
                    [Root Node]               ← Корінь (вся вибірка)
                    Age <= 30?
                   /          \
                 Yes           No
                /                \
        [Internal Node]      [Internal Node]  ← Внутрішні вузли
        Income <= 50k?        Income <= 70k?
         /        \              /         \
       Yes        No           Yes         No
       /            \           /             \
  [Leaf]         [Leaf]     [Leaf]         [Leaf]  ← Листки (передбачення)
  Class: 0       Class: 1   Class: 1       Class: 0
```

**Компоненти:**
- **Root Node** (корінь) — перший вузол, містить всі дані
- **Internal Nodes** (внутрішні вузли) — умови розбиття
- **Branches** (гілки) — результати умов (Yes/No)
- **Leaf Nodes** (листки) — фінальні передбачення
- **Depth** (глибина) — максимальна довжина шляху від кореня до листка

### Приклад: Схвалення кредиту

```
                    Зарплата <= 50k?
                   /                \
                 Так                  Ні
                /                      \
        Кредитний                  Вік <= 25?
        рейтинг <= 600?            /        \
         /        \               Так       Ні
       Так        Ні              /           \
       /            \        Відмовити    Схвалити
  Відмовити    Схвалити    (Class: 0)   (Class: 1)
  (Class: 0)   (Class: 1)
```

**Інтерпретація шляхів:**
- **Шлях 1:** Зарплата > 50k AND Вік > 25 → **Схвалити** ✓
- **Шлях 2:** Зарплата > 50k AND Вік ≤ 25 → **Відмовити** ✗
- **Шлях 3:** Зарплата ≤ 50k AND Рейтинг > 600 → **Схвалити** ✓
- **Шлях 4:** Зарплата ≤ 50k AND Рейтинг ≤ 600 → **Відмовити** ✗

---

## Як будується дерево?

### Жадібний алгоритм (Greedy Algorithm)

**Рекурсивний процес:**

1. **Почати з кореня** — усі дані в одному вузлі
2. **Знайти найкращу ознаку та поріг** для розбиття:
   - Перебрати всі ознаки
   - Для кожної ознаки перебрати можливі пороги
   - Вибрати розбиття з найбільшим information gain / найменшою impurity
3. **Розділити дані** на дві групи (ліву та праву гілки)
4. **Рекурсивно повторити** для кожної гілки
5. **Зупинитися** коли виконано критерій зупинки

### Критерії зупинки

Алгоритм зупиняється, коли:

- **Досягнута максимальна глибина** (`max_depth`)
- **Мінімальна кількість зразків у вузлі** (`min_samples_split`)
- **Мінімальна кількість зразків у листку** (`min_samples_leaf`)
- **Чистота 100%** — всі зразки одного класу
- **Не можна покращити** — information gain = 0

---

## Критерії розбиття (Splitting Criteria)

### Для класифікації

## 1. Gini Impurity (Gini Index)

### Формула

$$\text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2$$

де:
- $D$ — вузол (набір даних)
- $C$ — кількість класів
- $p_i$ — ймовірність класу $i$ у вузлі

### Діапазон

- **Gini = 0** → вузол **чистий** (всі зразки одного класу) ✓
- **Gini → max** → вузол **нечистий** (рівномірний розподіл класів) ✗

### Приклад

**Вузол:** 100 зразків, 60 класу A, 40 класу B

$$\text{Gini} = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48$$

**Після розбиття:**

**Ліва гілка:** 70 зразків, 55 класу A, 15 класу B
$$\text{Gini}_{\text{left}} = 1 - \left(\frac{55}{70}\right)^2 - \left(\frac{15}{70}\right)^2 = 1 - 0.617 - 0.046 = 0.337$$

**Права гілка:** 30 зразків, 5 класу A, 25 класу B
$$\text{Gini}_{\text{right}} = 1 - \left(\frac{5}{30}\right)^2 - \left(\frac{25}{30}\right)^2 = 1 - 0.028 - 0.694 = 0.278$$

**Зважений Gini після розбиття:**
$$\text{Gini}_{\text{split}} = \frac{70}{100} \times 0.337 + \frac{30}{100} \times 0.278 = 0.236 + 0.083 = 0.319$$

**Information Gain (зменшення Gini):**
$$\Delta \text{Gini} = 0.48 - 0.319 = 0.161$$

✓ Розбиття **покращує чистоту**!

---

## 2. Entropy (Information Gain)

### Формула Entropy

$$\text{Entropy}(D) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

де:
- $p_i$ — ймовірність класу $i$

### Діапазон

- **Entropy = 0** → вузол чистий (всі одного класу) ✓
- **Entropy → max** → максимальна невизначеність ✗

### Information Gain (Інформаційний виграш)

$$\text{IG}(D, A) = \text{Entropy}(D) - \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} \text{Entropy}(D_v)$$

де:
- $A$ — ознака для розбиття
- $D_v$ — підмножина даних після розбиття за значенням $v$

### Приклад

**Той самий вузол:** 60 A, 40 B

$$\text{Entropy} = -0.6 \log_2(0.6) - 0.4 \log_2(0.4)$$
$$= -0.6 \times (-0.737) - 0.4 \times (-1.322)$$
$$= 0.442 + 0.529 = 0.971$$

**Після розбиття:**

$$\text{Entropy}_{\text{left}} = -\frac{55}{70} \log_2\left(\frac{55}{70}\right) - \frac{15}{70} \log_2\left(\frac{15}{70}\right) = 0.779$$

$$\text{Entropy}_{\text{right}} = -\frac{5}{30} \log_2\left(\frac{5}{30}\right) - \frac{25}{30} \log_2\left(\frac{25}{30}\right) = 0.650$$

**Зважена Entropy після розбиття:**
$$\text{Entropy}_{\text{split}} = 0.7 \times 0.779 + 0.3 \times 0.650 = 0.545 + 0.195 = 0.740$$

**Information Gain:**
$$\text{IG} = 0.971 - 0.740 = 0.231$$

✓ Вибираємо розбиття з **найбільшим IG**!

---

## 3. Gini vs Entropy: що вибрати?

| Критерій | Gini Impurity | Entropy (IG) |
|----------|---------------|--------------|
| **Обчислення** | Швидше (без логарифмів) | Повільніше |
| **Чутливість** | Менш чутлива до змін | Більш чутлива |
| **Результати** | Зазвичай дуже схожі | Зазвичай дуже схожі |
| **За замовчуванням** | **sklearn** | CART, ID3, C4.5 |
| **Використання** | **Рекомендовано** (швидше) | Традиційний підхід |

**Висновок:** На практиці різниця мінімальна. **Gini** трохи швидше, тому за замовчуванням у scikit-learn.

---

### Для регресії

## 1. MSE (Mean Squared Error)

### Формула

$$\text{MSE}(D) = \frac{1}{|D|} \sum_{i \in D} (y_i - \bar{y})^2$$

де:
- $\bar{y}$ — середнє значення $y$ у вузлі

### Приклад

**Вузол:** [10, 20, 30, 40, 50]
$$\bar{y} = 30$$
$$\text{MSE} = \frac{1}{5}[(10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 + (50-30)^2]$$
$$= \frac{1}{5}[400 + 100 + 0 + 100 + 400] = \frac{1000}{5} = 200$$

**Після розбиття (x ≤ 25):**

**Ліва:** [10, 20] → $\bar{y}_L = 15$ → MSE = 25
**Права:** [30, 40, 50] → $\bar{y}_R = 40$ → MSE = 66.67

**Зважена MSE:**
$$\text{MSE}_{\text{split}} = \frac{2}{5} \times 25 + \frac{3}{5} \times 66.67 = 10 + 40 = 50$$

**Зменшення MSE:**
$$\Delta \text{MSE} = 200 - 50 = 150$$ ✓

---

## 2. MAE (Mean Absolute Error)

### Формула

$$\text{MAE}(D) = \frac{1}{|D|} \sum_{i \in D} |y_i - \text{median}(D)|$$

**Відмінність від MSE:**
- Використовує **median** замість mean
- Менш чутлива до **outliers**
- **L1** норма замість L2

### Коли використовувати MAE?

✅ Багато викидів у цільовій змінній
✅ Потрібна робастність
✅ Outliers не повинні сильно впливати

---

## Простий приклад: Класифікація "Грати у теніс?"

### Дані

| День | Погода | Температура | Вологість | Вітер | Грати? |
|------|---------|-------------|-----------|-------|--------|
| 1 | Сонячно | Спекотно | Висока | Ні | Ні |
| 2 | Сонячно | Спекотно | Висока | Так | Ні |
| 3 | Хмарно | Спекотно | Висока | Ні | Так |
| 4 | Дощ | Помірно | Висока | Ні | Так |
| 5 | Дощ | Прохолодно | Норма | Ні | Так |
| 6 | Дощ | Прохолодно | Норма | Так | Ні |
| 7 | Хмарно | Прохолодно | Норма | Так | Так |
| 8 | Сонячно | Помірно | Висока | Ні | Ні |
| 9 | Сонячно | Прохолодно | Норма | Ні | Так |
| 10 | Дощ | Помірно | Норма | Ні | Так |

**Розподіл:** 5 "Так", 5 "Ні"

### Крок 1: Обчислити Entropy кореня

$$\text{Entropy}_{\text{root}} = -\frac{5}{10} \log_2\left(\frac{5}{10}\right) - \frac{5}{10} \log_2\left(\frac{5}{10}\right) = 1.0$$

### Крок 2: Знайти найкращу ознаку

**Спробуємо "Погода":**

**Сонячно** (3 дні): 1 "Так", 2 "Ні"
$$\text{Entropy}_{\text{Сонячно}} = -\frac{1}{3} \log_2\left(\frac{1}{3}\right) - \frac{2}{3} \log_2\left(\frac{2}{3}\right) = 0.918$$

**Хмарно** (2 дні): 2 "Так", 0 "Ні"
$$\text{Entropy}_{\text{Хмарно}} = 0$$ (чисто!)

**Дощ** (5 днів): 2 "Так", 3 "Ні"
$$\text{Entropy}_{\text{Дощ}} = -\frac{2}{5} \log_2\left(\frac{2}{5}\right) - \frac{3}{5} \log_2\left(\frac{3}{5}\right) = 0.971$$

**Зважена Entropy:**
$$\text{Entropy}_{\text{split}} = \frac{3}{10} \times 0.918 + \frac{2}{10} \times 0 + \frac{5}{10} \times 0.971 = 0.761$$

**Information Gain:**
$$\text{IG}_{\text{Погода}} = 1.0 - 0.761 = 0.239$$

Аналогічно обчислюємо для інших ознак і вибираємо з найбільшим IG.

### Результат дерева

```
                    Погода?
                 /      |      \
            Сонячно  Хмарно    Дощ
               |        |        |
           Вологість  [Так]   Вітер?
            /    \            /    \
        Висока Норма        Ні    Так
          |      |          |      |
        [Ні]   [Так]      [Так]  [Ні]
```

---

## Складний приклад: Передбачення зарплати

### Дані

200 працівників з ознаками:

| Ознака | Тип | Діапазон |
|--------|-----|----------|
| Years_Experience | Числова | 0-20 |
| Education_Level | Категоріальна | [School, Bachelor, Master, PhD] |
| Age | Числова | 22-65 |
| City | Категоріальна | [Kyiv, Lviv, Dnipro] |
| **Salary** | Числова (target) | 20-200 тис. $ |

### Дерево регресії

```
                Years_Experience <= 5?
                /                    \
              Так                     Ні
             /                          \
    Education = School?          Years_Experience <= 10?
      /              \              /                  \
    Так              Ні           Так                  Ні
     |                |            |                    |
  [Pred: 35k]    [Pred: 55k]  [Pred: 85k]         Education = PhD?
                                                    /            \
                                                  Так            Ні
                                                   |              |
                                               [Pred: 150k]   [Pred: 110k]
```

**Інтерпретація:**
- **Досвід ≤ 5 років + School → 35k**
- **Досвід ≤ 5 років + вища освіта → 55k**
- **Досвід 5-10 років → 85k**
- **Досвід > 10 років + PhD → 150k**
- **Досвід > 10 років + інша освіта → 110k**

---

## Код (Python + scikit-learn)

### Класифікація

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Завантаження даних
iris = load_iris()
X = iris.data
y = iris.target

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Створення моделі
dt_clf = DecisionTreeClassifier(
    criterion='gini',        # або 'entropy'
    max_depth=3,             # Обмеження глибини
    min_samples_split=2,     # Мін. зразків для розбиття
    min_samples_leaf=1,      # Мін. зразків у листку
    random_state=42
)

# 3. Навчання
dt_clf.fit(X_train, y_train)

# 4. Передбачення
y_pred = dt_clf.predict(X_test)

# 5. Оцінка
print("=== Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 6. Важливість ознак
print("\n=== Feature Importance ===")
for name, importance in zip(iris.feature_names, dt_clf.feature_importances_):
    print(f"{name}: {importance:.4f}")

# 7. Візуалізація дерева
plt.figure(figsize=(20, 10))
plot_tree(
    dt_clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree - Iris Classification', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Регресія

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Генерація даних
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Модель
dt_reg = DecisionTreeRegressor(
    criterion='squared_error',  # або 'absolute_error', 'friedman_mse'
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Навчання
dt_reg.fit(X_train, y_train)

# Передбачення
y_pred = dt_reg.predict(X_test)

# Оцінка
print("=== Regression Metrics ===")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# Візуалізація
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_plot = dt_reg.predict(X_plot)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.5, s=30, label='Train', color='blue')
plt.scatter(X_test, y_test, alpha=0.5, s=50, label='Test', color='green')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Decision Tree')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Decision Tree Regression', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Гіперпараметри

### Основні параметри контролю складності

| Параметр | Опис | Типові значення | Ефект |
|----------|------|-----------------|-------|
| **max_depth** | Максимальна глибина дерева | 3-10 (або None) | Обмежує складність |
| **min_samples_split** | Мін. зразків для розбиття | 2-20 | Запобігає overfitting |
| **min_samples_leaf** | Мін. зразків у листку | 1-10 | Згладжує передбачення |
| **max_features** | Макс. ознак при розбитті | 'sqrt', 'log2', None | Feature subsampling |
| **max_leaf_nodes** | Макс. кількість листків | 10-100 | Обмежує складність |

### Критерії розбиття

| Параметр | Класифікація | Регресія |
|----------|--------------|----------|
| **criterion** | 'gini', 'entropy' | 'squared_error', 'absolute_error' |

### Інші параметри

```python
DecisionTreeClassifier(
    splitter='best',           # або 'random' (для стохастичності)
    class_weight='balanced',   # Ваги класів для незбалансованих даних
    min_impurity_decrease=0.0, # Мін. зменшення impurity для розбиття
    ccp_alpha=0.0              # Cost-complexity pruning
)
```

---

## Підбір гіперпараметрів

### Grid Search CV

```python
from sklearn.model_selection import GridSearchCV

# Сітка параметрів
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Grid Search
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Кращі параметри
print("Best parameters:")
print(grid_search.best_params_)
print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")

# Кращая модель
best_dt = grid_search.best_estimator_

# Оцінка на тесті
test_score = best_dt.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
```

### Randomized Search (швидше)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Розподіли параметрів
param_distributions = {
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy']
}

# Randomized Search
random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions,
    n_iter=50,              # Кількість комбінацій
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
```

---

## Overfitting та Pruning

### Проблема Overfitting

**Без обмежень дерево може:**
- Розділятися до тих пір, поки кожен лист не матиме 1 зразок
- Ідеально "запам'ятає" тренувальні дані (Train Acc = 100%)
- Погано узагальнюватиметься на нових даних

```
Overfitted Tree:

Train Accuracy: 100%
Test Accuracy: 65%  ← ПОГАНО!

Дерево дуже глибоке з багатьма листками,
кожен з яких пристосований до конкретних прикладів.
```

### Pre-Pruning (Обмеження під час побудови)

**Встановлюємо обмеження ДО навчання:**

```python
# Обмеження складності
dt = DecisionTreeClassifier(
    max_depth=5,              # Макс. глибина
    min_samples_split=10,     # Мін. зразків для розбиття
    min_samples_leaf=5,       # Мін. зразків у листку
    max_leaf_nodes=20         # Макс. листків
)
```

### Post-Pruning (Cost-Complexity Pruning)

**Спочатку будуємо повне дерево, потім "обрізаємо" непотрібні гілки:**

```python
# 1. Навчити повне дерево
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

# 2. Знайти оптимальний ccp_alpha через cross-validation
path = dt_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# 3. Тренувати дерева з різними alpha
train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

# 4. Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Alpha vs Accuracy
axes[0].plot(ccp_alphas, train_scores, marker='o', label='Train', linewidth=2)
axes[0].plot(ccp_alphas, test_scores, marker='s', label='Test', linewidth=2)
axes[0].set_xlabel('ccp_alpha', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy vs ccp_alpha', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Alpha vs Tree Size
node_counts = [dt.tree_.node_count for dt in 
               [DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_train, y_train) 
                for alpha in ccp_alphas]]

axes[1].plot(ccp_alphas, node_counts, marker='o', linewidth=2)
axes[1].set_xlabel('ccp_alpha', fontsize=12)
axes[1].set_ylabel('Number of Nodes', fontsize=12)
axes[1].set_title('Tree Size vs ccp_alpha', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Вибрати оптимальний alpha
optimal_idx = np.argmax(test_scores)
optimal_alpha = ccp_alphas[optimal_idx]
print(f"Optimal ccp_alpha: {optimal_alpha:.6f}")

# 6. Фінальна модель
dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
dt_pruned.fit(X_train, y_train)
print(f"Test Accuracy (pruned): {dt_pruned.score(X_test, y_test):.4f}")
```

---

## Feature Importance

### Як обчислюється?

**Важливість ознаки** = сума зменшень impurity, зважених на кількість зразків:

$$\text{Importance}(f) = \frac{\sum_{t \in \text{splits using } f} n_t \Delta I_t}{\sum_{t \in \text{all splits}} n_t \Delta I_t}$$

де:
- $n_t$ — кількість зразків у вузлі $t$
- $\Delta I_t$ — зменшення impurity після розбиття

**Властивості:**
- Сума всіх importance = 1.0
- Вища importance → ознака важливіша для передбачень
- Важливість 0 → ознака не використовувалась

### Код

```python
# Feature Importance
importances = dt_clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for i in range(X.shape[1]):
    print(f"{i+1}. {iris.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices], rotation=45)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importances', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Інтерпретованість** | Легко візуалізувати та пояснити |
| **Не потрібна нормалізація** | Працює з ознаками різних масштабів |
| **Категоріальні дані** | Обробляє без One-Hot Encoding |
| **Нелінійні залежності** | Автоматично виявляє |
| **Feature interactions** | Виявляє взаємодії між ознаками |
| **Швидкість передбачення** | O(log n) — дуже швидко |
| **Feature importance** | Показує важливість ознак |
| **Універсальність** | Регресія та класифікація |
| **Робастність до викидів** | Менш чутливі за лінійні моделі |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Overfitting** | Легко перенавчаються без обмежень |
| **Нестабільність** | Малі зміни даних → різні дерева |
| **Bias в ознаках** | Надає перевагу ознакам з більше значень |
| **Не екстраполюють** | Погано за межами тренувальних даних |
| **Лінійні залежності** | Неефективні для простих лінійних паттернів |
| **XOR problem** | Складно з діагональними границями |
| **Точність** | Поступаються ансамблям (RF, GBM) |

---

## Порівняння з іншими моделями

| Модель | Інтерпретованість | Точність | Швидкість | Підготовка даних |
|--------|-------------------|----------|-----------|------------------|
| **Decision Tree** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Linear Regression | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Random Forest | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Gradient Boosting | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Neural Networks | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## Коли використовувати Decision Trees

### Ідеально підходить ✓

- **Інтерпретованість критична** — медицина, фінанси, юриспруденція
- Потрібна **візуалізація** процесу прийняття рішень
- Категоріальні + числові ознаки
- **Швидкий baseline** перед ансамблями
- Малі/середні датасети
- **Дослідження даних** — feature importance

### Краще використати інше ✗

- Потрібна **максимальна точність** → **Random Forest, XGBoost**
- Лінійна залежність → Linear/Logistic Regression
- Дуже великі дані → Linear models, Neural Networks
- Потрібна **стабільність** → Ансамблі

---

## Практичні поради 💡

1. **Почни з обмежень** — встанови `max_depth=5-7` одразу
2. **Grid Search** — знайди оптимальні гіперпараметри
3. **Візуалізуй дерево** — перевір, чи має сенс
4. **Feature importance** — видали непотрібні ознаки
5. **Cost-Complexity Pruning** — для фінального tuning
6. **Порівняй з Random Forest** — часто RF краще
7. **class_weight='balanced'** для незбалансованих класів
8. **min_samples_leaf** — збільш для згладжування
9. **Не довіряй одному дереву** — використовуй ансамблі
10. **Експериментуй** — спробуй різні критерії (gini vs entropy)

---

## Поширені помилки ❌

### 1. Не обмежувати глибину

```python
# ❌ НЕПРАВИЛЬНО
dt = DecisionTreeClassifier()  # Overfitting майже гарантований

# ✅ ПРАВИЛЬНО
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
```

### 2. Використовувати для екстраполяції

```python
# Decision Trees НЕ ЕКСТРАПОЛЮЮТЬ
# Якщо train: X = [0, 10], а predict: X = 15
# Передбачення буде середнім найближчого листка, не продовженням тренду
```

### 3. Ігнорувати незбалансовані класи

```python
# ❌ НЕПРАВИЛЬНО
dt = DecisionTreeClassifier()

# ✅ ПРАВИЛЬНО
dt = DecisionTreeClassifier(class_weight='balanced')
```

### 4. Не візуалізувати дерево

```python
# Завжди дивись, що побудувало дерево!
plot_tree(dt, filled=True, feature_names=feature_names)
```

---

## Реальний приклад: Діагностика захворювань

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Синтетичні медичні дані
data = {
    'Age': [25, 45, 35, 50, 60, 30, 40, 55, 28, 52],
    'BP': [120, 140, 130, 150, 160, 125, 145, 155, 122, 148],  # Blood Pressure
    'Cholesterol': [200, 240, 220, 260, 280, 210, 250, 270, 205, 255],
    'BMI': [22, 28, 25, 30, 32, 23, 29, 31, 21, 30],
    'Disease': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  # 0=Healthy, 1=Disease
}

df = pd.DataFrame(data)

X = df.drop('Disease', axis=1)
y = df['Disease']

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Модель з обмеженнями
dt = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=2,
    criterion='gini',
    random_state=42
)

dt.fit(X_train, y_train)

# Текстове представлення дерева
tree_rules = export_text(dt, feature_names=list(X.columns))
print("=== Decision Tree Rules ===")
print(tree_rules)

# Приклад виходу:
# |--- Cholesterol <= 225.00
# |   |--- Age <= 32.50
# |   |   |--- class: 0 (Healthy)
# |   |--- Age >  32.50
# |   |   |--- class: 1 (Disease)
# |--- Cholesterol >  225.00
# |   |--- class: 1 (Disease)

# Оцінка
y_pred = dt.predict(X_test)
print("\n" + classification_report(y_test, y_pred, 
                                   target_names=['Healthy', 'Disease']))

# Feature Importance
print("\n=== Feature Importance ===")
for feature, importance in zip(X.columns, dt.feature_importances_):
    if importance > 0:
        print(f"{feature}: {importance:.4f}")
```

---

## Візуалізація Decision Boundaries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

# Генерація даних
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Різні max_depth
depths = [2, 5, 10, None]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, depth in enumerate(depths):
    # Модель
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X, y)
    
    # Сітка для візуалізації
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Передбачення на сітці
    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Візуалізація
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                     edgecolors='k', s=50)
    axes[idx].set_title(f'max_depth={depth}\nTrain Acc={dt.score(X, y):.3f}',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

---

## Пов'язані теми

- [[02_Random_Forest]] — ансамбль дерев
- [[03_Gradient_Boosting]] — послідовні дерева
- [[05_Ensemble_Methods]] — комбінування моделей
- [[06_Feature_Importance]] — аналіз ознак
- [[Information_Theory]] — Entropy, Information Gain

## Ресурси

- [Scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [CART Algorithm](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [Visualization Tool](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

---

## Ключові висновки

> Decision Trees будують деревоподібну структуру рішень через послідовні розбиття даних за ознаками для максимізації information gain або мінімізації impurity.

**Основні принципи:**
- Жадібний рекурсивний алгоритм
- Вибір найкращого розбиття на кожному кроці
- Критерії: Gini Impurity, Entropy (класифікація), MSE (регресія)
- Потребує обмежень для запобігання overfitting

**Формули:**
- **Gini:** $1 - \sum p_i^2$
- **Entropy:** $-\sum p_i \log_2(p_i)$
- **Information Gain:** $\text{Entropy}_{\text{parent}} - \text{weighted Entropy}_{\text{children}}$

**Коли використовувати:**
- Інтерпретованість + візуалізація + категоріальні дані = Decision Trees ✓
- Максимальна точність → Ансамблі (Random Forest, XGBoost) ✓

**Найважливіше:**
- ЗАВЖДИ обмежуй складність (`max_depth`, `min_samples_leaf`)
- Візуалізуй дерево для розуміння
- Використовуй як baseline, але розглядай ансамблі для production

---

#ml #supervised-learning #classification #regression #decision-trees #interpretability #tree-based
