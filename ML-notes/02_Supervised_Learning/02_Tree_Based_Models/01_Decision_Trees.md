# Decision Trees (Дерева рішень)

## Що це?

**Дерево рішень** — це алгоритм supervised learning, який приймає рішення шляхом послідовного розділення даних на основі питань про ознаки, формуючи деревоподібну структуру.

**Головна ідея:** модель як серія if-else правил, які легко зрозуміти людині — "якщо вік > 30 ТА дохід > 50k, то схвалити кредит".

## Навіщо потрібні?

- 🌳 **Інтерпретованість** — можна візуалізувати та пояснити кожне рішення
- 🎯 **Універсальність** — працює для регресії та класифікації
- 📊 **Нелінійні залежності** — автоматично виявляє складні патерни
- 🔧 **Мінімальна підготовка** — не потрібна нормалізація або encoding
- 💡 **Feature importance** — показує найважливіші ознаки
- 🚀 **Швидкість** — швидке передбачення

## Коли використовувати?

**Потрібно:**
- **Інтерпретованість критична** — медицина, фінанси, право
- Дані **змішаних типів** — числові + категоріальні
- **Нелінійні залежності** — складні interaction effects
- **Feature importance** — потрібно зрозуміти, що впливає
- Швидкий baseline перед ансамблями

**Не потрібно:**
- Потрібна **висока точність** → Random Forest, XGBoost
- Лінійні залежності → Linear/Logistic Regression
- Дуже **великі датасети** → може бути повільно
- **Екстраполяція** → погано працює за межами train даних

---

## Структура дерева

### Анатомія дерева

```
                    [Root Node]
                    Дохід ≤ 50k?
                    /          \
                 Так            Ні
                /                  \
        [Internal Node]        [Internal Node]
        Вік ≤ 25?              Освіта?
          /    \                /      \
       Так      Ні          Вища    Середня
       /          \           |         |
   [Leaf]      [Leaf]     [Leaf]    [Leaf]
   Відмова    Схвалити   Схвалити  Відмова
   (Class 0)  (Class 1)  (Class 1)  (Class 0)
```

### Компоненти

| Компонент | Опис | Приклад |
|-----------|------|---------|
| **Root Node** | Перший поділ (корінь) | Дохід ≤ 50k? |
| **Internal Node** | Проміжний поділ | Вік ≤ 25? |
| **Leaf Node** | Кінцеве рішення | Схвалити кредит |
| **Branch** | Гілка (результат питання) | Так/Ні, ≤/> |
| **Depth** | Кількість рівнів від кореня | Depth = 2 |
| **Split** | Правило поділу | Feature + threshold |

---

## Як працює? Алгоритм побудови

### Рекурсивний поділ (Recursive Binary Splitting)

**Алгоритм (Top-Down Greedy):**

1. **Почати з корня** (всі дані в одному вузлі)
2. **Для кожної ознаки та кожного порогу:**
   - Розділити дані на дві групи
   - Обчислити якість поділу (критерій)
3. **Вибрати найкращий поділ** (мінімізує impurity)
4. **Рекурсивно** повторити для лівої та правої частин
5. **Зупинитись**, коли досягнуто критерій:
   - Максимальна глибина
   - Мінімум прикладів у листі
   - Всі приклади одного класу (pure node)

### Критерії якості поділу

## 1. Для класифікації

### A) Gini Impurity (Індекс Джині)

**Формула:**
$$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$

де:
- $S$ — множина прикладів у вузлі
- $C$ — кількість класів
- $p_i$ — частка класу $i$ у вузлі

**Інтуїція:** ймовірність помилково класифікувати випадковий приклад, якби його клас вибрали випадково згідно з розподілом у вузлі.

**Діапазон:** $[0, 0.5]$ для бінарної класифікації
- **Gini = 0** → pure node (всі приклади одного класу) ✓
- **Gini = 0.5** → максимальна impurity (50/50)

**Приклад (бінарна класифікація):**

Вузол з 100 прикладами: 70 клас A, 30 клас B

$$\text{Gini} = 1 - (0.7^2 + 0.3^2) = 1 - (0.49 + 0.09) = 1 - 0.58 = 0.42$$

**Після поділу:**

Лівий вузол (60 прикладів): 55 A, 5 B
$$\text{Gini}_L = 1 - (0.917^2 + 0.083^2) = 1 - 0.848 = 0.152$$

Правий вузол (40 прикладів): 15 A, 25 B
$$\text{Gini}_R = 1 - (0.375^2 + 0.625^2) = 1 - 0.531 = 0.469$$

**Weighted Gini:**
$$\text{Gini}_{\text{split}} = \frac{60}{100} \times 0.152 + \frac{40}{100} \times 0.469 = 0.091 + 0.188 = 0.279$$

**Information Gain:**
$$\Delta\text{Gini} = 0.42 - 0.279 = 0.141$$

Чим більше $\Delta\text{Gini}$, тим кращий поділ! ✓

---

### B) Entropy (Ентропія) та Information Gain

**Формула ентропії:**
$$\text{Entropy}(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

**Інтуїція:** міра невизначеності або "безладу" у вузлі.

**Діапазон:** $[0, 1]$ для бінарної класифікації
- **Entropy = 0** → pure node (всі однакові)
- **Entropy = 1** → максимальна невизначеність (50/50)

**Приклад:**

Той же вузол: 70 A, 30 B

$$\text{Entropy} = -(0.7 \log_2(0.7) + 0.3 \log_2(0.3))$$
$$= -(0.7 \times (-0.515) + 0.3 \times (-1.737))$$
$$= -(-0.361 - 0.521) = 0.882$$

**Після поділу:**

Лівий: 55 A, 5 B
$$\text{Entropy}_L = -(0.917 \log_2(0.917) + 0.083 \log_2(0.083))$$
$$= 0.408$$

Правий: 15 A, 25 B
$$\text{Entropy}_R = -(0.375 \log_2(0.375) + 0.625 \log_2(0.625))$$
$$= 0.954$$

**Weighted Entropy:**
$$\text{Entropy}_{\text{split}} = 0.6 \times 0.408 + 0.4 \times 0.954 = 0.626$$

**Information Gain:**
$$\text{IG} = 0.882 - 0.626 = 0.256$$

Чим більше IG, тим кращий поділ!

---

### Порівняння Gini vs Entropy

| Критерій | Gini Impurity | Entropy |
|----------|---------------|---------|
| **Формула** | $1 - \sum p_i^2$ | $-\sum p_i \log_2(p_i)$ |
| **Обчислення** | Швидше ✓ | Повільніше (log) |
| **Результати** | Дуже схожі | Дуже схожі |
| **За замовчуванням** | scikit-learn | - |
| **Переваги** | Проста, швидка | Теоретично обґрунтована |
| **Використання** | Більш популярна | Information Theory |

**Висновок:** На практиці різниця мінімальна. Gini швидше, Entropy теоретично "чистіша".

---

## 2. Для регресії

### MSE (Mean Squared Error)

**Формула:**
$$\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2$$

де $\bar{y}$ — середнє значення $y$ у вузлі.

**Після поділу:**
$$\text{MSE}_{\text{split}} = \frac{|S_L|}{|S|} \text{MSE}(S_L) + \frac{|S_R|}{|S|} \text{MSE}(S_R)$$

**Reduction in MSE:**
$$\Delta\text{MSE} = \text{MSE}(S) - \text{MSE}_{\text{split}}$$

**Приклад:**

Вузол з 6 прикладами: $y = [10, 20, 15, 30, 25, 18]$

Середнє: $\bar{y} = 19.67$

$$\text{MSE} = \frac{1}{6}[(10-19.67)^2 + (20-19.67)^2 + ... + (18-19.67)^2]$$
$$= \frac{1}{6}[93.5 + 0.11 + 21.8 + 106.7 + 28.4 + 2.8] = 42.2$$

**Поділ:** $x \leq 17$

Лівий ($x \leq 17$): $y = [10, 15]$, $\bar{y}_L = 12.5$
$$\text{MSE}_L = \frac{1}{2}[(10-12.5)^2 + (15-12.5)^2] = 6.25$$

Правий ($x > 17$): $y = [20, 30, 25, 18]$, $\bar{y}_R = 23.25$
$$\text{MSE}_R = \frac{1}{4}[(20-23.25)^2 + (30-23.25)^2 + (25-23.25)^2 + (18-23.25)^2]$$
$$= 23.19$$

**Weighted MSE:**
$$\text{MSE}_{\text{split}} = \frac{2}{6} \times 6.25 + \frac{4}{6} \times 23.19 = 2.08 + 15.46 = 17.54$$

**Reduction:**
$$\Delta\text{MSE} = 42.2 - 17.54 = 24.66$$ ✓

Цей поділ суттєво зменшує MSE!

---

## Простий приклад: Схвалення кредиту

### Дані

| Дохід (тис. $) | Вік | Борг (тис. $) | Схвалено |
|----------------|-----|---------------|----------|
| 30 | 25 | 5 | 0 (Ні) |
| 50 | 35 | 10 | 1 (Так) |
| 40 | 28 | 15 | 0 (Ні) |
| 70 | 45 | 8 | 1 (Так) |
| 60 | 40 | 20 | 0 (Ні) |
| 80 | 50 | 5 | 1 (Так) |

### Побудова дерева

**Крок 1: Вибрати перший поділ (root)**

Розглянемо всі можливі поділи:

**Дохід ≤ 45:**
- Ліворуч: 30, 40 → 2 відмови
- Праворуч: 50, 70, 60, 80 → 2 схвалення, 1 відмова
- Gini = ...

**Дохід ≤ 55:**
- Ліворуч: 30, 50, 40 → 1 схвалення, 2 відмови
- Праворуч: 70, 60, 80 → 2 схвалення, 1 відмова
- Gini_left = $1 - (1/3)^2 - (2/3)^2 = 0.444$
- Gini_right = $1 - (2/3)^2 - (1/3)^2 = 0.444$
- Weighted Gini = $3/6 \times 0.444 + 3/6 \times 0.444 = 0.444$

**Борг ≤ 12:**
- Ліворуч: 5, 10, 8, 5 → 3 схвалення, 1 відмова
- Праворуч: 15, 20 → 2 відмови
- Gini_left = $1 - (3/4)^2 - (1/4)^2 = 0.375$
- Gini_right = $1 - 0 - 1 = 0$ (pure!)
- Weighted Gini = $4/6 \times 0.375 + 2/6 \times 0 = 0.25$ ✓

**Найкращий поділ:** Борг ≤ 12 (Gini = 0.25)

### Результуюче дерево

```
            [Root]
          Борг ≤ 12?
          /        \
       Так          Ні
       /              \
  [Internal]       [Leaf]
  Дохід ≤ 55?     Відмова
    /      \
  Так       Ні
  /           \
[Leaf]      [Leaf]
Відмова   Схвалити
```

---

## Складний приклад: Передбачення ціни будинку (регресія)

### Дані

| Площа (м²) | Кімнат | Район | Ціна (тис. $) |
|------------|--------|-------|---------------|
| 50 | 1 | A | 100 |
| 80 | 2 | B | 150 |
| 100 | 3 | A | 200 |
| 120 | 3 | B | 250 |
| 150 | 4 | A | 300 |
| 180 | 4 | B | 350 |

### Побудова дерева регресії

**Root node:** Всі 6 прикладів, $\bar{y} = 225$

**MSE(root) = $\frac{1}{6}[(100-225)^2 + (150-225)^2 + ... + (350-225)^2]$**
**= $\frac{1}{6}[15625 + 5625 + 625 + 625 + 5625 + 15625] = 7291.67$**

**Кандидати поділу:**

**Площа ≤ 90:**
- Лівий: 50, 80 → Ціни: 100, 150, $\bar{y}_L = 125$
- Правий: 100, 120, 150, 180 → Ціни: 200, 250, 300, 350, $\bar{y}_R = 275$
- MSE_L = 625
- MSE_R = 3125
- Weighted MSE = $2/6 \times 625 + 4/6 \times 3125 = 2291.67$
- Reduction = $7291.67 - 2291.67 = 5000$ ✓

**Кімнат ≤ 2:**
- Лівий: 1, 2 → Ціни: 100, 150, $\bar{y}_L = 125$
- Правий: 3, 3, 4, 4 → Ціни: 200, 250, 300, 350, $\bar{y}_R = 275$
- (Аналогічно до Площа ≤ 90)

**Найкращий поділ:** Площа ≤ 90

### Передбачення

Для нового будинку: Площа = 110 м², Кімнат = 3

```
Площа ≤ 90?
    → Ні (110 > 90)
    → Йдемо праворуч
    → Передбачення: $\bar{y}_R = 275$ тис. $
```

---

## Код (Python + scikit-learn)

### Класифікація

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification

# 1. Генерація даних
X, y = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Модель Decision Tree
tree_clf = DecisionTreeClassifier(
    criterion='gini',           # або 'entropy'
    max_depth=3,                # Максимальна глибина
    min_samples_split=20,       # Мінімум для поділу
    min_samples_leaf=10,        # Мінімум у листі
    random_state=42
)

tree_clf.fit(X_train, y_train)

# 3. Передбачення
y_pred = tree_clf.predict(X_test)
y_proba = tree_clf.predict_proba(X_test)

# 4. Оцінка
print("=== Classification Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n{classification_report(y_test, y_pred)}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Feature Importance
print("\n=== Feature Importance ===")
for i, importance in enumerate(tree_clf.feature_importances_):
    print(f"Feature {i}: {importance:.4f}")

# 6. Структура дерева
print(f"\n=== Tree Structure ===")
print(f"Number of leaves: {tree_clf.get_n_leaves()}")
print(f"Tree depth: {tree_clf.get_depth()}")

# 7. Візуалізація дерева
plt.figure(figsize=(20, 10))
plot_tree(
    tree_clf,
    feature_names=['Feature 0', 'Feature 1'],
    class_names=['Class 0', 'Class 1'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Регресія

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression

# Дані
X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Модель
tree_reg = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

tree_reg.fit(X_train, y_train)

# Передбачення
y_pred = tree_reg.predict(X_test)

# Метрики
print("=== Regression Metrics ===")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# Візуалізація
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_plot = tree_reg.predict(X_plot)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.4, s=30, label='Train')
plt.scatter(X_test, y_test, alpha=0.6, s=50, label='Test', color='green')
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

## Гіперпараметри Decision Trees

### Основні параметри

| Параметр | Опис | Діапазон | Рекомендації |
|----------|------|----------|--------------|
| **max_depth** | Максимальна глибина дерева | 1-∞ | 3-10 для простих, 10-20 для складних |
| **min_samples_split** | Мінімум прикладів для поділу | 2-∞ | 10-50 залежно від розміру даних |
| **min_samples_leaf** | Мінімум прикладів у листі | 1-∞ | 5-20 |
| **max_features** | Макс. ознак для поділу | int, float, auto | sqrt(n) для класифікації |
| **criterion** | Критерій поділу | gini, entropy | gini за замовчуванням |
| **max_leaf_nodes** | Макс. кількість листків | 2-∞ | Обмежує складність |
| **min_impurity_decrease** | Мін. зменшення impurity | 0.0-∞ | 0.01-0.1 для pruning |

### Вплив гіперпараметрів

```python
# Експеримент з різними max_depth
depths = [1, 2, 3, 5, 10, 20, None]
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_score = tree.score(X_train, y_train)
    test_score = tree.score(X_test, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Depth: {depth:>4} | Train: {train_score:.4f} | Test: {test_score:.4f} | Overfitting: {train_score - test_score:.4f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot([str(d) for d in depths], train_scores, 'o-', linewidth=2, label='Train Score')
plt.plot([str(d) for d in depths], test_scores, 's-', linewidth=2, label='Test Score')
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Effect of max_depth on Performance', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Типовий результат:**

```
Depth:    1 | Train: 0.8200 | Test: 0.8100 | Overfitting: 0.0100
Depth:    2 | Train: 0.8900 | Test: 0.8750 | Overfitting: 0.0150
Depth:    3 | Train: 0.9300 | Test: 0.9000 | Overfitting: 0.0300
Depth:    5 | Train: 0.9700 | Test: 0.8900 | Overfitting: 0.0800
Depth:   10 | Train: 0.9950 | Test: 0.8500 | Overfitting: 0.1450  ← Overfitting!
Depth:   20 | Train: 1.0000 | Test: 0.8200 | Overfitting: 0.1800  ← Сильний overfitting!
Depth: None | Train: 1.0000 | Test: 0.8000 | Overfitting: 0.2000  ← Найгірше!
```

**Висновок:** max_depth = 3-5 — оптимальний баланс!

---

## Pruning (Обрізання дерева)

### Проблема: Overfitting

```
Overfitting дерево:
                [Root]
              /    |    \
           /       |       \
        /          |          \
     [...]       [...]       [...]
    /  |  \     /  |  \     /  |  \
  [Leaf][...] [Leaf][...] [Leaf][...]
  
Занадто складне → запам'ятовує noise
```

### Рішення 1: Pre-pruning (Early Stopping)

**Зупинка під час побудови** через гіперпараметри:

```python
tree_pruned = DecisionTreeClassifier(
    max_depth=5,                    # Обмежити глибину
    min_samples_split=20,           # Мінімум для поділу
    min_samples_leaf=10,            # Мінімум у листі
    max_leaf_nodes=20,              # Макс. листків
    min_impurity_decrease=0.01,    # Мін. покращення
    random_state=42
)
```

### Рішення 2: Post-pruning (Cost Complexity Pruning)

**Обрізання після побудови** — scikit-learn підтримує через `ccp_alpha`:

```python
# 1. Знайти оптимальний ccp_alpha
path = tree_clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# 2. Тренувати дерева з різними alpha
train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# 3. Вибрати оптимальний alpha
best_alpha = ccp_alphas[np.argmax(test_scores)]
print(f"Best ccp_alpha: {best_alpha:.6f}")

# 4. Фінальна модель
tree_final = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
tree_final.fit(X_train, y_train)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Impurity vs alpha
axes[0].plot(ccp_alphas, impurities, marker='o')
axes[0].set_xlabel('ccp_alpha', fontsize=12)
axes[0].set_ylabel('Impurity', fontsize=12)
axes[0].set_title('Impurity vs alpha', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Scores vs alpha
axes[1].plot(ccp_alphas, train_scores, marker='o', label='Train')
axes[1].plot(ccp_alphas, test_scores, marker='s', label='Test')
axes[1].axvline(x=best_alpha, color='red', linestyle='--', label=f'Best alpha={best_alpha:.4f}')
axes[1].set_xlabel('ccp_alpha', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Accuracy vs alpha', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Feature Importance

### Як обчислюється?

**Для кожної ознаки:**
$$\text{Importance}(f) = \sum_{t \in \text{splits using } f} \frac{N_t}{N} \times \Delta\text{Impurity}_t$$

де:
- $N_t$ — кількість прикладів у вузлі $t$
- $N$ — загальна кількість прикладів
- $\Delta\text{Impurity}_t$ — зменшення impurity від поділу

**Нормалізація:** сума всіх importance = 1.0

### Код

```python
import pandas as pd

# Feature importance
feature_names = ['Дохід', 'Вік', 'Борг', 'Кредитний_рейтинг']
importances = tree_clf.feature_importances_

# DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_importance_df)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

**Приклад виходу:**

```
          Feature  Importance
0          Борг      0.5200
1    Дохід         0.2800
2          Вік       0.1500
3  Кредитний_рейтинг  0.0500
```

**Інтерпретація:**
- **Борг** — найважливіша ознака (52%)
- **Дохід** — друга за важливістю (28%)
- **Кредитний рейтинг** — майже не використовується (5%)

---

## Decision Boundaries (Межі рішення)

### Візуалізація для 2D даних

```python
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Візуалізація меж рішення для 2D даних"""
    h = 0.02  # Крок сітки
    
    # Створити сітку
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Передбачення для всієї сітки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Візуалізація
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), 
                edgecolor='black', s=50, alpha=0.7)
    plt.xlabel('Feature 0', fontsize=12)
    plt.ylabel('Feature 1', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Використання
tree_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_shallow.fit(X_train, y_train)
plot_decision_boundary(tree_shallow, X_train, y_train, "Decision Tree (depth=2)")

tree_deep = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_deep.fit(X_train, y_train)
plot_decision_boundary(tree_deep, X_train, y_train, "Decision Tree (depth=10)")
```

**Спостереження:**
- **Depth=2:** прості прямокутні області (underfitting можливо)
- **Depth=10:** дуже складні межі (overfitting!)

Decision Trees створюють **прямокутні** (axis-aligned) межі рішення!

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Інтерпретованість** | Легко візуалізувати та пояснити |
| **Мінімальна підготовка** | Не потрібна нормалізація, encoding |
| **Нелінійності** | Автоматично виявляє складні залежності |
| **Змішані типи** | Працює з числовими + категоріальними |
| **Feature importance** | Показує найважливіші ознаки |
| **Пропущені дані** | Може працювати з missing values |
| **Швидкість** | Швидке передбачення O(log n) |
| **Не чутливий до викидів** | Розділяє по рангам, не по значенням |
| **Без припущень** | Не припускає розподіл даних |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Overfitting** | Легко перенавчається без обмежень |
| **Нестабільність** | Малі зміни даних → інше дерево |
| **Bias до багатих класів** | Домінування частіших класів |
| **Axis-aligned splits** | Тільки прямокутні межі |
| **Не екстраполює** | Погано за межами train даних |
| **Локальний оптимум** | Greedy алгоритм (не глобальний) |
| **Великі дерева** | Можуть бути дуже складні |

---

## Порівняння з іншими алгоритмами

### Decision Trees vs Linear Models

| Критерій | Decision Trees | Linear Regression/Logistic |
|----------|----------------|----------------------------|
| **Лінійні залежності** | ❌ Неефективно | ✅ Ідеально |
| **Нелінійні залежності** | ✅ Автоматично | ❌ Потребує feature engineering |
| **Інтерпретованість** | ✅ Візуалізація | ✅ Коефіцієнти |
| **Підготовка даних** | ✅ Мінімальна | ❌ Нормалізація, encoding |
| **Overfitting** | ❌ Легко | ✅ Менше (з regularization) |
| **Екстраполяція** | ❌ Погано | ✅ Краще |
| **Точність** | ❌ Середня | ❌ Низька для нелінійних |

### Decision Trees vs Ensemble Methods

| Критерій | Single Tree | Random Forest | Gradient Boosting |
|----------|-------------|---------------|-------------------|
| **Точність** | ❌ Середня | ✅ Висока | ✅ Дуже висока |
| **Overfitting** | ❌ Високий ризик | ✅ Низький | ⚠️ Можливий |
| **Швидкість** | ✅ Дуже швидко | ⚠️ Середньо | ❌ Повільно |
| **Інтерпретованість** | ✅ Висока | ❌ Низька | ❌ Низька |
| **Стабільність** | ❌ Нестабільний | ✅ Стабільний | ✅ Стабільний |

---

## Коли використовувати Decision Trees

### Ідеально підходить ✓

- **Інтерпретованість критична** — медицина, право, фінанси
- **Швидкий baseline** — перед складнішими моделями
- **Експлораторний аналіз** — зрозуміти дані
- **Feature importance** — знайти ключові ознаки
- **Змішані типи даних** — числові + категоріальні
- **Невеликі/середні датасети**
- **Presentation для stakeholders** — легко пояснити

### Краще використати інше ✗

- **Потрібна висока точність** → Random Forest, XGBoost
- **Лінійні залежності** → Linear/Logistic Regression
- **Дуже великі дані** → SGD, online learning
- **Екстраполяція** → обережно, дерева погано екстраполюють
- **Виробництво (production)** → ансамблі стабільніші

---

## Практичні поради 💡

1. **Почни з невеликою глибиною** (3-5) — уникни overfitting
2. **Візуалізуй дерево** — зрозумій логіку моделі
3. **Feature importance** — знайди ключові ознаки
4. **Cross-validation** для підбору гіперпараметрів
5. **Pruning** — використовуй ccp_alpha або обмежуй глибину
6. **Не нормалізуй дані** — дерева не потребують
7. **min_samples_leaf** — збільши для згладжування
8. **Grid Search** для оптимальних параметрів
9. **Порівняй з Random Forest** — завжди перевір ансамбль
10. **Документуй рішення** — зберігай правила дерева

---

## Поширені помилки ❌

### 1. Дозволити повністю рости

```python
# ❌ НЕПРАВИЛЬНО
tree = DecisionTreeClassifier()  # Без обмежень → overfitting!

# ✅ ПРАВИЛЬНО
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10
)
```

### 2. Не перевіряти overfitting

```python
# ❌ НЕПРАВИЛЬНО
# Дивитись тільки на train accuracy

# ✅ ПРАВИЛЬНО
print(f"Train accuracy: {tree.score(X_train, y_train):.4f}")
print(f"Test accuracy: {tree.score(X_test, y_test):.4f}")
# Якщо Train >> Test → overfitting!
```

### 3. Використовувати для екстраполяції

```python
# ❌ НЕБЕЗПЕЧНО
# Передбачати далеко за межами train даних
# Дерево поверне найближчий лист, а не екстраполює!

# ✅ ПРАВИЛЬНО
if X_new > X_train.max():
    print("WARNING: Extrapolation! Use with caution.")
```

### 4. Нормалізувати дані

```python
# ❌ МАРНО (не шкодить, але не потрібно)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
tree.fit(X_scaled, y)

# ✅ ПРАВИЛЬНО (зекономити час)
tree.fit(X, y)  # Без нормалізації
```

---

## Візуалізація дерева

### Метод 1: plot_tree (scikit-learn)

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    tree_clf,
    feature_names=['Feature_1', 'Feature_2', 'Feature_3'],
    class_names=['Class_0', 'Class_1'],
    filled=True,           # Заповнити кольором
    rounded=True,          # Округлені рамки
    fontsize=10,
    proportion=True,       # Пропорції класів
    impurity=True,         # Показати impurity
    precision=2            # Точність чисел
)
plt.title('Decision Tree', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('tree_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Метод 2: Graphviz (більш красиво)

```python
from sklearn.tree import export_graphviz
import graphviz

# Експорт у DOT формат
dot_data = export_graphviz(
    tree_clf,
    out_file=None,
    feature_names=['Feature_1', 'Feature_2'],
    class_names=['Class_0', 'Class_1'],
    filled=True,
    rounded=True,
    special_characters=True
)

# Візуалізація
graph = graphviz.Source(dot_data)
graph.render('decision_tree', format='png', cleanup=True)
graph.view()
```

### Метод 3: Текстове представлення

```python
from sklearn.tree import export_text

tree_rules = export_text(
    tree_clf,
    feature_names=['Feature_1', 'Feature_2']
)

print(tree_rules)
```

**Приклад виходу:**

```
|--- Feature_1 <= 0.50
|   |--- Feature_2 <= -0.30
|   |   |--- class: 0
|   |--- Feature_2 >  -0.30
|   |   |--- class: 1
|--- Feature_1 >  0.50
|   |--- class: 1
```

---

## Робота з категоріальними змінними

### Scikit-learn вимагає числові дані

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Дані
df = pd.DataFrame({
    'Колір': ['червоний', 'зелений', 'синій', 'червоний'],
    'Розмір': ['S', 'M', 'L', 'M'],
    'Ціна': [10, 20, 30, 15],
    'Купили': [0, 1, 1, 0]
})

# Encoding категоріальних змінних
le_color = LabelEncoder()
le_size = LabelEncoder()

df['Колір_encoded'] = le_color.fit_transform(df['Колір'])
df['Розмір_encoded'] = le_size.fit_transform(df['Розмір'])

# Навчання
X = df[['Колір_encoded', 'Розмір_encoded', 'Ціна']]
y = df['Купили']

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)
```

**Альтернатива:** One-Hot Encoding для категорій без порядку

```python
# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Колір', 'Розмір'], drop_first=False)

X = df_encoded.drop('Купили', axis=1)
y = df_encoded['Купили']

tree.fit(X, y)
```

---

## Пов'язані теми

- [[02_Random_Forest]] — ансамбль дерев
- [[03_Gradient_Boosting]] — послідовний ансамбль
- [[05_Ensemble_Methods]] — загальна теорія
- [[06_Feature_Importance]] — аналіз важливості ознак
- [[Cross_Validation]] — оцінка моделей
- [[Hyperparameter_Tuning]] — Grid Search, Random Search

## Ресурси

- [Scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [CART: Classification and Regression Trees](https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418)
- [Interactive visualization](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

---

## Ключові висновки

> Decision Trees — інтерпретовані моделі, що будують деревоподібні правила рішень через рекурсивний поділ даних.

**Основні принципи:**
- Рекурсивно розділяє дані за ознаками
- Використовує Gini Impurity або Entropy для вибору поділів
- Створює прості if-else правила
- Легко візуалізувати та інтерпретувати

**Алгоритм:**
1. Вибрати найкращий поділ (мінімізує impurity)
2. Рекурсивно застосувати до підвузлів
3. Зупинитись при досягненні критерію

**Критерії поділу:**
- **Класифікація:** Gini Impurity або Entropy
- **Регресія:** MSE (Mean Squared Error)

**Коли використовувати:**
- Інтерпретованість + змішані дані + feature importance = Decision Trees ✓

**Важливо:**
- ЗАВЖДИ обмежуй глибину (max_depth)
- Перевіряй на overfitting (train vs test)
- Використовуй як baseline перед ансамблями
- Візуалізуй дерево для інсайтів

---

#ml #supervised-learning #decision-trees #classification #regression #interpretability
