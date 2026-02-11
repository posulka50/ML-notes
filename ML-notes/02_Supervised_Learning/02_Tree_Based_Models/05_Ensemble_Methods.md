# Ensemble Methods (Ансамблеві методи)

## Що це?

**Ensemble Methods** — це підхід у машинному навчанні, який комбінує **множину моделей** для отримання кращих передбачень, ніж будь-яка окрема модель.

**Головна ідея:** "мудрість натовпу" — багато різних, навіть слабких моделей разом дають сильні передбачення.

## Навіщо потрібні?

- 🎯 **Вища точність** — зменшення помилок
- 🛡️ **Робастність** — менша чутливість до шуму
- 📊 **Зменшення variance** — стабільніші передбачення
- 🔧 **Зменшення bias** — краще узагальнення
- 💡 **Універсальність** — працює з різними base learners

## Коли використовувати?

**Потрібно:**
- Потрібна **висока точність**
- Одна модель недостатньо точна
- **Kaggle competitions** — майже завжди
- Різні моделі доступні для комбінування
- Можна собі дозволити **обчислювальні ресурси**

**Не потрібно:**
- **Інтерпретованість критична** → одна проста модель
- Дуже обмежені ресурси
- Реал-тайм inference з жорсткими вимогами
- Одна модель вже досить точна

---

## Фундаментальна ідея

### Аналогія: Колективне рішення

**Сценарій:** Група експертів приймає рішення

**Варіант 1:** Один експерт
- Може помилятися через упередження або незнання
- Висока variance (різні експерти → різні рішення)

**Варіант 2:** Комітет експертів
- Кожен має свою думку
- Усереднення думок → помилки компенсуються
- Більш надійне та стабільне рішення ✓

**Умови успіху:**
1. **Різноманітність** — експерти не повинні думати однаково
2. **Незалежність** — кожен приймає рішення самостійно
3. **Компетентність** — кожен краще за випадкове гадання

---

## Математична інтуїція

### Зменшення Variance через усереднення

Припустимо, маємо $M$ моделей з variance $\sigma^2$ кожна.

**Якщо моделі НЕЗАЛЕЖНІ:**

$$\text{Var}(\text{average}) = \frac{\sigma^2}{M}$$

**Висновок:** Усереднення $M$ незалежних моделей зменшує variance в $M$ разів! ✓

**Якщо моделі КОРЕЛЬОВАНІ** (кореляція $\rho$):

$$\text{Var}(\text{average}) = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$$

**Спостереження:**
- При $\rho = 0$ (незалежні) → $\text{Var} = \frac{\sigma^2}{M}$
- При $\rho = 1$ (ідентичні) → $\text{Var} = \sigma^2$ (немає покращення!)

**Висновок:** Потрібна **різноманітність** моделей! 🌟

---

## Типи Ensemble Methods

```
Ensemble Methods
       |
       |--- Bagging (Bootstrap Aggregating)
       |     |
       |     |--- Random Forest
       |     |--- Bagged Trees
       |     |--- Extra Trees
       |
       |--- Boosting (Sequential Ensembling)
       |     |
       |     |--- AdaBoost
       |     |--- Gradient Boosting
       |     |--- XGBoost, LightGBM, CatBoost
       |
       |--- Stacking (Meta-Learning)
       |     |
       |     |--- Stacked Generalization
       |     |--- Blending
       |
       |--- Voting
             |
             |--- Hard Voting
             |--- Soft Voting
```

---

## 1. Bagging (Bootstrap Aggregating)

### Концепція

**Паралельно** навчаємо множину моделей на різних bootstrap samples та **усереднюємо** їх передбачення.

### Алгоритм

1. Створити $M$ bootstrap samples (вибірка з поверненням)
2. Навчити модель на кожному sample
3. Комбінувати передбачення:
   - **Регресія:** середнє арифметичне
   - **Класифікація:** голосування більшості

### Формула

**Регресія:**
$$\hat{y} = \frac{1}{M} \sum_{m=1}^{M} f_m(x)$$

**Класифікація:**
$$\hat{y} = \text{mode}\{f_1(x), f_2(x), ..., f_M(x)\}$$

### Переваги Bagging

- ✅ Зменшує **variance**
- ✅ Паралелізується
- ✅ Робастний до **overfitting**
- ✅ Працює з high-variance моделями (deep trees)

### Приклад: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,     # M моделей
    max_features='sqrt',  # Додаткова різноманітність
    bootstrap=True,       # Bootstrap sampling
    n_jobs=-1             # Паралелізація
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

---

## 2. Boosting

### Концепція

**Послідовно** навчаємо моделі, де кожна наступна **виправляє помилки** попередніх.

### Підходи

#### AdaBoost
- Змінює **ваги зразків**
- Фокус на важко класифікованих прикладах

#### Gradient Boosting
- Фітує на **residuals** (градієнт loss)
- Більша гнучкість (різні loss functions)

### Алгоритм (загальний)

1. Почати з простого передбачення (константа)
2. Для $m = 1$ до $M$:
   - Обчислити помилки попередньої моделі
   - Навчити нову модель на помилках
   - Додати нову модель до ансамблю (з вагою)
3. Фінальне передбачення = сума всіх моделей

### Формула

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu \cdot h_m(x)$$

де:
- $F_0(x)$ — початкове передбачення
- $h_m(x)$ — $m$-та модель
- $\nu$ — learning rate

### Переваги Boosting

- ✅ Зменшує **bias**
- ✅ Висока точність
- ✅ Працює з weak learners
- ✅ SOTA на табличних даних

### Недоліки Boosting

- ❌ Послідовне (не паралелиться)
- ❌ Ризик overfitting
- ❌ Чутливе до шуму

### Приклад: Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,          # Мілкі дерева!
    subsample=0.8
)

gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
```

---

## 3. Stacking (Stacked Generalization)

### Концепція

**Багаторівневий підхід:**
- **Level 0:** Різні base learners передбачають на даних
- **Level 1:** Meta-learner навчається на передбаченнях base learners

### Алгоритм

1. **Розділити дані:**
   - Train set → для base learners
   - Hold-out set → для meta-learner

2. **Навчити base learners:**
   - Кілька різних моделей на train set
   - Отримати передбачення на hold-out set

3. **Навчити meta-learner:**
   - Вхід = передбачення base learners
   - Вихід = справжні labels

4. **Фінальне передбачення:**
   - Base learners передбачають на test
   - Meta-learner комбінує їх передбачення

### Візуалізація

```
Train Data
    |
    |--- Model 1 (RF) ----→ Predictions 1 ┐
    |--- Model 2 (XGB) ---→ Predictions 2 ├→ Meta-learner (LR) → Final
    |--- Model 3 (SVM) ---→ Predictions 3 ┘
    |
    ↓
Test Data (same flow)
```

### Переваги Stacking

- ✅ Комбінує переваги різних моделей
- ✅ Часто найкраща точність
- ✅ Гнучкість у виборі моделей

### Недоліки Stacking

- ❌ Складна реалізація
- ❌ Ризик overfitting
- ❌ Обчислювально дорого
- ❌ Важко інтерпретувати

### Код (scikit-learn)

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Base learners
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Stacking with meta-learner
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),  # Meta-learner
    cv=5  # Cross-validation для генерації передбачень
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)

print(f"Stacking Accuracy: {stacking.score(X_test, y_test):.4f}")
```

---

## 4. Voting

### Hard Voting (Majority Vote)

**Кожна модель голосує за клас, обирається найпопулярніший.**

$$\hat{y} = \text{mode}\{f_1(x), f_2(x), ..., f_M(x)\}$$

**Приклад:**
```
Model 1: Class A
Model 2: Class A
Model 3: Class B
Model 4: Class A

Final: Class A (3 голоси)
```

### Soft Voting (Weighted Average of Probabilities)

**Усереднюємо ймовірності класів.**

$$\hat{p}_k = \frac{1}{M} \sum_{m=1}^{M} p_{m,k}(x)$$

$$\hat{y} = \arg\max_k \hat{p}_k$$

**Приклад:**
```
Model 1: [0.7, 0.3] (Class A prob, Class B prob)
Model 2: [0.6, 0.4]
Model 3: [0.4, 0.6]

Average: [0.57, 0.43]
Final: Class A (0.57 > 0.43)
```

### Код

```python
from sklearn.ensemble import VotingClassifier

# Моделі
clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(random_state=42)
clf3 = GradientBoostingClassifier(random_state=42)

# Hard Voting
voting_hard = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
    voting='hard'
)

# Soft Voting (краще, якщо є predict_proba)
voting_soft = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
    voting='soft'
)

voting_soft.fit(X_train, y_train)
y_pred = voting_soft.predict(X_test)

print(f"Voting Accuracy: {voting_soft.score(X_test, y_test):.4f}")
```

---

## Порівняння методів

| Метод | Тип | Різноманітність | Складність | Точність | Паралелізація |
|-------|-----|-----------------|------------|----------|---------------|
| **Bagging** | Паралельний | Bootstrap | ⭐ | ⭐⭐⭐⭐ | ✅ |
| **Random Forest** | Паралельний | Bootstrap + Features | ⭐⭐ | ⭐⭐⭐⭐ | ✅ |
| **Boosting** | Послідовний | Error-based | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| **Stacking** | Багаторівневий | Різні моделі | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ |
| **Voting** | Паралельний | Різні моделі | ⭐ | ⭐⭐⭐ | ✅ |

---

## Diversity (Різноманітність)

### Чому важлива?

**Теорема:** Ансамбль працює добре, коли моделі:
1. **Точні** (краще за випадкове гадання)
2. **Різноманітні** (роблять різні помилки)

### Способи створення різноманітності

#### 1. Data Diversity (Bagging)
- Bootstrap sampling
- Different subsets

#### 2. Feature Diversity (Random Forest)
- Random feature subsets
- Feature bagging

#### 3. Algorithm Diversity (Stacking, Voting)
- Різні типи моделей (Tree, Linear, SVM)
- Різні гіперпараметри

#### 4. Parameter Diversity
- Різна ініціалізація
- Різні random seeds

### Вимірювання різноманітності

**Q-statistic між двома моделями:**

$$Q_{ij} = \frac{N^{11}N^{00} - N^{01}N^{10}}{N^{11}N^{00} + N^{01}N^{10}}$$

де:
- $N^{11}$ — обидві правильні
- $N^{00}$ — обидві неправильні
- $N^{01}$ — $i$ правильна, $j$ неправильна
- $N^{10}$ — $i$ неправильна, $j$ правильна

**Діапазон:** [-1, 1]
- $Q = 1$ → завжди однакові (немає diversity)
- $Q = 0$ → незалежні
- $Q = -1$ → завжди різні

---

## Bias-Variance Decomposition для Ансамблів

### Загальна помилка

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

### Ефект ансамблів

**Bagging:**
- ✅ Зменшує **Variance** (через усереднення)
- ❌ Майже не впливає на **Bias**
- **Використання:** High-variance моделі (deep trees)

**Boosting:**
- ✅ Зменшує **Bias** (послідовне покращення)
- ✅ Може зменшити **Variance** (з регуляризацією)
- **Використання:** High-bias моделі (shallow trees)

### Візуалізація

```
               High Variance
                    |
       Random       |    Bagging ✓
       Forest ━━━━━━┼━━━━━━━→
                    |
Low Bias ━━━━━━━━━━┼━━━━━━━━━━ High Bias
                    |
       Boosting ✓   |
            ↓       |
       (зменшує     |
        і bias,     |
        і variance) |
                    |
```

---

## Практичні поради 💡

### Вибір методу

**1. Bagging / Random Forest:**
```python
# Коли:
# - Висока variance (overfitting)
# - Потрібна паралелізація
# - Швидкий baseline

rf = RandomForestClassifier(n_estimators=100)
```

**2. Boosting:**
```python
# Коли:
# - Потрібна максимальна точність
# - High bias (underfitting)
# - Є час на tuning

import xgboost as xgb
xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
```

**3. Stacking:**
```python
# Коли:
# - Kaggle competition
# - Максимальна точність важливіша за складність
# - Є різні моделі

stacking = StackingClassifier(estimators=[...], final_estimator=...)
```

**4. Voting:**
```python
# Коли:
# - Простий підхід до комбінування
# - Вже є навчені моделі
# - Потрібен швидкий boost точності

voting = VotingClassifier(estimators=[...], voting='soft')
```

---

## Повний приклад: Порівняння всіх методів

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Різні моделі
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier
)

# Генерація даних
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*70)
print("COMPARING ENSEMBLE METHODS")
print("="*70)

# 1. Single Decision Tree (baseline)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)
print(f"\n1. Single Decision Tree:        {dt_score:.4f}")

# 2. Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train, y_train)
bagging_score = bagging.score(X_test, y_test)
print(f"2. Bagging (50 trees):          {bagging_score:.4f} (+{bagging_score-dt_score:.4f})")

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print(f"3. Random Forest:               {rf_score:.4f} (+{rf_score-dt_score:.4f})")

# 4. AdaBoost
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)
ada_score = ada.score(X_test, y_test)
print(f"4. AdaBoost:                    {ada_score:.4f} (+{ada_score-dt_score:.4f})")

# 5. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)
print(f"5. Gradient Boosting:           {gb_score:.4f} (+{gb_score-dt_score:.4f})")

# 6. Voting (Soft)
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ],
    voting='soft'
)
voting.fit(X_train, y_train)
voting_score = voting.score(X_test, y_test)
print(f"6. Voting (RF+GB+LR):           {voting_score:.4f} (+{voting_score-dt_score:.4f})")

# 7. Stacking
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train, y_train)
stacking_score = stacking.score(X_test, y_test)
print(f"7. Stacking (RF+GB+Ada → LR):   {stacking_score:.4f} (+{stacking_score-dt_score:.4f})")

print("\n" + "="*70)
print(f"BEST METHOD: ", end="")
scores = {
    'Bagging': bagging_score,
    'Random Forest': rf_score,
    'AdaBoost': ada_score,
    'Gradient Boosting': gb_score,
    'Voting': voting_score,
    'Stacking': stacking_score
}
best_method = max(scores, key=scores.get)
print(f"{best_method} ({scores[best_method]:.4f})")
print("="*70)

# Детальний звіт для найкращого методу
print(f"\n=== {best_method} - Detailed Report ===")
best_model = eval(best_method.lower().replace(' ', '_'))
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Коли використовувати який метод?

### Decision Tree

**Використовуй коли:**
- ✅ Інтерпретованість критична
- ✅ Візуалізація рішень потрібна
- ✅ Швидкий прототип

### Bagging / Random Forest

**Використовуй коли:**
- ✅ Овerfitting з одним деревом
- ✅ Потрібна паралелізація
- ✅ Швидкий accurate baseline
- ✅ Робастність до шуму

### Boosting (GB, XGBoost)

**Використовуй коли:**
- ✅ Потрібна максимальна точність
- ✅ Kaggle / production ML
- ✅ Є час на tuning
- ✅ Табличні дані

### Stacking

**Використовуй коли:**
- ✅ Kaggle competition (top positions)
- ✅ Абсолютна точність важливіша за все
- ✅ Різні моделі вже навчені
- ✅ Є обчислювальні ресурси

### Voting

**Використовуй коли:**
- ✅ Простий спосіб комбінувати моделі
- ✅ Вже є кілька навчених моделей
- ✅ Потрібен швидкий boost

---

## Поширені помилки ❌

### 1. Використовувати ідентичні моделі

```python
# ❌ НЕПРАВИЛЬНО (немає різноманітності)
voting = VotingClassifier([
    ('rf1', RandomForestClassifier(random_state=42)),
    ('rf2', RandomForestClassifier(random_state=42)),
    ('rf3', RandomForestClassifier(random_state=42))
])

# ✅ ПРАВИЛЬНО (різні моделі)
voting = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('lr', LogisticRegression())
])
```

### 2. Overfitting у stacking

```python
# ❌ РИЗИК OVERFITTING
stacking = StackingClassifier(
    estimators=[...],
    cv=None  # Використовує train predictions
)

# ✅ ПРАВИЛЬНО
stacking = StackingClassifier(
    estimators=[...],
    cv=5  # Cross-validation
)
```

### 3. Ігнорувати різноманітність

```python
# Перевір diversity перед комбінуванням
from sklearn.metrics import confusion_matrix

predictions = [model.predict(X_test) for model in models]

# Якщо всі передбачення однакові → немає сенсу в ансамблі!
```

### 4. Надто складний ансамбль

```python
# ❌ OVERKILL
# Stacking of stacking of voting of bagging...

# ✅ Прості методи часто достатньо
# Random Forest або XGBoost часто все що потрібно
```

---

## Пов'язані теми

- [[01_Decision_Trees]] — базові learners
- [[02_Random_Forest]] — bagging ансамбль
- [[03_Gradient_Boosting]] — boosting ансамбль
- [[04_AdaBoost]] — класичний boosting
- [[Cross_Validation]] — оцінка ансамблів

## Ресурси

- [Scikit-learn: Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Zhou: Ensemble Methods - Foundations and Algorithms](https://www.routledge.com/Ensemble-Methods-Foundations-and-Algorithms/Zhou/p/book/9781439830031)
- [Kaggle: Ensemble Guide](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)

---

## Ключові висновки

> Ensemble Methods комбінують множину моделей для отримання кращих передбачень через зменшення bias та/або variance.

**Основні типи:**
- **Bagging** — паралельно, різні дані, зменшує variance
- **Boosting** — послідовно, фокус на помилках, зменшує bias
- **Stacking** — багаторівневий, різні моделі, максимальна точність
- **Voting** — простий, усереднення/голосування

**Ключові принципи:**
- **Різноманітність** моделей критична
- Моделі повинні бути **точними** та **різними**
- Баланс між **складністю** та **покращенням**

**Формула усереднення:**
$$\text{Var}(\text{average}) = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2$$

**Коли використовувати:**
- Швидкий baseline = Random Forest ✓
- Максимальна точність = Gradient Boosting (XGBoost) ✓
- Абсолютний максимум = Stacking ✓
- Простота комбінування = Voting ✓

**Найважливіше:**
- Різноманітність > Кількість моделей
- Random Forest — найкращий універсальний вибір
- XGBoost/LightGBM — для максимальної точності
- Stacking — для competitions, але складно

---

#ml #supervised-learning #ensemble #bagging #boosting #stacking #voting #random-forest #gradient-boosting #tree-based
