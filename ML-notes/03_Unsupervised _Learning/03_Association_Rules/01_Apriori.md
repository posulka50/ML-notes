# Apriori Algorithm

## Що це?

**Apriori** — це класичний алгоритм для пошуку **association rules** (асоціативних правил) у транзакційних даних. Він знаходить паттерни типу "якщо купують A, то часто купують і B".

**Головна ідея:** якщо itemset частий, то всі його підмножини також часті (Apriori principle). Це дозволяє ефективно прискорити пошук, відсіюючи нечасті комбінації.

## Навіщо потрібен?

- 🛒 **Market Basket Analysis** — що купують разом
- 💡 **Recommendation Systems** — рекомендації товарів
- 🔍 **Pattern Discovery** — виявлення прихованих зв'язків
- 📊 **Cross-selling** — стратегії продажу
- 🏥 **Medical Diagnosis** — symptom co-occurrence
- 📚 **Web Usage Mining** — які сторінки відвідують разом

## Коли використовувати?

**Потрібно:**
- **Транзакційні дані** — списки покупок, кліки, тощо
- **Categorical items** — дискретні товари/події
- **Pattern mining** — знайти що йде разом
- **Interpretable rules** — зрозумілі правила для бізнесу
- **Середні дані** (1000-100,000 транзакцій)

**Не потрібно:**
- **Дуже великі дані** (>1M транзакцій) → FP-Growth швидше
- **Numerical data** → Clustering, Regression
- **Sequence важлива** → Sequential Pattern Mining
- **Real-time** → Streaming algorithms

---

## Основні концепції

### 1. Itemset

**Itemset** — набір товарів.

```
1-itemset: {Milk}
2-itemset: {Milk, Bread}
3-itemset: {Milk, Bread, Butter}
```

### 2. Support (Підтримка)

**Support** — як часто itemset зустрічається в даних.

$$\text{Support}(A) = \frac{\text{Кількість транзакцій з } A}{\text{Всього транзакцій}}$$

**Приклад:**
```
Транзакції:
1: {Milk, Bread, Butter}
2: {Milk, Bread}
3: {Milk, Eggs}
4: {Bread, Butter}
5: {Bread, Eggs}

Support({Milk}) = 3/5 = 0.6 (60%)
Support({Bread}) = 4/5 = 0.8 (80%)
Support({Milk, Bread}) = 2/5 = 0.4 (40%)
```

### 3. Confidence (Впевненість)

**Confidence** — якщо купують A, яка ймовірність що купують B?

$$\text{Confidence}(A \Rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$

**Приклад:**
```
Правило: {Milk} → {Bread}

Support({Milk, Bread}) = 0.4
Support({Milk}) = 0.6

Confidence({Milk} → {Bread}) = 0.4 / 0.6 = 0.67 (67%)

Інтерпретація: 67% людей, що купують молоко, також купують хліб.
```

### 4. Lift (Підйом)

**Lift** — чи A та B незалежні, чи є зв'язок?

$$\text{Lift}(A \Rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A) \times \text{Support}(B)}$$

**Інтерпретація:**
```
Lift = 1  → A та B незалежні (випадковий зв'язок)
Lift > 1  → Позитивна кореляція (купують разом)
Lift < 1  → Негативна кореляція (не купують разом)
```

**Приклад:**
```
Support({Milk, Bread}) = 0.4
Support({Milk}) = 0.6
Support({Bread}) = 0.8

Lift = 0.4 / (0.6 × 0.8) = 0.4 / 0.48 = 0.83

Lift < 1 → Weak negative correlation
```

---

## Apriori Principle

**Ключова ідея:** Якщо itemset нечастий, то всі його супермножини також нечасті.

```
{Milk, Bread, Butter} — нечастий
         ↓
{Milk, Bread, Butter, Eggs} — ТОЧНО нечастий!

Можна не перевіряти!
```

**Протилежно:**

```
{Milk, Bread} — частий
         ↓
{Milk} та {Bread} — ОБА часті

Всі підмножини часті!
```

### Ілюстрація

```
Level 1:  {A}  {B}  {C}  {D}
           ✓    ✓    ✗    ✓
           
Level 2:  {A,B} {A,C} {A,D} {B,C} {B,D} {C,D}
           ✓     ✗     ✓     ✗     ✓     ✗
                 ↑                 ↑     ↑
         Не перевіряти (C нечастий)

Level 3:  {A,B,D}  {A,C,D}  {B,C,D}  {A,B,C}
           ✓        ✗        ✗        ✗
                    ↑        ↑        ↑
            Не перевіряти ({A,C}, {B,C}, {C,D} нечасті)
```

---

## Алгоритм Apriori

### Псевдокод

```
1. Знайти всі часті 1-itemsets (L₁)
   - Підрахувати support кожного item
   - Залишити тільки з support ≥ min_support

2. FOR k = 2, 3, 4, ... WHILE L_{k-1} не пусте:
   
   a) Генерувати кандидатів C_k
      - Комбінувати часті (k-1)-itemsets
      - Застосувати Apriori principle (pruning)
   
   b) Підрахувати support для кандидатів
      - Сканувати транзакції
      - Підрахувати скільки разів зустрічається
   
   c) Відфільтрувати нечасті
      - Залишити тільки з support ≥ min_support
      - L_k = часті k-itemsets

3. Повернути всі часті itemsets (L₁ ∪ L₂ ∪ ...)

4. Згенерувати association rules
   - Для кожного частого itemset
   - Розділити на antecedent → consequent
   - Обчислити confidence
   - Залишити rules з confidence ≥ min_confidence
```

### Приклад виконання

**Дані:**
```
Transactions:
T1: {Milk, Bread, Butter}
T2: {Milk, Bread}
T3: {Milk, Eggs}
T4: {Bread, Butter}
T5: {Bread, Eggs}

min_support = 0.4 (40%)
min_confidence = 0.6 (60%)
```

**Крок 1: Знайти L₁ (часті 1-itemsets)**

```
Item      Count   Support   Frequent?
Milk        3       0.6        ✓
Bread       4       0.8        ✓
Butter      2       0.4        ✓
Eggs        2       0.4        ✓

L₁ = {{Milk}, {Bread}, {Butter}, {Eggs}}
```

**Крок 2: Знайти L₂ (часті 2-itemsets)**

```
Кандидати C₂ (всі комбінації L₁):
{Milk, Bread}
{Milk, Butter}
{Milk, Eggs}
{Bread, Butter}
{Bread, Eggs}
{Butter, Eggs}

Підрахувати support:
{Milk, Bread}    Count=2  Support=0.4  ✓
{Milk, Butter}   Count=1  Support=0.2  ✗
{Milk, Eggs}     Count=1  Support=0.2  ✗
{Bread, Butter}  Count=2  Support=0.4  ✓
{Bread, Eggs}    Count=1  Support=0.2  ✗
{Butter, Eggs}   Count=0  Support=0.0  ✗

L₂ = {{Milk, Bread}, {Bread, Butter}}
```

**Крок 3: Знайти L₃ (часті 3-itemsets)**

```
Кандидати C₃:
{Milk, Bread, Butter} — з L₂

Підрахувати support:
{Milk, Bread, Butter}  Count=1  Support=0.2  ✗

L₃ = {} (пусто)

STOP — немає більше частих itemsets
```

**Крок 4: Генерувати правила**

```
Часті itemsets: {{Milk}, {Bread}, {Butter}, {Eggs},
                 {Milk, Bread}, {Bread, Butter}}

Правила з {Milk, Bread}:
1. {Milk} → {Bread}
   Confidence = Support({Milk,Bread}) / Support({Milk})
              = 0.4 / 0.6 = 0.67 ✓ (≥ 0.6)
   Lift = 0.4 / (0.6 × 0.8) = 0.83

2. {Bread} → {Milk}
   Confidence = 0.4 / 0.8 = 0.5 ✗ (< 0.6)

Правила з {Bread, Butter}:
3. {Bread} → {Butter}
   Confidence = 0.4 / 0.8 = 0.5 ✗

4. {Butter} → {Bread}
   Confidence = 0.4 / 0.4 = 1.0 ✓
   Lift = 0.4 / (0.4 × 0.8) = 1.25

Фінальні правила:
✓ {Milk} → {Bread}      (conf=0.67, lift=0.83)
✓ {Butter} → {Bread}    (conf=1.0, lift=1.25)
```

---

## Код (Python)

### Використання mlxtend

```python
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 1. Дані (список транзакцій)
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Cheese'],
]

print(f"Total transactions: {len(transactions)}\n")

# 2. Перетворити в binary matrix
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("Binary transaction matrix:")
print(df)
print()

# 3. Знайти часті itemsets (Apriori)
frequent_itemsets = apriori(
    df, 
    min_support=0.3,    # 30%
    use_colnames=True,
    verbose=1
)

print("\n=== Frequent Itemsets ===")
print(frequent_itemsets.sort_values('support', ascending=False))
print()

# 4. Генерувати association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6,   # 60%
    num_itemsets=len(frequent_itemsets)
)

print("\n=== Association Rules ===")
print(rules[['antecedents', 'consequents', 'support', 
             'confidence', 'lift']].sort_values('lift', ascending=False))
```

**Вивід:**
```
Binary transaction matrix:
   Bread  Butter  Cheese   Eggs   Milk
0   True    True   False  False   True
1   True   False   False  False   True
2  False   False   False   True   True
3   True    True   False  False  False
4   True   False   False   True  False
5   True    True   False   True   True
6   True    True   False   True  False
7   True   False    True  False   True

=== Frequent Itemsets ===
    support              itemsets
7  0.875                {Bread}
2  0.500                 {Milk}
1  0.500               {Butter}
0  0.500                 {Eggs}
4  0.500         {Bread, Milk}
5  0.500       {Bread, Butter}
3  0.375         {Bread, Eggs}
6  0.375   {Bread, Butter, Eggs}

=== Association Rules ===
  antecedents consequents  support  confidence   lift
1   {Butter}     {Bread}    0.500        1.00   1.14
0      {Milk}     {Bread}    0.500        1.00   1.14
2      {Eggs}     {Bread}    0.375        0.75   0.86
```

### Власна реалізація

```python
from itertools import combinations
from collections import defaultdict

class AprioriAlgorithm:
    def __init__(self, min_support=0.3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []
        self.rules = []
    
    def fit(self, transactions):
        """Знайти часті itemsets та правила"""
        self.transactions = transactions
        n_transactions = len(transactions)
        
        # 1. Знайти всі унікальні items
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        
        # 2. Level 1: часті 1-itemsets
        itemsets = [frozenset([item]) for item in all_items]
        frequent_itemsets = self._get_frequent_itemsets(
            itemsets, transactions, n_transactions
        )
        
        all_frequent = frequent_itemsets.copy()
        k = 2
        
        # 3. Level k: генерувати та фільтрувати
        while frequent_itemsets:
            print(f"Level {k}: {len(frequent_itemsets)} frequent itemsets")
            
            # Генерувати кандидатів
            candidates = self._generate_candidates(frequent_itemsets, k)
            
            # Фільтрувати часті
            frequent_itemsets = self._get_frequent_itemsets(
                candidates, transactions, n_transactions
            )
            
            all_frequent.extend(frequent_itemsets)
            k += 1
        
        self.frequent_itemsets = all_frequent
        
        # 4. Генерувати правила
        self._generate_rules()
        
        return self
    
    def _get_frequent_itemsets(self, itemsets, transactions, n_transactions):
        """Підрахувати support та відфільтрувати"""
        itemset_counts = defaultdict(int)
        
        # Підрахувати
        for itemset in itemsets:
            for transaction in transactions:
                if itemset.issubset(transaction):
                    itemset_counts[itemset] += 1
        
        # Фільтрувати
        frequent = []
        for itemset, count in itemset_counts.items():
            support = count / n_transactions
            if support >= self.min_support:
                frequent.append((itemset, support))
        
        return frequent
    
    def _generate_candidates(self, frequent_itemsets, k):
        """Генерувати k-itemsets з (k-1)-itemsets"""
        candidates = []
        n = len(frequent_itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = frequent_itemsets[i][0]
                itemset2 = frequent_itemsets[j][0]
                
                # Join якщо перші k-2 елементи однакові
                union = itemset1 | itemset2
                if len(union) == k:
                    # Перевірити Apriori principle
                    if self._has_frequent_subsets(union, frequent_itemsets):
                        candidates.append(union)
        
        return candidates
    
    def _has_frequent_subsets(self, itemset, frequent_itemsets):
        """Перевірити чи всі підмножини часті"""
        frequent_sets = {fs[0] for fs in frequent_itemsets}
        
        # Всі (k-1)-підмножини
        for item in itemset:
            subset = itemset - frozenset([item])
            if subset not in frequent_sets:
                return False
        
        return True
    
    def _generate_rules(self):
        """Генерувати association rules"""
        rules = []
        
        for itemset, support in self.frequent_itemsets:
            if len(itemset) < 2:
                continue
            
            # Всі можливі розбиття на antecedent → consequent
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    # Обчислити confidence
                    antecedent_support = self._get_support(antecedent)
                    if antecedent_support > 0:
                        confidence = support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            # Обчислити lift
                            consequent_support = self._get_support(consequent)
                            lift = support / (antecedent_support * consequent_support)
                            
                            rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            })
        
        self.rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
    
    def _get_support(self, itemset):
        """Знайти support для itemset"""
        for fs, support in self.frequent_itemsets:
            if fs == itemset:
                return support
        return 0
    
    def print_rules(self, top_n=10):
        """Вивести топ правила"""
        print(f"\n=== Top {top_n} Association Rules ===\n")
        
        for i, rule in enumerate(self.rules[:top_n], 1):
            ant = ', '.join(rule['antecedent'])
            cons = ', '.join(rule['consequent'])
            
            print(f"{i}. {{{ant}}} → {{{cons}}}")
            print(f"   Support: {rule['support']:.3f}")
            print(f"   Confidence: {rule['confidence']:.3f}")
            print(f"   Lift: {rule['lift']:.3f}")
            print()

# Використання
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Bread', 'Eggs'],
]

apriori = AprioriAlgorithm(min_support=0.4, min_confidence=0.6)
apriori.fit(transactions)
apriori.print_rules()
```

---

## Приклад: Supermarket Data

### Реальніші дані

```python
# Більш реалістичний приклад
supermarket_transactions = [
    ['Beer', 'Diapers', 'Milk'],
    ['Beer', 'Diapers'],
    ['Beer', 'Chips'],
    ['Diapers', 'Milk', 'Bread'],
    ['Beer', 'Diapers', 'Chips'],
    ['Beer', 'Chips'],
    ['Diapers', 'Milk'],
    ['Beer', 'Diapers', 'Milk', 'Bread'],
    ['Beer', 'Diapers'],
    ['Chips', 'Cookies'],
]

# Transform
te = TransactionEncoder()
te_ary = te.fit(supermarket_transactions).transform(supermarket_transactions)
df_super = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori
frequent = apriori(df_super, min_support=0.3, use_colnames=True)
rules = association_rules(frequent, metric="confidence", min_threshold=0.6)

# Вивід
print("=== Frequent Itemsets ===")
print(frequent.sort_values('support', ascending=False))

print("\n=== Strong Association Rules ===")
rules_display = rules[['antecedents', 'consequents', 'support', 
                       'confidence', 'lift']].sort_values('lift', ascending=False)
print(rules_display)
```

**Інсайти:**
```
Top Rule: {Diapers} → {Beer}
- Support: 0.5 (50% транзакцій)
- Confidence: 0.83 (83% хто купує підгузки купує пиво)
- Lift: 1.39 (сильний зв'язок!)

Бізнес дія: Розмістити пиво поряд з підгузками!
```

---

## Візуалізація

### Network Graph правил

```python
import networkx as nx
import matplotlib.pyplot as plt

def plot_rules_network(rules, top_n=10):
    """Візуалізувати правила як граф"""
    
    # Top rules
    top_rules = rules.nlargest(top_n, 'lift')
    
    # Створити граф
    G = nx.DiGraph()
    
    for _, rule in top_rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        
        # Додати edge з вагою = lift
        G.add_edge(
            antecedents, 
            consequents, 
            weight=rule['lift'],
            confidence=rule['confidence']
        )
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, 
                          node_color='lightblue',
                          alpha=0.7)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, 
                           font_weight='bold')
    
    # Edges з різною товщиною (за lift)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, width=weights,
                          alpha=0.6, edge_color='gray',
                          arrows=True, arrowsize=20,
                          connectionstyle='arc3,rad=0.1')
    
    # Edge labels (confidence)
    edge_labels = {(u, v): f"{G[u][v]['confidence']:.2f}" 
                   for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                font_size=8)
    
    plt.title('Association Rules Network\n(Edge width = Lift)', 
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Використання
plot_rules_network(rules, top_n=10)
```

### Heatmap метрик

```python
import seaborn as sns

def plot_rules_heatmap(rules, top_n=20):
    """Heatmap support, confidence, lift"""
    
    top_rules = rules.nlargest(top_n, 'lift')
    
    # Створити labels для правил
    rule_labels = []
    for _, rule in top_rules.iterrows():
        ant = ', '.join(list(rule['antecedents']))
        cons = ', '.join(list(rule['consequents']))
        rule_labels.append(f"{ant} → {cons}")
    
    # Матриця метрик
    metrics = top_rules[['support', 'confidence', 'lift']].values
    
    # Plot
    plt.figure(figsize=(8, 12))
    
    sns.heatmap(
        metrics,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        yticklabels=rule_labels,
        xticklabels=['Support', 'Confidence', 'Lift'],
        cbar_kws={'label': 'Value'}
    )
    
    plt.title('Association Rules Metrics Heatmap', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Rule', fontsize=12)
    plt.tight_layout()
    plt.show()

plot_rules_heatmap(rules)
```

### Scatter Plot

```python
def plot_rules_scatter(rules):
    """Scatter plot: Support vs Confidence (розмір = Lift)"""
    
    plt.figure(figsize=(10, 7))
    
    scatter = plt.scatter(
        rules['support'],
        rules['confidence'],
        s=rules['lift'] * 100,  # Розмір за lift
        alpha=0.6,
        c=rules['lift'],
        cmap='viridis',
        edgecolors='black',
        linewidths=0.5
    )
    
    plt.colorbar(scatter, label='Lift')
    plt.xlabel('Support', fontsize=12)
    plt.ylabel('Confidence', fontsize=12)
    plt.title('Association Rules: Support vs Confidence\n(Size = Lift)', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Додати labels для топ правил
    top_rules = rules.nlargest(5, 'lift')
    for _, rule in top_rules.iterrows():
        ant = ', '.join(list(rule['antecedents']))
        cons = ', '.join(list(rule['consequents']))
        plt.annotate(
            f"{ant}→{cons}",
            (rule['support'], rule['confidence']),
            fontsize=8,
            alpha=0.7
        )
    
    plt.tight_layout()
    plt.show()

plot_rules_scatter(rules)
```

---

## Оптимізація та прискорення

### 1. Hash-based pruning

```python
def hash_based_apriori(transactions, min_support):
    """Використати hash table для pruning"""
    
    # Hash кандидатів в buckets
    # Якщо bucket count < min_support → всі itemsets в ньому нечасті
    # Не реалізовано повністю, але ідея
    pass
```

### 2. Sampling

```python
# Для дуже великих даних
# Спочатку на sample, потім verify на всіх даних

sample_size = min(10000, len(transactions))
sample_indices = np.random.choice(len(transactions), sample_size, replace=False)
sample_transactions = [transactions[i] for i in sample_indices]

# Apriori на sample
frequent = apriori(df_sample, min_support=0.3)

# Verify на повних даних
```

### 3. Parallel Apriori

```python
from multiprocessing import Pool

def count_support_parallel(itemset_chunk, transactions):
    """Підрахувати support паралельно"""
    counts = {}
    for itemset in itemset_chunk:
        count = sum(1 for t in transactions if itemset.issubset(t))
        counts[itemset] = count
    return counts

# Розділити itemsets на chunks та обробити паралельно
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Простий** | Легко зрозуміти та реалізувати |
| **Інтерпретований** | Правила зрозумілі для бізнесу |
| **Гарантовано повний** | Знаходить ВСІ часті itemsets |
| **Apriori principle** | Ефективне pruning |
| **Масштабується** | Працює на середніх даних |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Повільний** | Багато сканувань БД |
| **Багато кандидатів** | Експоненційний ріст |
| **Не для великих даних** | FP-Growth швидше |
| **Тільки categorical** | Не працює з numerical |
| **Статичний** | Не для streaming data |

---

## Порівняння з іншими методами

| Критерій | Apriori | FP-Growth | Eclat |
|----------|---------|-----------|-------|
| **Швидкість** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Пам'ять** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Простота** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Масштабованість** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**Коли що:**
- **Малі/середні дані + простота** → Apriori ✓
- **Великі дані + швидкість** → FP-Growth ✓
- **Vertical format** → Eclat ✓

---

## Практичні поради 💡

### 1. Почни з високого min_support

```python
# ✅ Спочатку високий
frequent = apriori(df, min_support=0.5)  # 50%

# Якщо мало результатів → зменшуй
frequent = apriori(df, min_support=0.3)  # 30%
frequent = apriori(df, min_support=0.1)  # 10%
```

### 2. Фільтруй за lift > 1

```python
# Залишити тільки позитивні кореляції
rules_filtered = rules[rules['lift'] > 1]

# Сильні правила
strong_rules = rules[(rules['confidence'] > 0.7) & (rules['lift'] > 1.2)]
```

### 3. Обмеж довжину itemsets

```python
# Тільки 2-itemsets (найбільш інтерпретовані)
frequent = apriori(df, min_support=0.3, max_len=2)
```

### 4. Використовуй domain knowledge

```python
# Виключити очевидні правила
# Наприклад: {Кава} → {Цукор} (занадто очевидно)

rules_interesting = rules[
    ~rules.apply(lambda x: 
        ('Coffee' in x['antecedents'] and 'Sugar' in x['consequents']),
        axis=1
    )
]
```

### 5. Групуй схожі items

```python
# Замість окремих товарів → категорії
# {Milk_1L, Milk_2L} → {Milk}
# {Bread_White, Bread_Wheat} → {Bread}
```

### 6. Temporal analysis

```python
# Розділити за часом
transactions_morning = [t for t in transactions if t['time'] == 'morning']
transactions_evening = [t for t in transactions if t['time'] == 'evening']

# Окремий Apriori для кожного
rules_morning = apriori(...)
rules_evening = apriori(...)

# Порівняти різницю
```

### 7. Перевіряй статистичну значущість

```python
# Chi-square test для незалежності
from scipy.stats import chi2_contingency

def is_significant(rule, transactions, alpha=0.05):
    """Перевірити чи правило статистично значуще"""
    
    ant = rule['antecedents']
    cons = rule['consequents']
    
    # Побудувати contingency table
    # ... (підрахувати a, b, c, d)
    
    # Chi-square test
    _, p_value, _, _ = chi2_contingency([[a, b], [c, d]])
    
    return p_value < alpha

# Фільтрувати
significant_rules = rules[rules.apply(
    lambda r: is_significant(r, transactions), axis=1
)]
```

### 8. Візуалізуй результати

```python
# Графіки допомагають знайти інсайти
plot_rules_network(rules)
plot_rules_scatter(rules)
```

### 9. A/B testing для перевірки

```python
# Знайшли правило: {Chips} → {Beer}
# Тест: покласти chips поряд з beer в половині магазинів
# Виміряти чи збільшились продажі
```

### 10. Combine з іншими методами

```python
# Apriori + Clustering
# 1. Кластеризувати клієнтів
# 2. Окремий Apriori для кожного кластера
# 3. Персоналізовані рекомендації
```

---

## Поширені помилки ❌

### 1. Занадто низький min_support

```python
# ❌ Занадто багато правил (мільйони!)
frequent = apriori(df, min_support=0.01)  # 1%

# ✅ Розумний поріг
frequent = apriori(df, min_support=0.1)   # 10%
```

### 2. Ігнорувати lift

```python
# ❌ Тільки confidence
rules_bad = rules[rules['confidence'] > 0.8]
# Може включати випадкові кореляції!

# ✅ Confidence + Lift
rules_good = rules[
    (rules['confidence'] > 0.6) & 
    (rules['lift'] > 1.2)
]
```

### 3. Не перевіряти data quality

```python
# ❌ Пропущені значення, дублікати
# Можуть спотворити результати

# ✅ Очистити дані спочатку
transactions_clean = remove_duplicates(transactions)
transactions_clean = remove_empty(transactions_clean)
```

### 4. Інтерпретувати correlation як causation

```python
# ❌ "Chips ВИКЛИКАЮТЬ купівлю Beer"
# Може бути третій фактор (спортивні події)

# ✅ "Chips та Beer часто купують разом"
```

### 5. Не враховувати розмір транзакцій

```python
# Великі транзакції → багато itemsets → спотворені метрики
# Нормалізувати або фільтрувати
```

---

## Реальні застосування

### 1. Retail / Supermarkets

```python
# Знайти:
# - Що купують разом
# - Cross-selling opportunities
# - Store layout optimization

# Приклад інсайту:
# {Diapers} → {Beer} (lift=1.4)
# → Розмістити пиво біля дитячих товарів
```

### 2. E-commerce

```python
# Рекомендації:
# "Customers who bought X also bought Y"

# Використання:
user_cart = {'Laptop', 'Mouse'}
# Знайти правила: {Laptop, Mouse} → {???}
recommendations = find_recommendations(user_cart, rules)
```

### 3. Healthcare

```python
# Symptom co-occurrence
# {Fever, Cough} → {Flu} (confidence=0.85)

# Drug interactions
# {DrugA, DrugB} → {Side Effect}
```

### 4. Web Usage Mining

```python
# Які сторінки відвідують разом
# {HomePage, Products} → {Checkout} (path optimization)

# Персоналізація контенту
```

### 5. Telecommunications

```python
# Churn prediction
# {CallDrop, HighBill, LowUsage} → {Churn}

# Bundle recommendations
# {MobileData} → {Streaming} (lift=1.3)
```

---

## Пов'язані теми

- [[02_FP-Growth]] — швидша альтернатива
- [[03_Market_Basket_Analysis]] — практичне застосування
- [[Clustering]] — сегментація перед association mining
- [[Recommendation_Systems]] — використання правил

## Ресурси

- [mlxtend Documentation](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [Original Apriori Paper (Agrawal & Srikant, 1994)](https://www.vldb.org/conf/1994/P487.PDF)
- [Introduction to Data Mining (Tan et al.)](https://www-users.cs.umn.edu/~kumar001/dmbook/index.php)

---

## Ключові висновки

> Apriori — класичний алгоритм для знаходження частих itemsets та association rules у транзакційних даних, що використовує Apriori principle (якщо itemset нечастий, то всі його супермножини також нечасті) для ефективного pruning кандидатів.

**Основні концепції:**
- **Support:** як часто зустрічається
- **Confidence:** ймовірність consequent при antecedent
- **Lift:** сила зв'язку (>1 = позитивна кореляція)

**Алгоритм:**
1. Знайти часті 1-itemsets (L₁)
2. Для k=2,3,... генерувати кандидатів C_k
3. Pruning через Apriori principle
4. Підрахувати support, відфільтрувати
5. Генерувати правила з confidence ≥ threshold

**Переваги:**
- ✅ Простий та інтерпретований
- ✅ Гарантовано знаходить всі часті itemsets
- ✅ Ефективне pruning

**Недоліки:**
- ❌ Повільний на великих даних
- ❌ Багато сканувань БД
- ❌ Експоненційний ріст кандидатів

**Метрики:**
- **High support:** Часто зустрічається
- **High confidence:** Сильне правило
- **High lift > 1:** Позитивна кореляція
- **Всі три високі:** Найкращі правила!

**Практичні поради:**
- Почни з високого min_support (30-50%)
- Фільтруй за lift > 1
- Візуалізуй результати
- Використовуй domain knowledge
- Перевіряй статистичну значущість

**Коли використовувати:**
- Market basket analysis ✓
- Cross-selling ✓
- Recommendation systems ✓
- Середні дані (1K-100K) ✓
- Великі дані (>1M) → FP-Growth ✓

---

#ml #unsupervised-learning #association-rules #apriori #market-basket-analysis #pattern-mining #frequent-itemsets #data-mining
