# FP-Growth (Frequent Pattern Growth)

## Що це?

**FP-Growth** — це ефективний алгоритм для пошуку **frequent itemsets** без генерації кандидатів. Він використовує компактну структуру даних **FP-tree** (Frequent Pattern tree) і набагато швидше за Apriori на великих даних.

**Головна ідея:** замість генерувати кандидатів та сканувати БД багато разів (як Apriori), побудувати **FP-tree** за 2 проходи та добувати itemsets прямо з дерева.

## Навіщо потрібен?

- ⚡ **Швидкість** — значно швидше за Apriori (10-100x)
- 🗜️ **Compact representation** — FP-tree стискає дані
- 📊 **Масштабованість** — працює на великих даних (мільйони транзакцій)
- 🎯 **Ефективність** — тільки 2 проходи по БД
- 💾 **Memory efficient** — префікси діляться
- 🔍 **Divide-and-conquer** — розбиває задачу на підзадачі

## Коли використовувати?

**Потрібно:**
- **Великі дані** (> 100,000 транзакцій) — основна перевага!
- **Низький min_support** — багато frequent itemsets
- **Швидкість критична** — production systems
- **Обмежена пам'ять** — FP-tree компактніше
- **Довгі транзакції** — багато items в кожній

**Не потрібно:**
- **Дуже малі дані** (< 1000) → Apriori простіший
- **Простота важлива** → Apriori зрозуміліший
- **Incremental updates** → потрібно перебудовувати tree

---

## Відмінності від Apriori

### Apriori vs FP-Growth

| Аспект | Apriori | FP-Growth |
|--------|---------|-----------|
| **Підхід** | Generate-and-test | Pattern growth |
| **Кандидати** | Генерує явно | Не генерує |
| **Проходів по БД** | k+1 (для k-itemsets) | 2 |
| **Структура даних** | Hash tables | FP-tree |
| **Швидкість** | Повільний на великих даних | Швидкий |
| **Пам'ять** | Менше | Більше (tree) |
| **Складність** | Простіше | Складніше |

**Візуально:**

```
Apriori:
DB → Scan 1 → L₁ → Scan 2 → L₂ → Scan 3 → L₃ → ...
     ↓        ↓      ↓        ↓      ↓
   Count   Generate Count Generate Count
           C₂       C₂       C₃

FP-Growth:
DB → Scan 1 → Header table
     ↓
   Scan 2 → FP-tree → Mine patterns (NO more DB scans!)
```

---

## Структура FP-tree

### Що це?

**FP-tree (Frequent Pattern tree)** — компактне представлення транзакцій, де:
- Кожен шлях = транзакція (відсортована за frequency)
- Спільні префікси діляться (компресія!)
- Node = item + counter

### Приклад

**Дані:**
```
Transactions:
T1: {Bread, Milk}
T2: {Bread, Diapers, Beer, Eggs}
T3: {Milk, Diapers, Beer, Coke}
T4: {Bread, Milk, Diapers, Beer}
T5: {Bread, Milk, Diapers, Coke}

min_support = 3 (60%)
```

**Крок 1: Item frequencies**
```
Item      Count   Frequent?
Bread       4        ✓
Milk        4        ✓
Diapers     4        ✓
Beer        3        ✓
Eggs        1        ✗
Coke        2        ✗

Frequent items (sorted by count):
Bread(4), Milk(4), Diapers(4), Beer(3)
```

**Крок 2: Відсортувати транзакції**
```
Відсортувати кожну транзакцію за frequency (descending):

T1: {Bread, Milk}                    → [Bread, Milk]
T2: {Bread, Diapers, Beer, Eggs}     → [Bread, Diapers, Beer]
T3: {Milk, Diapers, Beer, Coke}      → [Milk, Diapers, Beer]
T4: {Bread, Milk, Diapers, Beer}     → [Bread, Milk, Diapers, Beer]
T5: {Bread, Milk, Diapers, Coke}     → [Bread, Milk, Diapers]
```

**Крок 3: Побудувати FP-tree**

```
          null (root)
           |
         Bread:4
           |
         Milk:3 ----→ Milk:1
           |            |
       Diapers:2    Diapers:1
           |            |
         Beer:1       Beer:1

Header Table:
Item      Count   Node links
Bread       4     → Bread:4
Milk        4     → Milk:3 → Milk:1
Diapers     4     → Diapers:2 → Diapers:1
Beer        3     → Beer:1 → Beer:1
```

**Пояснення:**
- T1 `[Bread, Milk]`: створює шлях Bread:1 → Milk:1
- T2 `[Bread, Diapers, Beer]`: розгалужує в Bread → Diapers:1 → Beer:1
- T3 `[Milk, Diapers, Beer]`: новий шлях з root → Milk:1 → Diapers:1 → Beer:1
- T4 `[Bread, Milk, Diapers, Beer]`: Bread:2 → Milk:2 → Diapers:1 → Beer:1
- T5 `[Bread, Milk, Diapers]`: Bread:3 → Milk:3 → Diapers:2

**Компресія:** Замість 5 окремих транзакцій → 1 дерево з спільними префіксами!

---

## Алгоритм FP-Growth

### Псевдокод

```
FP-Growth(Tree, α):
    IF Tree містить single path P:
        FOR кожна комбінація nodes в P:
            GENERATE pattern = комбінація ∪ α
    ELSE:
        FOR кожен item i в header table (знизу вгору):
            GENERATE pattern β = i ∪ α з support = support(i)
            
            CONSTRUCT conditional pattern base для β
            CONSTRUCT conditional FP-tree Tree_β
            
            IF Tree_β не пусте:
                CALL FP-Growth(Tree_β, β)
```

### Покроковий процес

**1. Побудувати FP-tree (2 проходи по БД)**

**Прохід 1:** Знайти frequent items та їх counts
```python
item_counts = count_items(transactions)
frequent_items = filter(item_counts, min_support)
```

**Прохід 2:** Побудувати FP-tree
```python
for transaction in transactions:
    sorted_items = sort(transaction, by=frequency, descending=True)
    insert_into_tree(sorted_items)
```

**2. Mining з FP-tree**

Для кожного frequent item (від найменш частого до найбільш):

a) **Знайти conditional pattern base**
   - Всі шляхи в FP-tree, що закінчуються на цей item
   
b) **Побудувати conditional FP-tree**
   - FP-tree тільки для цих шляхів
   
c) **Рекурсивно добувати patterns**
   - FP-Growth на conditional tree

---

## Детальний приклад

**Використаємо дані з вище:**

### Mining для Beer

**Крок 1: Conditional pattern base для Beer**

Знайти всі шляхи, що закінчуються на Beer:

```
Path 1: Bread → Diapers → Beer:1
        Prefix: [Bread:1, Diapers:1]

Path 2: Milk → Diapers → Beer:1
        Prefix: [Milk:1, Diapers:1]

Path 3: Bread → Milk → Diapers → Beer:1
        Prefix: [Bread:1, Milk:1, Diapers:1]

Conditional pattern base:
{Bread:1, Diapers:1}
{Milk:1, Diapers:1}
{Bread:1, Milk:1, Diapers:1}
```

**Крок 2: Conditional FP-tree для Beer**

Frequent items в pattern base (min_support=3):
```
Diapers: 1+1+1 = 3 ✓
Bread: 1+1 = 2 ✗
Milk: 1+1 = 2 ✗

Тільки Diapers частий!
```

Conditional tree:
```
    null
     |
  Diapers:3
```

**Крок 3: Generate patterns**
```
{Beer} support=3
{Diapers, Beer} support=3
```

### Mining для Diapers

**Conditional pattern base:**
```
From: Bread → Milk → Diapers:2
      Prefix: [Bread:2, Milk:2]

From: Milk → Diapers:1
      Prefix: [Milk:1]

Conditional pattern base:
{Bread:2, Milk:2}
{Milk:1}
```

**Conditional FP-tree:**
```
Frequent in pattern base:
Milk: 2+1 = 3 ✓
Bread: 2 ✗

Tree:
    null
     |
   Milk:3
```

**Patterns:**
```
{Diapers} support=4
{Milk, Diapers} support=3
```

### Всі frequent itemsets

```
1-itemsets:
{Bread}:4
{Milk}:4
{Diapers}:4
{Beer}:3

2-itemsets:
{Bread, Milk}:3
{Bread, Diapers}:2 (< min_support=3) ✗
{Milk, Diapers}:3
{Diapers, Beer}:3

3-itemsets:
{Bread, Milk, Diapers}:2 ✗
```

---

## Код (Python)

### Використання mlxtend

```python
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Дані
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diapers', 'Beer', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Coke'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Coke'],
]

# Transform
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("Transaction matrix:")
print(df)
print()

# FP-Growth
frequent_itemsets = fpgrowth(
    df,
    min_support=0.6,    # 60%
    use_colnames=True
)

print("=== Frequent Itemsets (FP-Growth) ===")
print(frequent_itemsets.sort_values('support', ascending=False))
print()

# Association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.7
)

print("=== Association Rules ===")
print(rules[['antecedents', 'consequents', 'support', 
             'confidence', 'lift']].sort_values('lift', ascending=False))
```

### Порівняння Apriori vs FP-Growth

```python
import time
from mlxtend.frequent_patterns import apriori, fpgrowth

# Великий dataset
import numpy as np

# Генерувати більше даних
np.random.seed(42)
n_transactions = 10000
items_pool = [f'Item_{i}' for i in range(100)]

transactions_large = []
for _ in range(n_transactions):
    n_items = np.random.randint(3, 10)
    trans = list(np.random.choice(items_pool, n_items, replace=False))
    transactions_large.append(trans)

# Transform
te = TransactionEncoder()
te_ary = te.fit(transactions_large).transform(transactions_large)
df_large = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Dataset: {n_transactions} transactions, {len(items_pool)} items")
print()

# Apriori
print("Running Apriori...")
start = time.time()
frequent_apriori = apriori(df_large, min_support=0.05, use_colnames=True)
time_apriori = time.time() - start
print(f"Apriori: {time_apriori:.2f}s")
print(f"Found {len(frequent_apriori)} frequent itemsets")
print()

# FP-Growth
print("Running FP-Growth...")
start = time.time()
frequent_fpgrowth = fpgrowth(df_large, min_support=0.05, use_colnames=True)
time_fpgrowth = time.time() - start
print(f"FP-Growth: {time_fpgrowth:.2f}s")
print(f"Found {len(frequent_fpgrowth)} frequent itemsets")
print()

print(f"=== Speedup ===")
print(f"FP-Growth is {time_apriori/time_fpgrowth:.1f}x faster!")
```

### Власна реалізація FP-tree

```python
class FPNode:
    """Node в FP-tree"""
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None  # Link до наступного node з тим же item
    
    def increment(self, count=1):
        self.count += count

class FPTree:
    """FP-tree structure"""
    def __init__(self, min_support):
        self.root = FPNode(None, 0)
        self.header_table = {}
        self.min_support = min_support
    
    def add_transaction(self, transaction, count=1):
        """Додати транзакцію до дерева"""
        current = self.root
        
        for item in transaction:
            if item in current.children:
                # Item вже є, increment count
                current.children[item].increment(count)
            else:
                # Створити новий node
                new_node = FPNode(item, count, current)
                current.children[item] = new_node
                
                # Оновити header table
                if item in self.header_table:
                    # Link до існуючого node
                    node = self.header_table[item]
                    while node.next:
                        node = node.next
                    node.next = new_node
                else:
                    self.header_table[item] = new_node
            
            current = current.children[item]
    
    def get_paths(self, item):
        """Знайти всі шляхи для item (conditional pattern base)"""
        paths = []
        
        # Знайти всі nodes з цим item
        node = self.header_table.get(item)
        
        while node:
            # Піднятись до root
            path = []
            count = node.count
            parent = node.parent
            
            while parent.parent is not None:  # До root
                path.append(parent.item)
                parent = parent.parent
            
            if path:
                paths.append((path[::-1], count))
            
            node = node.next
        
        return paths
    
    def print_tree(self, node=None, indent=0):
        """Вивести дерево (для debugging)"""
        if node is None:
            node = self.root
        
        if node.item is not None:
            print("  " * indent + f"{node.item}:{node.count}")
        
        for child in node.children.values():
            self.print_tree(child, indent + 1)

def fpgrowth_simple(transactions, min_support):
    """Проста реалізація FP-Growth"""
    
    # 1. Підрахувати item frequencies
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    # 2. Фільтрувати frequent items
    n_transactions = len(transactions)
    frequent_items = {
        item: count 
        for item, count in item_counts.items() 
        if count >= min_support * n_transactions
    }
    
    if not frequent_items:
        return []
    
    # 3. Відсортувати за frequency
    sorted_items = sorted(
        frequent_items.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # 4. Побудувати FP-tree
    tree = FPTree(min_support)
    
    for transaction in transactions:
        # Відсортувати та фільтрувати
        sorted_trans = sorted(
            [item for item in transaction if item in frequent_items],
            key=lambda x: frequent_items[x],
            reverse=True
        )
        
        if sorted_trans:
            tree.add_transaction(sorted_trans)
    
    print("=== FP-Tree ===")
    tree.print_tree()
    print()
    
    # 5. Mine patterns
    patterns = []
    
    # Для кожного item (від найменш частого)
    for item, count in reversed(sorted_items):
        # Додати 1-itemset
        patterns.append((frozenset([item]), count / n_transactions))
        
        # Знайти conditional pattern base
        paths = tree.get_paths(item)
        
        if paths:
            print(f"Item: {item}")
            print(f"Conditional pattern base: {paths}")
            
            # Побудувати conditional transactions
            conditional_trans = []
            for path, count in paths:
                for _ in range(count):
                    conditional_trans.append(path)
            
            # Рекурсивно mine
            if conditional_trans:
                conditional_patterns = fpgrowth_simple(
                    conditional_trans, 
                    min_support
                )
                
                # Додати item до patterns
                for pattern, support in conditional_patterns:
                    new_pattern = pattern | frozenset([item])
                    patterns.append((new_pattern, support))
    
    return patterns

# Використання
transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Diapers', 'Beer'],
    ['Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers'],
]

patterns = fpgrowth_simple(transactions, min_support=0.6)

print("\n=== Frequent Patterns ===")
for pattern, support in sorted(patterns, key=lambda x: (len(x[0]), -x[1])):
    items = ', '.join(sorted(pattern))
    print(f"{{{items}}}: {support:.2f}")
```

---

## Оптимізації

### 1. Single-path optimization

**Якщо FP-tree має тільки один шлях:**

```
    root
     |
     A:5
     |
     B:5
     |
     C:5

Всі комбінації automatically frequent!
{A}, {B}, {C}, {A,B}, {A,C}, {B,C}, {A,B,C}

Не потрібно рекурсивно mining!
```

### 2. Parallel FP-Growth

```python
# Розділити items між workers
# Кожен worker обробляє свої conditional trees паралельно

from multiprocessing import Pool

def mine_item(item, tree, min_support):
    """Mine patterns для одного item"""
    patterns = []
    # ... mining logic ...
    return patterns

# Parallel processing
with Pool(processes=4) as pool:
    results = pool.starmap(
        mine_item, 
        [(item, tree, min_support) for item in items]
    )
```

### 3. Memory optimization

```python
# Зберігати тільки frequent items в memory
# Для рідких items використовувати disk-based approach

class DiskBackedFPTree:
    """FP-tree з disk caching для великих даних"""
    # Зберігати частини дерева на disk
    # Завантажувати тільки при потребі
    pass
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Швидкість** | Набагато швидше за Apriori |
| **Тільки 2 проходи БД** | Не залежить від k-itemsets |
| **Compact** | FP-tree стискає дані |
| **Масштабованість** | Працює на мільйонах транзакцій |
| **Divide-and-conquer** | Ефективна декомпозиція |
| **Низький min_support** | Може знайти rare patterns |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Складність** | Важче зрозуміти ніж Apriori |
| **Пам'ять** | FP-tree займає багато |
| **Не incremental** | Треба перебудовувати tree |
| **Складна реалізація** | Багато edge cases |
| **Debugging** | Важко відстежити помилки |

---

## Порівняння з Apriori

### Performance на різних розмірах даних

```python
import matplotlib.pyplot as plt
import time

# Експеримент
dataset_sizes = [1000, 5000, 10000, 20000, 50000]
apriori_times = []
fpgrowth_times = []

for size in dataset_sizes:
    # Генерувати дані
    transactions = generate_transactions(size)
    te_ary = transform(transactions)
    df = pd.DataFrame(te_ary)
    
    # Apriori
    start = time.time()
    apriori(df, min_support=0.05)
    apriori_times.append(time.time() - start)
    
    # FP-Growth
    start = time.time()
    fpgrowth(df, min_support=0.05)
    fpgrowth_times.append(time.time() - start)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, apriori_times, 'o-', label='Apriori', linewidth=2)
plt.plot(dataset_sizes, fpgrowth_times, 's-', label='FP-Growth', linewidth=2)
plt.xlabel('Number of Transactions', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Apriori vs FP-Growth Performance', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Коли використовувати FP-Growth

### Decision Tree

```
Скільки транзакцій?
├─ < 10,000 → Apriori (простіше)
└─ ≥ 10,000
   │
   Який min_support?
   ├─ Високий (>0.5) → Apriori OK
   └─ Низький (<0.1)
      │
      Швидкість критична?
      ├─ Так → FP-Growth ✓
      └─ Ні → Apriori (простіше debug)
```

### Рекомендації

**Використовуй FP-Growth якщо:**
- ✅ Багато транзакцій (> 10K)
- ✅ Низький min_support
- ✅ Швидкість важлива
- ✅ Production system

**Використовуй Apriori якщо:**
- ✅ Малі дані (< 10K)
- ✅ Простота важлива
- ✅ Навчання/дослідження
- ✅ Легкий debugging потрібен

---

## Практичні поради 💡

### 1. Preprocessing для швидкості

```python
# Видалити рідкі items спочатку
item_counts = count_items(transactions)
min_count = min_support * len(transactions)

frequent_items = {
    item for item, count in item_counts.items() 
    if count >= min_count
}

# Фільтрувати транзакції
transactions_filtered = [
    [item for item in trans if item in frequent_items]
    for trans in transactions
]

# FP-Growth на filtered
fpgrowth(transactions_filtered, min_support)
```

### 2. Incremental updates

```python
# Для streaming data
# Періодично перебудовувати tree

class IncrementalFPGrowth:
    def __init__(self, min_support, rebuild_threshold=1000):
        self.min_support = min_support
        self.rebuild_threshold = rebuild_threshold
        self.buffer = []
        self.patterns = []
    
    def add_transaction(self, transaction):
        self.buffer.append(transaction)
        
        if len(self.buffer) >= self.rebuild_threshold:
            self.rebuild()
    
    def rebuild(self):
        """Перебудувати FP-tree"""
        all_transactions = self.get_all_transactions() + self.buffer
        self.patterns = fpgrowth(all_transactions, self.min_support)
        self.buffer = []
```

### 3. Parallel processing

```python
# Розділити items між threads
from concurrent.futures import ThreadPoolExecutor

def mine_conditional_tree(item, tree, min_support):
    """Mine для одного item"""
    # ... mining logic ...
    return patterns

# Parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(mine_conditional_tree, item, tree, min_support)
        for item in items
    ]
    
    all_patterns = []
    for future in futures:
        all_patterns.extend(future.result())
```

### 4. Memory monitoring

```python
import psutil
import os

def check_memory():
    """Перевірити використання пам'яті"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    return mem

# Перевіряти під час mining
initial_mem = check_memory()
patterns = fpgrowth(df, min_support=0.05)
final_mem = check_memory()

print(f"Memory used: {final_mem - initial_mem:.1f} MB")
```

### 5. Візуалізація FP-tree

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_fp_tree(tree, max_depth=3):
    """Візуалізувати FP-tree"""
    G = nx.DiGraph()
    
    def add_nodes(node, parent_id=None, depth=0):
        if depth > max_depth:
            return
        
        if node.item is not None:
            node_id = id(node)
            label = f"{node.item}:{node.count}"
            G.add_node(node_id, label=label)
            
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            
            for child in node.children.values():
                add_nodes(child, node_id, depth + 1)
    
    add_nodes(tree.root)
    
    # Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2)
    labels = nx.get_node_attributes(G, 'label')
    
    nx.draw(G, pos, labels=labels, with_labels=True,
           node_size=2000, node_color='lightblue',
           font_size=10, font_weight='bold',
           arrows=True, arrowsize=20)
    
    plt.title('FP-Tree Visualization', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

### 6. Optimize ordering

```python
# Експериментувати з різними сортуваннями
# Іноді інший порядок → кращі результати

# За frequency (standard)
sorted_by_freq = sorted(items, key=lambda x: counts[x], reverse=True)

# За alphabetical (для consistency)
sorted_alphabetical = sorted(items)

# Custom ordering (domain knowledge)
custom_order = ['Bread', 'Milk', 'Beer', ...]  # Most important first
```

### 7. Validate results

```python
# Перевірити чи FP-Growth дає ті ж результати що Apriori
frequent_apriori = set(map(frozenset, apriori_results))
frequent_fpgrowth = set(map(frozenset, fpgrowth_results))

# Повинні бути однакові!
assert frequent_apriori == frequent_fpgrowth, "Results mismatch!"
```

### 8. Transaction filtering

```python
# Видалити дуже великі транзакції (можуть спотворити results)
MAX_TRANSACTION_SIZE = 50

transactions_filtered = [
    t for t in transactions 
    if len(t) <= MAX_TRANSACTION_SIZE
]
```

### 9. Batch processing

```python
# Для ДУЖЕ великих даних
# Обробляти частинами

def fpgrowth_batch(transactions, min_support, batch_size=10000):
    """FP-Growth на batches"""
    
    all_patterns = []
    
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i+batch_size]
        patterns = fpgrowth(batch, min_support)
        all_patterns.extend(patterns)
    
    # Merge та deduplicate
    merged = merge_patterns(all_patterns)
    
    return merged
```

### 10. Profile performance

```python
import cProfile
import pstats

# Profile код
profiler = cProfile.Profile()
profiler.enable()

frequent = fpgrowth(df, min_support=0.05)

profiler.disable()

# Показати hotspots
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## Поширені помилки ❌

### 1. Не перевіряти розмір дерева

```python
# ❌ FP-tree може бути дуже великим!
# Може закінчитись memory

# ✅ Перевіряй та обмежуй
import sys

tree_size = sys.getsizeof(tree)
if tree_size > 1e9:  # 1 GB
    print("Warning: Tree too large!")
    # Використай вищий min_support або sampling
```

### 2. Не оптимізувати ordering

```python
# ❌ Випадковий порядок items
# Більше дерево, гірша компресія

# ✅ Сортувати за frequency
sorted_items = sorted(items, key=lambda x: counts[x], reverse=True)
```

### 3. Забути про preprocessing

```python
# ❌ Прямо на raw даних
fpgrowth(raw_transactions, min_support)

# ✅ Очистити спочатку
# Remove empty, duplicates, rare items
transactions_clean = preprocess(raw_transactions, min_support)
fpgrowth(transactions_clean, min_support)
```

### 4. Занадто низький min_support

```python
# ❌ min_support=0.001 (0.1%)
# Мільйони patterns! Memory overflow!

# ✅ Розумний поріг
min_support = max(0.01, 100 / len(transactions))  # Мінімум 100 транзакцій
```

### 5. Не використовувати parallel

```python
# ❌ Single-threaded на великих даних
# Дуже повільно

# ✅ Parallel якщо можливо
# Використай parallel libraries або batch processing
```

---

## Реальні застосування

### 1. E-commerce recommendations

```python
# Amazon-style "Frequently bought together"

# 1. Extract purchase history
transactions = get_user_purchases()

# 2. FP-Growth (швидко навіть на мільйонах)
frequent = fpgrowth(transactions, min_support=0.01)

# 3. Generate recommendations
def recommend(user_cart, frequent_itemsets):
    """Recommend based on cart"""
    recommendations = []
    
    for itemset, support in frequent_itemsets:
        if user_cart.issubset(itemset):
            # Items not yet in cart
            new_items = itemset - user_cart
            recommendations.extend(new_items)
    
    return list(set(recommendations))

# Usage
cart = {'Laptop', 'Mouse'}
recs = recommend(cart, frequent)
print(f"Recommended: {recs}")
```

### 2. Clickstream analysis

```python
# Web usage patterns

# Transactions = sequences of pages visited
clickstreams = [
    ['Home', 'Products', 'Cart', 'Checkout'],
    ['Home', 'Search', 'Products'],
    ['Home', 'Products', 'Product_Detail'],
    # ... millions more
]

# FP-Growth швидко знаходить patterns
patterns = fpgrowth(clickstreams, min_support=0.05)

# Optimize website structure based on patterns
```

### 3. Medical diagnosis

```python
# Symptom co-occurrence

# Transactions = patient symptoms
patient_data = [
    ['Fever', 'Cough', 'Headache'],
    ['Fever', 'Cough', 'Fatigue'],
    ['Nausea', 'Vomiting', 'Diarrhea'],
    # ...
]

# Find symptom clusters
symptom_patterns = fpgrowth(patient_data, min_support=0.1)

# Помогає в diagnosis
```

---

## Пов'язані теми

- [[01_Apriori]] — попередник FP-Growth
- [[03_Market_Basket_Analysis]] — практичне застосування
- [[Pattern_Mining]] — інші методи pattern discovery
- [[Sequential_Pattern_Mining]] — з урахуванням порядку

## Ресурси

- [FP-Growth Original Paper (Han et al., 2000)](https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf)
- [mlxtend FP-Growth Documentation](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
- [Mining of Massive Datasets (Chapter 6)](http://www.mmds.org/)

---

## Ключові висновки

> FP-Growth — ефективний алгоритм для пошуку frequent itemsets, що використовує компактну FP-tree структуру та divide-and-conquer підхід, уникаючи генерації кандидатів та досягаючи значного прискорення порівняно з Apriori.

**Основна ідея:**
- Побудувати FP-tree (2 проходи БД)
- Mining через conditional pattern bases
- Рекурсивна декомпозиція

**Алгоритм:**
1. **Прохід 1:** Підрахувати item frequencies
2. **Прохід 2:** Побудувати FP-tree (відсортовано за frequency)
3. **Mining:** Для кожного item:
   - Знайти conditional pattern base
   - Побудувати conditional FP-tree
   - Рекурсивно добувати patterns

**Переваги над Apriori:**
- ⚡ 10-100x швидше на великих даних
- 📊 Тільки 2 проходи БД (vs k+1)
- 🗜️ Compact representation
- 📈 Масштабується краще

**Недоліки:**
- ❌ Складніша реалізація
- ❌ Більше пам'яті (FP-tree)
- ❌ Не incremental

**Ключові компоненти:**
- **FP-tree:** Prefix tree зі спільними префіксами
- **Header table:** Links до nodes з однаковим item
- **Conditional pattern base:** Префікси шляхів
- **Divide-and-conquer:** Розбиття на підзадачі

**Коли використовувати:**
- Великі дані (>10K транзакцій) → FP-Growth ✓
- Малі дані + простота → Apriori ✓
- Низький min_support → FP-Growth ✓
- Production + швидкість → FP-Growth ✓

**Практичні поради:**
- Preprocessing для швидкості
- Parallel processing для scale
- Monitor memory usage
- Validate results проти Apriori
- Optimize item ordering

---

#ml #unsupervised-learning #association-rules #fp-growth #frequent-patterns #data-mining #pattern-mining #performance-optimization
