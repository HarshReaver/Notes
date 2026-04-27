# Association Rule Mining (Apriori)
Finds "people who buy X also buy Y" patterns.
- **Support** = how often items appear together
- **Confidence** = if X bought, chance Y is bought too
- **Lift** = strength of rule (lift > 1 = good)

## Using CSV Dataset
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Load data
df = pd.read_csv('Groceries_dataset.csv')  # CHANGE filename
print(df.head())
df.dropna(inplace=True)

# Step 2: Make transactions - CHANGE the column names below
transactions = df.groupby('Member_number')['itemDescription'].apply(list).tolist()

# Step 3: Top items bar chart
df['itemDescription'].value_counts().head(10).plot(kind='bar')  # CHANGE col name
plt.title('Top 10 Items')
plt.show()
# Step 4: Run Apriori
te = TransactionEncoder()
basket = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
freq = apriori(basket, min_support=0.02, use_colnames=True)  # CHANGE min_support per question
rules = association_rules(freq, metric='lift', min_threshold=1)

# Step 5: Show top rules
print(rules[['antecedents','consequents','support','confidence','lift']].head(10))

# Step 6: Scatter plot
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Rules')
plt.show()
## Using Custom (Generated) Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Make fake transactions - CHANGE item names per question
np.random.seed(42)
items = ['Python','Java','ML','DBMS','WebDev','Cloud','AI','DataSci','Cyber','IoT']
transactions = [list(np.random.choice(items, np.random.randint(2,6), replace=False)) for _ in range(200)]

# Step 2: Run Apriori (same every time)
te = TransactionEncoder()
basket = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
freq = apriori(basket, min_support=0.03, use_colnames=True)  # CHANGE min_support
rules = association_rules(freq, metric='lift', min_threshold=1)

print(rules[['antecedents','consequents','support','confidence','lift']].head(10))

plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Rules')
plt.show()