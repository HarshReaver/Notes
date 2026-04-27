# Decision Tree Classifier
Same as Naive Bayes but uses a tree to split data. You can visualize the tree!
- **criterion='gini'** or **'entropy'** (question will tell you)
- **Only difference from Naive Bayes:** model line + tree plot
## Using CSV Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Step 1: Load
df = pd.read_csv('diabetes.csv')  # CHANGE filename
print(df.head())
df.dropna(inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

df.hist(figsize=(10,6))
plt.tight_layout()
plt.show()
# Step 2: Train - CHANGE target and criterion
target = 'Outcome'  # CHANGE
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)  # CHANGE gini/entropy
model.fit(X_train, y_train)

# Step 3: Show tree (unique to Decision Tree!)
plt.figure(figsize=(15,8))
plot_tree(model, filled=True, feature_names=list(X.columns), max_depth=3, fontsize=8)
plt.title('Decision Tree')
plt.show()

# Step 4: Results
print(classification_report(y_test, model.predict(X_test)))
ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test))
plt.title('Confusion Matrix')
plt.show()
## Using Custom (Generated) Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Step 1: Generate - CHANGE names and ranges
np.random.seed(42)
df = pd.DataFrame({
    'StudyHours': np.random.uniform(1, 10, 200),
    'Attendance': np.random.randint(40, 100, 200),
    'AssignmentScore': np.random.randint(20, 100, 200)
})
df['Result'] = np.where((df['StudyHours']>4) & (df['Attendance']>60), 'Pass', 'Fail')

print(df.head())
df.hist(figsize=(10,6))
plt.tight_layout()
plt.show()

# Step 2: Train + Tree + Evaluate
target = 'Result'  # CHANGE
le = LabelEncoder()
df[target] = le.fit_transform(df[target])
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)  # CHANGE gini/entropy
model.fit(X_train, y_train)

plt.figure(figsize=(15,8))
plot_tree(model, filled=True, feature_names=list(X.columns), max_depth=3, fontsize=8)
plt.title('Decision Tree')
plt.show()

print(classification_report(y_test, model.predict(X_test)))
ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test))
plt.title('Confusion Matrix')
plt.show()