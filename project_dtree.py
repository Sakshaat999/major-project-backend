import pandas as pd# type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier# type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,f1_score,precision_score,recall_score# type: ignore
from sklearn.preprocessing import LabelEncoder# type: ignore
from sklearn.impute import SimpleImputer# type: ignore

ham_data = pd.read_csv('HAM10000_metadata.csv')
features = ham_data[['age', 'sex']]
labels = ham_data['dx']
le_sex = LabelEncoder()
features.loc[:, 'sex'] = le_sex.fit_transform(features['sex'])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train['age'] = imputer.fit_transform(X_train[['age']])
X_test['age'] = imputer.transform(X_test[['age']])
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred = decision_tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=1)
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)
precision_optimal = precision_score(y_test, y_pred, average='weighted')
recall_optimal = recall_score(y_test, y_pred, average='weighted')
f1_optimal = f1_score(y_test, y_pred, average='weighted')
print(f'\nWeighted Precision for Optimal k: {precision_optimal:.2f}')
print(f'Weighted Recall for Optimal k: {recall_optimal:.2f}')
print(f'Weighted F1 Score for Optimal k: {f1_optimal:.2f}')