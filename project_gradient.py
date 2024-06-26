# Import necessary libraries
import pandas as pd# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.preprocessing import StandardScaler# type: ignore
from sklearn.ensemble import GradientBoostingClassifier# type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score# type: ignore

data = pd.read_csv('HAM10000_metadata.csv')
data = data.drop(['lesion_id', 'image_id'], axis=1)
data = pd.get_dummies(data, columns=['dx_type', 'sex', 'localization'])
data = data.dropna()
X = data.drop('dx', axis=1)
y = data['dx']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
