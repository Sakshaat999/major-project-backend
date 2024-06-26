import pandas as pd # type: ignore
import numpy as np# type: ignore
from sklearn.impute import SimpleImputer# type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from tensorflow import keras# type: ignore
from sklearn.metrics import classification_report, confusion_matrix,precision_score,recall_score,f1_score# type: ignore

ham_data = pd.read_csv('HAM10000_metadata.csv')

features = ham_data[['age', 'sex']]
labels = ham_data['dx']
le_sex = LabelEncoder()
features.loc[:, 'sex'] = le_sex.fit_transform(features['sex']) 

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train['age'] = imputer.fit_transform(X_train[['age']])
X_test['age'] = imputer.transform(X_test[['age']])

X_train_reshaped = np.asarray(X_train['age']).reshape(-1, 1)
X_test_reshaped = np.asarray(X_test['age']).reshape(-1, 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)
y_train_onehot = keras.utils.to_categorical(le_sex.fit_transform(y_train), num_classes=len(le_sex.classes_))
y_test_onehot = keras.utils.to_categorical(le_sex.transform(y_test), num_classes=len(le_sex.classes_))
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(len(le_sex.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train_onehot, epochs=10, batch_size=32, validation_split=0.1)
y_pred_onehot = model.predict(X_test_scaled)
y_pred = le_sex.inverse_transform(np.argmax(y_pred_onehot, axis=1))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
precision_optimal = precision_score(y_test, y_pred, average='weighted')
recall_optimal = recall_score(y_test, y_pred, average='weighted')
f1_optimal = f1_score(y_test, y_pred, average='weighted')
print(f'\nWeighted Precision for Optimal k: {precision_optimal:.2f}')
print(f'Weighted Recall for Optimal k: {recall_optimal:.2f}')
print(f'Weighted F1 Score for Optimal k: {f1_optimal:.2f}')