import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib as jb

# Завантаження даних
heart_data = pd.read_csv('heart.csv')

# Підготовка даних
encoder = OneHotEncoder(sparse=False)
X_categorical = encoder.fit_transform(heart_data[['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']])
categorical_features = encoder.get_feature_names_out()
X_numerical_original = heart_data[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']].values
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
X_original_combined = np.hstack([X_categorical, X_numerical_original])
y = heart_data['HeartDisease'].values

# Розділення на тренувальну та тестову вибірки
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original_combined, y, test_size=0.25, random_state=42)

# Ініціалізація моделі
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_orig, y_train)

# Формування pickle-файлів
jb.dump(classifier, 'random_forest_model.pkl')
jb.dump(encoder, 'encoder.pkl')
