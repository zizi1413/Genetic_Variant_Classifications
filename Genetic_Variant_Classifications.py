import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers.legacy import Adam
import xgboost
from xgboost import XGBClassifier

#%% Data Import
data = pd.read_csv(r"/Genetic-VariantClassifications.csv")

#%% Data Inspection
print(data.head())  # Display the first few rows
print(data.shape)   # Check the shape of the dataset
print(data['CLASS'].value_counts())  # Check distribution of the target variable
data.info()  # Get information about the dataset
data.describe()  # Summary statistics

#%% Countplot to show CLASS distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='CLASS', data=data, palette='icefire')
plt.title('Distribution of CLASS')
plt.show()

#%% Heatmap for Missing Values
plt.figure(figsize=(12, 8))
sns.heatmap(data.isnull(), cmap='viridis', cbar=True)
plt.title('Missing Values Heatmap')
plt.show()

#%% Feature Selection and Dropping Missing Values
# Select relevant columns and drop rows with missing values
toBeConsidered = ['CHROM', 'POS', 'REF', 'ALT', 'AF_ESP', 'AF_EXAC', 'AF_TGP',
                  'CLNDISDB', 'CLNDN', 'CLNHGVS', 'CLNVC', 'MC', 'ORIGIN', 'CLASS',
                  'Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Feature', 'BIOTYPE', 'STRAND',
                  'CADD_PHRED', 'CADD_RAW']
df = data[toBeConsidered].dropna()
print(df.shape)  # Check the shape after dropping missing values

#%% Check for Remaining Missing Values
print(df.isnull().sum())  # Ensure no missing values remain
df.info()

#%% Convert 'CHROM' to String
df['CHROM'] = df['CHROM'].astype(str)

#%% Identify Object Columns for Encoding
object_cols = df.select_dtypes(include=['object']).columns
print(object_cols)

#%% Feature Encoding
# Use LabelEncoder for ordinal data and OneHotEncoder for categorical data
for col in object_cols:
    if df[col].nunique() < 10:  # Use OneHotEncoder for low cardinality columns
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    else:  # Use LabelEncoder for high cardinality columns
        df[col] = LabelEncoder().fit_transform(df[col])

#%% Feature Scaling
# Scale only numerical features
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = RobustScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

#%% Prepare X and y
y = df['CLASS']
X = df.drop(columns=['CLASS'], axis=1)

#%% Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#%% Logistic Regression Classifier
lr = LogisticRegression(class_weight='balanced', penalty='l2', max_iter=100, solver='saga')
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
print('Accuracy (Logistic Regression):', accuracy_score(y_test, y_predict))
print('F1 Score (Logistic Regression):', f1_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

#%% Decision Tree Classifier
dtc = DecisionTreeClassifier(max_depth=12, class_weight='balanced', random_state=0)
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)
print('Accuracy (Decision Tree):', accuracy_score(y_test, y_predict))
print('F1 Score (Decision Tree):', f1_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

#%% Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=180, class_weight='balanced', random_state=42)
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)
print('Accuracy (Random Forest):', accuracy_score(y_test, y_predict))
print('F1 Score (Random Forest):', f1_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

#%% AdaBoost Classifier
dtc = DecisionTreeClassifier(class_weight='balanced')
ada_model = AdaBoostClassifier(estimator=dtc, n_estimators=200, learning_rate=0.001)
ada_model.fit(X_train, y_train)
y_predict = ada_model.predict(X_test)
print('Accuracy (AdaBoost):', accuracy_score(y_test, y_predict))
print('F1 Score (AdaBoost):', f1_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

#%% K-Fold Cross Validation
classifiers = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('XGBoost', xgboost.XGBClassifier())
]

for name, clf in classifiers:
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f'Accuracy ({name}): {scores.mean():.4f}')

#%% Stacking Classifier
stack_model = StackingClassifier(estimators=classifiers, final_estimator=LogisticRegression())
stack_scores = cross_val_score(stack_model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Accuracy (Stacking): {stack_scores.mean():.4f}')

#%% Deep Learning Model
input_dim = X_train.shape[1]  # Dynamically set input dimension
model = Sequential([
    Dense(128, input_dim=input_dim, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model with early stopping
callback = EarlyStopping(monitor='val_loss', patience=3)
X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
history = model.fit(X_train_dl, y_train_dl, validation_data=(X_val_dl, y_val_dl), batch_size=64, epochs=50, verbose=1, callbacks=[callback])

# Evaluate the model
y_predict_dl = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_predict_dl))
