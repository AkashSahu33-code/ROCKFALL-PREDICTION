import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib

# Step 1: Load Data
df = pd.read_csv('/Users/mack/SIH_UI/SIH data.csv')  # Ye file yahi folder mein ho

features = [
    'displacement', 'velocity', 'cav', 'energy', 'rainfall', 'temp',
    'crack_length', 'bench_height', 'slope_angle', 'rmr', 'joint_spacing'
]
target = 'failure_label'
X = df[features]
y = df[target]

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Model Training (RandomForest/XGB/SVM)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
svm = SVC(probability=True, random_state=42)

rf.fit(X_train_scaled, y_train)
xgb.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

# Step 5: Save Models And Scaler
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(xgb, 'xgboost_model.pkl')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler saved. Now run your Streamlit dashboard!")
