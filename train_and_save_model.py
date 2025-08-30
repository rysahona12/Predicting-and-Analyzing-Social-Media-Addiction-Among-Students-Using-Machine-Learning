import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import json

def train_and_save_model():
    file_path = "Students Social Media Addiction.csv"
    data_frame = pd.read_csv(file_path)

    print("Dataset Shape: ", data_frame.shape)
    print("Dataset Information: ")
    print(data_frame.info())

    data_frame = data_frame.drop(columns=["Student_ID"])

    academic_order = {'High School': 1, 'Undergraduate': 2, 'Graduate': 3}
    data_frame['Academic_Level_encoded'] = data_frame['Academic_Level'].map(academic_order)
    data_frame['Affects_Academic_Performance_encoded'] = data_frame['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})

    data_frame = pd.get_dummies(data_frame, columns=[
        'Gender',
        'Country',
        'Most_Used_Platform',
        'Relationship_Status'
    ], drop_first=True, dtype=int)

    data_frame.drop(['Academic_Level', 'Affects_Academic_Performance'], axis=1, inplace=True)

    X = data_frame.drop('Addicted_Score', axis=1)
    y = data_frame['Addicted_Score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    print(f"Random Forest Regressor Performance:")
    print(f"  R² score: {r2_rf:.4f}")
    print(f"  RMSE: {rmse_rf:.4f}")
    print(f"  MAE: {mae_rf:.4f}")

    joblib.dump(rf, 'social_media_addiction_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    feature_info = {
        'feature_names': X.columns.tolist(),
        'numerical_features': num_features,
        'categorical_mappings': {
            'Academic_Level': academic_order,
            'Affects_Academic_Performance': {'Yes': 1, 'No': 0}
        },
        'one_hot_columns': {
            'Gender': [col for col in X.columns if col.startswith('Gender_')],
            'Country': [col for col in X.columns if col.startswith('Country_')],
            'Most_Used_Platform': [col for col in X.columns if col.startswith('Most_Used_Platform_')],
            'Relationship_Status': [col for col in X.columns if col.startswith('Relationship_Status_')]
        },
        'model_performance': {
            'r2_score': r2_rf,
            'rmse': rmse_rf,
            'mae': mae_rf
        }
    }

    with open('model_metadata.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    print("\nModel and preprocessing components saved successfully!")
    print("Files created:")
    print("- social_media_addiction_model.pkl")
    print("- scaler.pkl")
    print("- model_metadata.json")

    return rf, scaler, feature_info

if __name__ == "__main__":
    train_and_save_model()