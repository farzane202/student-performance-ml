import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_knn(X_train, X_test, y_train, y_test):
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=5))
    ])

    knn_pipeline.fit(X_train, y_train)

    y_pred = knn_pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n===== KNN (Scaled) Results =====")
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

    return knn_pipeline