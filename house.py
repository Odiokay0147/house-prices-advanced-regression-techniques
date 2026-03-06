# %%
#libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBRegressor
from src.wrang import wrangle_house

# %%
#training data
X_train, y = wrangle_house("data/train.csv")
#loading test data separately to get Id
test = pd.read_csv("data/test.csv")
test_ids = test["Id"]
#test data
X_test, _ = wrangle_house("data/test.csv", is_train=False)
#aligning test and train column
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# %%
#build and fit randomforestregressor model
model = RandomForestRegressor(
    n_estimators=200, 
    random_state=42
)
model.fit(X_train, y)

# %%
#predict model
pred = model.predict(X_test)
pred = np.expm1(pred)

# %%
#creating submission file
submission_rfr = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred
})

submission_rfr.to_csv("submissions/submission_rfr.csv", index=False)

# %%
#checking cross validation rmse for randomforestregressor
rmse = np.sqrt(-cross_val_score(
    model, X_train, y, scoring="neg_mean_squared_error"
))
print("CV RMSE:", rmse.mean()) 

# %%
#build and fit XGB model
model_1 = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_1.fit(X_train, y)


# %%
#predict model
pred_1 = np.expm1(model.predict(X_test))

# %%
#creating submission file for XGB predictions on XGB model
submission_xgb = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred_1 
})

submission_xgb.to_csv("submissions/submission_xgb.csv", index=False)

# %%
#checking cross validation rmse for XGB
rmse = np.sqrt(-cross_val_score(
    model_1, X_train, y, scoring="neg_mean_squared_error"
))
print("CV RMSE:", rmse.mean())

# %%
#Histogram to check the distribution of predicted saleprice
plt.figure(figsize=(8, 4))
plt.hist(pred_1, bins=50, color="green", edgecolor="black")
plt.title("Distribution of SalePrice Prediction")
plt.xlabel("Sale Price")
plt.ylabel("Frequency");
# %%
