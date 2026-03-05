# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBRegressor
from wrang import wrangle_house

# %%
#training data
X_train, y = wrangle_house("train.csv")
#loading test data separately to get Id
test = pd.read_csv("test.csv")
test_ids = test["Id"]
#test data
X_test, _ = wrangle_house("test.csv", is_train=False)
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
submission_1 = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred
})

submission_1.to_csv("submission1.csv", index=False)

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
submission_2 = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred_1 
})

submission_2.to_csv("submission2.csv", index=False)

# %%
#checking cross validation rmse for XGB
rmse = np.sqrt(-cross_val_score(
    model_1, X_train, y, scoring="neg_mean_squared_error"
))
print("CV RMSE:", rmse.mean())

# %%
plt.hist(pred_1, bins=50)



# %%
