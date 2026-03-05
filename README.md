# house-prices-advanced-regression-techniques
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. Rather, they focus on price negotiations. With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, i predict the final price of each home.

# Dataset
Kaggle: house-prices-advanced-regression-techniques
The dataset contains 79 explanatory variables describing various aspects of residential homes in Ames, Iowa.

# Structure
- data/ : train and test datasets(train.csv and test.csv)
- src/ : data wrangling functions(wrang.py)
- submissions/ : Kaggle submission files
- house.py : model training and prediction script

# Models Used
- RandomForestRegressor(scikit-learn)
- XGBRegressor XGBoost

# Evaluation metrics
Model was evaluated using the Root Mean Squared Error(RMSE).
RESULTS:
- CV RMSE for randomforestregressor: 0.14286325656974525
- CV RMSE for XGBRegressor: 0.12509518348380097

# Findings
From the predicted price distribution(histogram)
- Most houses between $100k – $200k
- Fewer houses above $300k
- Very few close to $500k
Which is normal for real estate prices
