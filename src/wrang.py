import pandas as pd
import numpy as np

def wrangle_house(filepath, is_train=True):
    #load data
    df = pd.read_csv(filepath)

    #drop column Id
    df.drop("Id", axis=1, inplace=True)

    #target variable
    if is_train:
        y = np.log1p(df["SalePrice"])
        df = df.drop("SalePrice", axis=1)
    else:
        y = None

    #converting the missing values in the below column to None
    cols_none = [
        "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
    ]
    for col in cols_none:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    #converting the missing values in the below column to 0
    cols_zero = [
        "MasVnrArea","BsmtFullBath","BsmtHalfBath", "BsmtFinSF1","GarageYrBlt", "BsmtFinSF2",
        "BsmtUnfSF","TotalBsmtSF", "GarageCars","GarageArea"
    ]
    for col in cols_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    #converting the missing values in the below column using mode
    mode_cols = [
        "MSZoning", "Functional", "Exterior1st", "Electrical", "KitchenQual", "SaleType", "Exterior2nd"
    ]
    for col in mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    #drop utilities column
    if "Utilities" in df.columns:
        df.drop("Utilities", axis=1, inplace=True)

    ##converting the missing values in LotFrontage column using median
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df.groupby("Neighborhood")["LotFrontage"].transform("median"))

    #fixing skewed feature
    numeric_feats = df.select_dtypes(include=np.number)
    skewed_feats = numeric_feats.apply(lambda x: x.skew())
    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
    skewed_cols = skewed_feats.index
    df[skewed_cols] = np.log1p(df[skewed_cols])

    #convert/encode categorical column
    df = pd.get_dummies(df)

    return df, y