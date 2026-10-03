import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandera.pandas as pa
from pandera.pandas import DataFrameSchema,Column,Check
########################PHASE1#############################
df=pd.read_csv("dataanalysis.csv")
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.isna().mean()*100)
df["CouponCode"]=df["CouponCode"].fillna("NoCoupon")
print(df.isna().sum())
num_cols = ["Quantity", "UnitPrice", "ItemsInCart", "TotalPrice"]
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    print(col, "lower:", lower, "upper:", upper, "outliers:", count)
    df[col] = np.clip(df[col], lower, upper)
print(df.shape)
print(df.isna().sum())
print(df[num_cols].describe())
df.to_csv("PHASE1.csv", index=False)
###################PHASE 2##############################
#adding three new features
df["Date"]=pd.to_datetime(df["Date"])
df["Month"]=df["Date"].dt.month
df["BigOrder"]=(df["Quantity"]>4).astype(int)
df["HasCoupon"]=(df["CouponCode"]!="NoCoupon").astype(int)
print(df.head())
print(df.info())
print(df.shape)
#saving changes
df.to_csv("phase2_features.csv", index=False)
#dropping column that have different values for each row
df = df.drop(columns=["OrderID", "CustomerID", "TrackingNumber", "ShippingAddress", "Date"])
print(df.shape)
#encoding categorical columns
cat_cols = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource"]
encoder = OneHotEncoder(drop="first", sparse_output=False)
res = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(res, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
df = df.drop(columns=cat_cols)
df = pd.concat([df, encoded_df], axis=1)
print(df.head())
print(df.info())
print(df.shape)
df.to_csv("phase2_encoded.csv", index=False)
corr = df.corr().abs()
top_half = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = top_half.stack()
pairs = pairs[pairs > 0.80]
print(pairs)
df = df.drop(columns=["CouponCode_NoCoupon"])
print(df.shape)
#######################phase3####################################
schema=DataFrameSchema({
    "Quantity": Column(int,Check.in_range(1, 5),coerce=True),
    "UnitPrice": Column(float,Check.greater_than(0),coerce=True),
    "ItemsInCart":Column(int,Check.in_range(1, 10),coerce=True),
    "TotalPrice":Column(float,Check.greater_than(0),coerce=True),
    "Month": Column(int,Check.in_range(1, 12),coerce=True),
    "BigOrder": Column(int,Check.isin([0, 1]),coerce=True),
    "HasCoupon":Column(int,Check.isin([0, 1]),coerce=True),
})
df=schema.validate(df)
print("passed")
print(df.shape)
df.to_csv("phase3_final.csv", index=False)
