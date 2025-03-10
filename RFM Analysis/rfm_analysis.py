import pandas as pd
import datetime as dt

pd.set_option("display.max_columns", None)
pd.set_option('display.width', 600)

# Step 1: Read the flo_data_20K.csv file and create a copy
df_ = pd.read_csv("/Users/pelinsukuzu/Desktop/Data Science Bootcamp/CRM Analitiği/Case Study 1/flo_data_20k.csv")
df = df_.copy()

# Step 2: Examine the dataset
print(df.head(10))  # First 10 observations
print(df.columns)  # Column names
print(df.describe().T)  # Descriptive statistics
print(df.isnull().sum())  # Missing values
print(df.dtypes)  # Data types
print(df.shape)  # Shape of dataset
print(df["master_id"].nunique())  # Unique customers

# Step 3: Create new variables for total orders and total spending
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = (df["order_num_total_ever_online"] * df["customer_value_total_ever_online"] + 
                     df["order_num_total_ever_offline"] * df["customer_value_total_ever_offline"])

# Step 4: Convert date-related columns to datetime format
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Step 5: Analyze the distribution of customer count, total products, and total spending by order channel
print(df.groupby("order_channel").agg({"master_id": "count", "total_order": "sum", "total_value": "sum"}))

# Step 6: Find the top 10 customers by total spending
print(df.groupby("master_id").agg({"total_value": "sum"}).sort_values("total_value", ascending=False).head(10))

# Step 7: Find the top 10 customers by order count
print(df.groupby("master_id").agg({"total_order": "sum"}).sort_values("total_order", ascending=False).head(10))

# Step 8: Define a preprocessing function
def preprocess_data(dataframe):
    dataframe.dropna(inplace=True)
    dataframe["total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_value"] = (dataframe["order_num_total_ever_online"] * dataframe["customer_value_total_ever_online"] +
                                dataframe["order_num_total_ever_offline"] * dataframe["customer_value_total_ever_offline"])
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return dataframe

df = preprocess_data(df)

# Step 9: Calculate RFM metrics
today_date = dt.datetime(2021, 6, 1)
rfm = df.groupby("master_id").agg({
    "last_order_date": lambda date: (today_date - date.max()).days,
    "total_order": "sum",
    "total_value": "sum"
})

# Rename columns
rfm.columns = ["recency", "frequency", "monetary"]

# Step 10: Convert RFM metrics to scores
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# Step 11: Create RF score
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# Step 12: Define customer segments
seg_map = {
    r'[1-2][1-2]': "hibernating",
    r'[1-2][3-4]': "at_risk",
    r'[1-2]5': "cant_loose",
    r'3[1-2]': "about_to_sleep",
    r'33': "need_attention",
    r'[3-4][4-5]': "loyal_customers",
    r'41': "promising",
    r'51': "new_customers",
    r'[4-5][2-3]': "potential_loyalists",
    r'5[4-5]': "champions"
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

# Step 13: Analyze segment statistics
print(rfm.groupby("segment").agg({"recency": "mean", "frequency": "mean", "monetary": "mean"}))

# Step 14: Identify target customers for campaigns

# a. Identify customers for the new women's shoe brand (loyal customers & champions who purchased women's products)
df_final = df.merge(rfm, how="left", on="master_id")
target_women = df_final[(df_final["segment"].isin(["loyal_customers", "champions"])) & 
                         (df_final["interested_in_categories_12"].str.contains("KADIN"))]

# Save the list of target customers
target_women["master_id"].to_csv("target_women_customers.csv", index=False)

# b. Identify customers for the 40% discount on men's and children's products
target_men_children = df_final[(df_final["segment"].isin(["cant_loose", "about_to_sleep", "new_customers"])) & 
                               (df_final["interested_in_categories_12"].str.contains("ERKEK|COCUK"))]

# Save the list of target customers
target_men_children["master_id"].to_csv("target_men_children_discount.csv", index=False)
