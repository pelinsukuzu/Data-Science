##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option('display.width', 600)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("/Users/pelinsukuzu/Desktop/Data Science Bootcamp/CRM Analitiği/Case Study 2/flo_data_20k.csv")
df = df_.copy()
df.head()


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe[variable] = dataframe[variable].round()

# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.

df.head()
df.isnull().sum()
df.describe().T

outlier_thresholds(df,["order_num_total_ever_online", "order_num_total_ever_offline","customer_value_total_ever_offline", "customer_value_total_ever_online"])
replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")

#def replace_with_thresholds(dataframe, col_names):
#   """
#   Birden fazla kolon için aykırı değerlerin yerine alt ve üst sınırları yerleştirir ve yuvarlar.

#   Parameters:
#   dataframe (pd.DataFrame): Veri çerçevesi
#   col_names (list): İşlem yapılacak kolonların isimleri listesi
#   """
#   for col in col_names:
#       lower_limit, upper_limit = outlier_thresholds(dataframe, col)
#       dataframe.loc[dataframe[col] < lower_limit, col] = lower_limit
#       dataframe.loc[dataframe[col] > upper_limit, col] = upper_limit
#
        # Yuvarlama işlemi
#       dataframe[col] = dataframe[col].round()
#Böylece
# Çoklu kolonlar için aykırı değerlerin baskılanması ve yuvarlanması
# cols_to_check = ["order_num_total_ever_online", "order_num_total_ever_offline",
#                 "customer_value_total_ever_offline", "customer_value_total_ever_online"]

# replace_with_thresholds(df, cols_to_check)

df.describe().T

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["order_num_total_ever_online"]*df["customer_value_total_ever_online"] + df["order_num_total_ever_offline"]*df["customer_value_total_ever_offline"]
df.head()

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

today_date = (df["last_order_date"].max()) + pd.Timedelta(days=2)
#pd.Timedelta(days=2) -> Bu, o tarihe 2 gün eklemek için kullanılır.
today_date

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency"] = (df["last_order_date"]-df["first_order_date"]).dt.days / 7
cltv_df["tenure"] = (today_date-df["first_order_date"]).dt.days / 7
cltv_df["frequency"] =df["total_order"]
cltv_df["monetary_cltv_avg"]= df["total_customer_value"] / df["total_order"]
cltv_df.columns=["customer_id","recency_cltv_weekly","T_weekly","frequency","monetary_cltv_avg",]
cltv_df.head()
#cltv_df = df.groupby("master_id").agg({"last_order_date": [lambda last_order_date: (last_order_date.max() - df["first_order_date"].min()).days/7,
#                                      lambda last_order_date: (today_date-last_order_date).days/7],
#                                       "total_order": "sum",
#                                      "total_customer_value": lambda total_customer_value: (total_customer_value.sum()/total_customer_value.count)})
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df.head()
cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12,
                                                         cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"]
                                           )
# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv
cltv_df.head()
# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv", ascending=False).head(20)

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması

# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.

cltv_df["segment"]= pd.qcut(cltv_df["cltv"],4,labels=["D","C","B","A"])
cltv_df.head()
# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz



# BONUS: Tüm süreci fonksiyonlaştırınız.

def create_cltv_df(dataframe):

    # Veriyi Hazırlama
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # CLTV veri yapısının oluşturulması
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # # Gamma-Gamma Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Cltv tahmini
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentleme
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)

