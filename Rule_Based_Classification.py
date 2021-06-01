import numpy as np
import pandas as pd
import seaborn as sns

dataset = pd.read_csv("datasets/persona.csv")
df = dataset.copy()

#####VERİ SETİNİ GÖZLEMLEYELİM#####

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


#Kaç unique SOURCE var ve frekansları neler?
df["SOURCE"].value_counts()

#Kaç unique PRICE var?
df["PRICE"].nunique()

#Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

#Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()

#Ülkelere göre satışlardan toplam ne kadar kazanılmış
df.groupby(["COUNTRY"]).agg({"PRICE": "sum"})

#SOURCE türlerine göre göre satış sayıları nedir?
df.groupby(["SOURCE"]).agg({"PRICE": "count"})

#Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE": "mean"})

#SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE":"mean"})

#COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir
df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE": "mean"})

##### COUNTRY, SOURCE, SEX, AGE kırılımında toplam kazançlar nedir? #####
df.groupby(["COUNTRY","SOURCE","SEX","AGE",]).agg({"PRICE": "sum"})

##### Çıktıyı PRICE’a göre sıralayalım #####
agg_df = (df.groupby(["COUNTRY","SOURCE","SEX","AGE",]).agg({"PRICE": "sum"} )).sort_values("PRICE",ascending=False)

##### Index’te yer alan isimleri değişken ismine çevirelim #####
agg_df = agg_df.reset_index()

##### Age sayısal değişkenini kategorik değişkene çevirelim #####
bins = [0,19,24,31,41,agg_df["AGE"].max()]
my_labels = ["0_19","20_24","25_31","32_41","42_+42"]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"],bins,labels = my_labels)

##### Yeni seviye tabanlı müşterileri (persona) tanımlayalım #####
agg_df["customers_level_based"] = [str(values[0].upper())+"_"+str(values[1].upper())+"_"+str(values[2].upper())+"_"+ str(values[5]) for values in agg_df.values]

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

agg_df.head()

##### Yeni müşterileri (personaları) segmentlere ayıralım #####
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4 , labels = ["D","C","B","A"])
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum"]})

#C Segmentini analiz edelim
agg_df[agg_df["SEGMENT"] == "C"]
agg_df[agg_df["SEGMENT"] == "C"].count()
agg_df[agg_df["SEGMENT"] == "C"].mean()

### Şirket artık yeni müşterilerin kazandıracağı ortalama geliri tahmin edebilir###

#new_user
agg_df = agg_df.reset_index()
new_user = "TUR_ANDROID_FEMALE_32_41"
agg_df[agg_df["customers_level_based"] == new_user] #C segmenti

new_user_2 = "FRA_IOS_FEMALE_32_41"
agg_df[agg_df["customers_level_based"] == new_user_2] #D segmenti
