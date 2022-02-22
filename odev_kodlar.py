# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:30:22 2021

@author: Win7
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#verilerin yüklenmesi
veriler1 = pd.read_csv("vgsales.csv")

#veri hakkında ön bilgi edinilmesi
veriler1.info() 
eksik1 = veriler1.isnull().sum()
print(eksik1)

#eksik verilerin incelenmesi & silinmes
veriler2 = veriler1.dropna(how="any")
eksik2 = veriler2.isnull().sum()
print(veriler2,eksik2)



#kolonlar arasında ki ilişkinin incelenmesi 
print(veriler2.corr())

#tahminleme için verinin hazırlığı 
'''
Kolon sayısının en aza indirilmesi ve yüzdelik olarak başarının artması için
global satış (ulaşılmak istenen) kolonunun ilişkisi incelenmis ve  en fazla ilişiği olan kolon 
NA _Sales olduğu içn tahminleme yapılırken NA_Sales kolonu kullanılmıştır.
'''


tahmin = veriler2.drop(["Year","Rank","Name","Platform","Publisher","Genre","EU_Sales","JP_Sales","Other_Sales"], axis=1)
print(tahmin)

x = tahmin.iloc[:,1:].values 
y = tahmin.iloc[:,0:1].values 


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

print("lnr başarı yüzdesi:",lin_reg.predict([[100]])) #değerleri buraya giriniz  
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x), color = 'blue')
plt.show()


from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(x,y.ravel())

print("rnd başarı yüzdesi:",rf_reg.predict([[100]])) #değerleri buraya giriniz  

plt.scatter(x,y,color='yellow')
plt.plot(x,rf_reg.predict(x),color='black')
plt.show()


from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)

print("dtr başarı yüzdesi:",r_dt.predict([[100]])) #değerleri buraya giriniz 

plt.scatter(x,y, color='green')
plt.plot(x,r_dt.predict(x), color='orange')
plt.show()





#kümeleme için verilerin hazırlığı
'''
global satış  ve diğer satış kolonu baz alınarak  satış hacminin kaç kümeye bölünebileceği
ve oluşan kümenin küme sayısının for döngüsü  ile kontrolünün yapılması ve en optimize 
haliyle 6 ya bölünmesi plot yardımıyla gösterilmiştir.  
'''


X = veriler2.iloc[:,9:11].values

from sklearn.cluster import KMeans

kmeans = KMeans ( n_clusters = 6, init = 'k-means++', random_state=0)
Y_tahmin = kmeans.fit_predict(X)

sonuclar = []

for i in range(1,10):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 0)
    Y_tahmin = kmeans.fit_predict(X)
    sonuclar.append(kmeans.inertia_)



from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80)
plt.scatter(X[Y_tahmin == 0, 0], X[Y_tahmin == 0, 1], s = 100, c = 'red', label = 'Küme 1')
plt.scatter(X[Y_tahmin == 1, 0], X[Y_tahmin == 1, 1], s = 100, c = 'blue', label = 'Küme 2')
plt.scatter(X[Y_tahmin == 2, 0], X[Y_tahmin == 2, 1], s = 100, c = 'green', label = 'Küme 3')
plt.scatter(X[Y_tahmin == 3, 0], X[Y_tahmin == 3, 1], s = 100, c = 'cyan', label = 'Küme 4')
plt.scatter(X[Y_tahmin == 4, 0], X[Y_tahmin == 4, 1], s = 100, c = 'magenta', label = 'Küme 5')
plt.scatter(X[Y_tahmin == 5, 0], X[Y_tahmin == 5, 1], s = 100, c = 'yellow', label = 'Küme 6')
plt.title("küme dağılımları")
plt.xlabel("küme sayısı")
plt.ylabel("global satışlar")
plt.legend()
plt.show()
plt.plot(range(1,10),sonuclar) # noktalı dağılım tablosu ile karışmaması için alt tarafta bastırılmıştır.






