#Kütüphanelerin eklenmesi
import pandas as pd
import numpy as np

#Veri setinin eklenip başlığının belirlenmesi
column = ['yorum']
df = pd.read_csv('yorumlar.csv', encoding ='iso-8859-9', sep='"')
df.columns=column
df.info()
df.head()
#df.dropna(inplace=True)

#Veri setindeki Türkçe dolgu kelimelerinin kaldırılması
def remove_stopwords(df_fon):
    stopwords = open('turkce-stop-words', 'r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
        [word for word in doc if word not in stopwords], df_fon['yorum']))

remove_stopwords(df)


#Verisetinde Positivity adlı bir sütunun oluşturalım ve başlangıçta tüm değerlere olumlu olarak 1 değerinin atayalım
df['Positivity'] = 1

#Veri setimizde 10003. veri ve sonrasındaki veriler olumsuz(negatif) verilerdir bu yüzden bu değerlerin
#Positivity değerleri 0 ile değiştirelim.
df.Positivity.iloc[10003:] = 0

#Şimdi, verileri "yorum" ve "Positivity" sütunlarını kullanarak rastgele eğitim ve test alt kümelerini bölüştürelim ve 
#ardından ilk girişi ve eğitim setinin şeklini yazalım.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['yorum'], df['Positivity'], random_state = 0)
print(X_train.head())
print('\n\nX_train shape: ', X_train.shape)

#CountVectorizer'ı başlatıyoruz ve eğitim verilerimize uyguluyoruz.
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(encoding ='iso-8859-9').fit(X_train)

#X_train'deki belgeleri bir belge terim matrisine dönüştürürüz
X_train_vectorized = vect.transform(X_train) 

#Bu özellik matrisi X_ train_ vectorized'e dayanarak Lojistik Regresyon sınıflandırıcısını eğiteceğiz
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

#Daha sonra, X_test kullanarak tahminler yapacağız ve eğri puanının altındaki alanı hesaplayacağız.
from sklearn.metrics import roc_auc_score
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

#Modelimizin bu tahminleri nasıl yaptığını daha iyi anlamak için, her bir özellik için katsayıları (bir kelime), 
#pozitifliği ve olumsuzluk açısından ağırlığını belirlemek için kullanabiliriz.
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Negatif: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Pozitif: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))


#tf-idf vectorizer'ı başlatacağız ve eğitim verilerimize uygulayacağız. 
#En az beş dokümanda görünen kelime dağarcığımızdaki kelimeleri kaldıracağımız min_df = 5 değerini belirtiyoruz.
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(X_train)

X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))  

feature_names = np.array(vect.get_feature_names())
sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
print('En küçük Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('En büyük Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:-11:-1]]))


#bigramlar, bitişik kelimelerin çiftlerini sayar ve bize kötü ve kötü olmayan gibi özellikler verebilir. 
#Bu nedenle, minimum 5 belge sıklığını belirten ve 1 gram ve 2 gram ekstrakt eden eğitim setimizi yeniden yerleştiriyoruz.
vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

#Her özelliği kontrol etmek için katsayıları kullanarak görebiliriz
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Negatif: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
print('Pozitif Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))

while(True):
    yorum=input("Yorumunuz Nedir?(Programdan çıkmak için \'F\' yazınız)")
    if(yorum == 'F' or yorum == 'f'):
        break
    else:
        print(model.predict(vect.transform([yorum])))