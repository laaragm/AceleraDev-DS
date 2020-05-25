#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[221]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[3]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[4]:


countries = pd.read_csv("countries.csv")


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[7]:


# Sua análise começa aqui.
pd.DataFrame({'dtypes': countries.dtypes,
             'missing values': countries.isna().sum()})


# In[8]:


countries.describe()


# In[25]:


#applymap: Apply a function to a Dataframe elementwise.
#map: Map values of Series according to input correspondence.
#strip: returns a copy of the string with both leading and trailing characters removed.
countries['Region'] = countries['Region'].map(lambda region: region.strip())
countries['Region']


# In[93]:


countries['Coastline_ratio'] = countries['Coastline_ratio'].str.replace(',', '.').astype(float)
countries['Infant_mortality'] = countries['Infant_mortality'].str.replace(',', '.').astype(float)
countries['Pop_density'] = countries['Pop_density'].str.replace(',', '.').astype(float)
countries['Net_migration'] = countries['Net_migration'].str.replace(',', '.').astype(float)
countries['Literacy'] = countries['Literacy'].str.replace(',', '.').astype(float)
countries['Phones_per_1000'] = countries['Phones_per_1000'].str.replace(',', '.').astype(float)
countries['Arable'] = countries['Arable'].str.replace(',', '.').astype(float)
countries['Crops'] = countries['Crops'].str.replace(',', '.').astype(float)
countries['Other'] = countries['Other'].str.replace(',', '.').astype(float)
countries['Birthrate'] = countries['Birthrate'].str.replace(',', '.').astype(float)
countries['Deathrate'] = countries['Deathrate'].str.replace(',', '.').astype(float)
countries['Agriculture'] = countries['Agriculture'].str.replace(',', '.').astype(float)
countries['Industry'] = countries['Industry'].str.replace(',', '.').astype(float)
countries['Service'] = countries['Service'].str.replace(',', '.').astype(float)


# In[94]:


countries.head()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[27]:


def q1():
    return list(sorted(countries['Region'].unique()))

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[111]:


#KBinsDiscretizer(n_bins=5, *, encode='onehot', strategy='quantile')
def q2():    
    kBinsDiscretizer = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'quantile')
    above90th = kBinsDiscretizer.fit_transform(countries[['Pop_density']])
                                                
    return len(above90th[above90th == 8])

q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[272]:


"""
OneHotEncoder cannot process string values directly. If your nominal features are strings, then you need to first
map them into integers.

pandas.get_dummies is kind of the opposite. By default, it only converts string columns into one-hot 
representation, unless columns are specified.
"""

def q3():    
    preprocessing = pd.get_dummies(countries[['Region', 'Climate']])
    #print(preprocessing)
    
    return preprocessing.shape[1]+1
    
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[264]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]
test_df = pd.DataFrame([test_country], columns=countries.columns)
test_df.head()


# In[247]:


numeric_features = ['Population', 'Area', 'Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality',
                    'GDP', 'Literacy', 'Phones_per_1000', 'Arable', 'Crops', 'Other', 'Birthrate', 'Deathrate',
                    'Agriculture', 'Industry', 'Service']
numeric_features


# In[270]:


def q4():   
    pipeline = Pipeline([("imputer", SimpleImputer(strategy='median')), ("standardScaler", StandardScaler())])
    pipeline.fit(countries[numeric_features])
    
    return float(round(pipeline.transform(test_df[numeric_features])[0][9], 3))
    
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[209]:


def q5():
    q1,q3 = np.quantile(countries.loc[:,'Net_migration'].dropna(), [0.25, 0.75])
    iqr = q3 - q1

    upper_outliers = len([val for val in countries.loc[:,'Net_migration'] if (val > (q3 + (1.5 * iqr)))])
    lower_outliers = len([val for val in countries.loc[:,'Net_migration'] if (val < (q1 - (1.5 * iqr)))])

    return tuple([lower_outliers, upper_outliers, False])

q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[211]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[228]:


def q6():
    countVectorizer = CountVectorizer()
    data = countVectorizer.fit_transform(newsgroup.data)
    
    return int(data[:, countVectorizer.vocabulary_['phone']].sum())
    
q6()    


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[234]:


def q7():
    tfidf = TfidfVectorizer()
    data = tfidf.fit_transform(newsgroup.data)
    
    return float(data[:, tfidf.vocabulary_['phone']].sum().round(3))

q7()

