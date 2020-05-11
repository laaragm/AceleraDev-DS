#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[80]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from loguru import logger


# In[60]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

sns.set()


# In[128]:


fifa = pd.read_csv("data.csv")
fifa.head()


# In[129]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[130]:


fifa.head()


# In[131]:


fifa.dtypes


# In[132]:


fifa.describe()


# In[133]:


fifa.info()


# In[134]:


fifa.isna().sum()


# In[137]:


fifa.shape


# In[138]:


#fifa.fillna(fifa.mean()) poderia ser uma abordagem para tratar dos missing values, mas aqui iremos removê-los
#removendo missing values
fifa.dropna(inplace=True)


# In[139]:


fifa.shape


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[140]:


def q1():
    # Retorne aqui o resultado da questão 1.
    pca = PCA(n_components=2)
    pca_transformation = pca.fit(fifa)
    val = pca_transformation.explained_variance_ratio_
    print(val)
    return float(val[0].round(3))
    
q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[141]:


def q2():
    # Retorne aqui o resultado da questão 2.
    pca = PCA(0.95)
    pca_transformation = pca.fit(fifa)
    
    cumulative_variance_ratio = np.cumsum(pca_transformation.explained_variance_ratio_)
    component_number = np.argmax(cumulative_variance_ratio >= 0.95) + 1 

    return int(component_number.round(3))
    
q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[142]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[143]:


def q3():
    # Retorne aqui o resultado da questão 3.    
    pca = PCA()
    pca.fit(fifa)
    pc = pca.components_.dot(x)
    
    cont = 0
    pc_list = []
    for val in pc:
        if cont < 2:
            pc_list.append(val.round(3))
        cont += 1
    
    return tuple(pc_list)

q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[144]:


target = 'Overall'
features = fifa.drop(columns=target).columns

y_train = fifa[target]
X_train = fifa[features]

def q4():
    # Retorne aqui o resultado da questão 4.
    model = LinearRegression()
    
    rfe = RFE(estimator=model, n_features_to_select=5)
    rfe.fit(X_train, y_train)
    
    features_list = []
    for feature in features[rfe.support_]:
        features_list.append(feature)
    
    return features_list
    
q4()
    

