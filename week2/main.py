#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head(10)


# In[4]:


black_friday.shape


# In[5]:


black_friday.dtypes


# In[6]:


black_friday.info()


# In[7]:


black_friday.columns


# In[8]:


black_friday.isna().sum()


# In[9]:


aux = pd.DataFrame({'Columns': black_friday.columns, 'Types': black_friday.dtypes,
                             'Missing Data Rate': black_friday.isna().sum()/black_friday.shape[0]})
aux


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[10]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape

ansq1 = q1()
print(ansq1)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[11]:


def q2():
    # Retorne aqui o resultado da questão 2.
    filtering = black_friday.loc[(black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')]
    return int(filtering['User_ID'].count())
    
ansq2 = q2()
print(ansq2)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[12]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday.loc[:, 'User_ID'].nunique() #nunique(): Count distinct observations over requested axis

ansq3 = q3()
print(ansq3)


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[13]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()

ansq4 = q4()
print(ansq4)


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[14]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return (len(black_friday) - len(black_friday.dropna())) / (len(black_friday)) 

ansq5 = q5()
print(ansq5)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[15]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return black_friday.isna().sum().max()

ansq6 = q6()
print(ansq6)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[16]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(black_friday['Product_Category_3'].mode())

ansq7 = q7()
print(ansq7)


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[17]:


def q8():
    # Retorne aqui o resultado da questão 8.
    min_value = black_friday.loc[:, 'Purchase'].min()
    max_value = black_friday.loc[:, 'Purchase'].max()
    return ((black_friday.loc[:,'Purchase'] - min_value) / (max_value - min_value)).mean()

ansq8 = q8()
print(ansq8)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[20]:


def q9():
    # Retorne aqui o resultado da questão 9.
    normalize = (black_friday.loc[:, 'Purchase'] - black_friday.loc[:, 'Purchase'].mean()) / black_friday.loc[:, 'Purchase'].std()
    
    return int(normalize[(normalize >= -1) & (normalize <= 1)].count())

    
ansq9 = q9()
print(ansq9)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[19]:


def q10():
    # Retorne aqui o resultado da questão 10.
    comparision = (black_friday['Product_Category_2'].isna() == black_friday['Product_Category_3'].isna())
    comparision = comparision.unique()
    if comparision[0]:
        return True
    return False
      
ansq10 = q10()
print(ansq10)


# In[ ]:




