#!/usr/bin/env python
# coding: utf-8

# # Medical Cost

# Prédiction des coûts médicaux est un projet visant à développer un modèle de machine learning pour estimer les dépenses de santé d'un individu.

# ## Importer les bibliothèques nécessaires.

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  
import os
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
os.getcwd()
import warnings
warnings.filterwarnings('ignore')


# ## Importer la base de donnéé

# In[11]:


data = pd.read_csv('insurance.csv')


# In[12]:


data.head()


# Le projet de Prédiction des coûts médicaux vise à utiliser des techniques de machine learning pour estimer les dépenses de santé d'un individu en fonction de plusieurs caractéristiques personnelles et comportements liés à la santé. Ce type de prédiction est particulièrement utile dans le domaine des assurances et de la santé publique, car il permet d'anticiper les frais médicaux futurs, d'ajuster les primes d'assurance et de soutenir les politiques de santé en fonction des besoins de populations spécifiques.
# 
# Les données de ce projet incluent les caractéristiques suivantes :
# ######  Âge 
# Indique l'âge de l'individu, souvent corrélé aux besoins en soins de santé, car les personnes plus âgées ont tendance à avoir des coûts médicaux plus élevés.
# 
# ######  Sexe 
# Variable catégorielle indiquant le genre de l'individu, ce qui peut influencer certains risques pour la santé. Les assureurs peuvent observer des différences de coûts médicaux moyens entre hommes et femmes.
# 
# ######  BMI (Body Mass Index) 
# Une valeur continue mesurant la corpulence de l'individu. Un BMI élevé est souvent associé à des risques accrus de maladies chroniques, telles que le diabète et les maladies cardiovasculaires, entraînant des coûts médicaux plus élevés.
# 
# ###### Nombre d'enfants 
# Indique le nombre d'enfants à charge de l'individu, ce qui peut influer sur certains comportements de santé et de consommation, surtout si l'individu souscrit à une assurance santé pour sa famille.
# 
# ######  Statut de fumeur 
# Variable catégorielle indiquant si l'individu est fumeur ou non. Le tabagisme est un facteur de risque majeur pour de nombreuses maladies chroniques (comme les maladies respiratoires et cardiovasculaires), augmentant significativement les frais médicaux.
# 
# ######  Région géographique 
# Catégorie représentant la région de résidence de l'individu (par exemple, Nord-Est, Sud-Ouest, etc.). Les frais médicaux peuvent varier selon les régions, en raison des différences dans l'accès aux soins, les coûts des services de santé locaux et les politiques régionales.
# 
# ######  Charges 
# Le coût médical réel encouru par l'individu, qui est la variable cible. Le modèle essaiera de prédire cette valeur en se basant sur les autres facteurs.
# 
# L'objectif principal est de comprendre l'impact de chacune de ces variables sur les frais médicaux afin de créer un modèle prédictif fiable, capable d'estimer ces coûts selon les caractéristiques spécifiques de chaque individu. Ce modèle fournirait aux compagnies d'assurance des informations précises pour ajuster les primes en fonction des risques et aiderait les responsables de santé publique à identifier les populations les plus à risque, ce qui faciliterait des stratégies de prévention ciblées et des initiatives de réduction des coûts de santé.

# In[13]:


print("Diemsions de la BDD : ", data.shape)


# In[14]:


print("\n\n Types de variables : \n", data.dtypes)


# In[15]:


data.describe()


# ## Analyser la base de donnée

# In[16]:


# Vérifier s'il y a des données manquantes
data.isna().sum()


# In[17]:


# Vérifier s'il y a des données manquantes:visualisation graphique
plt.figure(figsize=(4,3))
sns.heatmap(data.isna())


# In[18]:


# In[19]:


### grouping columns 

cat_features = data.select_dtypes(include = 'object').columns

num_features = data.select_dtypes(include = ['int', 'float']).columns


# In[20]:


cat_features


# In[21]:


num_features


# In[22]:


for i in cat_features:

    print(data[i].value_counts())

    print("\n"+"-"*50)


# ## Interprétation de ces résultats 

# Cette analyse montre la distribution des valeurs pour les variables catégorielles (sex, smoker, region) dans le jeu de données, en utilisant la méthode value_counts pour afficher la fréquence de chaque catégorie.
# 
# ###### Sexe (sex) 
# 
# * Il y a 676 hommes et 662 femmes dans le jeu de données.
# * La répartition est assez équilibrée entre les hommes et les femmes, ce qui permet de bien représenter les deux genres dans les analyses futures.
# 
# ###### Statut de fumeur (smoker) 
# 
# * 1064 personnes ne fument pas (no), tandis que 274 personnes fument (yes).
# * Cela montre que la majorité des individus (environ 80%) ne fument pas, alors qu'une minorité (environ 20%) sont fumeurs. Cette variable est importante car le tabagisme peut avoir un impact significatif sur les coûts médicaux.
# 
# ###### Région (region) 
# 
# La distribution des individus par région est assez homogène :
# * 364 viennent de la région du Sud-Est (southeast)
# * 325 de la région du Sud-Ouest (southwest)
# * 325 de la région du Nord-Ouest (northwest)
# * 324 de la région du Nord-Est (northeast)
# Chaque région est représentée de manière relativement égale, ce qui garantit que les analyses futures ne seront pas biaisées en faveur d'une région spécifique.
# 
# ### Conclusion
# La répartition de ces variables catégorielles indique un bon équilibre des données, avec des représentations proportionnées pour le sexe et les régions. Cependant, le statut de fumeur est légèrement déséquilibré, avec une majorité de non-fumeurs. Ce déséquilibre est néanmoins réaliste, car, dans la plupart des populations, les non-fumeurs sont plus nombreux que les fumeurs. Cette variable peut jouer un rôle clé dans la prédiction des coûts médicaux, car le tabagisme est un facteur de risque important pour de nombreuses maladies.

# In[23]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, ax = plt.subplots(nrows=int(np.ceil(len(cat_features) / 3)), ncols=3, figsize=(15, len(cat_features)*3))
ax = ax.flatten()


for i, column in enumerate(cat_features):
    if column in data.columns:  
        sns.countplot(x=data[column], ax=ax[i])  
        ax[i].set_title(f'Countplot of {column}')  
    else:
        ax[i].axis('off')  
plt.tight_layout()
plt.show()


# In[25]:


le = LabelEncoder()
data["sex"]=le.fit_transform(data["sex"])
data["smoker"]=le.fit_transform(data["smoker"])
data["region"]=le.fit_transform(data["region"])


# In[26]:


data.head()


# In[27]:


plt.figure(figsize=(25, 20))
num_columns = len(data.columns)
rows = (num_columns // 3) + (num_columns % 3 > 0)  # Calculate rows needed for the columns
for i, col in enumerate(data.columns, 1):
    plt.subplot(rows, 3, i)  # Adjust the number of rows dynamically
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# In[28]:


sns.pairplot(data)
plt.show()


# In[46]:


fig, axis = plt.subplots(nrows = 1, ncols = 2,figsize=(15, 6))
ax = axis.flatten()
sns.distplot(data[data.smoker == 1]["charges"], ax = ax[0])
ax[0].set_title("Distribution of charges for smokers")
sns.distplot(data[data.smoker == 0]["charges"], ax = ax[1])
ax[1].set_title("Distribution of charges for non-smokers")


# In[30]:


sns.countplot(data = data, x= 'smoker', hue = 'sex')

plt.legend({"male", "female"})


# In[31]:


sns.countplot(data = data[data["children"] > 0], x = "smoker", hue = "sex" )

plt.legend({"male", "female"})


# ### Séparer les variables explicatives (X) et la variable cible (y)

# In[32]:


X = data.drop('charges', axis=1)
y = data['charges']


# In[33]:


scaler = StandardScaler()


# ### Entrainer les modesls 

# In[42]:


# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'LASSO Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Random Forest': RandomForestRegressor(random_state=42),
}
# Dictionary to store the scores
scores = {}
# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"{name} R^2 score: {score:.4f}")
    scores[name] = score
    print(f"{name} R^2 score: {score:.4f}")


# In[40]:


# Fit the scaler on the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved as 'scaler.joblib'")


# In[44]:


# Find the model with the highest R^2 score
best_model = max(scores, key=scores.get)
print(f"\nBest model: {best_model} with R^2 score: {scores[best_model]:.4f}")


# ### 1. Linear Regression (R² = 0.7833)
# Le modèle de régression linéaire a un score R² de 0.7833, ce qui signifie qu'environ 78,33% de la variance dans les données est expliquée par ce modèle. En d'autres termes, le modèle est relativement bon pour prédire la variable cible, avec une capacité explicative assez élevée, mais il reste une marge d'amélioration.
# ### 2. Ridge Regression (R² = 0.7831)
# Le modèle de régression Ridge, qui ajoute une régularisation L2 pour réduire le sur-apprentissage, donne un score R² très proche de celui de la régression linéaire (0.7831). Cela suggère que la régularisation n'a pas amélioré de manière significative la capacité prédictive du modèle par rapport à la régression linéaire classique dans ce cas précis. Ce score reste élevé, mais légèrement inférieur.
# ### 3. LASSO Regression (R² = 0.7833)
# La régression LASSO, qui applique une régularisation L1 (en favorisant la parcimonie des coefficients), donne un score R² similaire à celui de la régression linéaire (0.7833). Cela signifie que LASSO a également conservé une forte capacité prédictive sans trop de complexité dans les coefficients, tout comme la régression linéaire.
# ### 4. Elastic Net (R² = 0.4193)
# L'Elastic Net combine les régularisations L1 et L2. Cependant, son score R² est beaucoup plus faible (0.4193), ce qui suggère que ce modèle n'a pas bien capté la structure des données dans ce cas précis. Ce score indique que l'Elastic Net explique seulement 41,93% de la variance des données, ce qui est bien inférieur aux autres modèles. Il pourrait être dû à une mauvaise sélection des hyperparamètres ou à une interaction complexe entre les caractéristiques qui n'est pas bien modélisée par Elastic Net.
# ### 5. Random Forest (R² = 0.8643)
# Le modèle de Random Forest, qui est un ensemble d'arbres de décision, donne un score R² de 0.8643, ce qui signifie qu'il explique environ 86,43% de la variance dans les données. C'est le modèle le plus performant parmi ceux testés, indiquant qu'il capture mieux les relations complexes entre les variables indépendantes et la variable cible. Random Forest est moins susceptible de sous-ajuster les données et peut capturer des interactions non linéaires.
# ### Conclusion 
# Random Forest se distingue clairement comme le modèle le plus performant, avec un R² nettement plus élevé que les autres. Cela suggère qu'il est plus adapté aux données dans ce cas, en particulier si les relations sont non linéaires ou complexes.
# Les modèles de régression linéaire, Ridge et LASSO donnent des scores similaires, indiquant qu'une relation linéaire pourrait suffire à expliquer une grande partie des données, bien qu'il y ait un léger avantage à utiliser la régression linéaire ou LASSO dans ce cas.
# Elastic Net présente des performances décevantes, ce qui pourrait indiquer qu'un ajustement plus fin des hyperparamètres est nécessaire ou que ses hypothèses de régularisation ne sont pas adaptées à vos données.
# Cela suggère qu'un modèle comme Random Forest pourrait être préféré pour sa meilleure capacité à capturer la complexité des données, tandis que les modèles linéaires peuvent être utilisés si la simplicité et l'interprétabilité sont essentielles.

# In[56]:
# Save the model
joblib.dump(LinearRegression(), 'model.joblib')

# Copy the model to a specific folder
source_path = 'model.joblib'
destination_path = 'C:\\Users\dell\PycharmProjects\Medical Cost> '
print("Model saved to project folder.")

