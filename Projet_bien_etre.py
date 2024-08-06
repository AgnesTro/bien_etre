import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib
import pickle
import plotly.express as px
import altair as alt
import streamlit as st
from streamlit_navigation_bar import st_navbar


file_path = os.path.join(current_dir,"world-happiness-report.csv")
happy = pd.read_csv(file_path)

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir,"world-happiness-report-2021.csv")
happy_2021 = pd.read_csv(file_path)
st.set_page_config(
  page_title="Projet bien-etre / Juin 2024",
  page_icon="☀️",
  layout="wide")

page = st_navbar(["Introduction 🤓","Exploration 🔎", "DataVizualization 📊", "Modélisation 🤖"])
pages = ["Introduction 🤓","Exploration 🔎", "DataVizualization 📊", "Modélisation 🤖"]
alt.themes.enable("dark")

if page == pages[0] : 
  st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Introduction</p>', unsafe_allow_html=True)
  st.write("##### Le projet porte sur l'étude du bonheur perçu à travers le monde.")
  st.write("Le Rapport mondial sur le bonheur (en anglais : World Happiness Report) est une mesure du bonheur qui est publiée par le Réseau des solutions pour le développement durable des Nations unies.")
  st.write("Ce rapport est édité chaque année depuis 2012.")
  st.write("")
  st.write("---")
  st.write("Issu de parcours différents, nous avons décidé de nous former en data analyse chez Datascientest. ")
  st.markdown("""
    ###### Nos profils :
    - Frederic : Contrôleur de gestion - [LinkedIn](https://www.linkedin.com/in/fr%C3%A9d%C3%A9ric-denhez-342a9b1a8/)
    - Delphine : Chargée d'études en évaluation sensorielle - [LinkedIn](www.linkedin.com/in/delphine-tandeo)
    - Agnès : Directrice des opérations et directrice financière - [LinkedIn](https://www.linkedin.com/in/agnes-trocard)""")
  st.write("Ayant tous les 3 beaucoup voyagé au cours de notre vie, c'est notre ouverture sur le monde qui nous a poussé à nous demander comment est perçu le bonheur aux 4 coins de la planète, et surtout, a quoi est-il dû.")
  # st.markdown('<div style="color:#006A89; font-size:15px;font-weight:bold">On dit souvent que l'argent ne fait pas le bonheur c'est ce que nous allons entre-autres essayer de voir à travers ce projet.</div>', unsafe_allow_html=True)
  # st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Introduction</p>', unsafe_allow_html=True)

if page == pages[1] : 
  st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Présentation des données</p>', unsafe_allow_html=True)
  st.write("#### Données Happy")
  st.write("Premières lignes de happy :")
  st.dataframe(happy.head(10))
  st.write("La shape de happy est :",happy.shape)
  st.write("Description de happy :")
  st.dataframe(happy.describe())
  if st.checkbox("Afficher les NA de Happy") :
    st.dataframe(happy.isna().sum())
    st.write("Il y aura un traitement de NA à faire sur happy")
  st.write("---")
  st.write("#### Données Happy_2021")
  st.write("Premières lignes de happy_2021 :")
  st.dataframe(happy_2021.head(10))
  st.write("La shape de happy_2021 est :",happy_2021.shape)
  st.write("Description de happy_2021:")
  st.dataframe(happy_2021.describe())
  if st.checkbox("Afficher les NA de Happy_2021") :
    st.dataframe(happy_2021.isna().sum())
    st.write("Il n'y aura pas de traitement de NA à faire sur happy_2021")

happy_2021.columns = happy_2021.columns.str.replace(" ", "_")
happy_2021.columns = happy_2021.columns.str.replace(":","")
happy.columns = happy.columns.str.replace(" ", "_")
happy.columns = happy.columns.str.replace(":","")
happy = happy.rename(columns={'Life_Ladder':'Ladder_score','Log_GDP_per_capita':'Logged_GDP_per_capita','Healthy_life_expectancy_at_birth':'Healthy_life_expectancy'})

happy_2021_drop = happy_2021.drop(["Standard_error_of_ladder_score", "upperwhisker", "lowerwhisker", "Ladder_score_in_Dystopia",
                                   "Explained_by_Log_GDP_per_capita","Explained_by_Social_support", "Explained_by_Healthy_life_expectancy",
                                   "Explained_by_Freedom_to_make_life_choices", "Explained_by_Generosity",
                                   "Explained_by_Perceptions_of_corruption", "Dystopia_+_residual"], axis = 1)
happy_2021_drop["year"] = 2021

happy_drop = happy.drop(["Positive_affect", "Negative_affect"], axis = 1)

happy_concat = pd.concat([happy_drop,happy_2021_drop], axis=0)


happy_concat = happy_concat.drop("Regional_indicator", axis = 1)

Regional_indicator = happy_2021_drop.drop(happy_2021_drop.columns[2:], axis = 1)

happy_complet= pd.merge(Regional_indicator, happy_concat, on = ("Country_name"), how = "inner")

if page == pages[1] : 
  st.markdown("""
    #### Traitement des données :
    - Ajustement des noms de colonne en remplaçant les 'espaces' par '_'
    - Retrait des ':'
    - Ajustement des colonnes pour ne garder que les colonnes identiques
    - Ajout d'une colonne year à happy_2021 avec poru valeur 2021
    - Ajout d'une colonne Regional_indicator à happy via les valeurs de happy_2021""")
  
  st.write("#### Rassemblement des données : happy_complet")
  st.write("Premières lignes de happy_complet :")
  st.dataframe(happy_complet.head())
  st.write("La shape est : ",happy_complet.shape)
  st.write("Description de happy_complet:")
  st.dataframe(happy_complet.describe())
  if st.checkbox("Afficher les doublons de Happy_complet") :
    st.dataframe(happy_complet.duplicated().sum()) #A corriger
    st.write("Il n'y a pas de doublon sur happy_complet")
  if st.checkbox("Afficher les NA de Happy_copmlet") :
    st.dataframe(happy_complet.isna().sum())
    st.write("Il aura un traitement de NA à faire sur happy_complet")

if page == pages[1] : 
  st.write("#### Traitement des NAN :")
  st.write("Dans le souci de perfomance du projet, nous avons décidé de ne pas avoir de fuite de données. Nos NAN seront donc traités avec train_test,split")
  st.write("Nous avons toutefois éliminé les lignes avec plus de 3 NAN")
happy_complet=happy_complet.loc[(happy_complet.isna().sum(axis=1))< 3]

if page == pages[1] : 
  st.write("---")
  st.write("## Nous pouvons passer à la visualisation des données.")


if page == pages[2] : 
  st.markdown('<p style="color:#006A89; font-size:30px;text-align:center;font-weight:bold">DataVizualization</p>', unsafe_allow_html=True)
  fig_1 = go.Figure()
  fig_1.add_trace(go.Box(y = happy_complet["Ladder_score"], x =happy_complet["Regional_indicator"],fillcolor='moccasin'))
  fig_1.update_layout(title=" Distribution du score de bonheur toutes années confondues en fonction des zones géographiques \n")
  fig_1.update_layout(
  title="Distribution du score de bonheur en fonction des zones géographiques \n",
  plot_bgcolor='white',  # fond blanc pour la zone de tracé
  paper_bgcolor='white',  # fond blanc pour la figure entière
  height=800,
  width=1100,
  xaxis=dict( showline=True,linewidth=2, linecolor='black', showgrid=True,gridcolor='lightgray'), 
  yaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridcolor='lightgray'))
  st.write(fig_1)
  st.write("On constate que les scores de bien-être les plus élevés sont attribués aux zones Western Europe et North America and ANZ, ce qui correspond à l hémisphère Nord de la planète.")
  st.write("Les scores de bien-être les plus faibles sont quant à eux détenus dans les zones Sub-Saharian, Africa et South Asia, ce qui correspond à l hémisphère Sud de la planète.")
  st.write("Il y a un fort consensus pour la zone North America and ANZ, ce qui est logique vu qu elle contient moins de pays.")
  st.write("La distribution la plus éparpillée concerne la zone Middle East and North Africa avec des notes plus disparates.")

happy_complet=happy_complet.sort_values(by='year') 
if page == pages[2] : 
  fig = px.choropleth(happy_complet, locations='Country_name', locationmode='country names',color='Ladder_score', hover_name='Country_name', title='Life Ladder per country over the years',animation_frame='year',color_continuous_scale=px.colors.sequential.Plasma, width=1800,height=800)
  fig["layout"].pop("updatemenus")
  st.write(fig)
  st.write("Nous remarquons qu'au fur et à mesure des années, nous disposons de plus de données.")
  st.write("En effet, en 2005, les données n'étaient collectées que pour 27 pays. Il faut attendre 2011 pour avoir une stabilité dans le nombre des pays participant à cette étude à l'exception de l'année 2020 où les données ne sont disponibles que pour 95 pays, quid du Covid ?")
  st.write("Globalement, les pays ayant des ladder score élevés se situent en Amérique du Nord, en Europe et en Océanie. A l'inverse, les pays ayant des ladder scores faibles se situent principalement en Afrique et en Asie du Sud.")

years = happy_complet.year.unique()
df_list = []
for annee in years:
    df_annee = happy_complet[happy_complet['year'] == annee].nlargest(10, 'Ladder_score')
    df_list.append(df_annee)
happy_complet_top10 = pd.concat(df_list)

if page == pages[2] : 
  plt.figure(figsize=(12, 8))
  fig_2 = px.bar(happy_complet_top10.sort_values(by='year'),
             x='Ladder_score',
             y='Country_name',
             animation_frame='year',
             orientation='h',
             title='Top 10 des pays ayant le Ladder_score le plus élevé par année')
  fig_2.update_layout(height=800, width=1100)
  fig_2["layout"].pop("updatemenus")
  st.write(fig_2)
  st.write("Le pays ayant le plus haut ladder score en 2021 est la Finlande, avec un score de 7.842/10. Dans le TOP 10 des pays au plus haut ladder score, les ladder score sont compris entre 7.268 (Autriche) et 7.842 (Finlande).Nous retrouvons 9 pays d’Europe (et plus précisément de la région Western Europe) ainsi que la Nouvelle-Zélande (région North America and ANZ). On constate également que tous les pays scandinaves sont présents dans ce top 10.L'hémisphère Nord est donc largement représenté dans ce TOP 10.")


years = happy_complet.year.unique()
df_list = []
for annee in years:
    df_annee = happy_complet[happy_complet.year == annee].sort_values(by='Ladder_score',ascending = False).tail(10)
    df_list.append(df_annee)

happy_complet_flop10 = pd.concat(df_list)

if page == pages[2] : 
  fig_3 = px.bar(happy_complet_flop10.sort_values(by='year'), #tris des valeurs par an pour avoir une barre chronologique
             x='Ladder_score',
             y='Country_name',
             animation_frame='year',
             orientation='h',
             title=' Flop 10 des pays ayant le Ladder_score le plus faible par année')
  fig_3.update_layout(height=800, width=1100) #taille de la figure
  fig_3["layout"].pop("updatemenus")
  st.write(fig_3)
  st.write("Le pays ayant le plus faible score en 2021 est l'Afghanistan, avec un score de 2.523/10. Dans le TOP 10 des pays au plus faible ladder score, les ladder score sont compris entre 2.523 (Afghanistan) et 3.775 (Burundi).Nous retrouvons 7 pays de la région Sub-Saharan Africa (Zimbabwe, Burundi, Rwanda, Tanzanie, Botswana, Lesotho et Malawi), 1 pays de la région Middle East and North Africa(Yemen), 1 pays de la région Latin America and Caribbean (Haiti), et 1 pays de la région South Asia (Afghanistan).L'hémisphère Sud est donc exclusivement représenté dans ce TOP 10.")

happy_complet_flop10['Top_Flop']='Flop'
happy_complet_top10['Top_Flop']= 'Top'

Happy_complet_TF = pd.concat([happy_complet_top10,happy_complet_flop10])

if page == pages[2] :
  fig_4 = plt.figure(figsize = (20,10)) 
  fig_4 = px.bar(Happy_complet_TF,
             x='Regional_indicator',
             animation_frame='year',
            color = Happy_complet_TF.Top_Flop,#coloration des top et des flops
             color_discrete_map={'Top': 'lightblue','Flop': 'lightgreen'})
  fig_4.update_layout(
    title="Top/Flop 10 des pays par région en fonction du Ladder Score",
    plot_bgcolor='white',  # fond blanc pour la zone de tracé
    paper_bgcolor='white',  # fond blanc pour la figure entière
    height=1000,
    width=1500,
    xaxis=dict( showline=True,
        linewidth=2,
        linecolor='black',
        showgrid=True,  # garder les lignes de la grille
        gridcolor='lightgray',),
    yaxis=dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        showgrid=True,  # garder les lignes de la grille
        gridcolor='lightgray'))
  fig_4["layout"].pop("updatemenus")
  st.write(fig_4)
  st.write("Ce graphique permet de visualiser rapidement les tops et flop au cours des années et de placer géographiquement les résultats obtenus.")

heatmap = happy_complet.drop(['Country_name','Regional_indicator'], axis = 1 )

if page == pages[2] :
  st.write("###### Corrélation des données de happy_content :")
  fig_5=plt.figure(figsize = (15,15)) 

  sns.heatmap (heatmap.corr(), cmap = "coolwarm", annot = True)
  st.write(fig_5)
  st.markdown("""
#### Les 2 variables les plus corrélées avec le ladder score sont :
- Logged GPD per capita : coefficient de corrélation = 0.6. 
Plus le Logged GPD per capita est élevé, et plus le ladder score semble élevé.
- Social Support : coefficient de corrélation = 0.63.
Plus le Social Support est élevé, et plus le ladder score semble élevé.
###### Nous observons également un forte corrélation entre les variables explicatives suivantes :
- Logged GPD per capita et Healthy Life Expectancy : coefficient de corrélation = 0.55
Plus le Logged GPD per capita augmente, et plus le Healthy Life Expectancy augmente aussi.""")
  st.write("###### Graphique représentant la corrélation entre Healthy_life_expectancy et GDP :")
  fig_6 = plt.figure(figsize = (8,8))
  sns.scatterplot(y= "Healthy_life_expectancy" , x = "Logged_GDP_per_capita", data = happy_complet)
  st.write(fig_6)

# Ladder score sera notre variable à prédire. Nous allons étudier sa correlation avec les autres variables pour analyser les données importantes à garder.
fig_7 = plt.figure(figsize = (80,75))
plt.suptitle('Analyse de la correlation des variables en fonction avec le ladder score', fontsize=95)

# year
plt.subplot(421)
year = sns.barplot(y = 'Ladder_score', x = 'year', data = happy_complet.groupby(['year'])['Ladder_score'].mean().reset_index())
year.set_title('Years vs Ladder score')
plt.xticks(rotation = 45)
year.set_ylabel('Ladder score')

# Logged_GDP_per_capita
plt.subplot(422)
GDP = sns.scatterplot(y="Ladder_score",x='Logged_GDP_per_capita',data = happy_complet.groupby(['Logged_GDP_per_capita'])['Ladder_score'].mean().reset_index())
GDP.set_title('GDP vs Ladder score')
plt.xticks(rotation = 45)
GDP.set_ylabel('Ladder score')

# Social_support
plt.subplot(423)
social = sns.scatterplot(y="Ladder_score",x='Social_support',data = happy_complet.groupby(['Social_support'])['Ladder_score'].mean().reset_index())
social.set_title('Social support vs Ladder score')
plt.xticks(rotation = 45)
social.set_ylabel('Ladder score')

# Healthy_life_expectancy	==> avec tranche d'age
plt.subplot(424)
health = sns.barplot(y = 'Ladder_score', x = pd.cut(x = happy_complet['Healthy_life_expectancy'], bins = [30,40,50,60,70,80], labels = ['30-40', '40-50', '50-60', '60-70', '70-80'], right = False, include_lowest = True), data = happy_complet)
health.set_title('Healthy life expectancy vs Ladder score')
plt.xticks(rotation = 45)
health.set_ylabel('Ladder score')

# Freedom_to_make_life_choices	==> par quartile
plt.subplot(425)
free = sns.barplot(y = 'Ladder_score', x = pd.cut(x = happy_complet['Freedom_to_make_life_choices'], bins = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], labels = ['0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'], right = False, include_lowest = True), data = happy_complet)
free.set_title('Freedom vs Ladder score')
plt.xticks(rotation = 45)
free.set_ylabel('Ladder score')

# Generosity	==> ok ?
plt.subplot(426)
gene = sns.scatterplot(y = 'Ladder_score', x = 'Generosity', data = happy_complet.groupby(['Generosity'])['Ladder_score'].mean().reset_index())
gene.set_title('Generosity vs Ladder score')
plt.xticks(rotation = 45)
gene.set_ylabel('Ladder score')

# Perceptions_of_corruption
plt.subplot(427)
corr = sns.scatterplot(y = 'Ladder_score', x = 'Perceptions_of_corruption', data = happy_complet.groupby(['Perceptions_of_corruption'])['Ladder_score'].mean().reset_index())
gene.set_title('Perceptions_of_corruption vs Ladder score')
plt.xticks(rotation = 45)
gene.set_ylabel('Ladder score')

if page == pages[2] :
  st.write(fig_7)

  st.write("#### Nous pouvons passer au machine Learning")

if page == pages[3] : 
  st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Modélisation</p>', unsafe_allow_html=True)
