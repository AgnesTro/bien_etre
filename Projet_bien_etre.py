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
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Projet bien-etre / Juin 2024",page_icon="🌍",layout="wide")
page = st_navbar(["Introduction 🤓","Exploration 🔎", "Data Visualisation 📊", "Modélisation 🤖","Conclusion🎯"])
pages = ["Introduction 🤓","Exploration 🔎", "Data Visualisation 📊", "Modélisation 🤖","Conclusion🎯"]
alt.themes.enable("dark")

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir,"world-happiness-report.csv")
happy = pd.read_csv(file_path)

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir,"world-happiness-report-2021.csv")
happy_2021 = pd.read_csv(file_path)

# #exemple
# @st.cache

# @st.cache_data
# def fetch_and_clean_data(csv):
#     return data

# happy = fetch_and_clean_data("world-happiness-report.csv")
# happy_2021 = fetch_and_clean_data('world-happiness-report-2021.csv')
# # fin


if page == pages[0] : 
  st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Introduction</p>', unsafe_allow_html=True)
  st.write("##### Le projet porte sur l'étude du bonheur perçu à travers le monde.🌍🫶")
  st.write("Le Rapport mondial sur le bonheur (en anglais : World Happiness Report) est une mesure du bonheur qui est publiée par le Réseau des solutions pour le développement durable des Nations unies.")
  st.write("Ce rapport est édité chaque année depuis 2012.")
  st.write('Les données dont nous disposons sont donc issues de la collecte du World Happiness report.')
  st.write("")
  st.write("---")
  st.write("Issu de parcours différents, nous avons décidé de nous former en data analyse chez Datascientest. ")
  st.markdown("""
    ###### Nos profils :
    - Frederic : Contrôleur de gestion - [LinkedIn](https://www.linkedin.com/in/fr%C3%A9d%C3%A9ric-denhez-342a9b1a8/)
    - Delphine : Chargée d'études en évaluation sensorielle - [LinkedIn](https://www.linkedin.com/in/delphine-tandeo)
    - Agnès : Directrice des opérations et directrice financière - [LinkedIn](https://www.linkedin.com/in/agnes-trocard)""")
  st.markdown('''
<p style="margin-bottom: 20px;">
    Ayant tous les 3 beaucoup voyagé au cours de notre vie, c'est notre ouverture sur le monde qui nous a poussé à nous demander comment est perçu le bonheur aux 4 coins de la planète, et surtout, a quoi est-il dû. <br>''', unsafe_allow_html=True)
  st.markdown('''
<div style="color:#003885; font-size:24px; font-weight:bold; text-align:center; margin-top: 20px;">
    On dit souvent que l'argent ne fait pas le bonheur. C'est ce que nous allons entre-autres essayer de voir à travers ce projet !
</div>
''', unsafe_allow_html=True)

if page == pages[1] : 
  st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Présentation des données</p>', unsafe_allow_html=True)
  st.markdown('''<div style="color:#006A89; font-size:24 px;font-weight:bold;">Données Happy</div>''', unsafe_allow_html=True)
  st.write("Premières lignes de happy :")
  st.dataframe(happy.head(10))
  st.write("La shape de happy est :",happy.shape)
  st.write("Description de happy :")
  st.dataframe(happy.describe())
  if st.checkbox("Afficher les NA de Happy") :
    st.dataframe(happy.isna().sum())
    st.write("Il y aura un traitement de NA à faire sur happy")
  st.write("---")
  st.markdown('''<div style="color:#006A89; font-size:24 px;font-weight:bold;">Données Happy_2021</div>''', unsafe_allow_html=True)
  st.write("Premières lignes de happy_2021 :")
  st.dataframe(happy_2021.head(10))
  st.write("La shape de happy_2021 est :",happy_2021.shape)
  st.write("Description de happy_2021:")
  st.dataframe(happy_2021.describe())
  if st.checkbox("Afficher les NA de Happy_2021") :
    st.dataframe(happy_2021.isna().sum())
    st.write("Il n'y aura pas de traitement de NA à faire sur happy_2021")
  st.write("---")

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
    ###### Traitement des données :
    - Ajustement des noms de colonne en remplaçant les 'espaces' par '_'
    - Retrait des ':'
    - Ajustement des colonnes pour ne garder que les colonnes identiques
    - Ajout d'une colonne year à happy_2021 avec poru valeur 2021
    - Ajout d'une colonne Regional_indicator à happy via les valeurs de happy_2021""")
  st.markdown('''<div style="color:#006A89; font-size:24 px;font-weight:bold;">Rassemblement des données : happy_complet</div>''', unsafe_allow_html=True)
  st.write("Premières lignes de happy_complet :")
  st.dataframe(happy_complet.head())
  st.write("La shape est : ",happy_complet.shape)
  st.write("Description de happy_complet:")
  st.dataframe(happy_complet.describe())
  if st.checkbox("Afficher les doublons de Happy_complet") :
     st.write("Il n'y a pas de doublon sur happy_complet")
  if st.checkbox("Afficher les NA de Happy_complet") :
    st.dataframe(happy_complet.isna().sum())
    st.write("Il aura un traitement de NA à faire sur happy_complet")
  st.write("---")
  st.markdown('''<div style="color:#006A89; font-size:24 px;font-weight:bold;">Traitement des NAN :</div>''', unsafe_allow_html=True)
  st.write("Dans le souci de perfomance du projet, nous avons décidé de ne pas avoir de fuite de données. Nos NAN seront donc traités avec train_test,split")
  st.write("Nous avons toutefois éliminé les lignes avec plus de 3 NAN")

happy_complet=happy_complet.loc[(happy_complet.isna().sum(axis=1))< 3]

if page == pages[1] : 
  st.markdown('''
              <div style="text-align:center; margin-top: 20px;color:#006A89">Nous pouvons passer à la visualisation des données</div>
</div>''', unsafe_allow_html=True)

if page == pages[2]: 
  st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Exploration graphique des données</p>', unsafe_allow_html=True)
  st.write("")
  st.write("")
  st.write("###### Dans un premier temps, nous visualiserons nos valeurs.")
  # Fonction de création de figure avec titres centrés et taille de police ajustée
  
  def create_fig(title, color, height, width):
      fig = go.Figure()
      fig.update_layout(
          title=dict(text=title, font=dict(size=24), x=0.5, xanchor='center'),
          plot_bgcolor='white',
          paper_bgcolor='white',
          height=height,
          width=width,
          xaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridcolor='lightgray'),
          yaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridcolor='lightgray'))
      return fig
  
  # Graphique Boxplot
  fig_1 = create_fig("Distribution du score de bonheur en fonction des zones géographiques",
      color='blue',
      height=600,
      width=400)
  fig_1.add_trace(go.Box(y=happy_complet["Ladder_score"], x=happy_complet["Regional_indicator"], fillcolor='#87AC99', line=dict(color='#69939D'), marker=dict(color='#873260')))
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
     st.plotly_chart(fig_1, use_container_width=True)
  
  show_text = st.checkbox("Interpretation du box plot")
# Afficher le texte en fonction de l'état de la case à cocher
  if show_text:
     st.write("On constate que les scores de bien-être les plus élevés sont attribués aux zones Western Europe et North America and ANZ, ce qui correspond à l'hémisphère Nord de la planète.")
     st.write("Les scores de bien-être les plus faibles sont quant à eux détenus dans les zones Sub-Saharian, Africa et South Asia, ce qui correspond à l'hémisphère Sud de la planète.")
     st.write("Il y a un fort consensus pour la zone North America and ANZ (ce qui est logique vu qu'elle contient moins de pays).")
     st.write("La distribution la plus éparpillée concerne la zone Middle East and North Africa avec des notes plus disparates. Enfin, on constate certains outliers :")
     st.write("- 2 pour la zone Western Europe : 4,72 et 4,756")
     st.write("- 10 pour la zone Latin America and Caribbean : 3.352, 3.57, 3.615, 3.754, 3.766, 3.824, 3.846, 3.889, 4.041 et 4.413")
     st.write("- 1 pour la zone North America and ANZ : 6.804")
     st.write("- 4 pour la zone South Asia : 2.375, 2.523, 2.662, 2.694")
     st.write("- 1 pour la zone Sub Saharian Africa. 6.241. C'est le seul outlier situé dans la partie supérieure de la boîte à moustaches.")
     st.write("En regardant de plus près ces outliers, on constate qu'il s'agit certes de valeurs extrêmes mais non de valeurs aberrantes, car elles sont bien comprises entre 0 et 10, soit l'échelle de notre ladder score. Ces scores sont tout à fait plausibles. Nous conservons donc ces données.")

  # Graphique Choropleth
  fig_2 = px.choropleth(
      happy_complet,
      locations='Country_name',
      locationmode='country names',
      color='Ladder_score',
      hover_name='Country_name',
      animation_frame='year',
      color_continuous_scale=px.colors.sequential.Plasma,
      width=800,
      height=600)
  fig_2.update_layout(
      title=dict(text="Carte du monde représentant l'évolution du ladder score au fil des années", font=dict(size=26), x=0.5, xanchor='center'),
      geo=dict(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray"))
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
     st.plotly_chart(fig_2, use_container_width=True)

  show_text_2 = st.checkbox("Interpretation de la carte")
  # Afficher le texte en fonction de l'état de la case à cocher
  if show_text_2:
      st.write("Nous remarquons qu'au fur et à mesure des années, nous disposons de plus de données.")
      st.write("En effet, en 2005, les données n'étaient collectées que pour 27 pays. Il faut attendre 2011 pour avoir une stabilité dans le nombre des pays participant à cette étude")
      st.write("A l'exception de l'année 2020 où les données ne sont disponibles que pour 95 pays, quid du Covid ?)")
      st.write("Globalement, les pays ayant des ladder score élevés se situent en Amérique du Nord, en Europe et en Océanie.")
      st.write("A l'inverse, les pays ayant des ladder scores faibles se situent principalement en Afrique et en Asie du Sud.")
   
  # Top 10 Ladder Score
  happy_complet_top10 = happy_complet.groupby('year').apply(lambda x: x.nlargest(10, 'Ladder_score')).reset_index(drop=True)
  fig_3 = px.bar(
      happy_complet_top10,
      x='Ladder_score',
      y='Country_name',
      color_discrete_sequence=['#6B9998'],
      animation_frame='year',
      orientation='h',
      title='Top 10 des pays ayant le Ladder_score le plus élevé par année',
      width=200,
      height=600)
  fig_3.update_layout(title=dict(text='Top 10 des pays ayant le Ladder_score le plus élevé par année', font=dict(size=24), x=0.5, xanchor='center'))
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
     st.plotly_chart(fig_3, use_container_width=True)

  
  # Flop 10 Ladder Score
  happy_complet_flop10 = happy_complet.groupby('year').apply(lambda x: x.nsmallest(10, 'Ladder_score')).reset_index(drop=True)
  fig_4 = px.bar(
      happy_complet_flop10,
      x='Ladder_score',
      y='Country_name',
      color_discrete_sequence=['#C30B4E'],
      animation_frame='year',
      orientation='h',
      title='Flop 10 des pays ayant le Ladder_score le plus faible par année',
      width=200,
      height=600)
  fig_4.update_layout(
      title=dict(text='Flop 10 des pays ayant le Ladder_score le plus faible par année', font=dict(size=24), x=0.5, xanchor='center'))
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
     st.plotly_chart(fig_4, use_container_width=True)

  # Top/Flop 10 Ladder Score par région
  happy_complet['Top_Flop'] = np.where(happy_complet['Ladder_score'] >= happy_complet['Ladder_score'].quantile(0.9), 'Top', 'Flop')
  fig_5 = px.bar(
      happy_complet,
      x='Regional_indicator',
      animation_frame='year',
      color='Top_Flop',
      color_discrete_map={'Top': '#6B9998', 'Flop': '#C30B4E'},
      title='Top/Flop 10 des pays par région en fonction du Ladder Score',
      width=200,
      height=600)
  fig_5.update_layout(
      title=dict(text='Top/Flop 10 des pays par région en fonction du Ladder Score', font=dict(size=24), x=0.5, xanchor='center'))
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
     st.plotly_chart(fig_5, use_container_width=True)

  show_text_3 = st.checkbox("### Interpretation des classements")
  # Afficher le texte en fonction de l'état de la case à cocher
  if show_text_3:
      st.write("Ces graphiques nous permettent de visualiser rapidement les tops et flop au cours des années et de placer géographiquement les résultats obtenus")

  st.write("")
  st.write("")
  st.write("###### Dans un deuxième temps, nous désirons analyser la corrélation entre nos valeurs.")

# Corrélation des données
  corr = happy_complet.select_dtypes('number').corr()
  fig_6 = create_fig("Corrélation des données de happy_complet", color=[0.5, 'red'], height=600, width=800)
  heatmap = go.Heatmap(
      z=corr.values,
      x=corr.columns,
      y=corr.columns,
      colorscale='pinkyl',
      showscale=True,
      zmin=-1,
      zmax=1,
      text=corr.values,
      texttemplate="%{text:.2f}",
      hoverinfo='z')
  fig_6.add_trace(heatmap)
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
     st.plotly_chart(fig_6, use_container_width=True)

  show_text_4 = st.checkbox("### Interpretation de la heatmap")
  # Afficher le texte en fonction de l'état de la case à cocher
  if show_text_4:
      st.write("Les 2 variables les plus corrélées avec le ladder score sont :")
      st.write("• Logged GPD per capita : coefficient de corrélation = 0.6")
      st.write("• Plus le Logged GPD per capita est élevé, et plus le ladder score semble élevé.")
      st.write("• Social Support : coefficient de corrélation = 0.63")
      st.write("• Plus le Social Support est élevé, et plus le ladder score semble élevé.")

  # Scatterplot entre Healthy_life_expectancy et GDP
  fig_7 = create_fig("Scatter plot du PIB par habitant et de l'espérance de vie en bonne santé", color='blue', height=800, width=1000)
  scatter = go.Scatter(
      x=happy_complet['Logged_GDP_per_capita'],
      y=happy_complet['Healthy_life_expectancy'],
      mode='markers',
      marker=dict(color='#779B52', size=10, line=dict(width=2, color='white')),
      text=happy_complet.index,
      hoverinfo='text')
  fig_7.add_trace(scatter)
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
     st.plotly_chart(fig_7, use_container_width=True)

  show_text_5 = st.checkbox("### Interpretation")
  # Afficher le texte en fonction de l'état de la case à cocher
  if show_text_5:
      st.write("Nous observons également un forte corrélation entre les variables explicatives suivantes :")
      st.write("Logged GPD per capita et Healthy Life Expectancy : coefficient de corrélation = 0.55")
      st.write("Nous avons donc fait le graphique représentant cette corrélation")
      st.write("La correlation et confirmée.")

  # Données pour chaque subplot
  data_years = happy_complet.groupby('year')['Ladder_score'].mean().reset_index()
  data_gdp = happy_complet.groupby('Logged_GDP_per_capita')['Ladder_score'].mean().reset_index()
  data_social = happy_complet.groupby('Social_support')['Ladder_score'].mean().reset_index()
  data_life_expectancy = happy_complet.copy()
  data_life_expectancy['Healthy_life_expectancy_bins'] = pd.cut(data_life_expectancy['Healthy_life_expectancy'], bins=[30, 40, 50, 60, 70, 80])
  data_freedom = happy_complet.copy()
  data_freedom['Freedom_bins'] = pd.cut(data_freedom['Freedom_to_make_life_choices'], bins=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
  data_generosity = happy_complet.groupby('Generosity')['Ladder_score'].mean().reset_index()

    # Création de la figure principale
  fig_8 = make_subplots(rows=2, cols=3, subplot_titles=(
      "Years vs Ladder score",
      "GDP vs Ladder score",
      "Social support vs Ladder score",
      "Healthy life expectancy vs Ladder score",
      "Freedom vs Ladder score",
      "Generosity vs Ladder score"))

  # Subplot 1: Years vs Ladder score
  fig_8.add_trace(go.Bar(x=data_years['year'], y=data_years['Ladder_score']), row=1, col=1)

  # Subplot 2: GDP vs Ladder score
  fig_8.add_trace(go.Scatter(x=data_gdp['Logged_GDP_per_capita'], y=data_gdp['Ladder_score'], mode='markers'), row=1, col=2)

  # Subplot 3: Social support vs Ladder score
  fig_8.add_trace(go.Scatter(x=data_social['Social_support'], y=data_social['Ladder_score'], mode='markers'), row=1, col=3)

  # Subplot 4: Healthy life expectancy vs Ladder score
  fig_8.add_trace(go.Bar(x=data_life_expectancy['Healthy_life_expectancy_bins'].astype(str), y=data_life_expectancy['Ladder_score']), row=2, col=1)

  # Subplot 5: Freedom vs Ladder score
  fig_8.add_trace(go.Bar(x=data_freedom['Freedom_bins'].astype(str), y=data_freedom['Ladder_score']), row=2, col=2)

  # Subplot 6: Generosity vs Ladder score
  fig_8.add_trace(go.Scatter(x=data_generosity['Generosity'], y=data_generosity['Ladder_score'], mode='markers'), row=2, col=3)

  # Mise à jour de la mise en page
  fig_8.update_layout(
      title=dict(text="Analyse de la corrélation des variables en fonction avec le ladder score", font=dict(size=24), x=0.5, xanchor='center'),
      height=1000,
      width=2000,
      showlegend=False,
      margin=dict(l=50, r=50, t=100, b=50))

  # Affichage avec Streamlit
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
     st.plotly_chart(fig_8, use_container_width=True)

  show_text_6 = st.checkbox("### Interpretation du subplot")
    # Afficher le texte en fonction de l'état de la case à cocher
  if show_text_6:
      st.write("La représentation de la corrélation de chaque variable avec le ladder score confirme l'analyse de la heatmap :")
      st.write("- Les variables les moins corrélés sont : l'année, la générosité et la corruption.")
      st.write("- Les variables les plus corrélées sont : Logged GPD per capita et Social Suppor")
      st.write("- Les variables les plus mitigées restent : Healthy Life Expectancy, Freedom to make a life choice")

if page == pages[2] : 
  st.markdown('''<div style="text-align:center; margin-top: 20px;style="color:#003885; font-size:20px; font-weight:bold;">Nous pouvons passer à la modélisation</div>''', unsafe_allow_html=True)

#étape1
#on récupère les fichiers
X_train=pd.read_csv('X_train.csv')
X_test=pd.read_csv('X_test.csv')
y_train=pd.read_csv('y_train.csv')
y_test=pd.read_csv('y_test.csv')  

#on sélectionne les variables
X_train=X_train.drop(['Country_name','Regional_indicator'],axis=1)
X_test=X_test.drop(['Country_name','Regional_indicator'],axis=1)

#on standardise
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#on récupère les fichiers entrainés
lrrecup = joblib.load('model_lin_reg.joblib')
# rdcrecup = joblib.load('model_reg_dec_tree.joblib')
rrfrecup = joblib.load('model_reg_random_forest.joblib')
rgbrecup = joblib.load('model_reg_gradient_boost.joblib')

  
# Calcul des métriques pour regression linéaire
y_pred_Test_lin_reg = lrrecup.predict(X_test_scaled)
y_pred_Train_lin_reg = lrrecup.predict(X_train_scaled)
# jeu d'entraînement
mae_lin_reg_train = mean_absolute_error(y_train,y_pred_Train_lin_reg)
mse_lin_reg_train = mean_squared_error(y_train,y_pred_Train_lin_reg,squared=True)
rmse_lin_reg_train = mean_squared_error(y_train,y_pred_Train_lin_reg,squared=False)
# jeu de test
mae_lin_reg_test = mean_absolute_error(y_test,y_pred_Test_lin_reg)
mse_lin_reg_test = mean_squared_error(y_test,y_pred_Test_lin_reg,squared=True)
rmse_lin_reg_test = mean_squared_error(y_test,y_pred_Test_lin_reg,squared=False)

# # Calcul des métriques pour l'arbre de décision
# y_pred_decision_tree = rdcrecup.predict(X_test_scaled)
# y_pred_train_decision_tree = rdcrecup.predict(X_train_scaled)

# # jeu d'entraînement
# mae_decision_tree_train = mean_absolute_error(y_train,y_pred_train_decision_tree)
# mse_decision_tree_train = mean_squared_error(y_train,y_pred_train_decision_tree,squared=True)
# rmse_decision_tree_train = mean_squared_error(y_train,y_pred_train_decision_tree,squared=False)
# # jeu de test
# mae_decision_tree_test = mean_absolute_error(y_test,y_pred_decision_tree)
# mse_decision_tree_test = mean_squared_error(y_test,y_pred_decision_tree,squared=True)
# rmse_decision_tree_test = mean_squared_error(y_test,y_pred_decision_tree,squared=False)

# Calcul des métriques pour random forest
y_pred_random_forest = rrfrecup.predict(X_test_scaled)
y_pred_random_forest_train = rrfrecup.predict(X_train_scaled)
# jeu d'entraînement
mae_random_forest_train = mean_absolute_error(y_train,y_pred_random_forest_train)
mse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=True)
rmse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=False)
# jeu de test
mae_random_forest_test = mean_absolute_error(y_test,y_pred_random_forest)
mse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=True)
rmse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=False)

# Calcul des métriques pour GradientBoost
y_pred_gradientBoost = rgbrecup.predict(X_test_scaled)
y_pred_gradientBoost_train = rgbrecup.predict(X_train_scaled)
# jeu d'entraînement
mae_gradientBoost_train = mean_absolute_error(y_train,y_pred_gradientBoost_train)
mse_gradientBoost_train = mean_squared_error(y_train,y_pred_gradientBoost_train,squared=True)
rmse_gradientBoost_train = mean_squared_error(y_train,y_pred_gradientBoost_train,squared=False)
# jeu de test
mae_gradientBoost_test = mean_absolute_error(y_test,y_pred_gradientBoost)
mse_gradientBoost_test = mean_squared_error(y_test,y_pred_gradientBoost,squared=True)
rmse_gradientBoost_test = mean_squared_error(y_test,y_pred_gradientBoost,squared=False)

# Creation d'un dataframe pour comparer les metriques des deux algorithmes
data_2 = {'MAE train': [mae_lin_reg_train,mae_decision_tree_train, mae_random_forest_train,mae_gradientBoost_train],
         'MAE test': [mae_lin_reg_test,mae_decision_tree_test, mae_random_forest_test,mae_gradientBoost_test],
         'MSE train': [mse_lin_reg_train,mse_decision_tree_train,mse_random_forest_train,mse_gradientBoost_train],
         'MSE test': [mse_lin_reg_test,mse_decision_tree_test,mse_random_forest_test,mse_gradientBoost_test],
         'RMSE train': [rmse_lin_reg_train,rmse_decision_tree_train, rmse_random_forest_train,rmse_gradientBoost_train],
         'RMSE test': [rmse_lin_reg_test,rmse_decision_tree_test, rmse_random_forest_test,rmse_gradientBoost_test]}

  # Creer DataFrame
metriques = pd.DataFrame(data_2, index = ['Lin Reg','Decision Tree', 'Random Forest ','GradientBoost'])

if page==pages[3] :
  st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">🤖 La modélisation 🤖</p>', unsafe_allow_html=True)
  st.write("")
  st.write("---")
  st.markdown("""
    ##### Nous avons testé quatre modèles :
      - La régression Linéaire
      - Le décision tree 
      - Le random Forest
      - Le Gradient boost""")
  st.write("")
  st.markdown('<p style="color:#006A89; font-size:22px;font-weight:bold;text-align:left">La Régression linéaire</p>', unsafe_allow_html=True)
  st.write("Les scores sont :")
  st.dataframe(metriques)
  
# if page==pages[3] :
#    st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Modélisation</p>', unsafe_allow_html=True)
#    st.write('--------------------Nuage de points entre le ladder score réel et le ladder score prédit---------------------\n')

# fig = plt.figure(figsize = (10,10))
# plt.scatter(y_pred_test_LR,y_test, c='blue')
# plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'black')
# plt.xlabel("Prediction")
# plt.ylabel("Vrai valeur")
# plt.title('Régression Linéaire pour la prédiction du ladder score')
# plt.show()
# print("\n On observe que le Ladder score est assez bien prédit")
# st.write("La courbe de régression ests :")

if page == pages[4] : 
  st.markdown('<p style="color:#006A89; font-size:30px;font-weight:bold;text-align:center">Conclusion</p>', unsafe_allow_html=True)
  
