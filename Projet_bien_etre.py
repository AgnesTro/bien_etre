# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import plotly.graph_objects as go
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# import joblib
# import pickle
# import plotly.express as px
# import altair as alt
# import streamlit as st
# from streamlit_navigation_bar import st_navbar
# from plotly.subplots import make_subplots
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir,"world-happiness-report.csv")
happy = pd.read_csv(file_path)

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir,"world-happiness-report-2021.csv")
happy_2021 = pd.read_csv(file_path)
st.set_page_config(
  page_title="Projet bien-etre / Juin 2024",
  page_icon="🌍",
  layout="wide")
page = st_navbar(["Introduction 🤓","Exploration 🔎", "Data Visualisation 📊", "Modélisation 🤖"])
pages = ["Introduction 🤓","Exploration 🔎", "Data Visualisation 📊", "Modélisation 🤖"]
alt.themes.enable("dark")

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
    st.dataframe(happy_complet.duplicated().sum()) #A corriger
    st.write("Il n'y a pas de doublon sur happy_complet")
  if st.checkbox("Afficher les NA de Happy_complet") :
    st.dataframe(happy_complet.isna().sum())
    st.write("Il aura un traitement de NA à faire sur happy_complet")

if page == pages[1] : 
  st.markdown('''<div style="color:#006A89; font-size:24 px;font-weight:bold;">Traitement des NAN :</div>''', unsafe_allow_html=True)
  st.write("Dans le souci de perfomance du projet, nous avons décidé de ne pas avoir de fuite de données. Nos NAN seront donc traités avec train_test,split")
  st.write("Nous avons toutefois éliminé les lignes avec plus de 3 NAN")
happy_complet=happy_complet.loc[(happy_complet.isna().sum(axis=1))< 3]

if page == pages[1] : 
  st.markdown('''
              <div style="text-align:center; margin-top: 20px;">
              <a href="/Data Visualisation 📊" style="color:#003885; font-size:20px; font-weight:bold;">Nous pouvons passer à la visualisation des données</a>
</div>
''', unsafe_allow_html=True)

if page == pages[2]: 
  st.write("#### Dans un premier temps, nous visualiserons nos valeurs.")
  # Fonction de création de figure avec titres centrés et taille de police ajustée
  def create_fig(title, color, height, width):
      fig = go.Figure()
      fig.update_layout(
          title=dict(text=title, font=dict(size=24), x=0.5, xanchor='center'),
          plot_bgcolor='white',
          paper_bgcolor='white',
          height=height,
          width=width,
          margin=dict(l=50, r=50, t=100, b=50),
          xaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridcolor='lightgray'),
          yaxis=dict(showline=True, linewidth=2, linecolor='black', showgrid=True, gridcolor='lightgray'))
      return fig
  
  # Graphique Boxplot
  fig_1 = create_fig("Distribution du score de bonheur en fonction des zones géographiques",
      color='blue',
      height=600,
      width=600)
  fig_1.add_trace(go.Box(y=happy_complet["Ladder_score"], x=happy_complet["Regional_indicator"], fillcolor='moccasin'))
  st.plotly_chart(fig_1, use_container_width=True)
    
  st.markdown("""
    <h3 style='text-align: left; font-weight: bold; font-size : 12 px'>Explications</h3>""",unsafe_allow_html=True)
  show_text = st.checkbox("Explication du box plot")
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
      title='Life Ladder per country over the years',
      animation_frame='year',
      color_continuous_scale=px.colors.sequential.Plasma,
      width=1500,
      height=1000)
  fig_2.update_layout(
      title=dict(text='Life Ladder per country over the years', font=dict(size=24), x=0.5, xanchor='center'),
      margin=dict(l=50, r=50, t=100, b=50),
      geo=dict(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray"))
  st.plotly_chart(fig_2, use_container_width=True)

  st.markdown("""
      <h3 style='text-align: left; font-weight: bold; font-size : 12 px'>Explications</h3>""",unsafe_allow_html=True)
  show_text_2 = st.checkbox("Explication de la carte")
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
      animation_frame='year',
      orientation='h',
      title='Top 10 des pays ayant le Ladder_score le plus élevé par année',
      width=600,
      height=600)
  fig_3.update_layout(title=dict(text='Top 10 des pays ayant le Ladder_score le plus élevé par année', font=dict(size=24), x=0.5, xanchor='center'))
  st.plotly_chart(fig_3, use_container_width=True)

  # Flop 10 Ladder Score
  happy_complet_flop10 = happy_complet.groupby('year').apply(lambda x: x.nsmallest(10, 'Ladder_score')).reset_index(drop=True)
  fig_4 = px.bar(
      happy_complet_flop10,
      x='Ladder_score',
      y='Country_name',
      animation_frame='year',
      orientation='h',
      title='Flop 10 des pays ayant le Ladder_score le plus faible par année',
      width=700,
      height=600)
  fig_4.update_layout(
      title=dict(text='Flop 10 des pays ayant le Ladder_score le plus faible par année', font=dict(size=24), x=0.5, xanchor='center'))
  st.plotly_chart(fig_4, use_container_width=True)

  # Top/Flop 10 Ladder Score par région
  happy_complet['Top_Flop'] = np.where(happy_complet['Ladder_score'] >= happy_complet['Ladder_score'].quantile(0.9), 'Top', 'Flop')
  fig_5 = px.bar(
      happy_complet,
      x='Regional_indicator',
      animation_frame='year',
      color='Top_Flop',
      color_discrete_map={'Top': 'lightblue', 'Flop': 'lightcoral'},
      title='Top/Flop 10 des pays par région en fonction du Ladder Score',
      width=700,
      height=600)
  fig_5.update_layout(
      title=dict(text='Top/Flop 10 des pays par région en fonction du Ladder Score', font=dict(size=24), x=0.5, xanchor='center'))
  st.plotly_chart(fig_5, use_container_width=True)

  show_text_3 = st.checkbox("### Explication des classements")
  # Afficher le texte en fonction de l'état de la case à cocher
  if show_text_3:
      st.write("Ces graphiques nous permettent de visualiser rapidement les tops et flop au cours des années et de placer géographiquement les résultats obtenus")

  st.write("#### Dans un deuxième temps, nous désirons analyser la corrélation entre nos valeurs.")
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
  st.plotly_chart(fig_6, use_container_width=True)

  st.markdown("""<h3 style='text-align: left; font-weight: bold; font-size : 12 px'>Explications</h3>""",unsafe_allow_html=True)  
  show_text_4 = st.checkbox("### Explication de la heatmap")
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
      marker=dict(color='light blue', size=10, line=dict(width=2, color='white')),
      text=happy_complet.index,
      hoverinfo='text')
  fig_7.add_trace(scatter)
  st.plotly_chart(fig_7, use_container_width=True)

  show_text_5 = st.checkbox("### Explication")
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
  st.plotly_chart(fig_8, use_container_width=True)

  show_text_6 = st.checkbox("### Explications du subplot")
    # Afficher le texte en fonction de l'état de la case à cocher
  if show_text_6:
      st.write("La représentation de la corrélation de chaque variable avec le ladder score confirme l'analyse de la heatmap :")
      st.write("- Les variables les moins corrélés sont : l'année, la générosité et la corruption.")
      st.write("- Les variables les plus corrélées sont : Logged GPD per capita et Social Suppor")
      st.write("- Les variables les plus mitigées restent : Healthy Life Expectancy, Freedom to make a life choice")

if page == pages[2] : 
  st.markdown('''
              <div style="text-align:center; margin-top: 20px;">
              <a href="/Modélisation 🤖 " style="color:#003885; font-size:20px; font-weight:bold;">Nous pouvons passer à la modélisation</a>
</div>''', unsafe_allow_html=True)

happy_model = happy_complet
# On sépare les données, la cible étant Ladder_score
feats = happy_model.drop('Ladder_score',axis =1)
target = happy_model['Ladder_score']

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state = 42)

# # # On commencer par gérer les NANS pour le jeu de données X_train

nan_cor = X_train[['Country_name','Perceptions_of_corruption']].groupby(['Country_name']).mean().reset_index()
countries_cor =nan_cor.Country_name
for country in countries_cor :
    countries = X_train.Country_name == country
    mean_value = X_train.Perceptions_of_corruption.loc[countries].mean()
    X_train.Perceptions_of_corruption.loc[countries] = X_train.Perceptions_of_corruption.loc[countries].fillna(mean_value)
nan_cor_2 = X_train[['Regional_indicator','Perceptions_of_corruption']].groupby(['Regional_indicator']).mean().reset_index()
regional_cor =nan_cor_2.Regional_indicator
for region in regional_cor :
    region = X_train.Regional_indicator == region
    mean_value = X_train.Perceptions_of_corruption.loc[region].mean()
    X_train.Perceptions_of_corruption.loc[region] = X_train.Perceptions_of_corruption.loc[region].fillna(mean_value)

        # Etape GDP :
nan_GDP=X_train.loc[X_train['Logged_GDP_per_capita'].isna()]

nan_gdp_2 = X_train[['Country_name','Logged_GDP_per_capita']].groupby(['Country_name']).mean().reset_index()

countries_gdp = nan_gdp_2.Country_name

for country in countries_gdp :
    countries = X_train.Country_name == country
    mean_value = X_train.Logged_GDP_per_capita.loc[countries].mean()
    X_train.Logged_GDP_per_capita.loc[countries] = X_train.Logged_GDP_per_capita.loc[countries].fillna(mean_value)

 # Etape freedom :
nan_free=X_train.loc[X_train['Freedom_to_make_life_choices'].isna()]

nan_free_2 = X_train[['Country_name','Freedom_to_make_life_choices']].groupby(['Country_name']).mean().reset_index()

countries_free =nan_free_2.Country_name

for country in countries_free :
    countries = X_train.Country_name == country
    mean_value = X_train.Freedom_to_make_life_choices.loc[countries].mean()
    X_train.Freedom_to_make_life_choices.loc[countries] = X_train.Freedom_to_make_life_choices.loc[countries].fillna(mean_value)

      # Etape social support :
nan_soc=X_train.loc[X_train['Social_support'].isna()]

nan_soc_2 = X_train[['Country_name','Social_support']].groupby(['Country_name']).mean().reset_index()

countries_soc =nan_soc_2.Country_name

for country in countries_soc :
    countries = X_train.Country_name == country
    mean_value = X_train.Social_support.loc[countries].mean()
    X_train.Social_support.loc[countries] = X_train.Social_support.loc[countries].fillna(mean_value)

      # Generosity --> Remplacer par la moyenne

nan_gen=X_train.loc[X_train['Generosity'].isna()]

nan_gen_2 = X_train[['Country_name','Social_support']].groupby(['Country_name']).mean().reset_index()

countries_gen =nan_gen_2.Country_name

for country in countries_gen :
    countries = X_train.Country_name == country
    mean_value = X_train.Generosity.loc[countries].mean()
    X_train.Generosity.loc[countries] = X_train.Generosity.loc[countries].fillna(mean_value)

    # Healthy_life_expectancy

# Recherche des NAN pour Healthy_life_expectancy
NAN_life = X_train.loc[X_train["Healthy_life_expectancy"].isna()]

data_to_fill_HK = [
    (2006, 82.38),
    (2008, 82.34),
    (2009, 82.78),
    (2010, 82.96),
    (2011,83.41),
    (2012,83.45),
    (2014,83.94),
    (2016,84.21),
    (2017,84.70),
    (2019,85.16)]

country_name = 'Hong Kong S.A.R. of China'
for year, value in data_to_fill_HK:
  X_train.Healthy_life_expectancy.loc[(X_train.Country_name == country_name) & (X_train.year == year)] = X_train.Healthy_life_expectancy.loc[(X_train.Country_name == country_name) & (X_train.year == year)].fillna(value)

data_to_fill_Kosovo = [
    (2007,69.20),
    (2008,69.35),
    (2009,69.65),
    (2010,69.90),
    (2011,70.15),
    (2012,70.50),
    (2013,70.80),
    (2014,71.10),
    (2015,71.35),
    (2016,71.65),
    (2017,71.95),
    (2018,72.20),
    (2019,72.20)]

# Pas de valeurs trouvée pour 2019; remplacer par 2018

country_name = 'Kosovo'
for year, value in data_to_fill_Kosovo:
  X_train.Healthy_life_expectancy.loc[(X_train.Country_name == country_name) & (X_train.year == year)] = X_train.Healthy_life_expectancy.loc[(X_train.Country_name == country_name) & (X_train.year == year)].fillna(value)

data_to_fill_Palestine = [
    (2011,73.24),
    (2012,73.47),
    (2013,74.03),
    (2014,72.62),
    (2015,74.41),
    (2016,74.55),
    (2017,74.83)]

country_name = 'Palestinian Territories'
for year, value in data_to_fill_Palestine:
  X_train.Healthy_life_expectancy.loc[(X_train.Country_name == country_name) & (X_train.year == year)] = X_train.Healthy_life_expectancy.loc[(X_train.Country_name == country_name) & (X_train.year == year)].fillna(value)

data_to_fill_Taiwan = [
    (2011,78.3),
    (2012,78.3),
    (2013,79),
    (2014,79.5),
    (2015,80.20),
    (2016,80.5),
    (2017,80.5)]

country_name = 'Taiwan Province of China'
for year, value in data_to_fill_Taiwan:
  X_train.Healthy_life_expectancy.loc[(X_train.Country_name == country_name) & (X_train.year == year)] = X_train.Healthy_life_expectancy.loc[(X_train.Country_name == country_name) & (X_train.year == year)].fillna(value)

# On recommence le meme processus avec X_test

# # CORRUPTION :

 # Remplacement des Nan par la moyenne
# Nous allons utiliser la moyenne des autres années pour le même pays ou la moyenne de la région pour les pays sans données.

nan_cor_2 = X_test[['Country_name','Perceptions_of_corruption']].groupby(['Country_name']).mean().reset_index()

countries_cor =nan_cor_2.Country_name

for country in countries_cor :
    countries = X_test.Country_name == country
    mean_value = X_test.Perceptions_of_corruption.loc[countries].mean()
    X_test.Perceptions_of_corruption.loc[countries] = X_test.Perceptions_of_corruption.loc[countries].fillna(mean_value)

nan_cor_2 = X_test[['Regional_indicator','Perceptions_of_corruption']].groupby(['Regional_indicator']).mean().reset_index()

regional_cor =nan_cor_2.Regional_indicator

for region in regional_cor :
    region = X_test.Regional_indicator == region
    mean_value = X_test.Perceptions_of_corruption.loc[region].mean()
    X_test.Perceptions_of_corruption.loc[region] = X_test.Perceptions_of_corruption.loc[region].fillna(mean_value)

 # Etape GDP :
nan_GDP=X_test.loc[X_test['Logged_GDP_per_capita'].isna()]

nan_gdp_2 = X_test[['Country_name','Logged_GDP_per_capita']].groupby(['Country_name']).mean().reset_index()

countries_gdp = nan_gdp_2.Country_name

for country in countries_gdp :
    countries = X_test.Country_name == country
    mean_value = X_test.Logged_GDP_per_capita.loc[countries].mean()
    X_test.Logged_GDP_per_capita.loc[countries] = X_test.Logged_GDP_per_capita.loc[countries].fillna(mean_value)


# Etape freedom :
nan_free=X_test.loc[X_test['Freedom_to_make_life_choices'].isna()]

nan_free_2 = X_test[['Country_name','Freedom_to_make_life_choices']].groupby(['Country_name']).mean().reset_index()

countries_free =nan_free_2.Country_name

for country in countries_free :
    countries = X_test.Country_name == country
    mean_value = X_test.Freedom_to_make_life_choices.loc[countries].mean()
    X_test.Freedom_to_make_life_choices.loc[countries] = X_test.Freedom_to_make_life_choices.loc[countries].fillna(mean_value)

 # Etape social support :
nan_soc=X_test.loc[X_test['Social_support'].isna()]

nan_soc_2 = X_test[['Country_name','Social_support']].groupby(['Country_name']).mean().reset_index()

countries_soc =nan_soc_2.Country_name

for country in countries_soc :
    countries = X_test.Country_name == country
    mean_value = X_test.Social_support.loc[countries].mean()
    X_test.Social_support.loc[countries] = X_test.Social_support.loc[countries].fillna(mean_value)

data_to_fill_Maroc = [
    (2008,0.621),
    (2010,0.621),
    (2012,0.631)]

country_name = 'Morocco'
for year, value in data_to_fill_Maroc:
  X_test.Social_support.loc[(X_test.Country_name == country_name) & (X_test.year == year)] = X_test.Social_support.loc[(X_test.Country_name == country_name) & (X_test.year == year)].fillna(value)

# Generosity --> Remplacer par la moyenne
nan_gen=X_test.loc[X_test['Generosity'].isna()]
nan_gen_2 = X_test[['Country_name','Social_support']].groupby(['Country_name']).mean().reset_index()

countries_gen =nan_gen_2.Country_name

for country in countries_gen :
    countries = X_test.Country_name == country
    mean_value = X_test.Generosity.loc[countries].mean()
    X_test.Generosity.loc[countries] = X_test.Generosity.loc[countries].fillna(mean_value)

 # Healthy_life_expectancy

# Recherche des NAN pour Healthy_life_expectancy
NAN_life = X_test.loc[X_test["Healthy_life_expectancy"].isna()]

data_to_fill_HK = [
    (2006, 82.38),
    (2008, 82.34),
    (2009, 82.78),
    (2010, 82.96),
    (2011,83.41),
    (2012,83.45),
    (2014,83.94),
    (2016,84.21),
    (2017,84.70),
    (2019,85.16)]

country_name = 'Hong Kong S.A.R. of China'
for year, value in data_to_fill_HK:
  X_test.Healthy_life_expectancy.loc[(X_test.Country_name == country_name) & (X_test.year == year)] = X_test.Healthy_life_expectancy.loc[(X_test.Country_name == country_name) & (X_test.year == year)].fillna(value)

data_to_fill_Kosovo = [
    (2007,69.20),
    (2008,69.35),
    (2009,69.65),
    (2010,69.90),
    (2011,70.15),
    (2012,70.50),
    (2013,70.80),
    (2014,71.10),
    (2015,71.35),
    (2016,71.65),
    (2017,71.95),
    (2018,72.20),
    (2019,72.20)]

# Pas de valeurs trouvée pour 2019; remplacer par 2018

country_name = 'Kosovo'
for year, value in data_to_fill_Kosovo:
  X_test.Healthy_life_expectancy.loc[(X_test.Country_name == country_name) & (X_test.year == year)] = X_test.Healthy_life_expectancy.loc[(X_test.Country_name == country_name) & (X_test.year == year)].fillna(value)

data_to_fill_Palestine = [
    (2011,73.24),
    (2012,73.47),
    (2013,74.03),
    (2014,72.62),
    (2015,74.41),
    (2016,74.55),
    (2017,74.83)]

country_name = 'Palestinian Territories'
for year, value in data_to_fill_Palestine:
  X_test.Healthy_life_expectancy.loc[(X_test.Country_name == country_name) & (X_test.year == year)] = X_test.Healthy_life_expectancy.loc[(X_test.Country_name == country_name) & (X_test.year == year)].fillna(value)

data_to_fill_Taiwan = [
    (2011,78.3),
    (2012,78.3),
    (2013,79),
    (2014,79.5),
     (2015,80.20),
      (2016,80.5),
    (2017,80.5)]

country_name = 'Taiwan Province of China'
for year, value in data_to_fill_Taiwan:
  X_test.Healthy_life_expectancy.loc[(X_test.Country_name == country_name) & (X_test.year == year)] = X_test.Healthy_life_expectancy.loc[(X_test.Country_name == country_name) & (X_test.year == year)].fillna(value)

X_train=X_train.drop(['Country_name','Regional_indicator'],axis=1)
X_test=X_test.drop(['Country_name','Regional_indicator'],axis=1)

# standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

