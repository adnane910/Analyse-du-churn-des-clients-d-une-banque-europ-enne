import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import plotly.express as px
import matplotlib.pyplot as plt


# Configuration page
# -------------------------------
st.set_page_config(
    page_title="🏦 Analyse du churn des clients d'une banque européenne 🏦",
    layout="wide"
)

st.markdown("""
    <div style="
        background: linear-gradient(90deg, #2E86C1, #5DADE2);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 32px;
        font-weight: bold;
    ">
        🏦 Analyse du churn des clients d'une banque européenne🏦
    </div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1,6,1])

with col2:
    st.image("../churn_app_streamlit/bank_image.jpg")


st.markdown(""" 📌 Contexte :
            
            Dans un environnement bancaire de plus en plus concurrentiel, la fidélisation des clients constitue un enjeu majeur.      
            Le départ des clients (désabonnement ou churn) peut entraîner une perte significative de revenus pour la banque.

            Dans ce cadre, il est essentiel de comprendre les facteurs qui influencent le désabonnement des clients et 
            d'identifier les profils les plus à risque.                           
            Cette analyse s'appuie sur un jeu de données de 10 000 clients d'une banque européenne, 
            contenant des informations démographiques, financières et comportementales.
""")
st.markdown("""           --------------------------------------------------------------------  """)

st.markdown("""Problematique :
            
            Comment la banque peut-elle identifier les clients à risque de désabonnement afin de mettre en place 
            des stratégies efficaces de fidélisation ? """)

st.markdown("""           --------------------------------------------------------------------  """)

st.markdown(""" 
🎯 Objectifs de l'analyse

Cette étude vise à répondre aux questions suivantes :

Analyse du désabonnement (churn) :
            
            Identifier les attributs les plus fréquents chez les clients qui se désabonnent par rapport à ceux qui restent fidèles
            
            Évaluer la possibilité de prédire le taux de désabonnement à partir des variables disponibles
            
Profil des clients:
            
            Décrire le profil démographique global des clients de la banque
            
Analyse géographique :
            
            Examiner les différences de comportement entre les clients allemands, français et espagnols
            
Segmentation client :
            
            Identifier les différents segments de clients présents dans la base de données
 """)

st.markdown(""" ------------------------------------------------------------------------- """)

st.markdown("""
Ce dashboard présente : 
            
- Les KPIs clés
            
- Une analyse exploratoire des données (EDA)
            
- Les facteurs influençant le churn
            
- La segmentation des clients
            
- La prédiction du churn via Machine Learning
""")
st.markdown(""" ------------------------------------------------------------------------- """)

# -------------------------------
# Charger les données et le modèle
# -------------------------------
df_bank_clean = pd.read_csv("../Data/processed/df_bank_clean.csv")
df_ml = pd.read_csv("../Data/processed/ML_data.csv")
cols_pipeline = df_ml.drop(columns=["Exited"]).columns.tolist()

with open("../models/pipeline_churn.pkl", "rb") as f:
    pipeline = pickle.load(f)

# -------------------------------
# Sidebar filtres
# -------------------------------

st.sidebar.header("Filtres")

genre_filter = st.sidebar.multiselect("Genre", ["Femme", "Homme"], default=["Femme", "Homme"])
pays_filter = st.sidebar.multiselect("Pays", ["France","Germany","Spain"], default=["France","Germany","Spain"])
produit_filter = st.sidebar.multiselect(
    "Nombre de produits", sorted(df_bank_clean["NumOfProducts"].unique()), 
    default=sorted(df_bank_clean["NumOfProducts"].unique())
)
carte_filter = st.sidebar.multiselect(
    "Carte bancaire", [0,1], default=[0,1],
    format_func=lambda x: "Non" if x==0 else "Oui"
)
actif_filter = st.sidebar.multiselect(
    "Client actif", [0,1], default=[0,1],
    format_func=lambda x: "Non" if x==0 else "Oui"
)

# Filtrage du DataFrame
df_bank_clean["Gender_label"] = df_bank_clean["Gender"].map({0:"Femme",1:"Homme"})

df_filtered = df_bank_clean[
    (df_bank_clean["Gender_label"].isin(genre_filter)) &
    (df_bank_clean["Geography"].isin(pays_filter)) &
    (df_bank_clean["NumOfProducts"].isin(produit_filter)) &
    (df_bank_clean["HasCrCard"].isin(carte_filter)) &
    (df_bank_clean["IsActiveMember"].isin(actif_filter))
]

#----------------
# KPI
#-----------------
st.subheader("Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Nombre de clients", len(df_filtered))
col2.metric("Taux de churn", f"{df_filtered['Exited'].mean()*100:.2f} %")
col3.metric("Age moyen", f"{df_filtered['Age'].mean():.1f} ans")
col4.metric("Solde Moyen", f"{df_filtered['Balance'].mean():.0f} €")

st.markdown(""" ------------------------------------------------------------------------- """)

#####################           EDA            ##########################################


st.header("EDA (Exploratory Data Analysis)")

# Presentation Dataset 
st.subheader("DataFrame")
st.write(df_bank_clean)

# Nombre de lignes et colonnes
st.subheader("Nombre de lignes et colonnes")
df_shape = df_bank_clean.shape
st.write(df_shape)


# Valeurs manquantes
st.subheader("Valeurs manquantes")

missing_values = df_bank_clean.isna().sum().sum()
st.metric("Total valeurs manquantes", missing_values)

# Doublons
st.subheader("Lignes dupliquées")

Duplicated = df_bank_clean.duplicated().sum()
st.write(f"le nombre de valeurs dupliquées : {Duplicated}")

# statistiques descriptives
st.subheader("Statistiques descriptives")
describ = df_bank_clean.drop(columns=["CustomerId","Exited","IsActiveMember","HasCrCard"]).describe()
st.write(describ)


# -------------------------------
# Section 1 : Répartition clients
# -------------------------------
st.subheader("1 - Répartition des clients par genre")
st.markdown("Ce graphique montre la répartition des clients par genre. Les femmes sont légèrement majoritaires.")

fig1 = px.pie(df_filtered, names="Gender_label", 
              width=400, height=400,
              color_discrete_sequence=["#121778", "#59B6F9"])
fig1.update_traces(textinfo="percent+label")
st.plotly_chart(fig1)


st.subheader("2 - Répartition des clients par pays")
st.markdown("Nombre de clients par pays, la France represente la moitié des clients de la Banque")

count_geo = df_filtered["Geography"].value_counts()
df_geo = pd.DataFrame({"Pays": count_geo.index, "Nombre de clients": count_geo.values})

fig2 = px.bar(
    df_geo, x="Pays", y="Nombre de clients", text="Nombre de clients",
    color="Nombre de clients", color_continuous_scale="Blues",
    labels={"Pays":"Pays","Nombre de clients":"Nombre de clients"}
)
fig2.update_traces(textposition="outside")
st.plotly_chart(fig2)


st.subheader("3 - Clients Fidèles vs désabonnés")
st.markdown("20 %  est la Proportion de clients désabonnés")
clients_churn = df_filtered["Exited"].value_counts(normalize=True).sort_index()*100
df_churn = pd.DataFrame({"Statut":["Fidèles","Désabonnés"], "Pourcentage":clients_churn.values})

fig3 = px.pie(
    df_churn,
    names="Statut",
    values="Pourcentage",
    hole=0.5,  # camembert creux
    color="Statut",
    color_discrete_sequence=["#121778", "#59B6F9"]
)
fig3.update_traces(textposition="inside")
st.plotly_chart(fig3)


# -------------------------------
# Section 3 : Distribution numériques
# -------------------------------
st.subheader("4 - 📊 Boxplot des variables numériques")
st.markdown("Distribution de l'âge, du salaire estimé et du solde des clients.")

fig, axes = plt.subplots(1,3,figsize=(16,8))
sns.boxplot(y=df_filtered["Age"], ax=axes[0]); axes[0].set_title("Âge")
sns.boxplot(y=df_filtered["EstimatedSalary"], ax=axes[1]); axes[1].set_title("Salaire")
sns.boxplot(y=df_filtered["Balance"], ax=axes[2]); axes[2].set_title("Balance")
st.pyplot(fig)

st.markdown("""

calcule de la Borne supérieure IQR pour les outliers :
            
            Q1 = 32  
            Q3 = 44 
            IQR = Q3 - Q1 ,   
            IQR = 44 - 32  ,
            IQR = 12
            Q3 + 1.5 * IQR
            Up = 44 + 1.5 * 12
            Up = 62

            # Nb de clients qui ont un Age superieur a 62 ans
            df_Bank_churn[df_Bank_churn["Age"] > 62].shape[0]

            = 359

            # calcule de Taux de churn des client plus agé de plus de 62 ans 
            df_Bank_churn[df_Bank_churn["Age"] > 62]["Exited"].mean()*100

            = 20.33 %

Bien que l'IQR ait identifié les clients de plus de 62 ans comme outliers statistiques, 
leur taux de churn (20,33 %) est quasiment identique au taux global (20,36 %). 
Ces observations ne présentent donc pas de comportement particulier et ont été conservées dans l'analyse. """)


# -------------------------------
# Section 4 : Corrélations
# -------------------------------
st.subheader("5 - Heatmap de Corrélations")
st.markdown("Corrélations entre variables numériques, et variable cible - Exited -")
numeric_df = df_bank_clean.select_dtypes(include=["int64","float64"]).drop(columns=["CustomerId"], errors='ignore')
corr = numeric_df.corr()
fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="blues")
fig_corr.update_layout(width=1000, height=600)
st.plotly_chart(fig_corr)




# Analyse heetmap de corr

st.markdown("""
Corrélation avec la variable cible Exited:
            
Age : corrélation = 0,29  Corrélation positive modérée.
Cela signifie que les clients plus âgés ont tendance à quitter davantage la banque.
            
👉 Toutefois, cette relation reste modérée et ne signifie pas que l'âge est une cause directe du churn.

Balance : corrélation = 0,12
Corrélation positive très faible.
            
👉 Le solde du compte influence très peu le départ des clients.
            
Geography_Germany : corrélation = 0,17
Corrélation positive faible.
            
👉 Les clients situés en Allemagne sont légèrement plus susceptibles de quitter la banque.
            
Geography_France : corrélation = -0,10
Corrélation négative faible.
            
👉 Les clients français ont tendance à être légèrement plus fidèles.
            
Gender : corrélation = -0,10
Corrélation négative faible.
            
👉 Le genre a un impact très limité sur le churn.
            
Corrélations entre variables explicatives :
            
Geography_Germany & Balance : corrélation = 0,40
Corrélation positive modérée.
            
👉 Les clients allemands ont tendance à avoir un solde plus élevé que la moyenne.
            
Geography_France & Balance : corrélation = -0,23
Corrélation négative faible à modérée.
            
👉 Les clients français ont tendance à avoir un solde inférieur à la moyenne.
""")
# -------------------------------
# Section 2 : Churn
# -------------------------------

# Distribution d'âge
st.subheader("Churn des clients en fonction de l'âge")

# Filtrer les clients désabonnés
df_churn = df_filtered[df_filtered["Exited"] == 1]

age_dist = (df_churn["Category_Age"].value_counts(normalize=True).mul(100).reset_index())
age_dist.columns = ["Category_Age", "percentage"]

fig = px.bar(
    age_dist,
    x="Category_Age",
    y="percentage",
    color="percentage",
    color_continuous_scale="Blues",
    labels={
        "percentage": "Pourcentage (%)",
        "Category_Age": "Catégorie d'âge"
    }
)
st.plotly_chart(fig)
st.markdown("""
Interprétation business:

Ce segment est en forte évolution, ce qui influence son comportement :

1- Changements de vie fréquents
            
- réévaluation régulière des dépenses
            
2- Sensibilité au rapport qualité/prix
            
- comparaison accrue des offres
            
3- Niveau d'exigence élevé
            
- attentes fortes en matière d'expérience et de service
            
4- Faible fidélité
            
- plus susceptibles à tester des alternatives
            
Implications stratégiques:
            
1 - Renforcer la proposition de valeur pour ce segment
            
2 - Mettre en place des actions de rétention ciblées
            
3 - Optimiser l'offre : prix, expérience client, flexibilité

4 - Collecter la voix du client :

- envoyer des questionnaires de satisfaction / de départ (churn)
- identifier les raisons principales de désengagement
- exploiter les retours pour ajuster l'offre et les parcours clients
            
""")


st.subheader("6 - Taux de churn par genre")
st.markdown("Les femmes ont un taux de churn plus élevé que les hommes.")
taux_churn_genre = df_filtered.groupby("Gender_label")["Exited"].mean().reset_index()
taux_churn_genre["Exited"] *=100
fig4 = px.bar(
    taux_churn_genre, x="Gender_label", y="Exited", text=taux_churn_genre["Exited"].round(1),
    labels={"Gender_label":"Genre","Exited":"Taux (%)"}, color="Exited", color_continuous_scale="blues"
)
fig4.update_traces(textposition="outside")
st.plotly_chart(fig4)

st.markdown("""
Le genre est un indicateur, pas une cause directe.
            
Le churn peut être lié à :

- type d'offre consommée
            
- usage du produit
            
- expérience vécue
            
Implications:
            
            -> Améliorer l'expérience utilisateur (fluidité, simplicité)
            
            -> Adapter la communication et le positionnement
            
            -> Analyser les points de friction spécifiques à ce segment
            
            -> Mettre en place des actions de rétention ciblées
            
""")


st.subheader("7 - Taux de churn par pays (%)")
st.markdown("Taux de churn par pays en pourcentage. la France faible Taux contrairement a l'Allemagne ")
taux_churn_pays = df_filtered.groupby("Geography")["Exited"].mean().reset_index()
taux_churn_pays["Taux (%)"] = taux_churn_pays["Exited"]*100

fig_churn_pays = px.bar(
    taux_churn_pays, x="Geography", y="Taux (%)", text=taux_churn_pays["Taux (%)"].round(1),
    labels={"Geography":"Pays","Taux (%)":"Taux de churn (%)"},
    color="Taux (%)", color_continuous_scale="blues",color_discrete_sequence=["#121778", "#59B6F9"]
)

fig_churn_pays.update_traces(textposition="outside")
st.plotly_chart(fig_churn_pays)

st.markdown(""" 
Constat :
            
            -> Le taux de churn est plus élevé chez les clients allemands 
            que chez les clients français et espagnols.
            
Interprétation business :

Plusieurs facteurs peuvent expliquer cet écart :

1- Pouvoir d'achat plus élevé
                        
            → les clients allemands ont en moyenne des revenus plus élevés, ce qui les rend
            moins sensibles au prix mais plus exigeants sur la qualité et la valeur perçue
                        
2- Exigence accrue
                        
            → attentes plus élevées en termes de service, fiabilité et expérience
                        
3- Plus grande liberté de choix
                        
            → capacité à changer facilement pour une offre jugée meilleure
                        
4- Marché concurrentiel
                        
            → présence d'alternatives attractives
            
Implications stratégiques :
            
            -> Mettre l'accent sur la qualité et la valeur plutôt que sur le prix

            -> Améliorer l'expérience client (fiabilité, service, simplicité)

            -> Différencier l'offre pour justifier le positionnement

            -> Personnaliser l'approche pour le marché allemand

            -> Collecter des feedbacks ciblés pour comprendre les attentes spécifiques
""")

# -------------------------------
# Section 5 : Analyse fidélité
# -------------------------------


st.subheader("""8 - Distribution du nombre de produits selon le churn
            ->  Les clients avec 1 seul produit = risque de churn
            """)

fig6 = px.histogram(df_filtered, x="NumOfProducts", color="Exited", barmode="group", text_auto=True)
st.plotly_chart(fig6)



st.markdown("""
Clients désabonnés (Exited=1) : 
            
            -> Femme, 30-49 ans, 1 produit, non actif, Allemagne  

Clients fidèles (Exited=0) : 
            
            -> Multi-produits, actifs, carte bancaire 

Variables discriminantes : 
            
            -> âge, genre, activité, nombre de produits, pays.  

Variables financières (Balance, Salaire, CreditScore) :
            
            -> peu impactantes.
""")

st.markdown("""          ------------------------------------------------------------------    """)
# -------------------------------
# Section 6 : Segments clients
# -------------------------------

st.subheader("Segments clients clés")
st.markdown("""
-> Clients à risque élevé : 
            
            Femme, 30-49 ans, non active, 1 produit, Allemagne 

-> Clients fidèles et engagés : 
            
            Multi-produits, actifs, carte bancaire, France 

-> Seniors (>62 ans) : 
            
            Churn similaire à la moyenne → peu discriminant  

-> Mono-produit & inactifs : 
            
            Risque élevé, peu importe âge/pays
""")

st.markdown("""      -------------------------------------------------------------     """)
# -------------------------------
# Section 7 : Prédiction ML
# -------------------------------
st.header("Prédiction du churn")
st.markdown("Entrez les informations d'un client pour estimer sa probabilité de churn")

with st.form("prediction_form"):
    features = {}
    gender_choice = st.selectbox("Genre", ["Femme","Homme"])
    features["Gender"] = 0 if gender_choice=="Femme" else 1
    features["CreditScore"] = st.slider("CreditScore", int(df_ml["CreditScore"].min()), int(df_ml["CreditScore"].max()), int(df_ml["CreditScore"].mean()))
    features["Age"] = st.slider("Age", int(df_ml["Age"].min()), int(df_ml["Age"].max()), int(df_ml["Age"].mean()))
    features["Tenure"] = st.slider("Tenure", int(df_ml["Tenure"].min()), int(df_ml["Tenure"].max()), int(df_ml["Tenure"].mean()))
    features["Balance"] = st.number_input("Balance", value=float(df_ml["Balance"].mean()))
    features["NumOfProducts"] = st.selectbox("NumOfProducts", sorted(df_ml["NumOfProducts"].unique()))
    features["HasCrCard"] = st.checkbox("Carte bancaire", value=bool(round(df_ml["HasCrCard"].mean())))
    features["IsActiveMember"] = st.checkbox("Client actif", value=bool(round(df_ml["IsActiveMember"].mean())))
    features["EstimatedSalary"] = st.number_input("EstimatedSalary", value=float(df_ml["EstimatedSalary"].mean()))
    geography = st.selectbox("Pays", ["France","Germany","Spain"])
    features["Geography_France"] = 1 if geography=="France" else 0
    features["Geography_Germany"] = 1 if geography=="Germany" else 0
    features["Geography_Spain"] = 1 if geography=="Spain" else 0
    submit = st.form_submit_button("Prédire")

if submit:
    input_df = pd.DataFrame([{col: features.get(col,0) for col in cols_pipeline}])
    probas = pipeline.predict_proba(input_df)[0]
    prob_non_churn, prob_churn = probas[0], probas[1]
    seuil = 0.3
    st.subheader("Résultat")
    st.progress(prob_churn)
    st.write(f"Probabilité Churn : {prob_churn:.2%}")
    if prob_churn<0.3: commentaire="Faible risque de churn ✅"
    elif prob_churn<0.5: commentaire="Risque modéré ⚠️"
    else: commentaire="Risque élevé ⚠️"
    st.write(commentaire)
    if prob_churn>seuil: st.error("⚠️⚠️ Client à risque de churn ⚠️⚠️")
    else: st.success("✅ Client fidèle")