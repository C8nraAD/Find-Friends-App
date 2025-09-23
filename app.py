import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import json

DATA = "welcome_survey_simple_v1.csv"
MODEL_NAME= "welcome_survey_clustering_pipeline_v1"
CLUSTER_NAME= "welcome_survey_cluster_names_and_descriptions_v1.json"

@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster():
    with open(CLUSTER_NAME, "r") as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model=get_model()
    all_df = pd.read_csv(DATA, sep=";")
    df_with_cluster=predict_model(model,data=all_df)
    return df_with_cluster

st.title("Znajdź znajomych")

with st.sidebar:
    st.header("Powiedz coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    
    age=st.selectbox("Wiek 👵🧓",["<18","25-34","45-54","25-44","18-24",">=65"])
    edu_lvl = st.selectbox("Wykształcenie 🎓", ["Podstawowe", "Średnie", "Wyżesze"])
    fav_animals=st.selectbox("Ulubione Zwierzę 🐾",["Brak ulubionych","Kot","Pies","Nosorożec"])
    fav_place=st.selectbox("Ulubione miejsce 🏕️", ["Nad wodą", "W górach", "Nad wodą"])
    gender = st.radio("Płeć 🚻", ["Mężczyzna", "Kobieta"])

    search_button = st.button("Wyszukaj 🔍")

    st.markdown("---")
    st.markdown("### O projekcie")
    st.markdown("Ta aplikacja wykorzystuje **uczenie maszynowe (PyCaret)**, a dokładnie technikę **klastrowania** (unsupervised learning) do grupowania osób na podstawie ich preferencji. Cel to znalezienie osób o podobnych zainteresowaniach, co może być przydatne w **analizie biznesowej** do **segmentacji klientów**.")
    st.markdown("---")


if search_button:
    person_df = pd.DataFrame([
        {
            "age": age,
            "edu_level": edu_lvl,
            "fav_animals": fav_animals,
            "fav_place": fav_place,
            "gender": gender
        }
    ])
    model = get_model()
    all_df = get_all_participants()
    cluster_desc = get_cluster()

    predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
    predicted_cluster_data = cluster_desc[predicted_cluster_id]
    
    cluster_number = int(predicted_cluster_id.split(" ")[1])
    st.image(f"{cluster_number}.png")
    
    st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])

    same_cluster = all_df[all_df["Cluster"] == predicted_cluster_id]
    st.metric("Liczba twoich znajomych", len(same_cluster))

    st.header("Osoby z grupy")
    fig = px.histogram(same_cluster.sort_values("age"), x="age")
    fig.update_layout(
        title="Rozkład wieku w grupie",
        xaxis_title="Wiek",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster, x="edu_level")
    fig.update_layout(
        title="Rozkład wykształcenia w grupie",
        xaxis_title="Wykształcenie",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster, x="fav_animals")
    fig.update_layout(
        title="Rozkład ulubionych zwierząt w grupie",
        xaxis_title="Ulubione zwierzęta",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster, x="fav_place")
    fig.update_layout(
        title="Rozkład ulubionych miejsc w grupie",
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster, x="gender")
    fig.update_layout(
        title="Rozkład płci w grupie",
        xaxis_title="Płeć",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)
