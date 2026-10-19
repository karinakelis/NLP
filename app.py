"""
Простой интерфейс.  Запуск:  py -m streamlit run app.py
"""
import streamlit as st
from ties import analyze, describe, build_graph, save_graph_png

st.title("Ties — character relationships")


# Streamlit перезапускает весь скрипт при каждом клике.
# cache_data запоминает результат: тот же текст + те же настройки -> мгновенно.
@st.cache_data(show_spinner="Analyzing the story…")
def run_analysis(text, min_mentions):
    return analyze(text, min_mentions=min_mentions)


@st.cache_data
def make_graph(text, min_mentions, max_nodes, top_links, _rel):
    # "_rel" с подчёркиванием Streamlit не хэширует; ключ кэша — текст + настройки
    save_graph_png(build_graph(_rel, max_nodes=max_nodes, top_links=top_links), "graph.png")
    return open("graph.png", "rb").read()


uploaded = st.file_uploader("Upload a story (.txt)", type="txt")
text = uploaded.read().decode("utf-8") if uploaded else open(
    "sample_story.txt", encoding="utf-8").read()
min_m = st.slider("Minimum mentions for a character", 1, 10, 2)

chars, rel = run_analysis(text, min_m)

st.subheader("Characters")
st.dataframe(chars)

st.subheader("Network")
c1, c2 = st.columns(2)
max_nodes = c1.slider("Characters in the graph", 3, 40, 15)
top_links = c2.slider("Strongest links per character", 1, 10, 3)
st.image(make_graph(text, min_m, max_nodes, top_links, rel))
st.caption("Line colour = relationship type; thickness = how often they appear together; "
           "dashed = negative tone. Node size = how connected the character is.")

st.subheader("Relationships")
st.dataframe(rel[["a", "b", "weight", "direct", "type", "tone", "arc"]])

st.subheader("Descriptions")
for _, r in rel.iterrows():
    st.write("•", describe(r))
