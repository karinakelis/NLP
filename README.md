# Ties — Character Relationship Extraction (no LLM)

Classical NLP pipeline that finds characters in a story, the connections between them,
the type of each relationship (family / romance / friendship / work / conflict), its tone,
and how it changes across chapters. Built with spaCy, VADER, networkx and Streamlit.

```
py -m pip install -r requirements.txt
py -m spacy download en_core_web_sm
py ties.py sample_story.txt
py evaluate.py holdout_story.txt gold_holdout.csv
py -m streamlit run app.py
```

Full explanation of every step (in Russian): [README_RU.md](README_RU.md)
