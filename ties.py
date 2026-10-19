"""
Ties — извлечение отношений между персонажами (без LLM).
Находит персонажей, связи между ними, тип и тон отношений,
и как тон меняется по главам. Только spaCy + VADER + наши правила.

Запуск:  py ties.py sample_story.txt
"""
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

import spacy
from spacy.tokens import DocBin
import networkx as nx
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL = "en_core_web_sm"
nlp = spacy.load(MODEL)
vader = SentimentIntensityAnalyzer()
CACHE_DIR = ".ties_cache"


# =====================================================================
# УСКОРЕНИЕ: разбор spaCy (самая медленная часть, ~80% времени)
# =====================================================================
def parse_chapters(chapters, n_process=1):
    """1) nlp.pipe обрабатывает главы пачками — примерно в 2 раза быстрее, чем по одной.
    2) Результат сохраняется на диск: повторный запуск того же текста — почти мгновенный.
    n_process=2 и больше — несколько ядер процессора (быстрее на длинных книгах)."""
    key = hashlib.md5(("\x00".join(chapters) + MODEL).encode("utf-8")).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".spacy")
    if os.path.exists(path):
        return list(DocBin().from_disk(path).get_docs(nlp.vocab))
    docs = list(nlp.pipe(chapters, batch_size=64, n_process=n_process))
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        DocBin(docs=docs).to_disk(path)
    except OSError:
        pass   # кэш — необязательная вещь
    return docs


# =====================================================================
# СЛОВАРИ
# =====================================================================

# Слова-признаки типа отношений (сравниваем с леммой: "kissed" -> "kiss")
KEYWORDS = {
    "family":     {"brother", "sister", "father", "mother", "daughter", "son",
                   "sibling", "parent", "child", "uncle", "aunt", "cousin",
                   "family", "grandmother", "grandfather"},
    "romance":    {"kiss", "marry", "love", "lover", "romance", "date",
                   "fiance", "fiancee", "wife", "husband", "propose", "court",
                   "beloved", "sweetheart", "wed"},
    "friendship": {"friend", "trust", "comfort", "talk", "help", "visit",
                   "walk", "laugh", "support", "confide"},
    "work":       {"boss", "factory", "work", "employ", "employee", "employer",
                   "fire", "hire", "colleague", "office", "owner", "own",
                   "nurse", "assistant", "servant", "master", "clinic"},
    "conflict":   {"argue", "blame", "threaten", "insult", "attack", "shout",
                   "hate", "enemy", "fight", "accuse", "betray", "rival",
                   "steal", "quarrel", "mock"},
}
# Фразы из нескольких слов
PHRASES = {"best friend": "friendship", "best man": "friendship",
           "in love": "romance"}

# Глаголы для тона: +1 тёплые, -1 враждебные
POSITIVE_VERBS = {"help", "thank", "hug", "kiss", "forgive", "love", "smile",
                  "laugh", "comfort", "trust", "marry", "welcome", "apologize",
                  "respect", "invite", "dance", "praise", "save", "protect",
                  "visit", "support", "defend", "friend"}
NEGATIVE_VERBS = {"threaten", "accuse", "blame", "attack", "insult", "shout",
                  "argue", "hate", "betray", "fight", "steal", "lie", "fire",
                  "leave", "abandon", "deny", "mock"}

# Роли: "her son Oliver" -> Оливер — ребёнок Клары
ROLE = {  # слово -> (нормализованная роль, категория, пол того, кто в этой роли)
    "son": ("child", "family", "M"), "daughter": ("child", "family", "F"),
    "father": ("parent", "family", "M"), "mother": ("parent", "family", "F"),
    "brother": ("sibling", "family", "M"), "sister": ("sibling", "family", "F"),
    "husband": ("spouse", "romance", "M"), "wife": ("spouse", "romance", "F"),
    "uncle": ("uncle", "family", "M"), "aunt": ("aunt", "family", "F"),
    "cousin": ("cousin", "family", None),
    "friend": ("friend", "friendship", None),
    "boss": ("boss", "work", None), "employer": ("boss", "work", None),
    "colleague": ("colleague", "work", None),
}
# Обратная роль: если Оливер — child Клары, то Клара — parent Оливера
INVERSE = {"child": "parent", "parent": "child", "sibling": "sibling",
           "spouse": "spouse", "friend": "friend", "boss": "employee",
           "colleague": "colleague", "cousin": "cousin",
           "uncle": "nephew", "aunt": "nephew"}

TITLES_F = ("Mrs.", "Mrs", "Ms.", "Ms", "Miss", "Lady", "Madam")
TITLES_M = ("Mr.", "Mr", "Sir", "Lord")
TITLES = TITLES_F + TITLES_M + ("Dr.", "Dr", "Professor", "Captain")

MALE_PRON = {"he", "him", "his", "himself"}
FEMALE_PRON = {"she", "her", "hers", "herself"}

# Предлоги места/темы: слова после них не говорят о типе отношений
# ("at the wedding", "about Rosa", "under the bridge")
SKIP_PREPS = {"at", "on", "beside", "near", "under", "behind", "across",
              "through", "about", "from", "by", "around", "over", "after",
              "before", "during"}
# Части предложения, которые являются отдельными мыслями (свой глагол)
CLAUSE_DEPS = {"ccomp", "advcl", "relcl", "conj", "parataxis", "csubj", "acl"}


def forms(tok):
    """Разные формы слова. spaCy иногда ошибается с леммой ("hated" -> "hat"),
    поэтому проверяем ещё само слово и слово без окончания."""
    low = tok.lower_
    out = {tok.lemma_.lower(), low}
    for suf in ("ied", "ed", "d", "ing", "s"):
        if low.endswith(suf) and len(low) - len(suf) >= 3:
            out.add(low[: -len(suf)] + ("y" if suf == "ied" else ""))
    return out


# =====================================================================
# ШАГ 1. Главы
# =====================================================================
def split_chapters(text):
    # убираем служебный текст Project Gutenberg, если он есть
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    if start:
        text = text[start.end(): end.start() if end else None]
    parts = re.split(r"(?im)^\s*chapter\s+[\w.]+.*$", text)
    chapters = [p.strip() for p in parts if p.strip()]
    return chapters if chapters else [text]


# =====================================================================
# ШАГ 2. Персонажи и псевдонимы
# =====================================================================
def clean_name(name):
    for t in sorted(TITLES, key=len, reverse=True):
        name = re.sub(rf"^{re.escape(t)}\s+", "", name)
    return name.replace("'s", "").strip()


def find_aliases(docs):
    counts = Counter()
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = clean_name(ent.text)
                if name and name[0].isupper() and len(name.split()) <= 3:
                    counts[name] += 1

    # Объединяем псевдонимы: "Anna" -> "Anna Petrova"
    full_names = [n for n in counts if " " in n]
    alias = {}
    for name in counts:
        if " " in name:
            alias[name] = name
            continue
        match = [f for f in full_names if f.split()[0] == name]
        if not match:
            match = [f for f in full_names if f.split()[-1] == name]
        alias[name] = match[0] if len(match) == 1 else name

    # Имя — всегда псевдоним; фамилия — только если она уникальна
    surnames = Counter(f.split()[-1] for f in full_names)
    for f in full_names:
        alias.setdefault(f.split()[0], f)
        if surnames[f.split()[-1]] == 1:
            alias.setdefault(f.split()[-1], f)
    return alias


def name_of(tok, alias):
    """Персонаж, которого называет это слово (или None).
    "Clara Hale": главное слово "Hale" общее, поэтому смотрим и на "Clara"."""
    if tok.text in alias:
        return alias[tok.text]
    for child in tok.children:
        if child.dep_ == "compound" and child.text in alias:
            return alias[child.text]
    return None


# =====================================================================
# ШАГ 3. Пол персонажей (нужен для местоимений)
# =====================================================================
def guess_genders(docs, alias, text):
    votes = defaultdict(Counter)

    # 1) Титулы: "Mrs. Hale" -> женщина
    for t in TITLES_F + TITLES_M:
        for m in re.finditer(rf"\b{re.escape(t)}\s+([A-Z]\w+)", text):
            if m.group(1) in alias:
                votes[alias[m.group(1)]]["F" if t in TITLES_F else "M"] += 3

    for doc in docs:
        sents = list(doc.sents)
        for i, sent in enumerate(sents):
            # 2) Роли: "her son Oliver" -> Оливер мужчина
            for tok in sent:
                if tok.lemma_.lower() in ROLE and ROLE[tok.lemma_.lower()][2]:
                    for child in tok.children:
                        c = name_of(child, alias)
                        if child.dep_ == "appos" and c:
                            votes[c][ROLE[tok.lemma_.lower()][2]] += 3
                    if tok.dep_ == "appos" and name_of(tok.head, alias):
                        votes[name_of(tok.head, alias)][ROLE[tok.lemma_.lower()][2]] += 3
            # 3) "Clara denied it. She went home." -> подлежащее, а затем he/she
            if i + 1 < len(sents):
                subj = [name_of(t, alias) for t in sent
                        if t.dep_ == "nsubj" and t.head.dep_ == "ROOT"]
                nxt = [t.lower_ for t in sents[i + 1]
                       if t.dep_ == "nsubj" and t.head.dep_ == "ROOT"]
                if len(subj) == 1 and subj[0] and nxt:
                    if nxt[0] == "he":
                        votes[subj[0]]["M"] += 1
                    elif nxt[0] == "she":
                        votes[subj[0]]["F"] += 1
    return {c: v.most_common(1)[0][0] for c, v in votes.items()}


# =====================================================================
# ШАГ 4. Кто упомянут: имена + местоимения + "her son"
# =====================================================================
def governing_verb(tok):
    t = tok
    while t.head is not t and t.pos_ not in ("VERB", "AUX"):
        t = t.head
    return t


def resolve_mentions(doc, alias, gender, characters, role_map, explicit):
    """Возвращает {номер слова: персонаж} для имён, местоимений и ролей.
    Местоимение = самый недавний персонаж того же пола (в этом или прошлом предложении)."""
    mention = {}
    recent = []  # [(персонаж, номер предложения)]
    for s_i, sent in enumerate(doc.sents):
        for tok in sent:
            c = name_of(tok, alias)
            if c and c not in characters:
                c = None
            low = tok.lower_

            # --- местоимение ---
            if c is None and (low in MALE_PRON or low in FEMALE_PRON) \
                    and not low.endswith("self"):
                g = "M" if low in MALE_PRON else "F"
                # "Victor shouted at him": him ≠ Victor (подлежащее того же глагола)
                exclude = set()
                if tok.dep_ in ("dobj", "pobj", "dative"):
                    exclude = subjects_of(governing_verb(tok), mention)
                seen = set()
                for cand, cand_s in reversed(recent):
                    if cand_s < s_i - 1:
                        break
                    if cand in seen or cand in exclude:
                        continue
                    seen.add(cand)
                    if gender.get(cand, g) == g:   # пол совпадает или неизвестен
                        c = cand
                        gender.setdefault(c, g)    # "She smiled" -> Роза — женщина
                        break

            # --- роль: "her son" -> Oliver ---
            lem = tok.lemma_.lower()
            if lem in ROLE:
                owner = next((mention.get(ch.i) or name_of(ch, alias)
                              for ch in tok.children if ch.dep_ == "poss"), None)
                named = next((name_of(ch, alias) for ch in tok.children
                              if ch.dep_ == "appos" and name_of(ch, alias)), None)
                if named is None and tok.i + 1 < len(doc):
                    # "Peter's cousin Nora Lowe" — имя сразу после роли
                    nb = doc[tok.i + 1]
                    named = name_of(nb, alias) or (name_of(nb.head, alias)
                                                   if nb.dep_ == "compound" else None)
                role, cat, _ = ROLE[lem]
                if owner and named and named in characters and owner != named:
                    # явное указание: "her son Oliver"
                    role_map[(owner, role)] = named
                    role_map[(named, INVERSE[role])] = owner
                    explicit.append((owner, named, cat))
                elif owner and not named and c is None:
                    # ссылка через роль: "her son" -> Oliver
                    c = role_map.get((owner, role))
                # "Lucy Kane, the daughter of Victor"
                if tok.dep_ == "appos":
                    head_c = mention.get(tok.head.i) or name_of(tok.head, alias)
                    of_c = [name_of(p, alias) for prep in tok.children
                            if prep.lower_ == "of" for p in prep.children]
                    of_c = [x for x in of_c if x in characters]
                    if head_c in characters and of_c and head_c != of_c[0]:
                        role_map[(of_c[0], role)] = head_c
                        role_map[(head_c, INVERSE[role])] = of_c[0]
                        explicit.append((head_c, of_c[0], cat))

            if c:
                mention[tok.i] = c
                recent.append((c, s_i))
    return mention


# =====================================================================
# ШАГ 5. Прямые действия: кто что сделал кому
# =====================================================================
def subjects_of(verb, mention):
    """Подлежащие глагола. "Anna and Mark talked" -> {Anna, Mark}.
    Если своего подлежащего нет ("...and hugged him"), берём у главного глагола."""
    subs = set()
    for ch in verb.children:
        if ch.dep_ in ("nsubj", "nsubjpass"):
            for t in [ch] + list(ch.conjuncts):
                if t.i in mention:
                    subs.add(mention[t.i])
                else:
                    # "Her daughter Mia helped": подлежащее "daughter" = Mia
                    subs |= {mention[a.i] for a in t.children
                             if a.dep_ == "appos" and a.i in mention}
    if not subs and verb.dep_ == "xcomp":
        # "Mia asked Peter to help her": кто помогает? Peter — дополнение главного глагола
        objs = [c for c in verb.head.children if c.dep_ in ("dobj", "dative")]
        if objs and objs[0].i in mention:
            return {mention[objs[0].i]}
    if not subs and verb.dep_ in ("conj", "xcomp") and verb.head is not verb:
        return subjects_of(verb.head, mention)
    return subs


def clause_tokens(verb):
    """Слова, принадлежащие этому глаголу (без подлежащего и вложенных предложений)."""
    out, stack = [], [c for c in verb.children
                      if c.dep_ not in ("nsubj", "nsubjpass")
                      and c.dep_ not in CLAUSE_DEPS]
    while stack:
        t = stack.pop()
        out.append(t)
        stack.extend(c for c in t.children if c.dep_ not in CLAUSE_DEPS)
    return sorted(out, key=lambda t: t.i)


def interactions(sent, mention):
    """Список действий: (подлежащие, объекты, слова-признаки, есть ли отрицание, глагол)."""
    acts = []
    for verb in sent:
        if verb.pos_ not in ("VERB", "AUX"):
            continue
        subs = subjects_of(verb, mention)
        if not subs:
            continue
        toks = clause_tokens(verb)
        tok_set = set(toks)
        objs = {mention[t.i] for t in toks if t.i in mention} - subs
        # слова-признаки: сам глагол + слова, кроме тех, что после предлогов места
        verb_forms = forms(verb)
        if "laugh" in verb_forms and any(c.lower_ == "at" for c in verb.children):
            verb_forms = {"mock"}          # "laughed at him" — насмешка, не дружба
        cues = [verb_forms]
        for t in toks:
            prep = next((a for a in t.ancestors if a.dep_ == "prep"), None)
            if prep is not None and prep in tok_set and prep.lower_ in SKIP_PREPS:
                continue
            cues.append(forms(t))
        text = " ".join(t.lower_ for t in toks)
        cues += [{p} for p in PHRASES if p in text]
        negated = any(c.dep_ == "neg" or c.lower_ == "never" for c in verb.children)
        if objs or len(subs) > 1:
            acts.append((subs, objs, cues, negated, verb_forms))
    return acts


def pairs_of(act):
    subs, objs, *_ = act
    pairs = {tuple(sorted((s, o))) for s in subs for o in objs if s != o}
    pairs |= {tuple(sorted(p)) for p in combinations(subs, 2)}
    return pairs


def cue_category(cue_forms):
    """cue_forms — множество форм одного слова."""
    for f in cue_forms:
        if f in PHRASES:
            return PHRASES[f]
    for cat, kws in KEYWORDS.items():
        if cue_forms & kws:
            return cat
    return None


# =====================================================================
# ШАГ 6. Тип отношений
# =====================================================================
def relation_type(scores):
    """Порядок важности: семья > романтика > работа/дружба > конфликт.
    Конфликт — это скорее тон: начальник может ругать сотрудника, но они всё равно коллеги."""
    if scores["family"] >= 2:
        return "family"
    if scores["romance"] >= 2:
        return "romance"
    if scores["work"] >= 2 and scores["work"] >= scores["friendship"]:
        return "work"
    if scores["friendship"] >= 2 and scores["friendship"] > scores["conflict"]:
        return "friendship"
    best = max(("work", "friendship"), key=lambda c: scores[c])
    if scores["conflict"] > 0:
        return "conflict"
    if scores[best] > 0:
        return best
    return "unknown"


# =====================================================================
# ШАГ 7. Тон
# =====================================================================
def verb_polarity(act):
    *_, negated, verb = act
    words = set().union(*act[2])
    if words & NEGATIVE_VERBS:
        return 0 if negated else -1
    if words & POSITIVE_VERBS:
        return -1 if negated else 1   # "did not trust" -> плохо
    return 0


def tone_label(score):
    if score >= 0.25:  return "positive"
    if score <= -0.25: return "negative"
    return "neutral"


# =====================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =====================================================================
def analyze(text, min_mentions=2, n_process=1):
    chapters = split_chapters(text)
    nlp.max_length = max(nlp.max_length, max(len(c) for c in chapters) + 1)
    docs = parse_chapters(chapters, n_process)
    alias = find_aliases(docs)

    # сколько раз названо по имени
    mentions = Counter()
    for doc in docs:
        for tok in doc:
            c = name_of(tok, alias)
            if c and not (tok.dep_ == "compound" and name_of(tok.head, alias) == c):
                mentions[c] += 1
    characters = {n for n, c in mentions.items() if c >= min_mentions}
    gender = guess_genders(docs, alias, text)

    role_map, explicit = {}, []
    pair = defaultdict(lambda: {"by_ch": defaultdict(list), "scores": Counter(),
                                "direct": 0, "evidence": []})

    for ch_i, doc in enumerate(docs, start=1):
        n_explicit = len(explicit)
        mention = resolve_mentions(doc, alias, gender, characters, role_map, explicit)
        for a, b, cat in explicit[n_explicit:]:        # явные роли: +5
            pair[tuple(sorted((a, b)))]["scores"][cat] += 5
            pair[tuple(sorted((a, b)))]["direct"] += 1

        for sent in doc.sents:
            people = {mention[t.i] for t in sent if t.i in mention}
            acts = interactions(sent, mention)
            vader_score = vader.polarity_scores(sent.text)["compound"]

            for a, b in combinations(sorted(people), 2):
                p = pair[(a, b)]
                my_acts = [act for act in acts if (a, b) in pairs_of(act)]
                # тип: слова из прямых действий этой пары, +2
                for act in my_acts:
                    if act[3]:   # отрицание: "did not trust" не дружба
                        continue
                    for cue in act[2]:
                        cat = cue_category(cue)
                        if cat:
                            p["scores"][cat] += 2
                    p["evidence"].append(sent.text.strip())
                # тип: слова в предложении, где только эти двое, +1
                if len(people) == 2 and not my_acts:
                    for t in sent:
                        cat = cue_category(forms(t))
                        if cat:
                            p["scores"][cat] += 1
                # тон: VADER + тёплые/враждебные глаголы этой пары
                pol = sum(verb_polarity(act) for act in my_acts)
                pol = max(-1, min(1, pol))
                p["by_ch"][ch_i].append(0.5 * vader_score + 0.5 * pol)
                p["direct"] += len(my_acts)

    rows = []
    for (a, b), p in pair.items():
        weight = sum(len(v) for v in p["by_ch"].values())
        # фильтр шума: нужна хотя бы 1 прямая связь или 2 совместных предложения
        if weight == 0 or (p["direct"] == 0 and weight < 2):
            continue
        all_scores = [x for v in p["by_ch"].values() for x in v]
        t = sum(all_scores) / len(all_scores)
        arc = {ch: tone_label(sum(v) / len(v)) for ch, v in sorted(p["by_ch"].items())}
        rows.append({
            "a": a, "b": b,
            "weight": weight,
            "direct": p["direct"],
            "type": relation_type(p["scores"]),
            "tone": tone_label(t),
            "tone_score": round(t, 2),
            "arc": " → ".join(f"Ch{ch}:{lab}" for ch, lab in arc.items()),
            "changed": len(set(arc.values())) > 1,
            "scores": dict(p["scores"]),
            "evidence": p["evidence"][:3],
        })

    cols = ["a", "b", "weight", "direct", "type", "tone", "tone_score", "arc",
            "changed", "scores", "evidence"]
    rel = pd.DataFrame(rows, columns=cols)
    rel = rel.sort_values("weight", ascending=False).reset_index(drop=True)
    chars = pd.DataFrame(
        [{"character": c, "mentions": mentions[c], "gender": gender.get(c, "?")}
         for c in characters]
    ).sort_values("mentions", ascending=False).reset_index(drop=True)
    return chars, rel


# =====================================================================
# ШАГ 8. Описание текстом (шаблон)
# =====================================================================
def describe(row):
    text = (f"{row['a']} and {row['b']} appear together in {row['weight']} "
            f"sentence(s) ({row['direct']} direct interaction(s)). "
            f"Their relationship looks like {row['type']}, with a {row['tone']} tone.")
    if row["changed"]:
        text += f" It changes over the story: {row['arc']}."
    return text


# =====================================================================
# ШАГ 9. Граф
# =====================================================================
def build_graph(rel):
    G = nx.Graph()
    for _, r in rel.iterrows():
        G.add_edge(r["a"], r["b"], weight=r["weight"], type=r["type"], tone=r["tone"])
    return G


def save_graph_png(G, path="graph.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"positive": "#2e7d32", "negative": "#c62828", "neutral": "#9e9e9e"}
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color="#ffd54f", node_size=1800)
    nx.draw_networkx_labels(G, pos, font_size=9)
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1 + min(d["weight"], 8),
                               edge_color=colors[d["tone"]])
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d["type"]
                                 for u, v, d in G.edges(data=True)}, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_story.txt"
    text = open(path, encoding="utf-8").read()
    # длинная книга: py ties.py book.txt 2   (2 = число ядер процессора)
    n_proc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    chars, rel = analyze(text, n_process=n_proc)

    print("\n=== CHARACTERS ===")
    print(chars.to_string(index=False))
    print("\n=== RELATIONSHIPS ===")
    print(rel[["a", "b", "weight", "direct", "type", "tone", "arc"]].to_string(index=False))
    print("\n=== DESCRIPTIONS ===")
    for _, r in rel.iterrows():
        print("-", describe(r))

    rel.drop(columns=["scores", "evidence"]).to_csv("relations.csv", index=False)
    save_graph_png(build_graph(rel))
    print("\nSaved: relations.csv, graph.png")
