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
    # Слова родства (brother, sister…) здесь НЕТ: "his sister" часто говорит о третьем
    # человеке. Родство берём из явных ролей ("her son Oliver") и общей фамилии.
    "family":     {"family", "relation", "relative", "kin", "sibling", "siblings",
                   "sisterly", "brotherly"},
    "romance":    {"kiss", "marry", "love", "lover", "romance", "date",
                   "fiance", "fiancee", "wife", "husband", "propose", "court",
                   "beloved", "sweetheart", "wed", "engage", "engagement",
                   "marriage", "wedding", "attachment", "admirer",
                   "suitor", "proposal", "courtship", "flirt",
                   "bride", "bridegroom", "caress"},
    "friendship": {"friend", "friendship", "trust", "comfort", "talk", "help",
                   "visit", "walk", "laugh", "support", "confide", "companion",
                   "kindness", "accompany", "share", "neighbour", "neighbor",
                   "dine", "chat", "play", "ally"},
    "work":       {"boss", "factory", "work", "employ", "employee", "employer",
                   "fire", "hire", "colleague", "office", "owner", "own",
                   "nurse", "assistant", "servant", "master", "clinic", "maid",
                   "butler", "clerk", "partner", "business", "tenant",
                   "landlord", "agent", "lawyer", "client", "pay", "wage",
                   "salary", "job", "shop", "deliver", "serve", "order",
                   "command", "officer", "crew", "duty"},
    "conflict":   {"argue", "blame", "threaten", "insult", "attack", "shout",
                   "hate", "enemy", "fight", "accuse", "betray", "rival",
                   "steal", "quarrel", "mock", "envy", "resent", "dislike",
                   "despise", "scorn", "offend", "oppose", "deceive", "cheat",
                   "liar", "rumour", "rumor", "confront", "dispute", "kill",
                   "murder", "strike", "curse", "revenge", "jealous"},
}
# Фразы из нескольких слов
PHRASES = {"best friend": "friendship", "best man": "friendship",
           "in love": "romance"}

# Глаголы для тона: +1 тёплые, -1 враждебные
POSITIVE_VERBS = {"help", "thank", "hug", "kiss", "forgive", "love", "smile",
                  "laugh", "comfort", "trust", "marry", "welcome", "apologize",
                  "respect", "invite", "dance", "praise", "save", "protect",
                  "visit", "support", "defend", "friend", "admire", "kindness",
                  "affection", "embrace", "please", "delight", "praise",
                  "comfort", "engage", "marry", "esteem", "cheer"}
NEGATIVE_VERBS = {"threaten", "accuse", "blame", "attack", "insult", "shout",
                  "argue", "hate", "betray", "fight", "steal", "lie", "fire",
                  "leave", "abandon", "deny", "mock", "envy", "resent",
                  "dislike", "despise", "scorn", "offend", "deceive", "cheat",
                  "liar", "confront", "kill", "murder", "strike", "curse",
                  "quarrel", "insult", "cruel", "anger", "angry"}

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

TITLES_F = ("Mrs.", "Mrs", "Ms.", "Ms", "Miss", "Lady", "Madam",
            # французские, итальянские и др. (Дюма, Толстой в переводе…)
            "Madame", "Mme.", "Mme", "Mademoiselle", "Mlle.", "Mlle", "Signora",
            "Señora", "Baroness", "Countess", "Comtesse", "Marquise", "Duchess",
            "Princess", "Frau", "Dame")
TITLES_M = ("Mr.", "Mr", "Sir", "Lord", "Captain", "Admiral", "Colonel",
            "General", "Major", "Uncle",
            "M.", "Monsieur", "Signor", "Señor", "Don", "Baron", "Count",
            "Comte", "Marquis", "Viscount", "Vicomte", "Duke", "Prince",
            "Abbé", "Abbe", "Herr")
TITLES = TITLES_F + TITLES_M + ("Dr.", "Dr", "Professor", "Aunt")

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
# Слова, которые spaCy часто ошибочно считает именами
NOT_NAMES = {"God", "Lord", "Heaven", "Heavens", "Sir", "Madam", "Miss", "Mrs",
             "Mr", "Majesty", "Highness", "Lady", "Ma'am", "Mamma", "Papa",
             "Christmas", "Sunday", "Monday", "Tuesday", "Wednesday",
             "Thursday", "Friday", "Saturday", "Oh", "O", "Ah", "Chapter",
             "I", "Me", "You", "Dear", "Poor", "Yes", "No", "Monsieur",
             "Madame", "Mademoiselle", "Signor", "Count", "Baron", "Abbé",
             "Saint", "Excellency", "Rue", "Château", "Chateau"}
# Первое слово названий мест: "Rue de Rivoli", "Château d'If"
PLACE_START = {"Rue", "Château", "Chateau", "Hôtel", "Hotel", "Isle", "Île",
               "Port", "Mount", "Fort", "Boulevard", "Avenue", "Pont", "Lake",
               "Cape", "Gulf", "Bay", "Faubourg", "Quai", "Palais", "Via", "Piazza",
               "Barrière", "Barriere", "Pointe", "Champs"}
# Слова в названиях мест: "Laura Place", "Kellynch Hall"
PLACE_WORDS = {"Place", "Street", "Hall", "Park", "House", "Lodge", "Square",
               "Road", "Lane", "Court", "Abbey", "Castle", "Cottage", "Manor",
               "Church", "Bridge", "River", "Hill", "Town", "City", "Inn"}
# Предлоги, после которых обычно стоит место: "in Bath", "at Kellynch"
PLACE_PREPS = {"in", "at", "into", "near", "towards", "toward", "through",
               "within", "inside", "around", "round", "across"}


def load_names():
    """Списки имён NLTK (Mark Kantrowitz): мужские и женские.
    Имя, которое есть в обоих списках (Leslie), не используем."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "names")
    try:
        male = set(open(os.path.join(base, "male.txt"), encoding="utf-8").read().split())
        female = set(open(os.path.join(base, "female.txt"), encoding="utf-8").read().split())
    except OSError:
        return {}
    out = {n: "M" for n in male - female}
    out.update({n: "F" for n in female - male})
    return out


NAME_GENDER = load_names()


def name_gender(first):
    """Пол по имени; "Mercédès" ищем как "Mercedes"."""
    import unicodedata
    plain = "".join(ch for ch in unicodedata.normalize("NFD", first)
                    if unicodedata.category(ch) != "Mn")
    return NAME_GENDER.get(first) or NAME_GENDER.get(plain)


def ent_name(ent):
    """Чистим имя от мусора spaCy:
    "A Mrs Smith" -> "Smith", "Elliot as Anne" -> "Elliot", "Mary knew" -> "Mary"."""
    words = []
    for tok in ent:
        if tok.is_punct and words:
            break                       # "Mr Elliot, Anne" — это два человека
        w = tok.text.replace("'s", "").replace("’s", "").strip(".,;:!?'\"")
        if not w:
            continue
        if not words and (tok.pos_ == "DET" or w in TITLES or w + "." in TITLES
                          or w in NOT_NAMES):
            continue                    # артикль или титул в начале
        if not w[0].isupper() or tok.pos_ not in ("PROPN", "NOUN", "ADJ"):
            break                       # дальше уже не имя
        words.append(w)
    name = " ".join(words[:3])
    if len(name) < 2 or name in NOT_NAMES or words[-1] in PLACE_WORDS \
            or words[0] in PLACE_START or (len(words) > 1 and words[-1] == "Saint"):
        return ""
    return name


def clean_name(name):
    for t in sorted(TITLES, key=len, reverse=True):
        name = re.sub(rf"^{re.escape(t)}\s+", "", name)
    return name.replace("'s", "").strip()


def find_aliases(docs):
    counts = Counter()
    as_person, as_place = Counter(), Counter()
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent_name(ent)
                if name:
                    counts[name] += 1
                    as_person.update(name.split())
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                as_place.update(t.text for t in ent)
    # Слово, которое spaCy намного чаще называет местом, чем человеком (Marseilles, Elba).
    # Порог x3: spaCy иногда путается ("the King" в «Алисе» бывает LOC).
    for name in list(counts):
        if all(as_place[w] > 3 * as_person[w] for w in name.split()):
            del counts[name]

    full_names = [n for n in counts if " " in n]

    # Какой пол у титулов перед фамилией: "Mrs Smith" -> F, "Captain Harville" -> M
    text = " ".join(d.text for d in docs)
    title_g = defaultdict(set)
    for t in TITLES_F + TITLES_M:
        for m in re.finditer(rf"\b{re.escape(t)}\s+([A-Z]\w+)", text):
            title_g[m.group(1)].add("F" if t in TITLES_F else "M")

    def same_person(surname, full):
        """"Mrs Smith" — не "Charles Smith": женский титул + мужское имя = разные люди."""
        g = name_gender(full.split()[0])
        return not g or not title_g[surname] or g in title_g[surname]

    def pick(match):
        """Один вариант — берём его. Несколько — берём тот, что встречается
        хотя бы в 3 раза чаще остальных ("Dantès": Edmond Dantès, а не его отец)."""
        if len(match) == 1:
            return match[0]
        if match:
            top = sorted(match, key=counts.get, reverse=True)
            if counts[top[0]] >= 3 * counts[top[1]]:
                return top[0]
        return None

    # Объединяем псевдонимы: "Anna" -> "Anna Petrova"
    alias = {}
    for name in counts:
        if " " in name:
            alias[name] = name
            continue
        best = pick([f for f in full_names if f.split()[0] == name])
        if best is None:
            best = pick([f for f in full_names
                         if f.split()[-1] == name and same_person(name, f)])
        alias[name] = best or name

    # Части полных имён тоже становятся псевдонимами
    for part in {f.split()[0] for f in full_names} | {f.split()[-1] for f in full_names}:
        if part in alias:
            continue
        best = pick([f for f in full_names if f.split()[0] == part])
        if best is None:
            best = pick([f for f in full_names
                         if f.split()[-1] == part and same_person(part, f)])
        if best:
            alias[part] = best
    return alias


def strip_accents(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower().strip()


def merge_same_people(alias, groups):
    """Ручное объединение: [["Edmond Dantès", "Monte Cristo"]] — один человек.
    Так бывает с тайными личностями: правила не могут знать сюжет."""
    for group in groups or []:
        names = {strip_accents(n) for n in group if n.strip()}
        # все варианты из текста, которые совпадают с одним из введённых имён
        targets = {v for k, v in alias.items()
                   if strip_accents(k) in names or strip_accents(v) in names}
        if not targets:
            continue
        # главное имя — первое введённое, если оно есть в тексте
        first = strip_accents(group[0])
        main = next((v for v in targets if strip_accents(v) == first),
                    max(targets, key=len))
        for k, v in alias.items():
            if v in targets:
                alias[k] = main
    return alias


# "the count" -> персонаж, которого чаще всего называют "Count …" / "Count of …"
REF_TITLES = {"count", "countess", "baron", "baroness", "abbé", "abbe", "marquis",
              "marquise", "viscount", "duke", "duchess", "prince", "princess",
              "captain", "admiral", "colonel", "major", "general", "doctor",
              "judge", "inspector", "professor"}


def title_references(docs, alias, characters):
    seen = defaultdict(Counter)
    for doc in docs:
        for tok in doc:
            low = tok.lower_
            if low not in REF_TITLES or tok.i + 1 >= len(doc):
                continue
            nxt = doc[tok.i + 1]
            if nxt.lower_ == "of" and tok.i + 2 < len(doc):
                nxt = doc[tok.i + 2]
            c = name_of(nxt, alias)
            if c in characters:
                seen[low][c] += 1
    ref = {}
    for t, cnt in seen.items():
        (top, n), total = cnt.most_common(1)[0], sum(cnt.values())
        if n >= 3 and n >= 0.6 * total:   # явный лидер
            ref[t] = top
    return ref


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

    # 0) Список имён: Charles — мужское, Anne — женское
    for c in set(alias.values()):
        g = name_gender(c.split()[0])
        if g:
            votes[c][g] += 6

    titled = defaultdict(set)   # пол титулов персонажа ("Lady Russell" -> {F})

    # 1) Титулы: "Mrs. Hale" -> женщина
    for t in TITLES_F + TITLES_M:
        g = "F" if t in TITLES_F else "M"
        for m in re.finditer(rf"\b{re.escape(t)}\s+([A-Z]\w+)( [A-Z]\w+)?", text):
            w = m.group(1)
            # "Mrs Charles Musgrove" — жена Чарльза, а не женщина по имени Charles.
            # Но "Lady Russell" — Russell тут фамилия, титул верный.
            if m.group(2) and (name_gender(w) or g) != g:
                continue
            if w in alias:
                votes[alias[w]][g] += 3
                titled[alias[w]].add(g)

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
    # Если имя есть в списке имён — он решает (надёжнее догадок по местоимениям)
    out = {c: v.most_common(1)[0][0] for c, v in votes.items()}
    # Исключение: "Lady Russell" — Russell тут фамилия (не стоит перед другой
    # фамилией, как "Charles Musgrove"), и титул говорит обратное -> верим титулу.
    first_names = {v.split()[0] for v in alias.values() if " " in v}
    for c in set(alias.values()):
        g = name_gender(c.split()[0])
        if not g:
            continue
        surname_like = " " not in c and c not in first_names
        if surname_like and titled[c] and g not in titled[c]:
            continue
        out[c] = g
    return out


# =====================================================================
# ШАГ 4. Кто упомянут: имена + местоимения + "her son"
# =====================================================================
def governing_verb(tok):
    t = tok
    while t.head.i != t.i and t.pos_ not in ("VERB", "AUX"):   # .i — номер слова
        t = t.head
    return t


def resolve_mentions(doc, alias, gender, characters, role_map, explicit, title_ref=None):
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

            # --- "the count" -> Monte Cristo ---
            if c is None and title_ref and low in title_ref and tok.i > 0 \
                    and doc[tok.i - 1].lower_ == "the" \
                    and not (tok.i + 1 < len(doc) and doc[tok.i + 1].lower_ == "of"):
                c = title_ref[low]

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
    if not subs and verb.dep_ in ("conj", "xcomp") and verb.head.i != verb.i:
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
            f = forms(t)
            # Романтика — только сам глагол ("loved", "kissed") или роль ("her lover").
            # "arranged Valentine's marriage" — это не роман между ними.
            if f & KEYWORDS["romance"] and t.dep_ not in ("attr", "appos", "oprd", "acomp") \
                    and not (t.lower_ == "love" and t.head.lower_ == "in"):
                continue
            cues.append(f)
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
def relation_type(scores, explicit=None):
    """1) Явная роль ("her son Oliver", общая фамилия) решает сразу.
    2) Иначе — категория с наибольшим числом очков (романтика x2). При равенстве порядок:
       семья > романтика > работа > дружба.
    3) Конфликт побеждает, только если у него строго больше очков:
       начальник может ругать сотрудника, но они всё равно коллеги."""
    order = ["family", "romance", "work", "friendship"]
    if explicit:
        return max(order, key=lambda c: (explicit[c], -order.index(c)))
    # романтические слова встречаются реже дружеских, но значат больше: x2
    scores = Counter(scores)
    scores["romance"] *= 2
    best = max(order, key=lambda c: (scores[c], -order.index(c)))
    if scores["conflict"] > scores[best]:
        return "conflict"
    if scores[best] >= 2:
        return best
    if scores["conflict"] > 0:
        return "conflict"
    if scores[best] > 0:
        return best
    # общаются, но слов-признаков нет: просто знакомые
    return "acquaintance"


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
def analyze(text, min_mentions=2, n_process=1, same_people=None):
    """same_people — один человек под разными именами, например
    [["Edmond Dantès", "Monte Cristo"]]. Можно передать строку "A = B; C = D"."""
    chapters = split_chapters(text)
    nlp.max_length = max(nlp.max_length, max(len(c) for c in chapters) + 1)
    docs = parse_chapters(chapters, n_process)
    alias = find_aliases(docs)
    if isinstance(same_people, str):
        same_people = [g.split("=") for g in re.split(r"[;\n]", same_people) if "=" in g]
    alias = merge_same_people(alias, same_people)

    # сколько раз названо по имени
    mentions = Counter()
    for doc in docs:
        for tok in doc:
            c = name_of(tok, alias)
            if c and not (tok.dep_ == "compound" and name_of(tok.head, alias) == c):
                mentions[c] += 1
    characters = {n for n, c in mentions.items() if c >= min_mentions}

    # Места: "in Bath", "at Kellynch" — если имя часто стоит после таких предлогов
    after_prep = Counter()
    for doc in docs:
        for tok in doc:
            c = name_of(tok, alias)
            if c and ((tok.dep_ == "pobj" and tok.head.lower_ in PLACE_PREPS)
                      or (tok.dep_ == "compound" and tok.head.text in PLACE_WORDS)):
                after_prep[c] += 1     # "in Bath", "Kellynch Hall"
    characters = {c for c in characters if after_prep[c] < 0.3 * mentions[c]}

    # "the count" (без имени) тоже считается упоминанием
    title_ref = title_references(docs, alias, characters)
    for doc in docs:
        for tok in doc:
            if tok.lower_ in title_ref and tok.i > 0 and doc[tok.i - 1].lower_ == "the":
                mentions[title_ref[tok.lower_]] += 1
    gender = guess_genders(docs, alias, text)

    role_map, explicit = {}, []
    pair = defaultdict(lambda: {"by_ch": defaultdict(list), "scores": Counter(),
                                "explicit": Counter(), "direct": 0, "evidence": []})

    for ch_i, doc in enumerate(docs, start=1):
        n_explicit = len(explicit)
        mention = resolve_mentions(doc, alias, gender, characters, role_map, explicit,
                                   title_ref)
        for a, b, cat in explicit[n_explicit:]:        # явные роли: +5
            pair[tuple(sorted((a, b)))]["scores"][cat] += 5
            pair[tuple(sorted((a, b)))]["explicit"][cat] += 1
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

    # Общая фамилия: "Anne Elliot" и "Walter Elliot" — скорее всего семья
    for (a, b), p in pair.items():
        # ("Louisa Musgrove" и "Musgrove" — родители, тоже считаем)
        if (" " in a or " " in b) and a.split()[-1] == b.split()[-1]:
            p["scores"]["family"] += 3
            p["explicit"]["family"] += 1

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
            "type": relation_type(p["scores"], p["explicit"]),
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
TYPE_COLORS = {"family": "#1f77b4", "romance": "#d6336c", "friendship": "#2b9348",
               "work": "#f08c00", "conflict": "#c92a2a", "acquaintance": "#adb5bd"}


def build_graph(rel, max_nodes=15, top_links=3, min_weight=1):
    """Граф для картинки. В книге десятки персонажей и сотни связей, поэтому:
    - только max_nodes главных персонажей (по силе связей);
    - у каждого — только top_links самых сильных связей ("скелет" сети)."""
    G = nx.Graph()
    for _, r in rel.iterrows():
        if r["weight"] >= min_weight:
            G.add_edge(r["a"], r["b"], weight=r["weight"], type=r["type"], tone=r["tone"])
    strength = dict(G.degree(weight="weight"))
    top = sorted(strength, key=strength.get, reverse=True)[:max_nodes]
    G = G.subgraph(top).copy()
    keep = set()
    for v in G:
        best = sorted(G.edges(v, data="weight"), key=lambda e: e[2], reverse=True)[:top_links]
        keep |= {tuple(sorted((a, b))) for a, b, _ in best}
    G.remove_edges_from([e for e in list(G.edges) if tuple(sorted(e)) not in keep])
    nx.set_node_attributes(G, {n: strength[n] for n in G}, "strength")
    return G


def save_graph_png(G, path="graph.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = max(len(G), 1)
    size = 8 + n * 0.35                       # больше персонажей — больше картинка
    fig = plt.figure(figsize=(size, size * 0.72))
    if len(G) == 0:
        plt.axis("off"); plt.savefig(path, dpi=130); plt.close(fig); return
    # Kamada-Kawai раскладывает узлы равномерно (меньше слипания, чем spring)
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=3 / n ** 0.5, iterations=200)
    max_w = max((d["weight"] for *_, d in G.edges(data=True)), default=1)
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=0.8,
                               width=1.5 + 7 * d["weight"] / max_w,
                               edge_color=TYPE_COLORS.get(d["type"], "#adb5bd"),
                               style="dashed" if d["tone"] == "negative" else "solid")
    strength = nx.get_node_attributes(G, "strength")
    max_s = max(strength.values()) or 1
    nx.draw_networkx_nodes(G, pos, node_color="#ffe066", edgecolors="#5c4033",
                           linewidths=1.2,
                           node_size=[500 + 2500 * strength[v] / max_s for v in G])
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    used = sorted({d["type"] for *_, d in G.edges(data=True)})
    handles = [Line2D([0], [0], color=TYPE_COLORS[t], lw=4, label=t) for t in used]
    handles.append(Line2D([0], [0], color="#555", lw=2, ls="--", label="negative tone"))
    # легенда под графом, чтобы не закрывала узлы
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=min(len(handles), 7), frameon=False, fontsize=10)
    plt.axis("off")
    plt.margins(0.1)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


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
