"""
Оценка: сравниваем ответы системы с ручной разметкой (gold).
Запуск:  py evaluate.py test_story.txt gold_test.csv
gold-файл: колонки a,b,type и (необязательно) tone.
"""
import sys
import pandas as pd
from ties import analyze


def first(name):
    # в gold пишем только имя ("Anna"), система выдаёт "Anna Petrova"
    return name.split()[0]


def key(a, b):
    return tuple(sorted([first(a), first(b)]))


def evaluate(story_path, gold_path, verbose=True):
    _, rel = analyze(open(story_path, encoding="utf-8").read())
    gold = pd.read_csv(gold_path).fillna("")
    if "tone" not in gold.columns:
        gold["tone"] = ""

    pred = {key(r.a, r.b): r for r in rel.itertuples()}
    true = {key(r.a, r.b): r for r in gold.itertuples()}

    # 1) Нашли ли мы правильные пары?
    tp = len(pred.keys() & true.keys())
    precision = tp / len(pred) if pred else 0     # из найденных — сколько верных
    recall = tp / len(true) if true else 0        # из настоящих — сколько нашли
    f1 = 2 * precision * recall / (precision + recall) if tp else 0

    # 2) Правильный ли тип у верно найденных пар?
    common = sorted(pred.keys() & true.keys())
    type_ok = sum(pred[k].type == true[k].type for k in common)

    # 3) Правильный ли тон (только там, где тон размечен)
    tone_keys = [k for k in common if true[k].tone]
    tone_ok = sum(pred[k].tone == true[k].tone for k in tone_keys)

    res = {
        "precision": precision, "recall": recall, "f1": f1,
        "type_acc": type_ok / len(common) if common else 0,
        "tone_acc": tone_ok / len(tone_keys) if tone_keys else 0,
    }
    if verbose:
        print(f"Pairs:  precision={precision:.2f}  recall={recall:.2f}  F1={f1:.2f}")
        print(f"Type accuracy: {res['type_acc']:.2f}  ({type_ok}/{len(common)})")
        print(f"Tone accuracy: {res['tone_acc']:.2f}  ({tone_ok}/{len(tone_keys)})")
        print("\nErrors:")
        for k in sorted(pred.keys() - true.keys()):
            print(f"  EXTRA   {k}: system found '{pred[k].type}', but there is no real relation")
        for k in sorted(true.keys() - pred.keys()):
            print(f"  MISSED  {k}: should be '{true[k].type}'")
        for k in common:
            if pred[k].type != true[k].type:
                print(f"  WRONG TYPE  {k}: system '{pred[k].type}', correct '{true[k].type}'")
            if true[k].tone and pred[k].tone != true[k].tone:
                print(f"  WRONG TONE  {k}: system '{pred[k].tone}', correct '{true[k].tone}'")
    return res


if __name__ == "__main__":
    story = sys.argv[1] if len(sys.argv) > 1 else "test_story.txt"
    gold_path = sys.argv[2] if len(sys.argv) > 2 else "gold_test.csv"
    evaluate(story, gold_path)
