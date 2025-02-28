import json
from pathlib import Path
import random


def sample():
    rules = {
        "S": [["NP", "VP"]],
        "NP": [["Det", "NP2"]],
        "NP2": [["Adj", "NP3"], ["Noun"]],
        "NP3": [["Noun", "RC"], ["Noun"]],
        "RC": [["Rel", "VP"]],
        "VP": [["Vt", "NP"], ["Vi"]],
        "Rel": ["who"],
        "Noun": ["cat", "dog", "mouse"],
        "Det": ["the", "a"],
        "Adj": ["sad", "scholarly", "wizened", "cowardly"],
        "Vt": ["chased", "ate", "befriended", "spurned"],
        "Vi": ["sulked", "smiled", "sighed"],
    }
    sent = ["S"]
    while any(w in rules for w in sent):
        new_sent = []
        for w in sent:
            if w in rules:
                pick = random.choice(rules[w])
                if type(pick) == str:
                    new_sent.append(pick)
                else:
                    new_sent += pick
            else:
                new_sent.append(w)
        sent = new_sent
    return " ".join(sent)

def generate(num_sentences=1000, output_dir="data"):
    random.seed(0)
    out = []
    for _ in range(num_sentences):
        out.append({"text": sample()})
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)
    fn = Path(output_dir) / "toy_cfg_sentences.json"
    print(f"Writing {len(out)} sentences to {fn}")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    generate()
