"""Analyse des habitudes d'achat d'un supermarché : règles d'association avec Apriori.

L'algorithme Apriori est implémenté ici "from scratch" (sans mlxtend) :
1. recherche des itemsets fréquents (support >= min_support) niveau par niveau,
2. génération des règles A -> B filtrées par confiance et lift.
"""
import argparse
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/market_basket.csv")
    p.add_argument("--min-support", type=float, default=0.005)
    p.add_argument("--min-confidence", type=float, default=0.2)
    p.add_argument("--min-lift", type=float, default=1.2)
    p.add_argument("--top", type=int, default=10, help="Nombre de règles affichées")
    return p.parse_args()


# ----------------------------------------------------------------------------
# 1. Chargement et nettoyage
# ----------------------------------------------------------------------------
def load_transactions(path):
    """Retourne une liste de transactions (chaque transaction = ensemble de produits).

    Formats gérés :
    - "large" : une ligne = un panier, produits dans les colonnes, sans en-tête
      (ex. Market_Basket_Optimisation.csv)
    - "long"  : colonnes Member_number, Date, itemDescription
      (ex. Groceries_dataset.csv)
    """
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()

    if "itemDescription" in first_line:
        df = pd.read_csv(path)
        df["itemDescription"] = df["itemDescription"].astype(str).str.strip().str.lower()
        baskets = df.groupby(["Member_number", "Date"])["itemDescription"].apply(set)
        transactions = [set(b) for b in baskets]
    else:
        transactions = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                basket = {x.strip().lower() for x in row if x.strip()}
                if basket:
                    transactions.append(basket)
    return transactions


# ----------------------------------------------------------------------------
# 2. Algorithme Apriori
# ----------------------------------------------------------------------------
def apriori(transactions, min_support):
    """Retourne {itemset (frozenset): support} pour tous les itemsets fréquents."""
    n = len(transactions)
    min_count = min_support * n

    counts = Counter(item for t in transactions for item in t)
    current = {frozenset([i]): c for i, c in counts.items() if c >= min_count}
    frequent = dict(current)

    k = 2
    while current:
        prev = set(current)
        # Génération des candidats : union de deux itemsets de taille k-1,
        # gardée seulement si tous ses sous-ensembles de taille k-1 sont fréquents.
        candidates = set()
        for a, b in combinations(prev, 2):
            union = a | b
            if len(union) == k and all(frozenset(s) in prev for s in combinations(union, k - 1)):
                candidates.add(union)

        # Comptage : pour chaque panier, on ne teste que les combinaisons de produits fréquents.
        freq_items = {i for s in prev for i in s}
        cand_counts = Counter()
        for t in transactions:
            items = sorted(t & freq_items)
            if len(items) < k:
                continue
            for combo in combinations(items, k):
                fs = frozenset(combo)
                if fs in candidates:
                    cand_counts[fs] += 1

        current = {s: c for s, c in cand_counts.items() if c >= min_count}
        frequent.update(current)
        k += 1

    return {s: c / n for s, c in frequent.items()}


# ----------------------------------------------------------------------------
# 3. Règles d'association
# ----------------------------------------------------------------------------
def association_rules(frequent, min_confidence, min_lift):
    rows = []
    for itemset, supp in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ant in combinations(itemset, r):
                ant = frozenset(ant)
                cons = itemset - ant
                confidence = supp / frequent[ant]
                lift = confidence / frequent[cons]
                if confidence >= min_confidence and lift >= min_lift:
                    rows.append({
                        "antecedent": ", ".join(sorted(ant)),
                        "consequent": ", ".join(sorted(cons)),
                        "support": supp,
                        "confidence": confidence,
                        "lift": lift,
                    })
    rules = pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "confidence", "lift"])
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. Graphiques
# ----------------------------------------------------------------------------
def plot_top_items(transactions, n_top=15, path="top_items.png"):
    counts = Counter(item for t in transactions for item in t)
    items, values = zip(*counts.most_common(n_top))
    plt.figure(figsize=(8, 6))
    plt.barh(items[::-1], values[::-1])
    plt.xlabel("Nombre de paniers contenant le produit")
    plt.title(f"Top {n_top} des produits les plus achetés")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_rules(rules, path="rules_scatter.png"):
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support")
    plt.ylabel("Confiance")
    plt.title("Règles d'association (support vs confiance)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def to_markdown(rules):
    lines = ["| Si le client achète | Il achète aussi | Support | Confiance | Lift |",
             "|---|---|---|---|---|"]
    for r in rules.itertuples():
        lines.append(f"| {r.antecedent} | {r.consequent} | {r.support:.3f} | {r.confidence:.2f} | {r.lift:.2f} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    transactions = load_transactions(args.data)
    n_items = len({i for t in transactions for i in t})
    print(f"{len(transactions)} paniers, {n_items} produits distincts")

    frequent = apriori(transactions, args.min_support)
    print(f"{len(frequent)} itemsets fréquents (support >= {args.min_support})")

    rules = association_rules(frequent, args.min_confidence, args.min_lift)
    print(f"{len(rules)} règles (confiance >= {args.min_confidence}, lift >= {args.min_lift})")
    if rules.empty:
        print("Aucune règle : baisse --min-support, --min-confidence ou --min-lift.")
        return

    Path("outputs").mkdir(exist_ok=True)
    rules.to_csv("outputs/rules.csv", index=False)

    top = rules.head(args.top)
    print(f"\n=== Top {args.top} règles (triées par lift) ===")
    print(top.round(3).to_string(index=False))

    md = to_markdown(top)
    Path("outputs/top_rules.md").write_text(md, encoding="utf-8")
    print("\n=== Tableau prêt à coller dans le README ===")
    print(md)

    plot_top_items(transactions)
    plot_rules(rules)
    print("\nGraphiques : top_items.png, rules_scatter.png")
    print("Fichiers : outputs/rules.csv, outputs/top_rules.md")


if __name__ == "__main__":
    main()
