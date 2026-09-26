import json

cases = {
    "B004I5KOM2": {"n":22, "labels": ["acceptable"]*3+["not acceptable"]*1+["acceptable"]+["not acceptable"]+["acceptable"]+["not acceptable"]+["acceptable"]*4+["not acceptable"]+["acceptable"]+["not acceptable"]*6+["borderline"]},
}

# Simpler: just rebuild the 3 known cases manually with exact label order
label_sets = {
    "B004I5KOM2": [ "acceptable","acceptable","acceptable","not acceptable","not acceptable",
        "acceptable","not acceptable","acceptable","not acceptable","acceptable","acceptable",
        "acceptable","acceptable","not acceptable","acceptable","not acceptable","not acceptable",
        "not acceptable","not acceptable","not acceptable","not acceptable","borderline"],
    "B000FA15OU": ["acceptable","acceptable","acceptable","acceptable","not acceptable"],
    "B000JX0A76": ["acceptable","acceptable","acceptable","acceptable","not acceptable","acceptable",
        "acceptable","acceptable","not acceptable","acceptable","acceptable","acceptable","acceptable",
        "acceptable","acceptable","acceptable","acceptable","borderline","acceptable","not acceptable"],
}

for pid, labels in label_sets.items():
    lines = "\n".join(f"- Item {i+1}: {lbl}" for i, lbl in enumerate(labels))
    prompt = (
        f"You are reviewing a batch of {len(labels)} items for product {pid}.\n"
        f"Each item has already been individually checked and labeled:\n\n{lines}\n\n"
        "Write one overall review summarizing this batch for a customer."
    )
    with open(f"data/stripped_prompt_{pid}.txt", "w") as f:
        f.write(prompt)
    print(f"=== {pid} (STRIPPED, no anti-fabrication instruction) ===")
    print(prompt)
    print()
