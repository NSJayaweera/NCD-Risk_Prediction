
import json
import re

notebook_path = 'c:/Users/Isum Enuka/Downloads/osteoporosis-risk-prediction/notebooks/MASTER_Complete_Pipeline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Replacements for Markdown Headers ONLY
header_replacements = [
    (r"PART 7: HYPERPARAMETER", "PART 6: HYPERPARAMETER"),
    (r"PART 8: CONFUSION", "PART 7: CONFUSION"),
    (r"PART 9: SHAP", "PART 8: SHAP"),
    (r"PART 10: LOSS", "PART 9: LOSS"),
    # Leaderboard is already PART 10, no change needed
]

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        new_source = []
        for line in source:
            new_line = line
            # Check if line is a header
            if line.strip().startswith("#"):
                for pattern, replacement in header_replacements:
                    if re.search(pattern, new_line):
                        new_line = re.sub(pattern, replacement, new_line)
            new_source.append(new_line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook headers corrected successfully.")
