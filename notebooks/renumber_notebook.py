
import json
import re

notebook_path = 'c:/Users/Isum Enuka/Downloads/osteoporosis-risk-prediction/notebooks/MASTER_Complete_Pipeline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new section mappings
section_map = {
    "PART 1": "PART 1",
    "PART 2": "PART 2",
    "PART 3": "PART 3",
    "PART 4": "PART 4",
    "PART 4.5": "PART 5",  # Gender-Specific Models -> PART 5
    "PART 5": "PART 6",    # Hyperparameter Tuning -> PART 6
    "PART 6": "PART 7",    # Confusion Matrices -> PART 7
    "PART 7": "PART 8",    # SHAP Analysis -> PART 8
    "PART 8": "PART 9",    # Loss Curves -> PART 9
    "PART 9": "PART 10",   # Leaderboard -> PART 10
    "PART 10": "PART 10"   # Just in case
}

# Regex for finding "PART X" in markdown or "SECTION X.Y" in code
part_pattern = re.compile(r"(PART\s+)(\d+(\.\d+)?)")
section_pattern = re.compile(r"(SECTION\s+)(\d+)(\.\d+)?")

# Helper to renumber PART X -> PART Y
def replace_part(match):
    prefix = match.group(1)
    old_num = match.group(2)
    key = f"PART {old_num}"
    if key in section_map:
        return section_map[key]
    return match.group(0) # No change if not in map

# Helper to renumber SECTION 4.5.X -> SECTION 5.X, SECTION 5.X -> SECTION 6.X, etc.
def replace_section(match):
    prefix = match.group(1)      # "SECTION "
    major = match.group(2)       # "4" or "5" etc.
    minor_suffix = match.group(3) if match.group(3) else "" # ".1", ".5", etc.
    
    # Logic:
    # If major is 4 and minor starts with .5 (e.g., 4.5.1), mapping to 5.X is tricky unless we reset subsections.
    # Actually, simpler logic based on the specific content we saw:
    # 4.5.x -> 5.x (Gender Specific)
    # 5.x -> 6.x (Hyperparameter Tuning)
    # 6.x -> 7.x (Confusion Matrices)
    # 7.x -> 8.x (SHAP)
    # 8.x -> 9.x (Loss Curves)
    # 9.x -> 10.x (Leaderboard)

    # We need to look at the full string to decide, but regex only gives us the number.
    # Let's try to map strictly based on the major/minor numbers found in the file.
    
    full_num_str = f"{major}{minor_suffix}"
    
    if full_num_str.startswith("4.5"):
        # Map 4.5.x -> 5.x
        # e.g. 4.5.1 -> 5.1
        # remove "4.5" replace with "5"
        return f"SECTION {full_num_str.replace('4.5', '5', 1)}"
    
    elif major == "5":
        # 5.x -> 6.x
        return f"SECTION {full_num_str.replace('5', '6', 1)}"
        
    elif major == "6":
        return f"SECTION {full_num_str.replace('6', '7', 1)}"
        
    elif major == "7":
        return f"SECTION {full_num_str.replace('7', '8', 1)}"

    elif major == "8":
        return f"SECTION {full_num_str.replace('8', '9', 1)}"

    elif major == "9":
        return f"SECTION {full_num_str.replace('9', '10', 1)}"
        
    return match.group(0)

# Iterate over cells
for cell in nb['cells']:
    source = cell['source']
    new_source = []
    
    for line in source:
        # Update Markdown Headers (PART X)
        if cell['cell_type'] == 'markdown':
            # Specific fix for the Table of Contents lines
            if "|" in line and "PART" in line:
                 # It's a table row, verify specific replacements manually for safety or usage regex
                 # | **PART 4.5** | Gender-Specific XGBoost Models | 15-20 min |
                 # -> | **PART 5** | ...
                 new_line = part_pattern.sub(replace_part, line)
                 new_source.append(new_line)
            elif line.strip().startswith("#"):
                 # It's a header
                 new_line = part_pattern.sub(replace_part, line)
                 new_source.append(new_line)
            # Fix text references like "This master notebook combines all 10..." or list items
            elif re.search(r"^\d+\.\s+.*PART", line):
                 # e.g. "4.5. Gender-Specific..." in a list
                 # This might be tricky, let's stick to "PART X" replacements generally in text too
                 new_line = part_pattern.sub(replace_part, line)
                 new_source.append(new_line)
            else:
                 # General text replacement for "PART X"
                 new_line = part_pattern.sub(replace_part, line)
                 # Also update the list in the Intro cell (1. ... 4. ... 4.5. ...)
                 if "4.5. " in new_line and "**Gender-Specific" in new_line:
                     new_line = new_line.replace("4.5. ", "5. ")
                 elif re.match(r"^\d+\.\s+.*", new_line.strip()) and "**" in new_line:
                     # Renumber the intro list items if they start with 5, 6, 7, 8, 9
                     # 5. -> 6.
                     # 6. -> 7.
                     # etc.
                     # This is fragile with regex, let's look for specific strings
                     if "5. " in new_line and "Hyperparameter" in new_line:
                         new_line = new_line.replace("5. ", "6. ")
                     elif "6. " in new_line and "Confusion" in new_line:
                         new_line = new_line.replace("6. ", "7. ")
                     elif "7. " in new_line and "SHAP" in new_line:
                         new_line = new_line.replace("7. ", "8. ")
                     elif "8. " in new_line and "Loss" in new_line:
                         new_line = new_line.replace("8. ", "9. ")
                     elif "9. " in new_line and "Leaderboard" in new_line:
                         new_line = new_line.replace("9. ", "10. ")
                 
                 new_source.append(new_line)

        # Update Code Comments (SECTION X.Y)
        elif cell['cell_type'] == 'code':
            # Look for comments like "# SECTION 4.5.1"
            if "SECTION" in line:
                new_line = section_pattern.sub(replace_section, line)
                
                # Manual fix for the print statements in code that say "PART 4.5" or similar?
                # The code usually prints things too.
                # e.g. print('PART 4.5 ...')
                if "PART" in new_line:
                     new_line = part_pattern.sub(replace_part, new_line)
                
                new_source.append(new_line)
            else:
                new_source.append(line)
        else:
            new_source.append(line)
            
    cell['source'] = new_source

# Rewrite the file
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1) # start with small indentation to minimize diff size if possible, typical ipynb uses 1 or 2

print("Notebook renumbered successfully.")
