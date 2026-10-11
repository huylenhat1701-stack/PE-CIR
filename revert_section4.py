# -*- coding: utf-8 -*-
"""Script to revert Section 4 and Conclusion in paper/experiments.tex back to the exact
state before expanding Section 4 and adding Section 5 Discussion."""

import re

# Read current experiments.tex
tex_path = r'C:\New folder (3)\PE-CIR\paper\experiments.tex'
with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Read the old texts from expand_part1, expand_part2, build_full_paper
from expand_part1 import old_sec42_text, new_sec42_text
from expand_part2 import old_sec43, new_sec43
from build_full_paper import old_target, replacement_text

# Revert build_full_paper changes (Section 4.5 additions + Section 5 Discussion + Section 6 Conclusion)
if replacement_text in text:
    text = text.replace(replacement_text, old_target)
    print("Reverted build_full_paper replacement successfully.")
else:
    print("WARNING: replacement_text not found in text.")

# Revert expand_part2 changes (Section 4.3 expansion + Table 4)
if new_sec43 in text:
    text = text.replace(new_sec43, old_sec43)
    print("Reverted expand_part2 replacement successfully.")
else:
    print("WARNING: new_sec43 not found in text.")

# Revert expand_part1 changes (Section 4.2 expansion)
if new_sec42_text in text:
    text = text.replace(new_sec42_text, old_sec42_text)
    print("Reverted expand_part1 replacement successfully.")
else:
    print("WARNING: new_sec42_text not found in text.")

with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Reversion to original Section 4 completed.")
