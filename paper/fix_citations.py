import re

# Read the bibliography file
with open('reunity_complete.bib', 'r') as f:
    content = f.read()

# Find all entries with fake DOI patterns (1234567, 0000000, etc.)
fake_patterns = [
    r'1234567',
    r'0000000',
    r'234567\.',
]

# Split into individual entries
entries = re.split(r'(@\w+\{)', content)

# Reconstruct keeping only valid entries
valid_entries = []
removed_count = 0
removed_keys = []

i = 0
while i < len(entries):
    if entries[i].startswith('@'):
        entry_type = entries[i]
        if i + 1 < len(entries):
            entry_body = entries[i + 1]
            full_entry = entry_type + entry_body
            
            # Check if this entry has a fake DOI
            is_fake = False
            for pattern in fake_patterns:
                if pattern in full_entry:
                    is_fake = True
                    # Extract the key
                    key_match = re.match(r'([^,]+),', entry_body)
                    if key_match:
                        removed_keys.append(key_match.group(1))
                    break
            
            if not is_fake:
                valid_entries.append(full_entry)
            else:
                removed_count += 1
            i += 2
        else:
            i += 1
    else:
        if entries[i].strip():
            valid_entries.append(entries[i])
        i += 1

# Write the cleaned bibliography
with open('reunity_complete_cleaned.bib', 'w') as f:
    f.write('\n'.join(valid_entries))

print(f"Removed {removed_count} fake citations")
print(f"Removed keys: {removed_keys[:20]}...")  # Show first 20
