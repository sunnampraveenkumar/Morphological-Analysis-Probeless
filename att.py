import pickle

# Load your data (assuming it's pickled)
with open('pickles/UM/bert/heb/train_parsed.pkl', 'rb') as f:
    data = pickle.load(f)

# Set to store all unique attributes
unique_attributes = set()

# Iterate through the data
for entry in data:
    # Assuming 'attributes' is the key where attributes are stored in each entry
    attributes = entry.get('attributes', {})
    
    # Add the keys (attribute names) to the unique_attributes set
    for attr in attributes.keys():
        unique_attributes.add(attr)

# Print all unique attributes
print("Unique attributes in the data:")
for attr in sorted(unique_attributes):
    print(attr)