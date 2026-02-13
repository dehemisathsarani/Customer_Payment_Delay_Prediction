import pickle

# Load feature names
feature_names = pickle.load(open('Model/feature_names.pkl', 'rb'))
print(f'Total features: {len(feature_names)}')
print('\nAll feature names:')
for i, name in enumerate(feature_names[:50]):  # Show first 50
    print(f'{i+1}. {name}')
