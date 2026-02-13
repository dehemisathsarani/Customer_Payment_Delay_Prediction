import pickle

scaler = pickle.load(open('Model/payment_delay_scaler.pkl', 'rb'))
print('Expected features:', scaler.n_features_in_)

# Also check feature names if available
try:
    feature_names = pickle.load(open('Model/feature_names.pkl', 'rb'))
    print('Number of feature names:', len(feature_names))
    print('First 10 feature names:', feature_names[:10])
except Exception as e:
    print('Error loading feature names:', e)
