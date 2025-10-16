import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('session_data.csv')

df = pd.DataFrame(data)

label_encoder = LabelEncoder()
df['winner_encoded'] = label_encoder.fit_transform(df['session_winner'])

features = ['inning_number', 'session_number', 'runs_scored', 'wickets_fallen']
X = df[features]
y = df['winner_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200,
                             max_features='sqrt',
                             max_depth=16,
                             min_samples_leaf=5,
                             bootstrap=True,
                             n_jobs=1,
                             random_state=42,
                             oob_score=True
                             )
clf.fit(X_train, y_train)

oob_score = clf.oob_score_
print(f'oob score: {oob_score}')
accuracy = clf.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

samples = np.array([
    [1, 1, 120, 2],
    [1, 2, 75, 5],
    [2, 1, 90, 3],
    [2, 2, 150, 4],
])

new_input = [[1, 1, 120, 2]]

predictions = clf.predict(new_input)

predicted_labels = label_encoder.inverse_transform(predictions)
print(predicted_labels)
