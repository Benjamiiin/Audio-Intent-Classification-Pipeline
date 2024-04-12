
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import librosa 
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

#===============================================================#

dev_path = "C:/Users/benya/Desktop/dsl_data/development.csv"
df = pd.read_csv(dev_path)   

y = df['action'] + df['object']

def process_audio(path):
    Wav, sample_rate = librosa.load(path, sr=16000)
    audio_trim, _ = librosa.effects.trim(Wav, top_db=20)
    p_add = 180000 - len(audio_trim)
    padding = np.pad(audio_trim, (0, p_add), mode='constant')
    mfccs = librosa.feature.mfcc(y=padding, sr=sample_rate)
    return mfccs.flatten()

X_processed = np.array(list(map(process_audio, df['path'])))


#===============================================================#

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y)

#===============================================================#

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#===============================================================#

pca = PCA(n_components=0.97)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


#===============================================================#
#                   RandomForest+SVM
#===============================================================#
clf = RandomForestClassifier()
param_grid = {'n_estimators': [420, 650, 700],
               'max_depth': [33, 46, 50]}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

svm = SVC()
svm_param_grid = {'C': [5, 35, 420], 'kernel': ['linear', 'rbf']}
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5)
svm_grid_search.fit(X_train, y_train)
print("Best parameters for SVM: ", svm_grid_search.best_params_)
print("Best score for SVM: ", svm_grid_search.best_score_)

clf_rf = RandomForestClassifier(**grid_search.best_params_)
clf_svm = SVC(**svm_grid_search.best_params_)
 
ensemble_clf = VotingClassifier(estimators=[('rf', clf_rf), ('svm', clf_svm)], voting='hard')
ensemble_clf.fit(X_train, y_train)

ypredict = ensemble_clf.predict(X_test)
acc = accuracy_score(y_test, ypredict) * 100
p, r, f1, s = precision_recall_fscore_support(y_test, ypredict)
print(f"accuracy ensemble classifier: {acc}%")
print(f"Mean precision ensemble classifier= {p.mean() * 100} %")
print(f"Mean recall ensemble classifier = {r.mean() * 100} %")
print(f"Mean f1score ensemble classifier = {f1.mean() * 100} %")

conf_mat_ens = confusion_matrix(y_test, ypredict)
print("CONFUSION MATRIX ENSEMBLE= ")
print(conf_mat_ens)


#===============================================================#
#===============================================================#
#===============================================================#
#===============================================================#
#===============================================================#
#===============================================================#
#===============================================================#

ev_path = "C:/Users/benya/Desktop/dsl_data/evaluation.csv"
df2 = pd.read_csv(ev_path)  
Id = df2.iloc[:,0].values 
z_processed = []
 
def process_audio(path):
    Wav, sample_rate = librosa.load(path, sr=16000)
    audio_trim, _ = librosa.effects.trim(Wav, top_db=20)
    p_add = 180000 - len(audio_trim)
    padding = np.pad(audio_trim, (0, p_add), mode='constant')
    mfccs = librosa.feature.mfcc(y=padding, sr=sample_rate)
    return mfccs.flatten()

z_processed = np.array(list(map(process_audio, df2['path'])))

z_processed = scaler.transform(z_processed)
z_processed = pca.transform(z_processed)

zpredict = ensemble_clf.predict(z_processed)

evv = pd.DataFrame({'Id': Id, 'Predicted': zpredict})
evv.to_csv('output1.csv', index=False)

