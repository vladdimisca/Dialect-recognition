import numpy as np
import csv
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer


# functie care retine id-urile pentru fiecare text dintr-un fisier
def getIds(data):
    index = {}

    for id_line, document in enumerate(data):
        for id_w, word in enumerate(document.split()):
            if id_w == 0:
                index[id_line] = word
            else:
                break

    return index


# folosesc tf-idf din sklearn pentru a retine datele normalizate
# si a le avea in ordine descrescatoare dupa frecventa de aparitie a fiecarui feature
def TfIdf():
    train_texts = []

    file_names = ['train_samples.txt', 'validation_samples.txt']

    # folosesc textele din train si textele din validation pentru extragerea de feature-uri
    # adaug linie cu linie textele din fisier intr-o lista de stringuri, fara id-urile din fata textelor
    for fname in file_names:
        with open(fname, "r", encoding="utf-8") as infile:
            for line in infile:
                text = line.split(None, 1)[1]
                train_texts.append(text)
            infile.close()

    test_texts = []

    # deschid fisierul cu texte pentru test si adaug textele intr-o lista de stringuri fara id-ul de la inceput
    ftest = open("test_samples.txt", "r", encoding="utf-8")
    test_data = ftest.readlines()
    test_indexes = getIds(test_data)

    for line in test_data:
        text = line.split(None, 1)[1]
        test_texts.append(text)

    # calculez valorile tf-idf si normalizez cele doua array-uri
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(4, 6))

    train_vector = vectorizer.fit_transform(train_texts)
    test_vector = vectorizer.transform(test_texts)

    return train_vector, test_vector, test_indexes


# functie pentru obtinerea label-urilor textelor din fisierele de train si validare
def getTrainLabels():
    file_names = ['train_labels.txt', 'validation_labels.txt']

    length = 0

    # calculez numarul de label-uri din cele doua fisiere
    for fname in file_names:
        with open(fname, "r", encoding="utf-8") as infile:
            length += len(infile.readlines())
            infile.close()

    labels = np.zeros(length, dtype='uint8')

    id_line = 0

    # pentru fiecare linie din fisierul de train/validare, adaug in array label-ul corespunzator, neglijand id-ul
    for fname in file_names:
        with open(fname, "r", encoding="utf-8") as infile:
            for line in infile:
                for id_w, word in enumerate(line.split()):
                    if id_w == 1:
                        labels[id_line] = word
                        id_line += 1
            infile.close()

    return labels


# extrag label-urile de training
training_labels = getTrainLabels()

# extrag feature-urile si scalez datele
scaled_train_data, scaled_test_data, test_index = TfIdf()
print("A scalat datele!")

# creez modelul de SVC
svm_model = svm.LinearSVC(C=0.5)
print("A creat modelul!")

# antrenez clasificatorul
svm_model.fit(scaled_train_data, training_labels)
print("A antrenat!")

# calculez vectorul cu predictii pentru datele de test
predicted_labels_svm = svm_model.predict(scaled_test_data)
print("A prezis!")

# scriu fisierul csv cu predictiile corespunzatoare fiecarui id
with open('submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for id_row, prediction in enumerate(predicted_labels_svm):
        writer.writerow([test_index[id_row], prediction])
