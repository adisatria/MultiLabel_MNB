import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import re, nltk, csv
import numpy as np
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer

import pandas
from sklearn.model_selection import ShuffleSplit

l_tanggapan_positif = []
l_tanggapan_negatif = []
factory1 = StopWordRemoverFactory()
stopword = factory1.create_stop_word_remover()

factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

akurasi_list = []


def preprocessing(dataset):
    teks = re.sub('(\d)+(\.)*(\d)*', '', dataset)  # hapus digit
    teks = re.sub('[/+@.,%-%^*"!#-$-\']', '', teks)  # hapus simbol

    # stopword removal
    prestopword = stopword.remove(teks)
    # teks = ' '.join([word for word in teks.split() if word not in prestopword])  # clearstopword

    # stemming
    hasilstem = stemmer.stem(prestopword)
    # teks = [stemmer.stem(word) for word in teks.split(" ")]

    tokenProcess = word_tokenize(hasilstem)
    teks = tokenProcess

    return teks


def preprocessing_uselist(listname):
    listteks = []
    for i in range(len(listname)):
        teks = re.sub('(\d)+(\.)*(\d)*', '', listname[i])  # hapus digit
        teks = re.sub('[/+@.,%-%^*"!#-$-\']', '', teks)  # hapus simbol

        # stopword removal
        prestopword = stopword.remove(teks)
        # teks = ' '.join([word for word in teks.split() if word not in prestopword])  # clearstopword

        # stemming
        hasilstem = stemmer.stem(prestopword)
        # teks = [stemmer.stem(word) for word in teks.split(" ")]

        tokenProcess = word_tokenize(hasilstem)
        listteks.extend(tokenProcess)

    return listteks


def preprocessing_uselist_test(listname):
    listteks = []
    for i in range(len(listname)):
        teks = re.sub('(\d)+(\.)*(\d)*', '', listname[i])  # hapus digit
        teks = re.sub('[/+@.,%-%^*"!#-$-\']', '', teks)  # hapus simbol

        # stopword removal
        prestopword = stopword.remove(teks)
        # teks = ' '.join([word for word in teks.split() if word not in prestopword])  # clearstopword

        # stemming
        hasilstem = stemmer.stem(prestopword)
        # teks = [stemmer.stem(word) for word in teks.split(" ")]

        tokenProcess = word_tokenize(hasilstem)
        listteks.append(tokenProcess)

    return listteks


def bacafile(filename):
    semua = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if row:
                HasilPrepro = preprocessing(row[0])
                semua.append(HasilPrepro)
            else:
                continue
    return semua


def bacafile_uselist(listname):
    semua = []
    for i in range(len(listname)):
        if i:
            HasilPrepro = preprocessing(listname[i])
            semua.append(HasilPrepro)
        else:
            continue
    return semua


def pisah_list_kelas(listname):  # ganti jadi untuk membaca list /sudah
    for i in range(len(listname)):
        if listname[i][1] == 0:
            l_tanggapan_negatif.append(listname[i][0])
        elif listname[i][1] == 1:
            l_tanggapan_positif.append(listname[i][0])


def join_words(listname):
    sentence = ""
    for word in listname:
        sentence += word + " "
    return sentence


def hitungkatadikelas(sentence):
    l_jumlah = []
    frequency = {}
    match_pattern = re.findall(r'\b[a-z]{3,15}\b', sentence)

    for word in match_pattern:
        count = frequency.get(word, 0)
        frequency[word] = count + 1  # count(w,c)

    frequency_list = frequency.keys()

    for words in frequency_list:
        insert = words, frequency[words] + 1
        l_jumlah.append(tuple(insert))

    return l_jumlah


def conditional_probabilistist(list_kata, jum_words, detailed_words, jum_vocab):  # p(w|c) untuk semua kata pada kelas
    for i in range(len(list_kata)):
        p_w_c = list_kata[i][1] / (jum_words + jum_vocab)
        words = list_kata[i] + (round(p_w_c, 5),)
        detailed_words.append(tuple(words))


def bow(listname):
    listmatrix = []
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(listname).todense()  # untuk tanggapan ada atau tidaknnya pada bag
    # print(matrix)
    listmatrix.append(matrix)
    vocab = vectorizer.vocabulary_
    # print(vocab)
    return vocab


def training_sentiment(data_train, tang_train, data_test, sent_test):
    n_tanggapan = 0
    n_kelas_negatif_in_all = 0
    n_kelas_positif_in_all = 0
    detailed_words_positif = []
    detailed_words_negatif = []
    print("Preprocessing............")
    # print("list tanggapan positif = ", l_tanggapan_positif)
    result_prepro_positif = preprocessing_uselist(l_tanggapan_positif)
    # print("=================================Prepro negatif DONE====================================")
    result_prepro_negatif = preprocessing_uselist(l_tanggapan_negatif)
    # print("=================================Prepro positif DONE====================================")
    # untuk bow, semua kalimat di satukan dan menjadi satu satu kata dalam list
    # print(tang_train)
    result_prepro_for_bow = preprocessing_uselist(tang_train)
    # print("=================================Prepro BOW DONE====================================")
    # print(result_prepro_for_bow)
    dict_vocab = bow(result_prepro_for_bow)
    # Prior P(c)
    # komponen Prior
    n_kelas_negatif_in_all = len(l_tanggapan_negatif)
    n_kelas_positif_in_all = len(l_tanggapan_positif)
    n_tanggapan = n_kelas_negatif_in_all + n_kelas_positif_in_all
    print("Training............")
    print("SUM OF OPINION = ", n_tanggapan)
    print("SUM OF NEGATIVE OPINION = ", n_kelas_negatif_in_all)
    print("SUM OF POSITIVE OPINION = ", n_kelas_positif_in_all)
    prior_positif = n_kelas_negatif_in_all / n_tanggapan  # P(positif)
    prior_negatif = n_kelas_positif_in_all / n_tanggapan  # P(negatif)
    print("P(Positif) = ", prior_positif)
    print("P(Negative) = ", prior_negatif)
    # Komponen conditional Probabilistic P(w|c)
    sentence_positif = (join_words(result_prepro_positif))
    sentence_negatif = (join_words(result_prepro_negatif))
    list_jum_kata_positif = hitungkatadikelas(sentence_positif)
    list_jum_kata_negatif = hitungkatadikelas(sentence_negatif)
    jum_words_positif = len(list_jum_kata_positif)
    jum_words_negatif = len(list_jum_kata_negatif)
    jum_vocab = len(dict_vocab)
    print("Count(Positif) = ", jum_words_positif)
    print("Count(Negatif) = ", jum_words_negatif)
    print("|V| = ", jum_vocab)
    # outputnya = (w, count(w,c)+1 , P(w|c))
    conditional_probabilistist(list_jum_kata_positif, jum_words_positif, detailed_words_positif,
                               jum_vocab)  # p(w|positif)
    conditional_probabilistist(list_jum_kata_negatif, jum_words_negatif, detailed_words_negatif,
                               jum_vocab)  # p(w|negatif)
    # print("Detailed Positif = ", detailed_words_positif) #list p(w|positif) semua w
    # print("Detailed Negatif = ", detailed_words_negatif) #list p(w|negatif) semua w

    # testing
    print("Testing............")
    testing_sentiment(data_test, prior_positif, prior_negatif, detailed_words_positif, detailed_words_negatif,
                      sent_test)


def testing_sentiment(data_test, prior_pos, prior_neg, detailed_words_pos, detailed_words_neg, sent_test):
    choosingclass = []
    result_sent_test = []
    allpwcpos = 1
    allpwcneg = 1
    chooseclasspos = 1
    chooseclassneg = 1
    tanggapan_test = preprocessing_uselist_test(data_test)
    for j in range(len(tanggapan_test)):
        if (len(tanggapan_test[j]) != 0):
            # look for all P(w|c)
            for k in range(len(tanggapan_test[j])):
                for l in range(len(detailed_words_pos)):
                    if (tanggapan_test[j][k] == detailed_words_pos[l][0]):
                        allpwcpos = allpwcpos * detailed_words_pos[l][2]
                    else:
                        continue
            for m in range(len(tanggapan_test[j])):
                for n in range(len(detailed_words_neg)):
                    if (tanggapan_test[j][m] == detailed_words_neg[n][0]):
                        allpwcneg = allpwcneg * detailed_words_neg[n][2]
                    else:
                        continue
            # 0 condition
            if (allpwcpos == 1):
                chooseclasspos = 0
                chooseclassneg = prior_neg * allpwcneg
            elif (allpwcneg == 1):
                chooseclasspos = prior_pos * allpwcpos
                chooseclassneg = 0
            else:
                chooseclasspos = prior_pos * allpwcpos
                chooseclassneg = prior_neg * allpwcneg
            # choosing class
            if (chooseclasspos > chooseclassneg):
                detailed_test = tanggapan_test[j], round(chooseclasspos, 5), 'POSITIVE'
                result_sent_test.append('1')
            elif (chooseclasspos < chooseclassneg):
                detailed_test = tanggapan_test[j], round(chooseclassneg, 5), 'NEGATIVE'
                result_sent_test.append('0')

            choosingclass.append(tuple(detailed_test))
            # reset the component
            allpwcpos = 1
            allpwcneg = 1
            chooseclasspos = 1
            chooseclassneg = 1
        else:
            detailed_test = tanggapan_test[j], 0, 'POSITIVE'
            result_sent_test.append('1')
            choosingclass.append(tuple(detailed_test))

    result_sent_test = list(map(int, result_sent_test))
    akurasi_list.append(accuracy_score(sent_test, result_sent_test))


def main():
    akurasi_list_I = []
    akurasi_list_SI = []
    print("Read Data............")
    before_all_data = pandas.read_csv("PusatBahasaP5000.csv", )
    # before_all_data = pandas.read_csv("data/data_50.csv", )
    print("Delete empty opinion............")
    df = pandas.DataFrame(before_all_data)
    all_data = df.dropna()
    ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    print("Cross Validation (9:1)............")
    for train_index, test_index in ss.split(all_data):
        all_train_data_I = []
        all_train_data_SI = []
        tanggapan_test = []
        tanggapan_train = []
        sentiment_test_I = []
        sentiment_test_SI = []
        for i in range(len(train_index)):
            train_data_SI = all_data["Tanggapan"].values[train_index[i]], all_data["L_Layanan"].values[train_index[i]]
            train_data_I = all_data["Tanggapan"].values[train_index[i]], all_data["L_Fasilitas"].values[train_index[i]]
            tanggapan_train.append(all_data["Tanggapan"].values[train_index[i]])
            all_train_data_SI.append(tuple(train_data_SI))
            all_train_data_I.append(tuple(train_data_I))
        for j in range(len(test_index)):
            tanggapan_test.append(all_data["Tanggapan"].values[test_index[j]])
            sentiment_test_SI.append(all_data["L_Layanan"].values[test_index[j]])
            sentiment_test_I.append(all_data["L_Fasilitas"].values[test_index[j]])
        pisah_list_kelas(all_train_data_SI)
        training_sentiment(all_train_data_SI, tanggapan_train, tanggapan_test, sentiment_test_SI)
        akurasi_list_SI.append(tuple(akurasi_list))

        akurasi_list.clear()
        l_tanggapan_negatif.clear()
        l_tanggapan_positif.clear()

        pisah_list_kelas(all_train_data_I)
        training_sentiment(all_train_data_I, tanggapan_train, tanggapan_test, sentiment_test_I)
        akurasi_list_I.append(tuple(akurasi_list))

        akurasi_list.clear()
        l_tanggapan_negatif.clear()
        l_tanggapan_positif.clear()

    print("Result (akurasi)............")
    print("Sistem Informasi = ", akurasi_list_SI)
    print("Internet = ", akurasi_list_I)


main()