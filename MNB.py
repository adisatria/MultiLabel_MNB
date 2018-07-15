import re, glob, nltk, string, csv, math
import numpy as np
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

dataset = 'PusatBahasaP100.csv'
num_folds = 10
factory1 = StopWordRemoverFactory()
stopword = factory1.create_stop_word_remover()

factory2 = StemmerFactory()
stemmer = factory2.create_stemmer()

with open('kamus/katadasar.txt', 'r') as filekatadasar,open('kamus/dict_vocab.txt', 'r') as filevocab, \
        open('kamus/alfabet.txt', 'r') as filealfabet, open('kamus/english.txt','r') as fileenglish:
    katadasar = filekatadasar.read().replace('\n', ',')
    vocab = filevocab.read().replace('\n', ',')
    alfabet = filealfabet.read().replace('\n', ',')
    english = fileenglish.read().replace('\n', ',')

def preprocessing(dataset):
    tokenizing = dataset.split(' ')
    hasilprepro = []
    for i in range(len(tokenizing)):
        teks = re.sub('(\d)+(\.)*(\d)*', '', tokenizing[i])  # hapus digit
        teks = re.sub('[/+@.,%-%^*"!#-$-\']', '', teks)  # hapus simbol
        teks = teks.lower()

        #stopword removal
        prestopword = stopword.remove(teks)

        if prestopword not in alfabet:
            if prestopword not in english:
                if len(prestopword)>3:
                    if prestopword in vocab and prestopword != '':
                        hasilstem = stemmer.stem(prestopword)
                        if hasilstem in katadasar and hasilstem != '':
                            hasilprepro.append(hasilstem)
    return hasilprepro

def preprocessingTanpaStemm(dataset):
    tokenizing = dataset.split(' ')
    hasilprepro = []
    for i in range(len(tokenizing)):
        teks = re.sub('(\d)+(\.)*(\d)*', '', tokenizing[i])  # hapus digit
        teks = re.sub('[/+@.,%-%^*"!#-$-\']', '', teks)  # hapus simbol
        teks = teks.lower()

        #stopword removal
        prestopword = stopword.remove(teks)

        if prestopword not in alfabet:
            if prestopword not in english:
                if len(prestopword)>3:
                    if prestopword in vocab and prestopword != '':
                        hasilprepro.append(prestopword)
    return hasilprepro

def bacafile(filename):
    semua = []
    idData = []
    trueLabelFasilitas = []
    trueLabelLayanan = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            print("Data ke-", row[0])
            if row:
                HasilPrepro = preprocessing(row[1])
                semua.append(HasilPrepro)
                print("hasil Prepro ", HasilPrepro)
                idData.append(int(row[0]))
                dataLabel = row[2:5]
                dataLabel = [int(j) for j in dataLabel]
                trueLabelFasilitas.append(dataLabel[0])
                trueLabelLayanan.append(dataLabel[1])
            else:
                continue
    return semua, idData, trueLabelFasilitas, trueLabelLayanan

def dataBersih():
    panjangTang = bacafile2('datapalsu.csv')
    print(panjangTang)
    with open('Data_Bersih.csv', 'w', newline='') as writeData:
        write = csv.writer(writeData, delimiter=',',quotechar=',')
        for row in range(len(panjangTang)):
            write.writerow(panjangTang[row])

def main7():
    panjangTang, idData, trueLabelFasilitas, trueLabelLayanan = bacafile(dataset)

    subset_size = int(len(idData) / num_folds)
    for i in range(num_folds):
        testing_this_round = idData[i * subset_size:][:subset_size]
        training_this_round = idData[:i * subset_size] + idData[(i + 1) * subset_size:]

        print('============== k', i + 1, ' ==============')
        print('testing : ', testing_this_round)
        print('length testing : ', len(testing_this_round))
        print('length training : ', len(training_this_round))

        daftarDokumen = {}
        #pisahkan tanggapan per label pada kategori fasilitas
        tanggapanPositifFasilitas = []
        tanggapanNegatifFasilitas = []

        # pisahkan tanggapan per label pada kategori layanan
        tanggapanPositifLayanan = []
        tanggapanNegatifLayanan = []

        # jumlah fasilitas yang berlabel pada data training
        jmlFasilitasPOS = 0
        jmlFasilitasNEG = 0

        #jumlah layanan yang berlabel pada data training
        jmlLayananPOS = 0
        jmlLayananNEG = 0

        for j in range(len(training_this_round)):
            id = training_this_round[j]
            daftarDokumen[id] = Counter(panjangTang[id])
            if trueLabelFasilitas[id] == 1:
                tanggapanPositifFasilitas.extend(panjangTang[id])
                jmlFasilitasPOS+=1
            elif trueLabelFasilitas[id] == 0:
                tanggapanNegatifFasilitas.extend(panjangTang[id])
                jmlFasilitasNEG+=1

            if trueLabelLayanan[id] == 1:
                tanggapanPositifLayanan.extend(panjangTang[id])
                jmlLayananPOS+=1
            elif trueLabelLayanan[id] == 0:
                tanggapanNegatifLayanan.extend(panjangTang[id])
                jmlLayananNEG+=1
        print("jml fasilitas pos : ", jmlFasilitasPOS)
        print("jml fasilitas neg : ", jmlFasilitasNEG)
        print("jml layanan pos : ", jmlLayananPOS)
        print("jml layanan neg : ", jmlLayananNEG)
        print('\n')
        print("tang Fasilitas Pos ",len(tanggapanPositifFasilitas)," : ", tanggapanPositifFasilitas)
        print("tang Fasilitas Neg ",len(tanggapanNegatifFasilitas)," : ", tanggapanNegatifFasilitas)
        print("tang Layanan Pos ",len(tanggapanPositifLayanan)," : ", tanggapanPositifLayanan)
        print("tang Layanan Neg ",len(tanggapanNegatifLayanan)," : ", tanggapanNegatifLayanan)
        # menampung daftar string
        daftarString = []
        for key, val in daftarDokumen.items():  # melompati(loop over) key di kamus
            for word, count_w in val.items():
                if word not in daftarString:
                    daftarString.append(word)  # append untuk menambah objek word baru kedalam list

        print("daftar string : ", daftarString)
        print("pjg string : ", len(daftarString))
        # print("Semua tanggapan ", semuaTanggapan)

main7()
# dataBersih()