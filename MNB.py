import re, glob, nltk, string, csv, math
import numpy as np
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics import accuracy_score

dataset = 'PusatBahasaP5000.csv'
num_folds = 10
simpanData = 'Hasil/MNBwithoutStemming.csv'

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
            if row:
                HasilPrepro = preprocessingTanpaStemm(row[1])
                semua.append(HasilPrepro)
                print("hasil Prepro ke ",row[0]," :", HasilPrepro)
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

def likelihood(data_testing, tanggapanPositif, tanggapanNegatif, daftarString):
    likeliPOS = []
    likeliNEG = []
    for i in range(len(data_testing)):
        kata = data_testing[i]
        hitungTkCPOS = tanggapanPositif.count(kata)  # tk(c) positif
        hitungTkCNEG = tanggapanNegatif.count(kata)  # tk(c) negatif
        hitungTkCPOS = hitungTkCPOS + 1  # tk(c) positif + smoothing
        hitungTkCNEG = hitungTkCNEG + 1  # tk(c) negatif + smoothing
        nilaiLikelihoodPOS = hitungTkCPOS / (len(tanggapanPositif) + len(daftarString))
        nilaiLikelihoodNEG = hitungTkCNEG / (len(tanggapanNegatif) + len(daftarString))

        likeliPOS.append(nilaiLikelihoodPOS)
        likeliNEG.append(nilaiLikelihoodNEG)
    return likeliPOS, likeliNEG

def hitungMNB(likeliPOS, likeliNEG, priorPOS, priorNEG):
    labelHasilTes = 0
    hasilMNBPOS = priorPOS*(np.prod(likeliPOS))
    hasilMNBNEG = priorNEG*(np.prod(likeliNEG))

    if hasilMNBPOS > hasilMNBNEG:
        labelHasilTes = 1
    else:
        labelHasilTes = 0

    return labelHasilTes


def main7():
    panjangTang, idData, trueLabelFasilitas, trueLabelLayanan = bacafile(dataset)

    totalLabelHasilTesFasilitas = []
    totalLabelHasilTesLayanan = []
    subset_size = int(len(idData) / num_folds)

    # nulis hasil testing akhir
    with open(simpanData, 'w', newline='')as nulishasil:
        DataSistem = csv.writer(nulishasil, delimiter=',', quotechar='|')
        headerHasilSistem = ['ID', 'Sentimen Fasilitas', 'Sentimen Layanan']
        DataSistem.writerow(headerHasilSistem)

        for i in range(num_folds):
            testing_this_round = idData[i * subset_size:][:subset_size]
            training_this_round = idData[:i * subset_size] + idData[(i + 1) * subset_size:]

            print('============== k', i + 1, ' ==============')
            print('testing : ', testing_this_round)
            print('length testing : ', len(testing_this_round))
            print('length training : ', len(training_this_round))
            print("Menghitung Prior . . . . . . . . . .")
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

            # menampung daftar string
            daftarString = [] #semua string pada data training
            for key, val in daftarDokumen.items():  # melompati(loop over) key di kamus
                for word, count_w in val.items():
                    if word not in daftarString:
                        daftarString.append(word)  # append untuk menambah objek word baru kedalam list
            #prior fasilitas
            priorFasilitasPOS = jmlFasilitasPOS/len(training_this_round)
            priorFasilitasNEG = jmlFasilitasNEG/len(training_this_round)

            #prior layanan
            priorLayananPOS = jmlLayananPOS/len(training_this_round)
            priorLayananNEG = jmlLayananNEG/len(training_this_round)

            totalLikelihoodFasilitasPOS = []
            totalLikelihoodFasilitasNEG = []
            totalLikelihoodLayananPOS = []
            totalLikelihoodLayananNEG = []

            labelHasilTestFasilitas = []
            labelHasilTestLayanan = []

            print("Perhitungan Likelihood dan Hasil Akhir MNB . . . . .")
            for j in range(len(testing_this_round)):
                id = testing_this_round[j]
                dataTesting = panjangTang[id]
                # print("dataTesting : ", dataTesting)
                likeliFasilitasPOS, likeliFasilitasNEG = likelihood(dataTesting,tanggapanPositifFasilitas,
                                                                    tanggapanNegatifFasilitas,daftarString)
                likeliLayananPOS, likeliLayananNEG = likelihood(dataTesting, tanggapanPositifLayanan,
                                                                tanggapanNegatifLayanan, daftarString)
                totalLikelihoodFasilitasPOS.append(likeliFasilitasPOS)
                totalLikelihoodFasilitasNEG.append(likeliFasilitasNEG)
                totalLikelihoodLayananPOS.append(likeliLayananPOS)
                totalLikelihoodLayananNEG.append(likeliLayananNEG)

                nilaiMNBFasilitas = hitungMNB(likeliFasilitasPOS,likeliFasilitasNEG,
                                                                       priorFasilitasPOS, priorFasilitasNEG)
                nilaiMNBLayanan = hitungMNB(likeliLayananPOS, likeliLayananNEG,
                                                                       priorLayananPOS, priorLayananNEG)

                duaHasil = []
                duaHasil.append(id)
                duaHasil.append(nilaiMNBFasilitas)
                duaHasil.append(nilaiMNBLayanan)
                labelHasilTestFasilitas.append(nilaiMNBFasilitas)
                labelHasilTestLayanan.append(nilaiMNBLayanan)
                totalLabelHasilTesFasilitas.append(nilaiMNBFasilitas)
                totalLabelHasilTesLayanan.append(nilaiMNBLayanan)
                DataSistem.writerow(duaHasil)

            # print("TotalnilaiMNBFasilitas : ",labelHasilTestFasilitas)
            # print("TotalnilaiMNBLayanan : ", labelHasilTestLayanan)
            # print('\n')
            # print("total likeli fasilitas pos:", totalLikelihoodFasilitasPOS)
            # print("total likeli fasilitas neg:", totalLikelihoodFasilitasNEG)
            # print("total liekli layanan pos :", totalLikelihoodLayananPOS)
            # print("total liekli layanan neg :", totalLikelihoodLayananNEG)
            # print("\n")
            #
            # print("prior fasilitas : positif(",priorFasilitasPOS,") dan negatif(",priorFasilitasNEG,")")
            # print("prior layanan : positif(", priorLayananPOS, ") dan negatif(", priorLayananNEG, ")")
            # print("jml fasilitas pos : ", jmlFasilitasPOS)
            # print("jml fasilitas neg : ", jmlFasilitasNEG)
            # print("jml layanan pos : ", jmlLayananPOS)
            # print("jml layanan neg : ", jmlLayananNEG)
            # print('\n')
            # print("tang Fasilitas Pos ",len(tanggapanPositifFasilitas)," : ", tanggapanPositifFasilitas)
            # print("tang Fasilitas Neg ",len(tanggapanNegatifFasilitas)," : ", tanggapanNegatifFasilitas)
            # print("tang Layanan Pos ",len(tanggapanPositifLayanan)," : ", tanggapanPositifLayanan)
            # print("tang Layanan Neg ",len(tanggapanNegatifLayanan)," : ", tanggapanNegatifLayanan)
            #
            #
            # print("daftar string : ", daftarString)
            # print("pjg string : ", len(daftarString))
            # print("Semua tanggapan ", semuaTanggapan)
        # print("totalLabelHasilTesFasilitas : ",totalLabelHasilTesFasilitas)
        # print("totalLabelHasilTesLayanan : ", totalLabelHasilTesLayanan)
        # print("trueLabelFasilitas : ", trueLabelFasilitas)
        # print("trueLabelLayanan : ", trueLabelLayanan)
        akurasiFasilitas = (accuracy_score(trueLabelFasilitas, totalLabelHasilTesFasilitas))*100
        akurasiLayanan = (accuracy_score(trueLabelLayanan, totalLabelHasilTesLayanan))*100
        totalAkurasi = ((akurasiFasilitas+akurasiLayanan)/2)
        print("Akurasi Kategori Fasilitas : ",akurasiFasilitas,'%')
        print("Akurasi Kategori Layanan : ", akurasiLayanan,'%')
        print("Total Akurasi : ", totalAkurasi,'%')
main7()
# dataBersih()