import jieba
import re
from zhon.hanzi import punctuation
import string
import os
import csv
from random import shuffle

'''def _join_path(path, *paths):
    if not paths:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    else:
        rst = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        for i in paths:
            rst = os.path.join(rst, i)
        return rst'''


def main():

    '''if not _join_path('txt_data'):
        os.makedirs(_join_path('txt_data'))'''

    '''f = open(_join_path('txt_data','test.txt'), 'r', encoding='utf-8')
    f_new = open(_join_path('new_test.txt'), 'w', encoding='utf-8')
    for l in f:
        f_new.write(l.split('\t')[0] + '\t' + l.split('\t')[1].replace(' ', ''))
    f.close()
    f_new.close()

    f = open(_join_path('txt_data','train.txt'), 'r', encoding='utf-8')
    f_new = open(_join_path( 'new_train.txt'), 'w', encoding='utf-8')
    for l in f:
        f_new.write(l.split('\t')[0] + '\t' + l.split('\t')[1].replace(' ', ''))
    f.close()
    f_new.close()'''
    r = 0
    vocab = {}
    cut_file = open('train_cut.txt', 'w', encoding='utf-8')
    with open("/home/0510894/final/train_data.csv", 'r', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        a = []
        for row in rows:
            if r != 0 :
                label = row[1]
                str_ = row[3].split('\n')[0] + row[4].split('\n')[0]
                str_ = re.sub(u"[%s]+" % string.punctuation, "", str_)
                str_ = re.sub(u"[%s]+" % punctuation, "", str_)
                cut_result = list(jieba.cut(str_, cut_all=False))
                a.append(cut_result)

                for w in cut_result:
                    if w in vocab.keys():
                        vocab[w] += 1
                    else:
                        vocab[w] = 1

                cut_file.write(label + '\t' + ' '.join(cut_result) + '\n')

            r = r + 1
    csvfile.close()
    cut_file.close()

    f_new = open('train_cut.txt', 'r', encoding='utf-8')
    f_out = open('train_cut_shuffle.txt', 'w', encoding='utf-8')
    temp_str = []
    for new_line in f_new:
        temp_str.append(new_line)
    shuffle(temp_str)
    f_out.write("".join(temp_str))
    f_out.close()

    r = 0
    cut_file = open('test_cut.txt', 'w', encoding='utf-8')
    with open("/home/0510894/final/sample.csv", 'r', encoding='utf-8') as cf:
        test_ans = list(csv.reader(cf))
    cf.close()
    with open("/home/0510894/final/test_data.csv", 'r', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        #b = []
        i = 0
        print(rows)
        for row in rows:
            if r != 0:
                #b.append(row)
                label = test_ans[i][1]
                str_ = row[1].split('\n')[0] + row[2].split('\n')[0]
                str_ = re.sub(u"[%s]+" % string.punctuation, "", str_)
                str_ = re.sub(u"[%s]+" % punctuation, "", str_)
                cut_result = list(jieba.cut(str_, cut_all=False))

                for w in cut_result:
                    if w in vocab.keys():
                        vocab[w] += 1
                    else:
                        vocab[w] = 1

                cut_file.write(label + '\t' + ' '.join(cut_result) + '\n')
            i = i + 1
            r = r + 1
    csvfile.close()
    cut_file.close()


    sorted_vocab = sorted(vocab.items(), key=lambda vocab: vocab[1], reverse=True)
    vocab_file = open('vocab.txt', 'w', encoding='utf-8')
    for k in sorted_vocab:
        vocab_file.write(k[0] + '\t' + str(k[1]) + '\n')
    vocab_file.close()
    '''vocab = {}
    f_new = open(_join_path('new_test.txt'), 'r', encoding='utf-8')
    cut_file = open(_join_path('test_cut.txt'), 'w', encoding='utf-8')
    for l in f_new:
        label = l.split('\t')[0]
        str_ = l.split('\t')[1].split('\n')[0]
        str_ = re.sub(u"[%s]+" %string.punctuation, "", str_)
        str_ = re.sub(u"[%s]+" %punctuation, "", str_)
        cut_result = list(jieba.cut(str_, cut_all=False))

        for w in cut_result:
            if w in vocab.keys():
                vocab[w] +=1
            else:
                vocab[w] = 1

        cut_file.write(label + '\t' + ' '.join(cut_result) + '\n')
    cut_file.close()
    f_new = open(_join_path('new_train.txt'), 'r', encoding='utf-8')
    cut_file = open(_join_path('train_cut.txt'), 'w', encoding='utf-8')

    for l in f_new:
        label = l.split('\t')[0]
        str_ = l.split('\t')[1].split('\n')[0]
        str_ = re.sub(u"[%s]+" %string.punctuation, "", str_)
        str_ = re.sub(u"[%s]+" %punctuation, "", str_)
        cut_result = list(jieba.cut(str_, cut_all=False))

        for w in cut_result:
            if w in vocab.keys():
                vocab[w] +=1
            else:
                vocab[w] = 1

        cut_file.write(label + '\t' + ' '.join(cut_result) + '\n')
    cut_file.close()

    sorted_vocab = sorted(vocab.items(), key=lambda vocab: vocab[1], reverse = True)
    vocab_file = open(_join_path('vocab.txt'), 'w', encoding='utf-8')
    for k in sorted_vocab:
        vocab_file.write(k[0] + '\t' + str(k[1]) + '\n')
    vocab_file.close()'''

if __name__ == '__main__':
    main()
