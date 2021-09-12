from gensim.models import Word2Vec
import os
#from random import shuffle
'''def _join_path(path, *paths):
    if not paths:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    else:
        rst = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        for i in paths:
            rst = os.path.join(rst, i)
        return rst'''

def main():
    f1 = open('train_cut_shuffle.txt', 'r', encoding='utf-8')
    f2 = open('test_cut.txt', 'r', encoding='utf-8')

    line_sent = []
    for l1 in f1:
        seq = l1.split('\t')[1].split('\n')[0].split(' ')
        line_sent.append(seq)

    for l2 in f2:
        seq = l2.split('\t')[1].split('\n')[0].split(' ')
        line_sent.append(seq)
    f1.close()
    f2.close()
    #shuffle(line_sent)
    #print(line_sent)
    model = Word2Vec(sentences=line_sent,
                     size = 256,
                     window = 5,
                     min_count = 1,
                     workers = 2)
    '''w2v_model_path = 'w2v_model'
    if not w2v_model_path:
        os.mkdir(w2v_model_path)'''

    model.save('w2v_model/word2vec.model')
    print('have trained')
if __name__ == '__main__':
    main()
