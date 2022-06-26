from gensim.models import word2vec

sentences = word2vec.Text8Corpus('./data/word.txt')
model = word2vec.Word2Vec(sentences, size=256)
model.save('./model/chemical256.w2v')

