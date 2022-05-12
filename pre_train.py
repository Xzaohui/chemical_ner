from gensim.models import word2vec
sentences = word2vec.Text8Corpus('./model/enwik9')
model = word2vec.Word2Vec(sentences, size=100)
word2vec.save('./model/chemical.model')
print(model.most_similar('oxygen'))