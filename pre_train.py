import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# model = Word2Vec(LineSentence('enwik9'), size = 200, window = 5, min_count = 5, workers = multiprocessing.cpu_count())
# model.save('chemical.model')

model=Word2Vec.load('chemical.model')
print(model['chemical'])