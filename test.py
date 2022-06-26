
import datetime
from gensim.models import word2vec
import torch
import torch.nn as nn

torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])