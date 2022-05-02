import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
random_data = []
for i in range(1,11):
    d = torch.normal(i*torch.ones(11-i,2),1)
    random_data.append(d)
packed_data = pack_sequence(random_data)
print(packed_data)
# lstm = nn.LSTM(
#     input_size = 2,
#     hidden_size = 10,
#     num_layers = 2,
#     batch_first = True,
#     )

# lstm_out,_ = lstm(packed_data)
# print(lstm_out.data.shape)
# print(lstm_out.data)
# lstm_out,_ =pad_packed_sequence(lstm_out)  #  pad_packed_sequence将打包的数据填充回来
# print(lstm_out.data.shape)  # 这之后就可以按照实际数据长度取自己所需数据
# print(lstm_out.data)

for i in enumerate(packed_data):
    print(i)