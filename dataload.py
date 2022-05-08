from numpy import dtype
import torch
from torch.utils.data import Dataset, DataLoader
import pre_data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

#读取手动填充
# class data_loader(Dataset):
#     def __init__(self,sent_index_pad,lab_index_pad):
#         self.sent_index=sent_index_pad
#         self.lab_index=lab_index_pad
#     def __len__(self):
#         return len(self.sent_index)
#     def __getitem__(self,idx):
#         return self.sent_index[idx],self.lab_index[idx]

class data_loader(Dataset):
    def __init__(self,sent_index,lab_index):
        self.sent_index=sent_index
        self.lab_index=lab_index
    def __len__(self):
        return len(self.sent_index)
    def __getitem__(self,idx):
        return self.sent_index[idx],self.lab_index[idx]

# #自动填充
# sent=pad_sequence([torch.LongTensor(i) for i in pre_data.sent_index], batch_first=True, padding_value=0)
# lab=pad_sequence([torch.LongTensor(i) for i in pre_data.lab_index], batch_first=True, padding_value=0)

sent_train=torch.Tensor(pre_data.sent_train_pad)
lab_train=torch.Tensor(pre_data.lab_train_pad)
sent_test=torch.Tensor(pre_data.sent_test_pad)
lab_test=torch.Tensor(pre_data.lab_test_pad)

sent_train=sent_train.cuda().long()
lab_train=lab_train.cuda().long()
sent_test=sent_test.cuda().long()
lab_test=lab_test.cuda().long()

train_dataset=data_loader(sent_train,lab_train)
train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True ,num_workers = 0)
test_dataset=data_loader(sent_test,lab_test)
test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=True ,num_workers = 0)
