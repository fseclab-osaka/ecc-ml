import torch
from torch import nn
input_dim = 10
embedding_dim = 2
embedding = nn.Embedding(input_dim, embedding_dim)
err = False
if err:
    #Any input more than input_dim - 1, here input_dim = 10
    #Any input less than zero
    input_to_embed = torch.tensor([10])
else:
    input_to_embed = torch.tensor([[3, 1, 2, 3, 4, 1, 6, 7, 8, 9], [4, 1, 2, 3, 8, 1, 6, 0, 2, 9]])
embed = embedding(input_to_embed)
#for p in embedding.parameters():
#    print(p.size())
#print(input_to_embed)
#print(embed)

from network.resnet import ResNet18
model = ResNet18()
for p in model.parameters():
    print(p.size())