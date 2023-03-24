import torch
from torch import nn

def contrastive_set(latent_vector, labels, temp=0.05):

    cos = nn.CosineSimilarity(dim=-1)

    sim_list = []
    sim_label = []
    for i in range(latent_vector.size(0)):
        for j in range(latent_vector.size(0)):

            if i == j:
                continue

            sim_value = cos(latent_vector[i, :], latent_vector[j, :]) / temp
            sim_list.append(sim_value)

            if labels[i] == labels[j]:
                sim_label.append(1)
            else:
                sim_label.append(0)
    
    sim_list = torch.tensor(sim_list)
    sim_label = torch.tensor(sim_label).to(torch.float32)

    return sim_list, sim_label