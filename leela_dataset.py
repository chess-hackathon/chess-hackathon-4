from pathlib import Path
from torch.utils.data import Dataset, TensorDataset

import torch

class LeelaDatasetReduced(TensorDataset):
    def __init__(self, source_dir):
        super().__init__()
        self.source_dir = source_dir
        filenames = list(Path(source_dir).glob("*"))

        inputs = []
        policy = []
        q = []

        for filename in filenames:
            print("Loading", filename)
            data = torch.load(filename)
            inputs.append(data["inputs"])
            policy.append(data["policy"])
            q.append(data["q"])
            #break

        super().__init__(torch.cat(inputs), torch.cat(policy), torch.cat(q))

        #self.inputs = torch.cat(inputs)
        #self.policy = torch.cat(policy)
        #self.q = torch.cat(q)
        #self.len = len(self.inputs)

    #def __len__(self):
    #    return self.len
            
    #def __getitem__(self, idx):
    #    return self.inputs[idx], self.policy[idx], self.q[idx]

#dataset = LeelaDatasetReduced("/Users/ralph/Data/lc0_filtered")
