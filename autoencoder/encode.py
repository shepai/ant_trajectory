from model import FlexibleAutoencoder
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=FlexibleAutoencoder().to(device)
model.load_state_dict(torch.load("/its/home/drs25/ant_trajectory/autoencoder/flexiencoder_weights.pth"))
model.eval()

def encode(data):
    return model.encode(torch.tensor(data).float().reshape((1,*data.shape)).to(device)).cpu().detach().numpy()

if __name__=="__main__":
    import numpy as np
    data=np.random.random((1,48,8))
    print(encode(data).shape)