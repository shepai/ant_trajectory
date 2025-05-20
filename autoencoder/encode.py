from model import FlexibleAutoencoder
model=FlexibleAutoencoder()
model.load_state_dict(torch.load("/its/home/drs25/ant_trajectory/autoencoder/flexiencoder_weights.pth"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def encode(data):
    return model.encode(data.to(device)).cpu().detach().numpy()