import torch 
from torchsummary import summary 
from torch.utils.data import DataLoader
from imageio import imread

from network import ShapeDecoder, AlbedoDecoder, SingleViewRecon

device = torch.device('cpu')

shape_path = './models/ShapeDecoder.pth'
albedo_path = './models/AlbedoDecoder.pth'

shape_model = ShapeDecoder()
shape_weights = torch.load(shape_path, map_location=device)
shape_state_dict = shape_weights['ShapeDecoder_state_dict']
# shape_model.load_state_dict(torch.load(shape_path, map_location=device))
shape_model.load_state_dict(shape_state_dict)
# summary(shape_model)
print(shape_model)

albedo_model = AlbedoDecoder()
albedo_weights = torch.load(albedo_path, map_location=device)
albedo_state_dict = albedo_weights['AlbedoDecoder_state_dict']
albedo_model.load_state_dict(albedo_state_dict)
# albedo_model.summary()
print(albedo_model)


# Get the image
def load_input_img(path):
    img = imread(path)
    img = img[:,:,:3]
    return img


model = SingleViewRecon(ShapeDecoder=shape_model, AlbedoDecoder=albedo_model)
print(model(img=load_input_img('/home/loaner/Documents/Research/3d-reconstruct/PASCAL3D+_release1.0/Images/aeroplane_pascal/2008_000021.jpg'), is_training=False))
