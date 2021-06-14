from torchsummary import summary
from models import Generator_Resnet, Discriminator

input_shape = (3, 250, 250)

G_AB = Generator_Resnet(input_shape, 5)
G_BA = Generator_Resnet(input_shape, 5)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

