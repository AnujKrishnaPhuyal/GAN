import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, input_channels, input_features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(input_channels, input_features, kernel_size=4, stride=2, padding=1),# 3*64*64
            nn.LeakyReLU(0.2),
            self.convolution_block(input_features, input_features*2, 4, 2, 1),
            self.convolution_block(input_features*2, input_features*4, 4, 2, 1),
            self.convolution_block(input_features*4, input_features*8, 4, 2, 1),
            self.convolution_block(input_features*8, input_features*16, 4, 2, 1),
            nn.Conv2d(input_features*16, 1, kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid()
        )

    def convolution_block(self, in_filters, out_filters, kernel_size, stride, padding):   
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size,stride, padding,bias=False,),

            nn.BatchNorm2d(out_filters),
            nn.LeakyReLU(0.2)
        )
        
        
    def forward(self, x):
        return self.disc(x)
    

a =Discriminator(3,64)

class Generator(nn.Module):
    def __init__(self,noise_channels, input_channels, output_channels, input_features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.transposed_convolution_block(noise_channels, input_features*16, 4, 1, 0),
            self.transposed_convolution_block(input_features*16, input_features*8, 4, 2, 1),
            self.transposed_convolution_block(input_features*8, input_features*4, 4, 2, 1),
            self.transposed_convolution_block(input_features*4, input_features*2, 4, 2, 1),
            nn.ConvTranspose2d(input_features*2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  
        )

    def transposed_convolution_block(self, in_filters, out_filters, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_filters),
            nn.ReLU()
            )
        
    def forward(self, x):
        return self.gen(x)
    
b = Generator(100,3,3,64)
# print(b)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    
