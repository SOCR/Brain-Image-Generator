import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaGenerator(nn.Module):
    def __init__(self,ouput_channels, embedding_dim, latent_dim, n_classes):
        super(VanillaGenerator, self).__init__()


        self.label_conditioned_generator = nn.Sequential(nn.Embedding(n_classes, embedding_dim),
                      nn.Linear(embedding_dim, 16))


        self.latent = nn.Sequential(nn.Linear(latent_dim, 4*4*512),
                                   nn.LeakyReLU(0.2, inplace=True))


        self.model = nn.Sequential(
                      nn.ConvTranspose2d(513, 64*8, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(64*8, momentum=0.1,  eps=0.8),
                      nn.ReLU(False),
                      nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1,bias=False),
                      nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
                      nn.ReLU(False),
                      nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1,bias=False),
                      nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
                      nn.ReLU(False),
                      nn.ConvTranspose2d(64*2, 64*1, 4, 2, 1,bias=False),
                      nn.BatchNorm2d(64*1, momentum=0.1,  eps=0.8),
                      nn.ReLU(False),
                      nn.ConvTranspose2d(64*1, ouput_channels, 4, 2, 1, bias=False),
                      nn.Tanh()
                    )

    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512,4,4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        #print(image.size())
        return image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionLayer, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x).view(batch_size, -1, H * W)  # B x C/8 x N
        key = self.key_conv(x).view(batch_size, -1, H * W)      # B x C/8 x N
        value = self.value_conv(x).view(batch_size, -1, H * W)  # B x C x N

        attention = self.softmax(torch.bmm(query.permute(0, 2, 1), key))  # B x N x N
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, H, W)
        return out + x  # residual connection for attention

class SpatialAttentionUNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, feature_dim=64):
        super(SpatialAttentionUNetGenerator, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True)
        )

        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(feature_dim * 2),
            ResidualBlock(feature_dim * 2)
        )

        # Attention Layer
        self.attention_layer = SpatialAttentionLayer(feature_dim * 2)

        # Decoder with skip connections
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 2 * 2, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # First feature map (skip connection 1)
        enc2 = self.encoder2(enc1)  # Second feature map (skip connection 2)

        # Residual Blocks
        x = self.residual_blocks(enc2)

        # Attention Layer
        x = self.attention_layer(x)

        # Decoder with skip connections
        x = self.decoder1(torch.cat([x, enc2], dim=1))  # Concatenate with skip connection 2
        x = self.decoder2(torch.cat([x, enc1], dim=1))  # Concatenate with skip connection 1

        return x

#################################################################################################################
################################ 3D models ######################################################################
##############################################################################################################
class Generator3D_32(nn.Module):
    def __init__ (self, noise_size=201, cube_resolution=32):
        super(Generator3D_32, self).__init__()
        
        self.noise_size = noise_size
        self.cube_resolution = cube_resolution
        
        self.gen_conv1 = torch.nn.ConvTranspose3d(self.noise_size, 256, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.gen_conv2 = torch.nn.ConvTranspose3d(256, 128, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.gen_conv3 = torch.nn.ConvTranspose3d(128, 64, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.gen_conv4 = torch.nn.ConvTranspose3d(64, 32, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        self.gen_conv5 = torch.nn.ConvTranspose3d(32, 1, kernel_size=[4,4,4], stride=[2,2,2], padding=1)
        
        self.gen_bn1 = nn.BatchNorm3d(256)
        self.gen_bn2 = nn.BatchNorm3d(128)
        self.gen_bn3 = nn.BatchNorm3d(64)
        self.gen_bn4 = nn.BatchNorm3d(32)
        
    
    def forward(self, x, condition):
        
        condition_tensor = condition * torch.ones([x.shape[0],1], device=x.device)
        x = torch.cat([x, condition_tensor], dim=1)
        x = x.view(x.shape[0],self.noise_size,1,1,1)
        
        x = F.relu(self.gen_bn1(self.gen_conv1(x)))
        x = F.relu(self.gen_bn2(self.gen_conv2(x)))
        x = F.relu(self.gen_bn3(self.gen_conv3(x)))
        x = F.relu(self.gen_bn4(self.gen_conv4(x)))
        x = self.gen_conv5(x)
        x = torch.sigmoid(x)
        
        return x.squeeze()

class Generator3D_64(nn.Module):
    def __init__(self, noise_size=201, cube_resolution=32):
        super(Generator3D_64, self).__init__()
        self.noise_size = noise_size
        m = 4

        self.gen_conv1 = nn.ConvTranspose3d(self.noise_size, 256 * m, kernel_size=4, stride=2, padding=1)
        self.gen_conv2 = nn.ConvTranspose3d(256 * m, 128 * m, kernel_size=4, stride=2, padding=1)
        self.gen_conv3 = nn.ConvTranspose3d(128 * m, 64 * m, kernel_size=4, stride=2, padding=1)
        self.gen_conv4 = nn.ConvTranspose3d(64 * m, 32 * m, kernel_size=4, stride=2, padding=1)
        self.gen_conv5 = nn.ConvTranspose3d(32 * m, 16 * m, kernel_size=4, stride=2, padding=1)
        self.gen_conv6 = nn.ConvTranspose3d(16 * m, 1, kernel_size=4, stride=2, padding=1)

        self.gen_bn1 = nn.BatchNorm3d(256 * m)
        self.gen_bn2 = nn.BatchNorm3d(128 * m)
        self.gen_bn3 = nn.BatchNorm3d(64 * m)
        self.gen_bn4 = nn.BatchNorm3d(32 * m)
        self.gen_bn5 = nn.BatchNorm3d(16 * m)
        
        
        self.skip1_4 = nn.Conv3d( 256 * m, 32 * m,kernel_size=1, stride=1)
        # self.skip1_5 = nn.Conv3d( 256 * m, 16 * m,kernel_size=1, stride=1)
        self.skip2_5 = nn.Conv3d( 128 * m, 16 * m,kernel_size=1, stride=1)
    
    def forward(self, x, condition):
        condition_tensor = condition * torch.ones([x.shape[0], 1], device=x.device)
        x = torch.cat([x, condition_tensor], dim=1)
        x = x.view(x.shape[0], self.noise_size, 1, 1, 1)

        out1 = F.relu(self.gen_bn1(self.gen_conv1(x)))
        out2 = F.relu(self.gen_bn2(self.gen_conv2(out1)))
        out3 = F.relu(self.gen_bn3(self.gen_conv3(out2)))
        out4 = F.relu(self.gen_bn4(self.gen_conv4(out3)))
        out5 = F.relu(self.gen_bn5(self.gen_conv5(out4)))
        
        # make the number of channels equal
        
        skip1_4 = self.skip1_4(out1)
        # skip1_5 = self.skip1_5(out1)
        skip2_5 = self.skip2_5(out2)
        
        
        # Upsample and adjust dimensions for skip connections
        out2_up = F.interpolate(skip2_5, size=out5.shape[2:], mode='trilinear', align_corners=True)
        out1_up = F.interpolate(skip1_4, size=out4.shape[2:], mode='trilinear', align_corners=True)
        # out1_up = F.interpolate(skip1_5, size=out5.shape[2:], mode='trilinear', align_corners=True)
        
        # Skip connections
        out5 = out5 + out2_up  # Adjust dimensions
        out4 = out4 + out1_up # Adjust dimensions
        
        x = self.gen_conv6(out5)
        x = torch.sigmoid(x)
        
        return x.squeeze()
