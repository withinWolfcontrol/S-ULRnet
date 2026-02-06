import torch
import torch.nn as nn
from torchsummary import summary

class Res_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(Res_Encoder, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class TransposedConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(TransposedConvLayer, self).__init__()

        self.transposed_conv = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.transposed_conv(x)

class S_ULRnet(nn.Module):
    def __init__(self, in_channel, out_channel, add_fc: bool = True, filters=[64, 128, 256, 512]):
        super(S_ULRnet, self).__init__()

        # --- U-CNN Part Starts ---

        self.res_encoder_1_main = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.res_encoder_1_skip = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1)
        )

        self.res_encoder_2 = Res_Encoder(filters[0], filters[1], 2, 1)
        
        self.res_encoder_3 = Res_Encoder(filters[1], filters[2], 2, 1)

        self.res_encoder_4 = Res_Encoder(filters[2], filters[3], 1, 1)


        # Block 5 (Res-Decoder 1): 对应 Table S2 Block 5 (23x23 -> 23x23)
        # 注意：代码中保留了 upsample_1 结构，但在 forward 中未使用，符合 Block 5 尺寸未变
        self.transposed_conv_1 = TransposedConvLayer(filters[3], filters[3], 2, 2) 
        self.res_decoder_1_conv = Res_Encoder(filters[3] + filters[2], filters[2], 1, 1)

        self.transposed_conv_2 = TransposedConvLayer(filters[2], filters[2], 2, 2)
        self.res_decoder_2_conv = Res_Encoder(filters[2] + filters[1], filters[1], 1, 1)

        self.transposed_conv_3 = TransposedConvLayer(filters[1], filters[1], 2, 2)
        self.res_decoder_3_conv = Res_Encoder(filters[1] + filters[0], filters[0], 1, 1)

        self.output_conv = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1),
            nn.Sigmoid(),
        )

        # --- FC Part Starts ---
        self.add_fc = add_fc
        self.input_size = 200
        self.middle_size = 92
        if self.add_fc:
            self.fc_layer = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(self.input_size * self.input_size),
                nn.Linear(self.input_size * self.input_size, self.middle_size * self.middle_size),
                nn.Dropout(0.02),
                nn.Unflatten(dim=1, unflattened_size=(1, self.middle_size, self.middle_size)),
            )

    def forward(self, x):
        # FC Layer processing
        if self.add_fc:
            x = self.fc_layer(x)
            
        # --- Encoder Path ---
        # Res-Encoder 1
        x1 = self.res_encoder_1_main(x) + self.res_encoder_1_skip(x)
        
        # Res-Encoder 2
        x2 = self.res_encoder_2(x1)
        
        # Res-Encoder 3
        x3 = self.res_encoder_3(x2)
        
        # Res-Encoder 4 (Bridge)
        x4 = self.res_encoder_4(x3)

        # --- Decoder Path ---
        # Res-Decoder 1 
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.res_decoder_1_conv(x5)

        # Res-Decoder 2 
        x6 = self.transposed_conv_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.res_decoder_2_conv(x7)

        # Res-Decoder 3 
        x8 = self.transposed_conv_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.res_decoder_3_conv(x9)

        # Output
        output = self.output_conv(x10)

        return output


if __name__ == '__main__':
    model = S_ULRnet(in_channel=1, out_channel=2, add_fc=True, filters=[64, 128, 256, 512])

    x = torch.randn(6, 1, 200, 200)
    b = model(x)
    print("Output shape:", b.shape)
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)
    # summary(model, (1, 200, 200))