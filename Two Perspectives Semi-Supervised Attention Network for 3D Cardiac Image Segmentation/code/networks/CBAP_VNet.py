import torch
from torch import nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # 利用1*1卷积代替全连接层
        self.shared_MLP = nn.Sequential(
            nn.Conv3d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv3d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel, ratio=16)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x

        return x


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        # self.cbam = CBAM3D(n_filters_out)

    def forward(self, x):
        x = self.conv(x)
        # x = self.cbam(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)  # 1 3 16
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)  # 16 32

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)  # 32 32
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)  # 32 64

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)  # 64 64
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)  # 64 128

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)  # 128 128
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16,
                                                   normalization=normalization)  # 128 256

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)  # 256 256
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.cbam1 = CBAM(channel=32)
        self.cbam2 = CBAM(channel=64)
        self.cbam3 = CBAM(channel=128)
        self.cbam4 = CBAM(channel=256)

        self.eca_attention = ECA()

    def forward(self, input):
        x1 = self.eca_attention(input)
        x1 = self.block_one(x1)
        x1_dw = self.block_one_dw(x1)
        x1_dw = self.cbam1(x1_dw) + x1_dw

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        x2_dw = self.cbam2(x2_dw) + x2_dw

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        x3_dw = self.cbam3(x3_dw) + x3_dw

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        x4_dw = self.cbam4(x4_dw) + x4_dw

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg


class Decoder_2out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder_2out, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.dropout2 = nn.Dropout3d(p=0.3, inplace=False)
        self.dsv_8 = nn.Sequential(nn.Conv3d(32, n_classes, kernel_size=3, stride=1, padding=1),
                                   nn.Upsample(scale_factor=2, mode='trilinear'), )

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg1 = self.out_conv(x9)

        if self.has_dropout:
            x8 = self.dropout2(x8)
        out_seg2 = self.dsv_8(x8)
        # out_seg原始差通道   X8一半  X7 1/4
        return out_seg1, out_seg2


class Decoder_3out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder_3out, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.dropout2 = nn.Dropout3d(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout3d(p=0.2, inplace=False)
        self.dsv_8 = nn.Sequential(nn.Conv3d(32, n_classes, kernel_size=1, stride=1, padding=0),
                                   nn.Upsample(scale_factor=2, mode='trilinear'), )
        self.dsv_8_2 = nn.Sequential(nn.Conv3d(64, n_classes, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(scale_factor=4, mode='nearest'), )

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg1 = self.out_conv(x9)

        if self.has_dropout:
            x8 = self.dropout2(x8)
        out_seg2 = self.dsv_8(x8)

        if self.has_dropout:
            x7 = self.dropout3(x7)  # 这里应该是3，之前的结果是2，有误
        out_seg3 = self.dsv_8_2(x7)

        # out_seg原始差通道   X8一半  X7 1/4
        return out_seg1, out_seg2, out_seg3


class Decoder_4out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder_4out, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.dropout2 = nn.Dropout3d(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout3d(p=0.2, inplace=False)
        self.dropout4 = nn.Dropout3d(p=0.1, inplace=False)
        self.dsv_8 = nn.Sequential(nn.Conv3d(32, n_classes, kernel_size=1, stride=1, padding=0),
                                   nn.Upsample(scale_factor=2, mode='trilinear'), )
        self.dsv_8_2 = nn.Sequential(nn.Conv3d(64, n_classes, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(scale_factor=4, mode='nearest'), )
        self.dsv_8_3 = nn.Sequential(nn.Conv3d(128, n_classes, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(scale_factor=8, mode='nearest'), )

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        # print("x5_up shape:", x5_up.shape)
        # print("x4 shape:", x4.shape)
        x5_up = x5_up + x4

        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg1 = self.out_conv(x9)

        if self.has_dropout:
            x8 = self.dropout2(x8)
        out_seg2 = self.dsv_8(x8)

        if self.has_dropout:
            x7 = self.dropout3(x7)  # 这里应该是3，之前的结果是2，有误
        out_seg3 = self.dsv_8_2(x7)

        if self.has_dropout:
            x6 = self.dropout4(x6)  # 这里应该是3，之前的结果是2，有误
        out_seg4 = self.dsv_8_3(x6)

        return out_seg1, out_seg2, out_seg3, out_seg4


class Decoder_5out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder_5out, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.dropout2 = nn.Dropout3d(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout3d(p=0.2, inplace=False)
        self.dropout4 = nn.Dropout3d(p=0.1, inplace=False)
        self.dropout5 = nn.Dropout3d(p=0.1, inplace=False)
        self.dsv_8 = nn.Sequential(nn.Conv3d(32, n_classes, kernel_size=1, stride=1, padding=0),
                                   nn.Upsample(scale_factor=2, mode='trilinear'), )
        self.dsv_8_2 = nn.Sequential(nn.Conv3d(64, n_classes, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(scale_factor=4, mode='nearest'), )
        self.dsv_8_3 = nn.Sequential(nn.Conv3d(128, n_classes, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(scale_factor=8, mode='nearest'), )
        self.dsv_8_4 = nn.Sequential(nn.Conv3d(256, n_classes, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(scale_factor=16, mode='trilinear'), )

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg1 = self.out_conv(x9)

        if self.has_dropout:
            x8 = self.dropout2(x8)
        out_seg2 = self.dsv_8(x8)

        if self.has_dropout:
            x7 = self.dropout3(x7)  # 这里应该是3，之前的结果是2，有误
        out_seg3 = self.dsv_8_2(x7)

        if self.has_dropout:
            x6 = self.dropout4(x6)  # 这里应该是3，之前的结果是2，有误
        out_seg4 = self.dsv_8_3(x6)
        if self.has_dropout:
            x5 = self.dropout5(x5)  # 这里应该是3，之前的结果是2，有误
        out_seg5 = self.dsv_8_4(x5)

        # out_seg原始差通道   X8一半  X7 1/4
        return out_seg1, out_seg2, out_seg3, out_seg4, out_seg5


class Decoder_2out_2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder_2out_2, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)
        self.block_eight_up_2 = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                    mode_upsampling=1)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_2 = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv_2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.dropout2 = nn.Dropout3d(p=0.5, inplace=False)
        self.dsv_8 = nn.Sequential(nn.Conv3d(32, n_classes, kernel_size=3, stride=1, padding=1),
                                   nn.Upsample(scale_factor=2, mode='trilinear'), )

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x8_up_2 = self.block_eight_up_2(x8)
        x8_up_2 = x8_up_2 + x1
        x9 = self.block_nine(x8_up)
        x9_2 = self.block_nine_2(x8_up_2)
        if self.has_dropout:
            x9 = self.dropout(x9)
            x9_2 = self.dropout(x9_2)
        out_seg1 = self.out_conv(x9)
        out_seg2 = self.out_conv(x9_2)
        return out_seg1, out_seg2


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1


class VNet_2out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_2out, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_2out(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, out_seg2 = self.decoder1(features)
        return out_seg1, out_seg2


class VNet_2out_2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_2out_2, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_2out(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, out_seg2 = self.decoder1(features)
        return out_seg1, out_seg2


class VNet_3out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_3out, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_3out(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, out_seg2, out_seg3 = self.decoder1(features)
        return out_seg1, out_seg2, out_seg3


class VNet_4out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_4out, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_4out(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, out_seg2, out_seg3, out_seg4 = self.decoder1(features)
        return out_seg1, out_seg2, out_seg3, out_seg4


class VNet_5out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet_5out, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_5out(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, out_seg2, out_seg3, out_seg4, out_seg5 = self.decoder1(features)
        return out_seg1, out_seg2, out_seg3, out_seg4, out_seg5


class MCNet3d_v1(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(MCNet3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2


class MCNet3d_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(MCNet3d_v2, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        out_seg3 = self.decoder3(features)
        return out_seg1, out_seg2, out_seg3


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info

    model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (1, 192, 192, 64), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb;

    ipdb.set_trace()
