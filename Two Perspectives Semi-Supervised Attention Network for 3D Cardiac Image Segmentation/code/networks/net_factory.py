from networks.VNet import VNet, VNet_2out,VNet_2out_2,VNet_3out,VNet_4out,VNet_5out
from networks.CBAP_VNet import VNet, VNet_2out,VNet_2out_2,VNet_3out,VNet_4out,VNet_5out
def net_factory(net_type="VNet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out" and mode == "test":
        net = VNet_2out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out" and mode == "train":
        net = VNet_2out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_2out_2" and mode == "test":
        net = VNet_2out_2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out_2" and mode == "train":
        net = VNet_2out_2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_3out" and mode == "test":
        net = VNet_3out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_3out" and mode == "train":
        net = VNet_3out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_4out" and mode == "test":
        net = VNet_4out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_4out" and mode == "train":
        net = VNet_4out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_5out" and mode == "test":
        net = VNet_5out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_5out" and mode == "train":
        net = VNet_5out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    return net

def net_factory_CBAP(net_type="VNet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "vnet" and mode == "train":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == "test":
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out" and mode == "test":
        net = VNet_2out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out" and mode == "train":
        net = VNet_2out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_2out_2" and mode == "test":
        net = VNet_2out_2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_2out_2" and mode == "train":
        net = VNet_2out_2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_3out" and mode == "test":
        net = VNet_3out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_3out" and mode == "train":
        net = VNet_3out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_4out" and mode == "test":
        net = VNet_4out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_4out" and mode == "train":
        net = VNet_4out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "VNet_5out" and mode == "test":
        net = VNet_5out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "VNet_5out" and mode == "train":
        net = VNet_5out(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    return net