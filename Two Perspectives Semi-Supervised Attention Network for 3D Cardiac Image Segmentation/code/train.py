import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory_CBAP
from utils.cutmix import cut_module
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='CETUS,LA')
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='debug', help='exp_name')
parser.add_argument('--model', type=str, default='VNet_4out', help='model_name')
parser.add_argument('--max_iteration', type=int, default=6000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=1, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=4, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--slice_weight', type=float,  default=0.95, help='initial slice_weight')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model)

num_classes = 2
if args.dataset_name == "LA":
    # patch_size = (32, 32, 32)   # for debug use, quickly the training process
    patch_size = (112,112,80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
    args.max_iteration =6000
elif args.dataset_name == "CETUS":
    patch_size = (192,192,64)
    # patch_size = (32, 32, 32)
    args.root_path = args.root_path + 'data/CETUS'
    args.max_samples = 40
    args.max_iteration = 6000

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr
def get_current_slice_weight(epoch):
    return ramps.cosine_rampdown(epoch, args.consistency_rampup)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.base_lr

        # Initialize the parameters of both models to the same value first
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        # Afterwards, perform smooth updates on the ema model every time
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                # 0.99 * previous model parameters+0.01 * updated model parameters
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('../code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = net_factory_CBAP(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    m_seg1 = create_model()
    m_seg2 = create_model()
    # init the dataset
    if args.dataset_name == "LA":
        db_train1 = LAHeart_no_read1(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                                RandomRotFlip1(),
                               RandomCrop1(patch_size,args.slice_weight),
                               ToTensor(),
                           ]))
        db_train2 = LAHeart_no_read2(base_dir=train_data_path,
                                   split='train',
                                   transform=transforms.Compose([
                                       RandomRotFlip2(),
                                       RandomCrop2(patch_size, args.slice_weight),
                                       ToTensor(),
                                   ]))
    elif args.dataset_name == "CETUS":
        db_train1 = CETUS_no_read1(base_dir=train_data_path,
                                split='train',
                                transform=transforms.Compose([
                                    RandomRotFlip1(),
                                    RandomCrop3(patch_size, args.slice_weight),
                                    ToTensor(),
                                ]))
        db_train2 = CETUS_no_read2(base_dir=train_data_path,
                                        split='train',
                                        transform=transforms.Compose([
                                            RandomRotFlip2(),
                                            RandomCrop4(patch_size, args.slice_weight),
                                            ToTensor(),
                                        ]))
    #set the labeled num selection for the training dataset
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader1 = DataLoader(db_train1, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    trainloader2 = DataLoader(db_train2, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer1 = optim.SGD(m_seg1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(m_seg2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ema_optimizer1 = WeightEMA(m_seg1, m_seg2, alpha=0.99)
    ema_optimizer2 = WeightEMA(m_seg2, m_seg1, alpha=0.99)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader1)))
    MSE_cri = losses.mse_loss

    iter_num,best_dice1,best_dice2 = 0,0,0
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    kl_distance = nn.KLDivLoss(reduction='none')
    max_epoch = max_iterations // len(trainloader1) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, (sampled_batch1, sampled_batch2) in enumerate(zip(trainloader1, trainloader2)):
            volume_batch, label_batch ,maskz= sampled_batch1['image'], sampled_batch1['label'],sampled_batch1['weight']
            label_batch[label_batch==255]=1
            maskz = maskz[0:labeled_bs].cuda()
            maskzz = torch.unsqueeze(maskz, 1).cuda()
            maskzz = maskzz.repeat(1, 2, 1, 1, 1).cuda()
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            l_image,l_label = volume_batch[:args.labeled_bs],label_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            X = list(zip(l_image, l_label))
            U = unlabeled_volume_batch
            X_prime, U_prime, pseudo_label = cut_module(X, U, eval_net=m_seg2, num_augments=2, alpha=0.75,
                                                         mixup_modes='_x', aug_factor=torch.tensor(1).cuda())
            m_seg1.train()
            X_data = torch.cat([torch.unsqueeze(X_prime[0][0],0)],0) #  需要unsqueeze 一下
            X_label = torch.cat([torch.unsqueeze(X_prime[0][1], 0)],0)
            U_data = torch.cat([torch.unsqueeze(U_prime[0][0], 0), torch.unsqueeze(U_prime[1][0], 0)])

            X_data ,X_label = X_data.cuda(),X_label.cuda().float()
            U_data ,U_data_pseudo = U_data.cuda(),pseudo_label.cuda().float()

            X = torch.cat((X_data, U_data,volume_batch[args.labeled_bs:]), 0)

            out_1_all, out_2_all,out_3_all,out_4_all= m_seg1(X)

            out_1,out_2,out_3,out_4            = \
                out_1_all[:-args.labeled_bs],out_2_all[:-args.labeled_bs],out_3_all[:-args.labeled_bs],out_4_all[:-args.labeled_bs]
            out_1_u,out_2_u,out_3_u,out_4_u    = \
                out_1_all[-args.labeled_bs:],out_2_all[-args.labeled_bs:],out_3_all[-args.labeled_bs:],out_4_all[-args.labeled_bs:]
            out_1_s,out_2_s,out_3_s,out_4_s       = \
                torch.softmax(out_1, dim=1), torch.softmax(out_2, dim=1),torch.softmax(out_3, dim=1),torch.softmax(out_4, dim=1)
            out_1_u_s,out_2_u_s,out_3_u_s,out_4_u_s = \
                torch.softmax(out_1_u, dim=1), torch.softmax(out_2_u, dim=1), torch.softmax(out_3_u, dim=1),torch.softmax(out_4_u, dim=1)

            o_1_u_s = torch.softmax(out_1_all[args.labeled_bs:], dim=1)
            o_2_u_s = torch.softmax(out_2_all[args.labeled_bs:], dim=1)
            o_3_u_s = torch.softmax(out_3_all[args.labeled_bs:], dim=1)
            o_4_u_s = torch.softmax(out_4_all[args.labeled_bs:], dim=1)

            loss_seg_ce_lab, loss_seg_ce_unlab = 0, 0
            loss_seg_dice_lab, loss_seg_dice_unlab = 0, 0
            num_outputs = len(out_1_all)


            loss_seg_ce_lab+=losses.wce(out_1[:labeled_bs],X_label[:labeled_bs],maskzz,args.labeled_bs,patch_size[0],patch_size[0],patch_size[2])+ \
                             losses.wce(out_2[:labeled_bs], X_label[:labeled_bs], maskzz, args.labeled_bs, patch_size[0],patch_size[0], patch_size[2])+ \
                             losses.wce(out_3[:labeled_bs], X_label[:labeled_bs], maskzz, args.labeled_bs,patch_size[0], patch_size[0], patch_size[2]) + \
                             losses.wce(out_4[:labeled_bs], X_label[:labeled_bs], maskzz, args.labeled_bs, patch_size[0], patch_size[0], patch_size[2])


            loss_seg_dice_lab+=losses.dice_loss_weight(out_1_s[:labeled_bs, 1, :, :, :],label_batch[:labeled_bs] == 1,maskz)+ \
                               losses.dice_loss_weight(out_2_s[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1,maskz) + \
                               losses.dice_loss_weight(out_3_s[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1,maskz) + \
                               losses.dice_loss_weight(out_4_s[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1, maskz)

            loss_seg_ce_unlab += losses.wce(out_1[args.labeled_bs:], U_data_pseudo[args.labeled_bs:].unsqueeze(1),maskzz,args.labeled_bs,patch_size[0],patch_size[0],patch_size[2]) + \
                                 losses.wce(out_2[args.labeled_bs:], U_data_pseudo[args.labeled_bs:].unsqueeze(1),maskzz,args.labeled_bs,patch_size[0],patch_size[0],patch_size[2]) + \
                                 losses.wce(out_3[args.labeled_bs:], U_data_pseudo[args.labeled_bs:].unsqueeze(1),maskzz,args.labeled_bs,patch_size[0],patch_size[0],patch_size[2]) + \
                                 losses.wce(out_4[args.labeled_bs:], U_data_pseudo[args.labeled_bs:].unsqueeze(1),maskzz,args.labeled_bs,patch_size[0],patch_size[0],patch_size[2])

            loss_seg_dice_unlab += losses.dice_loss_weight(out_1_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1),maskz) + \
                                   losses.dice_loss_weight(out_2_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1),maskz) + \
                                   losses.dice_loss_weight(out_3_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1),maskz) + \
                                   losses.dice_loss_weight(out_4_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1), maskz)

            supervised_loss = 0.5 * (loss_seg_ce_lab + loss_seg_dice_lab)
            pseudo_loss = 0.5 * (loss_seg_dice_unlab + loss_seg_ce_unlab)
            preds = (o_1_u_s + o_2_u_s + o_3_u_s + o_4_u_s) / 4

            variance_1 = torch.sum(kl_distance(torch.log(o_1_u_s), preds), dim=1, keepdim=True)
            exp_variance_1 = torch.exp(-variance_1)
            variance_2 = torch.sum(kl_distance(torch.log(o_2_u_s), preds), dim=1, keepdim=True)
            exp_variance_2 = torch.exp(-variance_2)
            variance_3 = torch.sum(kl_distance(torch.log(o_3_u_s), preds), dim=1, keepdim=True)
            exp_variance_3 = torch.exp(-variance_3)
            variance_4 = torch.sum(kl_distance(torch.log(o_4_u_s), preds), dim=1, keepdim=True)
            exp_variance_4 = torch.exp(-variance_4)


            consis_dist_1 = (preds - o_1_u_s) ** 3
            consis_loss_1 = torch.mean(consis_dist_1 * exp_variance_1) / (torch.mean(exp_variance_1) + 1e-8) + torch.mean(variance_1)
            consis_dist_2 = (preds - o_2_u_s) ** 3
            consis_loss_2 = torch.mean(consis_dist_2 * exp_variance_2) / (torch.mean(exp_variance_2) + 1e-8) + torch.mean( variance_2)
            consis_dist_3 = (preds - o_3_u_s) ** 3
            consis_loss_3 = torch.mean(consis_dist_3 * exp_variance_3) / (torch.mean(exp_variance_3) + 1e-8) + torch.mean(variance_3)
            consis_dist_4 = (preds - o_4_u_s) ** 3
            consis_loss_4 = torch.mean(consis_dist_4 * exp_variance_4) / (
                        torch.mean(exp_variance_4) + 1e-8) + torch.mean(variance_4)

            sharp1 = sharpening(out_1_u_s)
            sharp2 = sharpening(out_2_u_s)
            sharp3 = sharpening(out_3_u_s)
            sharp4 = sharpening(out_4_u_s)

            loss_consist =  (consis_loss_1 +  consis_loss_2 + consis_loss_3 + consis_loss_4)/4\
                            +(MSE_cri(sharp1,out_1_u_s) + MSE_cri(sharp2,out_2_u_s) +
                               MSE_cri(sharp3,out_3_u_s)+MSE_cri(sharp4,out_4_u_s))/4


            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss1 = supervised_loss + pseudo_loss + consistency_weight*(loss_consist)

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            ema_optimizer1.step()
            update_ema_variables(m_seg1, m_seg2, 0.99, iter_num)

#model2
            volume_batch, label_batch, masky = sampled_batch2['image'], sampled_batch2['label'], sampled_batch2[
                'weight']
            label_batch[label_batch==255]=1
            masky = masky[0:labeled_bs].cuda()
            maskyy = torch.unsqueeze(masky, 1).cuda()
            maskyy = maskyy.repeat(1, 2, 1, 1, 1).cuda()
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            l_image, l_label = volume_batch[:args.labeled_bs], label_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            X = list(zip(l_image, l_label))
            U = unlabeled_volume_batch
            X_prime, U_prime, pseudo_label = cut_module(X, U, eval_net=m_seg1, num_augments=2, alpha=0.75,
                                                        mixup_modes='_x', aug_factor=torch.tensor(1).cuda())
            m_seg2.train()
            X_data = torch.cat([torch.unsqueeze(X_prime[0][0], 0)], 0)  # 需要unsqueeze 一下
            X_label = torch.cat([torch.unsqueeze(X_prime[0][1], 0)], 0)
            U_data = torch.cat([torch.unsqueeze(U_prime[0][0], 0), torch.unsqueeze(U_prime[1][0], 0)])

            X_data, X_label = X_data.cuda(), X_label.cuda().float()
            U_data, U_data_pseudo = U_data.cuda(), pseudo_label.cuda().float()

            X = torch.cat((X_data, U_data, volume_batch[args.labeled_bs:]), 0)

            out_1_all, out_2_all, out_3_all, out_4_all    = m_seg1(X)

            out_1, out_2, out_3, out_4 = \
                out_1_all[:-args.labeled_bs], out_2_all[:-args.labeled_bs], out_3_all[:-args.labeled_bs], out_4_all[:-args.labeled_bs]
            out_1_u, out_2_u, out_3_u, out_4_u = \
                out_1_all[-args.labeled_bs:], out_2_all[-args.labeled_bs:], out_3_all[-args.labeled_bs:], out_4_all[-args.labeled_bs:]
            out_1_s, out_2_s, out_3_s, out_4_s = \
                torch.softmax(out_1, dim=1), torch.softmax(out_2, dim=1), torch.softmax(out_3, dim=1), torch.softmax(out_4, dim=1)
            out_1_u_s, out_2_u_s, out_3_u_s, out_4_u_s = \
                torch.softmax(out_1_u, dim=1), torch.softmax(out_2_u, dim=1), torch.softmax(out_3_u,dim=1), torch.softmax(out_4_u, dim=1)

            o_1_u_s = torch.softmax(out_1_all[args.labeled_bs:], dim=1)
            o_2_u_s = torch.softmax(out_2_all[args.labeled_bs:], dim=1)
            o_3_u_s = torch.softmax(out_3_all[args.labeled_bs:], dim=1)
            o_4_u_s = torch.softmax(out_4_all[args.labeled_bs:], dim=1)


            loss_seg_ce_lab, loss_seg_ce_unlab = 0, 0
            loss_seg_dice_lab, loss_seg_dice_unlab = 0, 0

            loss_seg_ce_lab += losses.wce(out_1[:labeled_bs], X_label[:labeled_bs], maskyy, args.labeled_bs,
                                          patch_size[0], patch_size[0], patch_size[2]) + \
                               losses.wce(out_2[:labeled_bs], X_label[:labeled_bs], maskyy, args.labeled_bs,
                                          patch_size[0], patch_size[0], patch_size[2]) + \
                               losses.wce(out_3[:labeled_bs], X_label[:labeled_bs], maskyy, args.labeled_bs,
                                          patch_size[0], patch_size[0], patch_size[2]) + \
                               losses.wce(out_4[:labeled_bs], X_label[:labeled_bs], maskyy, args.labeled_bs,
                                          patch_size[0], patch_size[0], patch_size[2])


            loss_seg_dice_lab += losses.dice_loss_weight(out_1_s[:labeled_bs, 1, :, :, :],
                                                         label_batch[:labeled_bs] == 1, masky) + \
                                 losses.dice_loss_weight(out_2_s[:labeled_bs, 1, :, :, :],
                                                         label_batch[:labeled_bs] == 1, masky) + \
                                 losses.dice_loss_weight(out_3_s[:labeled_bs, 1, :, :, :],
                                                         label_batch[:labeled_bs] == 1, masky) + \
                                 losses.dice_loss_weight(out_4_s[:labeled_bs, 1, :, :, :],
                                                         label_batch[:labeled_bs] == 1, masky)


            loss_seg_ce_unlab += losses.wce(out_1[args.labeled_bs:], U_data_pseudo[args.labeled_bs:].unsqueeze(1),
                                            maskzz, args.labeled_bs, patch_size[0], patch_size[0], patch_size[2]) + \
                                 losses.wce(out_2[args.labeled_bs:], U_data_pseudo[args.labeled_bs:].unsqueeze(1),
                                            maskzz, args.labeled_bs, patch_size[0], patch_size[0], patch_size[2]) + \
                                 losses.wce(out_3[args.labeled_bs:], U_data_pseudo[args.labeled_bs:].unsqueeze(1),
                                            maskzz, args.labeled_bs, patch_size[0], patch_size[0], patch_size[2]) + \
                                 losses.wce(out_4[args.labeled_bs:], U_data_pseudo[args.labeled_bs:].unsqueeze(1),
                                            maskzz, args.labeled_bs, patch_size[0], patch_size[0], patch_size[2])


            loss_seg_dice_unlab += losses.dice_loss_weight(out_1_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1),
                                                           masky) + \
                                   losses.dice_loss_weight(out_2_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1),
                                                           masky) + \
                                   losses.dice_loss_weight(out_3_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1),
                                                           masky) + \
                                   losses.dice_loss_weight(out_4_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1),
                                                           masky)

            supervised_loss = 0.5 * (loss_seg_ce_lab + loss_seg_dice_lab)
            pseudo_loss = 0.5 * (loss_seg_dice_unlab + loss_seg_ce_unlab)
            preds = (o_1_u_s + o_2_u_s + o_3_u_s + o_4_u_s) / 4

            variance_1 = torch.sum(kl_distance(torch.log(o_1_u_s), preds), dim=1, keepdim=True)  # 只是用来计算kl，固定操作，多加一个log
            exp_variance_1 = torch.exp(-variance_1)
            variance_2 = torch.sum(kl_distance(torch.log(o_2_u_s), preds), dim=1, keepdim=True)
            exp_variance_2 = torch.exp(-variance_2)
            variance_3 = torch.sum(kl_distance(torch.log(o_3_u_s), preds), dim=1, keepdim=True)
            exp_variance_3 = torch.exp(-variance_3)
            variance_4 = torch.sum(kl_distance(torch.log(o_4_u_s), preds), dim=1, keepdim=True)
            exp_variance_4 = torch.exp(-variance_4)


            consis_dist_1 = (preds - o_1_u_s) ** 3
            consis_loss_1 = torch.mean(consis_dist_1 * exp_variance_1) / (
                        torch.mean(exp_variance_1) + 1e-8) + torch.mean(variance_1)
            consis_dist_2 = (preds - o_2_u_s) ** 3
            consis_loss_2 = torch.mean(consis_dist_2 * exp_variance_2) / (
                        torch.mean(exp_variance_2) + 1e-8) + torch.mean(variance_2)
            consis_dist_3 = (preds - o_3_u_s) ** 3
            consis_loss_3 = torch.mean(consis_dist_3 * exp_variance_3) / (
                        torch.mean(exp_variance_3) + 1e-8) + torch.mean(variance_3)
            consis_dist_4 = (preds - o_4_u_s) ** 3
            consis_loss_4 = torch.mean(consis_dist_4 * exp_variance_4) / (
                    torch.mean(exp_variance_4) + 1e-8) + torch.mean(variance_4)
            sharp1 = sharpening(out_1_u_s)
            sharp2 = sharpening(out_2_u_s)
            sharp3 = sharpening(out_3_u_s)
            sharp4 = sharpening(out_4_u_s)

            loss_consist = (consis_loss_1 + consis_loss_2 + consis_loss_3 + consis_loss_4) / 4\
                           + (MSE_cri(sharp1, out_1_u_s) + MSE_cri(sharp2, out_2_u_s) +
                              MSE_cri(sharp3, out_3_u_s) + MSE_cri(sharp4, out_4_u_s)) / 4

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss2 = supervised_loss + pseudo_loss + consistency_weight*(loss_consist)


            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            ema_optimizer2.step()
            update_ema_variables(m_seg2, m_seg1, 0.99, iter_num)


            iter_num = iter_num + 1
            if iter_num % 50 == 0:
                logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' % (
                    iter_num, loss1, supervised_loss, loss_consist))
            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice_lab, iter_num)
            writer.add_scalar('Labeled_loss/pseudo_loss', pseudo_loss, iter_num)
            writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg_ce_lab, iter_num)
            writer.add_scalar('Co_loss/consi stency_loss', loss_consist, iter_num)
            writer.add_scalar('Co_loss/consist_weight', consistency_weight, iter_num)
            if iter_num % 100 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.argmax(out_1_s, dim=1, keepdim=True)[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num >= 800 and iter_num % 200 == 0:
                m_seg1.eval()
                if args.dataset_name == "LA":
                    dice_sample1 = test_patch.var_all_case(m_seg1, num_classes=num_classes, patch_size=patch_size,
                                                           stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "CETUS":
                    dice_sample1 = test_patch.var_all_case(m_seg1, num_classes=num_classes, patch_size=patch_size,
                                                           stride_xy=18, stride_z=4, dataset_name='CETUS')
                if dice_sample1 > best_dice1:
                    best_dice1 = dice_sample1
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice1))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(m_seg1.state_dict(), save_mode_path)
                    torch.save(m_seg1.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample1, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice1, iter_num)
                m_seg1.train()
            if iter_num >= 800 and iter_num % 200 == 0:
                m_seg2.eval()
                if args.dataset_name == "LA":
                    dice_sample2 = test_patch.var_all_case(m_seg2, num_classes=num_classes, patch_size=patch_size,
                                                           stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "CETUS":
                    dice_sample2 = test_patch.var_all_case(m_seg2, num_classes=num_classes, patch_size=patch_size,
                                                           stride_xy=18, stride_z=4, dataset_name='CETUS')
                if dice_sample2 > best_dice2:
                    best_dice2 = dice_sample2
                    save_mode_path2 = os.path.join(snapshot_path, 'iter_{}_dice_{}2.pth'.format(iter_num, best_dice2))
                    save_best_path2 = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))

                    torch.save(m_seg2.state_dict(), save_mode_path2)
                    torch.save(m_seg2.state_dict(), save_best_path2)
                    logging.info("save best model to {}".format(save_mode_path2))
                writer.add_scalar('Var_dice/Dice', dice_sample2, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice2, iter_num)
                m_seg1.train()

            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                save_mode_path2 = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '2.pth')
                torch.save(m_seg1.state_dict(), save_mode_path)
                torch.save(m_seg2.state_dict(), save_mode_path2)
                logging.info("save model to {}".format(save_mode_path))
                logging.info("save model to {}".format(save_mode_path2))
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    save_mode_path2 = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '2.pth')
    torch.save(m_seg1.state_dict(), save_mode_path)
    torch.save(m_seg2.state_dict(), save_mode_path2)
    logging.info("save model to {}".format(save_mode_path))
    logging.info("save model to {}".format(save_mode_path2))
    writer.close()
