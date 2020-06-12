import torch
import torchvision
import torch.nn as nn
import os
import glob
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
import random

rand_seed = 22
if rand_seed is not None:
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    

from modeling import *
from dataset import *
from utils.functions import *
from utils.losses import *
from utils import pytorch_ssim
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob


def cal_mse(est,gt,args):
    if args.curriculum == 'W':
        delta = (est - gt)**2
        k_w = 2e-3 * args.value_factor * args.downsample**2
        b_w = 5e-3 * args.value_factor * args.downsample**2
        T = torch.ones_like(gt, dtype=torch.float32) * epoch * k_w + b_w
        W = T / torch.max(T,est)
        delta = delta * W
        mse_loss = torch.mean(delta)
        return mse_loss
    else:
        return torch.mean((est - gt)**2)
    
    
def cal_loss(output, target, args):
    
    if args.loss == 'mse+lc':
        loss = cal_mse(output, target, args) + 1e2 * cal_lc_loss(output, target) * args.downsample
    elif args.loss == 'ssim':
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        loss = 1 - ssim_loss(output, target)
    elif args.loss == 'mse+ssim':
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        loss = cal_mse(output, target, args) + (1 - ssim_loss(output,target))
    elif args.loss == 'lsa+lsc':
        loss = cal_spatial_abstraction_loss(output, target) + cal_spatial_correlation_loss(output, target)
    elif args.loss == 'lsa':
        loss = cal_spatial_abstraction_loss(output, target)
    elif args.loss == 'dms-ssim':
        loss = cal_dms_ssim_loss(output, target, dilations=[1,2,3,6,9])
    elif args.loss == 'ms-ssim':
        loss = cal_ms_ssim(output, target)
    elif 'avg-ms-ssim' in args.loss:
        level = int(args.loss[0])
        loss = cal_avg_ms_ssim(output, target, level)
        '''est_c = output.sum()
        gt_c = target.sum()
        attention_factor = max(min((max(est_c, gt_c)+1e-1)/(min(est_c, gt_c)+1e-1), 5),1)
        loss *= attention_factor'''
    
    elif 'adver' in args.loss:
        valid = Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(1.0), requires_grad=False)
        g_loss = adversarial_loss(discriminator(output), valid)
        loss = cal_mse(output, target, args) + g_loss
    elif 'bayes' in args.loss:
        points, target, st_sizes, post_prob, bayes_criterion  = target
        st_sizes = st_sizes.cuda()
        
        points = [p.cuda() for p in points]
        target = [t.cuda() for t in target]
        
        prob_list = post_prob(points, st_sizes)
        loss = bayes_criterion(prob_list, target, output)
    elif args.loss == 'mse+mae':
        loss = cal_mse(output, target, args) + 2 * torch.mean(torch.abs(output - target))
    elif args.loss == 'mse+count':
        est = output.sum()+1e-2
        gt = target.sum()+1e-2
        loss = cal_mse(output, target, args) + 1e-5 * torch.abs(1 - est/gt)
    else:
        loss = cal_mse(output, target, args)
    return loss

def main(args):
    # use gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cur_device = torch.device('cuda:{}'.format(args.gpu))
    
    if 'bayes' in args.loss:
        if args.dataset=='sha':
            root = '/home/datamining/Datasets/CrowdCounting/sha_bayes_512/'
            train_path = root+'train/'
            test_path = root+'test/'
        elif args.dataset =='qnrf':
            root = '/home/datamining/Datasets/CrowdCounting/UCF-Train-Val-Test/'
            train_path = root+'train/'
            test_path = root+'test/'
        train_loader, test_loader, train_img_paths, test_img_paths = get_loader(train_path, test_path, args)
    else:
        train_loader, test_loader = get_loader_json(args)
    downsample_ratio = args.downsample
    model_dict = {'M_VGG': M_VGG, 'DefCcNet':DefCcNet, 'Res50':Res50, 'InceptionV3':Inception3CC, 'CAN':CANNet, 'CSRNet':CSRNet, 'U_VGG':U_VGG, 'M_VGG10':M_VGG10, 'SANet':SANet, 'M_VGG7':M_VGG7, 'Res2C':Res2C, 'RefCC': RefCC, 'RefUNet':RefUNet, 'PlusNet':PlusNet, 'UpNet':UpNet, 'MCNN':MCNN, 'MARNet':MARNet}
    model_name = args.model
    dataset_name = args.dataset
    if args.model in ['M_VGG', 'M_VGG10', 'U_VGG', 'RefUNet', 'PlusNet', 'UpNet', 'MARNet']:
        net = model_dict[model_name](downsample=args.downsample, bn=args.bn>0, objective=args.objective, sp=(args.sp>0), se=(args.se>0), NL=args.nl, block=args.block)
    else:
        net = model_dict[model_name]()
    net.cuda()
    if args.bn>0:
        save_name = '{}_{}_s{}_{}_lr{}'.format(model_name, dataset_name, str(args.crop_scale), args.loss, args.lr)
    else:
        save_name = '{}_d{}{}{}{}{}_{}_{}{}_{}{}{}{}{}{}{}{}'.format(model_name, str(args.downsample), '_sp' if args.sp else '', '_se' if args.se else '', '_'+args.nl if args.nl!='relu' else '', '_vp' if args.val_patch else '', dataset_name, args.crop_mode, '_cr'+str(args.crop_scale) if args.crop_mode!='one' else '', args.loss, '_wu' if args.warm_up else '', '_cl' if args.curriculum=='W' else '', '_v'+str(int(args.value_factor)) if args.value_factor!=1 else '', '_amp'+str(args.amp_k) if args.objective=='dmp+amp' else '', '_bg' if args.use_bg and 'bayes' in args.loss else '', '_'+args.block if args.block else '', '_lsn'+str(args.loss_n) if args.loss_n>1 else '')
    save_path = "/home/datamining/Models/CrowdCounting/"+save_name+".pth"
    logger = get_logger('logs/'+save_name+'.txt')
    for k, v in args.__dict__.items():  # save args
        logger.info("{}: {}".format(k, v))
    print()
    if os.path.exists(args.resume) and args.resume:
        net.load_state_dict(torch.load(args.resume))
        print('{} loaded!'.format(args.resume))
    
    value_factor=args.value_factor
    freq = args.print_freq
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == 'SGD':
        # sometimes not converage
        optimizer=torch.optim.SGD(net.parameters(),lr=args.lr, momentum=0.95, weight_decay=args.decay)
    
    if args.loss=='bayes+':
        bayes_criterion = Bay_Loss(True, cur_device)
        post_prob = Post_Prob(sigma=8, c_size=args.crop_scale, stride=1, background_ratio=0.1, use_background=True, device=cur_device)
    elif args.loss=='bayes++':
        bayes_criterion = Bay_Loss(True, cur_device)
        post_prob = Post_Prob(sigma=0, c_size=args.crop_scale, stride=1, background_ratio=0.1, use_background=True, device=cur_device)
    else:
        mse_criterion = nn.MSELoss().cuda()
        mae_critetion = nn.L1Loss().cuda()
    
    if args.scheduler == 'plt':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8,patience=30, verbose=True)
    elif args.scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=0)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,step_size=50, gamma=0.5)
    elif args.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.scheduler == 'cyclic' and args.optimizer == 'SGD':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.lr*0.01, max_lr=args.lr, step_size_up=25,)
    elif args.scheduler == 'None':
        scheduler = None
    else:
        print('scheduler name error!')
    
    if args.resume == 0:
        mae, rmse =1e6, 1e6
    elif args.dataset == 'we':
        mae, rmse = 0.0, 0.0
        for loader in test_loader:
            cur_mae, cur_rmse = val(net, loader, value_factor)
            mae += cur_mae
            rmse += cur_rmse
        mae /= len(test_loader)
        rmse /= len(test_loader)
    elif args.val_patch:
        mae, rmse = val_patch(net, test_loader, value_factor)
    elif 'bayes' in args.loss:
        mae, rmse = val_bayes(net, test_loader, value_factor)
    else:
        mae, rmse = val(net, test_loader, value_factor)
    best_mae, best_rmse = mae, rmse
    
    if 'adver' in args.loss:
        discriminator = Discriminator().cuda()
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
    for epoch in range(args.epochs):
        if args.crop_mode == 'curriculum':
            # every 20%, change the dataset
            if (epoch+1) % (args.epochs//5) == 0:
                print('change dataset')
                single_dataset = RawDataset(train_img_paths, transform, args.crop_mode, downsample_ratio, args.crop_scale, (epoch+1.0+args.epochs//5)/args.epochs)
                train_loader = torch.utils.data.DataLoader(single_dataset, shuffle=True, batch_size=1, num_workers=8)
        
        train_loss = 0.0
        if 'bayes' in args.loss:
            epoch_mae = AverageMeter()
            epoch_mse = AverageMeter()
        net.train()
        if args.warm_up and epoch < args.warm_up_steps:
            linear_warm_up_lr(optimizer, epoch, args.warm_up_steps,args.lr)
        for it, data in enumerate(train_loader):
            if 'bayes' in args.loss:
                #inputs, points, targets, st_sizes=data
                inputs, target = data[0], data[1:]
                img = inputs.to(cur_device)
                amp_gt = target[-1].cuda()
                target = [t for t in target[:-1]] + [post_prob, bayes_criterion]
            else:
                img, target, _, amp_gt = data
                img = img.cuda()
                target = value_factor * target.float().unsqueeze(1).cuda()
                amp_gt = amp_gt.cuda()
            #print(img.shape)
            optimizer.zero_grad()
            
            #print(target.shape)
            if args.model in ['RefCC', 'RefUNet']:
                if args.objective =='dmp+amp':
                    output, d0, d1, d2, d3, d4, d5, amp = net(img)
                else:
                    output, d0, d1, d2, d3, d4, d5 = net(img)
            elif args.model in ['U_VGG', 'PlusNet', 'UpNet']:
                if args.objective =='dmp+amp':
                    output, d0, d1, d2, d3, d4, amp = net(img)
                else:
                    output, d0, d1, d2, d3, d4 = net(img)
            elif args.model == 'MARNet':
                output, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = net(img)
            elif args.objective == 'dmp+amp':
                output, amp = net(img)
            else:
                output = net(img)
            
            loss = cal_loss(output, target, args)
            if args.loss_n>=2:
                loss += cal_loss(d0, target, args)
            if args.loss_n>=3:
                loss += cal_loss(d1, target, args)
            if args.loss_n>=4:
                loss += cal_loss(d2, target, args)
            if args.loss_n>=5:
                loss += cal_loss(d3, target, args)
            if args.loss_n>=6:
                loss += cal_loss(d4, target, args)
            if args.loss_n>=7:
                loss += cal_loss(d5, target, args)
            
             # add the cross entropy loss for attention map
            if args.objective == 'dmp+amp':
                if args.model == 'MARNet':
                    for amp in [amp41, amp31, amp21, amp11, amp01]:
                        amp_gt_us = amp_gt.unsqueeze(0)
                        if amp_gt_us.shape[2:]!=amp.shape[2:]:
                            amp_gt_us = F.interpolate(amp_gt_us, amp.shape[2:], mode='bilinear')
                        cross_entropy = (amp_gt_us * torch.log(amp+1e-10) + (1 - amp_gt_us) * torch.log(1 - amp+1e-10)) * -1
                        cross_entropy_loss = torch.mean(cross_entropy)
                        loss = loss + cross_entropy_loss * args.amp_k
                        #SsimLoss = pytorch_ssim.SSIM(window_size=11)
                        #ssim_loss = cal_avg_ms_ssim(amp, amp_gt_us, level=3)
                        #loss = loss + ssim_loss * args.amp_k
                else:
                    cross_entropy = (amp_gt * torch.log(amp+1e-10) + (1 - amp_gt) * torch.log(1 - amp+1e-10)) * -1
                    cross_entropy_loss = torch.mean(cross_entropy)
                    loss = loss + cross_entropy_loss * args.amp_k
                
            loss.backward()
            optimizer.step()
            data_loss = loss.item()
            train_loss += data_loss
            if 'bayes' in args.loss:
                N = inputs.size(0)
                pre_count = torch.sum(output.view(N, -1), dim=1).detach().cpu().numpy()
                points = target[0]
                gd_count = np.array([len(p) for p in points], dtype=np.float32)
                res = pre_count - gd_count
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
            if 'adver' in args.loss:
                optimizer_D.zero_grad()
                
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(target), valid)
                fake_loss = adversarial_loss(discriminator(output.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
        
                d_loss.backward()
                optimizer_D.step()
            if it%freq==0:
                if 'bayes' in args.loss:
                    print('[ep:{}], [it:{}], [loss:{:.4f}], pre_count:{:.1f}, gt_count:{:.1f}'.format(epoch+1, it, data_loss, pre_count[0], gd_count[0]))
                else:
                    print('[ep:{}], [it:{}], [loss:{:.8f}], [output:{:.2f}, target:{:.2f}]'.format(epoch+1, it, data_loss, output[0].sum().item(), target[0].sum().item()))
        if (args.lazy_val and epoch > 0.5 * args.epochs) or (args.lazy_val and epoch < 0.5 * args.epochs and epoch % 5 == 0) or args.lazy_val < 1:
            if args.dataset == 'we':
                mae, rmse = 0.0, 0.0
                for loader in test_loader:
                    cur_mae, cur_rmse = val(net, loader, value_factor)
                    mae += cur_mae
                    rmse += cur_rmse
                mae /= len(test_loader)
                rmse /= len(test_loader)
            elif args.val_patch:
                mae, rmse = val_patch(net, test_loader, value_factor)
            elif 'bayes' in args.loss:
                mae, rmse = val_bayes(net, test_loader, value_factor)
            else:
                mae, rmse = val(net, test_loader, value_factor)
            
            if mae + 0.1 * rmse < best_mae + 0.1 * best_rmse:
                best_mae, best_rmse = mae, rmse
                torch.save(net.state_dict(), save_path)
        
        if not (args.warm_up and epoch < args.warm_up_steps):
            if args.scheduler == 'plt':
                scheduler.step(train_loss/len(train_loader))
            elif args.scheduler != 'None':
                scheduler.step()
        if 'bayes' in args.loss:
            logger.info('{} Epoch {}/{} Loss:{:.4f},lr:{:.7f}, [Train]{:.1f}, {:.1f}, [VAL]:{mae:.1f}, {rmse:.1f}, [Best]:{b_mae:.1f}, {b_rmse:.1f}'.format(model_name, epoch+1, args.epochs, train_loss/len(train_loader), optimizer.param_groups[0]['lr'], epoch_mae.get_avg(), np.sqrt(epoch_mse.get_avg()), mae=mae, rmse=rmse, b_mae=best_mae, b_rmse=best_rmse))
        else:
            logger.info('{} Epoch {}/{} Loss:{:.6f}, lr:{:.7f}, [CUR]:{mae:.1f}, {rmse:.1f}, [Best]:{b_mae:.1f}, {b_rmse:.1f}'.format(model_name, epoch+1, args.epochs, train_loss/len(train_loader), optimizer.param_groups[0]['lr'], mae=mae, rmse=rmse, b_mae=best_mae, b_rmse=best_rmse))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Crowd Counting')
    parser.add_argument('--model', metavar='model name', default='MARNet', choices=['M_VGG', 'U_VGG', 'DefCcNet', 'InceptionV3', 'CAN', 'Res50', 'CSRNet', 'M_VGG10', 'SANet', 'M_VGG7', 'Res2C', 'RefCC', 'RefUNet', 'PlusNet', 'UpNet', 'MCNN', 'MARNet'], type=str)
    parser.add_argument('--downsample', metavar='downsample ratio', default=1, choices=[1, 2, 4, 8], type=int)
    parser.add_argument('--dataset', metavar='dataset name', default='sha', choices=['sha','shb','qnrf', 'we', 'mall'], type=str)
    parser.add_argument('--resume', metavar='resume model if exists', default='', type=str)
    parser.add_argument('--lr', type=float, default=1e-5, help='the initial learning rate')
    parser.add_argument('--gpu', default='0', help='assign device')
    parser.add_argument('--scheduler', default='step', help='lr scheduler', choices=['plt', 'cos', 'step', 'cyclic', 'exp', 'None'], type=str)
    parser.add_argument('--optimizer', default='Adam', help='optimizer', choices=['Adam','SGD'], type=str)
    parser.add_argument('--decay', default=1e-4, help='weight decay', type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lazy_val', default=1, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--train_json', metavar='TRAIN', type=str, default='json/sha_train.json', help='path to train json')
    parser.add_argument('--val_json', metavar='VAL', type=str, default='json/sha_val.json', help='path to val json')
    
    parser.add_argument('--loss', default='3avg-ms-ssim', choices=['mse','mse+lc','ssim','mse+ssim','lsa+lsc','lsa','dms-ssim', 'ms-ssim','bayes+','bayes++','adversial', '2avg-ms-ssim', '3avg-ms-ssim', '4avg-ms-ssim', '5avg-ms-ssim', 'mse+mae', 'mse+count'])
    parser.add_argument('--val_patch', metavar='val on patch if set to True', default=0, choices=[0,1], type=int)
    
    parser.add_argument('--crop_mode', default='random', choices=['random', 'one', 'fixed+random', 'fixed', 'mixed', 'curriculum'], type=str)
    parser.add_argument('--crop_scale', metavar='patch scale, crop size = size*scale when scale<1, crop size=(scale, scale) when scale>1', default=0.5, type=float)
    
    parser.add_argument('--warm_up', default=0, help='warm up from 0.1*lr to lr by warm up steps', type=int)
    parser.add_argument('--warm_up_steps', default=10, help='warm up steps', type=int)
    
    parser.add_argument('--curriculum', default='None', metavar='curriculum learning', choices=['None','W'])
    parser.add_argument('--value_factor', default=1.0, metavar='value factor * gt', type=float)
    parser.add_argument('--objective', default='dmp', choices=['dmp', 'dmp+amp'], type=str)
    parser.add_argument('--amp_k', default=0.1, help="only work when objective is 'dmp+amp'. loss = loss + k * cross_entropy_loss", type=float)
    parser.add_argument('--use_bg', default=1, help='if using background images(without any person) in training', choices=[0,1], type=int)
    
    parser.add_argument('--bn', default=0, help='if using batch normalization', type=int)
    parser.add_argument('--bs', default=1, help='batch size if using bn', type=int)
    parser.add_argument('--random_crop_n', default=1, metavar='random crop number for each image, only work when using bn', type=int)
    parser.add_argument('--loss_n', default=6, help='loss count', type=int)
    
    parser.add_argument('--sp', default=0, help='spatial pyramid module', type=int)
    parser.add_argument('--se', default=0, help='squeeze excitation module', type=int)
    parser.add_argument('--nl', default='relu', help='non-linear layer', choices=['relu', 'prelu', 'swish'], type=str)
    parser.add_argument('--block', default='', help='pyramid type or not', choices=['', 'dilation', 'depth', 'size', 'Dense', 'DenseRes', 'Res'], type=str)
    args = parser.parse_args()
    
    main(args)
    
