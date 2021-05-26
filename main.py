import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from datasets.mvtec import MVTecDataset
from models.resnet_backbone import modified_resnet18
from utils.util import  time_string, convert_secs2time, AverageMeter
from utils.functions import cal_anomaly_maps, cal_loss
from utils.visualization import plt_fig


class STPM():
    def __init__(self, args):
        self.device = args.device
        self.data_path = args.data_path
        self.obj = args.obj
        self.img_resize = args.img_resize
        self.img_cropsize = args.img_cropsize
        self.validation_ratio = args.validation_ratio
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.vis = args.vis
        self.model_dir = args.model_dir
        self.img_dir = args.img_dir

        self.load_model()
        self.load_dataset()

        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model_s.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)


    def load_dataset(self):
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=True, resize=self.img_resize, cropsize=self.img_cropsize)
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, **kwargs)

    def load_model(self):
        self.model_t = modified_resnet18().to(self.device)
        self.model_s = modified_resnet18(pretrained=False).to(self.device)
        for param in self.model_t.parameters():
            param.requires_grad = False
        self.model_t.eval()

    def train(self):
        self.model_s.train()
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        for epoch in range(1, self.num_epochs+1):
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * ((self.num_epochs+1) - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
            print('{:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, self.num_epochs, time_string(), need_time))
            losses = AverageMeter()
            for (data, _, _) in tqdm(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    features_t = self.model_t(data)
                    features_s = self.model_s(data)
                    loss = cal_loss(features_s, features_t)

                    losses.update(loss.sum().item(), data.size(0))
                    loss.backward()
                    self.optimizer.step()
            print('Train Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))

            val_loss = self.val(epoch)
            if best_score is None:
                best_score = val_loss
                self.save_checkpoint()
            elif val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()
            
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        print('Training end.')
    
    def val(self, epoch):
        self.model_s.eval()
        losses = AverageMeter()
        for (data, _, _) in tqdm(self.val_loader):
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t = self.model_t(data)
                features_s = self.model_s(data)
                loss = cal_loss(features_s, features_t)
                losses.update(loss.item(), data.size(0))
        print('Val Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))

        return losses.avg

    def save_checkpoint(self):
        print('Save model !!!')
        state = {'model':self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, 'model_s.pth'))

    def test(self):
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, 'model_s.pth'))
        except:
            raise Exception('Check saved model path.')
        self.model_s.load_state_dict(checkpoint['model'])
        self.model_s.eval()

        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        test_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=False, resize=self.img_resize, cropsize=self.img_cropsize)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)

        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        print('Testing')
        for (data, label, mask) in tqdm(test_loader):
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            gt_mask_list.extend(mask.squeeze().cpu().numpy())

            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t = self.model_t(data)
                features_s = self.model_s(data)

                score = cal_anomaly_maps(features_s, features_t, self.img_cropsize)
            scores.extend(score)

        scores = np.asarray(scores)
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        print('image ROCAUC: %.3f' % (img_roc_auc))

        precision, recall, thresholds = precision_recall_curve(gt_list.flatten(), img_scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        cls_threshold = thresholds[np.argmax(f1)]

        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        seg_threshold = thresholds[np.argmax(f1)]

        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        if self.vis:
            plt_fig(test_imgs, scores, img_scores, gt_mask_list, seg_threshold, cls_threshold, 
                    self.img_dir, self.obj)


def get_args():
    parser = argparse.ArgumentParser(description='STPM anomaly detection')
    parser.add_argument('--phase', default='train')
    parser.add_argument("--data_path", type=str, default="D:/dataset/mvtec_anomaly_detection")
    parser.add_argument('--obj', type=str, default='zipper')
    parser.add_argument('--img_resize', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=224)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--vis', type=eval, choices=[True, False], default=True)
    parser.add_argument("--save_path", type=str, default="./mvtec_results")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.model_dir = args.save_path + '/models' + '/' + args.obj
    args.img_dir = args.save_path + '/imgs' + '/' + args.obj
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)

    stpm = STPM(args)
    if args.phase == 'train':
        stpm.train()
        stpm.test()
    elif args.phase == 'test':
        stpm.test()
    else:
        print('Phase argument must be train or test.')







    

