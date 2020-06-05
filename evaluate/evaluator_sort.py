# coding=utf-8
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/03/06
@Version  ：0.1
@Desc   ：用于对生成的显著性图片进行测评，测评结果按照图片在该指标的得分进行排序。实现方式，直接将得分结果加在生成文件名称前
=================================================='''
import os
import time
import numpy as np
import torch
from torchvision import transforms
from shutil import copyfile
import logger as log
import traceback
mae_map = {}
sm_map = {}
em_map = {}
fm_map = {}


def save_img(self,image_path, gt_path, sod_result_path, score_compare_path, fm_map):

    # 其实这里可以不用排序
    # dict2 = sorted(fm_map.items(), key=lambda x: x[1], reverse=True)

    if not os.path.exists(score_compare_path):
        os.mkdir(score_compare_path)
    # print("fm_map.keys() is")
    # print(fm_map.keys())
    for dict in fm_map.keys():
        # print("dict is ==== ")
        # print(dict)
        img_name = dict
        try:
             img_score = '%.2f'%fm_map[dict]
        except Exception as e:
            exstr = traceback.format_exc()
            self.logger_result.error("errore dict is ",dict)
            self.logger_result.error(exstr)

        img_tmp_path = image_path + img_name + ".jpg"
        gt_tmp_path = gt_path + img_name + ".png"
        sod_result_tmp_path = sod_result_path + img_name + ".png"

        score_tmp_image_path = score_compare_path + img_score + "_" + img_name + "_image_0"+ ".png"
        copyfile(img_tmp_path,score_tmp_image_path)

        score_tmp_gt_path = score_compare_path + img_score + "_" + img_name + "_gt_1"+ ".png"
        copyfile(gt_tmp_path, score_tmp_gt_path)

        score_tmp_result_path = score_compare_path + img_score + "_" + img_name + "_result_2"+ ".png"
        copyfile(sod_result_tmp_path, score_tmp_result_path)


class Eval_thread():

    def __init__(self, loader, model_name, method, dataset, image_path, gt_path, result_path, score_image_save_path,
                 cuda, logger=None, isNeedSort=True):
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.model_name = model_name
        self.image_path = image_path
        self.gt_path = gt_path
        self.result_path = result_path
        self.fm_result_path = score_image_save_path + "/fm/"
        self.sm_result_path = score_image_save_path + "/sm/"
        # 是否需要存储排序结果
        self.isNeedSort = isNeedSort
        # self.logfile = os.path.join(output_dir, 'result.txt')
        if logger != None:
            self.logger_result = logger
        else:
            self.logger_result = log.setup_logger("compare_result", './log/', 0)

    def run(self):
        start_time = time.time()
        mae = 0
        max_e = 0
        max_f, mean_f, median_f = 0.00,0.00,0.00
        s = 0.00
        mae = self.Eval_mae()
        max_f, mean_f ,median_f= self.Eval_fmeasure()
        max_e = self.Eval_Emeasure()
        s = self.Eval_Smeasure()
        print("=============================================")
        print(" model:",self.model_name," test dataset: ", self.dataset," backbone: " ,self.method," mae: ",mae," max_f: ",max_f," mean_f: ",mean_f," median_f: ",median_f," s: ",s)
        self.logger_result.info("===============evaluate finished!!=================")
        self.logger_result.info('evaluate finished!!! [cost:{:.4f}s]   model: {}  test  dataset: {} with  method: {}'.format
            ((time.time() - start_time), self.model_name, self.dataset, self.method ))

        self.logger_result.info("==================result======================")
        self.logger_result.info(
            'model:{} test dataset: {} result is mae: {:.4f}, max-fmeasure: {:.4f},  mean-fmeasure: {:.4f},  median-fmeasure: {:.4f}， max-Emeasure: {:.4f} , S-measure: {:.4f}..\n'.format(
                self.model_name, self.dataset,mae,max_f, mean_f, median_f, max_e, s))

        if self.isNeedSort:
            self.logger_result.info("start generate score sort image!......")
            save_img(self,self.image_path,self.gt_path,self.result_path,self.fm_result_path,fm_map)
            save_img(self,self.image_path, self.gt_path, self.result_path, self.sm_result_path, sm_map)
            self.logger_result.info("Congratulation! generate score sort image finished!!......")


    def Eval_mae(self):
        self.logger_result.info('eval[MAE]:model:{} dataset:{} with method:{}.'.format(self.model_name, self.dataset, self.method))
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt, name in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item()

    def Eval_fmeasure(self):
        self.logger_result.info('eval[FMeasure]:model:{} dataset:{} with method:{}.'.format(self.model_name, self.dataset, self.method))
        beta2 = 0.3
        avg_p, avg_r, img_num = 0.0, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt, name in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                prec, recall = self._eval_pr(pred, gt, 255)
                # 记录某个图片的fm值
                fm = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                fm[fm != fm] = 0  # for Nan
                fm_map[name] = fm.mean().item()
                avg_p += prec
                avg_r += recall
                img_num += 1.0
            avg_p /= img_num
            avg_r /= img_num
            # TODO 将低分内容写到一个文件夹中，然后自动将内容将低分图片存放到一个文件夹中，方便分析
            score = (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
            score[score != score] = 0  # for Nan
            # 计算下 平均F
            return score.max().item(), score.mean().item(), score.median().item()

    def Eval_Emeasure(self):
        self.logger_result.info('eval[EMeasure]:model:{}  dataset:{} with method:{} .'.format(self.model_name,self.dataset, self.method))
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt, name in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                max_e = self._eval_e(pred, gt, 255)
                if max_e == max_e:
                    avg_e += max_e
                    img_num += 1.0
                # 计算下 平均E
            avg_e /= img_num
            return avg_e

    def Eval_Smeasure(self):
        self.logger_result.info('eval[SMeasure]:model:{}  dataset:{} with method:{}.'.format(self.model_name, self.dataset, self.method))
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        nan_size = 0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt, name in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                img_num += 1.0

                Q[Q != Q] = 0  # for Nan
                sm_map[name] = Q.item()

                if torch.isnan(Q.data):
                    nan_size += 1
                    self.logger_result.info(" NaN image is {}".format(name))
                    self.logger_result.info("Q is {:.3f}".format(Q.data))
                else:
                    avg_q += Q.data

            avg_q /= (img_num - nan_size)
            # logger_result.info("avg_q is {}"avg_q ," img_num is {}",(img_num-nan_size) )
            return avg_q


    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
        else:
            score = torch.zeros(num)
        for i in range(num):
            fm = y_pred - y_pred.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score.max()

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        # print(Q)
        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0, cols)).cuda().float()
                j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q


if __name__ == '__main__':

    fm_map = {"ILSVRC2012_test_00000003":"0.99", "ILSVRC2012_test_00000023":"0.61",
           "ILSVRC2012_test_00000025":"0.78"}

    # 从大到小排序
    dict2 = sorted(fm_map.items(), key=lambda x: x[1], reverse=True)

    image_path = r'C:/Users/yuan/Desktop/DUTS-TE-CPD-Edge/model2/img/'
    gt_path = "C:/Users/yuan/Desktop/DUTS-TE-CPD-Edge/model2/gt/"
    result_path = "C:/Users/yuan/Desktop/DUTS-TE-CPD-Edge/model2/result/"
    fm_result = "C:/Users/yuan/Desktop/DUTS-TE-CPD-Edge/model2/fm/"

    if not os.path.exists(fm_result):
        os.mkdir(fm_result)
    for dict in dict2:
        img_tmp_path = image_path + dict[0] + ".jpg"
        gt_tmp_path = gt_path + dict[0] + ".png"
        result_tmp_path = result_path + dict[0] + ".png"

        img_file = open(img_tmp_path, "r")
        gt_file = open(gt_tmp_path, "r")
        result_file = open(result_tmp_path, "r")

        fm_tmp_image_path = fm_result + dict[1] + "_" + dict[0] + "_0"+ ".png"
        copyfile(img_tmp_path,fm_tmp_image_path)

        fm_tmp_gt_path = fm_result + dict[1] + "_" + dict[0] + "_gt_1"+ ".png"
        copyfile(gt_tmp_path, fm_tmp_gt_path)

        fm_tmp_result_path = fm_result + dict[1] + "_" + dict[0] + "_result_2"+ ".png"
        copyfile(result_tmp_path, fm_tmp_result_path)

    # 需要保存 image 、gt、result
    # dict2 = sorted(map.items(), key=lambda x: x[1])
    # for data in dict2:
    #     print(data[0])
    # 根据文件将图片写到一个路径中

    # path = save_sort_results(dict2, "test_sort", "fm_sort.txt")
    # 根据文件将图片写到一个路径中
    # 知道文件名称 需要加入一个路径

    #
    # print(map.values())
    #
    # print(map)
