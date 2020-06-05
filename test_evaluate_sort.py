# encoding=UTF-8
import torch
import torch.nn.functional as F
from evaluate.evaluator_sort import Eval_thread
from evaluate.dataloader import EvalDataset
import numpy as np
import os, argparse
import torchvision.utils as vutils
from torch.autograd import Variable
from model.ResNet_models import SCRN
from evaluate.data import test_dataset,validate_dataset
import logger as log
import time
import torchvision.transforms as transforms
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/03/06
@Version  ：0.1
@Desc   ：用于测试模型效果，主要流程先利用模型识别图片，然后再对生成的图片进行跑分，并将图片得分按照得分进行排序
=================================================='''
# parser = argparse.ArgumentParser()
# parser.add_argument('--testsize', type=int, default=352, help='testing size')
# parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
# parser.add_argument('--is_unmerge', type=bool, default=False, help='True or False')
# parser.add_argument('--model_type', type=int, default=1, help='1 or 2 or 3')
# parser.add_argument('--is_need_process_image', type=bool, default=True, help='True or False')
# parser.add_argument('--test_datasets', type=str, default='ECSSD2', help='test_datasets Name')
#
# opt = parser.parse_args()

# dataset_path = '/home/ihavc01/LZY/SCRN/DUTS/'

# 测试数据集名称
test_datasets=['DUTS-TE']
# 测试模型名称
model_name ="SCRN-RFB-ASPP"
# 测试模型的backbone
method_names = ['Res_Net']
# 测试数据集路径
dataset_path = '/home/ihavc/datasets/'
# dataset_path = '/home/ihavc01/LZY/SCRN/DUTS/'
# 测试模型路径
model_path = '/home/ihavc/PycharmProjects/SCRN-RFB-ASPP/models/DUTS-TR_w.pth.49'
# 存储结果路径
result_file_name = 'evaluate_test_results'
# 日志路径
log_file_path= "./log/"
# 日志名称
log_file_name = model_name+"_evaluate"

def test_evaluate_sort(model_name, method_name, test_datasets, dataset_path, model_path, result_file_name, log_file_name, testsize, log_path="./log", is_need_process_image=True):

        model = SCRN()
        model.cuda()
        # model_path = './models/DUTS-TR_w.pth'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logger = log.setup_logger(log_file_name, log_path, 0)
        # logger.info("parameters is ", opt)
        logger.info("Test datasets {} begin!".format(test_datasets))
        model.load_state_dict(torch.load(model_path))

        logger.info("====【 model path 】====load model path is {}".format(model_path))

        for i,dataset in enumerate(test_datasets):
            startTime = time.time()
            logger.info("Model {} test NO:{} datasets {} begin test!  The main process is using the model to recoginze the test database's photo".format(model_name,i,dataset))


            path_prefix = './'+result_file_name+'/' + model_name + '/'
            #识别结果存放路径 （仅仅存放final）
            pre_image_merge_save_path = path_prefix + dataset + '/'+"recognize"+ '/'
            # 比较文件夹路径（将img gt edge final 等图片放在一起便于比较)
            compare_image_save_path = path_prefix + dataset + '/'+ 'compare' + '/'

            if not os.path.exists(pre_image_merge_save_path):
                os.makedirs(pre_image_merge_save_path)

            if not os.path.exists(compare_image_save_path):
                os.makedirs(compare_image_save_path)

            image_root = os.path.join(dataset_path, dataset)
            # 根据不同模型获取具体的测试集路径
            if dataset == "ECSSD":
                image_path = image_root + "/Imgs/"
                gt_path = image_path
            elif dataset == "ECSSD2":
                image_path = image_root + "/Imgs1/"
                gt_path = image_path
            elif dataset == "DUTS-TE":
                image_path = image_root + "/DUTS-TE-Image/"
                gt_path = image_root + "/DUTS-TE-Mask/"
            else:
                image_path = image_root + "/Img/"
                gt_path = image_root + "/GT/"

            test_loader = test_dataset(image_path, gt_path, testsize)
            total_test_time =0
            total_time =0

            model.eval()
            for i in range(test_loader.size):
                # if i==5:
                #     break
                image_tran, gt, image, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = Variable(image_tran).cuda()
                begin_time = time.time()
                res, edge = model(image)
                test_end_time = time.time()
                total_test_time +=(test_end_time-begin_time)
                # 加入验证集  根据gt对图片进行resize
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                det_edge_temp = F.interpolate(edge, size=gt.shape, mode='bilinear', align_corners=False)
                # um_merge_temp = F.interpolate(um_merge, size=gt.shape, mode='bilinear', align_corners=False)
                temp_name = name.split('.png')[0]
                transform = transforms.Compose([
                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                ]
                )

                # 直接保存生成结果的图像到一个文件夹中 这个不要注释，主要用于下面计算出得分的用途
                vutils.save_image(res, pre_image_merge_save_path + temp_name + ".png")

                # 便于肉眼直观比较的文件夹  有原图 gt 生成图像
                if is_need_process_image:
                    vutils.save_image(image, compare_image_save_path + temp_name+ "_0_image.jpg")
                    vutils.save_image(transform(gt), compare_image_save_path + temp_name+ "_1_gt.png")
                    vutils.save_image(det_edge_temp,compare_image_save_path + temp_name+ "_2_det_edge.png")
                    vutils.save_image(res, compare_image_save_path+temp_name+"_3_final.png")

                save_end_time = time.time()
                total_time += (save_end_time-begin_time)

            logger.info(" ave_test_time is {:.4f} s  total_test_time is {:.4f} s".format(total_test_time/test_loader.size,total_test_time))
            logger.info("total_time is {} s ".format(total_time))
            logger.info("Test Model {} test dataset {} [Cost:{:.4f}s] finished! ".format(model_name,dataset,(time.time()-startTime)))

        # 下面的代码用于评估模型生成图片的得分
        logger.info("begin merge  evaluate.....")

        # 图片按照得分排序的存储的文件路径
        score_image_save_path = path_prefix + dataset + '/'+ 'score_sort' + '/'
        if not os.path.exists(score_image_save_path):
            os.makedirs(score_image_save_path)


        for dataset in test_datasets:
            fm_score_compare_result_path = score_image_save_path+'fm/'
            sm_score_compare_result = score_image_save_path+'sm/'
            loader = EvalDataset(pre_image_merge_save_path,gt_path)
            thread = Eval_thread(loader, model_name, method_name, dataset, image_path, gt_path,
                                 pre_image_merge_save_path, score_image_save_path,
                                 True, logger, True)
            thread.run()


if __name__ == '__main__':
    test_evaluate_sort(model_name, 'Res_Net', ['DUTS-TE'], dataset_path,
                       model_path, result_file_name,
                       model_name + "_evaluate",352 , log_path=log_file_path, is_need_process_image=True)