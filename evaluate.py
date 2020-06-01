# encoding=UTF-8
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/06/01
@Version  ：0.2
@Desc   ：用于对生成的显著性图片进行测评，测评结果按照图片在该指标的得分进行排序。实现方式，直接将得分结果加在生成文件名称前
可以通过Eval_thread初始化方法中的isNeedSort参数控制是否需要输出分数排序的结果
=================================================='''
from evaluate.evaluator_sort import Eval_thread
from evaluate.dataloader import EvalDataset
import os, argparse
import logger as log



parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--is_unmerge', type=bool, default=False, help='True or False')
parser.add_argument('--model_type', type=int, default=1, help='1 or 2 or 3')
parser.add_argument('--is_need_process_image', type=bool, default=True, help='True or False')
parser.add_argument('--test_datasets', type=str, default='ECSSD2', help='test_datasets Name')

opt = parser.parse_args()

# dataset_path = '/home/ihavc01/LZY/SCRN/DUTS/DUTS-TE/'
dataset_path = '/home/ihavc/datasets/DUTS-TE/'

dataset = 'DUTS-TE'
modelName = 'CPD'
methodName ="VGG"

log_file_path = 'log/'
if not os.path.exists(log_file_path):
    os.mkdir(log_file_path)

logger = log.setup_logger("CPD_author_model_evaluate",log_file_path , 0)


# 下面的代码用于评估模型生成图片的得分
logger.info("begin merge  evaluate.....")

# 比较结果存放路径
result_file_name = 'result_new'
path_prefix = './' + result_file_name + '/' + modelName + '/'
score_image_save_path = path_prefix + dataset+ '__score_sort' + '/'


# 需要比较图片路径
pre_image_merge_save_path="/home/ihavc/PycharmProjects/CPD-master/results/VGG16_1/DUTS-TR/DUTS-TE/"
# image 的路径
image_path = dataset_path+"DUTS-TE-Image/"
# gt 的路径
gt_path = dataset_path+"DUTS-TE-Mask/"

if not os.path.exists(score_image_save_path):
    os.makedirs(score_image_save_path)


fm_score_compare_result_path = score_image_save_path+'fm/'
sm_score_compare_result = score_image_save_path+'sm/'
loader = EvalDataset(pre_image_merge_save_path,gt_path)
thread = Eval_thread(loader, modelName,methodName, dataset,image_path, gt_path, pre_image_merge_save_path, fm_score_compare_result_path, sm_score_compare_result, True, logger, False)
thread.run()



