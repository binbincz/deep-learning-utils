
import os
import time
from evaluate.data import validate_dataset
import  traceback
import  numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
from evaluate.evaluator_sort import Eval_thread
from evaluate.dataloader import EvalDataset
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/06/04
@Version  ：0.1
@Desc   ：用于训练过程中的验证
=================================================='''
def before_validate(validate_save_path):
    if not os.path.exists(validate_save_path):
        os.makedirs(validate_save_path)

def validate(model,model_name, method_name, vali_datasets_Name, validate_image_path, validate_gt_path, validate_save_path, validate_size, logger,score_image_save_path="",useCuda=True,isNeedSort=False):
    before_validate(validate_save_path)
    logger.info("begin the validate task.................................")
    # 先生成图片，然后再进行跑分 本来想将方法放到线程中，避免影响训练速度，但是使用threading 好像没有效果
    model.eval()
    vali_begin_time = time.time()
    validate_loader = validate_dataset(validate_image_path, validate_gt_path, validate_size)
    for i in range(validate_loader.size):
        try:
            image, gt, name = validate_loader.load_data()
        except Exception as e:
            exstr = traceback.format_exc()
            print(exstr)
        gt_tmp = np.asarray(gt, np.float32)
        gt_tmp /= (gt_tmp.max() + 1e-8)
        image = image.cuda()
        res, _ = model(image)
        res = F.interpolate(res, size=gt_tmp.shape, mode='bilinear', align_corners=False)
        vutils.save_image(res[0], validate_save_path + name)

    logger.info("get score is begining!............................")
    loader = EvalDataset(validate_save_path,validate_gt_path)

    thread = Eval_thread(loader, model_name, method_name, vali_datasets_Name, validate_image_path, validate_gt_path, validate_save_path, score_image_save_path, useCuda, logger, isNeedSort)
    thread.run()
    logger.info("get score finished!..........................................")

    vali_end_time = time.time()
    logger.info('validate finish  last time ............{:.4f} s'.format((vali_end_time - vali_begin_time)))
