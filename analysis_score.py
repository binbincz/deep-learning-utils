#encoding=utf-8
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/03/06
@Version  ：0.1
@Desc   ：主要用于找出A模型里面识别失败的内容，是否在B模型中识别成功。
=================================================='''

# 分析文件名中负号开头的文件数量

import  os
import  logger as log_set

logger = log_set.setup_logger("compare_SCRN-ASPP_CDP_EGNet_SCRN","log/",0)
# 分析得分情况，分析大于等于小于的数量
def anlaysis_score_condition(compare_result_file_path):

    # compare_result_file = open(compare_result_file_path,"r")
    bigger_zero = 0
    equal_zero = 0
    smaller_zero = 0
    file_count = len(os.listdir(compare_result_file_path))
    for file_name in sorted(os.listdir(compare_result_file_path)):
      if file_name.find("_a_"):
         if file_name.startswith("-"):
             smaller_zero+=1
         elif file_name.startswith("0.00"):
              equal_zero+=1
         else:
             bigger_zero+=1

    logger.info("比较路径是 "+compare_result_file_path)

    logger.info("比较图片数量 %s 张"%file_count)
    logger.info("比较结果是 小于%s 张"%smaller_zero)
    logger.info("比较结果是 等于%s 张"%equal_zero)
    logger.info("比较结果是 大于%s 张"%smaller_zero)

# 分析得分情况，分析大于等于小于的数量
def anlaysis_score_distribute_condition(compare_result_file_path):
    # compare_result_file_path =r"C:\Users\yuan\Desktop\compare1\各个模型优于scrn\fm cdp 优于 scrn 11\best"
    # compare_result_file = open(compare_result_file_path,"r")
    bigger = 0
    smaller =0
    equal = 0
    bigger_01 = 0
    bigger_12 = 0
    bigger_23 = 0
    bigger_34 = 0
    bigger_45 = 0
    bigger_56 = 0
    bigger_67 = 0
    bigger_78 = 0
    bigger_89 = 0
    bigger_91 = 0
    bigger_100 = 0

    smaller_01 = 0
    smaller_12 = 0
    smaller_23 = 0
    smaller_34 = 0
    smaller_45 = 0
    smaller_56 = 0
    smaller_67 = 0
    smaller_78 = 0
    smaller_89 = 0
    smaller_91 = 0
    smaller_100 = 0

    file_count = len(os.listdir(compare_result_file_path))
    for file_name in sorted(os.listdir(compare_result_file_path)):
      if file_name.find("_a_"):
         if file_name.startswith("-"):
            smaller += 1
            score = float(file_name.split("_")[0])
            if 0 > score >= -0.1:
               smaller_01+=1
            elif -0.1 > score >= -0.2:
               smaller_12+=1
            elif -0.2 > score >= -0.3:
               smaller_23 += 1
            elif  -0.3 > score >= -0.4:
               smaller_34 += 1
            elif  -0.4 > score >= -0.5:
               smaller_45 += 1
            elif  -0.5 > score >= -0.6:
               smaller_56 += 1
            elif -0.6 > score >= -0.7:
               smaller_67 += 1
            elif  -0.7 > score >= -0.8:
               smaller_78 += 1
            elif  -0.8 > score>= -0.9:
               smaller_89 += 1
            elif  -0.9 > score> -1:
               smaller_91 += 1
            elif score == -1:
                smaller_100 += 1
            else:
                print("小于的分数 ",score)
         elif file_name.startswith("0.00"):
              equal+=1
         else:
             bigger+=1
             score = float(file_name.split("_")[0])

             if 0 < score < 0.1:
                 bigger_01 += 1
             elif 0.1 <= score < 0.2:
                 bigger_12 += 1
             elif 0.2 <= score < 0.3:
                 bigger_23 += 1
             elif 0.3 <= score < 0.4:
                 bigger_34 += 1
             elif 0.4 <= score < 0.5:
                 bigger_45 += 1
             elif 0.5 <= score < 0.6:
                 bigger_56 += 1
             elif 0.6 <= score < 0.7:
                 bigger_67 += 1
             elif 0.7 <= score < 0.8:
                 bigger_78 += 1
             elif 0.8 <= score < 0.9:
                 bigger_89 += 1
             elif 0.9 <= score < 1:
                 bigger_91 += 1
             elif score == 1:
                 bigger_100 += 1
             else:
                print("大于的分数 ",score)
    logger.info("比较路径是 "+compare_result_file_path)

    logger.info("比较图片数量 %s 张"%file_count)
    logger.info("比较结果是 小于%s 张"%smaller)
    logger.info("比较结果是 等于%s 张"%equal)
    logger.info("比较结果是 大于%s 张"%bigger)

    logger.info("比较结果是 小于%s 张 "% smaller+"分布情况如下")
    logger.info(" 0 > score > -0.1  %s"%smaller_01)
    logger.info(" -0.1 >= score > -0.2  %s"%smaller_12)
    logger.info(" -0.2 >= score > -0.3  %s"%smaller_23)
    logger.info(" -0.3 >= score > -0.4  %s"%smaller_34)
    logger.info(" -0.4 >= score > -0.5  %s"%smaller_45)
    logger.info(" -0.5 >= score > -0.6  %s"%smaller_56)
    logger.info(" -0.6 >= score > -0.7  %s"%smaller_67)
    logger.info(" -0.7 >= score > -0.8  %s"%smaller_78)
    logger.info(" -0.8 >= score > -0.9  %s"%smaller_89)
    logger.info(" -0.9 >= score > -1  %s"%smaller_91)
    logger.info(" score =- 1  %s"  %smaller_100)

    logger.info("比较结果是 大于%s 张 " % bigger + "分布情况如下")
    logger.info(" 0 < score < -0.1  %s" % bigger_01)
    logger.info(" 0.1 <= score < 0.2  %s" % bigger_12)
    logger.info(" 0.2 <= score < 0.3  %s" % bigger_23)
    logger.info(" 0.3 <= score < 0.4  %s" % bigger_34)
    logger.info(" 0.4 <= score < 0.5  %s" % bigger_45)
    logger.info(" 0.5 <= score < 0.6  %s" % bigger_56)
    logger.info(" 0.6 <= score < 0.7  %s" % bigger_67)
    logger.info(" 0.7 <= score < 0.8  %s" % bigger_78)
    logger.info(" 0.8 <= score < 0.9  %s" % bigger_89)
    logger.info(" 0.9 <= score < 1  %s" % bigger_91)
    logger.info(" score = 1  %s" % bigger_100)
    # logger.info("<=0.1 %s" % bigger_10)
    # logger.info("<=0.2 %s" % bigger_20)
    # logger.info("<=0.3 %s" % bigger_30)
    # logger.info("<=0.4 %s" % bigger_40)
    # logger.info("<=0.5 %s" % bigger_50)
    # logger.info("<=0.6 %s" % bigger_60)
    # logger.info("<=0.7 %s" % bigger_70)
    # logger.info("<=0.8 %s" % bigger_80)
    # logger.info("<=0.9 %s" % bigger_90)
    # logger.info("<=1 %s"   % bigger_100)


if __name__ == '__main__':
    compare_result_file_path = r"/home/ihavc01/sod/compare_SCRN-ASPP_CDP_EGNet_SCRN/fm/CPD-best-SCRN-ASPP-EGNet"
    anlaysis_score_distribute_condition(compare_result_file_path)
