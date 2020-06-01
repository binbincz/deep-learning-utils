import os
from shutil import copyfile
import traceback
import logger as log_set
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/03/06
@Version  ：0.1
@Desc   ：主要用于找出A模型里面识别失败的内容，是否在B模型中识别成功。
=================================================='''

class image_info:
    def __init__(self, file_path, file_name, origin_file_name, score, img_type, file_name_no_score=None):
        self.file_path = file_path
        self.file_name = file_name
        self.origin_file_name = origin_file_name
        # self.file_name_no_score = file_name_no_score
        self.score = score
        # self.key = key_name
        self.type = img_type

    def __str__(self):
        image_map = {}
        image_map['file_path'] = self.file_path
        image_map['file_name'] = self.file_name
        image_map['origin_file_name'] = self.origin_file_name
        image_map['score'] = self.score
        return image_map



def get_image_info_map(model_result_file_list, image_path_prex):
    model_map = {}
    for file_name in model_result_file_list:
        file_name = file_name
        tmp_file_names = file_name.split("_")
        # 去掉分数
        tmp = file_name.replace(tmp_file_names[0] + "_", '')
        # 去掉最后标号
        tmp = tmp.replace("_" + tmp_file_names[-1], '')
        # # 去掉前后两条下划线
        # tmp = tmp[1:-1]
        # 获取图片类型 gt result
        img_type = tmp_file_names[-2]
        # 处理image 没有写上Image
        if img_type != 'gt' and img_type != 'result' and img_type != 'image':
            img_type = 'a'
        else:
            # 有image  需要去gt 或者result 可以兼容后期有image
            tmp = tmp.replace("_" + tmp_file_names[-2], '')
        origin_file_name = tmp
        score = tmp_file_names[0]
        if score == 'nan':
            # print("image no score ",image_path_prex," ",file_name)
            score = 0.00
        model_map[origin_file_name] = image_info(image_path_prex + file_name, file_name, origin_file_name, score,
                                                 img_type)
    return model_map

#分析分数代码
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


    # 比较 SRCN  与 lzy-srcn
    model_1_result_path = r"/home/ihavc01/sod/SCRN-ASSP/result_new/scrn_res/DUTS-TR/DUTS-TE__score_sort/fm/"
    model_2_result_path = r"/home/ihavc01/sod/CDP/results/VGG/fm/fm/"
    model_3_result_path = r"/home/ihavc01/LZY/EGNet-master/EGNet-master/EGNet-master/result_new/EGNet/DUTS-TE__score_sort/fm/"
    model_4_result_path = r"/home/ihavc01/sod/SCRN-master/result_new/scrn_res/DUTS-TR/DUTS-TE__score_sort/fm/"
    mode1_abbreviated = "SCRN-ASPP"
    mode2_abbreviated = "CPD"
    mode3_abbreviated = "EGNet"
    mode4_abbreviated = "SCRN"

    compare_file_name = "compare_"+mode1_abbreviated+"_"+mode2_abbreviated+"_"+mode3_abbreviated+"_"+mode4_abbreviated
    logger = log_set.setup_logger(compare_file_name, "log/", 0)
    image_max_score = 0.5
    is_only_origin = True
    compare_result_path_prex = r"/home/ihavc01/sod/" + compare_file_name + "/fm/"

    compare_result_path_best = compare_result_path_prex


    if not os.path.exists(compare_result_path_best):
        os.makedirs(compare_result_path_best)


    m1_image_file_path_list = [x for x in os.listdir(model_1_result_path) if x.find("_0.") != -1]

    m1_gt_file_path_list = [x for x in os.listdir(model_1_result_path) if x.find("gt") != -1 and x.find("_1") != -1]

    m1_result_file_path_list = [x for x in os.listdir(model_1_result_path) if
                                x.find("result") != -1 and x.find("_2") != -1]
    m2_result_file_path_list = [x for x in os.listdir(model_2_result_path) if
                                x.find("result") != -1 and x.find("_2") != -1]
    m3_result_file_path_list = [x for x in os.listdir(model_3_result_path) if
                                x.find("result") != -1 and x.find("_2") != -1]
    m4_result_file_path_list = [x for x in os.listdir(model_4_result_path) if
                                x.find("result") != -1 and x.find("_2") != -1]

    model_1_map = get_image_info_map(m1_result_file_path_list, model_1_result_path)
    model_2_map = get_image_info_map(m2_result_file_path_list, model_2_result_path)
    model_3_map = get_image_info_map(m3_result_file_path_list, model_3_result_path)
    model_4_map = get_image_info_map(m4_result_file_path_list, model_4_result_path)

    image_map = get_image_info_map(m1_image_file_path_list, model_1_result_path)
    gt_map = get_image_info_map(m1_gt_file_path_list, model_1_result_path)

    i = 0
    key_list = list(model_1_map.keys())
    length = len(key_list)
    compare_result_path = compare_result_path_best
    num = 0
    print("模型比较开始！")
    for i in range(0, length):
        # if i > 10:
        #     print("运行限制，提前结束")
        #     break;
        key = key_list[i]
        image1 = model_1_map[key]
        try:
            image2 = model_2_map[key]
            image3 = model_3_map[key]
            image4 = model_4_map[key]
            img = image_map[key]
            gt = gt_map[key]
        except KeyError as e:
            logger.error("KeyError ", e)
            exstr = traceback.format_exc()
            logger.error(exstr)
            continue
        image1_score = float(image1.score)
        image2_score = float(image2.score)
        image3_score = float(image3.score)
        image4_score = float(image4.score)

        different_value = image1_score - image2_score

        different_value = '%.2f' % different_value
        new_file_name_prex = str(different_value) + "_"


        # if image1_score<= image_max_score and image2_score<= image_max_score and image3_score<= image_max_score:
        #     num+=1
        # 由于图片得分不相等，所以分数只能放在后面，不然会导致相同图片无法相邻  window系统文件名排序是按照从前到后字符ascii值大小来排序
        new_img_file_name = new_file_name_prex + img.origin_file_name + "_" + "a" + "_1_" + str(img.score) + ".jpg"
        new_img_file_path = compare_result_path + new_img_file_name

        new_gt_file_name = new_file_name_prex + gt.origin_file_name + "_" + gt.type + "_2_" + str(gt.score) + ".png"
        new_gt_file_path = compare_result_path + new_gt_file_name
        # 由于图片得分不相等，所以 两个模型的生成结果是可以需要进行区分的 即不用加入后缀 3 4
        new_file_1_name = new_file_name_prex + image1.origin_file_name + "_" + image1.type + "_3_" + str(image1.score) +"_"+ mode1_abbreviated + ".png"
        new_file_1_path = compare_result_path + new_file_1_name

        new_file_2_name = new_file_name_prex + image2.origin_file_name + "_" + image2.type + "_4_" + str(image2.score) +"_"+ mode2_abbreviated+ ".png"
        new_file_2_path = compare_result_path + new_file_2_name

        new_file_3_name = new_file_name_prex + image3.origin_file_name + "_" + image3.type + "_5_" + str(image3.score) +"_"+ mode3_abbreviated + ".png"
        new_file_3_path = compare_result_path + new_file_3_name

        new_file_4_name = new_file_name_prex + image4.origin_file_name + "_" + image4.type + "_6_" + str(image4.score) +"_"+ mode4_abbreviated + ".png"
        new_file_4_path = compare_result_path + new_file_4_name


        copyfile(image1.file_path, new_file_1_path)
        copyfile(image2.file_path, new_file_2_path)
        copyfile(image3.file_path, new_file_3_path)
        copyfile(image4.file_path, new_file_4_path)
        copyfile(img.file_path, new_img_file_path)
        copyfile(gt.file_path, new_gt_file_path)

    logger.info("寻找4个模型 ",mode1_abbreviated,mode2_abbreviated,mode3_abbreviated,mode4_abbreviated," 识别都失败,分数低于或等于 ",image_max_score," 的图片数量有 ",num)
    logger.info("分数排序是 模型1 ",mode1_abbreviated,"与 模型2 ",mode2_abbreviated,"做差  负数表示 模型1差于模型2  正数表示 模型1优于模型2 ")
    logger.info("模型1是 ",model_1_result_path," 模型2是 ",model_2_result_path," 模型3是 ",model_3_result_path)
    logger.info("比较结果存放路径是 " + compare_result_path_best)
    logger.info("Congratualtion!! Compare finished!!!")

    # 分析比分分布情况
    anlaysis_score_distribute_condition(compare_result_path_best)

