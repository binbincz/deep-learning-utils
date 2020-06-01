import os
from shutil import copyfile
import traceback
import logger
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



if __name__ == '__main__':
    logger = logger.setup_logger("compare_error", "log/", 0)
    image_max_score = 0.5
    is_only_origin = True
    compare_result_path_prex =  r"/home/ihavc01/sod/compare_scrn_origin_assp1/fm/"


    # 比较 SRCN  与 lzy-srcn
    model_1_result_path = r"/home/ihavc01/sod/SCRN-ASSP/result_new/scrn_res/DUTS-TR/DUTS-TE__score_sort/fm/"
    model_2_result_path = r"/home/ihavc01/LZY/EGNet-master/EGNet-master/EGNet-master/result_new/EGNet/DUTS-TE__score_sort/fm/"
    model_3_result_path = r"/home/ihavc01/sod/SCRN-master/result_new/scrn_res/DUTS-TR/DUTS-TE__score_sort/fm/"
    mode1_abbreviated = "SCRN-ASPP"
    mode2_abbreviated = "EGNet"
    mode3_abbreviated = "SCRN"

    compare_result_path_best = compare_result_path_prex + mode2_abbreviated + "-best-" + mode1_abbreviated + "-"+mode3_abbreviated+"/"


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

    model_1_map = get_image_info_map(m1_result_file_path_list, model_1_result_path)
    model_2_map = get_image_info_map(m2_result_file_path_list, model_2_result_path)
    model_3_map = get_image_info_map(m3_result_file_path_list, model_3_result_path)

    image_map = get_image_info_map(m1_image_file_path_list, model_1_result_path)
    gt_map = get_image_info_map(m1_gt_file_path_list, model_1_result_path)

    i = 0
    key_list = list(model_1_map.keys())
    length = len(key_list)
    compare_result_path = compare_result_path_best
    num = 0
    print("模型比较开始！")
    for i in range(0, length):
        if i > 10:
            print("运行限制，提前结束")
            break;
        key = key_list[i]
        image1 = model_1_map[key]
        try:
            image2 = model_2_map[key]
            image3 = model_3_map[key]
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


        copyfile(image1.file_path, new_file_1_path)
        copyfile(image2.file_path, new_file_2_path)
        copyfile(image3.file_path, new_file_3_path)
        copyfile(img.file_path, new_img_file_path)
        copyfile(gt.file_path, new_gt_file_path)

    print("寻找3个模型 ",mode1_abbreviated,mode2_abbreviated,mode3_abbreviated," 识别都失败,分数低于或等于 ",image_max_score," 的图片数量有 ",num)
    print("模型1是 ",model_1_result_path," 模型2是 ",model_2_result_path," 模型3是 ",model_3_result_path)
    print("比较结果存放路径是 " + compare_result_path_best)
    print("Congratualtion!! Compare finished!!!")

