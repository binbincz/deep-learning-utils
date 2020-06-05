import  os
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/06/03
@Version  ：0.1
@Desc   ：主要用于有关模型读取方法
=================================================='''
# 不用指定关键词 直接逆排序即可
# 会有bug
def  get_last_model(model_path,reverse=True):
    if os.path.exists(model_path):
        file_names = os.listdir(model_path)
        file_names.sort(reverse=True)
        if file_names.__len__() > 0:
            last_model_path = os.path.join(model_path, file_names[0])
            print("获取的最新模型路径是",last_model_path)
            return last_model_path
        else:
            print("指定路径", model_path, "查找不到模型.......")
            return ""
    else:
        print("指定路径", model_path, "查找不到模型.......")
        return ""
if __name__ == '__main__':
    print(get_last_model(".\model1"))
