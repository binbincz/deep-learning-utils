# encoding: utf-8
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/03/06
@Version  ：0.1
@Desc   ：用于计算出训练过程中，剩余训练时间的工具
=================================================='''

class TimeUtil:
    def __init__(self,logger=None):
        # 记录最近十次的耗时
        self.last_10_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.total_time = 0
        self.logger = logger

    def my_print(self, str):
        if self.logger == None:
            print(str)
        else:
            self.logger.info(str)

    # 要多观察，特别是第一轮迭代，看是否抛异常
    def displayTime(self,epoch,total_epoch,last_time):

        index = epoch % 10
        # 更新近十次耗时
        self.last_10_times[index] = last_time
        # TODO:统计总共耗时,这个日期有可能时间长的话会出现溢出问题!!!!!  如果超过一天，那么可以将其转成天，然后剩余依旧是淼
        self.total_time += last_time
        # 统计近十次总耗时
        last_10_total_time = sum(st for st in self.last_10_times)
        # 统计近十次平均耗时，以此来计算剩余完成时间，更加准确
        is_first = False
        zero_times = 0
        for i in self.last_10_times:
            if i==0 :
                is_first = True
                zero_times +=1
        # 解决模型再次加载后，时间估算异常
        if  is_first:
            ave_10_time = last_10_total_time / (10-zero_times)
        elif epoch < 10:
            ave_10_time = last_10_total_time / (epoch + 1)
        else:
            ave_10_time = last_10_total_time / 10

        # 计算剩余时间 减去1的原因是 到此处已经跑了一轮，所以剩余迭代次数要减去1
        remain_times = total_epoch - epoch - 1
        if remain_times == 0:
            self.my_print("恭喜！训练已完成，已经耗费时间 {:.2f} s，过去十次迭代平均耗时{:.2f} s".format(self.total_time,  ave_10_time,))
        else:
            remain_time = ave_10_time * remain_times
            remain_time_hour = remain_time // (60 * 60)
            remain_time_min = remain_time % 3600 // 60
            remain_time_sec = remain_time % 3600 % 60
            self.my_print("本次训练，已经耗费时间 {:.2f} s，过去十次迭代平均耗时{:.2f} s，剩余完成时间【{:.0f}H {:.0f}M {:.2f}S】".format(self.total_time,
                                                                                                       ave_10_time,
                                                                                                       remain_time_hour,
                                                                                                       remain_time_min,
                                                                                                      remain_time_sec))

