import os

class file_logger_centralized():
    def __init__(self, path, time_num, item_list):
        head = ['time_'+str(i) for i in range(time_num)]
        head_str = ','.join(head)+','+','.join(item_list)
        self.path = path
        self.time_num = time_num
        self.item_length = len(item_list)
        with open(self.path,'a') as fp:
            print(head_str, file=fp)
    
    def update(self, time_list, item_list):
        if len(time_list)!=self.time_num or len(item_list)!=self.item_length:
            raise RuntimeError('incorrect log information')
        log_info = ','.join(map(str,time_list))+','+','.join(map(str,item_list))
        with open(self.path,'a') as fp:
            print(log_info, file=fp)