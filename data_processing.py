import tushare as ts
import numpy as np
import pickle
from sklearn import preprocessing
import os
import pandas as pd

f = open('tstoken.txt')
tstoken = f.readline()
tstoken = tstoken.strip()
print(tstoken)
ts.set_token(tstoken)

class data_processing:
    def __init__(self, name_list, date_start, date_end):
        self.name_list = name_list
        self.date_start = date_start
        self.date_end = date_end
        self.dataX = []
        self.dataY = []


    def data_download(self, clean_tmp=False):
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

        for item in self.name_list:
            savepath = "tmp/" + item + "_" + self.date_start + "_" + self.date_end +".xls"
            if os.path.isfile(savepath):
                if clean_tmp:
                    os.remove(savepath)
                else:
                    continue

            # data = ts.get_hist_data(item, start=self.date_start, end=self.date_end)
            data = ts.pro_bar(ts_code=item,  start_date=self.date_start, end_date=self.date_end, 
                 ma=[5, 20, 50] )
            data = data.reindex(index=data.index[::-1])
            data.to_excel(savepath)
    

    def data_prepare(self, time_step, time_step_add, clean_tmp=False):
        dataset_x = []
        dataset_y = []

        for item in self.name_list:
            srcdata = "tmp/" + item + "_" + self.date_start + "_" + self.date_end +".xls"
            if os.path.isfile(srcdata)==False or clean_tmp:
                self.data_download(clean_tmp)
            
            data = pd.read_excel(srcdata, usecols = [3,4,5,6,7,8,9,10,11,
                                                     12,13,14,15])
            data = data.dropna(axis = 0, subset = ['ma20'])

            datax = data.values.tolist()
            datay = data['close'].values.tolist()
            datay = [ [i] for i in datay]
            datalength = len(datax)

            idx_start = 0
            while(True):
                if idx_start+time_step >= datalength-1:
                    if datalength-time_step-1 < 0: break
                    dataset_x.append(datax[datalength-time_step-1:datalength-1])
                    dataset_y.append([datay[datalength-1]])
                    break
                
                dataset_x.append(datax[idx_start:idx_start+time_step])
                dataset_y.append([datay[idx_start+time_step]])

                idx_start += time_step_add

            

        # ss_x = preprocessing.StandardScaler()
        # data_fit_x = ss_x.fit_transform(np.array(dataset_x))
        # ss_y = preprocessing.StandardScaler()
        # data_fit_y = ss_y.fit_transform(np.array(dataset_y).reshape(-1, 1))

        return dataset_x,dataset_y


    def data_load(self, ):
        pass


if __name__ == "__main__":
    dp = data_processing(['600219.SH','600170.SH','600219.SH', '600369.SH', '600372.SH'],
                         "2019-01-01", 
                         "2019-12-31")

    datax,datay = dp.data_prepare(10, 2, clean_tmp=False)
    
    print(datax[0][0])
