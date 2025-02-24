import tushare as ts
import numpy as np
import pickle
from sklearn import preprocessing
import os
import pandas as pd
from tensorflow.python.keras.utils import np_utils
import random
from tqdm import tqdm

f = open('tstoken.txt')
tstoken = f.readline()
tstoken = tstoken.strip()
print(tstoken)
ts.set_token(tstoken)

class data_processing:
    def __init__(self, code_list, date_start, date_end, codefile=None):
        self.code_list = code_list
        if codefile is not None:
            self.codelist_from_txtfile(codefile)
        self.date_start = date_start
        self.date_end = date_end
        self.dataX = []
        self.dataY = []

        self.datalist = []


    def codelist_from_txtfile(self, txtfile):
        codelist = []
        f = open(txtfile)
        for line in f.readlines():
            line = line.strip()
            code = line #+ ".SH"

            codelist.append(code)
        
        self.code_list = codelist



    def data_download(self, clean_tmp=False, file_type='csv'):
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

        for item in tqdm(self.code_list):
            if file_type == 'csv':
                savepath = "tmp/" + item + "_" + self.date_start + "_" + self.date_end +".csv"
            else:
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
            if file_type == 'csv':
                data.to_csv(savepath)
            else:
                data.to_excel(savepath)

    def mean_norm(self, df_input):
        # df_input['open'] = (df_input['open']-df_input['open'].mean())/ df_input['open'].std()
        # df_input['high'] = (df_input['high']-df_input['high'].mean())/ df_input['high'].std()
        # df_input['low'] = (df_input['low']-df_input['low'].mean())/ df_input['low'].std()
        # df_input['close'] = (df_input['close']-df_input['close'].mean())/ df_input['close'].std()
        # df_input['pre_close'] = (df_input['pre_close']-df_input['pre_close'].mean())/ df_input['pre_close'].std()
        # df_input['ma5'] = (df_input['ma5']-df_input['ma5'].mean())/ df_input['ma5'].std()
        # df_input['ma20'] = (df_input['ma20']-df_input['ma20'].mean())/ df_input['ma20'].std()
        # df_input['ma_v_5'] = (df_input['ma_v_5']-df_input['ma_v_5'].mean())/ df_input['ma_v_5'].std()
        # df_input['ma_v_20'] = (df_input['ma_v_20']-df_input['ma_v_20'].mean())/ df_input['ma_v_20'].std()
        df_input['vol'] = (df_input['vol']-df_input['vol'].mean())/ df_input['vol'].std()
        df_input['amount'] = (df_input['amount']-df_input['amount'].mean())/ df_input['amount'].std()


        return df_input

    def inputdata_clean(self, dataset_x):
        # print(dataset_x[['open', 'close' , 'high', 'low',  'pre_close', 'ma5', 'ma20']])
        dataset_x['open'] = (dataset_x['open']-dataset_x['pre_close'])/dataset_x['pre_close'] * 100
        dataset_x['close'] = (dataset_x['close']-dataset_x['pre_close'])/dataset_x['pre_close'] * 100
        dataset_x['high'] = (dataset_x['high']-dataset_x['pre_close'])/dataset_x['pre_close'] * 100
        dataset_x['low'] = (dataset_x['low']-dataset_x['pre_close'])/dataset_x['pre_close'] * 100
        dataset_x['ma5'] = (dataset_x['ma5']-dataset_x['pre_close'])/dataset_x['pre_close'] * 100
        dataset_x['ma20'] = (dataset_x['ma20']-dataset_x['pre_close'])/dataset_x['pre_close'] * 100
        # print(dataset_x[['open', 'close' , 'high', 'low',  'pre_close', 'ma5', 'ma20']])

        dataset_x = self.mean_norm(dataset_x)


        return dataset_x

    def split_dataclass(self):

        dataset0 = []
        dataset1 = []
        dataset2 = []
        dataset3 = []
        for data in self.datalist:
            classlabel = np.argmax(data[1])
            if classlabel == 0:
                dataset0.append(data)
            elif classlabel == 1:
                dataset1.append(data)
            elif classlabel == 2:
                dataset2.append(data)
            elif classlabel == 3:
                dataset3.append(data)

        random.shuffle(dataset0)
        random.shuffle(dataset1)
        random.shuffle(dataset2)
        random.shuffle(dataset3)

        len0 = len(dataset0)
        len1 = len(dataset1)
        len2 = len(dataset2)
        len3 = len(dataset3)
        

        len_min = min([len0, len1, len2, len3])

        data_extend = dataset0[0:len_min] + dataset1[0:len_min] + dataset2[0:len_min] + dataset3[0:len_min]
        random.shuffle(data_extend)
        # print(len0 ,len1, len2, len3, len(data_extend))

        return data_extend



    def data_prepare(self, time_step, time_step_add, data_col=None, clean_tmp=False):
        dataset_x = []
        dataset_y = []

        for item in self.code_list:
            srcdata = "tmp/" + item + "_" + self.date_start + "_" + self.date_end +".xls"
            if os.path.isfile(srcdata)==False or clean_tmp:
                self.data_download(clean_tmp)
            
            data = pd.read_excel(srcdata, usecols = [1,2,3,4,5,6,7,8,9,10,11,
                                                     12,13,14,15])
            data = data.dropna(axis = 0, subset = ['ma20'])

            datay = data['pct_chg'].values.tolist()
            
            for i,v in enumerate(datay):

                if v <= -5:
                    datay[i] = 0
                elif v>-5 and v<=0:
                    datay[i] = 1
                elif v>0 and v<=5:
                    datay[i] = 2
                elif v>5:
                    datay[i] = 3

            datay = np_utils.to_categorical(datay,num_classes=4)
            

            data = self.inputdata_clean(data)
            if data_col is not None:
                data = data[data_col]
            datax = data.values.tolist()
            

            datalength = len(datax)
            idx_start = 0
            while(True):
                if idx_start+time_step >= datalength-1:
                    if datalength-time_step-1 < 0: break
                    dataset_x.append(datax[datalength-time_step-1:datalength-1])
                    dataset_y.append(datay[datalength-1])
                    break
                
                dataset_x.append(datax[idx_start:idx_start+time_step])
                dataset_y.append(datay[idx_start+time_step])

                idx_start += time_step_add

            

        # ss_x = preprocessing.StandardScaler()
        # data_fit_x = ss_x.fit_transform(np.array(dataset_x))
        # ss_y = preprocessing.StandardScaler()
        # data_fit_y = ss_y.fit_transform(np.array(dataset_y).reshape(-1, 1))

        self.datalist = list(zip(dataset_x,dataset_y))


        return self.datalist


    def data_load(self, ):
        pass



def kmean_analysis(test_data):
    random.shuffle(test_data)
    test_data = test_data[:1000]

    testx, testy = list(zip(*test_data))
    argmax_list = [0,0,0,0]
    for i in testy:
        temp = np.argmax(i)
        argmax_list[temp] += 1
    for i in argmax_list:
        print( i/len(testy) )

    testy = [np.argmax(i) for i in testy]
    print(testy)
    

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA, FactorAnalysis
    import matplotlib.pyplot as plt
    x = np.array(testx)
    x = x.reshape(x.shape[0], -1)
    print(x.shape)

    pca = PCA(n_components=2)
    pca.fit(x)
    x_reduced = pca.transform(x)

    fa_transformer = FactorAnalysis(n_components=2, random_state=0)
    fa_reduced = fa_transformer.fit_transform(x)

    print(x.shape)
    print(x_reduced.shape)

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(fa_reduced)
    y_pred = kmeans.predict(fa_reduced)
    print(y_pred)
    # plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=testy,
    #        cmap='RdYlBu')
    
    plt.scatter(fa_reduced[:, 0], fa_reduced[:, 1],  c=kmeans.labels_) # 按照label画图
    plt.show()



if __name__ == "__main__":
    dp = data_processing(['600219.SH','600170.SH','600219.SH', '600369.SH', '600372.SH'],
                         "2019-01-01", 
                         "2021-12-31", codefile="SZ50.txt")

    data_col = ['ts_code','trade_date','open','close','high','low','pct_chg','ma5','ma20',]
    test_data = dp.data_prepare(30, 1, data_col=data_col, clean_tmp=False)
    # test_data = dp.split_dataclass()
    testx, testy = list(zip(*test_data))
    for i,v in enumerate(testx):
        for item in v:
            print(item)
        print(testy[i])
        ttt = input()
    argmax_list = [0,0,0,0]
    for i in testy:
        temp = np.argmax(i)
        argmax_list[temp] += 1
    for i in argmax_list:
        print( i/len(testy) )