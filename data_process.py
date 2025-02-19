import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import preprocessing
import os
import pandas as pd
import numpy as np

import tushare as ts




class data_process():
    def __init__(self, code_list, date_start, date_end, codefile=None, data_savedir="tmp/"):
        self.code_list = code_list
        self.data_savedir = data_savedir
        if codefile is not None:
            self.codelist_from_txtfile(codefile)
        self.date_start = date_start
        self.date_end = date_end
        self.datalist = []

    def codelist_from_txtfile(self, txtfile):
        codelist = []
        f = open(txtfile)
        for line in f.readlines():
            line = line.strip()
            if(line[0]=='6'):
                code = line + ".SH"
            else:
                code = line + ".SZ"
            codelist.append(code)
        self.code_list = codelist


    def data_download(self, clean_tmp=False):
        f = open('tstoken.txt')
        tstoken = f.readline()
        tstoken = tstoken.strip()
        print(tstoken)
        ts.set_token(tstoken)

        if not os.path.exists(self.data_savedir):
            os.makedirs(self.data_savedir)

        for item in self.code_list:
            savepath = self.data_savedir + item + "_" + self.date_start + "_" + self.date_end +".xlsx"
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


    def data_load(self, filetime="2016-01-01_2022-05-31", restore="pre", clean_tmp=False):
        datadict = {}
        for codename in tqdm(self.code_list):
            srcdata = os.path.join(self.data_savedir , codename + "_" + filetime +".xlsx")
            
            if os.path.isfile(srcdata)==False or clean_tmp:
                self.data_download(clean_tmp)
            
            data = pd.read_excel(srcdata, usecols = [1,2,3,4,5,6,7,8,9,10,11,
                                        12,13,14,15,16,17])
            # print(pd.to_datetime(data["trade_date"]))
            # data["trade_date"] = pd.to_datetime(data["trade_date"])
            print("date", data.head(10))
            data['trade_date'] = data['trade_date'].apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8])
            #print(data.head(10))

            if restore=="pre":
                data.sort_index(ascending=False, inplace=True)
            elif restore=="nxt":
                data.sort_index(ascending=True, inplace=True)
            print("pre", data.head(10))
            close_data = data["close"]
            
            pct_chg_data = data["pct_chg"]
            data["close_restore"] = 0
            close_restore_tmp = 0
            close_temp = close_data[0]
            # print(pct_chg_data)
            for index, c in enumerate(close_data):
                if index == len(close_data): break
                if np.isnan(pct_chg_data[index]) : continue
                close_restore_tmp = close_temp / (1+pct_chg_data[index]/100)
                # print(close_temp, close_restore_tmp)
                data.loc[index,'close_restore'] = close_restore_tmp
                # data.loc[index+1,'close'] = close_restore_tmp
                close_temp = close_restore_tmp

            if restore=="pre":
                data.sort_index(ascending=True, inplace=True)
            elif restore=="nxt":
                data.sort_index(ascending=False, inplace=True)
                
            print("return", data.head(10))

            data["ma5_restore"] = data["close_restore"].rolling(5).mean()
            data["ma10_restore"] = data["close_restore"].rolling(10).mean()
            data["ma20_restore"] = data["close_restore"].rolling(20).mean()
            data["ma30_restore"] = data["close_restore"].rolling(30).mean()
            data["ma60_restore"] = data["close_restore"].rolling(60).mean()

            datadict[codename] = data

        return datadict


if __name__ == "__main__":
    data_hd = data_process(["002508.SZ"],
                         "2016-01-01", "2024-12-31", data_savedir="./tmp_test/")
    
    tradedata = data_hd.data_load(filetime="2016-01-01_2024-12-31", restore="pre")