##### ライブラリの読み込み #####
from datetime import datetime
import numpy as np
import pandas as pd
import time

def download(a,b,c,d,e,f):
  year = a
  month = b
  day = c
  prec = d
  block = e
  name = f
  print('get_data --->',a,'/', b,'/', c,'/',d,'/',e,'/',f)


   ##### urlを指定#####
  url = 'http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=' + str(prec) +'&block_no=' + str(block) +'&year=' + str(year) + '&month=' + str(month) + '&day=' + str(day) + '&view='

 

   ##### スクレイピング #####
  tables = pd.io.html.read_html(url)
  df = tables[0].iloc[:,1:] # --- 必要な行と列を抽出
  df = df.reset_index(drop = True)

 

   ##### 列名を指定 #####
  df.columns = ['G_HPA','O_HPA','PRC','TEMP','DP','VP_HPA','HM', 'WS_MEAN', 'WD_MEAN','SUN','ALSUN','SN','SNM','WT','CLO','EY']

 

   ##### 欠測値の処理 #####
  df = df.replace('///', '-999.9') # --- '///' を欠測値として処理
  df = df.replace('×', '-999.9') # --- '×' を欠測値として処理
  df = df.replace(r'\s\)', '', regex = True) # --- ')' が含まれる値を正常値として処理
  df = df.replace(r'.*\s\]', '-999.9', regex = True) # --- ']' が含まれる値を欠測値として処理
  df = df.replace('#', '-999.9') # --- '#'が含まれる値を欠測値として処理
  df = df.replace('--', '-999.9') # --- '--'が含まれる値を欠測値として処理
  df = df.fillna(0) # --- NaN を欠測値として処理

 

   ##### 風向を北0°で時計回りの表記に変更 #####
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('北北東', '22.5')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('東北東', '67.5')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('東南東', '112.5')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('南南東', '157.5')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('南南西', '202.5')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('西南西', '247.5')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('西北西', '292.5')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('北北西', '337.5')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('北東', '45.0')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('南東', '135.0')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('南西', '225.0')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('北西', '315.0')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('北', '360.0')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('東', '90.0')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('南', '180.0')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('西', '270.0')
  df.loc[:,['WD_MEAN']] = df.loc[:,['WD_MEAN']].replace('静穏', '-888.8')

 

   ##### 時刻列を追加 #####
  df['DATE'] = pd.date_range(datetime(year, month, day, hour=0), periods = len(df), freq = '60T')

 

   ##### 年/月/日/時/分/秒 の各列を追加 #####
  # df['YEAR'] = df['DATE'].dt.year
  # df['MONTH'] = df['DATE'].dt.month
  # df['DAY'] = df['DATE'].dt.day
  # df['HOUR'] = df['DATE'].dt.hour

 

   ##### 処理の停止時間を指定 #####
  time.sleep(1)

 

   ##### csvファイルとして出力 #####
  if c == 1:
    df.to_csv(
    'amd10_31001_'+str(name)+str(year)+(str(month).zfill(2))+'.csv',
    columns
     = ['DATE','TEMP'],
    header = False, index = False
    )
  else:
    df.to_csv(
    'amd10_31001_'+str(name)+str(year)+(str(month).zfill(2))+'.csv',
    columns = ['DATE','TEMP'],
    header = False, index = False, mode = 'a'
    )

def main():
##### 日付を指定 #####
  ylist=[2000]  # yearの指定
  mlist=[2]  # monthの指定
  prec_no = [
    62,
    14,
    48,
    44,
    91
  ]
  block_no = [
    47772,
    47412,
    47620,
    47662,
    47936

  ]
  place_name=[
    '大阪',
    '札幌',
    '諏訪',
    '東京',
    '那覇'

  ]

  for l in range(0,6):
      for y in range (2000,2021):
        ylist =[y]
        for m in range(1,13):
          mlist =[m]
          pre = prec_no[l]
          bloc = block_no[l]
          name = place_name[l]
          for k in ylist : 
            for j in mlist:
              if j == 1 or j == 3 or j == 5 or j == 7 or j == 8 or j ==10 or j ==12:
                for i in range (1, 32) :  # --- 日ループ
                  download(k,j,i,pre,bloc,name)
              elif j == 4 or j == 6 or j == 9 or j == 11:
                for i in range (1, 31) :  # --- 日ループ
                  download(k,j,i,pre,bloc,name)
              elif j == 2 and k%4 == 0:
                if k%100==0:
                  if k%400 ==0:
                    for i in range (1,30):
                      download(k,j,i,pre,bloc,name)
                  else:
                    for i in range (1,29):
                      download(k,j,i,pre,bloc,name)
                else:
                  for i in range (1,30):
                    download(k,j,i,pre,bloc,name)
              else:
                for i in range (1,29):
                  download(k,j,i,pre,bloc,name)

  exit()


if __name__ == '__main__':
  main()