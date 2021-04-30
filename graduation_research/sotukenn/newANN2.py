#ライブラリの読み込み
import os
import keras
import numpy as np
import pandas as pd
import codecs
import csv
from pandas.core.frame import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

#予測結果のグラフ比較出力
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy : ')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.close()

def get_Weather_data(FEATURE_VALUE,start1,end1,place):
  dfWeather_all = pd.read_csv("datasample"+str(place)+".csv", index_col=0, parse_dates=True, skiprows = 0, encoding = 'shift-jis', header=None)

  dfWeather_all.columns =[
      'average_temperature'
  ]
  dfWeather = dfWeather_all.loc[start1:end1, FEATURE_VALUE]
  dfWeather = dfWeather.sort_index()
  dfWeather = dfWeather.dropna() #欠損値を取り除く
  
  return  dfWeather


def generate_data(data, length_per_unit,dimension):
    #DataFrame→array変換
    data_array = data.values
    #時系列データを入れる箱
    sequences = []
    #正解データを入れる箱
    target = []
    #正解データの日付を入れる箱
    target_date =[]
    #グループごとに寺家列データと正解をセットしていく
    for i in range(0, data_array.shape[0] - length_per_unit):
        #print(i)
        sequences.append(data_array[i:i + length_per_unit])
        target.append(data_array[i + length_per_unit])
        target_date.append(data[i + length_per_unit: i + length_per_unit + 1].index.strftime('%Y/%m/%d %H'))
    # 時系列データを成形
    X = np.array(sequences).reshape(len(sequences),length_per_unit)
    # 正解データを成形
    Y = np.array(target).reshape(len(sequences), -1)
    # 正解データの日付データを成形
    Y_date = np.array(target_date).reshape(len(sequences), 1)

    return X, Y, Y_date

def generate_data_test(length_per_unit,testWeather,dimension):
    test_temple =[]
    test_anser =[]
    test_anser_date = []
    test_data = testWeather.values
    for i in range(0, test_data.shape[0] - length_per_unit):
      test_temple.append(test_data[i:i + length_per_unit])
      test_anser.append(test_data[i + length_per_unit])
      test_anser_date.append(testWeather[i + length_per_unit: i + length_per_unit + 1].index.strftime('%Y/%m/%d %H'))

    test_temple = np.array(test_temple)
    test_anser = np.array(test_anser).reshape(len(test_anser),-1)
    test_anser_date = np.array(test_anser_date).reshape(len(test_anser_date),-1)

    return test_temple,test_anser,test_anser_date

#モデル学習関数
def model_study(PLACE,HIDDEN,EPOCH):
  #定数の定義
  FEATURE_VALUE = ['average_temperature']
  DIMENSION = len(FEATURE_VALUE)
  LENGTH_PER_UNIT = 365
  timestep = 365
  in_out_neurons = 1
  #1学習期間
  leran_periodstart ='2016-01-01 01:00:00'
  leran_periodend ='2019-12-31 23:00:00'
  TEST_periodstart =['2004-01-01 01:00:00','2009-01-01 01:00:00','2014-01-01 01:00:00','2019-01-01 01:00:00']
  TEST_periodend =['2005-12-31 23:00:00','2010-12-31 23:00:00','2015-12-31 23:00:00','2020-12-31 23:00:00']
  save_period_start =['2005/01/01 01','2010/01/01 01','2015/01/01 01','2020/01/01 01']
  save_period_end =['2005/12/31 23','2010/12/31 23','2015/12/31 23','2020/12/31 23']
  #隠れ層の数
  n_hidden = HIDDEN
  #学習回数
  n_epoch =EPOCH
  #学習定数
  learn_rate = 0.001
  batch = 100
  #入力の形状
  dfWeather= get_Weather_data(FEATURE_VALUE,leran_periodstart,leran_periodend,PLACE)
  z, y_train ,Y_train_data= generate_data(dfWeather, LENGTH_PER_UNIT,DIMENSION)
  input_shape = np.shape(z)
  ########ここから学習#########
  model = Sequential()
  # 入力層
  model.add(Dense(n_hidden, activation='sigmoid', input_shape=input_shape))
  # 出力層
  model.add(Dense(in_out_neurons, activation='linear'))
  # コンパイル（勾配法：RMSprop、損失関数：mean_squared_error、評価関数：accuracy）
  model.compile(loss='mean_squared_error', optimizer=RMSprop(lr = learn_rate), metrics=['accuracy'])
  model.summary()
  # 構築したモデルで学習
  history = model.fit(z, y_train, batch_size=batch, epochs=n_epoch)
  # モデルの性能評価
  score = model.evaluate(z, y_train, verbose=0)
  # print('Score:', score[0])    # 損失値
  # print('Accuracy:', score[1]) # 精度
  # 学習履歴をプロット
  #plot_history(history)
  new_dir_path = '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ANN/predictANN'+str(n_epoch)+str(n_hidden)+PLACE
  os.mkdir(new_dir_path)
  for l in range(0,4):
    Test_Data = get_Weather_data(FEATURE_VALUE,TEST_periodstart[l],TEST_periodend[l],PLACE)
    dfWeatherTest = Test_Data['average_temperature']
    TESTTEMPLE,y_test,Y_test_date = generate_data_test( LENGTH_PER_UNIT,dfWeatherTest, DIMENSION)
    # モデルを使った予測
    y_ = model.predict(TESTTEMPLE)
    y_test = y_test.tolist()
    predicted_data =[]
    Y =[]
    for i in range(0,len(y_test)):
      koko = []
      koko.append(y_test[i][0])
      koko.append(y_[i][0])
      koko.append(y_test[i][0]-y_[i][0])
      koko.append(np.abs(y_[i][0]-y_test[i][0]))
      koko.append((y_[i][0]-y_test[i][0])/y_test[i][0])
      koko.append(np.abs((y_[i][0]-y_test[i][0])/y_test[i][0]))
      Y.append(Y_test_date[i][0])
      predicted_data.append(koko)
    df_predicted_data = pd.DataFrame(predicted_data,columns=['anser','predict','error','error_abs','relative_error','relative_error_abs'],index=Y)
    df_predicted_data=df_predicted_data.loc[save_period_start[l]:save_period_end[l]]
    df_predicted_data= df_predicted_data.sort_index()
    df_predicted_data.to_csv("/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ANN/predictANN"+str(n_epoch)+str(n_hidden)+PLACE+"/predictANN"+str(n_epoch)+str(n_hidden)+TEST_periodstart[l]+'-'+TEST_periodend[l]+PLACE+".csv")
    #print('モデルを使った予測',y_)

  
  # 予測と正解のグラフ作成
  
  
  # 学習モデルの保存
  #model.save('modelANN'+str(n_epoch)+str(n_hidden)+PLACE+'.h7')

def main():
  place =[
    '大阪',
    '東京',
    '諏訪',
    '那覇',
    '札幌'

  ]
  hidenn = [15,30,60,30,30]
  epoch =[60,60,60,30,120]
  for l in range(0,5):
    for i in range(0,5):
      print('start'+place[i]+str(hidenn[l])+str(epoch[l])+'learing')
      model_study(place[i],hidenn[l],epoch[l])

if __name__ =="__main__":
  main()