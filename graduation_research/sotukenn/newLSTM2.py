import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import losses

def get_Weather_data(FEATURE_VALUE,start1,end1,place):

  dfWeather_all = pd.read_csv("datasample"+str(place)+".csv", index_col=0, parse_dates=True, skiprows = 0, encoding = 'shift-jis', header=None)

  dfWeather_all.columns =[
      'average_temperature'
  ]
  dfWeather = dfWeather_all.loc[start1:end1, FEATURE_VALUE]
  dfWeather = dfWeather.sort_index()
  dfWeather = dfWeather.dropna() #欠損値を取り除く
  
  return  dfWeather

def generate_data(data, length_per_unit, dimension):
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
    X = np.array(sequences).reshape(len(sequences),length_per_unit,dimension)
    # 正解データを成形
    Y = np.array(target).reshape(len(sequences), dimension)
    # 正解データの日付データを成形
    Y_date = np.array(target_date).reshape(len(sequences), 1)

    return (X, Y, Y_date)

def buildmodel(input_shape,length_per_unit,hidden, neurons,time_step):
    #モデルの構築
    model = Sequential()
    #中間層
    model.add(LSTM(hidden, batch_input_shape = (None,length_per_unit,neurons), return_sequences = False ))
    model.add(Dropout(0.5))
    # input_shape = (input_shape)
    model.add(Dense(neurons))
    model.add(Activation('linear'))
    optimizer = Adam(lr = 0.001)
    model.compile(loss = losses.mean_squared_error, optimizer = optimizer,metrics= ['accuracy'])
    model.summary()

    return model

def study(PLACE,HIDDEN,EPOCH):
    #定数の定義
    FEATURE_VALUE = ['average_temperature']
    DIMENSION = len(FEATURE_VALUE)
    LENGTH_PER_UNIT = 365
    timestep = 365
    in_out_neurons = 1
    #隠れ層の数
    n_hidden = HIDDEN
    #学習用パラメータ
    batch = 90
    n_epoch = EPOCH
    #1学習期間
    leran_periodstart ='2016-01-01 01:00:00'
    leran_periodend ='2019-12-31 23:00:00'
    #予測期間
    TEST_periodstart =['2004-01-01 01:00:00','2009-01-01 01:00:00','2014-01-01 01:00:00','2019-01-01 01:00:00']
    TEST_periodend =['2005-12-31 23:00:00','2010-12-31 23:00:00','2015-12-31 23:00:00','2020-12-31 23:00:00']
    save_period_start =['2005/01/01 01','2010/01/01 01','2015/01/01 01','2020/01/01 01']
    save_period_end =['2005/12/31 23','2010/12/31 23','2015/12/31 23','2020/12/31 23']
    #入力の形状
    input_shape = (LENGTH_PER_UNIT,DIMENSION)
    #データ取得
    dfWeather= get_Weather_data(FEATURE_VALUE,leran_periodstart,leran_periodend,PLACE)
    #データの成形
    X_train, Y_train ,Y_train_data = generate_data(dfWeather, LENGTH_PER_UNIT, DIMENSION)
    model = buildmodel(input_shape,LENGTH_PER_UNIT,n_hidden,in_out_neurons,timestep)
    filepath = 'model-{epoch:02d}.h5'
    model_ckp=ModelCheckpoint(filepath,minitor='val_loss',verbose= 0,save_best_only = True,save_weight_only=False, mode = 'auto', save_freq= n_epoch)
    #学習
    hist = model.fit(X_train,Y_train,epochs = n_epoch,validation_split =0.5,batch_size= batch,callbacks=[model_ckp])
    #Accuracyの推移
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    #保存用のフォルダを作成
    new_dir_path = '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/LSTM/predictLSTM'+str(n_epoch)+str(n_hidden)+PLACE
    os.mkdir(new_dir_path)
    for l in range(0,4):
      Test_Data = get_Weather_data(FEATURE_VALUE,TEST_periodstart[l],TEST_periodend[l],PLACE)
      dfWeatherTest = Test_Data['average_temperature']
      TESTTEMPLE,y_test,Y_test_date = generate_data( dfWeatherTest,LENGTH_PER_UNIT, DIMENSION)
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
      df_predicted_data.to_csv("/Users/konishitakashidai/Desktop/卒研/卒研プログラム/LSTM/predictLSTM"+str(n_epoch)+str(n_hidden)+PLACE+"/predictRNN"+str(n_epoch)+str(n_hidden)+TEST_periodstart[l]+'-'+TEST_periodend[l]+PLACE+".csv")




    #  # モデルと学習結果を保存する
    #model.save('LSTM.h5')






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
      study(place[i],hidenn[l],epoch[l])



if __name__ == "__main__":
    main()
