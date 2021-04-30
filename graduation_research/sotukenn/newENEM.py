#ライブラリ定義
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM,Dropout,SimpleRNN
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
from keras import losses
#データ取得用関数
def get_Weather_data(FEATURE_VALUE,start1,end1,place):
  dfWeather_all = pd.read_csv("datasample"+str(place)+".csv", index_col=0, parse_dates=True, skiprows = 0, encoding = 'shift-jis', header=None)
  dfWeather_all.columns =[
      'average_temperature'
  ]
  dfWeather = dfWeather_all.loc[start1:end1, FEATURE_VALUE]
  dfWeather = dfWeather.sort_index()
  dfWeather = dfWeather.dropna() #欠損値を取り除く
  return  dfWeather
#データ成形
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
#LSTMモデル構築
def LSTM_templebuildmodel(input_shape,length_per_unit,hidden, neurons,time_step):
  #モデルの構築
  model = Sequential()
  #中間層
  model.add(LSTM(hidden, batch_input_shape = (None,length_per_unit,neurons), return_sequences = False))
  model.add(Dropout(0.5))
  model.add(Dense(neurons))
  model.add(Activation('linear'))
  optimizer = Adam(lr = 0.001)
  model.compile(loss = losses.mean_squared_error, optimizer = optimizer,metrics= ['accuracy'])
  model.summary()
  LSTM_templemodel = model
  return LSTM_templemodel
#LSTMモデルの学習
def LSTMtemple_Training(model,X_train,Y_train,TEST,batch,n_epoch,hidden,place,number):
  filepath = 'model-{epoch:02d}.h5'
  #学習用パラメータ
  model_ckp=ModelCheckpoint(filepath,minitor='val_loss',verbose= 0,save_best_only = True,save_weight_only=False, mode = 'auto', save_freq= n_epoch)
  #学習
  hist = model.fit(X_train,Y_train,epochs = n_epoch,validation_split =0.1,batch_size= batch,callbacks=[model_ckp])
  #グラフを表示
  predicted = model.predict(X_train)
  #モデルと学習結果を保存する
  model.save(
    '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/モデルENEM'+str(n_epoch)+str(hidden)+place+
    '/LSTM'
    +str(n_epoch)+'-'+str(hidden)+place+str(number)+'.h5')
  return predicted
#アンサンブル用学習器の構築
def build_model(input_shape,length_per_unit,hidden, neurons,time_step):
  #モデルの構築
  model = Sequential()
  #中間層
  model.add(LSTM(hidden, batch_input_shape = (None,length_per_unit,neurons), return_sequences = False))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('linear'))
  optimizer = Adam(lr = 0.001)
  model.compile(loss = losses.mean_squared_error, optimizer = optimizer,metrics= ['accuracy'])
  model.summary()
  return model
#アンサンブル学習器の学習
def ensamble_model_Training(model,X_train,Y_train,batch,n_epoch,hidden,place):
  filepath = 'model-{epoch:02d}.h5'
  #学習用パラメータ
  model_ckp=ModelCheckpoint(filepath,minitor='val_loss',verbose= 0,save_best_only = True,save_weight_only=False, mode = 'auto', save_freq= n_epoch)
  #学習
  hist = model.fit(X_train,Y_train,epochs = n_epoch,validation_split =0.3,batch_size= batch,callbacks=[model_ckp])
  
  # モデルと学習結果を保存する
  model.save(
    '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/モデルENEM'
    +str(n_epoch)+str(hidden)+place+
    '/ENEM'
    +str(n_epoch)+'-'+str(hidden)+place+'.h5')
  return 
#予測結果のデータを合成
def make_dataset(temple_data,predicted_temple_data):
  t = temple_data.tolist()
  tp= predicted_temple_data.tolist()
  Stack_predicted_data = [ ]
  tem2 =[]
  for i in range(0,len(t)):
    temple =t[i]
    tem = []
    for j in range(0,len(temple)):
      tem.append(temple[j][0])

    tem.append(tp[i][0])
    tem2.append(tem)
    

  Stack_predicted_data = np.array(tem2)

  Stack_predicted_data = Stack_predicted_data.reshape(len(Stack_predicted_data),len(temple)+1,1)
  return Stack_predicted_data
#RNNモデルの構築
def buildRNNmodel(input_shape,length_per_unit,hidden, neurons,time_step):
    #モデルの構築
    model = Sequential()
    #中間層
    model.add(SimpleRNN(hidden, batch_input_shape = (None,length_per_unit,neurons), return_sequences = False ))
    model.add(Dropout(0.5))
    # input_shape = (input_shape)
    model.add(Dense(neurons))
    model.add(Activation('linear'))
    optimizer = SGD(lr = 0.001)
    model.compile(loss = losses.mean_squared_error, optimizer = optimizer,metrics= ['accuracy'])
    model.summary()
    return model
#RNNモデルの学習
def RNNmodel_Training(model,temple_train,RNNtemple_Ytrain,n_epoch,batch_size,hidden,place):
  filepath = 'model-{epoch:02d}.h5'
  #学習用パラメータ
  model_ckp=ModelCheckpoint(filepath,minitor='val_loss',verbose= 0,save_best_only = True,save_weight_only=False, mode = 'auto', save_freq= n_epoch)
  #学習
  hist = model.fit(temple_train,RNNtemple_Ytrain,epochs = n_epoch,validation_split =0.1,batch_size= batch_size,callbacks=[model_ckp])
  model.save(
    '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/モデルENEM'
    +str(n_epoch)+str(hidden)+place+
    '/RNN'
    +str(n_epoch)+'-'+str(hidden)+place+'.h5')
#各モデル予測
def model_load(n_epoch,hidden,place,number1,number2):
  new_model1 = tf.keras.models.load_model(
    '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/モデルENEM'
    +str(n_epoch)+str(hidden)+place+
    '/LSTM'
    +str(n_epoch)+'-'+str(hidden)+place+str(number1)+'.h5'
    )
  new_model1_2 = tf.keras.models.load_model(
    '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/モデルENEM'
    +str(n_epoch)+str(hidden)+place+
    '/LSTM'
    +str(n_epoch)+'-'+str(hidden)+place+str(number2)+'.h5'
    )
  new_model2 = tf.keras.models.load_model(
    '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/モデルENEM'
    +str(n_epoch)+str(hidden)+place+
    '/ENEM'
    +str(n_epoch)+'-'+str(hidden)+place+'.h5'
    )
  new_model3 = tf.keras.models.load_model(
    '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/モデルENEM'
    +str(n_epoch)+str(hidden)+place+
    '/RNN'
    +str(n_epoch)+'-'+str(hidden)+place+'.h5'
    )
  return new_model1,new_model1_2,new_model2,new_model3


def study(PLACE,HIDDEN,EPOCH):
#各種定数_学習期間の定義
  #LSTM_templeの定数の定義
  FEATURE_VALUE = ['average_temperature']
  LENGTH_PER_UNIT = 719
  DIMENSION = len(FEATURE_VALUE)
  LSTM_temple_input_shape = (LENGTH_PER_UNIT,DIMENSION)
  #アンサンブル用の定数
  DIMENSION_ensemble = 1
  LENGTH_PER_UNIT_ensemble = LENGTH_PER_UNIT+2
  ensemble_in_out_neurons = 1
  ensemble_input_shape =  (LENGTH_PER_UNIT_ensemble,DIMENSION_ensemble)
  #共通の定数を定義
  time_step = 100
  n_hidden = 1
  LSTMtemple_in_out_neurons = 1
  in_out_neurons = 1
  epoch = 1
  batch_size = 720
  input_shape = (LENGTH_PER_UNIT,DIMENSION)
  #学習期間
  leran_periodstart ='2016-01-01 01:00:00'
  leran_periodend ='2019-12-31 23:00:00'
  #予測期間
  TEST_periodstart =['2004-06-01 01:00:00','2009-06-01 01:00:00','2014-06-01 01:00:00','2019-06-01 01:00:00']
  TEST_periodend =['2005-12-31 23:00:00','2010-12-31 23:00:00','2015-12-31 23:00:00','2020-12-31 23:00:00']
  save_period_start =['2005/01/01 01','2010/01/01 01','2015/01/01 01','2020/01/01 01']
  save_period_end =['2005/12/31 23','2010/12/31 23','2015/12/31 23','2020/12/31 23']
#学習データの取得と成形
  #学習データ取得
  dfWeather= get_Weather_data(FEATURE_VALUE,leran_periodstart,leran_periodend,PLACE)
  #データ成形
  temple_Xtrain,temple_Ytrain,LSTMtemple_Ytrain_data = generate_data(dfWeather, LENGTH_PER_UNIT, DIMENSION)
  LSTM_X_test,LSTM_Y_test,LSTM_Y_date_test = generate_data(dfWeather, LENGTH_PER_UNIT, DIMENSION)
#気温予測アンサンブル部分モデル
  #モデル保存用のフォルダ作成
  new_dir_path = '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/モデルENEM'+str(epoch)+str(n_hidden)+PLACE
  os.mkdir(new_dir_path)
  #モデル構築
  print("----------学習器1------------")
  LSTM_temple_model1 = LSTM_templebuildmodel(LSTM_temple_input_shape,LENGTH_PER_UNIT,n_hidden,LSTMtemple_in_out_neurons,time_step)
  temple_predicted1= LSTMtemple_Training(LSTM_temple_model1,temple_Xtrain,temple_Ytrain,LSTM_X_test,batch_size,epoch,n_hidden,PLACE,1)
  temple_train = temple_Xtrain
  #訓練用のデータを追加する
  temple1 = make_dataset(temple_train,temple_predicted1)
  print("----------学習器2------------")
  LSTM_temple_model2 = LSTM_templebuildmodel(LSTM_temple_input_shape,LENGTH_PER_UNIT+1,n_hidden,LSTMtemple_in_out_neurons,time_step)
  temple_predicted2= LSTMtemple_Training(LSTM_temple_model2,temple1,temple_Ytrain,LSTM_X_test,batch_size,epoch,n_hidden,PLACE,2)
  #訓練用のデータを追加する
  temple2 = make_dataset(temple1,temple_predicted2 )
  
  print("---------------ここからアンサンブル学習開始-----------------")
  #temple_data2 = np.array(temple2)

  ensemble_model = build_model(ensemble_input_shape,LENGTH_PER_UNIT_ensemble,n_hidden,ensemble_in_out_neurons,time_step)
  ensamble_model_Training(ensemble_model,temple2,temple_Ytrain,batch_size,epoch,n_hidden,PLACE)
  print("----------学習器3------------")
#RNNのモデルで予測
  model = buildRNNmodel(input_shape,LENGTH_PER_UNIT,n_hidden,in_out_neurons,time_step)
  RNNmodel_Training(model,temple_Xtrain,temple_Ytrain,epoch,batch_size,n_hidden,PLACE)
#予測結果保存用フォルダ作成
  new_dir_path = '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/ENEM'+PLACE
  os.mkdir(new_dir_path)
  new_dir_path = '/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/ENEM'+PLACE+'/predictENEM'+str(epoch)+str(n_hidden)+PLACE
  os.mkdir(new_dir_path)
#保存したモデルをダウンロード
  LSTM_model1,LSTM_model2,ENEM_model,RNN_model = model_load(epoch,n_hidden,PLACE,1,2)
#モデルを使用してテストデータを予測する
  print("期間ごとの予測開始")
  for l in range(0,4):
    print(l)
    #テスト用データを取得、成形
    Test_Data = get_Weather_data(FEATURE_VALUE,TEST_periodstart[l],TEST_periodend[l],PLACE)
    dfWeatherTest = Test_Data['average_temperature']
    TESTTEMPLE,y_test,Y_test_date = generate_data( dfWeatherTest,LENGTH_PER_UNIT, DIMENSION)
    # モデルを使った予測    
    print("モデル使って予測")
    LSTM_predict = LSTM_model1.predict(TESTTEMPLE)
    temple=make_dataset(TESTTEMPLE,LSTM_predict)
    LSTM_predict2 = LSTM_model2.predict(temple)
    ENEMtemp=make_dataset(temple,LSTM_predict2)
    ensemble_predict = ENEM_model.predict(ENEMtemp)
    RNN_predict = RNN_model.predict(TESTTEMPLE)
    #予測結果の平均を算出
    predicted_ave =[]
    print("結果の平均値の算出")
    print(np.shape(ensemble_predict))
    i=0
    while i<len(ensemble_predict):
      temple_value =[]
      sum = ensemble_predict[i][0]+RNN_predict[i][0]
      ave = sum/2
      temple_value.append(ave)
      predicted_ave.append(predicted_ave)
      print(i)
      i +=1
    #予測結果の誤差等の計算

    predicted_data =[]
    Y =[]
    print("学習結果の保存_______")
    print(len(y_test))
    for j in range(0,len(y_test)):
      koko = []
      koko.append(y_test[j][0])
      koko.append(predicted_ave[j][0])
      koko.append(y_test[j][0]-predicted_ave[j][0])
      koko.append(np.abs(predicted_ave[j][0]-y_test[j][0]))
      Y.append(Y_test_date[j][0])
      predicted_data.append(koko)
    #データフレーム型にして保存、必要な期間に成形
    df_predicted_data = pd.DataFrame(predicted_data,columns=['anser','predict','error','error_abs'],index=Y)
    df_predicted_data=df_predicted_data.loc[save_period_start[l]:save_period_end[l]]
    df_predicted_data= df_predicted_data.sort_index()
    df_predicted_data.to_csv('/Users/konishitakashidai/Desktop/卒研/卒研プログラム/ENEM/ENEM'+PLACE+
    '/predictENEM'+str(epoch)+str(n_hidden)+PLACE+
    "/predictENEM"+str(epoch)+str(n_hidden)+TEST_periodstart[l]+'-'+TEST_periodend[l]+PLACE+".csv")
    print("学習結果の保存_______終了")
  print("期間ごとの予測終了")

#メインプログラム
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
