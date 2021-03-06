#!/usr/bin/python3
# coding: utf-8
import os
import glob
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

def data_split(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=1)
  X_train, X_val, y_train, y_valid = train_test_split(X_train, y_train, test_size=10, random_state=1)
  return X_train, X_val, X_test, y_train, y_valid, y_test

class ImgPathManager:
  def __init__(self, func): 
    self.showPopup = func

    self.path_train = np.empty(0)
    self.y_train = np.empty(0)
    self.path_valid = np.empty(0)
    self.y_valid = np.empty(0)
    self.path_test = np.empty(0)
    self.y_test = np.empty(0)

    rootpath = "./data/"
    try:
      names = os.listdir(rootpath)
      self.category_num = len(names)
      if self.category_num == 0:
        raise FileNotFoundError("error!")
    except:
      self.showPopup()
      return

    self.idx_to_names = {idx: name for idx, name in zip(range(self.category_num), names)}
    for idx in self.idx_to_names.keys():
      target_path = os.path.join(rootpath, self.idx_to_names[idx], "*.jpg") 

      path_list = np.empty(0)
      idx_list = np.empty(0)
      for path in glob.glob(target_path):
        path_list = np.append(path_list, path)
        idx_list = np.append(idx_list, idx)

      Xtr, Xva, Xte, ytr, yva, yte = data_split(path_list, idx_list)
      self.path_train = np.append(self.path_train, Xtr)
      self.y_train = np.append(self.y_train, ytr)
      self.path_valid = np.append(self.path_valid, Xva)
      self.y_valid = np.append(self.y_valid, yva)
      self.path_test = np.append(self.path_test, Xte)
      self.y_test = np.append(self.y_test, yte)
  
  def getData(self, phase="train"):
    if phase == "train":
      return self.path_train, self.y_train
    elif phase == "validation":
      return self.path_valid, self.y_valid
    elif phase == "test":
      return self.path_test, self.y_test

  def getLabel(self, idx):
    return self.idx_to_names[idx]
     
class ImageTransform():
  def __init__(self):
    self.data_transform = transforms.Compose([
      transforms.RandomResizedCrop(size=(30, 30), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
      transforms.ToTensor(), 
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  def __call__(self, img):
    return self.data_transform(img)

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data, transform=None):
    self.file_list, labels = data
    self.labels = torch.tensor(labels, dtype=torch.long)
    self.file_transform = transform # 前処理クラスのインスタンス
  
  def __len__(self):
    return len(self.file_list)
  
  def __getitem__(self, index):
    img_path = self.file_list[index]           # index番目のpath
    img = Image.open(img_path)                 # index番目の画像ロード
    img_transformed = self.file_transform(img) # 前処理クラスでtensorに変換
    # print(type(self.labels))
    # print(type(self.labels[index]))
    label = self.labels[index]
    # print(type(label))
    return img_transformed, label


# モデルの更新と評価(modeによって切り替え)
def calculate_model(model, loss_fn, opt, dataloader, device, mode):
  if mode == 'evaluate':
    model.eval() # 学習を行わない時はevaluateモードに切り替え

  sum_loss = 0
  correct = 0
  count = len(dataloader.dataset)
  
  for X, y in dataloader:
    X = X.to(device)      # GPUへデータを転送
    y = y.to(device)      # GPUへデータを転送
    y_pred = model(X)     # Xからyを予測（softmaxを行う前の値が出力される）
    
    _, predicted = torch.max(y_pred.data, 1) # 6クラスのうち、予測確率最大のクラス番号を取得
    correct += (predicted == y).sum().item() # 予測に成功した件数をカウント（accuracy計算用）
    
    loss = loss_fn(y_pred, y)        # ミニバッチ内の訓練誤差の 平均 を計算
    sum_loss += loss.item()*len(y) # エポック全体の訓練誤差の 合計 を計算しておく
    
    if mode == 'update':
      # 重みの更新
      opt.zero_grad()
      loss.backward()
      opt.step()
    
  # エポック内の訓練誤差の平均値と予測精度を計算
  mean_train_loss = sum_loss / count
  train_accuracy = correct / count
  
  if mode == 'evaluate':
    model.train() # evaluate状態からtrain状態に戻しておく

  return mean_train_loss, train_accuracy    


def predict(model, path, device):
  model.eval() 
  img = Image.open(path)
  transform = ImageTransform()
  img_transformed = transform(img)
  X = img_transformed.unsqueeze(0)
  X = X.to(device) # GPUへデータを転送
  y_pred = model(X) # Xからyを予測    
  _, predicted = torch.max(y_pred.data, 1) # 予測確率最大のクラス番号を取得
  model.train()
  return predicted


# ==============================================================
class FaceLearner(QThread):
  changeProgress = pyqtSignal(int)
  triggerPopup = pyqtSignal()

  def __init__(self):
    super().__init__()
    self.data_exist = True

    # モデルの作成と学習の実行
    torch.manual_seed(0) # 学習結果の再現性を担保

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.loadData()

    # モデルパラメータのロード
    try:
      self.cnn.load_state_dict(torch.load("./model_state.pt"))
    except (RuntimeError, FileNotFoundError, AttributeError) as e:
      # print('catch RuntimeError:', e)
      print("モデルパラメータのロードに失敗しました．再学習を行って下さい．")
      pass
  
  def showPopup(self):
    self.triggerPopup.emit()

  def loadData(self):
    batch_size = 4
    self.ipm = ImgPathManager(self.showPopup)

    try: 
      self.train_dataset = Dataset(data=self.ipm.getData(phase="train"), transform=ImageTransform())
      self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

      self.valid_dataset = Dataset(data=self.ipm.getData(phase="validation"), transform=ImageTransform())
      self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=True)

      self.test_dataset = Dataset(data=self.ipm.getData(phase="test"), transform=ImageTransform())
      self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

      # 入力画像のサイズ(30,30)
      self.cnn = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, (5, 5), padding=2),    # (28,28)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),                        # (14,14)
        torch.nn.Conv2d(16, 32, (3, 3)),              # (12,12)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),                        # ( 6, 6)
        torch.nn.Conv2d(32, 64, (3, 3)),              # ( 4, 4)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),                        # ( 2, 2)
        torch.nn.Dropout(0.25),
        torch.nn.Flatten(), 
        torch.nn.Linear(64*2*2, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(128, self.ipm.category_num),
      )
      self.cnn.to(self.device)
    
      # 誤差関数と最適化手法を準備
      self.loss_fn = torch.nn.CrossEntropyLoss()
      # optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)
      self.optimizer = torch.optim.Adagrad(self.cnn.parameters(), lr=0.01)
      self.data_exist = True
    except ValueError as e:
      self.data_exist = False
      return

  def run(self):
    self.loadData()

    if self.data_exist:
      # 学習の実行
      trained_model = self.train(self.cnn, self.loss_fn, self.optimizer, self.train_loader, self.valid_loader, self.device, epoch=20)

      # モデルパラメータの保存
      torch.save(trained_model.state_dict(), "./model_state.pt")

  def test(self):
    # 予測の実行
    test_loss, test_accuracy = calculate_model(self.cnn, self.loss_fn, None, self.test_loader, self.device, mode='evaluate')
    print("テストデータの予測正解率 : ", test_accuracy)

  def predict(self, file_path):
    if self.data_exist:
      label_predicted = predict(self.cnn, file_path, self.device)
      os.remove(file_path)
      return self.ipm.getLabel(label_predicted.numpy()[0])
    else:
      os.remove(file_path)
      return "error"
  
  # 学習
  def train(self, model, loss_fn, opt, train_loader, valid_loader, device, epoch=50):
    self.changeProgress.emit(0)
    # liveloss = PlotLosses() # 描画の初期化
    for i in range(epoch):
      print(str(i+1) + " / " + str(epoch))
      train_loss, train_accuracy = calculate_model(model, loss_fn, opt, train_loader, device, mode='update')
      valid_loss, valid_accuracy = calculate_model(model, loss_fn, None, valid_loader, device, mode='evaluate')
      
      self.changeProgress.emit((i+1)/epoch*100)
    
      # Visualize the loss and accuracy values.
      # liveloss.update({
      #     'log loss': train_loss,
      #     'val_log loss': valid_loss,
      #     'accuracy': train_accuracy,
      #     'val_accuracy': valid_accuracy,
      # })
      # liveloss.draw()  
    print('Accuracy: {:.4f} (valid), {:.4f} (train)'.format(valid_accuracy, train_accuracy))
    return model # 学習したモデルを返す

# fc = FaceLearner()
# fc.start()
# fc.wait()
# fc.test()


