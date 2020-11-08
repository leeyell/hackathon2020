import timeit
from datetime import datetime

import os
import glob
from tqdm import tqdm
import pickle

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


import model
from data_loader import TimeStampDataset
# from convert_from_caffe2 import load_checkpoint




############### 하이퍼 파라미터
nEpochs = 100       # Number of epochs for training
snapshot = 3       # Store a model every snapshot epochs
lr = 0.03
batch_size = 10
threshold = 0.2        # 클래스 0 ~ 3 중 하나를 예측하는 가장 높은 확률이 이 값 미만이면 해당 예측값은 해당되는 class 가 없다는, 즉 클래스 4 를 예측하는 것으로 친다.


save_path = "./save/1108_22_20"
dataset_path = "D:/news_frame"
xlsx_path = "./video_labeling.xlsx"
log_dir = os.path.join(save_path, 'logs')



def train():
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)


    ############### 모델 생성
    net = model.resnet50(class_num=4)     # 앵커, 기자, 인터뷰, 자료화면에 대한 타임 스탬프 확률

    # pre-trained model 불러오기
    # pretrained_path = "SLOWFAST_4x16_R50.pkl" 조졌음 이건 망했어

    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))


    # Loss Function
    sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # 20 에폭마다 learning rate 0.5배


    net.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_dir=log_dir)


    ############### Dataset, DataLoader 생성
    dataset = TimeStampDataset(xlsx_path=xlsx_path, root_path=dataset_path, sigma=3.0, image_size=(180,120))
    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) * 0.2)
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    print(f"Total number of train datas : {len(train_dataset)}")
    print(f"Total number of validation datas : {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Total number of train batches : {len(train_loader)}")
    print(f"Total number of train batches : {len(val_loader)}")



    ############### 학습 시작
    best_acc_on_train = 0.0
    best_acc_on_test = 0.0
    for epoch in range(0, nEpochs):
        ############### Train
        net.train()
        running_loss = 0.0          # 한 epoch 동안의 loss
        running_corrects = 0.0      # 한 epoch 동안의 corrects

        start_time = timeit.default_timer()

        for inputs, labels in tqdm(train_loader):
            inputs = Variable(inputs, requires_grad=True).to(device=device, dtype=torch.float32)
            labels = Variable(labels).to(device=device, dtype=torch.float32)
            #### labels 를 1차원으로 펴준다. --> [0 ~ n 프레임의 클래스 0 일 확률 쭉 ... ,
            ####                                         0 ~ n 프레임의 클래스 1 일 확률 쭉 ... ,
            ####                                               0 ~ n 프레임의 클래스 2 일 확률 쭉 ..., ...]
            #### labels = labels.view(labels.shape[0], -1)    # shape = (batch_size, 4*frames)
            
            optimizer.zero_grad()

            outputs = net(inputs)

            # outputs 은 지금 이 꼴로 나옴 --> 배치 개수 x [0 ~ n 프레임의 클래스 0 일 확률 쭉 ... ,
            #                                                   0 ~ n 프레임의 클래스 1 일 확률 쭉 ... ,
            #                                                           0 ~ n 프레임의 클래스 2 일 확률 쭉 ..., ...]    --> (Batch, Class*Frame)
            outputs = outputs.reshape(labels.shape[0], labels.shape[1], labels.shape[2])        # labels 모양이랑 맞춰주고 --> (Batch, Class, Frame)
            probs = sigmoid(outputs)



            ################## Loss 측정
            loss = criterion(probs, labels)
            running_loss += loss.item() * inputs.size(0)


            ################## Accuracy 측정
            # labels 에서 각 클래스 별 가장 높은 확률들만 가져옴
            # [class 0 을 나타내는 확률 중 가장 높은 값, class 1 을 나타내는 확률 중 가장 높은 값, ... ] --> (10, 4)
            labels = torch.max(labels, dim=2)[0]        # 여기서 labels 값은 각 클래스 별 가장 높은 확률임
            # 그 중에서도 제일 높은 확률을 갖는 class 가 해당 프레임 시퀀스의 class 로 결정   --> (10)
            labels = torch.max(labels, dim=1)[1]        # 여기서 labels 값은 클래스 그 자체임

            # outputs 에서 각 클래스 별 가장 높은 확률들만 가져옴
            # [class 0 을 나타내는 확률 중 가장 높은 값, class 1 을 나타내는 확률 중 가장 높은 값, ... ] --> (10, 4)
            outputs = torch.max(probs, dim=2)[0]
            # 어떤 class 인지는 몰라도... 가장 높은 확률값인데? threshold 이상의 확률이 아니라면 아무 클래스도 아니라는... 4 의 class 를 부여
            # threshold 를 넘는 확률값이라면 그 class 를 가져옴 (argmax = 클래스니까 인덱스를 가져와야 함)
            outputs = torch.tensor([output.argmax() if output.max() > threshold else 4 for output in outputs])      # --> (10)
            
            outputs = outputs.to(device=device, dtype=torch.float32)
            running_corrects += torch.sum(outputs == labels)


            
            loss.backward()
            optimizer.step()


        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # if phase == 'train':
        writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
        writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
        # else:
        #     writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
        #     writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

        if epoch_acc > best_acc_on_train:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_path, 'train_epoch-' + str(epoch) + '-best_model.pth'))
            best_acc_on_train = epoch_acc

        print("[Train] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        ############### validation
        net.eval()

        running_loss = 0.0
        running_corrects = 0.0

        start_time = timeit.default_timer()

        for inputs, labels in tqdm(val_loader):
            inputs = Variable(inputs, requires_grad=True).to(device=device, dtype=torch.float32)
            labels = Variable(labels).to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                outputs = net(inputs)
            
            outputs = outputs.reshape(labels.shape[0], labels.shape[1], labels.shape[2])
            probs = sigmoid(outputs)


            ################## Loss 측정
            loss = criterion(probs, labels)
            running_loss += loss.item() * inputs.size(0)



            ################## Accuracy 측정
            labels = torch.max(labels, dim=2)[0]
            labels = torch.max(labels, dim=1)[1]

            outputs = torch.max(probs, dim=2)[0]
            outputs = torch.tensor([output.argmax() if output.max() > threshold else 4 for output in outputs])
            
            outputs = outputs.to(device=device, dtype=torch.float32)
            running_corrects += torch.sum(outputs == labels)


        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
        writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

        if epoch_acc > best_acc_on_test:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_path, 'test_epoch-' + str(epoch) + '-best_model.pth'))
            best_acc_on_test = epoch_acc

        print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")



    writer.close()


if __name__ == '__main__':
    train()