import timeit
from datetime import datetime

import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


import model
# from data_loader import TimeStampDataset
from range_data_loader import RangeDataset





############### 하이퍼 파라미터
nEpochs = 30       # Number of epochs for training
snapshot = 3       # Store a model every snapshot epochs
lr = 0.03
batch_size = 10
threshold = 0.95

save_path = "./save/1110_16_00"
dataset_path = "D:/news_frame"
xlsx_path = "./video_labeling.xlsx"
log_dir = os.path.join(save_path, 'logs')



def train():
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)


    ############### 모델 생성
    net = model.resnet50(class_num=3)     # 앵커, 기자, 인터뷰, (자료화면->지금은 나가리)에 대한 타임 스탬프 확률


    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))


    # Loss Function
    softmax = nn.Softmax(dim=2)
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


    net.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_dir=log_dir)


    ############### Dataset, DataLoader 생성
    # dataset = TimeStampDataset(xlsx_path=xlsx_path, root_path=dataset_path, sigma=3.0, image_size=(180,120))
    dataset = RangeDataset(xlsx_path=xlsx_path, root_path=dataset_path, sigma=6.0, spacing=8, image_size=(180,120))
    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) * 0.2)
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    print(f"Total number of train datas : {len(train_dataset)}")
    print(f"Total number of validation datas : {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
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


            optimizer.zero_grad()

            outputs = net(inputs)
            outputs = outputs.reshape(labels.shape[0], labels.shape[1], labels.shape[2])        # labels 모양이랑 맞춰주고 --> (Batch, Class, Frame)

            probs = softmax(outputs)

            labels = torch.clamp(labels, 0.0, 1.0)
            probs = torch.clamp(probs, 0.0, 1.0)

            ################## Loss 측정
            loss = criterion(probs, labels)
            running_loss += loss.item() * inputs.size(0)


            ################## Accuracy 측정
            labels_mean_max = labels.mean(dim=2).max(dim=1)
            labels = torch.where(labels_mean_max.values > threshold, labels_mean_max.indices, torch.sum(labels[:, :, 1:] - labels[:, :, :-1], dim=2).argmax(dim=1))
            probs_mean_max = probs.mean(dim=2).max(dim=1)
            pred_labels = torch.where(probs_mean_max.values > threshold, probs_mean_max.indices, torch.sum(probs[:, :, 1:] - probs[:, :, :-1], dim=2).argmax(dim=1))


            running_corrects += torch.sum(pred_labels == labels)

            
            loss.backward()
            optimizer.step()


        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)


        writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
        writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
        writer.add_scalar('data/learning_rate', lr, epoch)


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
            probs = softmax(outputs)


            ################## Loss 측정
            loss = criterion(probs, labels)
            running_loss += loss.item() * inputs.size(0)



            ################## Accuracy 측정
            labels_mean_max = labels.mean(dim=2).max(dim=1)
            labels = torch.where(labels_mean_max.values > threshold, labels_mean_max.indices, torch.sum(labels[:, :, 1:] - labels[:, :, :-1], dim=2).argmax(dim=1))
            probs_mean_max = probs.mean(dim=2).max(dim=1)
            pred_labels = torch.where(probs_mean_max.values > threshold, probs_mean_max.indices, torch.sum(probs[:, :, 1:] - probs[:, :, :-1], dim=2).argmax(dim=1))
            
            running_corrects += torch.sum(pred_labels == labels)



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

        scheduler.step()


    writer.close()


if __name__ == '__main__':
    train()