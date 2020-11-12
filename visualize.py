
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

import model
from class_data_loader import ClassDataset

import os
import xlrd
import json
import matplotlib.pyplot as plt

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, xlsx_path, root_path, transform=None, image_size=None):
        self.root_path = root_path

        self.transform = []
        if transform is not None:
            self.transform.append(transform)
        if image_size is not None:
            self.transform.append(transforms.Resize(image_size))
        self.transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(self.transform)

        wb = xlrd.open_workbook(xlsx_path)
        sheet = wb.sheet_by_index(0)
        headers = [sheet.cell_value(0, col).lower() for col in range(5)]

        self.labels = []

        row = 1
        while True: 
            try:
                if sheet.cell_value(row, 0) == '':
                    break
            except:
                break
            row_labels = {}     
            for col in range(5):
                row_labels[headers[col]] = sheet.cell_value(row, col)
            self.labels.append(row_labels)
            row += 1

        for row_labels in self.labels:
            row_labels['video_id'] = int(row_labels['video_id']) if row_labels['video_id'] != '' else -1

            time_stamps = filter(lambda x: x != '', row_labels['time_stamp'].split(','))
            time_stamps_split = map(lambda x: x.split(':'), time_stamps)
            row_labels['time_stamp'] = list(map(lambda x: int(x[0]) * 60 + int(x[1]), time_stamps_split))
            row_labels['time_stamp_class'] = list(map(int, filter(lambda x: x != '', row_labels['time_stamp_class'].split(','))))
            
            video_name = row_labels['video_name']
            ext = video_name.rfind('.')
            if ext > 0:
                video_name = video_name[:ext]
                row_labels['video_name'] = video_name
            
            row_labels['frames'] = len([path for path in os.listdir(os.path.join(self.root_path, video_name)) if path[-4:] == '.jpg'])

    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, idx):
        labels = self.labels[idx]
        images = torch.stack([self.transform(Image.open(os.path.join(self.root_path, labels['video_name'], f'{str(frame + 1).zfill(5)}.jpg'))) for frame in range(0, labels['frames'])], dim=1)
        return labels['video_id'], images

save_path = "./save/1111_13_40"
dataset_path = "C:/news_frame/"
xlsx_path = "./video_labeling.xlsx"

batch_size = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = model.resnet50(class_num=4)
net.to(device)

net.load_state_dict(torch.load('./save/models/train_epoch-46-best_model.pth')['state_dict'])
net.eval()

dataset = ImageDataset(xlsx_path, dataset_path, image_size=(180, 120))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

samples = []
outputs = []

softmax = nn.Softmax()

with torch.no_grad():
    for i, (video_id, sample) in enumerate(data_loader):
        for j in range(0, sample.size(2), 32):
            samples.append(sample[0, :, j])

        sample = sample.to(device)

        for j in range(0, sample.size(2) - 16):
            output = softmax(net(sample[:, :, j:j + 16]))
            outputs.append(output)
            print(j)

        '''print(output.argmax(dim=1))

        fig, axes = plt.subplots(1, len(sample), figsize=(len(sample), 1))
        for ax, img in zip(axes, sample.permute(0, 2, 3, 1)):
            ax.imshow(img)
            ax.axis('off')
        plt.show()'''
        break

samples = torch.stack(samples)
outputs = torch.cat(outputs)

fig, axes = plt.subplots(1, len(samples), figsize=(len(samples), 1))
for ax, img in zip(axes, samples.permute(0, 2, 3, 1)):
    ax.imshow(img)
    ax.axis('off')
plt.savefig('./images.png')
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(len(samples), 1))
axes.imshow(torch.repeat_interleave(outputs.T, 10, dim=0).cpu())
axes.axis('off')
plt.savefig('./outputs.png')
plt.show()