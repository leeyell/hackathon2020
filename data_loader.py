import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np

import os
import xlrd

class TimeStampDataset(Dataset):
  def __init__(self, xlsx_path, root_path, sigma, fps=8, frames=24, frame_overlaps=8):
    self.root_path = root_path
    self.sigma = sigma
    self.fps = fps
    self.frames = frames
    self.frame_overlaps = frame_overlaps

    self.to_tensor = transforms.ToTensor()

    wb = xlrd.open_workbook(xlsx_path)
    sheet = wb.sheet_by_index(0)
    headers = [sheet.cell_value(0, col) for col in range(5)]
    headers

    self.labels = []

    row = 1
    while True:
      row_labels = {}
      for col in range(5):
        try:
          row_labels[headers[col]] = sheet.cell_value(row, col)
        except:
          break
      else:
        self.labels.append(row_labels)
        row += 1
        continue
      break

    self.total_frame_count = 0
    self.total_data_count = 0

    for row_labels in self.labels:
      row_labels['Video_Id'] = int(row_labels['Video_Id']) if row_labels['Video_Id'] != '' else -1

      time_stamps = filter(lambda x: x != '', row_labels['Time_stamp'].split(','))
      time_stamps_split = map(lambda x: x.split(':'), time_stamps)
      row_labels['Time_stamp'] = list(map(lambda x: int(x[0]) * 60 + int(x[1]), time_stamps_split))
      row_labels['time_stamp_class'] = list(map(int, filter(lambda x: x != '', row_labels['time_stamp_class'].split(','))))
      
      video_name = row_labels['Video_name']
      ext = video_name.rfind('.')
      if ext > 0:
        video_name = video_name[:ext]
      row_labels['Video_name'] = video_name
      
      row_labels['frames'] = len([path for path in os.listdir(os.path.join(self.root_path, video_name)) if path[-4:] == '.jpg'])
      row_labels['data_count'] = int(np.ceil((row_labels['frames'] - self.frame_overlaps) / (self.frames - self.frame_overlaps)))

      self.total_frame_count += row_labels['frames']
      self.total_data_count += row_labels['data_count']

    self.label = torch.zeros(4, self.total_frame_count)
    
    frame_offset = 0
    for row_labels in self.labels:
      timestamps = row_labels['Time_stamp']
      ts_classes = row_labels['time_stamp_class']
      frames = row_labels['frames']

      frame_indices = torch.arange(frames, dtype=torch.float)
      for i in range(len(timestamps)):
        self.label[ts_classes[i], frame_offset:frame_offset + frames] += torch.exp(-0.5 * (((timestamps[i] + 0.5) * self.fps - frame_indices) / self.sigma) ** 2)
      frame_offset += frames

    self.label = torch.clamp(self.label, 0.0, 1.0)

  def __len__(self):
    return self.total_data_count
  
  def __getitem__(self, idx):
    offset, frame_offset = 0, 0
    for row_labels in self.labels:
      if idx - offset < row_labels['data_count']:
        break
      offset += row_labels['data_count']
      frame_offset += row_labels['frames']

    frame_end = min((self.frames - self.frame_overlaps) * (idx - offset + 1) + self.frame_overlaps, row_labels['frames'])
    frame_start = frame_end - self.frames

    images = []
    for frame in range(frame_start, frame_end):
      images.append(self.to_tensor(Image.open(os.path.join(self.root_path, row_labels['Video_name'], f'{str(frame + 1).zfill(5)}.jpg'))))

    return torch.stack(images, dim=1), self.label[:, frame_offset + frame_start:frame_offset + frame_end]

if __name__ == '__main__':
  import matplotlib.pyplot as plt

  dataset = TimeStampDataset('/content/drive/My Drive/video_labeling.xlsx', '/content/drive/My Drive/해커톤2020/AI 학습용 구축 원천 데이터/찐뉴스/news_frame/', 3.0)
  dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

  print(len(dataset))

  for data, label in dataloader:
    for dat, lbl in zip(data, label):
      fig, axes = plt.subplots(1, 24, figsize=(72, 3))
      for ax, img in zip(axes, dat.permute(1, 2, 3, 0)):
        ax.imshow(img)
      plt.show()
      fig, axes = plt.subplots(4, 1, figsize=(10, 3))
      for ax, img in zip(axes, lbl):
        ax.imshow(img.view(1, 24), vmin=0, vmax=1)
      plt.show()
    break