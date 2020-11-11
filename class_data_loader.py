import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np

import os
import xlrd

class ClassDataset(Dataset):
    def __init__(self, xlsx_path, root_path, spacing, seed, transform=None, image_size=None, fps=8, frames=16, classes=4):
        self.root_path = root_path
        self.spacing = spacing
        self.fps = fps
        self.frames = frames
        self.classes = classes

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

        self.class_indices = []
        self.lookup = []
        self.lookup_per_class = [[] for _ in range(self.classes)]

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
            
            row_labels['frames'] = frames = len([path for path in os.listdir(os.path.join(self.root_path, video_name)) if path[-4:] == '.jpg'])

            data_count = 0
            frame_start = int(row_labels['time_stamp'][0] * self.fps)

            for timestamp, ts_class in zip(row_labels['time_stamp'][1:] + [int(np.ceil((frames + spacing) / self.fps))], row_labels['time_stamp_class']):
                timestamp_frame = int(timestamp * self.fps)
                timestamp_data_count = (timestamp_frame - spacing - frame_start) // self.frames
                for i in range(timestamp_data_count):
                    frame_end = min(frame_start + self.frames * (i + 1), frames)
                    self.lookup_per_class[ts_class].append(len(self.lookup))
                    self.lookup.append((video_name, frame_end - self.frames, frame_end))
                    self.class_indices.append(ts_class)
                data_count += timestamp_data_count
                frame_start = timestamp_frame + spacing

            row_labels['data_count'] = data_count
            
        np.random.seed(seed)
        min_class_samples = min([len(lookup) for lookup in self.lookup_per_class])
        lookup_discarded_indices = []
        for cl in range(self.classes):
            class_samples = len(self.lookup_per_class[cl])
            discarded_indices = np.sort(np.random.permutation(class_samples)[:class_samples - min_class_samples])[::-1]
            for discarded_index in discarded_indices:
                lookup_discarded_indices.append(self.lookup_per_class[cl][discarded_index])
                del self.lookup_per_class[cl][discarded_index]
        lookup_discarded_indices = sorted(lookup_discarded_indices, reverse=True)
        for lookup_discarded_index in lookup_discarded_indices:
            del self.lookup[lookup_discarded_index]
            del self.class_indices[lookup_discarded_index]

        print([len(lookup) for lookup in self.lookup_per_class])

    def __len__(self):
        return len(self.lookup)
  
    def __getitem__(self, idx):
        video_name, frame_start, frame_end = self.lookup[idx]
        images = torch.stack([self.transform(Image.open(os.path.join(self.root_path, video_name, f'{str(frame + 1).zfill(5)}.jpg'))) for frame in range(frame_start, frame_end)], dim=1)
        label = self.class_indices[idx]
        return images, label

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = ClassDataset('./video_labeling.xlsx', 'C:/news_frame/', spacing=8, seed=420, image_size=(200, 200))
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    print(len(dataset))

    for data, label in dataloader:
        for dat, lbl in zip(data, label):
            print(lbl)
            fig, axes = plt.subplots(1, dataset.frames, figsize=(10, 1))
            for ax, img in zip(axes, dat.permute(1, 2, 3, 0)):
                ax.imshow(img)
                ax.axis('off')
            '''fig, axes = plt.subplots(4, 1, figsize=(10, 3))
            for ax, img in zip(axes, lbl):
                ax.imshow(img.view(1, 24), vmin=0, vmax=1)
                ax.axis('off')'''
        break
    plt.show()