# import os
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset
# import torch
# from sklearn.preprocessing import OneHotEncoder
#
# class OCTData(Dataset):
#     def __init__(self, data_frame, root_dir, encoder, transform=None):
#         self.original_data_frame = data_frame
#         self.data_frame = data_frame.drop(columns=['id_patient', 'split_type', 'case'])
#         self.root_dir = root_dir
#         self.transform = transform
#         self.encoder = encoder
#
#         # 指定不需要的列
#         self.exclude_columns = ['label', 'image', 'LOCALIZER']
#         # 提取所有列
#         self.patient_info_columns = [col for col in self.data_frame.columns if col not in self.exclude_columns]
#
#         # 打印列名进行检查
#         print("所有列名:", self.data_frame.columns)
#         print("患者信息列:", self.patient_info_columns)
#
#         # 独热编码患者信息列
#         patient_info = self.data_frame[self.patient_info_columns]
#         self.encoded_patient_info = self.encoder.transform(patient_info)
#         self.patient_info_columns = [f"feature_{i}" for i in range(self.encoded_patient_info.shape[1])]
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.original_data_frame.iloc[idx]['image'])
#         loc_img_name = os.path.join(self.root_dir, self.original_data_frame.iloc[idx]['LOCALIZER'])
#
#         image = Image.open(img_name).convert('L')  # Convert to grayscale
#         loc_image = Image.open(loc_img_name).convert('L')  # Convert to grayscale
#
#         if self.transform:
#             image = self.transform(image)
#             loc_image = self.transform(loc_image)
#
#         # Extract encoded patient information
#         patient_info = self.encoded_patient_info[idx].astype('float32')
#         patient_info = torch.tensor(patient_info)
#
#         # # 打印患者信息的形状进行调试
#         # print(f"患者信息的形状: {patient_info.shape}")
#
#         label = int(self.original_data_frame.iloc[idx]['label'])
#
#         sample = {
#             'image': image,
#             'loc_image': loc_image,
#             'patient_info': patient_info,
#             'label': label
#         }
#
#         return sample
#
# class TestData(Dataset):
#     def __init__(self, data_frame, root_dir, encoder, transform=None):
#         self.original_data_frame = data_frame
#         self.data_frame = data_frame.drop(columns=['id_patient', 'split_type', 'case'])
#         self.root_dir = root_dir
#         self.transform = transform
#         self.encoder = encoder
#
#         # 指定不需要的列
#         self.exclude_columns = ['label', 'image', 'LOCALIZER']
#         # 提取所有列
#         self.patient_info_columns = [col for col in self.data_frame.columns if col not in self.exclude_columns]
#
#         # # 打印列名进行检查
#         # print("所有列名:", self.data_frame.columns)
#         # print("患者信息列:", self.patient_info_columns)
#
#         # 独热编码患者信息列
#         patient_info = self.data_frame[self.patient_info_columns]
#         self.encoded_patient_info = self.encoder.transform(patient_info)
#         self.patient_info_columns = [f"feature_{i}" for i in range(self.encoded_patient_info.shape[1])]
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.original_data_frame.iloc[idx]['image'])
#         loc_img_name = os.path.join(self.root_dir, self.original_data_frame.iloc[idx]['LOCALIZER'])
#
#         image = Image.open(img_name).convert('L')  # Convert to grayscale
#         loc_image = Image.open(loc_img_name).convert('L')  # Convert to grayscale
#
#         if self.transform:
#             image = self.transform(image)
#             loc_image = self.transform(loc_image)
#
#         # Extract encoded patient information
#         patient_info = self.encoded_patient_info[idx].astype('float32')
#         patient_info = torch.tensor(patient_info)
#
#         # # 打印患者信息的形状进行调试
#         # print(f"患者信息的形状: {patient_info.shape}")
#
#         case_id = self.original_data_frame.iloc[idx]['case']
#
#         sample = {
#             'image': image,
#             'loc_image': loc_image,
#             'patient_info': patient_info,
#             'case_id': case_id
#         }
#
#         return sample


##v1

import os
import pandas as pd
from PIL import Image
# from torch.utils.data import Dataset
# import torch
#
# class OCTData(Dataset):
#     def __init__(self, data_frame, root_dir, transform=None):
#         self.data_frame = data_frame
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image'])
#         loc_img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['LOCALIZER'])
#
#         image = Image.open(img_name).convert('L')  # Convert to grayscale
#         loc_image = Image.open(loc_img_name).convert('L')  # Convert to grayscale
#
#         if self.transform:
#             image = self.transform(image)
#             loc_image = self.transform(loc_image)
#
#         label = int(self.data_frame.iloc[idx]['label'])
#
#         sample = {
#             'image': image,
#             'loc_image': loc_image,
#             'label': label
#         }
#
#         return sample
#
# class TestData(Dataset):
#     def __init__(self, data_frame, root_dir, transform=None):
#         self.data_frame = data_frame
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image'])
#         loc_img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['LOCALIZER'])
#
#         image = Image.open(img_name).convert('L')  # Convert to grayscale
#         loc_image = Image.open(loc_img_name).convert('L')  # Convert to grayscale
#
#         if self.transform:
#             image = self.transform(image)
#             loc_image = self.transform(loc_image)
#
#         case_id = self.data_frame.iloc[idx]['case']
#
#         sample = {
#             'image': image,
#             'loc_image': loc_image,
#             'case_id': case_id
#         }
#
#         return sample


##v2

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class OCTData(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image'])
        # loc_img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['LOCALIZER'])

        image = Image.open(img_name).convert('L')  # Convert to grayscale
        # loc_image = Image.open(loc_img_name).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
            # loc_image = self.transform(loc_image)

        label = int(self.data_frame.iloc[idx]['label'])

        sample = {
            'image': image,
            # 'loc_image': loc_image,
            'label': label
        }

        return sample


class TestData(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['image'])
        # loc_img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['LOCALIZER'])

        image = Image.open(img_name).convert('L')  # Convert to grayscale
        # loc_image = Image.open(loc_img_name).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
            # loc_image = self.transform(loc_image)

        case_id = self.data_frame.iloc[idx]['case']

        sample = {
            'image': image,
            # 'loc_image': loc_image,
            'case_id': case_id
        }

        return sample


class MILDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

        # Group the data by LOCALIZER
        self.grouped = self.data_frame.groupby('LOCALIZER')
        self.localizers = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.localizers)

    def __getitem__(self, idx):
        localizer = self.localizers[idx]
        instances = self.grouped.get_group(localizer)

        images = []
        for _, row in instances.iterrows():
            img_path = os.path.join(self.root_dir, row['image'])
            image = Image.open(img_path).convert('L')  # Convert to grayscale

            if self.transform:
                image = self.transform(image)

            images.append(image)

        # Stack images into a single tensor
        images_tensor = torch.stack(images)

        # Get label (assuming all instances in a bag have the same label)
        label = torch.tensor(instances['label'].iloc[0], dtype=torch.long)

        return {'image': images_tensor, 'label': label}


# class LocalizerGroupedDataset(Dataset):
#     def __init__(self, data_frame, root_dir, transform=None, feature_extractor=None):
#         self.data_frame = data_frame
#         self.root_dir = root_dir
#         self.transform = transform
#         self.feature_extractor = feature_extractor
#         self.feature_extractor.eval()  # Set to evaluation mode
#
#         # Group the data by localizer
#         self.grouped = self.data_frame.groupby('LOCALIZER')
#
#     def __len__(self):
#         return len(self.grouped)
#
#     def __getitem__(self, idx):
#         # if idx >= len(self.grouped):
#         #     raise IndexError(f"Index {idx} out of range for grouped data (max index: {len(self.grouped) - 1}).")
#         #
#         #     # 添加调试信息
#         # print(f"Fetching data for idx: {idx}, max index: {len(self.grouped) - 1}")
#
#         # Get the localizer key
#         localizer_key = list(self.grouped.groups.keys())[idx]
#
#         # Get all rows corresponding to this localizer
#         localizer_data = self.grouped.get_group(localizer_key)
#
#         features = []
#         for _, row in localizer_data.iterrows():
#             img_name = os.path.join(self.root_dir, row['image'])
#             image = Image.open(img_name).convert('L')  # Convert to grayscale
#
#             if self.transform:
#                 image = self.transform(image)
#
#             with torch.no_grad():
#                 feature = self.feature_extractor(image.unsqueeze(0))  # Extract features
#
#             features.append(feature.squeeze(0))  # Add to the list of features
#
#         # Stack features to create a tensor of size [N, Ndim] where N is the number of images and Ndim is the feature dimension
#         features = torch.stack(features)
#
#         # All images in the group should have the same label
#         label = int(localizer_data['label'].iloc[0])
#
#         return {'features': features, 'label': label}


class LocalizerGroupedDataset(Dataset):
    def __init__(self, features_dict, labels_dict):
        self.features_dict = features_dict
        self.labels_dict = labels_dict
        self.localizers = list(self.features_dict.keys())

    def __len__(self):
        return len(self.localizers)

    def __getitem__(self, idx):
        localizer_key = self.localizers[idx]
        features = torch.stack(self.features_dict[localizer_key])
        label = self.labels_dict[localizer_key]  # Get the label for the localizer
        return {'features': features, 'label': label}



## DTFD-MIL

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

import random

# class MILDataset(Dataset):
#     def __init__(self, data_frame, root_dir, transform=None, max_instances=64):
#         self.data_frame = data_frame
#         self.root_dir = root_dir
#         self.transform = transform
#         self.max_instances = max_instances  # 限制最大实例数量
#         self.grouped = self.data_frame.groupby('LOCALIZER')
#
#     def __len__(self):
#         return len(self.grouped)
#
#     def __getitem__(self, idx):
#         localizer = list(self.grouped.groups.keys())[idx]
#         instances = self.grouped.get_group(localizer)
#
#         if len(instances) > self.max_instances:
#             instances = instances.sample(self.max_instances)  # 随机选择 max_instances 数量的图像
#
#         images = []
#         for _, row in instances.iterrows():
#             img_name = os.path.join(self.root_dir, row['image'])
#             image = Image.open(img_name).convert('L')  # Convert to grayscale
#             if self.transform:
#                 image = self.transform(image)
#             images.append(image)
#
#         images = torch.stack(images)
#         label = instances['label'].iloc[0]
#
#         return {'image': images, 'label': label}


# class MILDataset(Dataset):
#     def __init__(self, data_frame, root_dir, transform=None):
#         self.data_frame = data_frame
#         self.root_dir = root_dir
#         self.transform = transform
#         self.grouped = self.data_frame.groupby('LOCALIZER')
#         self.localizers = list(self.grouped.groups.keys())  # 获取localizer的列表
#
#         # 调试信息
#         print(f"Total localizers: {len(self.localizers)}")
#         print(f"Total groups: {len(self.grouped)}")
#
#     def __len__(self):
#         return len(self.localizers)
#
#     def __getitem__(self, idx):
#         if idx >= len(self.localizers):
#             print(f"Index {idx} is out of range for localizers (total localizers: {len(self.localizers)})")
#             raise IndexError("Index out of range for localizers")
#
#         # 调试信息
#         print(f"Fetching data for idx: {idx}, localizer: {self.localizers[idx]}")
#
#         localizer = self.localizers[idx]
#         instances = self.grouped.get_group(localizer)
#
#         # 调试信息
#         print(f"Number of instances for localizer {localizer}: {len(instances)}")
#
#         images = []
#         for _, row in instances.iterrows():
#             img_name = os.path.join(self.root_dir, row['image'])
#             image = Image.open(img_name).convert('L')  # 转换为灰度图
#             if self.transform:
#                 image = self.transform(image)
#             images.append(image)
#
#         images = torch.stack(images)
#         label = instances['label'].iloc[0]
#
#         return images, label
