# import torch
#
#
# class Config:
#     def __init__(self):
#
#         self.seed = 25
#         self.split = 0.8
#
#         # 模型参数
#         self.model_name = 'convnextv2_large'
#         self.pretrained = True
#         self.input_channels = 1
#         self.num_classes = 4
#
#
#         # 训练参数
#
#         self.num_epochs = 50
#         self.lr = 0.0001  # 0.0005
#         self.step_size = 2
#         self.gamma = 0.8
#         self.batch_size = 16
#
#         # 数据增强和预处理
#         self.image_size = (224, 224)
#         self.normalize_mean = [0.5]
#         self.normalize_std = [0.5]
#
#         # 其他
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.output_file = f'data/submission/{self.model_name}_split_0.8_bs_16_lr_{self.lr}_seed_{self.seed}_test.csv'
#         self.train_csv_file = 'data/data_1/df_task1_train_challenge.csv'
#         self.train_root_dir = 'data/data_1/train'
#         self.test_csv_file = 'data/data_1/df_task1_val_challenge.csv'
#         self.test_root_dir = 'data/data_1/val'
#
#
# # 实例化配置类
# config = Config()


import torch


class Config:
    def __init__(self):
        self.seed = 1
        self.fold = 5
        self.split = 0.8

        # 模型参数
        self.model_name = 'abmil_convnext2_avg_weight_clamsb_test_seed1'
        # self.model_name = 'resnet50'
        self.pretrained = True
        self.input_channels = 2  # 因为现在只用两张图片
        self.num_classes = 3

        # 训练参数
        self.num_epochs = 301
        self.lr = 0.0002
        self.step_size = 2
        self.gamma = 0.8
        self.batch_size = 1
        self.max_instances = 64

        self.data_number = f'{self.seed}_{self.fold}'
        # 数据增强和预处理
        self.image_size = (224, 224)
        self.normalize_mean = [0.5]
        self.normalize_std = [0.5]

        # 其他
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_file = f'submission/{self.model_name}_split_0.8_bs_16_lr_{self.lr}_seed_{self.seed}_final.csv'
        self.train_csv_file = 'data_2/df_task2_train_challenge.csv'
        self.train_root_dir = 'data_2/data_task2/train'
        self.test_csv_file = 'data_2/df_task2_val_challenge.csv'
        self.test_root_dir = 'data_2/data_task2/val'
        self.final_csv_file = 'data_2/combined_dataset.csv'
        self.final_root_dir = 'data_2/data_task2/final'
        self.final_train_csv_file = f'data_2/train_fold_{self.data_number}.csv'
        self.final_val_csv_file = f'data_2/val_fold_{self.data_number}.csv'

        # self.feature_train_save_path = f'model/{self.model_name}/train_feature_{self.data_number}.pth'
        # self.feature_val_save_path = f'model/{self.model_name}/val_feature_{self.data_number}.pth'
        self.feature_train_save_path = f'feature/convnext2/train_feature_{self.data_number}_new.pth'
        self.feature_val_save_path = f'feature/convnext2/val_feature_{self.data_number}_new.pth'
# 实例化配置类
config = Config()
