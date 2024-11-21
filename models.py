import torch
import torch.nn as nn
import timm


## clam


# class CLAMResNet50(nn.Module):
#     def __init__(self, num_classes=3, feature_size=1024):
#         super(CLAMResNet50, self).__init__()
#         # 使用 ResNet50 from timm
#         self.feature_extractor = timm.create_model('resnet50', pretrained=True, num_classes=0)
#         self.feature_dim = feature_size
#
#         # 适应单通道输入（灰度图像）
#         self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.feature_extractor.fc = nn.Identity()  # Remove the fully connected layer
#
#         # Attention-based pooling
#         self.attention = nn.Sequential(
#             nn.Linear(self.feature_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1)
#         )
#
#         # Classifier
#         self.classifier = nn.Linear(self.feature_dim, num_classes)
#
#     def forward(self, x):
#         # x 是一组实例的批次
#         batch_size, num_instances, channels, height, width = x.size()
#
#         # 将批次的实例展平为单个实例批次
#         x = x.view(-1, channels, height, width)
#
#         # 对每个实例提取特征
#         features = self.feature_extractor(x)
#
#         # 重新调整为一组实例的批次
#         features = features.view(batch_size, num_instances, -1)
#
#         # 基于注意力的聚合
#         attention_scores = self.attention(features)
#         attention_weights = torch.softmax(attention_scores, dim=1)
#         aggregated_features = torch.sum(attention_weights * features, dim=1)
#
#         # 分类
#         logits = self.classifier(aggregated_features)
#         return logits


class CLAMResNet50(nn.Module):
    def __init__(self, num_classes=3, instance_classes=512, k_sample=8, dropout=True, dropout_rate=0.25):
        super(CLAMResNet50, self).__init__()
        self.k_sample = k_sample  # Top-k instances to sample
        self.instance_classes = instance_classes
        # 使用 ResNet-50 作为特征提取器
        self.feature_extractor = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature_dim = self.feature_extractor.feature_info[-1]['num_chs']  # Get feature dimensions
        # 替换ResNet-50的全连接层
        self.feature_extractor.fc = nn.Identity()

        # 线性分类层（实例级别）
        self.instance_classifiers = nn.Linear(self.feature_dim, self.instance_classes)

        # 注意力池化
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.feature_dim, self.instance_classes),
            nn.Tanh(),
            nn.Linear(self.instance_classes, 1)
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate) if dropout else nn.Identity()

        # 聚合后的分类层（bag级别）
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x 是一组实例的批次
        batch_size, num_instances, channels, height, width = x.size()

        # 将批次的实例展平为单个实例批次
        x = x.view(-1, channels, height, width)

        # 对每个实例提取特征
        features = self.feature_extractor(x)

        # 重新调整为一组实例的批次
        features = features.view(batch_size, num_instances, -1)

        # 实例级别分类
        instance_logits = self.instance_classifiers(features)

        # 注意力机制
        attention_scores = self.attention_pooling(features)
        attention_weights = torch.softmax(attention_scores, dim=1)

        k = min(self.k_sample, num_instances)

        # 选取top-k实例进行聚合
        top_k_weights, top_k_indices = torch.topk(attention_weights, k, dim=1)
        top_k_features = torch.gather(features, 1, top_k_indices.expand(-1, -1, features.size(-1)))

        # 聚合特征
        aggregated_features = torch.sum(top_k_weights * top_k_features, dim=1)
        aggregated_features = self.dropout(aggregated_features)

        # Bag级别分类
        logits = self.classifier(aggregated_features)
        return logits, instance_logits, attention_scores


### DTFD MIL
class ConvNeXtMILModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNeXtMILModel, self).__init__()
        # 使用 ConvNeXt V2 Large from timm
        self.feature_extractor = timm.create_model('convnextv2_base', pretrained=True, num_classes=0)
        self.feature_dim = self.feature_extractor.num_features

        # 修改输入层，使其接受单通道输入（灰度图像）
        self.feature_extractor.stem[0] = nn.Conv2d(1, self.feature_extractor.stem[0].out_channels, kernel_size=4,
                                                   stride=4, bias=False)

        # 聚合层 (attention-based pooling)
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 分类器
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x 是一组实例的批次
        batch_size, num_instances, channels, height, width = x.size()

        # 将批次的实例展平为单个实例批次
        x = x.view(-1, channels, height, width)  # 这里 channels 应该是 1

        # 对每个实例提取特征
        features = self.feature_extractor(x)

        # 重新调整为一组实例的批次
        features = features.view(batch_size, num_instances, -1)

        # 基于注意力的聚合
        attention_scores = self.attention(features)
        attention_weights = torch.softmax(attention_scores, dim=1)
        aggregated_features = torch.sum(attention_weights * features, dim=1)

        # 分类
        logits = self.classifier(aggregated_features)
        return logits


# #binary
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         self.model = timm.create_model('resnet50', pretrained=True)
#         self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
#         num_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_features, 2)
#
#     def forward(self, image):
#         return self.model(image)


# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         self.model = timm.create_model('resnet50', pretrained=True)
#         self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
#         num_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_features, 3)
#
#     def forward(self, image):
#         return self.model(image)

import torch


# import torch.nn as nn
# import timm
#
###v2
class OCTClassifier(nn.Module):
    def __init__(self):
        super(OCTClassifier, self).__init__()
        # 使用 timm 导入 ConvNeXt V2-L 模型
        self.model = timm.create_model('convnextv2_large', pretrained=True)

        # 修改第一层以接受多通道输入
        self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)

        # 替换最后一层分类器
        num_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(num_features, 3)  # Assuming 3 classes for classification

    def forward(self, image):
        # Concatenate the images along the channel dimension
        # x = torch.cat((image, loc_image), dim=1)
        features = self.model(image)
        return features


# import torch
# import torch.nn as nn
# import timm
#
# ###v1
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受多通道输入
#         self.model.stem[0] = nn.Conv2d(2, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 3)  # Assuming 3 classes for classification
#
#     def forward(self, image, loc_image):
#         # Concatenate the images along the channel dimension
#         x = torch.cat((image, loc_image), dim=1)
#         features = self.model(x)
#         return features


# ###v0
# class OCTClassifier(nn.Module):
#     def __init__(self, num_patient_info_features):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受多通道输入
#         self.model.stem[0] = nn.Conv2d(2, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Identity()  # 保留特征提取部分，移除原始分类层
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Sequential(
#             nn.Linear(num_patient_info_features, 128),
#             nn.ReLU()
#         )
#
#         # 定义最后的分类器
#         self.fc_combined = nn.Sequential(
#             nn.Linear(num_features + 128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)  # Assuming 3 classes for classification
#         )
#
#     def forward(self, image, loc_image, patient_info):
#         # Concatenate the images along the channel dimension
#         x = torch.cat((image, loc_image), dim=1)
#         features = self.model(x)
#         #
#         # # 打印特征张量的形状进行调试
#         # print(f"图像特征的形状: {features.shape}")
#         # print(f"患者信息的形状: {patient_info.shape}")
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#
#         # # 打印处理后的患者信息特征的形状
#         # print(f"处理后的患者信息特征的形状: {patient_info.shape}")
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         #
#         # # 打印拼接后的特征的形状
#         # print(f"拼接后的特征的形状: {combined.shape}")
#
#         combined = self.fc_combined(combined)
#
#         return combined


# import torch
# import torch.nn as nn
# import timm
#
# class OCTClassifier(nn.Module):
#     def __init__(self, num_patient_info_features):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受多通道输入
#         self.model.stem[0] = nn.Conv2d(2, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 128)  # 中间层输出128维特征
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Linear(num_patient_info_features, 128)
#         self.fc_combined = nn.Linear(256, 3)  # Assuming 3 classes for classification
#
#     def forward(self, image, loc_image, patient_info):
#         # Concatenate the images along the channel dimension
#         x = torch.cat((image, loc_image), dim=1)
#         features = self.model(x)
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#         patient_info = torch.relu(patient_info)
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         combined = self.fc_combined(combined)
#
#         return combined


# import torch
# import torch.nn as nn
# import timm
#
#
# class OCTClassifier(nn.Module):
#     def __init__(self, num_patient_info_features):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受多通道输入
#         self.model.stem[0] = nn.Conv2d(2, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Identity()  # 保留特征提取部分，移除原始分类层
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Sequential(
#             nn.Linear(num_patient_info_features, 128),
#             nn.ReLU()
#         )
#
#         # 定义最后的分类器
#         self.fc_combined = nn.Sequential(
#             nn.Linear(num_features + 128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)  # Assuming 3 classes for classification
#         )
#
#     def forward(self, image, loc_image, patient_info):
#         # Concatenate the images along the channel dimension
#         x = torch.cat((image, loc_image), dim=1)
#         features = self.model(x)
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         combined = self.fc_combined(combined)
#
#         return combined


#
# import torch
# import torchvision
# from torch import nn
# from torchvision import models
# from efficientnet_pytorch import EfficientNet
# from torchvision.models import EfficientNet_V2_L_Weights
# from torchvision.models import vit_b_16, ViT_B_16_Weights
# import timm


# resnet50
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         self.model = models.resnet50(pretrained=True)
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一层以接受单通道输入
#         self.model.fc = nn.Identity()  # 移除最后一层全连接层，因为我们要提取特征
#         self.fc = nn.Linear(2048, 4)  # 添加自己的分类层，根据ResNet50的输出特征维度
#
#     def forward(self, image_ti, image_ti_1):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         # 提取特征
#         x = self.model(image_diff)
#         # 分类
#         x = self.fc(x)
#         return x


##### efficient v2


# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # 使用 EfficientNet V2
#         self.model = torchvision.models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
#
#         # 修改第一层以接受单通道输入
#         self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.classifier[1].in_features
#         self.model.classifier[1] = nn.Linear(num_features, 4)  # Replace with your desired output size
#
#
#     def forward(self, image_ti, image_ti_1):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         # 提取特征并分类
#         x = self.model(image_diff)
#         return x
#
#


### VIT

# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # Load Vision Transformer (ViT) model
#         self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#
#         # Modify the first convolutional layer to accept 1 channel instead of 3
#         self.model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
#
#         # Modify the classifier layer for the desired output size
#         num_features = self.model.heads.head.in_features
#         self.model.heads.head = nn.Linear(num_features, 4)
#
#     def forward(self, image_ti, image_ti_1):
#         # Calculate absolute difference between images
#         image_diff = torch.abs(image_ti - image_ti_1)
#
#         # Resize to match ViT input size if necessary (224x224)
#         image_diff = nn.functional.interpolate(image_diff, size=(224, 224), mode='bilinear', align_corners=False)
#
#         # Extract features and classify
#         x = self.model(image_diff)
#         return x

### convnext2_large


# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # Load the ConvNeXt V2 model from timm
#         self.model = timm.create_model('convnextv2_large', pretrained=True, in_chans=1, num_classes=4)
#
#     def forward(self, image_ti, image_ti_1):
#         # Calculate the absolute difference between the two images
#         image_diff = torch.abs(image_ti - image_ti_1)
#         # Extract features and classify
#         x = self.model(image_diff)
#         return x
#
# import torch
# import torch.nn as nn
# import timm


### convnext2_large_test
# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受单通道输入
#         self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 128)  # 中间层输出128维特征
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Linear(len(self.patient_info), 128)
#         self.fc_combined = nn.Linear(128 + 128, 4)  # Assuming 4 classes for classification
#
#     def forward(self, image_ti, image_ti_1, loc_image_ti, loc_image_ti_1, patient_info):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         loc_image_diff = torch.abs(loc_image_ti - loc_image_ti_1)
#
#         # Concatenate the image differences and pass through the model
#         x = torch.cat((image_diff, loc_image_diff), dim=1)  # Concatenate along the channel dimension
#         features = self.model(x)
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#         patient_info = torch.relu(patient_info)
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         combined = self.fc_combined(combined)
#
#         return combined

# class OCTClassifier(nn.Module):
#     def __init__(self, num_patient_info_features):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受单通道输入
#         self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 128)  # 中间层输出128维特征
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Linear(num_patient_info_features, 128)
#         self.fc_combined = nn.Linear(256, 4)  # Assuming 4 classes for classification
#
#     def forward(self, image_ti, image_ti_1, loc_image_ti, loc_image_ti_1, patient_info):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         loc_image_diff = torch.abs(loc_image_ti - loc_image_ti_1)
#
#         # Concatenate the image differences and pass through the model
#         x = torch.cat((image_diff, loc_image_diff), dim=1)  # Concatenate along the channel dimension
#         features = self.model(x)
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#         patient_info = torch.relu(patient_info)
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         combined = self.fc_combined(combined)
#
#         return combined
#
# import torch
# import torch.nn as nn
# import timm
#
# class OCTClassifier(nn.Module):
#     def __init__(self, num_patient_info_features):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受多通道输入
#         self.model.stem[0] = nn.Conv2d(2, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 128)  # 中间层输出128维特征
#
#         # 定义用于融合病人信息的全连接层
#         self.fc_patient = nn.Linear(num_patient_info_features, 128)
#         self.fc_combined = nn.Linear(256, 4)  # Assuming 4 classes for classification
#
#     def forward(self, image_ti, image_ti_1, loc_image_ti, loc_image_ti_1, patient_info):
#         # 计算图像的绝对差异
#         image_diff = torch.abs(image_ti - image_ti_1)
#         loc_image_diff = torch.abs(loc_image_ti - loc_image_ti_1)
#
#         # Concatenate the image differences and pass through the model
#         x = torch.cat((image_diff, loc_image_diff), dim=1)  # Concatenate along the channel dimension
#         features = self.model(x)
#
#         # 处理病人信息
#         patient_info = self.fc_patient(patient_info)
#         patient_info = torch.relu(patient_info)
#
#         # Concatenate the extracted features with patient information
#         combined = torch.cat((features, patient_info), dim=1)
#         combined = self.fc_combined(combined)
#
#         return combined


# def resnet50_feature_extractor(pretrained=True, output_dim=1024):
#     # Load a ResNet-50 model pre-trained on ImageNet
#     model = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
#
#     # Modify the first convolutional layer to accept single-channel (grayscale) input
#     model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#     # Replace the fully connected layer with an Identity layer
#     model.fc = nn.Identity()
#
#     # Optionally, add a linear layer to reduce the feature dimension to the desired output dimension
#     if output_dim and output_dim != model.feature_info[-1]['num_chs']:
#         model = nn.Sequential(
#             model,
#             nn.Linear(model.feature_info[-1]['num_chs'], output_dim)
#         )
#
#     return model

def resnet50_feature_extractor(pretrained=True, output_dim=2048):
    # Load a ResNet-50 model pre-trained on ImageNet
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)

    # Modify the first convolutional layer to accept single-channel (grayscale) input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the fully connected layer with an Identity layer
    model.fc = nn.Identity()

    return model


# class OCTClassifier(nn.Module):
#     def __init__(self):
#         super(OCTClassifier, self).__init__()
#         # 使用 timm 导入 ConvNeXt V2-L 模型
#         self.model = timm.create_model('convnextv2_large', pretrained=True)
#
#         # 修改第一层以接受多通道输入
#         self.model.stem[0] = nn.Conv2d(1, self.model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)
#
#         # 替换最后一层分类器
#         num_features = self.model.head.fc.in_features
#         self.model.head.fc = nn.Linear(num_features, 3)  # Assuming 3 classes for classification
#
#     def forward(self, image):
#         # Concatenate the images along the channel dimension
#         # x = torch.cat((image, loc_image), dim=1)
#         features = self.model(image)
#         return features
def convnext2_feature_extractor(pretrained=True, output_dim=2048):
    model = timm.create_model('convnextv2_large', pretrained=True)

    # 修改第一层以接受多通道输入
    model.stem[0] = nn.Conv2d(1, model.stem[0].out_channels, kernel_size=4, stride=4, bias=False)

    model.fc = nn.Identity()

    return model
