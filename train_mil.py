import torch
from torch.optim import lr_scheduler
import torch.cuda.amp as amp
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import os
from config import config


def unfreeze_layers(model, unfreeze_from):
    """
    Unfreeze layers starting from `unfreeze_from`.
    """
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific layers from `unfreeze_from` onwards
    for name, param in model.named_parameters():
        if unfreeze_from in name:
            param.requires_grad = True


def calculate_metrics(true_labels, predictions, num_classes=3):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    spearman_corr, _ = spearmanr(true_labels, predictions)
    qwk = cohen_kappa_score(true_labels, predictions, weights='quadratic')

    cm = confusion_matrix(true_labels, predictions)
    specificity_per_class = []
    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class)

    mean_metrics = (f1 + spearman_corr + specificity + qwk) / 4

    return accuracy, f1, spearman_corr, specificity, qwk, mean_metrics, cm


# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     scaler = amp.GradScaler()
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)
#
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0
#         all_labels = []
#         all_preds = []
#
#         # Training Loop
#         train_loader_tqdm = tqdm(train_loader, desc="Training")
#
#         for samples in train_loader_tqdm:
#             images = samples['image']  # List of images (batch of images from the same localizer)
#             labels = samples['label'].to(device, dtype=torch.long)
#
#             # Stack images into a tensor
#             inputs_image = torch.stack(images).to(device, dtype=torch.float)
#
#             optimizer.zero_grad()
#
#             with amp.autocast():
#                 outputs = model(inputs_image)
#                 loss = criterion(outputs, labels)
#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             running_loss += loss.item() * inputs_image.size(0)
#             _, preds = torch.max(outputs, 1)
#             running_corrects += torch.sum(preds == labels.data)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#
#         scheduler.step()
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / len(train_loader.dataset)
#
#         print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#         train_accuracy, train_f1, train_spearman, train_specificity, train_qwk, train_mean_metrics, train_cm = calculate_metrics(
#             all_labels, all_preds, num_classes=num_classes)
#         print(
#             f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} QWK: {train_qwk:.4f} Mean Metrics: {train_mean_metrics:.4f}')
#
#         # Validation Loop
#         model.eval()
#         val_loss = 0.0
#         val_corrects = 0
#         val_labels = []
#         val_preds = []
#
#         val_loader_tqdm = tqdm(val_loader, desc="Validation")
#
#         for samples in val_loader_tqdm:
#             images = samples['image']  # List of images (batch of images from the same localizer)
#             labels = samples['label'].to(device, dtype=torch.long)
#
#             # Stack images into a tensor
#             inputs_image = torch.stack(images).to(device, dtype=torch.float)
#
#             with torch.no_grad():
#                 with amp.autocast():
#                     outputs = model(inputs_image)
#                     loss = criterion(outputs, labels)
#
#             val_loss += loss.item() * inputs_image.size(0)
#             _, preds = torch.max(outputs, 1)
#             val_corrects += torch.sum(preds == labels.data)
#             val_labels.extend(labels.cpu().numpy())
#             val_preds.extend(preds.cpu().numpy())
#
#         val_loss = val_loss / len(val_loader.dataset)
#         val_acc = val_corrects.double() / len(val_loader.dataset)
#
#         print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
#
#         val_accuracy, val_f1, val_spearman, val_specificity, val_qwk, val_mean_metrics, val_cm = calculate_metrics(
#             val_labels, val_preds, num_classes=num_classes)
#         print(
#             f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} QWK: {val_qwk:.4f} Mean Metrics: {val_mean_metrics:.4f}')
#         torch.save(model.state_dict(),
#                    f'model/{config.model_name}/epoch_{epoch}_split_{config.split}_bs_{config.batch_size}_lr_{config.lr}_{config.data_number}')
#
#     return model


# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, num_classes=3):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     scaler = amp.GradScaler()
#
#     # Learning rate scheduler
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)
#
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0
#         all_labels = []
#         all_preds = []
#
#         train_loader_tqdm = tqdm(train_loader, desc="Training")
#
#         for samples in train_loader_tqdm:
#             images = samples['image']  # This is a list of images (batch of images from the same localizer)
#             labels = samples['label'].to(device, dtype=torch.long)
#             if isinstance(images, torch.Tensor):
#                 images = [images]
#
#             #     # Stack images into a tensor
#             inputs_image = torch.stack(images).to(device, dtype=torch.float)
#             # Stack images into a tensor
#             # Remove extra dimensions
#             # Ensure the tensor has the correct shape: [batch_size, num_instances, channels, height, width]
#             # We will enforce that the shape is 5-dimensional
#             if inputs_image.dim() == 4:  # [num_instances, channels, height, width]
#                 inputs_image = inputs_image.unsqueeze(0)  # Add batch dimension
#             elif inputs_image.dim() == 3:  # [channels, height, width] (single image case)
#                 inputs_image = inputs_image.unsqueeze(0).unsqueeze(0)  # Add batch and num_instances dimensions
#
#             # 确保转换后的形状是 [batch_size, num_instances, channels, height, width]
#             inputs_image = inputs_image.to(device, dtype=torch.float)
#
#             optimizer.zero_grad()
#
#             with amp.autocast():
#                 outputs = model(inputs_image)
#                 loss = criterion(outputs, labels)
#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             running_loss += loss.item() * inputs_image.size(0)
#             _, preds = torch.max(outputs, 1)
#             running_corrects += torch.sum(preds == labels.data)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#
#         scheduler.step()
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / len(train_loader.dataset)
#
#         print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#         train_accuracy, train_f1, train_spearman, train_specificity, train_qwk, train_mean_metrics, train_cm = calculate_metrics(
#             all_labels, all_preds, num_classes=num_classes)
#         print(
#             f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} QWK: {train_qwk:.4f} Mean Metrics: {train_mean_metrics:.4f}')
#         # print(f'Train Confusion Matrix:\n{train_cm}')
#
#         # Save training confusion matrix as CSV
#         train_cm_df = pd.DataFrame(train_cm, index=[f"True_{i}" for i in range(num_classes)], columns=[f"Pred_{i}" for i in range(num_classes)])
#         train_cm_path = os.path.join(f'model/{config.model_name}/', f'mil_train_cm_epoch_{epoch}_lr_{config.lr}_{config.data_number}.csv')
#         train_cm_df.to_csv(train_cm_path, index=True)
#
#         model.eval()
#         val_loss = 0.0
#         val_corrects = 0
#         val_labels = []
#         val_preds = []
#
#         val_loader_tqdm = tqdm(val_loader, desc="Validation")
#
#         for samples in val_loader_tqdm:
#             images = samples['image']
#             labels = samples['label'].to(device, dtype=torch.long)
#
#             if isinstance(images, torch.Tensor):
#                 images = [images]
#
#             #     # Stack images into a tensor
#             # inputs_image = torch.stack(images).to(device, dtype=torch.float)
#             inputs_image = torch.stack(images).to(device, dtype=torch.float)
#
#             # Remove extra dimensions
#             inputs_image = inputs_image.squeeze(1).squeeze(2)  # 将 (1, 19, 1, 224, 224) 转换为 (19, 224, 224)
#
#             # 将批次大小恢复为预期的维度
#             inputs_image = inputs_image.unsqueeze(0)  # 转换为 (1, 19, 224, 224)
#
#             print(f"Reshaped inputs_image: {inputs_image.shape}")
#
#             # 确保转换后的形状是 [batch_size, num_instances, channels, height, width]
#             inputs_image = inputs_image.to(device, dtype=torch.float)
#
#             with torch.no_grad():
#                 with amp.autocast():
#                     outputs = model(inputs_image)
#                     loss = criterion(outputs, labels)
#
#             val_loss += loss.item() * inputs_image.size(0)
#             _, preds = torch.max(outputs, 1)
#             val_corrects += torch.sum(preds == labels.data)
#             val_labels.extend(labels.cpu().numpy())
#             val_preds.extend(preds.cpu().numpy())
#
#         val_loss = val_loss / len(val_loader.dataset)
#         val_acc = val_corrects.double() / len(val_loader.dataset)
#
#         print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
#
#         val_accuracy, val_f1, val_spearman, val_specificity, val_qwk, val_mean_metrics, val_cm = calculate_metrics(
#             val_labels, val_preds, num_classes=num_classes)
#         print(
#             f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} QWK: {val_qwk:.4f} Mean Metrics: {val_mean_metrics:.4f}')
#         # print(f'Validation Confusion Matrix:\n{val_cm}')
#
#         # Save validation confusion matrix as CSV
#         val_cm_df = pd.DataFrame(val_cm, index=[f"True_{i}" for i in range(num_classes)], columns=[f"Pred_{i}" for i in range(num_classes)])
#         val_cm_path = os.path.join(f'model/{config.model_name}/', f'val_cm_epoch_{epoch}_lr_{config.lr}_{config.data_number}.csv')
#         val_cm_df.to_csv(val_cm_path, index=True)
#
#         # Save model for each epoch
#         torch.save(model.state_dict(),
#                    f'model/{config.model_name}/mil_epoch_{epoch}_split_{config.split}_bs_{config.batch_size}_lr_{config.lr}_{config.data_number}.pth')
#
#     return model


import torch
from torch.optim import lr_scheduler
import torch.cuda.amp as amp
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from config import config



def train_abmil_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, num_classes=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler()

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    #
    # unfreeze_steps = [("layer4", 1), ("layer3", 2), ("layer2", 3), ("layer1", 4)]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        #
        # for layer_name, step in unfreeze_steps:
        #     if epoch >= step:
        #         unfreeze_layers(model, unfreeze_from=layer_name)

            # Reset optimizer to include newly unfrozen layers
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []

        train_loader_tqdm = tqdm(train_loader, desc="Training")

        for samples in train_loader_tqdm:
            images = samples['features']  # This is a list of images (batch of images from the same localizer)
            labels = samples['label'].to(device, dtype=torch.long)
            if isinstance(images, torch.Tensor):
                images = [images]

            # Stack images into a tensor
            inputs_image = torch.stack(images).to(device, dtype=torch.float)

            # 确保形状为期望的5维：[batch_size, num_instances, channels, height, width]
            if inputs_image.dim() == 4:  # [num_instances, channels, height, width]
                inputs_image = inputs_image.unsqueeze(0)  # Add batch dimension
            elif inputs_image.dim() == 3:  # [channels, height, width] (single image case)
                inputs_image = inputs_image.unsqueeze(0).unsqueeze(0)  # Add batch and num_instances dimensions

            inputs_image = inputs_image.view(1, -1, inputs_image.size(-3), inputs_image.size(-2), inputs_image.size(-1))

            # print(f"Final inputs_image shape: {inputs_image.shape}")  # 打印最终的输入形状

            optimizer.zero_grad()

            with amp.autocast():
                # Forward pass
                # outputs= model(inputs_image,label=labels)
                # outputs,_= model(inputs_image)
                outputs = model(inputs_image)

                # Use only logits for loss calculation
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs_image.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            del outputs, loss, images, inputs_image, labels, preds
            torch.cuda.empty_cache()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        train_accuracy, train_f1, train_spearman, train_specificity, train_qwk, train_mean_metrics, train_cm = calculate_metrics(
            all_labels, all_preds, num_classes=num_classes)
        print(
            f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} QWK: {train_qwk:.4f} Mean Metrics: {train_mean_metrics:.4f}')


        train_cm_df = pd.DataFrame(train_cm, index=[f"True_{i}" for i in range(num_classes)],
                                   columns=[f"Pred_{i}" for i in range(num_classes)])
        train_cm_df.to_csv(f'model/{config.model_name}/train_cm_lr_{config.lr}_{config.data_number}_epoch_{epoch}.csv', index=True)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_labels = []
        val_preds = []

        val_loader_tqdm = tqdm(val_loader, desc="Validation")

        for samples in val_loader_tqdm:
            images = samples['features']
            labels = samples['label'].to(device, dtype=torch.long)

            if isinstance(images, torch.Tensor):
                images = [images]

            inputs_image = torch.stack(images).to(device, dtype=torch.float)

            # 确保形状为5维
            if inputs_image.dim() == 4:  # [num_instances, channels, height, width]
                inputs_image = inputs_image.unsqueeze(0)  # Add batch dimension
            elif inputs_image.dim() == 3:  # [channels, height, width] (single image case)
                inputs_image = inputs_image.unsqueeze(0).unsqueeze(0)  # Add batch and num_instances dimensions

            inputs_image = inputs_image.view(1, -1, inputs_image.size(-3), inputs_image.size(-2), inputs_image.size(-1))

            # print(f"Final inputs_image shape: {inputs_image.shape}")  # 打印最终的输入形状

            with torch.no_grad():
                with amp.autocast():
                    # outputs,_= model(inputs_image)
                    outputs = model(inputs_image)
                    # outputs = model(inputs_image, label=labels)
                    loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs_image.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

            del outputs, loss, images, inputs_image, labels, preds
            torch.cuda.empty_cache()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        val_accuracy, val_f1, val_spearman, val_specificity, val_qwk, val_mean_metrics, val_cm = calculate_metrics(
            val_labels, val_preds, num_classes=num_classes)
        print(
            f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} QWK: {val_qwk:.4f} Mean Metrics: {val_mean_metrics:.4f}')

        # 保存验证集的混淆矩阵
        val_cm_df = pd.DataFrame(val_cm, index=[f"True_{i}" for i in range(num_classes)],
                                 columns=[f"Pred_{i}" for i in range(num_classes)])
        val_cm_df.to_csv(f'model/{config.model_name}/val_cm_lr_{config.lr}_{config.data_number}_epoch_{epoch}.csv',
                         index=True)

        torch.save(model.state_dict(),
                   f'model/{config.model_name}/epoch_{epoch}_split_{config.split}_bs_{config.batch_size}_lr_{config.lr}_{config.data_number}.pth')

    return model



### cuda out of memory

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, num_classes=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler()

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    unfreeze_steps = [("layer4", 1), ("layer3", 2), ("layer2", 3), ("layer1", 4)]

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for layer_name, step in unfreeze_steps:
            if epoch >= step:
                unfreeze_layers(model, unfreeze_from=layer_name)

            # Reset optimizer to include newly unfrozen layers
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []

        train_loader_tqdm = tqdm(train_loader, desc="Training")

        for samples in train_loader_tqdm:
            images = samples['image']  # This is a list of images (batch of images from the same localizer)
            labels = samples['label'].to(device, dtype=torch.long)
            if isinstance(images, torch.Tensor):
                images = [images]

            # Stack images into a tensor
            inputs_image = torch.stack(images).to(device, dtype=torch.float)

            # 确保形状为期望的5维：[batch_size, num_instances, channels, height, width]
            if inputs_image.dim() == 4:  # [num_instances, channels, height, width]
                inputs_image = inputs_image.unsqueeze(0)  # Add batch dimension
            elif inputs_image.dim() == 3:  # [channels, height, width] (single image case)
                inputs_image = inputs_image.unsqueeze(0).unsqueeze(0)  # Add batch and num_instances dimensions

            inputs_image = inputs_image.view(1, -1, inputs_image.size(-3), inputs_image.size(-2), inputs_image.size(-1))

            # print(f"Final inputs_image shape: {inputs_image.shape}")  # 打印最终的输入形状

            optimizer.zero_grad()

            with amp.autocast():
                # Forward pass
                outputs, instance_logits, attention_scores = model(inputs_image)

                # Use only logits for loss calculation
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs_image.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            del outputs, loss, images, inputs_image, labels, preds
            torch.cuda.empty_cache()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        train_accuracy, train_f1, train_spearman, train_specificity, train_qwk, train_mean_metrics, train_cm = calculate_metrics(
            all_labels, all_preds, num_classes=num_classes)
        print(
            f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} QWK: {train_qwk:.4f} Mean Metrics: {train_mean_metrics:.4f}')

        # 保存训练集的混淆矩阵
        # train_cm_df = pd.DataFrame(train_cm, index=[f"True_{i}" for i in range(num_classes)],
        #                            columns=[f"Pred_{i}" for i in range(num_classes)])
        # train_cm_df.to_csv(f'confusion_matrices/train_cm_lr_{config.lr}_{config.data_number}_epoch_{epoch}.csv', index=True)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_labels = []
        val_preds = []

        val_loader_tqdm = tqdm(val_loader, desc="Validation")

        for samples in val_loader_tqdm:
            images = samples['image']
            labels = samples['label'].to(device, dtype=torch.long)

            if isinstance(images, torch.Tensor):
                images = [images]

            inputs_image = torch.stack(images).to(device, dtype=torch.float)

            # 确保形状为5维
            if inputs_image.dim() == 4:  # [num_instances, channels, height, width]
                inputs_image = inputs_image.unsqueeze(0)  # Add batch dimension
            elif inputs_image.dim() == 3:  # [channels, height, width] (single image case)
                inputs_image = inputs_image.unsqueeze(0).unsqueeze(0)  # Add batch and num_instances dimensions

            inputs_image = inputs_image.view(1, -1, inputs_image.size(-3), inputs_image.size(-2), inputs_image.size(-1))

            # print(f"Final inputs_image shape: {inputs_image.shape}")  # 打印最终的输入形状

            with torch.no_grad():
                with amp.autocast():
                    outputs, instance_logits, attention_scores = model(inputs_image)
                    loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs_image.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

            del outputs, loss, images, inputs_image, labels, preds
            torch.cuda.empty_cache()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        val_accuracy, val_f1, val_spearman, val_specificity, val_qwk, val_mean_metrics, val_cm = calculate_metrics(
            val_labels, val_preds, num_classes=num_classes)
        print(
            f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} QWK: {val_qwk:.4f} Mean Metrics: {val_mean_metrics:.4f}')

        # 保存验证集的混淆矩阵
        val_cm_df = pd.DataFrame(val_cm, index=[f"True_{i}" for i in range(num_classes)],
                                 columns=[f"Pred_{i}" for i in range(num_classes)])
        val_cm_df.to_csv(f'model/{config.model_name}/val_cm_lr_{config.lr}_{config.data_number}_epoch_{epoch}.csv',
                         index=True)

        torch.save(model.state_dict(),
                   f'model/{config.model_name}/epoch_{epoch}_split_{config.split}_bs_{config.batch_size}_lr_{config.lr}_{config.data_number}.pth')

    return model

# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, num_classes=3):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     scaler = amp.GradScaler()
#
#     # Learning rate scheduler
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)
#
#         model.train()
#         running_loss = 0.0
#         running_corrects = 0
#         all_labels = []
#         all_preds = []
#
#         train_loader_tqdm = tqdm(train_loader, desc="Training")
#
#         for samples in train_loader_tqdm:
#             images = samples['image']  # This is a list of images (batch of images from the same localizer)
#             labels = samples['label'].to(device, dtype=torch.long)
#             if isinstance(images, torch.Tensor):
#                 images = [images]
#
#             # 打印初始images的形状
#             print(f"Initial images shape: {images[0].shape}")
#
#             # Stack images into a tensor
#             inputs_image = torch.stack(images).to(device, dtype=torch.float)
#             print(f"After stacking images: {inputs_image.shape}")  # 打印stack后的形状
#
#             # 调整形状到5维：[batch_size, num_instances, channels, height, width]
#             if inputs_image.dim() == 4:  # [num_instances, channels, height, width]
#                 inputs_image = inputs_image.unsqueeze(0)  # Add batch dimension
#             elif inputs_image.dim() == 3:  # [channels, height, width] (single image case)
#                 inputs_image = inputs_image.unsqueeze(0).unsqueeze(0)  # Add batch and num_instances dimensions
#
#             print(f"After adding dimensions: {inputs_image.shape}")  # 打印添加维度后的形状
#
#             # 期望的最终形状
#             expected_shape = (1, inputs_image.size(1), 1, 224, 224)
#             print(f"Expected inputs_image shape: {expected_shape}")
#
#             # 确保形状为期望的5维
#             inputs_image = inputs_image.view(1, -1, inputs_image.size(-3), inputs_image.size(-2), inputs_image.size(-1))
#
#             print(f"Final inputs_image shape: {inputs_image.shape}")  # 打印最终的输入形状
#
#             optimizer.zero_grad()
#
#             with amp.autocast():
#                 outputs = model(inputs_image)
#                 loss = criterion(outputs, labels)
#
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             running_loss += loss.item() * inputs_image.size(0)
#             _, preds = torch.max(outputs, 1)
#             running_corrects += torch.sum(preds == labels.data)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#
#         scheduler.step()
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / len(train_loader.dataset)
#
#         print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#         train_accuracy, train_f1, train_spearman, train_specificity, train_qwk, train_mean_metrics, train_cm = calculate_metrics(
#             all_labels, all_preds, num_classes=num_classes)
#         print(
#             f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} QWK: {train_qwk:.4f} Mean Metrics: {train_mean_metrics:.4f}')
#
#         model.eval()
#         val_loss = 0.0
#         val_corrects = 0
#         val_labels = []
#         val_preds = []
#
#         val_loader_tqdm = tqdm(val_loader, desc="Validation")
#
#         for samples in val_loader_tqdm:
#             images = samples['image']
#             labels = samples['label'].to(device, dtype=torch.long)
#
#             if isinstance(images, torch.Tensor):
#                 images = [images]
#
#             inputs_image = torch.stack(images).to(device, dtype=torch.float)
#             print(f"Validation - After stacking images: {inputs_image.shape}")  # 打印stack后的形状
#
#             # 确保形状为5维
#             if inputs_image.dim() == 4:  # [num_instances, channels, height, width]
#                 inputs_image = inputs_image.unsqueeze(0)  # Add batch dimension
#             elif inputs_image.dim() == 3:  # [channels, height, width] (single image case)
#                 inputs_image = inputs_image.unsqueeze(0).unsqueeze(0)  # Add batch and num_instances dimensions
#
#             print(f"Validation - After adding dimensions: {inputs_image.shape}")  # 打印添加维度后的形状
#
#             inputs_image = inputs_image.view(1, -1, inputs_image.size(-3), inputs_image.size(-2), inputs_image.size(-1))
#
#             print(f"Validation - Final inputs_image shape: {inputs_image.shape}")  # 打印最终的输入形状
#
#             with torch.no_grad():
#                 with amp.autocast():
#                     outputs = model(inputs_image)
#                     loss = criterion(outputs, labels)
#
#             val_loss += loss.item() * inputs_image.size(0)
#             _, preds = torch.max(outputs, 1)
#             val_corrects += torch.sum(preds == labels.data)
#             val_labels.extend(labels.cpu().numpy())
#             val_preds.extend(preds.cpu().numpy())
#
#         val_loss = val_loss / len(val_loader.dataset)
#         val_acc = val_corrects.double() / len(val_loader.dataset)
#
#         print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
#
#         val_accuracy, val_f1, val_spearman, val_specificity, val_qwk, val_mean_metrics, val_cm = calculate_metrics(
#             val_labels, val_preds, num_classes=num_classes)
#         print(
#             f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} QWK: {val_qwk:.4f} Mean Metrics: {val_mean_metrics:.4f}')
#
#         torch.save(model.state_dict(),
#                    f'model/{config.model_name}/epoch_{epoch}_split_{config.split}_bs_{config.batch_size}_lr_{config.lr}_{config.data_number}.pth')
#
#     return model
