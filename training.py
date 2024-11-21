import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy.stats import spearmanr
import numpy as np
from torch.optim import lr_scheduler
import torch.cuda.amp as amp
from config import config

###v1
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from scipy.stats import spearmanr
import numpy as np


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def calculate_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    spearman_corr, _ = spearmanr(true_labels, predictions)
    qwk = quadratic_weighted_kappa(true_labels, predictions)

    cm = confusion_matrix(true_labels, predictions)
    specificity_per_class = []
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    specificity = np.mean(specificity_per_class)

    mean_metrics = (f1 + spearman_corr + specificity + qwk) / 4

    return accuracy, f1, spearman_corr, specificity, qwk, mean_metrics


###v1
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_metrics = (0, 0, 0, 0, 0)  # Accuracy, F1, Spearman, Specificity, Mean Metrics
#
#     # Learning rate scheduler
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
#     scaler = amp.GradScaler()
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)
#
#         model.train()
#
#         running_loss = 0.0
#         running_corrects = 0
#         all_labels = []
#         all_preds = []
#
#         # Adding tqdm progress bar for training
#         train_loader_tqdm = tqdm(train_loader, desc="Training")
#
#         for samples in train_loader_tqdm:
#             inputs_image = samples['image'].to(device, dtype=torch.float)
#             inputs_loc_image = samples['loc_image'].to(device, dtype=torch.float)
#             labels = samples['label'].to(device, dtype=torch.long)
#
#             optimizer.zero_grad()
#
#             with amp.autocast():
#                 outputs = model(inputs_image, inputs_loc_image)
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
#         train_accuracy, train_f1, train_spearman, train_specificity, train_mean_metrics = calculate_metrics(all_labels,
#                                                                                                             all_preds)
#         print(
#             f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} Mean Metrics: {train_mean_metrics:.4f}')
#
#         model.eval()
#         val_loss = 0.0
#         val_corrects = 0
#         val_labels = []
#         val_preds = []
#
#         # Adding tqdm progress bar for validation
#         val_loader_tqdm = tqdm(val_loader, desc="Validation")
#
#         for samples in val_loader_tqdm:
#             inputs_image = samples['image'].to(device, dtype=torch.float)
#             inputs_loc_image = samples['loc_image'].to(device, dtype=torch.float)
#             labels = samples['label'].to(device, dtype=torch.long)
#
#             with torch.no_grad():
#                 with amp.autocast():
#                     outputs = model(inputs_image, inputs_loc_image)
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
#         val_accuracy, val_f1, val_spearman, val_specificity, val_mean_metrics = calculate_metrics(val_labels, val_preds)
#         print(
#             f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} Mean Metrics: {val_mean_metrics:.4f}')
#
#         if val_mean_metrics > best_metrics[-1]:
#             best_metrics = (val_accuracy, val_f1, val_spearman, val_specificity, val_mean_metrics)
#             best_model_wts = copy.deepcopy(model.state_dict())
#
#         torch.save(model.state_dict(),
#                    f'model/{config.model_name}/epoch_{epoch}_split_{config.split}_bs_{config.batch_size}_lr_{config.lr}_loc_info.pth')
#
#     model.load_state_dict(best_model_wts)
#     return model

### v2 all
def train_model_all(model, train_loader, criterion, optimizer, num_epochs=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_metrics = (0, 0, 0, 0, 0)  # Accuracy, F1, Spearman, Specificity, Mean Metrics

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []

        # Adding tqdm progress bar for training
        train_loader_tqdm = tqdm(train_loader, desc="Training")

        for samples in train_loader_tqdm:
            inputs_image = samples['image'].to(device, dtype=torch.float)
            # inputs_loc_image = samples['loc_image'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(inputs_image)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs_image.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        train_accuracy, train_f1, train_spearman, train_specificity, train_mean_metrics = calculate_metrics(all_labels,
                                                                                                            all_preds)
        print(
            f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} Mean Metrics: {train_mean_metrics:.4f}')

        torch.save(model.state_dict(),
                   f'model/{config.model_name}/epoch_{epoch}_split_{config.split}_bs_{config.batch_size}_lr_{config.lr}_loc_info_all.pth')

    # model.load_state_dict(best_model_wts)
    return model


#v2

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_metrics = (0, 0, 0, 0, 0)  # Accuracy, F1, Spearman, Specificity, Mean Metrics

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []

        # Adding tqdm progress bar for training
        train_loader_tqdm = tqdm(train_loader, desc="Training")

        for samples in train_loader_tqdm:
            inputs_image = samples['image'].to(device, dtype=torch.float)
            # inputs_loc_image = samples['loc_image'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(inputs_image)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs_image.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        train_accuracy, train_f1, train_spearman, train_specificity, qwk, train_mean_metrics = calculate_metrics(all_labels,
                                                                                                            all_preds)
        print(
            f'Train Accuracy: {train_accuracy:.4f} F1 Score: {train_f1:.4f} Spearman Corr: {train_spearman:.4f} Specificity: {train_specificity:.4f} QWK: {qwk:.4f} Mean Metrics: {train_mean_metrics:.4f}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_labels = []
        val_preds = []

        # Adding tqdm progress bar for validation
        val_loader_tqdm = tqdm(val_loader, desc="Validation")

        for samples in val_loader_tqdm:
            inputs_image = samples['image'].to(device, dtype=torch.float)
            # inputs_loc_image = samples['loc_image'].to(device, dtype=torch.float)
            labels = samples['label'].to(device, dtype=torch.long)

            with torch.no_grad():
                with amp.autocast():
                    outputs = model(inputs_image)
                    loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs_image.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        val_accuracy, val_f1, val_spearman, val_specificity, val_qwk, val_mean_metrics = calculate_metrics(val_labels, val_preds)
        print(
            f'Validation Accuracy: {val_accuracy:.4f} F1 Score: {val_f1:.4f} Spearman Corr: {val_spearman:.4f} Specificity: {val_specificity:.4f} QWK: {val_qwk:.4f} Mean Metrics: {val_mean_metrics:.4f}')
        #
        # if val_mean_metrics > best_metrics[-1]:
        #     best_metrics = (val_accuracy, val_f1, val_spearman, val_specificity, val_mean_metrics)
        #     best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(),
                   f'model/{config.model_name}/epoch_{epoch}_split_{config.split}_bs_{config.batch_size}_lr_{config.lr}_{config.data_number}')

    # model.load_state_dict(best_model_wts)
    return model
