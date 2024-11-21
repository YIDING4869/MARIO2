import pandas as pd
import torch
from config import config

# def evaluate_model(model, test_loader, output_file=config.output_file):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     predictions = []
#
#     with torch.no_grad():
#         for samples in test_loader:
#             inputs_image = samples['image'].to(device)
#             inputs_loc_image = samples['loc_image'].to(device)
#             case_ids = samples['case_id']
#             patient_info = samples['patient_info'].to(device)
#
#             outputs = model(inputs_image, inputs_loc_image, patient_info)
#
#             _, predicted = torch.max(outputs, 1)
#
#             for case_id, prediction in zip(case_ids, predicted):
#                 predictions.append({'case': case_id.item(), 'prediction': prediction.item()})
#
#     df = pd.DataFrame(predictions)
#     df.to_csv(output_file, index=False)
#
#     print(f'Submission file saved as {output_file}')

###v1

# import torch
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# import numpy as np
# from scipy.stats import spearmanr
#
#
# def evaluate_model(model, test_loader, output_file=config.output_file):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     predictions = []
#
#     with torch.no_grad():
#         for samples in test_loader:
#             inputs_image = samples['image'].to(device)
#             inputs_loc_image = samples['loc_image'].to(device)
#             case_ids = samples['case_id']
#
#             outputs = model(inputs_image, inputs_loc_image)
#
#             _, predicted = torch.max(outputs, 1)
#
#             for case_id, prediction in zip(case_ids, predicted):
#                 predictions.append({'case': case_id.item(), 'prediction': prediction.item()})
#
#     df = pd.DataFrame(predictions)
#     df.to_csv(output_file, index=False)
#
#     print(f'Submission file saved as {output_file}')


###v2

import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from scipy.stats import spearmanr


def evaluate_model(model, test_loader, output_file=config.output_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []

    with torch.no_grad():
        for samples in test_loader:
            inputs_image = samples['image'].to(device)
            # inputs_loc_image = samples['loc_image'].to(device)
            case_ids = samples['case_id']

            outputs = model(inputs_image)

            _, predicted = torch.max(outputs, 1)

            for case_id, prediction in zip(case_ids, predicted):
                predictions.append({'case': case_id.item(), 'prediction': prediction.item()})

    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)

    print(f'Submission file saved as {output_file}')
