#!/bin/python

import copy

import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import statsmodels.stats.api as sms

from tqdm import tqdm
from sklearn.metrics import\
    precision_recall_curve, roc_curve, auc, roc_auc_score,\
    accuracy_score, f1_score, confusion_matrix

from config import Config

torch.manual_seed(Config.random_state)


class MLP(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 2)
        self.probability_reduction = nn.LogSoftmax(dim=1)  # Feature dimension

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return self.probability_reduction(out)


class LASSO(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 1, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.Sigmoid()(out)

        return out


# GPU execution takes a lot longer for this. Not sure why.
# Federation has to be repeated for each fold of data
class Federated:
    def __init__(self, split_training_datasets, split_testing_datasets, model_type, noise):
        # Store metrics here
        self.dict_metrics = {}

        # This is the type of the model. It's either an MLP or a LASSO
        # ALL CAPS
        if model_type not in ('MLP', 'LASSO'):
            raise NotImplementedError
        self.model_type = model_type

        # Notifications
        self.noise = noise
        if self.noise:
            print('Adding noise to training')

        # Load datasets
        self.hospitals = list(split_training_datasets.keys())
        for split in range(Config.cross_val_folds):
            print('Current fold:', split)
            self.current_split = split

            self.dict_metrics_split = {}

            # These have to be nulled after each split
            # Datasets
            self.training_datasets_dataloaders = {}
            self.training_datasets = {}
            self.testing_datasets = {}

            # Number of patients for scaling
            self.total_patients = 0
            self.patients_per_hospital = {}

            for facility in self.hospitals:
                self.dict_metrics_split[facility] = {}
                self.dict_metrics_split[facility]['loss'] = []
                self.dict_metrics_split[facility]['auroc'] = []
                self.dict_metrics_split[facility]['auprc'] = []

                df_train_hosp = split_training_datasets[facility][split]
                self.total_patients += df_train_hosp.shape[0]
                self.patients_per_hospital[facility] = df_train_hosp.shape[0]

                # Create float tensors for each hospital
                self.training_datasets_dataloaders[facility] = torch.utils.data.DataLoader(
                    df_train_hosp.values.astype('float32'),
                    batch_size=64, shuffle=True, num_workers=8)

                self.training_datasets[facility] = df_train_hosp
                self.testing_datasets[facility] = split_testing_datasets[facility][split]

            # Number of patients
            self.input_shape = df_train_hosp.shape[1] - 1

            # Start a per split federation
            self.gaping_maw()

            # Record metrics
            self.dict_metrics[split] = copy.deepcopy(self.dict_metrics_split)

        # Save final metrics
        pd.to_pickle(
            self.dict_metrics,
            f'{Config.metrics_save_dir}/Federated{self.model_type}TrainingMetrics_Noise{self.noise}.pickle',
            protocol=4)

    def gaussian_noise(self, model_params):
        return_dict = {}

        for param, tensor in model_params.items():
            # Mean:0, SD: 1
            noise_tensor = torch.randn(
                tensor.shape,
                dtype=tensor.dtype)
            noise_tensor *= Config.gaussian_noise_scale

            return_dict[param] = tensor + noise_tensor

        return return_dict

    def weight_scaling_factor(self, hospital):
        # Returns proportion of datapoints
        local_count = self.patients_per_hospital[hospital]
        return local_count / self.total_patients

    def scale_params(self, model_params, scaling_factor):
        return_dict = {}

        # Not a huge fan of modifying what I'm iterating over
        # Should work either way
        for param, tensor in model_params.items():
            return_dict[param] = tensor * scaling_factor

        return return_dict

    def sum_params(self, scaled_hosp_params):
        # scaled_hosp_params is a list
        # Sum the corresponding element from each and return the sum
        # The sum in this case is the federated average
        return_dict = {layer: None for layer in scaled_hosp_params[0]}
        for parameter in return_dict:
            scaled_param_list = [i[parameter] for i in scaled_hosp_params]

            # Combines iterable of tensors into 1
            scaled_param_tensor = torch.stack(scaled_param_list)

            # dim is important or it just returns a scalar
            return_dict[parameter] = torch.sum(scaled_param_tensor, dim=0)

        return return_dict

    def test_global_model(self, model, epoch):
        model.eval()
        test_results = []

        # NOTE
        # This is a continuous evaluation on each facility's TRAINING dataset
        for facility, df_test in self.training_datasets.items():
            X_test = torch.Tensor(
                df_test.drop('MORTALITY', axis=1).values.astype('float32'))
            y_test = torch.Tensor(
                df_test[['MORTALITY']].values.astype('float32'))

            with torch.no_grad():
                y_pred = model(X_test)

            # Predictions need to be reduced to where they can be understood by BCE
            # and the sklearn functions
            if self.model_type == 'MLP':
                y_pred = y_pred.exp()[:, 1]
                loss = torch.nn.functional.binary_cross_entropy(
                    y_pred, y_test.squeeze())

            elif self.model_type == 'LASSO':
                loss = torch.nn.functional.binary_cross_entropy(
                    y_pred, y_test)

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            auprc = auc(recall, precision)
            auroc = auc(fpr, tpr)

            # # Record results
            self.dict_metrics_split[facility]['loss'].append(loss.item())
            self.dict_metrics_split[facility]['auroc'].append(auroc)
            self.dict_metrics_split[facility]['auprc'].append(auprc)

            test_results.append((facility, loss.item(), auroc, auprc))

        if epoch == Config.n_epochs - 1:
            df_test_results = pd.DataFrame(
                test_results,
                columns=['FACILITY', 'BCE LOSS', 'AUROC', 'AUPRC'])
            print(df_test_results.sort_values('FACILITY').reset_index(drop=True))

    def gaping_maw(self):
        if self.model_type == 'MLP':
            global_model = MLP(self.input_shape)
        elif self.model_type == 'LASSO':
            global_model = LASSO(self.input_shape)

        # Repeat training for each epoch for each hospital
        for epoch in tqdm(range(Config.n_epochs)):
            # Get global model weights
            # Will be the init weights for all local models
            global_params = global_model.state_dict()

            # Local weights go here
            scaled_hosp_params = []

            # Train a NEW client model for each epoch
            for client in self.hospitals:

                if self.model_type == 'MLP':
                    local_model = MLP(self.input_shape)
                    local_model.load_state_dict(global_params)

                    # Train this model
                    criterion = torch.nn.NLLLoss()
                    optim = torch.optim.Adam(local_model.parameters(), lr=0.001)

                elif self.model_type == 'LASSO':
                    local_model = LASSO(self.input_shape)
                    local_model.load_state_dict(global_params)

                    # Train this model
                    criterion = torch.nn.MSELoss(reduction='sum')
                    optim = torch.optim.SGD(local_model.parameters(), lr=0.001)

                # Each hospital is trained once only within a global epoch
                for train_data in self.training_datasets_dataloaders[client]:
                    x_train = train_data[:, :-1]
                    y_train = train_data[:, -1]

                    pred = local_model(x_train)

                    if self.model_type == 'MLP':
                        loss = criterion(pred.squeeze(), y_train.long())

                    elif self.model_type == 'LASSO':
                        loss = criterion(pred.squeeze(), y_train)
                        l1_norm = 0.1 * torch.norm(local_model.fc1.weight, p=1)
                        loss += l1_norm

                    loss.backward()
                    optim.step()

                scaling_factor = self.weight_scaling_factor(client)
                scaled_params = self.scale_params(
                    local_model.state_dict(), scaling_factor)

                # Add random gaussian noise
                # Only for the first epoch currently
                if self.noise:
                    scaled_params = self.gaussian_noise(scaled_params)

                scaled_hosp_params.append(scaled_params)

            # Tensor summation to derive federated average
            federated_state_dict = self.sum_params(scaled_hosp_params)

            # Update global model
            global_model.load_state_dict(federated_state_dict)

            # Break training loop for LASSO
            if self.model_type == 'LASSO':
                norm = torch.norm(torch.tensor(0.) - global_model.fc1.weight).item()
                if norm < 1e-4:
                    print('Threshold reached')
                    break

            # Test global model and print out metrics after each communications round
            self.test_global_model(global_model, epoch)

        # Save federated model after each split
        save_path = f'{Config.model_save_dir}/{self.model_type}_Federated_{self.current_split}.pth'
        if self.noise:
            save_path = f'{Config.model_save_dir}/{self.model_type}_Federated_{self.current_split}_noisy.pth'

        state_dict = global_model.state_dict()
        torch.save(state_dict, save_path)


def global_pooling(training_datasets, testing_datasets):
    facilities = list(training_datasets.keys())
    cols = training_datasets['MSH'][0].columns

    # Iterate through each split
    for split in range(Config.cross_val_folds):
        print('Current fold:', split)

        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        for facility in facilities:
            df_train = df_train.append(training_datasets[facility][split], sort=True)
            df_test = df_test.append(testing_datasets[facility][split], sort=True)

        # Rearrange columns to make sure everything is situated correctly
        df_train = df_train[cols]
        df_test = df_test[cols]

        if df_train.columns[-1] != 'MORTALITY':
            print('Nope.')
            exit(1)

        print('Total patients:', len(df_train))
        print('Outcomes:', len(df_train.query('MORTALITY == 1')))

        # Create datasets
        # Training gets a dataloder
        # Testing is fine as it is
        training_dataloader = torch.utils.data.DataLoader(
            df_train.values.astype('float32'),
            batch_size=Config.batch_size, shuffle=True, num_workers=8)

        # Testing
        x_test = torch.tensor(df_test.values[:, :-1], dtype=torch.float32)
        y_test = df_test.values[:, -1]
        best_testing_auroc = 0.

        # Instantiate pooled global model and related criteria
        model = MLP(df_train.shape[1] - 1)
        criterion = torch.nn.NLLLoss()
        optim = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in tqdm(range(Config.n_epochs)):
            model.train()
            for training_data in training_dataloader:
                # Get data
                x_train = training_data[:, :-1]
                y_train = training_data[:, -1]

                # The usual
                optim.zero_grad()
                pred = model(x_train)

                loss = criterion(pred.squeeze(), y_train.long())

                loss.backward()
                optim.step()

            # Some garden variety checkpointing
            model.eval()
            with torch.no_grad():
                testing_pred = model(x_test).exp()[:, 1]
                auroc = roc_auc_score(y_test, testing_pred.detach())
                if auroc > best_testing_auroc:
                    best_testing_auroc = copy.copy(auroc)
                    best_state_dict = model.state_dict()

        # Train and save global LASSO as well
        lso = LASSO(df_train.shape[1] - 1)
        criterion = torch.nn.MSELoss(reduction='sum')
        optim = torch.optim.SGD(lso.parameters(), lr=0.001)

        for _ in tqdm(range(Config.n_epochs)):
            lso.train()
            for training_data in training_dataloader:
                # Get data
                x_train = training_data[:, :-1]
                y_train = training_data[:, -1]

                # The usual
                optim.zero_grad()
                pred = lso(x_train)

                loss = criterion(pred.squeeze(), y_train)
                l1_norm = 0.1 * torch.norm(lso.fc1.weight, p=1)
                loss += l1_norm

                loss.backward()
                optim.step()

            norm = torch.norm(torch.tensor(0.) - lso.fc1.weight).item()
            if norm < 1e-4:
                print('Threshold reached')
                break

        # Save best models named according to the split
        torch.save(best_state_dict, f'{Config.model_save_dir}/MLP_Pooled_{split}.pth')
        torch.save(lso.state_dict(), f'{Config.model_save_dir}/LASSO_Pooled_{split}.pth')


def model_comparisons(training_datasets, testing_datasets):
    # Also does comparisons against the saved global models
    # and local LASSO
    hospitals = list(training_datasets.keys())

    all_results = []
    dict_metrics = {}

    for split in range(Config.cross_val_folds):
        print('Current fold:', split)
        dict_metrics[split] = {}

        for facility in hospitals:
            print('Facility:', facility)

            # Training and testing data
            df_train = training_datasets[facility][split]
            train_loader = torch.utils.data.DataLoader(
                df_train.values.astype('float32'),
                batch_size=Config.batch_size, shuffle=True, num_workers=4)

            # Doing all of testing data at once
            df_test = testing_datasets[facility][split]
            x_test = torch.Tensor(df_test.drop('MORTALITY', axis=1).values.astype('float32'))
            y_test = df_test[['MORTALITY']].values.astype('float32')

            # Patient proportions
            total_train = len(df_train)
            train_perc = df_train['MORTALITY'].sum() / len(df_train)
            test_perc = df_test['MORTALITY'].sum() / len(df_test)

            # Sanity check
            if df_train.columns[-1] != df_test.columns[-1] != 'MORTALITY':
                print('Yeah, no.')
                exit(1)

            # Declare model and related parameters
            model = MLP(df_train.shape[1] - 1)

            criterion = torch.nn.NLLLoss()
            optim = torch.optim.Adam(model.parameters(), lr=0.001)

            # Training - LOCAL MLP ___
            for _ in tqdm(range(Config.n_epochs)):
                model.train()
                for train_data in train_loader:
                    x_train = train_data[:, :-1]
                    y_train = train_data[:, -1].long()  # This is a float tensor - must be cast as long

                    optim.zero_grad()
                    pred = model(x_train)

                    # The .squeeze() is needed because of the batch dimension
                    loss = criterion(pred.squeeze(), y_train)

                    loss.backward()
                    optim.step()

            # Training - LOCAL LASSO ___
            lso = LASSO(df_train.shape[1] - 1)
            criterion = torch.nn.MSELoss(reduction='sum')
            optim = torch.optim.SGD(lso.parameters(), lr=0.001)

            for _ in tqdm(range(Config.n_epochs)):
                lso.train()
                for train_data in train_loader:
                    # Get data
                    x_train = train_data[:, :-1]
                    y_train = train_data[:, -1]

                    # The usual
                    optim.zero_grad()
                    pred = lso(x_train)

                    loss = criterion(pred.squeeze(), y_train)
                    l1_norm = 0.1 * torch.norm(lso.fc1.weight, p=1)
                    loss += l1_norm

                    loss.backward()
                    optim.step()

                norm = torch.norm(torch.tensor(0.) - lso.fc1.weight).item()
                if norm < 1e-4:
                    print('Threshold reached')
                    break

            # Testing - Get all predictions one by one
            # LOCAL MLP ___
            model.eval()
            with torch.no_grad():
                mlp_pred = model(x_test)

            # Eval metrics on CPU
            # There's an .exp() here since the auroc / auprc are calculated on the probabilities
            # NLL loss is calculated on Log Softmax output
            mlp_pred = mlp_pred.exp().cpu().numpy()[:, 1]

            # LOCAL LASSO ___
            lso.eval()
            with torch.no_grad():
                lso_pred = lso(x_test)

            # GLOBAL LASSO ___
            pooled_lso = LASSO(df_train.shape[1] - 1)
            pooled_LASSO_state_dict_path = f'{Config.model_save_dir}/LASSO_Pooled_{split}.pth'
            pooled_lso.load_state_dict(torch.load(pooled_LASSO_state_dict_path))
            pooled_lso.eval()
            with torch.no_grad():
                lso_pooled_pred = pooled_lso(x_test)
                lso_pooled_pred = lso_pooled_pred.numpy()

            # FEDERATED LASSO ___
            global_lso = LASSO(df_train.shape[1] - 1)
            federated_LASSO_state_dict_path = f'{Config.model_save_dir}/LASSO_Federated_{split}.pth'
            global_lso.load_state_dict(torch.load(federated_LASSO_state_dict_path))
            global_lso.eval()
            with torch.no_grad():
                global_lso_pred = global_lso(x_test)
                global_lso_pred = global_lso_pred.numpy()

            # FEDERATED MLP (No noise) ___
            global_model = MLP(df_train.shape[1] - 1)
            federated_MLP_state_dict_path = f'{Config.model_save_dir}/MLP_Federated_{split}.pth'
            global_model.load_state_dict(torch.load(federated_MLP_state_dict_path))
            global_model.eval()
            with torch.no_grad():
                global_pred = global_model(x_test)
                global_pred = global_pred.exp().numpy()[:, 1]

            # FEDERATED MLP (With noise) ___
            global_model_noisy = MLP(df_train.shape[1] - 1)
            federated_MLP_noisy_state_dict_path = f'{Config.model_save_dir}/MLP_Federated_{split}_noisy.pth'
            global_model_noisy.load_state_dict(torch.load(federated_MLP_noisy_state_dict_path))
            global_model_noisy.eval()
            with torch.no_grad():
                global_pred_noisy = global_model_noisy(x_test)
                global_pred_noisy = global_pred_noisy.exp().numpy()[:, 1]

            # Pooled MLP ___
            global_model_pooled = MLP(df_train.shape[1] - 1)
            pooled_MLP_state_dict_path = f'{Config.model_save_dir}/MLP_Pooled_{split}.pth'
            global_model_pooled.load_state_dict(torch.load(pooled_MLP_state_dict_path))
            global_model_pooled.eval()
            with torch.no_grad():
                global_pred_pooled = global_model_pooled(x_test)
                global_pred_pooled = global_pred_pooled.exp().numpy()[:, 1]

            # Put all predictions together
            pred_dict = {
                'MLP: Local': mlp_pred,
                'MLP: Federated (no Gaussian noise)': global_pred,
                'MLP: Federated (with Gaussian noise)': global_pred_noisy,
                'MLP: Global (Pooled data)': global_pred_pooled,
                'LASSO: Federated': global_lso_pred,
                'LASSO: Global (Pooled data)': lso_pooled_pred,
                'LASSO: Local': lso_pred
            }

            # Store results
            dict_metrics[split][facility] = {}
            for desc in pred_dict:
                dict_metrics[split][facility][desc] = {}

            for model_desc, predictions in pred_dict.items():
                fpr, tpr, thresholds = roc_curve(y_test, predictions)
                precision, recall, _ = precision_recall_curve(y_test, predictions)
                auroc = auc(fpr, tpr)
                auprc = auc(recall, precision)

                optimal_threshold = thresholds[np.argmax(tpr - fpr)]
                predictions_binary = predictions >= optimal_threshold

                accuracy, sens, spec, f1s = metrics_calculator(y_test, predictions_binary)

                dict_level = dict_metrics[split][facility][model_desc]

                dict_level['auroc'] = auroc
                dict_level['auprc'] = auprc
                dict_level['roc_curve'] = (fpr, tpr, thresholds)
                dict_level['pr_curve'] = (recall, precision)
                dict_level['accuracy'] = accuracy
                dict_level['sens'] = sens
                dict_level['spec'] = spec
                dict_level['f1s'] = f1s

                # AUROC and AUPRC in the final iteration
                all_results.append((
                    split, facility, model_desc, auroc, auprc, accuracy, sens, spec, f1s))

    # Tabulate results
    columns = [
        'SPLIT', 'FACILITY', 'MODEL', 'AUROC', 'AUPRC',
        'ACC', 'SENS', 'SPEC', 'F1S']
    df_results = pd.DataFrame(all_results, columns=columns).round(3)
    df_results = df_results.sort_values('FACILITY').reset_index(drop=True)
    df_results = df_results.set_index(['FACILITY', 'MODEL']).sort_index()
    df_results.to_excel('ModelPerformanceComparisonsALL.xlsx')

    df_results = df_results.reset_index().\
        groupby(['FACILITY', 'MODEL']).apply(mean_ci_calculator)
    df_results = pd.DataFrame.from_records(
        df_results, index=df_results.index,
        columns=['AUROC', 'AUPRC', 'ACC', 'SENS', 'SPEC', 'F1S'])

    df_results.to_excel('ResultsWithCI.xlsx')
    pd.to_pickle(dict_metrics, 'PerformanceCurves.pickle', protocol=4)

    print(df_results)
    breakpoint()


def mean_ci_calculator(df):
    processed_results = []
    for metric in ['AUROC', 'AUPRC', 'ACC', 'SENS', 'SPEC', 'F1S']:
        metric_mean = df[metric].mean().round(3)
        metric_ci = sms.DescrStatsW(df[metric]).tconfint_mean()
        metric_ci = (round(metric_ci[0], 3), round(metric_ci[1], 3))
        processed_results.append(f'{metric_mean} ({metric_ci[0]} - {metric_ci[1]})')

    return processed_results


def metrics_calculator(y_true, y_pred):
    # y_pred is supposed to be binary

    accuracy = accuracy_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred)

    c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    sens = c_matrix[0, 0] / (c_matrix[0, 0] + c_matrix[0, 1])
    spec = c_matrix[1, 1] / (c_matrix[1, 1] + c_matrix[1, 0])

    return accuracy, sens, spec, f1s
