import pandas as pd
import torch
import os
# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")
from datasets import get_benchmark_for_task
from FairPFN.scripts.transformer_prediction_interface.base import FairPFNClassifier
from FairPFN.scripts.tabular_metrics import roc_auc_score
from experiments.metrics import treatment_effect
from experiments.tasks_config import (
    FAIRPFN_CLASSIFIER,
    TABPFN_CLASSIFIER,
    PREDICTOR_NAMES_TO_FUNC,
    ROC_AUC_ERROR_METRIC,
    TREATMENT_EFFECT_METRIC
)

task_type = "fairness_multiclass"
if "datasets_dict" not in locals():
    datasets_dict = {}

datasets_dict[f"valid_{task_type}"], df = get_benchmark_for_task(
    task_type,
    split="valid",
    max_samples=10000,
    return_as_lists=False,
    sel=False
)

dataset_map = {}
for dataset in datasets_dict[f"valid_{task_type}"]:
    dataset_map[dataset.name] = dataset

fairpfn = FairPFNClassifier()

rows = []
causal_casestudies = ['Direct_effect', 'Indirect_effect', 'Indirect_effect_biased', 'Total_effect_level_one', 'Total_effect_level_two', 'Total_effect_level_three']
cntf_casestudies = ['Direct_effect_counterfactual', 'Indirect_effect_counterfactual', 'Indirect_effect_biased_counterfactual',  'Total_effect_level_one_counterfactual', 'Total_effect_level_two_counterfactual', 'Total_effect_level_three_counterfactual']
rows_base_ate_analysis = []

           
def ate_on_all_datasets(same_context: bool = True):
    predictors = [FAIRPFN_CLASSIFIER, TABPFN_CLASSIFIER]
    number_of_splits = 5
    number_of_datasets = 0
    for predictor in predictors:
        print(f"\nEvaluating predictor: {predictor}")
        for causal_casetudy in causal_casestudies:
            print(f"\nProcessing causal casestudy: {causal_casetudy}")
            for i in range(100):
                dataset = dataset_map[f'{causal_casetudy}_{i}']
                roc_auc_error = []
                for fold in range(1, number_of_splits + 1):
                    train_ds, test_ds = dataset.generate_valid_split(
                        n_splits=number_of_splits,
                        split_number=fold
                    )
                    if train_ds is None or test_ds is None:
                        print(f"Skipping fold {fold} for dataset {causal_casetudy}_{i} due to invalid split.")
                        continue

                    X_train_data = torch.tensor(train_ds.dowhy_data['df'].drop(columns=['y'] + dataset.fair_unobservables).values).float()
                    X_test_data = torch.tensor(test_ds.dowhy_data['df'].drop(columns=['y'] + dataset.fair_unobservables).values).float()
                    X_data = torch.cat([X_train_data, X_test_data], dim=0)

                    Y_train_data = torch.tensor(train_ds.dowhy_data['df']['y'].values).float()
                    Y_test_data = torch.tensor(test_ds.dowhy_data['df']['y'].values).float()
                    Y_data = torch.cat([Y_train_data, Y_test_data], dim=0)
                    obs_dataset = torch.cat([X_data, Y_data.unsqueeze(1)], dim=1)

                    pred_func = PREDICTOR_NAMES_TO_FUNC[predictor]
                    result = pred_func(obs_dataset.unsqueeze(0), train_size=X_train_data.shape[0])
                    y_probs = result.preds.probs[0].cpu().numpy()
                    auc = roc_auc_score(Y_test_data.numpy(), y_probs[:, 1])
                    roc_error = 1 - auc
                    roc_auc_error.append(roc_error)

                roc_error_mean = sum(roc_auc_error) / len(roc_auc_error)
                roc_error_var = sum((x - roc_error_mean) ** 2 for x in roc_auc_error) / len(roc_auc_error)
                roc_error_std_err = (roc_error_var ** 0.5) / (len(roc_auc_error) ** 0.5)

                rows.append({'case_study': causal_casetudy, 'dataset': f'{causal_casetudy}_{i}', 'predictor': predictor, 'metric': ROC_AUC_ERROR_METRIC, 'value': [round(roc_error_mean, 3)], 'case_type_Y': 'binary'})
                rows.append({'case_study': causal_casetudy, 'dataset': f'{causal_casetudy}_{i}', 'predictor': predictor, 'metric': 'Var_' + ROC_AUC_ERROR_METRIC, 'value': [round(roc_error_var, 6)], 'case_type_Y': 'binary'})
                rows.append({'case_study': causal_casetudy, 'dataset': f'{causal_casetudy}_{i}', 'predictor': predictor, 'metric': 'Std_Err_' + ROC_AUC_ERROR_METRIC, 'value': [round(roc_error_std_err, 6)], 'case_type_Y': 'binary'})
                number_of_datasets += 1
                print(f"Processed dataset {number_of_datasets}: {causal_casetudy}_{i}")

        number_of_datasets = 0
        for cntf_casestudy in cntf_casestudies:
            print(f"\nProcessing counterfactual casestudy: {cntf_casestudy}")
            for i in range(100):
                dataset = dataset_map[f"{cntf_casestudy}_{i}"]
                ates = []
                base_ates = []
                roc_errors_first_cft = []
                roc_errors_second_cft = []
                for fold in range(1, number_of_splits + 1):
                    train_ds, test_ds = dataset.generate_valid_split(
                        n_splits=number_of_splits,
                        split_number=fold
                    )
                    if train_ds is None or test_ds is None:
                        print(f"Skipping fold {fold} for dataset {cntf_casestudy}_{i} due to invalid split.")
                        continue

                    X_train_data = torch.tensor(train_ds.dowhy_data['df'].drop(columns=['y'] + dataset.fair_unobservables).values).float()
                    X_test_data = torch.tensor(test_ds.dowhy_data['df'].drop(columns=['y'] + dataset.fair_unobservables).values).float()
                    X_data = torch.cat([X_train_data, X_test_data], dim=0)

                    Y_train_data = torch.tensor(train_ds.dowhy_data['df']['y'].values).float()
                    Y_test_data = torch.tensor(test_ds.dowhy_data['df']['y'].values).float()
                    Y_data = torch.cat([Y_train_data, Y_test_data], dim=0)
                    first_cft_data = torch.cat([X_data, Y_data.unsqueeze(1)], dim=1)
                    train_size = X_train_data.shape[0]
                    pred_func = PREDICTOR_NAMES_TO_FUNC[predictor]
                    result_1 = pred_func(first_cft_data.unsqueeze(0), train_size=train_size)

                    y_probs_1 = result_1.preds.probs[0].cpu().numpy()
                    auc_1 = roc_auc_score(Y_test_data.numpy(), y_probs_1[:, 1])
                    roc_error_1 = 1 - auc_1
                    roc_errors_first_cft.append(roc_error_1)

                    X_cft_train_data = None
                    if same_context:
                        X_cft_train_data = X_train_data.clone()
                    else:
                        X_cft_train_data = torch.tensor(train_ds.dowhy_data['df_cntf'].drop(columns=['y'] + dataset.fair_unobservables).values).float()

                    X_cft_test_data = torch.tensor(test_ds.dowhy_data['df_cntf'].drop(columns=['y'] + dataset.fair_unobservables).values).float()
                    X_cft_data = torch.cat([X_cft_train_data, X_cft_test_data], dim=0)

                    Y_cft_train_data = torch.tensor(train_ds.dowhy_data['df_cntf']['y'].values).float()
                    Y_cft_test_data = torch.tensor(test_ds.dowhy_data['df_cntf']['y'].values).float()
                    Y_cft_data = torch.cat([Y_cft_train_data, Y_cft_test_data], dim=0)
                    second_cft_data = torch.cat([X_cft_data, Y_cft_data.unsqueeze(1)], dim=1)
                    pred_func = PREDICTOR_NAMES_TO_FUNC[predictor]
                    result_2 = pred_func(second_cft_data.unsqueeze(0), train_size=train_size)
                    y_probs_2 = result_2.preds.probs[0].cpu().numpy()
                    auc_2 = roc_auc_score(Y_cft_test_data.numpy(), y_probs_2[:, 1])
                    roc_error_2 = 1 - auc_2
                    roc_errors_second_cft.append(roc_error_2)

                    prot_attr_test_obs = X_test_data[:, 0]
                    prot_attr_test_cft = X_cft_test_data[:, 0]
                    if torch.any(prot_attr_test_obs == prot_attr_test_cft):
                        print("Warning: There are some matching protected attribute values between observational and counterfactual test sets.")
                    # get indices of Y where prot_attr is 0
                    indices_do_0_obs = (prot_attr_test_obs == 0).nonzero(as_tuple=True)[0]
                    if len(indices_do_0_obs) == 0:
                        print("Warning: No instances with protected attribute value 0 in the observational test set.")
                    indices_do_1_obs = (prot_attr_test_obs == 1).nonzero(as_tuple=True)[0]
                    if len(indices_do_1_obs) == 0:
                        print("Warning: No instances with protected attribute value 1 in the observational test set.")
                    indices_do_0_cft = (prot_attr_test_cft == 0).nonzero(as_tuple=True)[0]
                    if len(indices_do_0_cft) == 0:
                        print("Warning: No instances with protected attribute value 0 in the counterfactual test set.")
                    indices_do_1_cft = (prot_attr_test_cft == 1).nonzero(as_tuple=True)[0]
                    if len(indices_do_1_cft) == 0:
                        print("Warning: No instances with protected attribute value 1 in the counterfactual test set.")
                    # calculate base ATE
                    y_base_do_0 = torch.cat([Y_test_data[indices_do_0_obs], Y_cft_test_data[indices_do_0_cft]], dim=0)
                    y_base_do_1 = torch.cat([Y_test_data[indices_do_1_obs], Y_cft_test_data[indices_do_1_cft]], dim=0)
                    base_ate = torch.mean(y_base_do_1, dim=0) - torch.mean(y_base_do_0, dim=0)
                    base_ates.append(base_ate.item())

                    # calculate ATE as E[Y(1)] - E[Y(0)]
                    y_preds_do_0 = torch.cat([result_1.preds.values[0][indices_do_0_obs], result_2.preds.values[0][indices_do_0_cft]], dim=0)
                    y_preds_do_1 = torch.cat([result_1.preds.values[0][indices_do_1_obs], result_2.preds.values[0][indices_do_1_cft]], dim=0)
                    if len(y_preds_do_0) == 0 or len(y_preds_do_1) == 0:
                        print("Warning: Cannot compute ATE due to lack of predictions for one of the groups.")
                    result_1.preds.values[0] = y_preds_do_0
                    result_2.preds.values[0] = y_preds_do_1
                    ate = treatment_effect([result_1, result_2])
                    ate_equivalent = torch.mean(y_preds_do_1, dim=0) - torch.mean(y_preds_do_0, dim=0)
                    if not torch.isclose(torch.tensor(ate[0]), ate_equivalent, atol=1e-5):
                        print(f"Warning: ATE calculation mismatch: treatment_effect {ate[0]} vs direct calculation {ate_equivalent.item()}")
                    ates.append(ate[0])
                mean_roc_error_first_cft = sum(roc_errors_first_cft) / len(roc_errors_first_cft)
                var_roc_error_first_cft = sum((x - mean_roc_error_first_cft) ** 2 for x in roc_errors_first_cft) / len(roc_errors_first_cft)
                std_err_roc_error_first_cft = (var_roc_error_first_cft ** 0.5) / (len(roc_errors_first_cft) ** 0.5)

                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'First_CFT_' + ROC_AUC_ERROR_METRIC, 'value': [round(mean_roc_error_first_cft, 3)], 'case_type_Y': 'binary'})
                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'Var_First_CFT_' + ROC_AUC_ERROR_METRIC, 'value': [round(var_roc_error_first_cft, 6)], 'case_type_Y': 'binary'})
                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'Std_Err_First_CFT_' + ROC_AUC_ERROR_METRIC, 'value': [round(std_err_roc_error_first_cft, 6)], 'case_type_Y': 'binary'})

                mean_roc_error_second_cft = sum(roc_errors_second_cft) / len(roc_errors_second_cft)
                var_roc_error_second_cft = sum((x - mean_roc_error_second_cft) ** 2 for x in roc_errors_second_cft) / len(roc_errors_second_cft)
                std_err_roc_error_second_cft = (var_roc_error_second_cft ** 0.5) / (len(roc_errors_second_cft) ** 0.5)

                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'Second_CFT_' + ROC_AUC_ERROR_METRIC, 'value': [round(mean_roc_error_second_cft, 3)], 'case_type_Y': 'binary'})
                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'Var_Second_CFT_' + ROC_AUC_ERROR_METRIC, 'value': [round(var_roc_error_second_cft, 6)], 'case_type_Y': 'binary'})
                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'Std_Err_Second_CFT_' + ROC_AUC_ERROR_METRIC, 'value': [round(std_err_roc_error_second_cft, 6)], 'case_type_Y': 'binary'})

                mean_base_ate = sum(base_ates) / len(base_ates)
                var_base_ate = sum((x - mean_base_ate) ** 2 for x in base_ates) / len(base_ates)
                std_err_base_ate = (var_base_ate ** 0.5) / (len(base_ates) ** 0.5)

                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'Base_ATE', 'value': [round(mean_base_ate, 3)], 'case_type_Y': 'binary'})
                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'Var_Base_ATE', 'value': [round(var_base_ate, 6)], 'case_type_Y': 'binary'})
                rows_base_ate_analysis.append({'case_study': cntf_casestudy, 'dataset': f'{cntf_casestudy}_{i}', 'predictor': predictor, 'metric': 'Std_Err_Base_ATE', 'value': [round(std_err_base_ate, 6)], 'case_type_Y': 'binary'})

                mean_ate = sum(ates) / len(ates)
                var_ate = sum((x - mean_ate) ** 2 for x in ates) / len(ates)
                std_err_ate = (var_ate ** 0.5) / (len(ates) ** 0.5)
                # remove '_counterfactual' from cntf_casestudy to get the original casestudy name
                original_casestudy = cntf_casestudy.replace('_counterfactual', '')
                rows.append({'case_study': original_casestudy, 'dataset': f'{original_casestudy}_{i}', 'predictor': predictor, 'metric': TREATMENT_EFFECT_METRIC, 'value': [round(mean_ate, 3)], 'case_type_Y': 'binary'})
                rows.append({'case_study': original_casestudy, 'dataset': f'{original_casestudy}_{i}', 'predictor': predictor, 'metric': 'Var_ATE', 'value': [round(var_ate, 6)], 'case_type_Y': 'binary'})
                rows.append({'case_study': original_casestudy, 'dataset': f'{original_casestudy}_{i}', 'predictor': predictor, 'metric': 'Std_Err_ATE', 'value': [round(std_err_ate, 6)], 'case_type_Y': 'binary'})
                number_of_datasets += 1
                print(f"Processed dataset {number_of_datasets}: {cntf_casestudy}_{i}")

    print(f"Total number of datasets processed: {number_of_datasets*2}")
    return rows, rows_base_ate_analysis


rows, rows_base_ate_analysis = ate_on_all_datasets(same_context=True)
df = pd.DataFrame(rows)
df_base_ate_analysis = pd.DataFrame(rows_base_ate_analysis)
if not os.path.exists('data/csv/fairpfn_results_same_context_2nd_exp.csv'):
    df.to_csv('data/csv/fairpfn_results_same_context_2nd_exp.csv', index=False)
else:
    df_existing = pd.read_csv('data/csv/fairpfn_results_same_context_2nd_exp.csv')
    df_combined = pd.concat([df_existing, df], ignore_index=True)
    df_combined.to_csv('data/csv/fairpfn_results_same_context_2nd_exp.csv', index=False)

if not os.path.exists('data/csv/fairpfn_base_ate_analysis_same_context.csv'):
    df_base_ate_analysis.to_csv('data/csv/fairpfn_base_ate_analysis_same_context.csv', index=False)
else:
    df_existing = pd.read_csv('data/csv/fairpfn_base_ate_analysis_same_context.csv')
    df_combined = pd.concat([df_existing, df_base_ate_analysis], ignore_index=True)
    df_combined.to_csv('data/csv/fairpfn_base_ate_analysis_same_context.csv', index=False)


# create a pandas DataFrame to store results with columns: Dataset, function_used, ATE, AUC
# number_of_not_identical_X = 0
# number_of_not_identical_Y = 0
# for causal_casetudy, cntf_casestudy in zip(causal_casestudies, cntf_casestudies):
#     print(f"\nProcessing causal casestudy: {causal_casetudy} and counterfactual casestudy: {cntf_casestudy}")
#     for i in range(100):
#         number_of_datasets += 1
#         cft_results = []
#         dataset = dataset_map[f'{causal_casetudy}_{i}']
#         n = dataset.x.shape[0]
#         # print(f"Dataset attribute names: {dataset.attribute_names}")
#         # print(f"n is: {n}")
#         indices = torch.randperm(n)
#         train_size = int(0.7 * n)
#         train_indices = indices[:train_size].tolist()
#         test_indices = indices[train_size:].tolist()
#         splits = [(train_indices, test_indices)]
#         train_ds, test_ds = dataset.generate_valid_split(splits=splits, split_number=1)
#         # print(f"Shape of train_ds.x: {train_ds.x.shape}, train_ds.y: {train_ds.y.shape}")
#         # print(f"Shape of test_ds.x: {test_ds.x.shape}, test_ds.y: {test_ds.y.shape}")
#         # print(f"train_ds.dowhy_data.keys(): ", train_ds.dowhy_data.keys())
#         # print head of dowhy_data['df']
#         # print(f"train_ds.dowhy_data['df'].head(): \n", train_ds.dowhy_data['df'].head())
#         X_1 = torch.cat([train_ds.x, test_ds.x], dim=0)
#         Y_1 = torch.cat([train_ds.y, test_ds.y], dim=0)
#         dataset = torch.cat([X_1, Y_1.unsqueeze(1)], dim=1)
#         results = fairpfn_predictor(dataset.unsqueeze(0), train_size=train_ds.x.shape[0])
#         y_pred = results.preds.values[0].cpu().numpy()
#         ate_cfte = causal_fairness_total_effect(target=test_ds.y, pred=results.preds.probs[0].cpu().numpy(), x=test_ds.x, prot_attr=test_ds.x[:, 0], name=test_ds.name, dowhy_data=test_ds.dowhy_data)
#         auc = roc_auc_score(test_ds.y, y_pred)
#         rows.append({'Dataset': f'{causal_casetudy}_{i}', 'Metric': 'ATE_CFTE', 'Value': round(ate_cfte, 3)})
#         rows.append({'Dataset': f'{causal_casetudy}_{i}', 'Metric': 'AUC', 'Value': round(auc, 3)})
#         cft_results.append(results)

#         dataset_cft = dataset_map[f'{cntf_casestudy}_{i}']
#         print(f"Observational ds name: {dataset_map[f'{causal_casetudy}_{i}'].name}, Counterfactual ds name: {dataset_map[f'{cntf_casestudy}_{i}'].name}")
#         train_ds_cft, test_ds_cft = dataset_cft.generate_valid_split(splits=splits, split_number=1)
#         print(f"pointers of observational: {train_ds.x.data_ptr()}, {test_ds.x.data_ptr()} and counterfactual: {train_ds_cft.x.data_ptr()}, {test_ds_cft.x.data_ptr()}")
#         # print(f"Shape of train_ds_cft.x: {train_ds_cft.x.shape}, train_ds_cft.y: {train_ds_cft.y.shape}")
#         # print(f"Shape of test_ds_cft.x: {test_ds_cft.x.shape}, test_ds_cft.y: {test_ds_cft.y.shape}")
#         # print(f"train_ds_cft.dowhy_data.keys(): ", train_ds_cft.dowhy_data.keys())
#         # print(f"train_ds_cft.dowhy_data['df'].head(): \n", train_ds_cft.dowhy_data['df'].head())
#         X_2 = torch.cat([train_ds_cft.x, test_ds_cft.x], dim=0)
#         Y_2 = torch.cat([train_ds_cft.y, test_ds_cft.y], dim=0)
#         dataset_cft = torch.cat([X_2, Y_2.unsqueeze(1)], dim=1)
#         # check how many of the protected attribute column values are the same
#         num_same = (X_1[:, 0] == X_2[:, 0]).sum().item()
#         # print(f"Number of same values in protected attribute column: {num_same} out of {n}, which is {num_same/n*100:.2f}%")

#         if not torch.equal(X_1, X_2):
#             # print("The original and counterfactual X do not match!")
#             number_of_not_identical_X += 1
#         if not torch.equal(Y_1, Y_2):
#             # print("The original and counterfactual Y do not match!")
#             number_of_not_identical_Y += 1

#         # for col in range(X_1.shape[1]):
#         #     unique_1, counts_1 = torch.unique(X_1[:, col], return_counts=True)
#         #     unique_2, counts_2 = torch.unique(X_2[:, col], return_counts=True)
#         #     print(f"Column {col}: Dataset 1 unique values and counts: {dict(zip(unique_1.tolist(), counts_1.tolist()))}")
#         #     print(f"Column {col}: Dataset 2 unique values and counts: {dict(zip(unique_2.tolist(), counts_2.tolist()))}")

#         results_cft = fairpfn_predictor(dataset_cft.unsqueeze(0), train_size=train_ds_cft.x.shape[0])
#         cft_results.append(results_cft)
#         y_pred_cft = results_cft.preds.values[0].cpu().numpy()
#         ate_cfte_cft = causal_fairness_total_effect(target=test_ds_cft.y, pred=results_cft.preds.probs[0].cpu().numpy(), x=test_ds_cft.x, prot_attr=test_ds_cft.x[:, 0], name=test_ds_cft.name, dowhy_data=test_ds_cft.dowhy_data)
#         rows.append({'Dataset': f'{cntf_casestudy}_{i}', 'Metric': 'ATE_CFTE', 'Value': round(ate_cfte_cft, 3)})
#         auc = roc_auc_score(test_ds_cft.y, y_pred_cft)
#         rows.append({'Dataset': f'{cntf_casestudy}_{i}', 'Metric': 'AUC', 'Value': round(auc, 3)})
#         ate = treatment_effect(cft_results)
#         rows.append({'Dataset': f'{causal_casetudy}_{i}', 'Metric': 'ATE_treatment_effect', 'Value': round(ate[0], 3)})


    # print(f"Dataset attribute names: {dataset.attribute_names}")
        # train_ds, test_ds = dataset.generate_valid_split(n_splits=2)
        # print(f"Shape of dataset: {dataset.x.shape}, train_ds.x: {train_ds.x.shape}, train_ds.y: {train_ds.y.shape}")

        # print(f"train_ds.dowhy_data.keys(): ", train_ds.dowhy_data.keys())
        # print head of dowhy_data['df']
        # print(f"train_ds.dowhy_data['df'].head(): \n", train_ds.dowhy_data['df'].head())
        # X_1 = torch.cat([train_ds.x, test_ds.x], dim=0)
        # Y_1 = torch.cat([train_ds.y, test_ds.y], dim=0)
        # dataset = torch.cat([X_1, Y_1.unsqueeze(1)], dim=1)
        # results = fairpfn_predictor(dataset.unsqueeze(0), train_size=train_ds.x.shape[0])
        # y_pred = results.preds.values[0].cpu().numpy()
        # ate_cfte = causal_fairness_total_effect(target=test_ds.y, pred=results.preds.probs[0].cpu().numpy(), x=test_ds.x, prot_attr=test_ds.x[:, 0], name=test_ds.name, dowhy_data=test_ds.dowhy_data)
        # auc = roc_auc_score(test_ds.y, y_pred)
        # rows.append({'Dataset': f'{causal_casetudy}_{i}', 'Metric': 'ATE_CFTE', 'Value': round(ate_cfte, 3)})
        # rows.append({'Dataset': f'{causal_casetudy}_{i}', 'Metric': 'AUC', 'Value': round(auc, 3)})
        # cft_results.append(results)

        # dataset_cft = dataset_map[f'{cntf_casestudy}_{i}']
        # print(f"Observational ds name: {dataset_map[f'{causal_casetudy}_{i}'].name}, Counterfactual ds name: {dataset_map[f'{cntf_casestudy}_{i}'].name}")
        # train_ds_cft, test_ds_cft = dataset_cft.generate_valid_split(splits=splits, split_number=1)
        # print(f"pointers of observational: {train_ds.x.data_ptr()}, {test_ds.x.data_ptr()} and counterfactual: {train_ds_cft.x.data_ptr()}, {test_ds_cft.x.data_ptr()}")
        # print(f"Shape of train_ds_cft.x: {train_ds_cft.x.shape}, train_ds_cft.y: {train_ds_cft.y.shape}")
        # print(f"Shape of test_ds_cft.x: {test_ds_cft.x.shape}, test_ds_cft.y: {test_ds_cft.y.shape}")
        # print(f"train_ds_cft.dowhy_data.keys(): ", train_ds_cft.dowhy_data.keys())
        # print(f"train_ds_cft.dowhy_data['df'].head(): \n", train_ds_cft.dowhy_data['df'].head())
        # X_2 = torch.cat([train_ds_cft.x, test_ds_cft.x], dim=0)
        # Y_2 = torch.cat([train_ds_cft.y, test_ds_cft.y], dim=0)
        # dataset_cft = torch.cat([X_2, Y_2.unsqueeze(1)], dim=1)
        # check how many of the protected attribute column values are the same
        # num_same = (X_1[:, 0] == X_2[:, 0]).sum().item()
        # print(f"Number of same values in protected attribute column: {num_same} out of {n}, which is {num_same/n*100:.2f}%")

        # if not torch.equal(X_1, X_2):
            # print("The original and counterfactual X do not match!")
            # number_of_not_identical_X += 1
        # if not torch.equal(Y_1, Y_2):
            # print("The original and counterfactual Y do not match!")
            # number_of_not_identical_Y += 1

        # for col in range(X_1.shape[1]):
        #     unique_1, counts_1 = torch.unique(X_1[:, col], return_counts=True)
        #     unique_2, counts_2 = torch.unique(X_2[:, col], return_counts=True)
        #     print(f"Column {col}: Dataset 1 unique values and counts: {dict(zip(unique_1.tolist(), counts_1.tolist()))}")
        #     print(f"Column {col}: Dataset 2 unique values and counts: {dict(zip(unique_2.tolist(), counts_2.tolist()))}")

        # results_cft = fairpfn_predictor(dataset_cft.unsqueeze(0), train_size=train_ds_cft.x.shape[0])
        # cft_results.append(results_cft)
        # y_pred_cft = results_cft.preds.values[0].cpu().numpy()
        # ate_cfte_cft = causal_fairness_total_effect(target=test_ds_cft.y, pred=results_cft.preds.probs[0].cpu().numpy(), x=test_ds_cft.x, prot_attr=test_ds_cft.x[:, 0], name=test_ds_cft.name, dowhy_data=test_ds_cft.dowhy_data)
        # rows.append({'Dataset': f'{cntf_casestudy}_{i}', 'Metric': 'ATE_CFTE', 'Value': round(ate_cfte_cft, 3)})
        # auc = roc_auc_score(test_ds_cft.y, y_pred_cft)
        # rows.append({'Dataset': f'{cntf_casestudy}_{i}', 'Metric': 'AUC', 'Value': round(auc, 3)})
        # ate = treatment_effect(cft_results)
        # rows.append({'Dataset': f'{causal_casetudy}_{i}', 'Metric': 'ATE_treatment_effect', 'Value': round(ate[0], 3)})
        # print(f"Number of datasets where original and counterfactual X do not match: {number_of_not_identical_X} out of {number_of_datasets}, which is {number_of_not_identical_X/number_of_datasets*100:.2f}%")
        # print(f"Number of datasets where original and counterfactual Y do not match: {number_of_not_identical_Y} out of {number_of_datasets}, which is {number_of_not_identical_Y/number_of_datasets*100:.2f}%")
