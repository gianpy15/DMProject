import numpy as np


def export_sub(sub_base_structure, predictions, group_rows=False):
    predictions = predictions.flatten()
    sub_base_structure['SPEED_AVG'] = predictions
    if group_rows:
        print('the predictions for the same datetime road and sensor are computed with the mean')

        sub_base_structure = sub_base_structure[['KEY', 'KM', 'DATETIME_UTC', 'PREDICTION_STEP', 'SPEED_AVG']]\
            .groupby(['KEY', 'KM', 'DATETIME_UTC', 'PREDICTION_STEP'], as_index=False).mean()
    else:
        sub_base_structure = sub_base_structure[['KEY', 'KM', 'DATETIME_UTC', 'PREDICTION_STEP', 'SPEED_AVG']]
    return sub_base_structure


def hybrid_score(dict_scores):
    """
    dictionary containing

    dict['bs']=[bs1, bs2,...., bsn] --> the prediction in base structure of N models

    dict['weights'] = [a1, a2,...., an] --> the weights to give to the N models

    :return: bs structure resulting from the hybridation of the N models
    """
    models = dict_scores['bs']
    weights = dict_scores['weights']

    assert len(models) == len(weights), 'NOTE LEN OF MODELS AND WEIGHTS DO NOT MATCH!'
    assert np.sum(np.array(weights)) == 1, 'NOTE THE WEIGHTS DO NOT SUM TO 1 !'

    hybrid_speed_avg = None
    count = 0
    for m in models:
        if hybrid_speed_avg is None:
            hybrid_speed_avg = (m['SPEED_AVG'].values)*weights[count]
        else:
            hybrid_speed_avg = np.sum([hybrid_speed_avg, m['SPEED_AVG'].values*weights[count]], axis=0)
        count+=1

    # do the average and assign it to the first model it will be the hybrid model
    models[0]['SPEED_AVG']=hybrid_speed_avg
    return models[0]


def compute_MAE(bd_predictions, y):
    y = y.values.flatten()
    bd_predictions['y'] = y
    # filter the none y rows
    bd_predictions=bd_predictions[bd_predictions['y'].notnull()]
    bd_predictions['MAE']=abs(bd_predictions['y']-bd_predictions['SPEED_AVG'])
    MAE = np.mean(bd_predictions['MAE'].values)
    print(f'MAE is:{MAE}')
    return MAE