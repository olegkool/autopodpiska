import datetime
from typing import BinaryIO

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.metrics import roc_auc_score
import dill



def do_model() -> object:
    df = pd.read_csv('data/ga_innerjoin.csv.zip',
                     compression='zip',
                     encoding='utf-8',
                     index_col='session_id',
                     dtype={
                         'utm_keyword': 'object',
                         'device_os': 'object',
                         'device_model': 'object'
                     })

    # set random_state for all manipulations
    random_state = 42

    x = df.drop(columns=['target'])
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=random_state, stratify=y)

    # функция выравнивания количества в классах 0 и 1
    def class_balancer(a, b):
        df_merge = pd.merge(a, b, on='session_id', how='inner')
        df_work = shuffle(
            pd.concat(
                [df_merge[df_merge['target'] == 1],
                 df_merge[df_merge['target'] == 0].sample(
                     len(df_merge[df_merge['target'] == 1]
                         ),
                     random_state=random_state
                 )
                 ],
                axis=0
            ),
            random_state=random_state
        )
        return df_work

    x_test_bal = class_balancer(x_test, y_test).drop(columns=['target'])
    y_test_bal = class_balancer(x_test, y_test)['target']

    x_train_bal = class_balancer(x_train, y_train).drop(columns=['target'])
    y_train_bal = class_balancer(x_train, y_train)['target']

    # заполнение Nan, создание новых признаков
    def input_data_transform(x, y=None):
        import pandas as pd
        df = x.copy()

        def feature_engineering(frame):
            # feature engineering block
            frame['visit_date'] = pd.to_datetime(frame['visit_date'] + ' ' + frame['visit_time'])
            frame = frame.drop(columns='visit_time')
            frame['day_of_year'] = frame['visit_date'].dt.dayofyear
            frame['day_of_week'] = frame['visit_date'].dt.day_of_week
            frame['week_of_year'] = frame['visit_date'].dt.isocalendar().week
            frame['month'] = frame['visit_date'].dt.month
            frame['time_in_hours'] = frame.visit_date.dt.hour

            def change_part_day(item):
                if item < 6:
                    return 'night'
                elif item < 12:
                    return 'morning'
                elif item < 18:
                    return 'day'
                else:
                    return 'evening'

            frame['part_of_day'] = frame.visit_date.dt.hour.map(change_part_day)
            frame = frame.drop(columns='visit_date')
            # fillna block
            frame['utm_campaign'] = frame['utm_campaign'].fillna('No_campaign')
            frame['utm_adcontent'] = frame['utm_adcontent'].fillna('No_adcontent')
            frame['utm_keyword'] = frame['utm_keyword'].fillna('No_keyword')
            frame['utm_source'] = frame['utm_source'].fillna('No_source')
            frame = frame.drop(columns=['device_model', 'device_os'])
            frame['device_brand'] = frame['device_brand'].fillna('No_brand')
            frame = frame.drop(columns=['client_id'])
            # typing block
            category_cols = [
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
                'utm_keyword', 'device_category', 'device_brand',
                'device_screen_resolution', 'device_browser', 'geo_country',
                'geo_city', 'part_of_day'
            ]
            frame[category_cols] = frame[category_cols].astype('category')
            number_cols = [
                'day_of_year', 'week_of_year', 'time_in_hours', 'month',
                'day_of_week', 'visit_number'
            ]
            frame[number_cols] = frame[number_cols].astype('int32')

            return frame

        if y is not None:
            return feature_engineering(df), y
        else:
            return feature_engineering(df)

    catboost_model = CatBoostClassifier(
        iterations=2000,
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=random_state,
        early_stopping_rounds=50,
        learning_rate=0.07
    )

    train_pool = Pool(data=input_data_transform(x_train_bal), label=y_train_bal,
                      cat_features=(
                          'visit_number', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                          'device_category', 'device_brand', 'device_screen_resolution', 'device_browser',
                          'geo_country', 'geo_city', 'part_of_day', 'time_in_hours', 'month', 'day_of_week',
                          'day_of_year', 'week_of_year'
                      )
                      )
    test_pool = Pool(data=input_data_transform(x_test_bal), label=y_test_bal,
                     cat_features=(
                         'visit_number', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
                         'device_category', 'device_brand', 'device_screen_resolution', 'device_browser',
                         'geo_country', 'geo_city', 'part_of_day', 'time_in_hours', 'month', 'day_of_week',
                         'day_of_year', 'week_of_year'
                     )
                     )

    catboost_model.fit(train_pool, eval_set=test_pool)

    catboost_model_full = make_pipeline(
        FunctionTransformer(input_data_transform),
        catboost_model
    )

    catboost_auc: float = roc_auc_score(y_test_bal, catboost_model_full.predict_proba(x_test_bal)[:, 1])

    print('catboost_auc_full', catboost_auc)

    file: BinaryIO
    with open('model/catboost_model.pkl', 'wb') as file:
        dill.dump(
            {'model': catboost_model_full,
             'metadata': {'name': 'SberAutopodpiska_click_predict', 'author': 'Oleg Kulikov', 'version': '1.0',
                          'date': datetime.datetime.now(), 'type': str(catboost_model).split('.')[2].split(' ')[0],
                          'ROC_AUC': catboost_auc}},
            file
        )


if __name__ == '__main__':
    do_model()
