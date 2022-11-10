import pandas as pd
import json


def import_data():
    with open('data/URL_for_load.json', 'rb') as file:
        url_dict = json.load(file)

    ga_sessions_zip = url_dict['ga_sessions_zip']
    ga_hits_zip = url_dict['ga_hits_zip']

    # Функции пайплайна загрузки и обработки файла ga_hits для извлечения колонки target:

    def load_ga_hits(file):  # загружаем колонки session_id и event_action из ga_hits
        return pd.read_csv(file,
                           compression='zip',
                           usecols=['session_id', 'event_action'],
                           dtype={
                               'session_id': 'category',
                               'event_action': 'category'
                           })

    def create_target(df: object):  # Добавляем в датафрейм колонку target согласно списку целевых действий
        event_action_list = [
            'sub_car_claim_click', 'sub_car_claim_submit_click',
            'sub_open_dialog_click', 'sub_custom_question_submit_click',
            'sub_call_number_click', 'sub_callback_submit_click',
            'sub_submit_success', 'sub_car_request_submit_click'
        ]

        df['target'] = df.event_action \
            .isin(event_action_list) \
            .map({True: 1, False: 0})
        return df

    def groupby_ga_hits(df):  # группируем датафрейм по колонке session_id и выводим в target итог совершенных действий
        # внутри сессии (если была хотя бы одна 1 - получаем 1)
        return df.groupby(['session_id']) \
            .agg(target=pd.NamedAgg(column='target', aggfunc='max'))

    # Функции загрузки и обработки файла ga_sessions
    def load_ga_sessions(file):
        return pd.read_csv(file,
                           compression='zip',
                           dtype={
                               'session_id': 'category',
                               'client_id': 'category'
                           },
                           parse_dates=['visit_date'])

    # Функция сохранения в файл csv итогового датафрейма,
    # полученного объединением данных по столбцу session_id из файлов
    # ga_hits и ga_sessions
    def save_csv(df: object) -> object:
        df.to_csv('data/ga_innerjoin.csv.zip',
                  compression='zip',
                  encoding='utf-8',
                  index=False)
        return None

    pd.merge(
        left=load_ga_hits(ga_hits_zip)
        .pipe(create_target)
        .pipe(groupby_ga_hits),
        right=load_ga_sessions(ga_sessions_zip),
        on='session_id',
        how='inner')\
        .pipe(save_csv)


if __name__ == '__main__':
    import_data()
