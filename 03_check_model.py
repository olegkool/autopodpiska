import json
import pandas as pd
import dill


def model_check() -> object:
    with open('model/catboost_model.pkl', 'rb') as file:
        model = dill.load(file)


    with open('data/01.json') as js:
        form = json.load(js)


    form_in_list = list()
    form_in_list.append(form)
    df = pd.DataFrame.from_dict(form_in_list)
    df.index = df['session_id']
    df = df.drop(columns='session_id')
    y = model['model'].predict(df)
    print(
        pd.DataFrame(
            dict(
                session_id=form['session_id'],
                predict_label=y
            )
        )
    )


if __name__ == '__main__':
    model_check()
