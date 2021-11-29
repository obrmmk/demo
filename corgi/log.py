import pandas as pd
import ast

def trans_log(PATH):
    json_open = open(PATH, 'r')
    json_load = json.load(json_open)
    df = pd.DataFrame()
    for i in json_load:
        df_tmp = pd.DataFrame(ast.literal_eval(i['text']).values(), index=ast.literal_eval(i['text']).keys()).T
        df = pd.concat([df, df_tmp], axis=0)
    df = df.reindex(columns=['session', 'index', 'input', 'translated'])
    return df