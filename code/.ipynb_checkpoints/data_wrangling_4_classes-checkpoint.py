import pandas as pd

def data_wrangling_4_classes(data_frames, names):
    '''
    Wrangles data for all 4 dataframes in one function. data_frames must be a list of dataframes.
    '''
    wrangled_df_list = []

    for df in data_frames:
        
        ball_4 = df[df.b_count == 4].index
        df.drop(ball_4, axis=0, inplace=True)
        df.replace(True, 1, inplace=True)
        
        df = df.merge(names, left_on='pitcher_id', right_on='id')
        
        df['pitch_type'].replace(['FF', 'FT', 'SI'], 'FB', inplace=True)
        df['pitch_type'].replace(['CU', 'FC', 'KC', 'SL', 'SC'], 'BB', inplace=True)
        df['pitch_type'].replace(['CH', 'EP', 'FS', 'KN'], 'OS', inplace=True)
        df['pitch_type'].replace(['FO', 'PO', 'IN', 'UN', 'FA'], 'OT', inplace=True)
    
        d = {i: 0 for i in df['ab_id']}
        p_in_ab_list = []
        
        for i in df['ab_id']:
            if i in d.keys():
                d[i] += 1
                p_in_ab_list.append(d[i])
                
        df['p_in_ab'] = p_in_ab_list
        
        wrangled_df_list.append(df)
    
    return wrangled_df_list