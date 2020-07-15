import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bokeh.io import output_notebook, show, curdoc
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, Axis, Dropdown
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import widgetbox, row

def get_df_p_in_ab(target_variable, lst):
    
    df_2015, df_2016, df_2017, df_2018 = lst
    
    dic_2015 = {'{}_2015'.format(a): [df_2015[(df_2015['pitch_type'] == i) & (df_2015[target_variable] == j)]
                                       ['pitch_type'].count() 
                                       for j in list(df_2015[target_variable].unique()) 
                                       for i in list(df_2015['pitch_type'].unique()) if i == a] 
                                       for a in list(df_2015['pitch_type'].unique())}
    dic_2016 = {'{}_2016'.format(a): [df_2016[(df_2016['pitch_type'] == i) & (df_2016[target_variable] == j)]
                                       ['pitch_type'].count() 
                                       for j in list(df_2016[target_variable].unique()) 
                                       for i in list(df_2016['pitch_type'].unique()) if i == a] 
                                       for a in list(df_2016['pitch_type'].unique())}
    dic_2017 = {'{}_2017'.format(a): [df_2017[(df_2017['pitch_type'] == i) & (df_2017[target_variable] == j)]
                                       ['pitch_type'].count() 
                                       for j in list(df_2017[target_variable].unique()) 
                                       for i in list(df_2017['pitch_type'].unique()) if i == a] 
                                       for a in list(df_2017['pitch_type'].unique())}
    dic_2018 = {'{}_2018'.format(a): [df_2018[(df_2018['pitch_type'] == i) & (df_2018[target_variable] == j)]
                                       ['pitch_type'].count() 
                                       for j in list(df_2018[target_variable].unique()) 
                                       for i in list(df_2018['pitch_type'].unique()) if i == a] 
                                       for a in list(df_2018['pitch_type'].unique())}

    df_2015_count = pd.DataFrame({'count': df_2015[target_variable].unique(), 
                                  'year': [2015] * len(df_2015[target_variable].unique())
                                 })
    for x in [i[:2] for i in dic_2015.keys()]:
        df_2015_count['number_of_{}'.format(x)] = dic_2015['{}_2015'.format(x)]

    df_2016_count = pd.DataFrame({'count': df_2016[target_variable].unique(), 
                                  'year': [2016] * len(df_2016[target_variable].unique())
                                 })
    for x in [i[:2] for i in dic_2016.keys()]:
        df_2016_count['number_of_{}'.format(x)] = dic_2016['{}_2016'.format(x)]

    df_2017_count = pd.DataFrame({'count': df_2017[target_variable].unique(), 
                                  'year': [2017] * len(df_2017[target_variable].unique())
                                 })
    for x in [i[:2] for i in dic_2017.keys()]:
        df_2017_count['number_of_{}'.format(x)] = dic_2017['{}_2017'.format(x)]

    df_2018_count = pd.DataFrame({'count': df_2018[target_variable].unique(), 
                                  'year': [2018] * len(df_2018[target_variable].unique())
                                 })
    for x in [i[:2] for i in dic_2018.keys()]:
        df_2018_count['number_of_{}'.format(x)] = dic_2018['{}_2018'.format(x)]

    df = df_2015_count.append([df_2016_count, df_2017_count, df_2018_count], sort=True)
    df = df.reset_index().drop('index', axis=1)
        
    for col in [i[-2:] for i in list(df.columns[1:5])]:
        df['percent_{}'.format(col)] = (df['number_of_{}'.format(col)] / 
                                        df.iloc[:,1:5].sum(axis=1) * 100)
    return df

def get_df(target_df, target_variable, **kwargs):
    
    years = list(target_df['year'].unique())
    
    if kwargs:
        kwarg_list = list(kwargs.items())
        df = target_df[(target_df[kwarg_list[0][0]] == kwarg_list[0][1]) & 
                         (target_df[kwarg_list[1][0]] == kwarg_list[1][1])]
        
        full_dic = {'dic_{}'.format(year): {'{}_{}'.format(p, year): [df[(df['year'] == year) & (df['pitch_type'] == i) & 
                                                                         (df[target_variable] == j)]['pitch_type'].count() 
                                           for j in list(df[target_variable].unique()) 
                                           for i in list(df['pitch_type'].unique()) if i == p] 
                                           for p in list(df['pitch_type'].unique())} for year in years}
    else:
        full_dic = {'dic_{}'.format(year): {'{}_{}'.format(p, year): 
                                            [target_df[(target_df['year'] == year) & (target_df['pitch_type'] == i) & 
                                                         (target_df[target_variable] == j)]['pitch_type'].count() 
                                           for j in list(target_df[target_variable].unique()) 
                                           for i in list(target_df['pitch_type'].unique()) if i == p] 
                                           for p in list(target_df['pitch_type'].unique())} for year in years}
    df_list = []
    for year in years:
        df_count = pd.DataFrame({'count': target_df[target_df['year'] == year][target_variable].unique(), 
                                 'year': [year] * len(target_df[target_df['year'] == year][target_variable].unique())
                               })
        
        for x in [i[:2] for i in full_dic['dic_{}'.format(year)].keys()]:
            df_count['number_of_{}'.format(x)] = full_dic['dic_{}'.format(year)]['{}_{}'.format(x, year)]
            
        df_list.append(df_count)

    df = pd.concat(df_list).reset_index().drop('index', axis=1)
    
    for col in [i[-2:] for i in list(df.columns[2:])]:
            df['percent_{}'.format(col)] = (df['number_of_{}'.format(col)] / 
                                            df.iloc[:,2:6].sum(axis=1) * 100)
            
    return df

def get_plots(target_df, target_variable, **kwargs):

    if target_variable == 'p_in_ab' or target_variable == 'inning':
        df = get_df_p_in_ab(target_variable, **kwargs)
    else:
        df = get_df(target_df, target_variable, **kwargs)
    
    p_tab_list = []
    for year in df.year.unique():
        p = figure(x_range=[str(i) for i in list(df[df.year == year]['count'].unique())])
        p.vbar_stack(['number_of_FB', 
                      'number_of_BB', 
                      'number_of_OS', 
                      'number_of_OT'],
                      x='count',
                      width=0.8,
                      color=('red', 'blue', 'green', 'yellow'),
                      source=df[df.year == year],
                      legend_label = ['Fastballs', 'Breaking balls', 'Offspeed', 'Other'])
        yaxis = p.select(dict(type=Axis, layout="left"))[0]
        yaxis.formatter.use_scientific = False
        hover = HoverTool(tooltips=[(target_variable, '@count'),
                                    ('Number of Fastballs', '@number_of_FB'), 
                                    ('Number of Breaking Balls', '@number_of_BB'),
                                    ('Number of Offspeed', '@number_of_OS'),
                                    ('Number of Other', '@number_of_OT')])
        p.add_tools(hover)
        p_tab_list.append(p)
        p.xaxis.axis_label = target_variable
        p.yaxis.axis_label = 'Number of pitches'
        if len(kwargs) > 1:
            kwarg_list = list(kwargs.items())
            p.title.text = '{} for {} {}'.format(target_variable, kwarg_list[0][1], kwarg_list[1][1])
        else:
            p.title.text = '{} for all pitchers'.format(target_variable)

    tab1 = Panel(child=p_tab_list[0], title='2015')
    tab2 = Panel(child=p_tab_list[1], title='2016')
    tab3 = Panel(child=p_tab_list[2], title='2017')
    tab4 = Panel(child=p_tab_list[3], title='2018')

    layout1 = Tabs(tabs=[tab1, tab2, tab3, tab4])
    output_notebook()
    
    pp_tab_list = []
    for year in df.year.unique():
        pp = figure(x_range=[str(i) for i in list(df[df.year == year]['count'].unique())])
        pp.vbar_stack(['percent_FB', 
                       'percent_BB', 
                       'percent_OS', 
                       'percent_OT'],
                      x='count',
                      width=0.8,
                      color=('red', 'blue', 'green', 'yellow'),
                      source=df[df.year == year],
                      legend_label = ['Fastballs', 'Breaking Balls', 'Offspeed', 'Other'])
        yaxis = pp.select(dict(type=Axis, layout="left"))[0]
        yaxis.formatter.use_scientific = False
        hover = HoverTool(tooltips=[(target_variable, '@count'),
                                    ('Percent Fastballs', '@percent_FB'), 
                                    ('Percent Breaking Balls', '@percent_BB'),
                                    ('Percent Offspeed', '@percent_OS'),
                                    ('Percent Other', '@percent_OT')])
        pp.add_tools(hover)
        pp_tab_list.append(pp)
        pp.xaxis.axis_label = target_variable
        pp.yaxis.axis_label = 'Percent of pitches'
        
        if len(kwargs) > 1:
            kwarg_list = list(kwargs.items())
            pp.title.text = '{} for {} {}'.format(target_variable, kwarg_list[0][1], kwarg_list[1][1])
        else:
            pp.title.text = '{} for all pitchers'.format(target_variable)

    tab1 = Panel(child=pp_tab_list[0], title='2015')
    tab2 = Panel(child=pp_tab_list[1], title='2016')
    tab3 = Panel(child=pp_tab_list[2], title='2017')
    tab4 = Panel(child=pp_tab_list[3], title='2018')

    layout2 = Tabs(tabs=[tab1, tab2, tab3, tab4])
    output_notebook()
    show(row(layout1, layout2))