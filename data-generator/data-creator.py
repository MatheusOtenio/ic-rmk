import pandas as pd
import os
import re
from unidecode import unidecode
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='openpyxl')

# Cria o diretório 'datasets' se não existir
if not os.path.exists('datasets'):
    os.makedirs('datasets')

def make_valid_name(name):
    if(name =='Situação'):
      return 'situacao'
    else:
        name = unidecode(name)
        name = re.sub(r'\W+', '_', name)
        name = '_'.join(filter(None, name.lower().split('_')))
        return name
    
# Remove identifiers
def remove_columns(df):
    columns_to_remove = ['Campus', 'Sede', 'Nome', 'Código', 'Data de ingresso', 'Data de nascimento', 'Matriz curricular', 
                         'Município', 'Município SiSU', 'Município (SISU)', 'UF SISU', 'UF (SISU)', 'E-mail', 'Funcionamento', 'Sede']
    return df.drop(columns=[col for col in columns_to_remove if col in df.columns])

# Rename columns
def rename_columns(df):
    bad_names = df.columns.tolist()
    good_names = [make_valid_name(name) for name in bad_names]
    bad_to_good_converter = dict(zip(bad_names, good_names))
    df.rename(columns=bad_to_good_converter, inplace=True)
    return df

def apply_transformations(df, df_name):
    df_name = make_valid_name(df_name)
   
    df.to_csv(f'datasets/{df_name}_FormReg_DesisTran.csv', index=False)

    # First: Formado+Regular vs Desistente+Trancado
    data = df[(df['situacao'] == 'Desistente') | (df['situacao'] == 'Formado') | (df['situacao'] == 'Regular') | (df['situacao'] == 'Trancado')]
    data.loc[data['situacao'] == 'Formado', 'situacao'] = 'Regular'
    data.loc[data['situacao'] == 'Trancado', 'situacao'] = 'Desistente'
    data.loc[:, 'situacao'] = data['situacao'].replace({'Regular': 0, 'Desistente': 1})
    data = data.rename(columns={'situacao': 'target'})
    data.to_csv(f'datasets/{df_name}_FormReg_DesisTran.csv', index=False)
    apply_period_transformations(data, f'{df_name}_FormReg_DesisTran')
    apply_course_transformations(data, f'{df_name}_FormReg_DesisTran')
    apply_ger_per_period(data, f'{df_name}_FormReg_DesisTran')
    
    # Second: Regular vs Desistente+Trancado
    data = df[(df['situacao'] == 'Desistente') | (df['situacao'] == 'Regular') | (df['situacao'] == 'Trancado')].copy()
    data.loc[data['situacao'] == 'Trancado', 'situacao'] = 'Desistente'
    data.loc[:, 'situacao'] = data['situacao'].replace({'Regular': 0, 'Desistente': 1})
    data = data.rename(columns={'situacao': 'target'})
    data.to_csv(f'datasets/{df_name}_Reg_DesisTran.csv', index=False)
    apply_period_transformations(data, f'{df_name}_Reg_DesisTran')
    apply_course_transformations(data, f'{df_name}_Reg_DesisTran')
    apply_ger_per_period(data, f'{df_name}_Reg_DesisTran')
    
    # Third: Formado+Regular vs Desistente
    data = df[(df['situacao'] == 'Desistente') | (df['situacao'] == 'Formado') | (df['situacao'] == 'Regular')].copy()
    data.loc[data['situacao'] == 'Formado', 'situacao'] = 'Regular'
    data.loc[:, 'situacao'] = data['situacao'].replace({'Regular': 0, 'Desistente': 1})
    data = data.rename(columns={'situacao': 'target'})
    data.to_csv(f'datasets/{df_name}_FormReg_Desis.csv', index=False)
    apply_period_transformations(data, f'{df_name}_FormReg_Desis')
    apply_course_transformations(data, f'{df_name}_FormReg_Desis')
    apply_ger_per_period(data, f'{df_name}_FormReg_Desis')
    
    # Fourth: Formado vs Desistente
    data = df[(df['situacao'] == 'Desistente') | (df['situacao'] == 'Formado')].copy()
    data.loc[:, 'situacao'] = data['situacao'].replace({'Formado': 0, 'Desistente': 1})
    data = data.rename(columns={'situacao': 'target'})
    data.to_csv(f'datasets/{df_name}_Form_Desis.csv', index=False)
    apply_period_transformations(data, f'{df_name}_Form_Desis')
    apply_course_transformations(data, f'{df_name}_Form_Desis')
    apply_ger_per_period(data, f'{df_name}_Form_Desis')
    
    # Fifth: Regular vs Desistente
    data = df[(df['situacao'] == 'Desistente') | (df['situacao'] == 'Regular')].copy()
    data.loc[:, 'situacao'] = data['situacao'].replace({'Regular': 0, 'Desistente': 1})
    data = data.rename(columns={'situacao': 'target'})
    data.to_csv(f'datasets/{df_name}_Reg_Desis.csv', index=False)
    apply_period_transformations(data, f'{df_name}_Reg_Desis')
    apply_course_transformations(data, f'{df_name}_Reg_Desis')
    apply_ger_per_period(data, f'{df_name}_Reg_Desis')

def apply_period_transformations(df, df_name):
    df_name = make_valid_name(df_name)
    if 'periodo' in df.columns:
        for period_range, suffix in [((1, 3), '_1a3'), ((1, 3), '_1a3_sem_periodo'),
                                      ((1, 5), '_1a5'), ((1, 5), '_1a5_sem_periodo'),
                                      ((4, 10), '_4a10'), ((4, 10), '_4a10_sem_periodo')]:
            if 'sem_periodo' in suffix:
                data = df.loc[(df['periodo'] >= period_range[0]) & (df['periodo'] <= period_range[1])].copy()
                data = data.drop(['periodo', 'ano', 'semestre', 'ano_de_ingresso', 'ano_semestre_da_situacao'], axis=1, errors='ignore')
            else:
                data = df.loc[(df['periodo'] >= period_range[0]) & (df['periodo'] <= period_range[1])].copy()
                
            data.to_csv(f'datasets/{df_name}{suffix}.csv', index=False)

def apply_ger_per_period(df, df_name):
    df_name = make_valid_name(df_name)
    if 'periodo' in df.columns:
        periodos = df['periodo'].unique()
        for periodo in sorted(periodos):
            data_periodo = df[df['periodo'] == periodo].copy()
            data_periodo_sem_periodo = data_periodo.drop(['periodo','ano','semestre','ano_de_ingresso','ano_semestre_da_situacao'], axis=1, errors='ignore')
            data_periodo_sem_periodo.to_csv(f'datasets/{df_name}_{periodo}_periodo.csv', index=False)

def apply_course_transformations(df, df_name):
    df_name = make_valid_name(df_name)
    if 'curso' in df.columns:
        cursos = df['curso'].unique()
        for curso in cursos:
            data = df[df['curso'] == curso].copy()
            data.to_csv(f'datasets/{df_name}_{make_valid_name(curso)}.csv', index=False)
            if 'periodo' in data.columns:
                data_sem_periodo = data.drop('periodo', axis=1)
                data_sem_periodo.to_csv(f'datasets/{df_name}_{make_valid_name(curso)}_sem_periodo.csv', index=False)

# Load datasets 
data_dv = pd.read_csv('data-generator/newest-data-base/full-dv-data.csv')

data_dv = data_dv[(data_dv['Nível de ensino'] == "Graduação") & 
        (data_dv['Curso'] != 'Curso Superior De Tecnologia Em Horticultura')]

# Taking only the last semester of each student at ap data

data_dv_sorted = data_dv.sort_values(by=['Nome', 'Ano', 'Semestre', 'Período'], ascending=[True, True, True, True])
data_dv_last_occurence = data_dv_sorted.drop_duplicates(subset='Nome', keep='last')
data_dv_last_occurence.reset_index(drop=True, inplace=True)

# Map DataFrames to their variable names
dfs = {
    'data_dv': data_dv,
    'data_dv_last_occurence': data_dv_last_occurence,
}

# Loop iteration
for var_name, df in dfs.items():

    # Removing identifiers 
    df = remove_columns(df)

    # Renaming columns
    df = rename_columns(df)

    # Saving the initial processed dataframe
   # df.to_csv(f'datasets/{var_name}.csv', index=False)

    if 'situacao' in df.columns:
        # Apply transformations for 'situacao'
        apply_transformations(df, var_name)
