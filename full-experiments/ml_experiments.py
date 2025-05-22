from imports import *

def class_plot(data,dataset_name):

    cont_periodo = data['target'].value_counts()

    plt.figure(figsize=(10, 6))
    ax = cont_periodo.plot(kind='bar', color='black')
    plt.title(f'Distribuição de classes - {dataset_name} ')
    plt.xlabel('Situação')
    plt.ylabel('Nº de Desistentes')
    plt.xticks(rotation=45)
    plt.tight_layout()

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0f} ({height / cont_periodo.sum() * 100:.1f}%)', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', fontsize=10, color='gray', xytext=(0, 10), textcoords='offset points')
    
    if not os.path.exists('plots/class-distribution'):
        os.makedirs('plots/class-distribution')
    plt.savefig(f'plots/class-distribution/class-distribution-{dataset_name}.pdf', format='pdf')
    plt.close()

def pre_process(data,dataset_name):

    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    #class_plot(data,dataset_name)
   
    # removendo colunas com mais de 50% de valores ausentes
    data = data.dropna(axis=1, thresh=len(data) * 0.5)
    
    columns = data.columns.tolist()

    for column in columns:
        if data[column].dtype == 'object':
            data.loc[:, column] = pd.Categorical(data[column])
            data.loc[:, column] = LabelEncoder().fit_transform(data[column])
        
        data.loc[:, column] = data[column].astype(str).str.replace(',', '.', regex=True)
        data.loc[:, column] = pd.to_numeric(data[column], errors='coerce')


    return data

def const_remove(data,threshold):
    constant_filter = VarianceThreshold(threshold)
    constant_filter.fit(data)
    data = data.iloc[:, constant_filter.get_support()]
    
    return data

def imputation(data):
    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(data)
    data = pd.DataFrame(data=df_imputed, columns=data.columns)

    return data

def correlation(data,threshold):
    corr = data.corr()
    mask = (corr > threshold) | (corr < (-threshold))
    columns_to_drop = []
    for col in mask.columns:
        correlated_cols = mask.index[mask[col]].tolist()
        if len(correlated_cols) > 1:
            columns_to_drop.extend(correlated_cols[1:])
    
    data = data.drop(columns=columns_to_drop)
    return data

def load_models():
    model_mapping = {
        'dt': DecisionTreeClassifier(random_state=0, ccp_alpha=0.02),
        'rf': RandomForestClassifier(n_estimators=100, random_state=0),
        'neigh': KNeighborsClassifier(n_neighbors=3),
        'nb': GaussianNB()
    }
      
    return model_mapping

def plot_rf(model,features,dataset_name,algorithm,corr_threshold,const_threshold,seed):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importância': importance})
    importance_df = importance_df.sort_values(by='Importância', ascending=False)

    plt.figure(figsize=(25, 10))
    plt.barh(importance_df['Feature'], importance_df['Importância'], color='skyblue')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.title('Importância das Features')
    plt.gca().invert_yaxis()  # Inverter a ordem das features para exibir a mais importante no topo

    if not os.path.exists('plots/rf'):
        os.makedirs('plots/rf') 
    plt.savefig(f'plots/rf/rf-{dataset_name}-{algorithm}-{corr_threshold}-{const_threshold}-{seed}.pdf', format='pdf')
    plt.close()


def plot_dt(model,features,dataset_name,algorithm,corr_threshold,const_threshold,seed):
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=features)
   
    if not os.path.exists('plots/dt'):
        os.makedirs('plots/dt') 
    plt.savefig(f'plots/dt/dt-{dataset_name}-{algorithm}-{corr_threshold}-{const_threshold}-{seed}.pdf', format='pdf')
    plt.close()

def plot_cm(cm,dataset_name,algorithm,corr_threshold,const_threshold,seed):
    opt = ["0", "1"]  
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=opt, yticklabels=opt)
    plt.xlabel('Predições')
    plt.ylabel('Valores Reais')
    plt.title('Matriz de Confusão')

    if not os.path.exists('plots/cm'):
        os.makedirs('plots/cm') 
    plt.savefig(f'plots/cm/cm-{dataset_name}-{algorithm}-{corr_threshold}-{const_threshold}-{seed}.pdf', format='pdf')
    plt.close()
    
def train_test(x, y, dataset_name, algorithm, corr_threshold, const_threshold, seed):
    
    n_samples = len(y)
    n_splits = min(10, n_samples)

    if n_splits < 2:
        raise ValueError(f"Não é possível dividir o conjunto de dados {dataset_name} com {n_samples} amostras em {n_splits} partes")

    model_mapping = load_models()
    results = {'Model': [], 'Accuracy': [], 'Balanced Accuracy': [],  'f1-score': [],  'recall': [],  'precision': [], 'CM - True Positive': [], 'CM - False Negative': [], 'CM - False Positive': [], 'CM - True Negative': []}

    classifier = model_mapping[algorithm]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(classifier, x, y, cv=cv)
    classifier.fit(x, y)

    predict_df = pd.DataFrame({'True_Label': y, 'Predicted_Label': y_pred})

    if not os.path.exists('predictions'):
        os.makedirs('predictions') 
    predict_df.to_csv(f'predictions/predict-{dataset_name}-{algorithm}-{corr_threshold}-{const_threshold}-{seed}.csv', index=False)

    acc = accuracy_score(y, y_pred)
    bac = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)  

    cm = confusion_matrix(y, y_pred)
    
    results['Model'].append(algorithm)
    results['Accuracy'].append(acc)
    results['Balanced Accuracy'].append(bac)
    results['f1-score'].append(f1)
    results['recall'].append(recall)
    results['precision'].append(precision)

    # Verifique se a matriz de confusão tem o tamanho esperado (2x2)
    if cm.shape == (2, 2):
        results['CM - True Positive'].append(cm[0][0])
        results['CM - False Negative'].append(cm[0][1])
        results['CM - False Positive'].append(cm[1][0])
        results['CM - True Negative'].append(cm[1][1])
    else:
        results['CM - True Positive'].append(None)
        results['CM - False Negative'].append(None)
        results['CM - False Positive'].append(None)
        results['CM - True Negative'].append(None)

    features = x.columns.tolist()

    # if f1 > 0.9:
    #     if algorithm == 'rf':
    #         plot_rf(classifier, features, dataset_name, algorithm, corr_threshold, const_threshold, seed)
    #     elif algorithm == 'dt':
    #         plot_dt(classifier, features, dataset_name, algorithm, corr_threshold, const_threshold, seed)

    #     if cm.shape == (2, 2):
    #         plot_cm(cm,dataset_name,algorithm,corr_threshold,const_threshold,seed)   

    infos = {
        'Dataset': [dataset_name], 
        'corr_threshold': [corr_threshold], 
        'const_threshold': [const_threshold], 
        'seed': [seed], 
        'class_distribution': [y.value_counts()],
        'dropout': [y.value_counts().get(1, 0)],  
        'regular': [y.value_counts().get(0, 0)]   
    }

    df_temp1 = pd.DataFrame(infos)
    df_temp2 = pd.DataFrame(results)
    df_temp1 = df_temp1.reindex(df_temp2.index)

    df_results = pd.concat([df_temp1, df_temp2], axis=1)

    return df_results


def execute(dataset_name,algorithm,corr_threshold,const_threshold,seed,execution):
    print(f'Execução numero {execution}')

    data = pd.read_csv(f'../datasets/{dataset_name}')

    # Cria a coluna target a partir de "Situação atual"
    data['target'] = data['Situação atual'].apply(lambda x: 1 if x == 'Desistente' else 0)

    y = data['target']
    data = pre_process(data,dataset_name)

    features = data.columns.tolist()
    features.remove('target')

    x = data[features]

    data = const_remove(x,const_threshold)

    data = imputation(data)

    # Removendo novamente features constantes após imputação
    data = const_remove(data,const_threshold)

    data = correlation(data,corr_threshold)

    results = train_test(data,y,dataset_name,algorithm,corr_threshold,const_threshold,seed)

    if not os.path.exists('results'):
        os.makedirs('results') 
    results.to_csv(f'results/resultados-{dataset_name}-{algorithm}-{corr_threshold}-{const_threshold}-{seed}.csv', index=False)

    return results

#execute('data_ap_Reg_Desis.csv', 'rf', 0.8, 0.05, 145,1)
