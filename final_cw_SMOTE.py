import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.utils import check_random_state
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features
from functools import reduce
import glob
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ConvergenceWarning)
sns.set_style('white')

def load_hydraulicsystems_data(extract=True, WINDOW=30, STEP=10, tsf = False):
    '''
    Number of Instances: 2205
    
    Number of Attributes: 43680 (8x60 (1 Hz) + 2x600 (10 Hz) + 7x6000 (100 Hz))
    
    Relevant Information:
    The data set was experimentally obtained with a hydraulic test rig. This test 
    rig consists of a primary working and a secondary cooling-filtration circuit 
    which are connected via the oil tank [1], [2]. The system cyclically repeats 
    constant load cycles (duration 60 seconds) and measures process values such as 
    pressures, volume flows and temperatures while the condition of four hydraulic 
    components (cooler, valve, pump and accumulator) is quantitatively varied. 
    
    Attribute Information:
    The data set contains raw process sensor data (i.e. without feature extraction) 
    which are structured as matrices (tab-delimited) with the rows representing the 
    cycles and the columns the data points within a cycle. The sensors involved are:
    Sensor		Physical quantity				Unit		Sampling rate
    PS1			Pressure						bar			100 Hz
    PS2			Pressure						bar			100 Hz
    PS3			Pressure						bar			100 Hz
    PS4			Pressure						bar			100 Hz
    PS5			Pressure						bar			100 Hz
    PS6			Pressure						bar			100 Hz
    EPS1		Motor power						W			100 Hz
    FS1			Volume flow						l/min		10 Hz
    FS2			Volume flow						l/min		10 Hz
    TS1			Temperature						°C			1 Hz
    TS2			Temperature						°C			1 Hz
    TS3			Temperature						°C			1 Hz
    TS4			Temperature						°C			1 Hz
    VS1			Vibration						mm/s		1 Hz
    CE			Cooling efficiency (virtual)	%			1 Hz
    CP			Cooling power (virtual)			kW			1 Hz
    SE			Efficiency factor				%			1 Hz
    
    The target condition values are cycle-wise annotated in ‘profile.txt‘ 
    (tab-delimited). As before, the row number represents the cycle number. 
    The columns are
    
    1: Cooler condition / %:
    	3: close to total failure
    	20: reduced effifiency
    	100: full efficiency
    
    2: Valve condition / %:
    	100: optimal switching behavior
    	90: small lag
    	80: severe lag
    	73: close to total failure
    
    3: Internal pump leakage:
    	0: no leakage
    	1: weak leakage
    	2: severe leakage
    
    4: Hydraulic accumulator / bar:
    	130: optimal pressure
    	115: slightly reduced pressure
    	100: severely reduced pressure
    	90: close to total failure
    
    5: stable flag:
    	0: conditions were stable
    	1: static conditions might not have been reached yet
    '''
    print('Loading Hydraulic Systems Data')
    names = ['cooler_condition', 'valve_condition', 'pump_leak', 
             'hydraulic_accumulator', 'stable_flag']
    target_path = 'UCI Engineering/Hydraulic Systems/data/target/'
    features_path = 'UCI Engineering/Hydraulic Systems/data/features/*.txt'
    target = pd.read_csv(target_path+'profile.txt', delim_whitespace=True, 
                         header=None, names=names)
    files = glob.glob(features_path)
    df = {}

    if tsf:
        # based on https://github.com/zhou100/SensorDefaults
        for file in files:
            temp = pd.read_csv(file, sep='\t', header=None)
            temp.index.name = 'cycle'
            temp.reset_index(inplace=True)
            temp_transposed= temp.T
            temp_transposed.index.name = 'time'
            temp_transposed.reset_index(inplace=True)
            string = ' cycle'.join(str(e) for e in list(temp_transposed.columns))
            temp_transposed.columns = string.split(" ")
            temp_long = pd.wide_to_long(temp_transposed.iloc[1:,:],
                                       stubnames='cycle', i=['time'], j='c')
            temp_long.reset_index(inplace=True)
            df[file[48:-4]] = temp_long
        for key in list(df.keys()):
            df[key].columns=['time','cycle', key]
        dfs= [df['SE'], df['CP'], df['CE'], df['VS1'], df['TS4'], df['TS3'],
              df['TS2'], df['TS1'], df['FS2'], df['FS1'],  df['EPS1'], df['PS6'],
              df['PS5'], df['PS4'], df['PS3'], df['PS2'], df['PS1']]
              
        features = reduce(lambda left,right: pd.merge(left,right,
                                                      on=['time','cycle']), dfs)
        return features, target
    
    if extract:
        for file in files:
            temp = pd.read_csv(file, sep='\t', header=None)
            df[file[48:-4]] = temp
        t0 = time.time()
        print('Extracting Features...')
        # all PS = 100 Hz
        FREQ100 = 100
        PS1 = window_features(df['PS1'].transpose(), list(df['PS1'].index), 
                              FREQ100*WINDOW, FREQ100*STEP, 'PS1')
        PS2 = window_features(df['PS2'].transpose(), list(df['PS2'].index), 
                              FREQ100*WINDOW, FREQ100*STEP, 'PS2')
        PS3 = window_features(df['PS3'].transpose(), list(df['PS3'].index), 
                              FREQ100*WINDOW, FREQ100*STEP, 'PS3')
        PS4 = window_features(df['PS4'].transpose(), list(df['PS4'].index), 
                              FREQ100*WINDOW, FREQ100*STEP, 'PS4')
        PS5 = window_features(df['PS5'].transpose(), list(df['PS5'].index), 
                              FREQ100*WINDOW, FREQ100*STEP, 'PS5')
        PS6 = window_features(df['PS6'].transpose(), list(df['PS6'].index), 
                              FREQ100*WINDOW, FREQ100*STEP, 'PS6')
        EPS1 = window_features(df['EPS1'].transpose(), list(df['EPS1'].index), 
                              FREQ100*WINDOW, FREQ100*STEP, 'EPS1')
        # all FS = 10 Hz
        FREQ10 = 10
        FS1 = window_features(df['FS1'].transpose(), list(df['FS1'].index), 
                              FREQ10*WINDOW, FREQ10*STEP, 'FS1')
        FS2 = window_features(df['FS2'].transpose(), list(df['FS2'].index), 
                              FREQ10*WINDOW, FREQ10*STEP, 'FS2')
        # all those = 1 Hz
        FREQ1 = 1
        TS1 = window_features(df['TS1'].transpose(), list(df['TS1'].index), 
                              FREQ1*WINDOW, FREQ1*STEP, 'TS1')
        TS2 = window_features(df['TS2'].transpose(), list(df['TS2'].index), 
                              FREQ1*WINDOW, FREQ1*STEP, 'TS2')
        TS3 = window_features(df['TS3'].transpose(), list(df['TS3'].index), 
                              FREQ1*WINDOW, FREQ1*STEP, 'TS3')
        TS4 = window_features(df['TS4'].transpose(), list(df['TS4'].index), 
                              FREQ1*WINDOW, FREQ1*STEP, 'TS4')
        VS1 = window_features(df['VS1'].transpose(), list(df['VS1'].index), 
                              FREQ1*WINDOW, FREQ1*STEP, 'VS1')
        CE = window_features(df['CE'].transpose(), list(df['CE'].index), 
                              FREQ1*WINDOW, FREQ1*STEP, 'CE')
        CP = window_features(df['CP'].transpose(), list(df['CP'].index), 
                              FREQ1*WINDOW, FREQ1*STEP, 'CP')
        SE = window_features(df['SE'].transpose(), list(df['SE'].index), 
                              FREQ1*WINDOW, FREQ1*STEP, 'SE')
        # combine all dataframes
        features = pd.concat([PS1, PS2, PS3, PS4, PS5, PS6, EPS1, FS1, FS2, TS1, 
                              TS2, TS3, TS4, VS1, CE, CP, SE], axis=1)
        print(f'Done... {((time.time())-t0):.2f} seconds')
        return features, target
    else:
        print('PLEASE MAKE A CHOICE!')
    
def window_features(X_in, names, WINDOW, STEP, name):
    t0 = time.time()
    print('Extracting features...')
    X = pd.DataFrame(names)
    X.set_index(0, inplace=True)
    size = X_in.shape[0]
    first = 0
    last = WINDOW
    # Statistical features
    print(f'Creating statistical based features for {name}...')
    while last <= (size + STEP/2):
        print(f'Window = {first}-{last}')
        col_name = f'{name}_{first}-{last}_mean'
        X[col_name] = 0
        for name in names:
            X[col_name].loc[name] = X_in[name][first:last].mean()
        col_name = f'{name}_{first}-{last}_median'
        X[col_name] = 0
        for name in names:
            X[col_name].loc[name] = X_in[name][first:last].median()
        col_name = f'{name}_{first}-{last}std'
        X[col_name] = 0
        for name in names:
            X[col_name].loc[name] = X_in[name][first:last].std()
        col_name = f'{name}_{first}-{last}min'
        X[col_name] = 0
        for name in names:
            X[col_name].loc[name] = X_in[name][first:last].min()
        col_name = f'{name}_{first}-{last}max'
        X[col_name] = 0
        for name in names:
            X[col_name].loc[name] = X_in[name][first:last].max()
        col_name = f'{name}_{first}-{last}skew'
        X[col_name] = 0
        for name in names:
            X[col_name].loc[name] = X_in[name][first:last].skew()
        col_name = f'{name}_{first}-{last}kurt'
        X[col_name] = 0
        for name in names:
            X[col_name].loc[name] = X_in[name][first:last].kurt()
        first += STEP
        last += STEP
    print(f'Done, {len(list(X))} features in {((time.time())-t0):.2f} seconds')
    return X

def create_pca(X, name, **kwargs):
    t0 = time.time()
    print('Normalizing the data...')
    X_normalize = StandardScaler().fit_transform(X)
    with open(f'results/{name}.txt', 'a') as f:
        if kwargs:
            print(f'Normalized shape {kwargs["cond"]} = {X_normalize.shape}', file=f)
        else:
            print(f'Normalized shape = {X_normalize.shape}', file=f)
    pca = PCA(0.999)
    try:
        X_pca = pca.fit_transform(X_normalize)
    except:
        X_pca = pca.fit_transform(X_normalize)
    with open(f'results/{name}.txt', 'a') as f:
        if kwargs:
            print(f'PCA shape {kwargs["cond"]} = {X_pca.shape} \n', file=f)
        else:
            print(f'PCA shape = {X_pca.shape} \n', file=f)
    print(f'Done, PCA returned {((time.time())-t0):.2f} seconds')
    np.savetxt(f'results/{name}_pca.csv', X_pca, delimiter=",")
    return X_pca

@ignore_warnings(category=ConvergenceWarning)
def training_models(X_pca, y_true, name, **kwargs):
    with ignore_warnings(category=ConvergenceWarning):
        t0 = time.time()
        # Training models
        check_random_state(13)
        FOLDS = 5
        SPLIT = 0.50
        X_train1, X_test, y_train1, y_test = train_test_split(X_pca, y_true, 
                                                            stratify=y_true, 
                                                            test_size=SPLIT)
        
        # SMOTE oversampling technique
        smt = SMOTE(random_state=13)
        X_train, y_train = smt.fit_resample(X_train1, y_train1)
        
        # parameters
        N_ITER = 100
        C = [(i/100) for i in range(1,10001,5)]
        CRIT = ['gini', 'entropy']
        MAX_DEPTH = [i for i in range(10,501,2)]
        MAX_DEPTH.append(None)
        MIN_LEAF = [i/100 for i in range(1,26)]
        MIN_SPLIT = [i/100 for i in range(1,26)]
        GAMMA = [(0.01 * i / len(list(X_train))) for i in range(1,1001)]
    
        print('Starting LogisticRegression')
        ta = time.time()
        # class_weight = balanced, datasets are slightly unbalanced, so this will help
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        parameters = {'C':C}
        clf_lr = RandomizedSearchCV(lr, parameters, n_iter=N_ITER, cv=FOLDS, 
                                    n_jobs=-1, pre_dispatch='2*n_jobs', 
                                    verbose=10)
        clf_lr.fit(X_train, y_train)
        if kwargs:
            save_results(name, X_test, y_test, clf_lr, 
                          'Logistic Regression Classification'+'_'+kwargs['cond'])
            confusion(y_test, clf_lr.predict(X_test), name+'_'+kwargs['cond'],
                      'Logistic Regression Classification')
        else:
            save_results(name, X_test, y_test, clf_lr, 
                          'Logistic Regression Classification')
            confusion(y_test, clf_lr.predict(X_test), name,
                      'Logistic Regression Classification')
        print(f' done = {(time.time()) - ta}')
        
        print('Starting DecisionTreeClassifier')
        ta = time.time()
        dtc = DecisionTreeClassifier()
        parameters = {'criterion': CRIT, 'max_depth': MAX_DEPTH, 
                      'min_samples_leaf': MIN_LEAF, 'min_samples_split': MIN_SPLIT}
        clf_dtc = RandomizedSearchCV(dtc, parameters, n_iter=N_ITER, cv=FOLDS, 
                                      n_jobs=-1, pre_dispatch='2*n_jobs', 
                                      verbose=10)
        clf_dtc.fit(X_train, y_train)
        if kwargs:
            save_results(name, X_test, y_test, clf_dtc, 
                          'Decision Tree Classification'+'_'+kwargs['cond'])
            confusion(y_test, clf_dtc.predict(X_test), name+'_'+kwargs['cond'],
                      'Decision Tree Classification')
        else:
            save_results(name, X_test, y_test, clf_dtc, 
                          'Decision Tree Classification')
            confusion(y_test, clf_dtc.predict(X_test), name,
                      'Decision Tree Classification')
        print(f' done = {(time.time()) - ta}')
        
        print('Starting LinearSVC')
        ta = time.time()
        lsvm = LinearSVC(class_weight='balanced', max_iter=2000)
        parameters = {'C': C}
        clf_lsvm = RandomizedSearchCV(lsvm, parameters, n_iter=N_ITER, cv=FOLDS, 
                                      n_jobs=-1, pre_dispatch='2*n_jobs', 
                                      verbose=10)
        clf_lsvm.fit(X_train, y_train)
        if kwargs:
            save_results(name, X_test, y_test, clf_lsvm, 
                          'Linear Support Vector Classification'+'_'+kwargs['cond'])
            confusion(y_test, clf_lsvm.predict(X_test), name+'_'+kwargs['cond'],
                      'Linear Support Vector Classification')
        else:
            save_results(name, X_test, y_test, clf_lsvm, 
                          'Linear Support Vector Classification')
            confusion(y_test, clf_lsvm.predict(X_test), name,
                      'Linear Support Vector Classification')
        print(f' done = {(time.time()) - ta}')
    
        print('Starting Polynomial SVC')
        DEGREE = [i for i in range(2,5)]
        ta = time.time()
        polsvm = SVC(class_weight='balanced', kernel='poly')
        parameters = {'C': C, 'degree': DEGREE, 'gamma': GAMMA}
        clf_polsvm = RandomizedSearchCV(polsvm, parameters, n_iter=N_ITER, 
                                        cv=FOLDS, n_jobs=-1, pre_dispatch='2*n_jobs', 
                                        verbose=10)
        clf_polsvm.fit(X_train, y_train)
        if kwargs:
            save_results(name, X_test, y_test, clf_polsvm, 
                          'Polynomial Support Vector Classification'+'_'+kwargs['cond'])
            confusion(y_test, clf_polsvm.predict(X_test), name+'_'+kwargs['cond'],
                      'Polynomial Support Vector Classification')
        else:
            save_results(name, X_test, y_test, clf_polsvm, 
                          'Polynomial Support Vector Classification')
            confusion(y_test, clf_polsvm.predict(X_test), name,
                      'Polynomial Support Vector Classification')
        print(f' done = {(time.time()) - ta}')
    
        print('Starting RBF SVC')
        ta = time.time()
        rbfsvm = SVC(class_weight='balanced', kernel='rbf')
        parameters = {'C': C, 'gamma': GAMMA}
        clf_rbfsvm = RandomizedSearchCV(rbfsvm, parameters, n_iter=N_ITER, 
                                        cv=FOLDS, n_jobs=-1, pre_dispatch='2*n_jobs', 
                                        verbose=10)
        clf_rbfsvm.fit(X_train, y_train)
        if kwargs:
            save_results(name, X_test, y_test, clf_rbfsvm, 
                          'RBF Support Vector Classification'+'_'+kwargs['cond'])
            confusion(y_test, clf_rbfsvm.predict(X_test), name+'_'+kwargs['cond'],
                      'RBF Support Vector Classification')
        else:
            save_results(name, X_test, y_test, clf_rbfsvm, 
                          'RBF Support Vector Classification')
            confusion(y_test, clf_rbfsvm.predict(X_test), name,
                      'RBF Support Vector Classification')
        print(f' done = {(time.time()) - ta}')
       
        print(f'Done, buiding the models in {((time.time())-t0):.2f} seconds')
        return clf_lr, clf_dtc, clf_lsvm, clf_polsvm, clf_rbfsvm

def confusion(test, predict, name, title='Confusion Matrix'):
    # based on https://github.com/zhou100/SensorDefaults
    '''
    The result metrics are mainly using three measures:
        Accuracy. The percent of data that we accurately predict in each category.
        Preicision. The number of true positives divided by the number of true 
        positives plus the number of false positives.
        Recall. The number of true positives divided by the number of true positives 
        plus the number of false negatives.
    '''
    names = np.unique(test)
    bins = [i for i in np.unique(test)]
    bins.append(sum(bins))
    # Make a 2D histogram from the test and result arrays
    pts, xe, ye = np.histogram2d(test, predict, bins)
    # For simplicity we create a new DataFrame
    pd_pts = pd.DataFrame(pts.astype(int).T, index=names, columns=names)
    # Display heatmap and add decorations
    plt.close()
    sns.heatmap(pd_pts, annot=True, fmt="d", cmap='GnBu')    
    plt.title(title, fontsize=20)
    plt.xlabel('Actual', fontsize=18)
    plt.ylabel('Predicted', fontsize=18)
    plt.savefig(f'results/{name}_{title}.png')
    plt.close()
    pts = (pts.T/sum(pts.T))
    pd_pts = pd.DataFrame(pts, index=names, columns=names )
    hm = sns.heatmap(pd_pts, annot=True, fmt=".2%", cmap='GnBu')    
    hm.axes.set_title(title, fontsize=20)
    hm.axes.set_xlabel('Actual', fontsize=18)
    hm.axes.set_ylabel('Predicted', fontsize=18)
    plt.savefig(f'results/{name}_norm_{title}.png')
    plt.close()
    return None

def save_results(name, X_test, y_test, clf, title='Classifier'):
    with open(f'results/{name}.txt', 'a') as f:
        print(f'Best params: {clf.best_params_}', file=f)
        # Compute and save accuracy score for random forest
        score = 100.0 * clf.score(X_test, y_test)
        print(f"{title} prediction accuracy = {score:5.1f}%", file=f)
        print(metrics.classification_report(y_test, clf.predict(X_test)), file=f)
    return None

def main_HS():
# Hydraulic Systems Dataset, using top 2 WINDOW/STEP and TSFRESH
    name = 'HS_X1_20-10'
    X1, y_true = load_hydraulicsystems_data(True, 20, 10)
    with open(f'results/{name}.txt', 'w') as f:
        print(name, file=f)
    X1_pca = create_pca(X1, name)
    # for each condition we need a model:
    X1_models = {}
    for condition in list(y_true):
        clf_lr, clf_dtc, clf_lsvm, clf_polsvm, clf_rbfsvm =\
            training_models(X1_pca, y_true[condition], name, cond=condition)
        X1_models[condition] = {'clf_lr': clf_lr, 'clf_dtc': clf_dtc, 
                                'clf_lsvm': clf_lsvm, 'clf_polsvm': clf_polsvm, 
                                'clf_rbfsvm': clf_rbfsvm}
        
    name = 'HS_X1_30-15'
    X1, y_true = load_hydraulicsystems_data(True, 30, 15)
    with open(f'results/{name}.txt', 'w') as f:
        print(name, file=f)
    X1_pca = create_pca(X1, name)
    # for each condition we need a model:
    X1_models = {}
    for condition in list(y_true):
        clf_lr, clf_dtc, clf_lsvm, clf_polsvm, clf_rbfsvm =\
            training_models(X1_pca, y_true[condition], name, cond=condition)
        X1_models[condition] = {'clf_lr': clf_lr, 'clf_dtc': clf_dtc, 
                                'clf_lsvm': clf_lsvm, 'clf_polsvm': clf_polsvm, 
                                'clf_rbfsvm': clf_rbfsvm}

    df, y_true = load_hydraulicsystems_data(tsf=True)
    
    name = 'HS_X2_TSFRESH1'
    with open(f'results/{name}.txt', 'w') as f:
        print(name, file=f)
    # for each condition we need a model:
    X2_models = {}
    X2 = extract_features(df, column_id="cycle", column_sort="time")
    impute(X2)
    for condition in list(y_true):
        X2_pca = create_pca(X2, name, cond=condition)
        clf_lr, clf_dtc, clf_lsvm, clf_polsvm, clf_rbfsvm =\
            training_models(X2_pca, y_true[condition], name, cond=condition)
        X2_models[condition] = {'clf_lr': clf_lr, 'clf_dtc': clf_dtc, 
                                'clf_lsvm': clf_lsvm, 'clf_polsvm': clf_polsvm, 
                                'clf_rbfsvm': clf_rbfsvm}

    name = 'HS_X3_TSFRESH2'
    with open(f'results/{name}.txt', 'w') as f:
        print(name, file=f)
    # for each condition we need a model:
    X3_models = {}
    for condition in list(y_true):
        X3 = extract_relevant_features(df, y_true[condition], 
                                        column_id="cycle", column_sort="time")
        impute(X3)
        X3_pca = create_pca(X3, name, cond=condition)
        clf_lr, clf_dtc, clf_lsvm, clf_polsvm, clf_rbfsvm =\
            training_models(X3_pca, y_true[condition], name, cond=condition)
        X3_models[condition] = {'clf_lr': clf_lr, 'clf_dtc': clf_dtc, 
                                'clf_lsvm': clf_lsvm, 'clf_polsvm': clf_polsvm, 
                                'clf_rbfsvm': clf_rbfsvm}

    return None

def main():
    # main_HS()
    return None

if __name__ == "__main__": main()
