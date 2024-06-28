# Optimizing timeseries multiclass classification utilizing feature evaluation methods

## Summary

In this study, fault classification was performed using multiclass classification with the aim of optimizing the used features and required computational power. Feature evaluation and selection was performed in two ways: calculating statistical **p-values** for each feature, and utilizing a **Minimum Redundance - Maximum Relevance** (**mRMR**) algorithm. Feature sets of different sizes were evaluated in terms of classification accuracy and computational performance.

## Method

![Framework](https://github.com/AbidAbdulAzeez/Time_Series_Classification/assets/81526615/215ea439-7ffb-4f53-9fbf-63849c5cda57)


## Provided data

The data was obtained from a simulated model of a hydraulic system. The data consists of the following signals:
- Motor-pump speed reference (MREF)
- Motor-pump speed response (MRES)
- Current responses of the motor (U, V, W)
- Piston and rod side pressures (PSP, RSP)
- Accumulator pressure (AP).

The measurements consist of 12 fault cases of a check valve and 1 healthy case. Each data file has 50 000 samples with sample time of 1ms. The classes are:
- Leakage: L1, L2, L3
- Friction: F1, F2, F3
- Spring: SP1, SP2, SP3
- Stiction: ST1, ST2, ST3
- Healthy: H.

## Preprocessing and feature extraction

The data preprocessing is implemented in Preprocessing.ipynib, where statistical features are calculated from the data using sliding window technique. The extracted statistical features include: mean, medium, variance, standard deviation, minimum, maximum, etc. These features are calculated from each signal. In the code, it is possible to change the size of the windows, and the step size between windows.

After feature extraction, the new data can be saved individually for each signal and also as one file combining all data.

## Feature selection

The goal of feature selection is to enhance the performance of classification models. This is done by selecting only the most relevant features to be used in the model training. Performing feature selection improves model interpretability and also reduces computational complexity. Different methods for feature selection are introduced here, focusing on the filter methods which were used in the study.

### 1. Filter methods

Filter methods use the statistical properties of the data to measure feature performance with respect to specific criteria, and they are independent of a classifier. Generally, filter methods are computationally less demanding compared to the other methods. Different ways of categorizing the methods exist, one of which is briefly introduced here.

#### Univariate methods

These rank features separately by their relations with the outcome variable, in this case the classes. They do this regardless of other classes. For classification, one possible implementation is to calculate **p-values**. They measure the probability that there is no relationship between a specific feature and a fault type.

#### Multivariate methods

As the name may suggest, this methods takes into account not only the features' relationship to the target class, but also their interactions with each other. One example is **mRMR**, which works in an iterative way, selecting one feature at a time. At each iteration, we want to select the feature with **maximum relevance** with respect to the target variable and **minimum redundancy** with respect to the chosen features at previous iterations.

#### Other methods

**Mutual information -based:** These methods utilize mutual information, which is a measure of mutual dependence between two features. Entropy measures the uncertainty of a class. A high mutual information value indicates a large uncertainty reduction. Meaning that when we realize the value of one class, there is a relatively high certainty in the other's expected value.  
**Relief-based:** These methods use knowledge of the nearest neighbors to derive feature statistics. The algorithm assigns higher importance to features in which samples from one class are further away from examples that belong to the other class. In general, more neighbors result in more accurate scores, to some extent, but take a longer time to converge.

### 2. Wrapper methods

This method employs any stand-alone modeling algorithm to train a predictive model using a candidate feature subset. The testing performance on a subset is typically used to score the feature set. For this method to get an idea about the features that actually are the most relevant, it takes many rounds with different feature sets. This makes the method computationally very complex and time consuming. Also, the selected subset is optimal only for the model that was used for process of selecting it. Therefore, if we wanted to compare performance of different classification models using this method, an optimal subset would need to be defined on each of those models we wish to compare.

### 3. Embedded methods

As a some type of combination of the filter and wrapper methods, embedded methods perform feature selection as a part of the modeling algorithm’s execution, meaning that they take advantage of the feature importance estimations which are embedded in the algorithm. These methods are usually more efficient compared to wrappers in terms of computational complexity. This is possible because they simultaneously integrate modeling with feature selection.

## Algorithm steps

#### 1. Install needed packages  
The data handling is done using Numpy and Pandas libraries, the classifier implementations are from Scikit-learn and Lgbm libraries, the feature evaluation and selection use Tsfresh and mRMR libraries, and the result visualization as a confusion matrix is done using Matplotlib and Seaborn. Installation via pip is presented here. For other installation methods, please refer to the installation guides.

- **python:** [3.11.9 or above](https://www.python.org/downloads/)
- **numpy:** pip install [numpy](https://numpy.org/install/)
- **pandas:** pip install [pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- **scikit-learn:** pip install -U [scikit-learn](https://scikit-learn.org/stable/install.html)
- **lgbm:** pip install [lightgbm](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)
- **tsfresh:** pip install [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/quick_start.html)
- **mrmr:** pip install [mrmr-selection](https://pypi.org/project/mrmr-selection/)
- **matplotlib:** pip install -U [matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- **seaborn:** pip install [seaborn](https://seaborn.pydata.org/installing.html)

#### 2. Download the data and code files

#### 3. Run the codes
**a. Run *Preprocessing.ipynib*.**  
- Specify the corresponding file paths to the folder you have the data in. Also check where the preprocessed data will be saved.  
- Example of how files from a folder "Raw_data" are read in this code:
```
path='C:/Users/username/Classification/Raw_data' # replace this with your file location
all_files = sorted(glob.glob(path+"/*.csv"),key=numericalSort) # read and sort all the files with an ending ".csv"
```
- After the preprocessing, data can be saved separately:
```
# change the labels according to the used data
labels = ['F1','F2','F3','H','L1','L2','L3','SP1','SP2','SP3','ST1','ST2','ST3']

save_path = 'C:/Users/username/Classification/Preprocessed_data/' # replace this with your own save location

for i,l in zip(all_files,labels):
    
    df = pd.read_csv(i,delimiter=',')
    
    # Check for the correct column names (modify according to your data)
    df.columns = ['t', 'U', 'V', 'W', 'MRES', 'MREF', 'PSP','RSP', 'AP']
    
    # Create variables for each column
    v1=df["t"]; v2=df["U"]; v3=df["V"]; v4=df["W"]; v5=df["MRES"];
    v6=df["MREF"]; v7=df["PSP"]; v8=df["RSP"]; v9=df["AP"]
    
    #df = transform_to_freq(df) # Use for the frequency conversion if needed!
    
    df = full(df) # run the function full to implement preprocessing
    df.to_csv(l+'_win 5 100.csv',index=False) # Saving the preprocessed data with its fault label "l"
    print('The current file is '+i)
```
and also combined, which is used for the classification:  
```
pathpp = 'C:/Users/username/Classification/Preprocessed_data'
all_files = sorted(glob.glob(pathpp"/*intval win.csv"),key=numericalSort) # replace intval and win with the correct numbers
all_pp = pd.concat( [pd.read_csv(f) for f in all_files] )
all_pp.to_csv("DATA_combined_win "+intval+" "+win+".csv", index=False, encoding='utf-8-sig')
```

**b. Run *Multiclass_classification_step-by-step.ipynb*.**  
- Again, check that the file paths and names correspond.  
- *Step 1*: The data is **scaled** to the range of -1...1 and a **feature relevance table** is generated.  
    - According to the **p-values** calculated in the feature relevance table, a set of "best features" can be chosen.  
    - Example feature sets *5 features_p-values.csv* and *10 features_p-values.csv* are provided and used in the code.  
- *Step 2*: The features calculated from the motor responses is separated into a new variable *dataE*.  
```
dataE = data.drop(data.loc[:, 'MREF minimum':'AP mean_abs_change'].columns, axis=1)
```  
- *Step 3*: Feature selection
    - Read the provided feature sets *5 features_p-values.csv* and *10 features_p-values.csv* **or** other set of features.
    - Perform feature selection using the **mRMR** algorithm. The code is constructed to choose an equal amount of features as are in the sets chosen based on the p-values.
```
feature_path = "C:/Users/username/Classification/Feature_sets" # replace this with your own location
                                                               # of the feature sets
feature_files = sorted(glob.glob(feature_path+"/p_values*.csv"),key=numericalSort)

p_val_sets = []
for f in feature_files:
    features = pd.read_csv(f).to_numpy().flatten()
    p_val_sets.append(features)
    
setnames = ["5 features", "10 features"] # Name the feature sets (here they are named by
                                         # the number of features and the used method)
mrmr_sets = []

for name,s in zip(setnames,p_val_sets):
    n = s.size
    features = mrmr_classif(data, Y, K=n) # Here the features are chosen from all the data.
                                          # Change to dataE to select only from electric features.
    df = pd.DataFrame(features)
    df.to_csv(feature_path+"/"+name+"_mrmr.csv", encoding='utf-8-sig') # Saving the selected features
    mrmr_sets.append(features)
```  
- *Step 4*: Splitting the data to train and test data, and performing the classification.
    - The classifiers used are shown below.
    - The training is implemented on both the selected feature sets, **and** all and electric data.
    - Note that training all the classifiers can take several minutes depending on the amount of data and the number of features used. The **accuracy score** of each classifier is printed when the training is completed, showing the progress.
    - The classification results are saved as **classification reports**.
```
classifiers = [DecisionTreeClassifier(),
               RandomForestClassifier(n_estimators=100),,
               svm.SVC(random_state=8, max_iter=1000),
               KNeighborsClassifier(n_neighbors=3),
               SGDClassifier(loss='squared_hinge', max_iter=2000),
               MLPClassifier(hidden_layer_sizes=(64, 16, 4), random_state=8, max_iter = 300),
               LGBMClassifier()]
```  
- *Step 5*: Two confusion matrices are produced as an example.  
- *Step 6*: A comparison between different sizes of feature sets is made.
- *Step 7*: Memory profiling is performed using the code file *Training_for_memory_profiling.py*.
    - Results are saved in separate text files in the same location as the code is run at.

 ## Sample results - confusion matrix for all signals

<p align="center">
  <img width="720" height="600" src="https://github.com/AbidAbdulAzeez/Time_Series_Classification/assets/81526615/af450516-fad2-42e5-adfe-f7d52f239412">
</p>

 ## Sample results - confusion matrix for electric signals
 
<p align="center">
  <img width="720" height="600" src="https://github.com/AbidAbdulAzeez/Time_Series_Classification/assets/81526615/88e1261e-e115-416a-a49a-42a1e771f7c7">
</p>
    
**Paper link:**  [AI-Based Condition Monitoring of Hydraulic Valves in Zonal Hydraulics Using Simulated Electric Motor Signals](https://doi.org/10.1115/FPMC2021-68615)

**Note:** If you use the code or data provided here, please refer to the above paper.  

**Bibtex:**  

```
@proceedings{10.1115/FPMC2021-68615,
    author = {Abdul Azeez, Abid and Han, Xu and Zakharov, Viacheslav and Minav, Tatiana},
    title = "{AI-Based Condition Monitoring of Hydraulic Valves in Zonal Hydraulics Using Simulated Electric Motor Signals}",
    volume = {ASME/BATH 2021 Symposium on Fluid Power and Motion Control},
    series = {Fluid Power Systems Technology},
    pages = {V001T01A016},
    year = {2021},
    month = {10},
    abstract = "{Zonal hydraulics, in particular Direct Driven Hydraulics (DDH), is an emerging transmission and actuation technique that is proposed to be used for electrification of heavy-duty mobile machinery. In addition to the already demonstrated advantages of DDH, which include high efficiency, compactness, and ease of maintenance, it is also capable of condition monitoring. The condition monitoring features can be obtained through indirect analysis of the existing electric motor signals (voltage and current) using artificial intelligence-based algorithms rather than by adding extra sensors, which are normally required for conventional realization. In this paper, the valve condition monitoring method of the DDH through electrical motor signals is explored at an early development stage. Firstly, the hydraulic valve models, which involve the valve fault behaviors, are added to the basic DDH model. Secondly, healthy and faulty scenarios for the valves are simulated, and the data are generated. Thirdly, the preliminary artificial intelligence-based condition monitoring classifier is developed using the simulation data, including feature extraction, algorithm training, testing, and comparison of accuracy. The effects of modeling error on developing the condition monitoring function are analyzed. In conclusion, the preliminary outcomes for the valve condition monitoring of the DDH are achieved by taking advantage of modeling and simulation and by utilizing the existing electric motor signals.}",
    doi = {10.1115/FPMC2021-68615},
    url = {https://doi.org/10.1115/FPMC2021-68615},
    eprint = {https://asmedigitalcollection.asme.org/FPST/proceedings-pdf/FPMC2021/85239/V001T01A016/6811428/v001t01a016-fpmc2021-68615.pdf},
}
```
