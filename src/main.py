import FakeNewsDetector
import LoadDataSets
import PlotMetric


if __name__ == '__main__':
    '''
    This code runs multiple of machine learning models to detect fake news using 
    four datasets: COVID-19, FakeNewsNet, ISOT, and LIAR. 

    File structure: 
        - main.py: runs the entire project 
        - LoadDataSets.py: Loads and  the four datasets listed above
        - FakeNewsDetector.py: Trains data using multiple classifiers(LogReg, NB, SVM,RF, Bert, and Ensemble) 
        - PlotMetric.py: Plots evaluation results (metrics, runtime, n-grams,etc.) 
        
    Data location: 
        By default the data path is set to one level up to the code: 
        data_path = '../datasets'
        If user datasets are located elsewhere, user can override this path when initalizing in main.py
        LoadDatasets(data_path='path/to/your/data')  
    
    Running the code: 
     1. Open a terminal and navigate to the folder containing main.py
     2. Run: python main.py 
     
     NOTE: For portability, the datasets delivered with this package have been reduced to 100 rows each.
    '''

    # Load datasets
    df = LoadDataSets.LoadDataSets(datalen=None)
    covid, fakenews, isot, lair  = df.get_all_datasets()
    cleanedcoviddata = df.preprocess_data(covid)
    cleanedfakenews = df.preprocess_data(fakenews)
    cleanedisot = df.preprocess_data(isot)
    cleanedlair = df.preprocess_data(lair)

    # Run classifiers
    ml = FakeNewsDetector.FakeNewsDetector()
    summary_covid, results_covid = ml.evaluate_classifiers(cleanedcoviddata,test_size=0.3, random_state=42, loop=1,save_model=False)
    summary_fakenews, results_fakenews = ml.evaluate_classifiers(cleanedfakenews,test_size=0.3, random_state=42, loop=1,save_model=False)
    summary_isot, results_isot = ml.evaluate_classifiers(cleanedisot,test_size=0.3, random_state=42, loop=1,save_model=False)
    summary_lair, results_lair = ml.evaluate_classifiers(cleanedlair,test_size=0.3, random_state=42, loop=1,save_model=False)

    datasets = ['covid','fakenews', 'isot', 'liar']
    all_results_dict = dict(zip(datasets, [results_covid, results_fakenews, results_isot, results_lair]))

    pd = PlotMetric.PlotMetric()
    metric = ['acc', 'precval', 'recall', 'f1']
    for m in metric:
        pd.visualize_text_ngram(cleanedfakenews, ngramrange=(2, 3), top_k=30)
        pd.plot_all_data_metric(all_results_dict, m, 3)

        pd.plot_eval_metric(results_covid,m, 3)
        pd.plot_eval_metric(results_fakenews,m, 3)
        pd.plot_eval_metric(results_isot,m, 3)
        pd.plot_eval_metric(results_lair,m, 3)

        pd.plot_metric_boxplot(results_covid,m, 3)
        pd.plot_metric_boxplot(results_fakenews,m, 3)
        pd.plot_metric_boxplot(results_isot,m, 3)
        pd.plot_metric_boxplot(results_lair,m, 3)

    pd.plot_runtime(all_results_dict, 3)
