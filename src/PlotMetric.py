
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud
# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


class PlotMetric:
    def __init__(self):
        self.figure_counter = 100


    def plot_all_data_metric(self, all_results_dict, metric='acc', loop=3, title_prefix=''):
        """
        This function plots a grouped bar chart comparing a metric (Accuracy, Precision, Recall, F1 Score) for each model across datasets.

        Parameters:
        - all_results_dict: all results from evaluate_classifiers()
        - model_names: list of model names (strings)
        - metric: one of ['acc', 'pr', 'recall', 'f1']
        - loop: number of repetitions used
        - title_prefix: optional text to prefix chart titles
        """

        metric = metric.lower()
        allowed_metrics = {'acc': 'Accuracy', 'precval': 'Precision', 'recall': 'Recall', 'f1': 'F1 Score'}
        modelnames = ['NB', 'LogReg', 'SVM', 'RF', 'GMM', 'LOF', 'BERT']
        metric_label = allowed_metrics.get(metric, metric)

        # metric_label = allowed_metrics[metric]
        results = list(all_results_dict.keys())
        first_result = next(iter(all_results_dict.values()))
        model_names = list(first_result.keys())
        nresults = len(results)
        nmodels = len(model_names)

        x = np.arange(nmodels)
        barwidth = 0.8 / nresults
        opacity = 0.8
        colors = np.array(['#0198E1', '#FF3333', '#BF5FFF', '#FCD116', '#FF7216', '#4DBD33', '#87421F'])

        plt.figure(figsize=(max(10, nmodels + 6), 6))
        errconfig = {'ecolor': '0.3'}
        scoremean = np.zeros((nresults, nmodels))
        scorestd = np.zeros((nresults, nmodels))

        for i, dname in enumerate(results):
            for j, mname in enumerate(model_names):
                model_result = all_results_dict[dname][mname]
                # Handling 'precval' vs 'prec'
                if metric not in model_result:
                    if metric == 'precval' and 'prec' in model_result:
                        value_list = model_result['prec']
                    else:
                        value_list = []
                else:
                    value_list = model_result[metric]

                if len(value_list) > 0:
                    scoremean[i, j] = np.mean(value_list)
                    scorestd[i, j] = np.std(value_list)
                else:
                    scoremean[i, j] = 0
                    scorestd[i, j] = 0

        # Start plotting
        fig, ax = plt.subplots(figsize=(max(10, nmodels + 4), 6), facecolor='white')

        for idx, dname in enumerate(results):
            ax.bar(x + idx * barwidth, scoremean[idx], width=barwidth, alpha=opacity,
                   color=colors[idx % len(colors)], label=dname, yerr=scorestd[idx], error_kw=errconfig)

        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel(f"{metric_label} Score", fontsize=12)
        ax.set_title(f"{metric_label} Score Across Datasets ({loop} iterations)\n{title_prefix}", fontsize=14)
        ax.set_xticks(x + (barwidth * (nresults - 1) / 2))
        ax.set_xticklabels(model_names, rotation=0)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title='Datasets', loc='best', fontsize=8)

        plt.tight_layout()
        self.save_figure()

        # plt.show()


    def plot_eval_metric(self, results, metric='acc', loop=3, title_prefix=''):
        """
        This function plots mean/max/min scores(acc, precision, recall, f1) for each model using `results` from evaluate_classifiers().

        Parameters:
        - results: dictionary with model names as keys and dicts of metric lists as values
        - metric: one of ['acc', 'prec', 'recall', 'f1']
        - loop: number of repetitions used
        - title_prefix: optional string to prefix the chart title
        """

        metric = metric.lower()
        # allowed_metrics = ['acc', 'prec', 'recall', 'f1']
        # assert metric in allowed_metrics, f"Metric must be one of {allowed_metrics}"

        model_names = list(results.keys())
        metric_means = [np.mean(results[m][metric]) for m in model_names]
        metric_maxs = [np.max(results[m][metric]) for m in model_names]
        metric_mins = [np.min(results[m][metric]) for m in model_names]

        df = pd.DataFrame({
            'Model': model_names,
            'mean': metric_means,
            'max': metric_maxs,
            'min': metric_mins
        }).sort_values(by='mean')

        x = np.arange(len(df))
        color_main = '#1B84CC' if metric != 'acc' else '#CC4F1B'
        color_fill = '#84C2FF' if metric != 'acc' else '#FF9848'

        plt.figure(figsize=(max(8, len(model_names)), 6))
        plt.ylim(0, 1)
        plt.plot(x, df['mean'], 'o-', color=color_main)
        plt.fill_between(x, df['max'], df['min'], alpha=0.2, facecolor=color_fill)
        plt.xticks(x, df['Model'], rotation=45, ha='right')
        plt.title(f"{title_prefix}Mean {metric.upper()} per Model over {loop} Iterations", fontsize=12)
        plt.legend([f"mean {metric}", "max/min range"], loc='best')
        plt.ylabel(f"{metric.upper()} Score")
        plt.xlabel("Models")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        self.save_figure()
        # plt.show()

        return df

    def plot_metric_boxplot(self, results, metric='prec', title='', save_as=None):
        """
        This function plots a boxplot of metric distribution per model using results from evaluate_classifiers().

        Parameters:
        - results: dict from evaluate_classifiers()
        - metric: 'acc', 'prec', 'recall', or 'f1'
        - title: optional plot title
        - save_as: optional filename to save figure (e.g., 'fig3.png')
        """
        metric = metric.lower()

        labels = list(results.keys())
        metric_data = [results[m][metric] for m in labels]  # list of lists

        x = np.arange(1, len(labels) + 1)

        plt.figure(figsize=(len(labels) + 3, 6), dpi=100, facecolor='w', edgecolor='k')
        plt.ylim(0, 1)
        plt.boxplot(metric_data, patch_artist=True)
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.title(title or f"Distribution of {metric.upper()} Scores per Algorithm", fontsize=12)
        plt.ylabel(f"{metric.upper()} Score")
        plt.xlabel("Models")
        ax = plt.gca()
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        # Optional: golden ratio
        width = 7.5
        height = width / 1.618
        fig = plt.gcf()
        fig.set_size_inches(width, height)
        if save_as:
            fig.savefig(save_as, bbox_inches='tight')
        plt.tight_layout()
        self.save_figure()
        # plt.show()

    def save_figure(self):
        # global FIGURE_COUNTER
        filename = f"{'.'}/{'fig'}{self.figure_counter}.png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"saved:{filename}")
        self.figure_counter += 1

    def plot_label_distribution(self, df, label_col='label'):
        counts = df[label_col].value_counts().sort_index()
        labels = ['Fake', 'Real'] if set(counts.index) == {0, 1} else counts.index

        sns.barplot(x=labels, y=counts.values, palette='coolwarm')
        plt.title("Distribution of Fake vs Real News")
        plt.ylabel("Number of Articles")
        plt.xlabel("Label")
        for i, val in enumerate(counts.values):
            plt.text(i, val + 2, f"{val}", ha='center', fontsize=12)
        self.save_figure()
        # plt.show()

    def visualize_text_ngram(self, df, ngramrange=(2, 3), top_k=30):
        """
        Visualize n-gram frequencies for different classes.

        Parameters:
        - df: DataFrame with 'text' and 'label' columns
        - ngramrange: tuple for n-grams, e.g. (2, 3) for bigrams & trigrams
        - top_k: top n-grams to plot
        """
        assert 'text' in df.columns and 'label' in df.columns, "Input DataFrame must contain 'text' and 'label' columns"

        # Generate n-gram frequencies separately for each class
        def get_ngram_freq(corpus):
            vec = CountVectorizer(ngram_range=ngramrange, stop_words='english').fit(corpus)
            ngramcounts = vec.transform(corpus).sum(axis=0)
            return {ngram: int(ngramcounts[0, idx]) for ngram, idx in vec.vocabulary_.items()}

        fakefreq = get_ngram_freq(df[df['label'] == 0]['text'])
        realfreq = get_ngram_freq(df[df['label'] == 1]['text'])

        dffake = pd.DataFrame(fakefreq.items(), columns=['ngram', 'count_0']).sort_values(by='count_0',
                                                                                          ascending=False).head(top_k)
        dfreal = pd.DataFrame(realfreq.items(), columns=['ngram', 'count_1']).sort_values(by='count_1',
                                                                                          ascending=False).head(top_k)

        # Plotting
        plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(3, 2)

        # WordClouds
        wc0 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fakefreq)
        wc1 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(realfreq)

        ax0 = plt.subplot(gs[0, 0])
        ax0.imshow(wc0, interpolation='bilinear')
        ax0.set_title("N-Gram WordCloud - Class 0 (Fake)", fontsize=14)
        ax0.axis('off')

        ax1 = plt.subplot(gs[0, 1])
        ax1.imshow(wc1, interpolation='bilinear')
        ax1.set_title("N-Gram WordCloud - Class 1 (Real)", fontsize=14)
        ax1.axis('off')

        # Top barplots
        ax2 = plt.subplot(gs[1, 0])
        sns.barplot(x='count_0', y='ngram', data=dffake, ax=ax2, palette='Reds_r')
        ax2.set_title("Top N-Grams in Class 0 (Fake)")

        ax3 = plt.subplot(gs[1, 1])
        sns.barplot(x='count_1', y='ngram', data=dfreal, ax=ax3, palette='Blues_r')
        ax3.set_title("Top N-Grams in Class 1 (Real)")

        # Combined stacked n-gram histogram
        allngrams = pd.DataFrame(dffake[['ngram', 'count_0']]).merge(dfreal[['ngram', 'count_1']], on='ngram',
                                                                     how='outer').fillna(0)

        # Instead of sns.barplot with hue, use pivot + stacked bar
        pivotdf = allngrams.pivot_table(index='ngram', values=['count_0', 'count_1'], aggfunc='sum').fillna(0)

        pivotdf[['count_0', 'count_1']].plot(
            kind='bar',
            stacked=True,
            color=['#FF6666', '#6699FF'],
            figsize=(12, 6)
        )
        plt.title("Stacked N-Gram Frequency: Fake vs Real")
        plt.xlabel("N-Gram")
        plt.ylabel("Frequency")
        plt.legend(['Fake', 'Real'])
        plt.xticks(rotation=90)
        plt.tight_layout()
        self.save_figure()
        # plt.show()

    def plot_bar_metric(self, results_dict, datasetname='covid', metric='acc', loop=3, title_prefix=''):
        """
        This function plots a grouped bar chart comparing a metric (Accuracy, Precision, Recall, F1 Score) for each model across datasets.

        Parameters:
        - results: results from evaluate_classifiers()
        - datasets: list of dataset names
        - model_names: list of model names (strings)
        - metric: one of ['acc', 'pr', 'recall', 'f1']
        - loop: number of repetitions used
        - title_prefix: optional text to prefix chart titles
        """

        metric = metric.lower()
        allowed_metrics = {'acc': 'Accuracy', 'precval': 'Precision', 'recall': 'Recall', 'f1': 'F1 Score'}
        modelnames = ['NB', 'LogReg', 'SVM', 'RF', 'GMM', 'LOF', 'BERT']

        metric_label = allowed_metrics[metric]
        model_names = list(results_dict.keys())
        model_means = [np.mean(results_dict[m][metric]) for m in model_names]
        model_errs = [np.max(results_dict[m][metric]) - np.min(results_dict[m][metric]) for m in model_names]

        x = np.arange(len(model_names))
        bar_width = 0.5
        opacity = 0.8
        colors = np.array(['#0198E1', '#FF3333', '#BF5FFF', '#FCD116', '#FF7216', '#4DBD33', '#87421F'])

        plt.figure(figsize=(max(8, len(model_names) * 1.5), 6))
        plt.bar(x, model_means, yerr=model_errs, width=bar_width, alpha=opacity,
                color=[colors[i % len(colors)] for i in x], capsize=5)

        plt.xticks(x, model_names)
        plt.ylim(0, 1)
        plt.ylabel(f"{metric_label} Score")
        plt.xlabel("Models")
        plt.title(f"{metric_label} Comparison on {datasetname} ({loop} runs)\n{title_prefix}", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        self.save_figure()
        # plt.show()

    def plot_runtime(self, all_results_dict, loop=3):
        '''
        Plot average runtime per algorithm based on results dictionary
        '''

        datasets = list(all_results_dict.keys())
        models = list(next(iter(all_results_dict.values())).keys())

        times = np.zeros((len(datasets), len(models)))
        for i, dataset in enumerate(datasets):
            result = all_results_dict[dataset]
            for j, model in enumerate(models):
                if 'times' in result[model]:
                    times[i, j] = np.mean(result[model]['times'])
                else:
                    times[i, j] = np.nan  # or 0

        x = np.arange(len(models))  # 0,1,2,...number of models
        plt.figure(figsize=(len(models) + 3, len(datasets) + 3), dpi=100)

        for i in range(len(datasets)):
            plt.plot(x, times[i], '-o', label=datasets[i])
        plt.xticks(x, models, rotation=45)
        plt.title(f"Average Runtime across Datasets over {loop} Iterations", fontsize=12)
        plt.xlabel("Algorithm")
        plt.ylabel("Computation Time (seconds)")
        plt.grid(True)
        plt.legend(title="Dataset", loc='upper left')
        plt.tight_layout()
        self.save_figure()
        # plt.show()

