import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from scipy.stats import norm


from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA 

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

OUTPUT_TEMPLATE = (
    'Daily\n'
    'Bayesian classifier: {bayes_daily:.3g}\n'
    'kNN classifier:      {knn_daily:.3g}\n'
    'Standard SVM classifier:      {svm_daily_standard:.3g}\n'
    'MinMax SVM classifier:      {svm_daily_minmax:.3g}\n'
    'PCA(2),clusters:\n'
    '{df_daily}\n'              
    'Monthly\n'
    'Bayesian classifier: {bayes_monthly:.3g}\n'
    'kNN classifier:      {knn_monthly:.3g}\n'
    'Standard SVM classifier:      {svm_monthly_standard:.3g}\n'
    'MinMax SVM classifier:      {svm_monthly_minmax:.3g}\n'
    'PCA(2),clusters:\n'
    '{df_monthly}\n'        
    'Quarterly\n'
    'Bayesian classifier: {bayes_quarterly:.3g}\n'
    'kNN classifier:      {knn_quarterly:.3g}\n'
    'Standard SVM classifier:      {svm_quarterly_standard:.3g}\n'
    'MinMax SVM classifier:      {svm_quarterly_minmax:.3g}\n'
        'PCA(2),clusters:\n'
    '{df_quarterly}\n'        
    'Yearly\n'
    'Bayesian classifier: {bayes_yearly:.3g}\n'
    'kNN classifier:      {knn_yearly:.3g}\n'
    'Standard SVM classifier:      {svm_yearly_standard:.3g}\n'
    'MinMax SVM classifier:      {svm_yearly_minmax:.3g}\n'
    'PCA(2),clusters:\n'
    '{df_yearly}\n'        
)
def get_pca(X):

    flatten_model = make_pipeline(
        MinMaxScaler(),
        PCA(2)
    )
    X2 = flatten_model.fit_transform(X)
    assert X2.shape == (X.shape[0], 2)
    return X2

    
def get_clusters(X,no_clusters):
    model=make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=no_clusters)
    )
    model.fit(X)
    return model.predict(X)

def GNB_KN_SVC_SVC1 (X,y,n,no_clusters):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #sns.countplot(y, palette='hls').set_title(title + " Count")
    #plt.show()

    GuassianNB_model = GaussianNB()
    GuassianNB_model.fit(X_train, y_train)
    GNB = GuassianNB_model.score(X_test, y_test)


    KNeighbor_model = KNeighborsClassifier(n_neighbors=n)
    KNeighbor_model.fit(X_train, y_train)
    KN= KNeighbor_model.score(X_test, y_test)

    SVC_model_pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1)
    )
    SVC_model_pipeline.fit(X_train, y_train)
    SVM= SVC_model_pipeline.score(X_test, y_test)


    SVC_model_pipline = make_pipeline(
        MinMaxScaler(),
        SVC(kernel='rbf', C=1)
    )

    SVC_model_pipline.fit(X_train, y_train)
    SVM1= SVC_model_pipline.score(X_test, y_test)

    X2 = get_pca(X)
    clusters = get_clusters(X,no_clusters)
    plt.subplot(1, 2, 1)
    plt.title('data after PCA transform ')
    plt.ylabel('X_pca[:,1]')
    plt.xlabel('X_pca[:,0]')

    plt.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap='Set1', edgecolor='k')
    df = pd.DataFrame({
        'cluster': clusters,
        'long_short': y,
    })
    df= pd.crosstab(df['long_short'], df['cluster'])

    
    return GNB,KN,SVM,SVM1,df


def main():

    xlabel = ['pct_GDP', 'pct_MonetaryBase', 'pct_CPI', 'pct_HomePrice', 'pct_Loans', 'pct_Employment',
               'pct_Income', 'pct_ConstructionSpending', 'pct_FedFundRate','pct_USDollar', 'pct_CrudeOil',
               'pct_Import_Unit_Value', 'pct_Import_Volume', 'pct_Import_Value', 'pct_Export_Unit_Value', 'pct_Export_Volume', 'pct_Export_Value'
               ]

    n = 5 #knn
    m=24 #no_cluster
    ylabel='long_short'

    filename = "Ultimate_Data" + "\\" + "Ultimate_Assortion_pct_Change_Daily_Insoo.csv"
    data = pd.read_csv(filename, sep=',', encoding='utf-8')
    data_x = data[xlabel].values
    data_y= data[ylabel].values
    GuassianNB_model_daily,KNeighbor_model_daily,SVC_model_pipline_daily_standard, SVC_model_pipline_daily_minmax,df_daily =GNB_KN_SVC_SVC1 (data_x,data_y,n,m)
    plt.subplot(1, 2, 2)
    plt.hist(data['pct_SnP'], bins=405, alpha=0.5)
    plt.suptitle("Daily Distribution")
    plt.axis('tight')
    plt.xlim(-0.065, 0.065)
    plt.show()
    
    filename = "Ultimate_Data" + "\\" + "Ultimate_Assortion_pct_Change_Monthly_Insoo.csv"
    monthly_data = pd.read_csv(filename, sep=',', encoding='utf-8')
    monthly_x = monthly_data[xlabel].values
    monthly_y= monthly_data[ylabel].values
    m=12
    GuassianNB_model_monthly,KNeighbor_model_monthly,SVC_model_pipline_monthly_standard, SVC_model_pipline_monthly_minmax,df_monthly =GNB_KN_SVC_SVC1 ( monthly_x,monthly_y,n, m)
    plt.subplot(1, 2, 2)
    plt.hist(monthly_data['pct_SnP'], bins=45, alpha=0.5)
    plt.suptitle("Monthly Distribution")
    plt.axis('tight')
    plt.xlim(-0.125, 0.125)
    plt.show()

    filename = "Ultimate_Data" + "\\" + "Ultimate_Assortion_pct_Change_Quarterly_Insoo.csv"
    quarterly_data = pd.read_csv(filename, sep=',', encoding='utf-8')
    quarterly_x = quarterly_data[xlabel].values
    quarterly_y= quarterly_data[ylabel].values
    m=6
    GuassianNB_model_quarterly,KNeighbor_model_quarterly,SVC_model_pipline_quarterly_standard, SVC_model_pipline_quarterly_minmax,df_quarterly =GNB_KN_SVC_SVC1 ( quarterly_x,quarterly_y,n,m)
    plt.subplot(1, 2, 2)
    plt.hist(quarterly_data['pct_SnP'], bins=15, alpha=0.5)
    plt.suptitle("Quarterly Distribution")
    plt.axis('tight')
    plt.show()


    filename = "Ultimate_Data" + "\\" + "Ultimate_Assortion_pct_Change_Yearly_Insoo.csv"
    yearly_data = pd.read_csv(filename, sep=',', encoding='utf-8')
    yearly_x = yearly_data[xlabel].values
    yearly_y= yearly_data[ylabel].values
    m=3
    GuassianNB_model_yearly,KNeighbor_model_yearly,SVC_model_pipline_yearly_standard, SVC_model_pipline_yearly_minmax,df_yearly =GNB_KN_SVC_SVC1 ( yearly_x,yearly_y,n,m)
    plt.subplot(1, 2, 2)
    plt.hist(yearly_data['pct_SnP'], bins=7, alpha=0.5)
    plt.suptitle("yearly Distribution")
    plt.axis('tight')
    plt.show()


    print(OUTPUT_TEMPLATE.format(
        bayes_daily=GuassianNB_model_daily,
        knn_daily=KNeighbor_model_daily,
        svm_daily_standard=SVC_model_pipline_daily_standard,
        svm_daily_minmax=SVC_model_pipline_daily_minmax,
        df_daily=df_daily,
        bayes_monthly=GuassianNB_model_monthly,
        knn_monthly=KNeighbor_model_monthly,
        svm_monthly_standard=SVC_model_pipline_monthly_standard,
        svm_monthly_minmax=SVC_model_pipline_monthly_minmax,
        df_monthly=df_monthly,
        bayes_quarterly=GuassianNB_model_quarterly,
        knn_quarterly=KNeighbor_model_quarterly,
        svm_quarterly_standard=SVC_model_pipline_quarterly_standard,
        svm_quarterly_minmax=SVC_model_pipline_quarterly_minmax,
        df_quarterly=df_quarterly,
        bayes_yearly=GuassianNB_model_yearly,
        knn_yearly=KNeighbor_model_yearly,
        svm_yearly_standard=SVC_model_pipline_yearly_standard,
        svm_yearly_minmax=SVC_model_pipline_yearly_minmax,
        df_yearly=df_yearly

    ))


if __name__=='__main__':
    main()
