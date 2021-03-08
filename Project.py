import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
import operator
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer
from surprise import SVD,BaselineOnly, Reader,KNNBasic, NormalPredictor,KNNBaseline
from surprise import Reader, Dataset
from surprise.model_selection.validation import cross_validate
from surprise import accuracy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import time
st.set_option('deprecation.showPyplotGlobalUse', False)


st.sidebar.header('Section')
section = st.sidebar.radio("Choose a section:", 
                              ("EDA & Data Preprocessing Yelp DataSet", 
                              "Collabortive Filetring On Yelp Dataset",
                              'Evaluation of Collabortive Filetring On Yelp Dataset',
                              "EDA & Data Preprocessing KL DataSet", 
                              "Collabortive Filetring On KL Dataset",
                              'Evaluation of Collabortive Filetring On KL Dataset',
                              "KL Restaurant Recommender System")
                              )

def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return " ".join([word for word in nopunc.split() if word.lower() not in stop])

    
if section == "EDA & Data Preprocessing Yelp DataSet":
    user = pd.read_csv('yelp_review.csv')
    business = pd.read_csv('yelp_business.csv')
    
    st.title('Final Year Project')
    st.header("EDA & Data Preprocessing Yelp DataSet")  
    
    st.subheader("Null Values")
    st.write(business.isnull().sum())
    
    business["address"].fillna("No Address", inplace = True) 
    business["city"].fillna("No City", inplace = True) 
    business["postal_code"].fillna("No Postal", inplace = True) 
    business["hours"].fillna("No Operation Hours", inplace = True) 
    business["attributes"].fillna("No Review", inplace = True) 
    
    df = business.dropna(axis=0, subset=['categories'], inplace = True)
    st.subheader("After Cleaned Null Values")
    st.write(business.isnull().sum())
    
    
    select_feature = ( 'Bar one' , 'Bar two')
    select_feature1 = ( 'Bar two' )
    
    st.subheader("Ratings Count")
    sns.set(style="darkgrid")
    sns.countplot(x=business['stars'], data=business)
    st.pyplot()
        
    st.subheader("Top 10 Restuarant")
    business['name'].value_counts().sort_values(ascending=False).head(10).plot(kind='pie',figsize=(10,6), 
    title="10 Most Popular Cuisines", autopct='%1.2f%%')
    st.pyplot()
    
    st.subheader("Yelp User Reviews Data")
    select_feature2 = st.radio('Select a choice:', ('Original Dataset','Final DataFrame after cleaned'))
    
    if select_feature2 == 'Original Dataset':
        st.write(user)
        
    elif select_feature2 == 'Final DataFrame after cleaned': 
        yelp_idf = pd.read_csv('yelp_TDI-DF_scores.csv', encoding='latin-1')
        yelp_data = user[['business_id', 'user_id', 'stars', 'text']]
    
        stop = []
        for word in stopwords.words('english'):
            s = [char for char in word if char not in string.punctuation]
            stop.append(''.join(s))

        yelp_data['text'] = yelp_data['text'].apply(text_process)
        yelp_data['text'] = [word for word in yelp_data['text'] if word not in stopwords.words('english')]
        yelp_data['text']=yelp_data['text'].str.replace('\d+', '')
        
    
        df = pd.merge(yelp_data, yelp_idf, left_index=True, right_index=True)
        df['Word_Rating'] = pd.cut(df['Highest_word_score'], [0 , 0.2 , 0.4 , 0.6 , 0.8 , 1], labels=[1,2,3,4,5])
        st.write(df.astype('object'))

    
    
    
        

elif section == "Collabortive Filetring On Yelp Dataset":

    df = pd.read_csv('CF_Algo.csv')
    st.title('Final Year Project')
    st.header("Collabortive Filetring On YELP Dataset")
    
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id2', 'business_id2', 'Word_Rating']], reader)
    
    train_data = Dataset.load_from_df(df[['user_id2', 'business_id2', 'Word_Rating']], reader)
    test_data = Dataset.load_from_df(df[['user_id2', 'business_id2', 'Word_Rating']], reader)
    val_data = Dataset.load_from_df(df[['user_id2', 'business_id2', 'Word_Rating']], reader)
    train = train_data.build_full_trainset()
    val_before = val_data.build_full_trainset()
    val = val_before.build_testset()
    test_before = test_data.build_full_trainset()   
    test = test_before.build_testset()
    
    benchmark = []
    for algorithm in [SVD(), BaselineOnly(), KNNBaseline() , NormalPredictor() , KNNBasic()]:
        results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
        
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]],index=['Algorithm']))
        benchmark.append(tmp)
    
    surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
    st.subheader("Evluation of yelp data with collborative filetring withount parameter tuning")
    st.write(surprise_results)
    
    st.subheader("Model Prediction and Evaluation")
    select_algo = st.radio('Select a Algorithmn:', ('Baseline Only','KNNBasic','KNNBaseline', 'SVD' , 'Normal Predictor'))
    
    if select_algo == 'Baseline Only':
        bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
        algo = BaselineOnly(bsl_options=bsl_options)
        
        bias_baseline = BaselineOnly(bsl_options)
        bias_baseline.fit(train)
        
        bbase = bias_baseline.test(test)
        start_time = time.time()
        bias_baseline.fit(train)
        bias_time = ((time.time() - start_time))
        bias_time = round(bias_time,4)
        bbase_df = pd.DataFrame(bbase, columns = ['userId','itemId','rating','pred_rating','x'])
        baseline_rsme = accuracy.rmse(bbase)
        baseline_r2 = r2_score(bbase_df.rating , bbase_df.pred_rating)
        baseline_mae = mean_absolute_error(bbase_df.rating, bbase_df.pred_rating)
        
        data = [[baseline_rsme , baseline_mae ,baseline_r2 , bias_time]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        bbase_df['err'] = abs(bbase_df.pred_rating - bbase_df.rating)
        base_best_predictions = bbase_df.sort_values(by='err')[:5]
        base_worst_predictions = bbase_df.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions) 
        
        
            
    elif select_algo == 'KNNBasic':
        sim_options = {'name': 'cosine','user_based': False}
        
        algo_knn_basic = KNNBasic(sim_options=sim_options)
        algo_knn_basic.fit(train)
        
        start_time = time.time()
        algo_knn_basic.fit(train)
        knnb_time = time.time() - start_time
        knnb_time = round(knnb_time,4)
        
        knnB = algo_knn_basic.test(test)
        knn = pd.DataFrame(knnB, columns = ['userId','itemId','rating','pred_rating','x'])
        knnb_rmse = accuracy.rmse(knnB)
        knnb_r2 =  r2_score(knn.rating , knn.pred_rating)
        knnb_mae = mean_absolute_error(knn.rating, knn.pred_rating)
        
        data = [[knnb_rmse , knnb_mae ,knnb_r2 , knnb_time ]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        knn['err'] = abs(knn.pred_rating - knn.rating)
        base_best_predictions = knn.sort_values(by='err')[:5]
        base_worst_predictions = knn.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions) 
            
    elif select_algo == 'KNNBaseline':
        sim_options = {'name': 'cosine','user_based': False}
        
        Knnb = KNNBaseline(sim_options=sim_options)
        Knnb.fit(train)
        
        knnBS = Knnb.test(test)
        start_time = time.time()
        Knnb.fit(train)
        knnb_time = time.time() - start_time
        knnb_time = round(knnb_time,4)
        Knnbb = pd.DataFrame(knnBS, columns = ['userId','itemId','rating','pred_rating','x'])
        knnbs_rmse = accuracy.rmse(knnBS)
        knnbs_r2 = r2_score(Knnbb.rating , Knnbb.pred_rating) 
        knnbs_mae =  mean_absolute_error(Knnbb.rating, Knnbb.pred_rating)
        
        data = [[knnbs_rmse , knnbs_mae ,knnbs_r2 , knnb_time ]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        Knnbb['err'] = abs(Knnbb.pred_rating - Knnbb.rating)
        base_best_predictions = Knnbb.sort_values(by='err')[:5]
        base_worst_predictions = Knnbb.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions)
            
    
    elif select_algo == 'SVD':  
        RMSE_tune = {}
        n_epochs = [10, 20, 30]  # the number of iteration of the SGD procedure
        lr_all = [0.001, 0.003, 0.005] # the learning rate for all parameters
        reg_all =  [0.02, 0.05, 0.1, 0.4, 0.5] # the regularization term for all parameters

        for n in n_epochs:
            for l in lr_all:
                for r in reg_all:
                    print('Fitting n: {0}, l: {1}, r: {2}'.format(n, l, r))
                    algo = SVD(n_epochs = n, lr_all = l, reg_all = r)
                    algo.fit(train)
                    predictions = algo.test(val)
                    RMSE_tune[n,l,r] = accuracy.rmse(predictions)
        
        best_param = min(RMSE_tune.items(), key=operator.itemgetter(1))[0]
        st.write('The Best Paramater for SVD is:', best_param )
        
        
        algo_real = SVD(n_epochs = 30, lr_all = 0.005, reg_all = 0.02)
        algo_real.fit(train)
        
        svdd = algo_real.test(test)
        start_time = time.time()
        algo_real.fit(train)
        svd_time = time.time() - start_time
        svd_time = round(svd_time,4)
        
        svd = pd.DataFrame(svdd, columns = ['userId','businessId','rating','pred_rating','x' ])
        svd_rmse = accuracy.rmse(svdd)
        svd_r2 = r2_score(svd.rating , svd.pred_rating)
        svd_mae =  mean_absolute_error(svd.rating, svd.pred_rating)
        
        data = [[svd_rmse , svd_mae ,svd_r2 , svd_time ]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        svd['err'] = abs(svd.pred_rating - svd.rating)
        base_best_predictions = svd.sort_values(by='err')[:5]
        base_worst_predictions = svd.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions)
            
            
            
    elif select_algo == 'Normal Predictor':
        algo = NormalPredictor()
        algo.fit(train)
        
        norm = algo.test(test)
        start_time = time.time()
        algo.fit(train)
        norm_time = time.time() - start_time
        norm_time = round(norm_time,4)
        norm_p = pd.DataFrame(norm, columns = ['userId','itemId','rating','pred_rating','x'])
        norm_rmse = accuracy.rmse(norm)
        norm_r2 = r2_score(norm_p.rating , norm_p.pred_rating) 
        norm_mae =  mean_absolute_error(norm_p.rating, norm_p.pred_rating)
        
        data = [[norm_rmse , norm_mae ,norm_r2 , norm_time ]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        norm_p['err'] = abs(norm_p.pred_rating - norm_p.rating)
        base_best_predictions = norm_p.sort_values(by='err')[:5]
        base_worst_predictions = norm_p.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions)
            
            

elif section == 'Evaluation of Collabortive Filetring On Yelp Dataset':
        st.header("Comparison of Collaborative Filetring Algorithm")
        
        df = pd.read_csv('CF_Algo.csv')
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id2', 'business_id2', 'Word_Rating']], reader)
    
        train_data = Dataset.load_from_df(df[['user_id2', 'business_id2', 'Word_Rating']], reader)
        test_data = Dataset.load_from_df(df[['user_id2', 'business_id2', 'Word_Rating']], reader)
        val_data = Dataset.load_from_df(df[['user_id2', 'business_id2', 'Word_Rating']], reader)
        train = train_data.build_full_trainset()
        val_before = val_data.build_full_trainset()
        val = val_before.build_testset()
        test_before = test_data.build_full_trainset()   
        test = test_before.build_testset()
        
        bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
        algo = BaselineOnly(bsl_options=bsl_options)
        
        bias_baseline = BaselineOnly(bsl_options)
        bias_baseline.fit(train)
        
        bbase = bias_baseline.test(test)
        start_time = time.time()
        bias_baseline.fit(train)
        bias_time = ((time.time() - start_time))
        bias_time = round(bias_time,4)
        bbase_df = pd.DataFrame(bbase, columns = ['userId','itemId','rating','pred_rating','x'])
        baseline_rsme = accuracy.rmse(bbase)
        baseline_r2 = r2_score(bbase_df.rating , bbase_df.pred_rating)
        baseline_mae = mean_absolute_error(bbase_df.rating, bbase_df.pred_rating)
        
        sim_options = {'name': 'cosine','user_based': False}
        
        algo_knn_basic = KNNBasic(sim_options=sim_options)
        algo_knn_basic.fit(train)
        
        start_time = time.time()
        algo_knn_basic.fit(train)
        knnb_time = time.time() - start_time
        knnb_time = round(knnb_time,4)
        
        knnB = algo_knn_basic.test(test)
        knn = pd.DataFrame(knnB, columns = ['userId','itemId','rating','pred_rating','x'])
        knnb_rmse = accuracy.rmse(knnB)
        knnb_r2 =  r2_score(knn.rating , knn.pred_rating)
        knnb_mae = mean_absolute_error(knn.rating, knn.pred_rating)
        
        
        Knnb = KNNBaseline(sim_options=sim_options)
        Knnb.fit(train)
        
        knnBS = Knnb.test(test)
        start_time = time.time()
        algo_knn_basic.fit(train)
        knnb_time = time.time() - start_time
        knnb_time = round(knnb_time,4)
        Knnbb = pd.DataFrame(knnBS, columns = ['userId','itemId','rating','pred_rating','x'])
        knnbs_rmse = accuracy.rmse(knnBS)
        knnbs_r2 = r2_score(Knnbb.rating , Knnbb.pred_rating) 
        knnbs_mae =  mean_absolute_error(Knnbb.rating, Knnbb.pred_rating)
        
        algo_real = SVD(n_epochs = 30, lr_all = 0.005, reg_all = 0.02)
        algo_real.fit(train)
        
        svdd = algo_real.test(test)
        start_time = time.time()
        algo_real.fit(train)
        svd_time = time.time() - start_time
        svd_time = round(svd_time,4)
        
        svd = pd.DataFrame(svdd, columns = ['userId','businessId','rating','pred_rating','x' ])
        svd_rmse = accuracy.rmse(svdd)
        svd_r2 = r2_score(svd.rating , svd.pred_rating)
        svd_mae =  mean_absolute_error(svd.rating, svd.pred_rating)
        
        
        algo = NormalPredictor()
        algo.fit(train)
        
        norm = algo.test(test)
        start_time = time.time()
        algo.fit(train)
        norm_time = time.time() - start_time
        norm_time = round(norm_time,4)
        norm_p = pd.DataFrame(norm, columns = ['userId','itemId','rating','pred_rating','x'])
        norm_rmse = accuracy.rmse(norm)
        norm_r2 = r2_score(norm_p.rating , norm_p.pred_rating) 
        norm_mae =  mean_absolute_error(norm_p.rating, norm_p.pred_rating)
        
        x_algo = [ 'BaselineOnly' , 'SVD' , 'KNNBasic' , 'KNNBaseline' , 'Normal Predictor' ]
        all_algos_cv = [baseline_rsme,svd_rmse,knnb_rmse,knnbs_rmse , norm_rmse]
        mae_cv= [baseline_mae,svd_mae,knnb_mae,knnbs_mae , norm_mae ]
        r2_cv = [baseline_r2,svd_r2,knnb_r2,knnbs_r2 , norm_r2]
           
        
        select_pred = st.radio('Evaluation:', ('RMSE', 'MAE', 'R^2'))
        
        if select_pred == 'RMSE':
           plt.figure(figsize=(10,5))
           plt.title('Comparison of Algorithms on RMSE', loc='center', fontsize=15)
           plt.plot(x_algo, all_algos_cv, label='RMSE', color='darkgreen', marker='o')
           plt.xlabel('Algorithms', fontsize=15)
           plt.ylabel('RMSE Value', fontsize=15)
           plt.legend()
           plt.grid(ls='dashed')
           st.pyplot()
           
        elif select_pred == 'MAE':
           plt.title('Comparison of Algorithms on MAE', loc='center', fontsize=15)
           plt.plot(x_algo, mae_cv, label='MAE', color='navy', marker='o')
           plt.xlabel('Algorithms', fontsize=15)
           plt.ylabel('MAE Value', fontsize=15)
           plt.legend()
           plt.grid(ls='dashed')
           st.pyplot()
           
           
        elif select_pred == 'R^2':
           plt.title('Comparison of Algorithms on R^2', loc='center', fontsize=15)
           plt.plot(x_algo, r2_cv, label='R^2', color='red', marker='o')
           plt.xlabel('Algorithms', fontsize=15)
           plt.ylabel('R^2 Value', fontsize=15)
           plt.legend()
           plt.grid(ls='dashed')
           st.pyplot()

elif section == "EDA & Data Preprocessing KL DataSet":   
    kl_data = pd.read_csv('kl_data.csv')
    
    st.subheader('Original DataFrame')
    st.write(kl_data)
    
    
    st.subheader('Null Values')
    st.write(kl_data.isnull().sum())
    
    kl_data["contact_number"].fillna("No contact", inplace = True)
    kl_data["address"].fillna("No Address", inplace = True) 
    kl_data["review_count"].fillna("No Review", inplace = True) 
    kl_data["categories"].fillna("No Categories", inplace = True) 
    kl_data["pricing"].fillna("No pricing", inplace = True) 
    kl_data["rating"].fillna("No Review", inplace = True) 
    
    st.subheader('Null Values Removed')
    st.write(kl_data.isnull().sum())
    
    st.subheader('Ratings Count')
    plt.figure(figsize=(10,5))
    sns.set(style="darkgrid")
    sns.countplot(x=kl_data['rating'], data=kl_data)
    st.pyplot()
    
    
    st.subheader('Top 10 restuarant in KL')
    kl_data['restuarant_name'].value_counts().sort_values(ascending=False).head(10).plot(kind='pie',figsize=(10,6), 
    title="10 Most Popular Cuisines", autopct='%1.2f%%')
    plt.axis('equal')
    st.pyplot()
    
    select_data = st.radio('Evaluation:', ('User Reviews Before Clean', 'User Reviews After Cleaned'))
    if select_data == 'User Reviews Before Clean':
        data = pd.read_csv('modelll.csv')
        st.write(data)
        
    if select_data == 'User Reviews After Cleaned':
        df = pd.read_csv('clean_user.csv')
        df["restaurant_name"] = df["restaurant_name"].astype('category')
        df["restaurant_id"] = df["restaurant_name"].cat.codes
        df['Word_Rating'] = pd.cut(df['Highest_word_score'], [0 , 0.2 , 0.4 , 0.6 , 0.8 , 1], labels=[1,2,3,4,5])
        st.write(df.astype('object'))
       

elif section == "Collabortive Filetring On KL Dataset":
    st.title('Final Year Project')
    st.header("Collabortive Filetring On KL Dataset")
    df = pd.read_csv('clean_user.csv')
    df["restaurant_name"] = df["restaurant_name"].astype('category')
    df["restaurant_id"] = df["restaurant_name"].cat.codes
    df['Word_Rating'] = pd.cut(df['Highest_word_score'], [0 , 0.2 , 0.4 , 0.6 , 0.8 , 1], labels=[1,2,3,4,5])
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['User_ID', 'restaurant_id', 'Word_Rating']], reader)
    
    benchmark = []
    for algorithm in [SVD(), BaselineOnly(), KNNBaseline() , NormalPredictor() , KNNBasic()]:
        results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
        
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]],index=['Algorithm']))
        benchmark.append(tmp)
    
    surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
    st.subheader("Evluation of KL data with collborative filetring withount parameter tuning")
    st.write(surprise_results)
    
    
    train = Dataset.load_from_df(df[['User_ID', 'restaurant_id', 'Word_Rating']], reader)
    test = Dataset.load_from_df(df[['User_ID', 'restaurant_id', 'Word_Rating']], reader)
    val = Dataset.load_from_df(df[['User_ID', 'restaurant_id', 'Word_Rating']], reader)
    train = train.build_full_trainset()
    val_before = val.build_full_trainset()
    val = val_before.build_testset()
    test_before = test.build_full_trainset()
    test = test_before.build_testset()
    
    st.subheader("Model Prediction and Evaluation")
    select_algo = st.radio('Select a Algorithmn:', ('Baseline Only','KNNBasic','KNNBaseline', 'SVD' , 'Normal Predictor'))
    
    if select_algo == 'Baseline Only':
        bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
        algo = BaselineOnly(bsl_options=bsl_options)
        
        bias_baseline = BaselineOnly(bsl_options)
        bias_baseline.fit(train)
        
        bbase = bias_baseline.test(test)
        start_time = time.time()
        bias_baseline.fit(train)
        bias_time = ((time.time() - start_time))
        bias_time = round(bias_time,4)
        bbase_df = pd.DataFrame(bbase, columns = ['userId','itemId','rating','pred_rating','x'])
        baseline_rsme = accuracy.rmse(bbase)
        baseline_r2 = r2_score(bbase_df.rating , bbase_df.pred_rating)
        baseline_mae = mean_absolute_error(bbase_df.rating, bbase_df.pred_rating)
        
        data = [[baseline_rsme , baseline_mae ,baseline_r2 , bias_time]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        bbase_df['err'] = abs(bbase_df.pred_rating - bbase_df.rating)
        base_best_predictions = bbase_df.sort_values(by='err')[:5]
        base_worst_predictions = bbase_df.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions) 
        
        
     
            
            
    elif select_algo == 'KNNBasic':
        sim_options = {'name': 'cosine','user_based': False}
        
        algo_knn_basic = KNNBasic(sim_options=sim_options)
        algo_knn_basic.fit(train)
        
        start_time = time.time()
        algo_knn_basic.fit(train)
        knnb_time = time.time() - start_time
        knnb_time = round(knnb_time,4)
        
        knnB = algo_knn_basic.test(test)
        knn = pd.DataFrame(knnB, columns = ['userId','itemId','rating','pred_rating','x'])
        knnb_rmse = accuracy.rmse(knnB)
        knnb_r2 =  r2_score(knn.rating , knn.pred_rating)
        knnb_mae = mean_absolute_error(knn.rating, knn.pred_rating)
        
        data = [[knnb_rmse , knnb_mae ,knnb_r2 ,knnb_time ]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        knn['err'] = abs(knn.pred_rating - knn.rating)
        base_best_predictions = knn.sort_values(by='err')[:5]
        base_worst_predictions = knn.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions) 
            
    elif select_algo == 'KNNBaseline':
        sim_options = {'name': 'cosine','user_based': False}
        
        Knnb = KNNBaseline(sim_options=sim_options)
        Knnb.fit(train)
        
        knnBS = Knnb.test(test)
        start_time = time.time()
        Knnb.fit(train)
        knnb_time = time.time() - start_time
        knnb_time = round(knnb_time,4)
        Knnbb = pd.DataFrame(knnBS, columns = ['userId','itemId','rating','pred_rating','x'])
        knnbs_rmse = accuracy.rmse(knnBS)
        knnbs_r2 = r2_score(Knnbb.rating , Knnbb.pred_rating) 
        knnbs_mae =  mean_absolute_error(Knnbb.rating, Knnbb.pred_rating)
        
        data = [[knnbs_rmse , knnbs_mae ,knnbs_r2 , knnb_time ]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        Knnbb['err'] = abs(Knnbb.pred_rating - Knnbb.rating)
        base_best_predictions = Knnbb.sort_values(by='err')[:5]
        base_worst_predictions = Knnbb.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions)
            
    
    elif select_algo == 'SVD':  
        RMSE_tune = {}
        n_epochs = [10, 20, 30]  # the number of iteration of the SGD procedure
        lr_all = [0.001, 0.003, 0.005] # the learning rate for all parameters
        reg_all =  [0.02, 0.05, 0.1, 0.4, 0.5] # the regularization term for all parameters

        for n in n_epochs:
            for l in lr_all:
                for r in reg_all:
                    print('Fitting n: {0}, l: {1}, r: {2}'.format(n, l, r))
                    algo = SVD(n_epochs = n, lr_all = l, reg_all = r)
                    algo.fit(train)
                    predictions = algo.test(val)
                    RMSE_tune[n,l,r] = accuracy.rmse(predictions)
        
        best_param = min(RMSE_tune.items(), key=operator.itemgetter(1))[0]
        st.write('The Best Paramater for SVD is:', best_param )
        
        
        algo_real = SVD(n_epochs = 30, lr_all = 0.005, reg_all = 0.02)
        algo_real.fit(train)
        
        svdd = algo_real.test(test)
        start_time = time.time()
        algo_real.fit(train)
        svd_time = time.time() - start_time
        svd_time = round(svd_time,4)
        
        svd = pd.DataFrame(svdd, columns = ['userId','businessId','rating','pred_rating','x' ])
        svd_rmse = accuracy.rmse(svdd)
        svd_r2 = r2_score(svd.rating , svd.pred_rating)
        svd_mae =  mean_absolute_error(svd.rating, svd.pred_rating)
        
        data = [[svd_rmse , svd_mae ,svd_r2 , svd_time ]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        svd['err'] = abs(svd.pred_rating - svd.rating)
        base_best_predictions = svd.sort_values(by='err')[:5]
        base_worst_predictions = svd.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions)
            
            
            
    elif select_algo == 'Normal Predictor':
        algo = NormalPredictor()
        algo.fit(train)
        
        norm = algo.test(test)
        start_time = time.time()
        algo.fit(train)
        norm_time = time.time() - start_time
        norm_time = round(norm_time,4)
        norm_p = pd.DataFrame(norm, columns = ['userId','itemId','rating','pred_rating','x'])
        norm_rmse = accuracy.rmse(norm)
        norm_r2 = r2_score(norm_p.rating , norm_p.pred_rating) 
        norm_mae =  mean_absolute_error(norm_p.rating, norm_p.pred_rating)
        
        data = [[norm_rmse , norm_mae ,norm_r2 , norm_time ]]
        df = pd.DataFrame (data, columns = ['RMSE','MAE' , 'R^2' , 'Run_Time'])
        st.write(df)
        
        norm_p['err'] = abs(norm_p.pred_rating - norm_p.rating)
        base_best_predictions = norm_p.sort_values(by='err')[:5]
        base_worst_predictions = norm_p.sort_values(by='err')[-5:]
        
        select_pred = st.radio('Prediction:', ('Top 5', 'Bottom 5'))
        
        if select_pred == 'Top 5':
            st.write(base_best_predictions)
            
        elif select_pred == 'Bottom 5':
            st.write(base_worst_predictions)
            
            
            
elif section == 'Evaluation of Collabortive Filetring On KL Dataset':
        st.header("Comparison of Collaborative Filetring Algorithm")
        
        df = pd.read_csv('clean_user.csv')
        df["restaurant_name"] = df["restaurant_name"].astype('category')
        df["restaurant_id"] = df["restaurant_name"].cat.codes
        df['Word_Rating'] = pd.cut(df['Highest_word_score'], [0 , 0.2 , 0.4 , 0.6 , 0.8 , 1], labels=[1,2,3,4,5])
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['User_ID', 'restaurant_id', 'Word_Rating']], reader)
        
        train = Dataset.load_from_df(df[['User_ID', 'restaurant_id', 'Word_Rating']], reader)
        test = Dataset.load_from_df(df[['User_ID', 'restaurant_id', 'Word_Rating']], reader)
        val = Dataset.load_from_df(df[['User_ID', 'restaurant_id', 'Word_Rating']], reader)
        train = train.build_full_trainset()
        val_before = val.build_full_trainset()
        val = val_before.build_testset()
        test_before = test.build_full_trainset()
        test = test_before.build_testset()
        
        bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
        algo = BaselineOnly(bsl_options=bsl_options)
        
        bias_baseline = BaselineOnly(bsl_options)
        bias_baseline.fit(train)
        
        bbase = bias_baseline.test(test)
        start_time = time.time()
        bias_baseline.fit(train)
        bias_time = ((time.time() - start_time))
        bias_time = round(bias_time,4)
        bbase_df = pd.DataFrame(bbase, columns = ['userId','itemId','rating','pred_rating','x'])
        baseline_rsme = accuracy.rmse(bbase)
        baseline_r2 = r2_score(bbase_df.rating , bbase_df.pred_rating)
        baseline_mae = mean_absolute_error(bbase_df.rating, bbase_df.pred_rating)
        
        sim_options = {'name': 'cosine','user_based': False}
        
        algo_knn_basic = KNNBasic(sim_options=sim_options)
        algo_knn_basic.fit(train)
        
        start_time = time.time()
        algo_knn_basic.fit(train)
        knnb_time = time.time() - start_time
        knnb_time = round(knnb_time,4)
        
        knnB = algo_knn_basic.test(test)
        knn = pd.DataFrame(knnB, columns = ['userId','itemId','rating','pred_rating','x'])
        knnb_rmse = accuracy.rmse(knnB)
        knnb_r2 =  r2_score(knn.rating , knn.pred_rating)
        knnb_mae = mean_absolute_error(knn.rating, knn.pred_rating)
        
        
        Knnb = KNNBaseline(sim_options=sim_options)
        Knnb.fit(train)
        
        knnBS = Knnb.test(test)
        start_time = time.time()
        algo_knn_basic.fit(train)
        knnb_time = time.time() - start_time
        knnb_time = round(knnb_time,4)
        Knnbb = pd.DataFrame(knnBS, columns = ['userId','itemId','rating','pred_rating','x'])
        knnbs_rmse = accuracy.rmse(knnBS)
        knnbs_r2 = r2_score(Knnbb.rating , Knnbb.pred_rating) 
        knnbs_mae =  mean_absolute_error(Knnbb.rating, Knnbb.pred_rating)
        
        algo_real = SVD(n_epochs = 30, lr_all = 0.005, reg_all = 0.02)
        algo_real.fit(train)
        
        svdd = algo_real.test(test)
        start_time = time.time()
        algo_real.fit(train)
        svd_time = time.time() - start_time
        svd_time = round(svd_time,4)
        
        svd = pd.DataFrame(svdd, columns = ['userId','businessId','rating','pred_rating','x' ])
        svd_rmse = accuracy.rmse(svdd)
        svd_r2 = r2_score(svd.rating , svd.pred_rating)
        svd_mae =  mean_absolute_error(svd.rating, svd.pred_rating)
        
        
        algo = NormalPredictor()
        algo.fit(train)
        
        norm = algo.test(test)
        start_time = time.time()
        algo.fit(train)
        norm_time = time.time() - start_time
        norm_time = round(norm_time,4)
        norm_p = pd.DataFrame(norm, columns = ['userId','itemId','rating','pred_rating','x'])
        norm_rmse = accuracy.rmse(norm)
        norm_r2 = r2_score(norm_p.rating , norm_p.pred_rating) 
        norm_mae =  mean_absolute_error(norm_p.rating, norm_p.pred_rating)
        
        x_algo = [ 'BaselineOnly' , 'SVD' , 'KNNBasic' , 'KNNBaseline' , 'Normal Predictor' ]
        all_algos_cv = [baseline_rsme,svd_rmse,knnb_rmse,knnbs_rmse , norm_rmse]
        mae_cv= [baseline_mae,svd_mae,knnb_mae,knnbs_mae , norm_mae ]
        r2_cv = [baseline_r2,svd_r2,knnb_r2,knnbs_r2 , norm_r2]
           
        
        select_pred = st.radio('Evaluation:', ('RMSE', 'MAE', 'R^2'))
        
        if select_pred == 'RMSE':
           plt.figure(figsize=(10,5))
           plt.title('Comparison of Algorithms on RMSE', loc='center', fontsize=15)
           plt.plot(x_algo, all_algos_cv, label='RMSE', color='darkgreen', marker='o')
           plt.xlabel('Algorithms', fontsize=15)
           plt.ylabel('RMSE Value', fontsize=15)
           plt.legend()
           plt.grid(ls='dashed')
           st.pyplot()
           
        elif select_pred == 'MAE':
           plt.title('Comparison of Algorithms on MAE', loc='center', fontsize=15)
           plt.plot(x_algo, mae_cv, label='MAE', color='navy', marker='o')
           plt.xlabel('Algorithms', fontsize=15)
           plt.ylabel('MAE Value', fontsize=15)
           plt.legend()
           plt.grid(ls='dashed')
           st.pyplot()
           
           
        elif select_pred == 'R^2':
           plt.title('Comparison of Algorithms on R^2', loc='center', fontsize=15)
           plt.plot(x_algo, r2_cv, label='R^2', color='red', marker='o')
           plt.xlabel('Algorithms', fontsize=15)
           plt.ylabel('R^2 Value', fontsize=15)
           plt.legend()
           plt.grid(ls='dashed')
           st.pyplot()
   
    
elif section == "KL Restaurant Recommender System":

    st.title('Final Year Project')
    st.header("Recommender System")
    st.subheader('What is on mind to eat?')
    
    kl_data = pd.read_csv('kl_data.csv' , delimiter = ",")
    kl_data["contact_number"].fillna("No contact", inplace = True)
    kl_data["address"].fillna("No Address", inplace = True) 
    kl_data["review_count"].fillna("No Review", inplace = True) 
    kl_data["categories"].fillna("No Categories", inplace = True) 
    kl_data["pricing"].fillna("No pricing", inplace = True) 
    kl_data["rating"].fillna("No Review", inplace = True) 

    
    data = pd.read_csv('clean_user.csv')
    username = data[['User_ID','comment']]
    userid_df = username.groupby('User_ID').agg({'comment': ' '.join})
    username = data[['User_ID','comment']]
    userid_df = username.groupby('User_ID').agg({'comment': ' '.join})
    resturant_id = data[['restaurant_name','comment']]
    business_df = resturant_id.groupby('restaurant_name').agg({'comment': ' '.join})
    
    
    userid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=3000)
    userid_vectors = userid_vectorizer.fit_transform(userid_df['comment'])

    
    
    businessid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=3000)
    businessid_vectors = businessid_vectorizer.fit_transform(business_df['comment'])
    business_matrix = pd.DataFrame(businessid_vectors.toarray(), index=business_df.index, columns=businessid_vectorizer.get_feature_names())
    
    stop = []
    for word in stopwords.words('english'):
        s = [char for char in word if char not in string.punctuation]
        stop.append(''.join(s))
    
    user_input = st.text_input("Enter choice" , 'food')
    
    test_df= pd.DataFrame([user_input], columns=['comment'])
    test_df['comment'] = test_df['comment'].apply(text_process)
    test_vectors = userid_vectorizer.transform(test_df['comment'])
    test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())
    predict_item_rating=pd.DataFrame(np.dot(test_v_df.loc[0],business_matrix.T),index=business_matrix.index,columns=['Rating'])
    top_recommendations=pd.DataFrame.sort_values(predict_item_rating,['Rating'],ascending=[0])[:3]
    
    for i in top_recommendations.index:
        st.write((kl_data[kl_data['restuarant_name']==i]['restuarant_name'].iloc[0]))
        st.write((kl_data[kl_data['restuarant_name']==i]['categories'].iloc[0]))
        st.write(str(kl_data[kl_data['restuarant_name']==i]['rating'].iloc[0]))
        st.write(str(kl_data[kl_data['restuarant_name']==i]['restuarant_url'].iloc[0]))
        st.write('')
        st.write('')
    
     
    
    
