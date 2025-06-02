import pandas as pd
from dash import html, dcc
from dash import dash_table as dt
from titanic_model import TitanicModel


class ML:
  def __init__(self):
    self.tm = TitanicModel()

    # Dash layout component list
    self.dcl = []

    self.df = None


  def run(self):
    self.df = self.tm.load_data('titanic_train.csv')

    title = html.H1('A machine model for predicting Titanic survival')
    bg_text = html.H2('Background')

    background_text = '''
    The Titanic was the largest and most luxurious ocean liner that operated in the
    1900 era. It struck an iceberg during its maiden voyage on 15 April 1912 and sunk.
    Statistics of the disaster were collected into the Titanic data set which we will
    use to build a machine model that predicts the chance of survival.'
    '''
    bg_text_md = dcc.Markdown(background_text)
    hdr = html.Div([html.Br(), title, bg_text_md, html.Br(), html.Br()],
                   style={
                     'background-color': 'skyblue',
                     'border-radius': '30px',
                     'padding': '10px'
                   })

    eda_hdr = html.H2('Exploratory Data Analysis')
    eda_text1 = html.Div('We begin by checking out the data structure of a few samples:')
    eda_table = dt.DataTable(self.df.head().to_dict('records'))

    eda_add_info = html.H5(children='Additional Information:')

    eda_info = html.Div(self.tm.df_info(),
             style={'font-family': 'monospace',
                    'font-weight': 'lighter',
                    'font-size': 'math',
                    'whiteSpace': 'pre-wrap'})

    eda_info_typ = html.Div(self.tm.df_info_type(),
             style={'font-family': 'monospace',
                    'font-weight': 'lighter',
                    'font-size': 'math',
                    'whiteSpace': 'pre-wrap'})

    info_text = (f'From the above data, we can see that the data columns are {", ".join(self.df.columns.values.tolist())}.'
                 f' The data types are a mixture of integers and strings. The data set contains 891 rows with 12 columns each.')
    eda_info_text = dcc.Markdown(info_text)

    eda_stats = html.H5('Statistical Summary:')
    eda_stats_table = dt.DataTable(self.df.describe().reset_index().to_dict('records'))

    mdata = html.Div('Next, check for missing data.')
    mdata_table = dt.DataTable(pd.DataFrame(self.df.isnull().sum()).transpose().to_dict('records'))
    mdata_graph = dcc.Graph(figure=self.tm.missing_data())
    mdata_text = html.Div('The above information shows that there are missing data ' \
              + 'in the "Age", "Cabin" and "Embarked" columns. ' \
              + 'The missing "Age" data accounts for only ' \
              + str(round(self.df["Age"].isnull().sum() / self.df.shape[0] * 100, 1)) + '%. ' \
              + 'This is small enough for applying the imputation method. ' \
              + 'On the other hand, the missing "Cabin" data is too much and requires a different way to be resolved. ' \
              + 'One way would be to convert it to a categorical feature such as "Cabin Known: 1 or 0". ' \
              + 'This will be examined further later in the workflow.')

    desc_text = '''
    Continuing with the analysis, the data set contains known features
    and known target label(i.e. the "Survived" column).
    Therefore, lets visualize that data to get a sense of its
    content and distribution.
    '''
    eda_text2 = dcc.Markdown(desc_text)

    surv_graph = dcc.Graph(figure=self.tm.survival_plot())
    surv_text = html.Div('There is no extreme difference between the number of ' \
              + 'and deceased; meaning that the target labels are reasonably ' \
              + 'well balanced. Therefore, the accuracy metric can be used as ' \
              + 'a good indicator of the model performance.')

    surv_sex_graph = dcc.Graph(figure=self.tm.survival_plot_by_sex(self.df))
    surv_sex_text = html.Div('It seems females survived twice more than males!')

    surv_pclass_graph = dcc.Graph(figure=self.tm.survival_by_pclass())
    surv_pclass_text = html.Div('Class 3 passengers are much more likely to die! ' \
              + 'That is probably because it is at the lowest deck of the ' \
              + 'ship where rescue efforts were scarce, difficult and ' \
              + 'time-consuming.')

    age_text = html.Div('Now, lets examine the age of the passengers...')
    age_graph = dcc.Graph(figure=self.tm.age_plot())
    age_text2 = html.Div('Looks like the majority of the passengers are aged between 20 and 30.')

    rel_text = html.Div('This data set also contain information on relationship ' \
              + 'between the passengers, namely, the "SibSp" column. It specifies ' \
              + 'if there are singles, spouse or children, etc on board. ' \
              + 'Lets check it out...')
    rel_graph = dcc.Graph(figure=self.tm.relationship_plot())
    rel_text2 = html.Div('Looking at this graph, it is clear that the majority of ' \
              + 'passengers are single. They probably made up a large portion ' \
              + ' of the passengers in class 3. The next biggest group ' \
              + ' are the couples, followed by "single parent with child". ')

    fare_text = html.Div('Lastly, lets explore what the "Fare" column is telling us...')
    fare_graph = dcc.Graph(figure=self.tm.fare_plot())
    fare_text2 = html.Div('From this graph, we can see that the majority of ' \
              + 'passengers paid the lowest amount which again correlated ' \
              + 'with the large number of passengers in the lowest class (class 3).')

    eda_summ = html.H5('Exploratory Data Analysis Summary')
    eda_summ_text = html.Div('The above exploratory analysis has given us some ' \
                      + 'valuable insights and a good sense of the contextual ' \
                      + 'information of the data set. We have learned that:')
    eda_summ_list = html.Div(
      html.Ul([
        html.Li(
          'There are some missing data that needs pre-processing, specifically the '
          '"Age", "Cabin" and "Embarked" columns'),
        html.Li('Females were twice as likely to survive compared to males'),
        html.Li(
          'In particular, mortality rate of males aged between 20 and 30 who travelled in '
          'class 3 is significantly higher'),
        html.Li('The target label "Survived" is known and contains binary data, therefore, '
                'we will build a logistic model to predict chance of survival')
      ]))

    dc_hdr = html.H2(children='Data Cleaning')
    dc_text = html.Div(children='In the exploratory data analysis phase, we have ' \
             + 'identified some columns that have missing data. The "Age" ' \
             + 'column has low amount of missing data and can be filled in ' \
             + 'using the mean age of all passengers. Lets examine that ' \
             + 'statistic visually.')

    ma_graph = dcc.Graph(figure=self.tm.mean_age_box_plot())
    ma_text = html.Div(children='From this graph, we can see that the mean age of ' \
             + 'passengers tend to increase from third class to ' \
             + 'first class, thus, we can infer that the wealthier ' \
             + 'passengers tend to be older. This parameter is suitable ' \
             + 'for imputing the missing data in the "Age" column. After ' \
             + 'applying this imputation, confirm that the "Age" column ' \
             + 'has no missing data:')

    ima = self.tm.impute_missing_age()
    ima_table = dt.DataTable(data=pd.DataFrame(ima.isnull().sum()).transpose().to_dict('records'))
    ima_graph = dcc.Graph(figure=self.tm.missing_data())

    cabin_text = html.Div(children='Examining the cabin data further, they look like ' \
             + 'id of the cabin that the passenger was residing in. ' \
             + 'Since we already know the categorical class of the cabin, ' \
             + 'the specificity of this information does not add any ' \
             + 'further insights. Therefore, this column can be dropped. ' \
             + 'After applying this change, check the data set to confirm: ')
    ccabin = self.tm.clean_cabin()
    ccabin_table = dt.DataTable(data=pd.DataFrame(ccabin.isnull().sum()).transpose().to_dict('records'))
    ccabin_graph = dcc.Graph(figure=self.tm.missing_data())

    emb_text = html.Div(children='The "Embarked" column tells us where the passenger got on ' \
            + 'the ship and only has two missing items, thus, dropping ' \
            + 'their respective rows would be alright. After applying ' \
            + 'these changes, check the data set to confirm: ')
    cemb = self.tm.clean_embarked()
    cemb_table = dt.DataTable(data=pd.DataFrame(cemb.isnull().sum()).transpose().to_dict('records'))
    cemb_graph = dcc.Graph(figure=self.tm.missing_data())

    clean_text = html.Div('The data set is now clean. It is now ready for the ' \
             + 'next phase of data cleaning, categorical feature processing. ')

    cfp_hdr = html.H2('Categorical Feature Processing')
    cfp_text = html.Div('A close look at the data set reveals that some of the ' \
             + 'columns contain categorical data i.e. their values come from ' \
             + 'a pre-defined finite set. These columns are "Sex" and "Embarked" ' \
             + 'but the values are in string format. Machine learning algorithm ' \
             + 'do not understand this format. They needs to be converted to ' \
             + 'dummy variables (0 or 1). Furthermore, the conversion should also ' \
             + 'resolve the multi-collinearity problem. This can be achieved by ' \
             + 'dropping the first converted column values. Lets do that.')

    dfe_hdr = html.H5('Dummy Feature Extraction')
    dfe_text = html.Div('The "Sex" column values are "Male" and "Female" which ' \
             + 'can be represented with boolean values using the dummy variable ' \
             + 'conversion step. Note that the original column is no longer ' \
             + 'needed after the conversion and thus is removed. After this ' \
             + 'conversion, the data looks like:')
    dm_sex = self.tm.dummy_variables_for_sex()
    dm_sex_table = dt.DataTable(data=dm_sex.to_dict('records'), page_size=10)

    dm_emb_text = html.Div('Similarly, apply the dummy variable conversion to the ' \
             + '"Embarked" column. After this conversion, the data looks like:')
    dm_emb = self.tm.dummy_variables_for_embarked()
    dm_emb_table = dt.DataTable(data=dm_emb.to_dict('records'), page_size=10)

    nm_tkt_text = html.Div('Regarding the "name" column, it may be possible to extract ' \
                    + 'some feature from it but for the purpose of the model we ' \
                    + 'trying to build, they may not be useful. Therefore, we will ' \
                    + 'drop this column. Similarly, the "Ticket" column will be ' \
                    + 'dropped as well. After this, the data look like:')
    no_nm_tkt = self.tm.drop_name_ticket()
    no_nm_tkt_table = dt.DataTable(data=no_nm_tkt.to_dict('records'), page_size=10)

    ready_text = html.Div('Now, the dataset is ready for next phase of the workflow i.e. building a model.')

    mod_hdr = html.H2('Choosing and building a model')
    mod_text = html.Div('How do we determine what model should be used? The answer lies in the context of ' \
             + 'the problem. Since we are trying to build a model to predict Titanic survival and the ' \
             + 'data set contains a "Survived" column which contains discrete binary data (i.e. class ' \
             + '0 or class 1), we can conclude that the model should be a binary logistic classifier.')

    sd_hdr = html.H5('Split data set')
    sd_text = html.Div('The next step we need to take is split the data set into training set and test set. ' \
             + 'A list of features are constructed from all the columns in the data set except the "Survived" ' \
             + 'column which will be the target label. Therefore, we have the following:')

    feat_text = html.Div('Features:')
    features = self.tm.features()
    target = self.tm.target()
    feat_table = dt.DataTable(data=features.to_dict('records'), page_size=10)
    targ_text = html.Div('Target:')
    targ_table = dt.DataTable(data=target.to_dict('records'), page_size=10)

    train_text = html.Div('After the data split, we have the following training data:')
    X_train, X_test, y_train, y_test = self.tm.train_test_split()
    xtrain_table = dt.DataTable(data=X_train.to_dict('records'), page_size=10)
    ytrain_table = dt.DataTable(data=y_test.to_dict('records'), page_size=10)

    tmod_hdr = html.H5('Train the logistic regression model')
    tmod_text = html.Div('After creating a logistic regression model and training it with '\
            + 'the above training data, it has the following parameters:')
    coef = self.tm.train_logistic_regression_model()
    coef_table = dt.DataTable(data=coef.to_dict('records'), page_size=10)
    coef_graph = dcc.Graph(figure=self.tm.model_coef_plot(coef))

    coef_text = html.Div('This shows that the most influential factors that decreases ' \
      + 'the chance of survival are: ')
    coef_list = html.Div(
      html.Ol([
        html.Li('Being a male'),
        html.Li('Passenger class'),
      ]))
    coef_text2 = html.Div('At this point, we should validate if the above data make sense by cross checking with the ' \
             + 'statistics we extracted from the data set during exploratory data analysis, as well as, validating ' \
             + 'them with domain knowledge. We can infer that these data correlate well in both respect. Therefore, ' \
             + 'the model has been trained reasonably well.')

    eval_hdr = html.H2('Evaluation')
    eval_text = html.Div('We can now use the model to make predictions using the test data. '\
                  + 'Here is a comparison between the actual outcome vs the predicted outcome:')
    preds = self.tm.predict()
    comp = self.tm.compare_act_pred()
    eval_table = dt.DataTable(data=comp.to_dict('records'), page_size=10, fill_width=False)

    perf_text = html.Div("Subsequently, we can evaluate the model's performance using the "\
                  + "confusion matrix and classification report.")

    cmplot = self.tm.confusion_mat()
    self.tm.classification_rep()
    self.tm.classification_report_plot()
    cm_graph = dcc.Graph(figure=cmplot)
    cm_text = html.Div('The Confusion Matrix above showed that the model has misclassified ' + str(14 + 32) + ' out of '\
             + str(149 + 72 + 14 + 32) + ' target labels i.e. a 83% accuracy. This is also shown in the Classification ' \
             + 'Report below.')
    cr_graph = html.Img(src='assets/titanic_cr.png', alt='Classification Report image')

    pred_hdr = html.H2('Making new prediction')
    pred_text = html.Div('The model is now ready to make prediction for new passengers with different '\
          + 'features. Here is a listing of new passengers with randomly generated features and their '\
          + 'survival predicted by the model:')
    np = self.tm.new_passengers()
    pred_table = dt.DataTable(data=np.to_dict('records'))

    fn_hdr = html.H5('Final Note')
    fn_text = html.Div('We have examined the Titanic disaster data set and have learned that:')
    fn_list = html.Div(
      html.Ul([
        html.Li('A class 3 male passenger has the highest likelihood of dying, and'),
        html.Li('A female passenger is much more likely to survive than a male passenger in the same class'),
      ]))
    fn_text2 = html.Div('We have built a machine model that makes prediction based on these insights.')


    self.dcl = [
      hdr,
      html.Hr(), eda_hdr, eda_text1, eda_table,
      html.Br(), eda_add_info, eda_info, eda_info_typ, eda_info_text,
      html.Br(), eda_stats, eda_stats_table,
      html.Br(), mdata, mdata_table, mdata_graph, mdata_text,
      html.Br(), eda_text2,
      html.Br(), surv_graph, surv_text,
      html.Br(), surv_sex_graph, surv_sex_text,
      html.Br(), surv_pclass_graph, surv_pclass_text,
      html.Br(), age_text, age_graph, age_text2,
      html.Br(), rel_text, rel_graph, rel_text2,
      html.Br(), fare_text, fare_graph, fare_text2,
      html.Br(), eda_summ, eda_summ_text, eda_summ_list,
      html.Hr(), dc_hdr, dc_text,
      html.Br(), ma_graph, ma_text,
      html.Br(), ima_table, ima_graph,
      html.Br(), cabin_text, ccabin_table, ccabin_graph,
      html.Br(), emb_text, cemb_table, cemb_graph,
      html.Br(), clean_text,
      html.Hr(), cfp_hdr, cfp_text,
      html.Br(), dfe_hdr, dfe_text, dm_sex_table,
      html.Br(), dm_emb_text, dm_emb_table,
      html.Br(), nm_tkt_text, no_nm_tkt_table,
      html.Br(), ready_text,
      html.Br(), mod_hdr, mod_text,
      html.Br(), sd_hdr, sd_text,
      html.Br(), feat_text, feat_table, targ_text, targ_table,
      html.Br(), train_text, xtrain_table, ytrain_table,
      html.Br(), tmod_hdr, tmod_text, coef_table, coef_graph,
      html.Br(), coef_text, coef_list, coef_text2,
      html.Hr(), eval_hdr, eval_text, eval_table,
      html.Br(), cm_graph, cm_text, cr_graph,
      html.Br(), pred_hdr, pred_text, pred_table,
      html.Br(), fn_hdr, fn_text, fn_list, fn_text2
    ]

    return self.dcl
