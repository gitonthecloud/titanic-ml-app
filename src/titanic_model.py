import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from contextlib import redirect_stdout

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class TitanicModel:
  def __init__(self):
    self.df = None
    self.feat = None
    self.targ = None
    self.X_test = None
    self.y_test = None
    self.model = None
    self.predictions = None
    self.cm = None
    self.cr = None
    self.new_passenger_df = None


  def load_data(self, csv_file: str):
    # Incorporate data
    # This is the best practice for storing and reading local source file
    csv_path = Path(__file__).parent/'../data'/csv_file
    self.df = pd.read_csv(csv_path)

    # self.df = pd.read_csv(csv_file)
    return self.df.copy()

  def df(self):
    return self.df

  def df_info(self):
    # Prints to the stream fs instead of stdout
    fs = io.StringIO()
    with redirect_stdout(fs):
      print(self.df.info())
    return fs.getvalue()

  def df_info_type(self):
    # Prints to the stream fs instead of stdout
    fs = io.StringIO()
    with redirect_stdout(fs):
      cols = self.df.columns.values.tolist()
      for col in cols:
        print(col, ' ', type(self.df[col][0]))
    return fs.getvalue()

  ''' Returns an image showing missing data as horizontal lines'''
  def missing_data(self):
    missing_data_hm = px.imshow(
      self.df.isnull(),
      width=600,
      height=800,
      color_continuous_scale='viridis',
      title='Visualising missing data (if any, will show up as horizontal lines)')
    return missing_data_hm

  def survival_plot(self):
    fig = px.histogram(data_frame=self.df, x='Survived', width=500, height=300,
                       color='Survived',
                       color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig.update_layout(
      title='How many survivors and deceased are there <br>in the Titanic disaster?',
      bargap=0.1,
      template='plotly_white')
    return fig

  def survival_plot_by_sex(self,df=None):
    src = df if df is not None else self.df
    fig = px.histogram(data_frame=src, x='Survived', width=500, height=300,
                       color='Sex',
                       barmode='group',
                       color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig.update_layout(
      title='How many male and female survivors/deceased <br>are there in the Titanic disaster?',
      bargap=0.1,
      template='plotly_white')
    return fig

  def survival_by_pclass(self):
    fig = px.histogram(data_frame=self.df, x='Survived', width=500, height=300,
                       color='Pclass', barmode='group',
                       color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig.update_layout(
      title='Does passenger class affect survival rate?',
      bargap=0.1,
      template='plotly_white')
    return fig

  def age_plot(self):
    fig = px.histogram(data_frame=self.df['Age'].dropna(), x='Age',
                       nbins=40, width=500, height=300)
    fig.update_layout(
      title='What age are the passengers?',
      bargap=0.1,
      template='plotly_white')
    return fig

  def relationship_plot(self):
    fig = px.histogram(data_frame=self.df, x='SibSp', width=500, height=300,
                       color='SibSp',
                       color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig.update_layout(
      title='Are the passengers related to each other?',
      bargap=0.1,
      template='plotly_white')
    return fig

  def fare_plot(self):
    fig = px.histogram(data_frame=self.df, x='Fare', width=500, height=300,
                       nbins=60)
    fig.update_layout(
      title='How much did the passengers pay?',
      bargap=0.1,
      template='plotly_white')
    return fig

  def mean_age_box_plot(self):
    fig = px.box(data_frame=self.df, x='Pclass', y='Age', color='Pclass',
                 width=700, height=500)
    fig.update_layout(
      title='Mean age of passengers by class',
      bargap=0.1,
      template='plotly_white')
    return fig

  def impute_missing_age(self):
    lookup_tab = self.df.groupby('Pclass')['Age'].mean().to_dict()
    self.df['Age'] = self.df.apply(lambda x: lookup_tab[x['Pclass']] if np.isnan(x['Age']) else x['Age'], axis=1)
    return self.df.copy()

  def clean_cabin(self):
    self.df.drop('Cabin',axis=1,inplace=True)
    return self.df.copy()

  def clean_embarked(self):
    self.df.dropna(inplace=True)
    return self.df.copy()

  def dummy_variables_for_sex(self):
    sex = pd.get_dummies(data=self.df['Sex'],drop_first=True,dtype='int')
    try:
      self.df.drop('Sex',axis=1,inplace=True)
    except KeyError as err:
      print(err)
    self.df = pd.concat([self.df,sex],axis=1)
    return self.df.copy()

  def dummy_variables_for_embarked(self):
    embarked = pd.get_dummies(data=self.df['Embarked'],drop_first=True,dtype='int')
    try:
      self.df.drop('Embarked',axis=1,inplace=True)
    except KeyError as err:
      print(err)
    self.df = pd.concat([self.df,embarked],axis=1)
    return self.df.copy()

  def drop_name_ticket(self):
    try:
      self.df.drop(['Name','Ticket'],axis=1,inplace=True)
    except KeyError:
      pass
    return self.df.copy()

  def features(self):
    return self.df.drop('Survived',axis=1)

  def target(self):
    return pd.DataFrame(self.df['Survived'])

  def train_test_split(self):
    X_train, X_test, y_train, y_test = train_test_split(self.df.drop('Survived', axis=1),
                                                        self.df['Survived'], test_size=0.30,
                                                        random_state=101)
    self.feat = X_train
    self.targ = y_train
    self.X_test = X_test
    self.y_test = y_test
    return (X_train,X_test,pd.DataFrame(y_train),pd.DataFrame(y_test))

  def train_logistic_regression_model(self):
    self.model = LogisticRegression(max_iter=1000)
    self.model.fit(self.feat, self.targ)
    explain = ['Passenger Id does not affect the survival rate',
               'Every unit of increase in passenger class decreases survival rate significantly',
               'Every unit of increase in age leads to less likelihood of survival',
               'Every unit of increase in sibling/spouse decreases survival rate alot',
               'NA',
               'Every unit of increase in fare leads to a very small increase in survival',
               'Being a male decreases the survival rate significantly',
               'Embarking from location Q decreases the survival rate a bit',
               'Embarking from location S decreases survival rate three times more than from location S']
    df_coef = pd.concat(
      [pd.DataFrame(self.feat.columns.values), pd.DataFrame(self.model.coef_).transpose(), pd.DataFrame(explain)], axis=1)
    df_coef.columns = ['Feature', 'Coefficient', 'Explanation']
    return df_coef

  def model_coef_plot(self, df_coef):
    fig = px.bar(data_frame=df_coef.sort_values(by='Coefficient'), x='Feature', y='Coefficient')
    fig.update_layout(
      title='Which feature influences survival rate the most?',
      bargap=0.1,
      template='plotly_white')
    return fig

  def predict(self):
    self.predictions = self.model.predict(self.X_test)
    return self.predictions

  def compare_act_pred(self):
    df = pd.concat([pd.DataFrame(self.y_test.values), pd.DataFrame(self.predictions)], axis=1)
    df.columns = ['Actual', 'Predicted']
    return df

  def confusion_mat(self):
    self.cm = confusion_matrix(self.y_test,self.predictions)
    return px.imshow(self.cm, title='Confusion Matrix (Actual vs Predicted Survival)',
                     width=500, height=500, text_auto=True)

  def classification_rep(self):
    self.cr = classification_report(self.y_test,self.predictions,output_dict=True)

  def classification_report_matrix(self,cr):
    indexes = list(cr.keys())
    cols = ['precision', 'recall', 'f1-score', 'support']
    crm = []

    # In dictionary form, accuracy only has a singular value for the f1-score
    # and others are not present. Therefore, assume accuracy's support is same as weighted avg's
    # NOTE: Use a sentinel value to denote None ... how else?
    accu_f1_score = cr['accuracy']
    cr['accuracy'] = {
      'precision': -1,
      'recall': -1,
      'f1-score': accu_f1_score,
      'support': cr['weighted avg']['support']
    }

    for idx in indexes:
      try:
        vals = []
        for col in cols:
          vals.append(cr[idx][col])
        crm.append(vals)
      except Exception as err:
        print(err)

    return (indexes, cols, crm)

  def classification_report_plot(self):
    (idxs, metrics, crm) = self.classification_report_matrix(self.cr)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Adding the text
    for (i, j), val in np.ndenumerate(crm):
      if val != -1:
        if j == 3:
          ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
        else:
          ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    ax.set_xticks(ticks=range(len(metrics)), labels=metrics)
    ax.set_yticks(ticks=range(len(idxs)), labels=idxs)

    ms = ax.matshow(Z=crm)
    plt.colorbar(ms)

    file = Path(__file__).parent/'../assets/titanic_cr.png'
    plt.savefig(file)


  def new_passenger(self):
    # new_passenger = [
    #   random.randint(1000, 2000),
    #   random.randint(1, 3),
    #   random.randint(1, 100),
    #   random.randint(0, 1),
    #   random.randint(0, 1),
    #   random.randint(0, 100),
    #   bool(random.randint(0, 1)),
    #   bool(random.randint(0, 1)),
    #   bool(random.randint(0, 1))
    # ]
    # np_df = pd.DataFrame(new_passenger).transpose()
    # np_df.columns = self.feat.columns

    new_passenger = {
        'PassengerId':random.randint(1000,2000),
        'Pclass':random.randint(1,3),
        'Age':random.randint(1,100),
        'SibSp':random.randint(0,1),
        'Parch':random.randint(0,1),
        'Fare':random.randint(0,100),
        'male':bool(random.randint(0,1)),
        'Q':bool(random.randint(0,1)),
        'S':bool(random.randint(0,1))
    }
    np_df = pd.DataFrame(new_passenger,index=[0])

    self.new_passenger_df = np_df
    np_pred = self.model.predict(np_df)
    np_pred_df = pd.DataFrame([np_pred[0]], columns=['Survived^'])

    return pd.concat([np_df,np_pred_df],axis=1)

  def new_passengers(self):
    passengers = []
    new_pred = []
    np_df = None
    for i in range(5):
      embarked_at_Q = bool(random.randint(0, 1))
      embarked_at_S = not embarked_at_Q
      new_passenger = [
        random.randint(1000, 2000),
        random.randint(1, 3),
        random.randint(1, 100),
        random.randint(0, 1),
        random.randint(0, 1),
        random.randint(0, 100),
        bool(random.randint(0, 1)),
        embarked_at_Q,
        embarked_at_S
      ]
      passengers.append(new_passenger)

      np_df = pd.DataFrame(new_passenger).transpose()
      np_df.columns = self.feat.columns
      np_pred = self.model.predict(np_df)
      new_pred.append(np_pred[0])

      # self.new_passenger_df = np_df

    nps_df = pd.DataFrame(passengers, columns=self.feat.columns)
    new_pred_df = pd.DataFrame(new_pred, columns=['Survived^'])

    return pd.concat([nps_df,new_pred_df],axis=1)