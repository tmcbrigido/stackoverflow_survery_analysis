import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import style

# Import data

df = pd.read_csv('survey_results_public.csv')

# Print head

df.head(5)

# List of the questions

list(df.columns)

df.info()

# Total Sample

df.shape

# Number of Respondends by Country

df.country_count = df.groupby(['Country'])['Respondent'].count().sort_values(ascending=False)
df.country_count.head(10)

# Print the top 15 Countries

mpl.rc('figure',figsize=(10,10))
style.use('fast')
df.country_count.head(15).plot(kind='bar',label='Top 15 Countries')
plt.legend

# Analyse Employment for US

df['Employment'].unique()

filter_by_usa = df[(df['Country'] == 'United States')]

filter_by_usa.head(5)

employment = filter_by_usa['Employment'].value_counts(normalize=True)

print(employment)

# Display Results

mpl.rc('figure',figsize=(10,10))
style.use('fast')
employment.plot(kind='bar',label='Employment %')
plt.legend

# Contribution for Open Source among americans

df['OpenSourcer'].unique() #Contribution to Open Source

df['OpenSource'].unique() # Quality of OSS Projects vs Closed Source Code


# Contribution to Open Source

open_sourcer =filter_by_usa['OpenSourcer'].value_counts(normalize=True)

open_sourcer.plot(kind='bar')

# Quality of OSS

open_source = filter_by_usa['OpenSource'].value_counts(normalize=True)

open_source.plot(kind='bar')




developers = df[df['DevType'].notnull()]

developers.count()

unique_developers = {}

for developer_set in developers['DevType'].apply(lambda row:str(row).split(';')):
    for developer in developer_set:
        if developer not in unique_developers.keys():
            unique_developers[developer] = 1
        else:
            unique_developers[developer] += 1

print(unique_developers)


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.barh(*zip(*sorted(unique_developers.items())))
plt.show()

# Most Popular Languages in the US Community

filter_by_usa['LanguageWorkedWith'].unique()

languages = filter_by_usa['LanguageWorkedWith'].str.split(';',expand=True)

print(languages)

languages_stack = languages.stack().value_counts()

print(languages_stack)

languages_stack.plot(kind='bar', figsize=(15,15), color="b")

# Are you happy with your job?

filter_by_usa['JobSat'].unique()

satisfaction = filter_by_usa['JobSat'].value_counts(normalize=True)

print(satisfaction)

satisfaction.plot(kind='barh')

# BetterLife

betterlife = filter_by_usa['BetterLife'].value_counts(normalize=True)

betterlife.plot(kind='pie')


# OrgSize and Better Life [WIP]

OrgSize = filter_by_usa['OrgSize'].value_counts(normalize=True)


OrgSize.plot(kind='bar')


x = filter_by_usa.groupby('OrgSize').apply(lambda x:x[x['BetterLife'] == "Yes"]).reset_index(drop=True)


x['OrgSize'].value_counts(normalize=True)


y = filter_by_usa.groupby('OrgSize').apply(lambda x:x[x['BetterLife'] == "No"]).reset_index(drop=True)

y['OrgSize'].value_counts(normalize=True)


z = filter_by_usa.groupby(['OrgSize','BetterLife']).size().value_counts(normalize=True)

z.plot(kind='bar')


# Plot the salary distribution for US 

fig = plt.figure(figsize=(15,10))
distribution = filter_by_usa['ConvertedComp'].value_counts().sort_values(ascending=False).index.tolist()

sns.distplot(distribution, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


# Average Salary US


filter_by_usa['ConvertedComp'].mean() # Gives an average of $249,546

filter_by_usa['ConvertedComp'].dropna().reset_index().mean() # Gives the same average $249,546


# Histogram with 4 different binwidths

for i, binwidth in enumerate([1, 5, 10, 15]):
    
    # Set up the plot
    ax = plt.subplot(2, 2, i + 1)
    
    # Draw the plot
    ax.hist(distribution, bins = int(180/binwidth),
             color = 'blue', edgecolor = 'black')
    
    # Title and labels
    ax.set_title('Histogram with Binwidth = %d' % binwidth, size = 30)
    ax.set_xlabel('Salaries', size = 22)
    ax.set_ylabel('Respondents', size= 22)

plt.tight_layout()
plt.show()


# Check the standard deviation since the histograms and the density plot are too skewed

filter_by_usa['ConvertedComp'].std() #Std is $452,103

# What is the max and min salaries?

filter_by_usa['ConvertedComp'].max() # Max = $2 000 000

filter_by_usa['ConvertedComp'].min() # Min = $0

filter_by_usa['ConvertedComp'].dropna().reset_index().min()


# CodeRev

CodeRev = df['CodeRev'].value_counts(normalize=True)

CodeRev.plot(kind='barh',figsize=(6,6), color="b")


# Salary distribuition for Python over top 10 countries


python = df.dropna().reset_index()

python_salaries = python[python['LanguageWorkedWith'].str.contains('Python')]

python_salaries['LanguageWorkedWith'].unique()


fig = plt.figure(figsize=(15,10))

countries = python_salaries['Country'].value_counts().sort_values(ascending=False)[:9].index.tolist()

for i,country in enumerate(countries):
    plt.subplot(3,3,i+1)
    temp_salaries = python_salaries.loc[python_salaries['Country']==country,'ConvertedComp']

    ax = temp_salaries.plot(kind='kde')
    ax.axvline(temp_salaries.mean(), linestyle = '-', color = 'red')
    ax.text((temp_salaries.mean() + 1500), (float(ax.get_ylim()[1])*0.55), 'mean = $ ' + str(round(temp_salaries.mean(),0)), fontsize = 12)
    ax.set_xlabel('Annual Salary in USD')
    ax.set_xlim(-temp_salaries.mean(),temp_salaries.mean()+2*temp_salaries.std())
    
    ax.set_title('Annual Salary Distribution in {}'.format(country))

plt.tight_layout()
plt.savefig('developer_salaries_by_country.png',bbox_inches = 'tight')
plt.show()


# Gender Equality in Tech

gender = df.groupby(['Gender'])['Respondent'].count().sort_values(ascending=False)
gender.head(10)


mpl.rc('figure',figsize=(10,10))
style.use('fast')
gender.head(15).plot(kind='bar',label='Gender')
plt.legend