'''
Justin Ehly
MSDS 7335 - Machine Learning II, early class, Wednesdays 630p
Homeworork 3: Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations. 
# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.  
# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.  
'''

from io import IncrementalNewlineDecoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
from faker import Faker
import os
import json
os.chdir(r'C:\Users\justi\Google Drive\_SMU\7335 Machine Learning II\HW3\Ehly_HW3_ML2_Early\data')
'''
####################################################################################################################

# Transform the restaurant data into a matrix(M_resturants) use the same column index.

# The most imporant idea in this project is the idea of a linear combination.  

# Problem 1: Informally describe what a linear combination is  and how it will relate to our resturant matrix.

- Linear combination is where 2 matrices (W (w by n) * H(n by h)) are combined to form a dot product V(w by h).
- Where we use column vectors in W and the H vectors are the coefficients
- From V we can sum each column to determine an overall score by person or rows to determine an overall score by restaurant
'''


####################################################################################################################
# Data Preparation
####################################################################################################################
# People
####################################################################################################################
random.seed(42)
fake = Faker()
Faker.seed(42)

#names = [fake.first_name() for _ in range(10)]


names = ['Joseph', 'William', 'Connie', 'Dana', 'Kimberly', 'Angela',
       'Jason', 'Matthew', 'Nicole', 'Daniel']

'''
people={key: {
		 'willingness to travel': random.randrange(1,5), # 1 low, 5 high
         'desire for new experience': random.randrange(1,5), # 1 low, 5 high
         'cost': random.randrange(1,10), #1 cheap, 10 expensive
         'hipster points': random.randrange(1,10), #1 worse, 10 best
         'cuisine': random.randrange(2), # 0=indian, 1 = mexican
         'vegetarian': random.randrange(2)#binary 0=no, 1 =yes
         } for key in names} 




jsonString = json.dumps(people)
jsonFile = open('people.json', 'w')
jsonFile.write(jsonString)
jsonFile.close()


np.savetxt("names.csv", np.array(names), fmt='%s', delimiter=",") # save to file so I can maintain a consistent data source
'''
# import people
fileObj = open('people.json', 'r')
jsonContent = fileObj.read()
people = json.loads(jsonContent)


#TODO turn people dict into a matrix
# convery dictionary to list to preserve order, make array from the values of the inner dictionary using the outer dictionary as index
M_people = np.array([[value for (key, value) in people[i].items()] for i in people.keys()]) #scores per username/ category
M_people_names = [key for key in people.keys()] #row names scores
M_people_scores = [key for (key, value) in people[names[0]].items()] #col names are people

####################################################################################################################
# Restaurants Data
####################################################################################################################

# Create restaurant data
rest = ['flacos', 'Joes', 'Poke', 'Sush-shi', 'Chick Fillet', 'Mackie Des', 'Michaels', 'Amaze', 'Kappa', 'Mu']
'''
restaurants  = {key: {'distance' : random.randrange(1,5),
						'novelty' : random.randrange(1,5),
						'cost': random.uniform(1,10),
						'average rating': random.uniform(1,10),
						'cuisine': random.randrange(2), #binary 0=indian, 1=mexican
						'vegetarian': random.randrange(2) #binary 0=no, 1 =yes
						} for key in rest}
			  

jsonString = json.dumps(restaurants)
jsonFile = open('restaurants.json', 'w')
jsonFile.write(jsonString)
jsonFile.close()

'''

# import restaurants
fileObj = open('restaurants.json', 'r')
jsonContent = fileObj.read()
restaurants = json.loads(jsonContent)

# Transform the restaurant data into a matrix(M_resturants) use the same column index.

# TODO Turn restaurants dict into matrix
M_restaurants = np.array([[value for (key, value) in restaurants[i].items()] for i in restaurants.keys()]) #scores per username/ category
M_rest_names = [key for key in restaurants.keys()] # row indicies = restaurant names
M_rest_scores = [key for (key, value) in restaurants[rest[0]].items()] # column names = scores




####################################################################################################################
# Questions
####################################################################################################################

# Problem 2: Choose a person and compute(using a linear combination) the top restaurant for them.  

M_people_names[0] # the person we are doing the top restaurant match for
'''
>>> M_people_names[0] # the person we are doing the top restaurant match for
'Joseph'
'''

from scipy.stats import rankdata, iqr
top_rest_score = np.dot(M_restaurants, M_people[0]) # get scores for all restaurants for that person
'''
>>> top_rest_score
array([77.06056757, 89.44427131, 63.1989789 , 21.08453724, 53.27046491,
       77.5390318 , 59.14507594, 18.48379129, 14.41664567, 57.60558386])
'''

top_rest = rankdata(top_rest_score) # rank the scores, we want the highest rank because that's the highest score
'''
>>> top_rest
array([ 8., 10.,  7.,  3.,  4.,  9.,  6.,  2.,  1.,  5.])
'''

where = list(list(np.where(top_rest == np.amax(top_rest)))[0]) # find the index of the highest rank so we can compare that to our restuarant list
'''
>>> where
[1]
'''
if len(where) == 1:
  print((M_people_names[0]+'\'s'),'top restaurant is', rest[where[0]]) # print the name of the person and their top scored restaurant
else:  
  print((M_people_names[0]+'\'s'),'top restaurants are ', *[(rest[i]+',') if x+1 < len(where) else rest[i] for x,i in enumerate(where)]) # print the name of the person and their top scored restaurant
'''
Joseph's top restaurant is Joes
'''

# What does each entry in the resulting vector represent. 

'''
>>> top_rest_score
array([77.06056757, 89.44427131, 63.1989789 , 21.08453724, 53.27046491,
       77.5390318 , 59.14507594, 18.48379129, 14.41664567, 57.60558386])

each entry represents the linear combination of names[0]'s (my chosen person) combined scoring for each restaurant.
'''

## Problem 3 ##################################################################################################################

# Problem 3: Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people. 
M_usr_x_rest = np.dot(M_restaurants, M_people.T)
'''
>>> M_usr_x_rest
array([[ 77.06056757,  64.62312197,  82.87807156,  54.36817238,
         65.80561798,  76.63973998,  92.10886996,  54.97149598,
         53.97149598,  85.26721478],
       [ 89.44427131,  82.38971976, 107.96490304,  70.81453648,
         79.86908803,  81.83465295, 113.84148132,  71.90352312,
         70.90352312, 101.62224459],
       [ 63.1989789 ,  58.04058794,  76.42870732,  51.65246856,
         60.81085952,  59.48114006,  81.61087743,  54.14057898,
         53.14057898,  77.17519975],
       [ 21.08453724,  22.85399552,  31.44803421,  22.25995684,
         20.49049856,  20.24319948,  33.72289434,  19.73779763,
         19.73779763,  24.04021882],
       [ 53.27046491,  65.27349716,  89.51902096,  63.02797335,
         52.02494111,  42.73050691,  92.93318481,  56.3193753 ,
         52.3193753 ,  58.85326881],
       [ 77.5390318 ,  75.71789927,  96.79029625,  60.64550229,
         63.46663481,  66.859456  , 105.75480609,  58.07381363,
         57.07381363,  78.39565449],
       [ 59.14507594,  72.1189153 ,  95.01722706,  61.22060354,
         48.24676418,  40.74725656,  98.9920512 ,  53.74627179,
         49.74627179,  54.19641244],
       [ 18.48379129,  12.74714503,  17.79103972,  17.70325035,
         23.43989661,  27.60497656,  24.38455336,  18.27481666,
         15.27481666,  29.6269239 ],
       [ 14.41664567,  13.32662065,  18.92573691,  17.7275044 ,
         17.81752941,  18.9773988 ,  23.15545067,  16.65766003,
         12.65766003,  21.27695693],
       [ 57.60558386,  39.47801133,  54.37422537,  46.58179729,
         65.70936983,  74.94378719,  64.71548756,  51.47495246,
         47.47495246,  86.39189421]])

here we can see the the first person M_people[0] is the first column of the matix
>>> M_usr_x_rest[:,0]
array([77.06056757, 89.44427131, 63.1989789 , 21.08453724, 53.27046491,
       77.5390318 , 59.14507594, 18.48379129, 14.41664567, 57.60558386])
'''
#  What does the a_ij matrix represent? 
'''The a_ij matrix represents each person (columns) scores per restuarant(rows)'''
>>> top_rest_score == M_usr_x_rest[:,0]
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True])
## Problem 4 ##################################################################################################################

# Problem 4: Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
optimal_rest = sum(M_usr_x_rest.T)# this puts the people in the rows/ restaurants in the columns
optimal_rest
'''
>>> optimal_rest
array([707.69436813, 870.58794371, 635.67997744, 235.61893028,
       626.27160861, 740.31690825, 633.1768498 , 205.33121014,
       174.93916348, 588.75006156])
 
>>> max(optimal_rest)
870.5879437127807
>>> M_rest_names[list(list(np.where(np.isin(optimal_rest, max(optimal_rest))))[0])[0]]
'Joes'

The entry's represent the sum of all scores for each person by restaurant, from the looks the highest score is 870.587 in position 1 
and that is restaurant 'Joes'
'''
## Problem 5 ##################################################################################################################

# Problem 5: Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   
M_usr_x_rest_rank  = rankdata(M_usr_x_rest.T, axis=1) # this puts the people in the rows/ restaurants in the columns
M_usr_x_rest_rank 
'''
>>> M_usr_x_rest_rank
array([[ 8., 10.,  7.,  3.,  4.,  9.,  6.,  2.,  1.,  5.],
       [ 6., 10.,  5.,  3.,  7.,  9.,  8.,  1.,  2.,  4.],
       [ 6., 10.,  5.,  3.,  7.,  9.,  8.,  1.,  2.,  4.],
       [ 6., 10.,  5.,  3.,  9.,  7.,  8.,  1.,  2.,  4.],
       [ 9., 10.,  6.,  2.,  5.,  7.,  4.,  3.,  1.,  8.],
       [ 9., 10.,  6.,  2.,  5.,  7.,  4.,  3.,  1.,  8.],
       [ 6., 10.,  5.,  3.,  7.,  9.,  8.,  2.,  1.,  4.],
       [ 7., 10.,  6.,  3.,  8.,  9.,  5.,  2.,  1.,  4.],
       [ 8., 10.,  7.,  3.,  6.,  9.,  5.,  2.,  1.,  4.],
       [ 8., 10.,  6.,  2.,  5.,  7.,  4.,  3.,  1.,  9.]])
'''
rankdata(optimal_rest)
'''
>>> rankdata(optimal_rest)
array([ 8., 10.,  7.,  3.,  5.,  9.,  6.,  2.,  1.,  4.])
'''

# Do the same as above to generate the optimal resturant choice.  
optimal_rest_rank = sum(M_usr_x_rest_rank)
optimal_rest_rank
'''
>>> optimal_rest_rank
array([ 73., 100.,  58.,  27.,  63.,  82.,  60.,  20.,  13.,  54.])
'''
rankdata(optimal_rest_rank)
'''
>>> rankdata(optimal_rest_rank)
array([ 8., 10.,  5.,  3.,  7.,  9.,  6.,  2.,  1.,  4.])
'''
## Problem 6 ##################################################################################################################

# Problem 6: Why is there a difference between the two?  What problem arrives?  What does represent in the real world?
'''
raw score ranking:             [ 8., 10.,  7.,  3.,  5.,  9.,  6.,  2.,  1.,  4.]
user restaurant ranking rank:  [ 8., 10.,  5.,  3.,  7.,  9.,  6.,  2.,  1.,  4.]


1)a. There is a difference because one uses raw scores (aka subjective input from each of your coworkers) vs 
the other uses individual user ranking of restaurants per individual based on their individual raw scores, essentially 
standardizing the raw scores across each individual into a class of ranking

2) When we use raw scores and just tally up the totals by each user, we are solely relying on a very simple assessment
of which restuarant we should choose for lunch. BUT if we take into account each user's independent assessment of each 
scoring metric, then we don't have any standardization because we can't assume each individual has a unified thought process, background or bias

3) In the real world this happens all the time where you have different tastes among people and you need to find a way to balance all those tastes 
to generate a fair ranking system. It also happens in event or delivery route patterns where there are several solutions to a problem, but which one is the best based on 
the circumstances. This is also a good representation of AP Top 25 NCAA football teams that is based on scoring from a panel of 62 sports writer and broadcasters in the USA vs the CFP
Top 25 that using more in-depth analysis of each football program and combines both subjective information with non-subjective information.
In other words, in the real world whenever you use raw scoring, you run the risk of bais in the scoring. When you use the rank data, it helps to balance out any bias, but at the
same time removing inter-user variance
'''

## Problem 7 ##################################################################################################################

# Problem 7: How should you preprocess your data to remove this problem. 
'''
- One method is to transform the data...for example, if you are comparing an image from different camera angeles, you would first need to 
alter the angles to all be the same across all photos within the 3-d space (distance to the subject being the 3rd dimension).

- Ranking in and of itself is a method of pre-processing because it converts the raw scores into a list of distinct scores by user...essentially if each 
user is judging each restaurant using the same standard criteria (for example I prefer Mexican food most days, so most days  will reflect that in how 
I might repetitiously score mexican restuarants higher than Indian restaurants, but my colleague might like Indian restaurants better and will continiously score them better - but we may both agree on 
distance and expense)...ultimately the consistency of individuals to score will result in their rankings being a more reliable metric to use than the raw scores.

- The challenge with using ranking data is that you lose the variability in the data or between the people, so it makes more sense to use something like PCA or kmeans clustering or a combination
of the two to pre-define groups if necessary to keep the variability

'''
## Problem 8 ##################################################################################################################

# Problem 8: Find  user profiles that are problematic, explain why?
'''
My intiuitive thought is to look for user profiles that are crazy outliers compared to the rest as this will or will highly likely overly inflate or deflate ratings
'''

M_usr_x_rest_avg = np.mean(M_usr_x_rest.T, axis=1) # get mean of each ranking by user 
M_usr_x_rest_avg
'''
array([53.12489485, 50.65695139, 67.11372624, 46.60017655, 49.76812   ,
       51.00621145, 73.12196567, 45.53002856, 43.23002856, 61.68459887])
       '''
mean = np.mean(M_usr_x_rest.T) # get the mean of all scores in the matrix
'''
>>> mean
54.18367021396597
'''
std = np.std(M_usr_x_rest.T) # get the SD of all scores in the matrix
variance = np.var(M_usr_x_rest.T)
# get the range based on 1-SD to find outliers in the data
lower = mean-std # lower range using 1 standard deviation
upper = mean+std # upper range using 1 standard deviation

print('Range from lower: %d to mean: %d to upper %d using 1 standard deviation' %(lower, mean,  upper)) # print Range of lower to mean to upper standard deviation
'''
Range from lower: 27 to mean: 54 to upper 80 using 1 standard deviation
'''
# Find outliers using 1 standard deviation
pot_outliers = [x for x in M_usr_x_rest_avg if x < lower or x > upper] #find outliers
pot_outliers
'''
[]
We don't see any outliers using 1-SD, so no dissassisfied people using this method
'''
if len(pot_outliers) == 0:
  print('Basd on 1-SD, we do not have any dissassified people in our group')
else:
  outlier_idx = np.where(np.isin(M_usr_x_rest_avg, pot_outliers)) #find indeices of outliers
  idx = list(outlier_idx[0]) # unpack the tuple into a list
  trouble_profiles = [M_people_names[i] for i in idx]
  print('These users have profiles that are outliers based on 1-standard deviation: %s' % trouble_profiles)

'''
Basd on 1-SD, we do not have any dissassified people in our group
'''

### Explaination: ###
'''
Using 1-SD to determing outliers (we did not find any), but if we did the reason these would be considered outliers or trouble profiles 
is because using rule of 1-standard deviation based on the range of per user mean scores, these would be considered outliers.
The reason they would have exceptionally high or low ratings compared to the rest of the people and that can disproportionately inflate or deflate the raw scores for
each restaurant. 
'''
## Problem 9 ##################################################################################################################

# Problem 9: Think of two metrics to compute the disatistifaction with the group.  
# metrics: We can use standard deviation and IQR to determine if there are dissassified people within the group
#
##### Standard Deviation #####
#
'''
we already have the SD from above that shows we have potentially no dissassified people.

'''

# graph the mean for each person
x = np.array([y for y in range(len(M_usr_x_rest_avg))])
y = M_usr_x_rest_avg
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.plot(x, y, 'o', color='red')
ax.hlines(y=[upper, mean, lower], xmin=0, xmax=len(x), colors=['blue'], linestyles='dashed')
ax.text(0,upper, '1-SD Upper', ha='left', va='bottom', color='blue')
ax.text(0,mean, 'Group Mean', ha='left', va='bottom', color='blue')
ax.text(0,lower, '1-SD Lower', ha='left', va='bottom', color='blue')
ax.set_title('Plot For Each User\'s Mean Restaurant Score To Visualize Outliers at 1-Standard Deviation') 
ax.set_xticklabels(M_people_names)
ax.set_xticks(x)
ax.set_ylabel('Mean Raw Scores for Each User')
plt.show()



#####   IQR    #####
# establish the group and user IQRs
group_iqr = iqr(M_usr_x_rest)
user_iqr = iqr(M_usr_x_rest, axis=1)
group_median = np.median(M_usr_x_rest)
group_q1 = np.quantile(M_usr_x_rest, 0.25)
group_q2 = np.quantile(M_usr_x_rest, 0.5)
group_q3 = np.quantile(M_usr_x_rest, 0.75)
group_q4 = np.max(M_usr_x_rest)


'''
# graph the boxplots for each user to see if there is any vidual evidence of outliers using IQRs
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.boxplot(M_usr_x_rest)
ax.set_title('Boxplots For Each User Raw Restaurant Scores')
ax.set_xticklabels(M_people_names)
ax.set_ylabel('Raw Scores for Each Restaurant')
ax.set_xlabel('People in the Lunch Group')
plt.show()

# don't see much evidence of outliers or problem profiles

'''
# Chart the IQR for each person
x_plot = [1 for x in range(10)]

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.boxplot(user_iqr, manage_ticks=True, whis=1)
ax.plot(x_plot, user_iqr, 'o', color='blue')
ax.set_title('Boxplot Showing IQR Distribution of Each User')
ax.set_ylabel('IQR')
for i, n in enumerate(M_people_names):
  ax.annotate(n, (x_plot[i], user_iqr[i]), xycoords='data', xytext=(5,5),textcoords='offset points')
plt.show()
'''
with the whiskers set to 1, we don't see any outliers with the groups so no problem profiles or dissassified people
'''

## Problem 10 ##################################################################################################################

# Problem 10: Should you split in two groups today? 
'''
use PCA to reduce to 2 vectors
code adapted from office hours &
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
'''
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans

# PCA
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(M_usr_x_rest)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
pca.get_params()
'''
>>> pca = PCA(n_components=2, random_state=42)
>>> reduced = pca.fit_transform(M_usr_x_rest)
>>> print(pca.explained_variance_ratio_)
[0.92295088 0.07151824]
>>> print(pca.singular_values_)
[237.90545483  66.22523601]
>>> pca.get_params()
{'copy': True, 'iterated_power': 'auto', 'n_components': 2, 'random_state': 42, 'svd_solver': 'auto', 'tol': 0.0, 'whiten': False}
>>>

'''
# KMeans
kmeans = KMeans(n_clusters=2, random_state=42) # set number of clusters
kmeans.get_params() # just curious what the params are
kmeans.fit(reduced) # fit kmeans
kmeans_group = kmeans.predict(reduced)
kmeans_group
labels = kmeans.labels_
labels
'''
>>> kmeans = KMeans(n_clusters=2, random_state=42) # set number of clusters
>>> kmeans.get_params() # just curious what the params are
{'algorithm': 'auto', 'copy_x': True, 'init': 'k-means++', 'max_iter': 300, 'n_clusters': 2, 'n_init': 10, 'random_state': 42, 'tol': 0.0001, 'verbose': 0}
>>> kmeans.fit(reduced) # fit kmeans
KMeans(n_clusters=2, random_state=42)
>>> kmeans_group = kmeans.predict(reduced)
>>> kmeans_group
array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0])
>>> labels = kmeans.labels_
>>> labels
array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0])
'''

# plot results

fig, ax = plt.subplots(1,1, figsize=(10,10))
colors = ['b','r','g']
ax.set_title('K-Means Clustering of People in 3 Clusters', size=15)
print('Location:','\t\t\t','Class Label')
for i in range(len(reduced)):
  ax.plot(reduced[i][0], reduced[i][1],'k.', color = colors[labels[i]],
          markersize=20)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=100, linewidths=3,
            color='orange', zorder=10)
plt.show()

print('It does appear with randomly drawn scores, we coule break into 2 distinct groups.')
'''
It does appear with randomly drawn scores, we coule break into 2 distinct groups.
'''

## Problem 11 ##################################################################################################################

# Problem 11: Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?
# if the boss is paying, let's pick a restaurant that is far away

max_dist = np.max(np.unique(M_restaurants[:,0]))
max_dist
boss_idx = list(list(np.where(np.isin(M_restaurants[:,0], max_dist)))[0]) # reduce to the restaurants that are farthest away
boss_rest = M_restaurants[boss_idx,1:] # removing distance score and only keeping the restaurants that are farthest away
boss_rest_name = [M_rest_names[i] for i in boss_idx]



#boss_rest = np.delete(boss_rest,3,0 ) # remove cheapest restaurant
min_cost = np.min(boss_rest[:,1])
min_cost_idx = list(list(np.where(np.isin(boss_rest[:,1], min_cost)))[0])
boss_rest = np.delete(boss_rest, min_cost_idx, 0) # delete the cheapest restaurant
for i in min_cost_idx:
  del boss_rest_name[i] # clean up shortened restaurant list
boss_rest = np.delete(boss_rest, 1, 1) # remove cost column

# update the restaurant list
boss_rest_name = [M_rest_names[i] for i in boss_idx]
del boss_rest_name[-1] # delete the last item in the list since it was the cheapest 

boss_people = np.delete(M_people, [0,2], 1)
boss_rest.shape
boss_people.shape
'''
>>> boss_rest.shape
(3, 4)
>>> boss_people.T.shape
(4, 10)
'''
boss_usr_x_rest = np.dot(boss_rest, boss_people.T)
boss_rank = rankdata(boss_usr_x_rest.T, axis=1)
boss_rest_rank = sum(boss_rank) 
final_boss_rank = rankdata(boss_rest_rank) # rank the restaurants based on the summary of the people's individual restaurant rankings
boss_idx = [i for i, best in enumerate(final_boss_rank) if best == np.max(final_boss_rank)] # find the top ranked restaurants index
boss_best_rest_name = [boss_rest_name[i] for i in boss_idx] # in the event there is a list of ties, this will return all the restaurant names
if len(boss_best_rest_name) == 1:
  boss_best_rest_name = boss_best_rest_name[0]

boss_usr_x_rest
boss_rest_rank
final_boss_rank
print('The optimal restuarant based on the %d most expensive restaurants the farthest away and our people input is %s' % (len(boss_rank), boss_best_rest_name))
'''
>>> boss_usr_x_rest
array([[36.50631311,  9.87657828, 11.87657828, 13.87657828, 38.50631311,
        58.25946967, 21.75315656, 20.75315656, 19.75315656, 56.25946967],
       [25.72868045,  7.18217011,  8.18217011, 10.18217011, 28.72868045,
        42.09302068, 17.36434023, 15.36434023, 14.36434023, 41.09302068],
       [ 8.6566554 ,  4.41416385,  8.41416385, 14.41416385, 15.6566554 ,
        22.48498311, 15.8283277 , 11.8283277 ,  7.8283277 , 18.48498311]])
>>> boss_rest_rank
array([29., 18., 13.])
>>> final_boss_rank
array([3., 2., 1.])
>>> print('The optimal restuarant based on the %d most expensive restaurants the farthest away and our people input is %s' % (len(boss_rank), boss_best_rest_name))
The optimal restuarant based on the 10 most expensive restaurants the farthest away and our people input is Joes
'''

## Problem 12 ##################################################################################################################

# Problem 12: Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 
'''
Without more detailed data beyond just the rankings I cannot find their weight matrix

'''
