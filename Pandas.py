import pandas as pd
pd.DataFrame({'Yes': [121, 324], 'No': [345,789]}, index=['Product A', 'Product B'])
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
#We can use the shape attribute to check how large the resulting DataFrame is:
wine_reviews.shape
#To make pandas use that column for the index (instead of creating a new one from scratch), we can specify an index_col.
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame({'Apples':[35,41], 'Bananas':[21,34]}, index=['2017 Sales', '2018 Sales'])
ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour','Milk', 'Eggs','Spam'],name='Dinner')
# Your code goes here
ingredients.to_csv('cows_and_goats.csv')

pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']},  index=['Product A', 'Product B'])
pd.Series([1, 2, 3, 4, 5])
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
wine_reviews.shape
wine_reviews.head()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()

# q1: In the cell below, create a DataFrame fruits that looks like this
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])

#q2: Create a dataframe `fruit_sales` that matches the diagram below:
fruit_sales = pd.DataFrame([[35, 21], [41, 34]], columns=['Apples', 'Bananas'],index=['2017 Sales', '2018 Sales'])

#q3: Create a variable ingredients with a Series that looks like
quantities = ['4 cups', '1 cup', '2 large', '1 can']
items = ['Flour', 'Milk', 'Eggs', 'Spam']
recipe = pd.Series(quantities, index=items, name='Dinner')

#q4: Read the following csv dataset of wine reviews into a DataFrame called reviews
reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)

#q5: Run the cell below to create and display a DataFrame called animals
animals.to_csv("cows_and_goats.csv")

reviews.country
reviews['country']
reviews['country'][0]
reviews.iloc[0]
reviews.iloc[:, 0]
reviews.iloc[:3, 0]
reviews.iloc[1:3, 0]
reviews.iloc[[0, 1, 2], 0]
reviews.iloc[-5:]
reviews.loc[0, 'country']
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
reviews.set_index("title")
reviews.country == 'Italy'
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]
reviews['critic'] = 'everyone'
reviews['critic']
reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']
#q1: Select the description column from reviews and assign the result to the variable desc.
# Your code here
desc = reviews['description']
# q2: Select the first value from the description column of reviews, assigning it to variable first_description
first_description = reviews['description'][0]

# q3:Select the first row of data (the first record) from reviews, assigning it to the variable first_row.
first_row = reviews.iloc[0]

#q4: Select the first 10 values from the description column in reviews, assigning the result to variable first_descriptions
first_descriptions = reviews.description.loc[0:9]

#q5: Select the records with index labels 1, 2, 3, 5, and 8, assigning the result to the variable sample_reviews
sample_reviews = reviews.iloc[[1,2,3,5,8]] 
#OR
indices = [1, 2, 3, 5, 8]
sample_reviews = reviews.loc[indices]

#Q6: Create a variable `df` containing the `country`, `province`, `region_1`, and `region_2` columns of the records with the index labels `0`, `1`, `10`, and `100`. In other words, generate the following DataFrame:
mf = reviews.loc[:, ['country', 'province', 'region_1', 'region_2']]
df = mf.loc[[0,1,10,100]]
#ORD
cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = reviews.loc[indices, cols]

#q7: Create a variable df containing the country and variety columns of the first 100 records
mf = reviews.loc[:, ['country', 'variety']]
df = mf.iloc[0:100] 
#OR
cols = ['country', 'variety']
df = reviews.loc[:99, cols]
#or

cols_idx = [0, 11]
df = reviews.iloc[:100, cols_idx]

#q8: Create a DataFrame italian_wines containing reviews of wines made in Italy. Hint: reviews.country equals what?
italian_wines = reviews.loc[reviews.country == 'Italy']

#q9: Create a DataFrame top_oceania_wines containing all reviews with at least 95 points (out of 100) for wines from Australia or New Zealand
top_oceania_wines =reviews.loc[(reviews.country.isin(['Australia', 'New Zealand'])) & (reviews.points >= 95)]

reviews.points.describe() #best used for numerical data
reviews.taster_name.describe() #best used for string data
reviews.points.mean()
#To see a list of unique values we can use the unique() function:
reviews.taster_name.unique()
#To see a list of unique values and how often they occur in the dataset,
reviews.taster_name.value_counts()

#remean the scores the wines received to 0.
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)

#using appl(.) for mapping 
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')

#Pandas provides many common mapping operations as built-ins. For example, here's a faster way of remeaning our points column:
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean

#an easy way of combining country and region information in the dataset would be to do the following:
reviews.country + " - " + reviews.region_1

#EXERCISE
# q1: What is the median of the points column in the reviews DataFrame?
median_points = reviews.points.median()

#q2: What countries are represented in the dataset? (Your answer should not include any duplicates.)
countries = reviews.country.unique()
#q4: How often does each country appear in the dataset? Create a Series reviews_per_country mapping countries to the count of reviews of wines from that country
reviews_per_country = reviews.country.value_counts()

#q4: Create variable centered_price containing a version of the price column with the mean price subtracted.
reviews_price_mean = reviews.price.mean()
centered_price = reviews.price - reviews_price_mean

#q6: I'm an economical wine buyer. Which wine is the "best bargain"? Create a variable bargain_wine with the title of the wine with the highest points-to-price ratio in the dataset.
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

#q6: describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series descriptor_counts counting how many times each of these two words appears in the descriptio

n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

#q6We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.

#Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.

#Create a series star_ratings with the number of stars corresponding to each review in the dataset.
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1
    
star_ratings = reviews.apply(stars, axis='columns')

#One function we've been using heavily thus far is the value_counts() function. We can replicate what value_counts()
reviews.groupby('points').points.count()

#We can use any of the summary functions we've used before with this data. For example, to get the cheapest wine in each point value category, we can do the following:
reviews.groupby('points').price.min()

#For example, here's one way of selecting the name of the first wine reviewed from each winery in the dataset:
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])

#For even more fine-grained control, you can also group by more than one column. For an example, here's how we would pick out the best wine by country and province:
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])

#Another groupby() method worth mentioning is agg(), which lets you run a bunch of different functions on your DataFrame simultaneously. For example, we can generate a simple statistical summary of the dataset as follows:
reviews.groupby(['country']).price.agg([len, min, max])

countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed
mi = countries_reviewed.index
type(mi)

#MULTI-INDEXING 
#However, in general the multi-index method you will use most often is the one for converting back to a regular index, the reset_index() method:
countries_reviewed.reset_index()

#SORTING
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')

#sort_values() defaults to an ascending sort, where the lowest values go first. However, most of the time we want a descending sort, where the higher numbers go first. That goes thusly:

countries_reviewed.sort_values(by='len', ascending=False)

#To sort by index values, use the companion method sort_index(). This method has the same arguments and default order:
countries_reviewed.sort_index()

#Finally, know that you can sort by more than one column at a time:
countries_reviewed.sort_values(by=['country', 'len'])

#GROUPING AND SORTING EXCERCISE 
#q1: Who are the most common wine reviewers in the dataset? Create a `Series` whose index is the `taster_twitter_handle` category from the dataset, and whose values count how many reviews each person wrote.
reviews_written = reviews.groupby('taster_twitter_handle').points.count()
#q2: What is the best wine I can buy for a given amount of money? Create a `Series` whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the values by price, ascending (so that `4.0` dollars is at the top and `3300.0` dollars is at the bottom).
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()
#q3: What are the minimum and maximum prices for each `variety` of wine? Create a `DataFrame` whose index is the `variety` category from the dataset and whose values are the `min` and `max` values thereof.
price_extremes = reviews.groupby(['variety']).price.agg([min, max])
#q4: What are the most expensive wine varieties? Create a variable `sorted_varieties` containing a copy of the dataframe from the previous question where varieties are sorted in descending order based on minimum price, then on maximum price (to break ties).
orted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)
#q5: Create a `Series` whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the `taster_name` and `points` columns.
reviewer_mean_ratings =reviews.groupby(['taster_name']).points.mean()
#q6: What combination of countries and varieties are most common? Create a `Series` whose index is a `MultiIndex`of `{country, variety}` pairs. For example, a pinot noir produced in the US should map to `{"US", "Pinot Noir"}`. Sort the values in the `Series` in descending order based on wine count.
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)

#DATA TYPES & MISSING VALUES
reviews.price.dtype
reviews.dtypes #returns the type of data in all column
reviews.points.astype('float64') # to change from one data type to aother 
reviews.index.dtype  #A DataFrame or Series index has its own dtype, too:
reviews[pd.isnull(reviews.country)] #to locate or compare missing columns or missing data
reviews[pd.notnull(reviews.country)]
reviews.region_2.fillna("Unknown") # replacing missing values
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino") # to replace a particular value

#EXERCISE
#Q1 :What is the data type of the `points` column in the dataset?
dtype = reviews.points.dtype
#q2: Create a Series from entries in the `points` column, but convert the entries to strings. Hint: strings are `str` in native Python.
point_strings = reviews.points.astype('str')
#Q3: Sometimes the price column is null. How many reviews in the dataset are missing a price?
missing_price_reviews = reviews[reviews.price.isnull()]
n_missing_prices = len(missing_price_reviews)
# cute alternative solution:  if we sum a bolean series, True is treated as 1 and False as 0
n_missing_prices = reviews.price.isnull().sum()
# or equivalently
n-missing_prices = pd.isnull(reviews.price).sum()

#Q4: What are the most common wine-producing regions? Create a Series counting the number of times each value occurs in the `region_1` field. This field is often missing data, so replace missing values with `Unknown`. Sort in descending order.  Your output should look something like this:
reviews_per_region1 = reviews.region_1.fillna("Unknown") 
reviews_per_region = reviews_per_region1.value_counts()
#OR
reviews_per_region =reviews.region_1.fillna('Unknown').value_counts.sort_values(ascending=False)

#RENAMING & COMBINNING
reviews.rename(columns={'points': 'score'}) #to change the points column in our dataset to score, we would do:
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'}) #rename() lets you rename index or column values by specifying a index or column keyword parameter, respectively. It supports a variety of input formats, but usually a Python dictionary is the most convenient.
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns') #Both the row index and the column index can have their own name attribute.The complimentary rename_axis() method may be used to change these names. Fo
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_youtube, british_youtube]) # to combinne two seperate dataset: you open the combinations on python first , then u combine 

left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK') #The middlemost combiner in terms of complexity is join(). join() lets you combine different DataFrame objects which have an index in common.

#EXERCISE 
#Q1: `region_1` and `region_2` are pretty uninformative names for locale columns in the dataset. Create a copy of `reviews` with these columns renamed to `region` and `locale`, respectively.
renamed = reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})
#Q2: Set the index name in the dataset to `wines`
reindexed = reviews.rename_axis('wines', axis='rows')
#Q3: 
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
#Create a `DataFrame` of products mentioned on *either* subreddit.
combined_products = pd.concat([gaming_products, movie_products]) #first run the cell above 

#Q$
#The [Powerlifting Database](https://www.kaggle.com/open-powerlifting/powerlifting-database) dataset on Kaggle includes one CSV table for powerlifting meets and a separate one for powerlifting competitors. Run the cell below to load these datasets into dataframes:
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
#Both tables include references to a `MeetID`, a unique key for each meet (competition) included in the database. Using this, generate a dataset combining the two tables into one.
powerlifting_combined = powerlifting_meets.set_index('MeetID').join(powerlifting_competitors.set_index('MeetID'))



