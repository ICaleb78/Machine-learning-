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
