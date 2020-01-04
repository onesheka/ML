from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_df = home_listings.sample(frac=0.75,random_state=1)
test_df = home_listings.drop(train_df.index)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

hyper_params = [1,2,3,4,5]
mse_values = []
for r in hyper_params:
        knn = KNeighborsRegressor(n_neighbors = r, algorithm='brute')
        train_columns = ['accommodates', 'bathrooms', 'bedrooms', 'number_of_reviews']
        knn.fit(train_df[train_columns], train_df['price'])
        predictions = knn.predict(test_df[train_columns])
        
        mse_values = mean_squared_error(test_df["price"], predictions)
        print(mse_values)
