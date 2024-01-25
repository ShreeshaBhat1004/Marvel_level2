So Exploratory Data analysis means various ways in which you can unlock secrets from the given data.
We have to make sure data is clean, no errors or empty spaces
EDA can include many skills and techniques
________
Steps of EDA:
- Understand the data(.head,.tail,.describe methods)
- Cleaning the data(detecting for anomalies and deleting them)
- Relationship analysis(visulisation techniques - seaborn library)

Let's take an example of AirBnB dataset. AirBnB is an online platform where people are looking forward to rent their house while others are looking for accomodations. Inside AirBnB is independent, non commercial set of tools and data that shows how AirBnB really works. We will be using their data for our exploratory data analysis. 

We use .head() method of pandas to get first 5 coloumns 
```python 
nyc_df.head()
```
|index|id|name|host\_id|host\_name|neighbourhood\_group|neighbourhood|latitude|longitude|room\_type|price|minimum\_nights|number\_of\_reviews|last\_review|reviews\_per\_month|calculated\_host\_listings\_count|availability\_365|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|2539|Clean & quiet apt home by the park|2787|John|Brooklyn|Kensington|40\.64749|-73\.97237|Private room|149|1|9|2018-10-19|0\.21|6|365|
|1|2595|Skylit Midtown Castle|2845|Jennifer|Manhattan|Midtown|40\.75362|-73\.98377|Entire home/apt|225|1|45|2019-05-21|0\.38|2|355|
|2|3647|THE VILLAGE OF HARLEM\.\.\.\.NEW YORK \!|4632|Elisabeth|Manhattan|Harlem|40\.80902|-73\.9419|Private room|150|3|0|NaN|NaN|1|365|
|3|3831|Cozy Entire Floor of Brownstone|4869|LisaRoxanne|Brooklyn|Clinton Hill|40\.68514|-73\.95976|Entire home/apt|89|1|270|2019-07-05|4\.64|1|194|
|4|5022|Entire Apt: Spacious Studio/Loft by central park|7192|Laura|Manhattan|East Harlem|40\.79851|-73\.94399|Entire home/apt|80|10|9|2018-11-19|0\.1|1|0|

We use .info() method to get a summary of our dataset. 
```python
nyc_df.info()
```
RangeIndex: 48895 entries, 0 to 48894
Data columns (total 16 columns):
    Column                          Non-Null Count  Dtype    \n                      
 0   id                              48895 non-null  int64   \n
 1   name                            48879 non-null  object  \n
 2   host_id                         48895 non-null  int64   \n
 3   host_name                       48874 non-null  object  \n
 4   neighbourhood_group             48895 non-null  object  \n
 5   neighbourhood                   48895 non-null  object  \n
 6   latitude                        48895 non-null  float64 \n
 7   longitude                       48895 non-null  float64 \n
 8   room_type                       48895 non-null  object  \n
 9   price                           48895 non-null  int64   \n
 10  minimum_nights                  48895 non-null  int64   \n
 11  number_of_reviews               48895 non-null  int64   \n
 12  last_review                     38843 non-null  object  \n
 13  reviews_per_month               38843 non-null  float64 \n
 14  calculated_host_listings_count  48895 non-null  int64   \n
 15  availability_365                48895 non-null  int64   \n
dtypes: float64(3), int64(7), object(6)

We get to know that there are 15 features, out of which 4 have missing values, those are host_name,name,last_review,reviews_per_month. Also the id feature is useless here. So we drop these 4 features from our dataset. 
```python
nyc_df.drop([‘id’,’name’,’host_name’,’last_review’], axis=1, inplace=True)
```
We will check for any feature that contains null value even after cleaning

```python
print(nyc_df.isnull().any())
```
host_id                           False
neighbourhood_group               False
neighbourhood                     False
latitude                          False
longitude                         False
room_type                         False
price                             False
minimum_nights                    False
number_of_reviews                 False
reviews_per_month                 False
calculated_host_listings_count    False
availability_365                  False
dtype: bool
We havent missed any value anymore. 


