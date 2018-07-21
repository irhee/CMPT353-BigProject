####
#applying function pandas
###
#https://www.youtube.com/watch?v=P_q0tkYqvSk&index=30&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y
#map
train['sex_num'] = train.sex.map({'female':0,'male':1})
#apply
train['name_len'] = train.name.apply(len)
train['fair_ceil'] = train.fare.apply(np.ceil)
#lambda
def get_element(my_list,position)
  return my_list[position]
train['first_name'] = train.name.str.split(',').apply(get_element,position=0)
train['first_name'] = train.name.str.split(',').apply(lambda x: x[0])
#applymap = apply function to every element of the data
drinks.loc[:,'beer':'wine'] = drinks.loc[:, 'beer':'wine'].apploymap(float)
#Using string methods https://www.youtube.com/watch?v=4tTO_xH4aQE&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=6
ri['frisk'] = ri.search_type.str.contains('Protective Frisk')
ri['frisk'].value_counts()
ri['frisk'].value_counts(dropna=False)
ri.frisk.sum()

###
#joining and merging 
###
#https://www.youtube.com/watch?v=XMjSGGej9y8

df1 = pd.DataFrame({'year':[2001, 2002, 2003, 2004], 'Int_rate':[2,3,2,2]})
df3 = pd.DataFrame({'year':[2001, 2003, 2004, 2005], 'Int_rate':[7,8,9,10]})

merged = pd.merge(df1, df3 , on='year', how='right, left, outer, inner')
merged.set_index('year', inplace=True)


###
#display options
###
#https://www.youtube.com/watch?v=yiO43TQ4xvc&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=28


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.float_format', '{:}'.format)

###
#How do I find and remove duplicate rows in pandas?
###
#https://www.youtube.com/watch?v=ht5buXUMqkQ&index=26&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y

users.loc[users.duplicated(keep='first','last', False)]
users.drop_duplicates(subset=['age','zip_code'], keep=false)


###
#dates and times
###
#https://www.youtube.com/watch?v=yCgJGsg0Xa4&index=25&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y

ufo['Time'] = pd.to_datetime(ufo.Time)
Time.dt.dayofyear


###
#missing values
###
#https://www.youtube.com/watch?v=fCMrO_VzeL8&index=16&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y

ufo[ufo.city.isnull()]
ufo.dropna(how='any')
ufo.dropna(subset=['city','shapeReport'], how='all')
ufo['shapeReport'].value_counts(dropna=False)
ufo['shapeReport'].value_counts(normalize=True)
ufo['shapeReport'].fillna(value='Various', inplace= True)

###
#groupby
###
#https://www.youtube.com/watch?v=qy0fDqoMJx8&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=14

drinks.groupby('continent').beer_serving.mean()
drinks[drinks.continent == 'Europe'].beer_serving.mean()
drinks.groupby('continent').beer_serving.agg(['count','min','max','mean'])


###
#Removing Columns
###
#https://www.youtube.com/watch?v=TW5RqdDBasg&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=2
#
ri.drop('count_name', axis='columns', inplace=True) 



###
#shift by one row
###
#https://www.youtube.com/watch?v=TW5RqdDBasg&list=PL5-da3qGB5IBITZj_dYSFqnd_15JgqwA6&index=2
#
points['lat2']=points.lat.shift(-1)
points['lon2']=points.lon.shift(-1)

####
#drop NaN
###
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
dow = dow.dropna()
