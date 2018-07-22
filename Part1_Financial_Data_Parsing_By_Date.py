import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.float_format', '{:}'.format)

#Classify Long==1, Short==0
def is_long_short(num):
    ret = ''
    if (1.0 <= num):
        ret = 1
    else:
        ret = 0
    return ret

#Seperate DATE into Year, Month and Day
def DATE_DIVIDER(DataFrame):
    DataFrame['DATE'] = pd.to_datetime(DataFrame['DATE'])
    DataFrame['Year'] = DataFrame.DATE.dt.year
    DataFrame['Month'] = DataFrame.DATE.dt.month
    DataFrame['Day'] = DataFrame.DATE.dt.day
    #DataFrame = DataFrame.drop(columns=['DATE'])
    return DataFrame
#Convert Yearly to Daily
def Converter_Yearly_Daily(Name_O,  Indicator, start_date):
    usb = CustomBusinessDay(calendar = USFederalHolidayCalendar())
    rng = pd.date_range(start=start_date,end="1/1/2018",freq=usb)
    Name_Daily = pd.DataFrame({'DATE': rng})
    Name_Daily['Year'] = Name_Daily.DATE.dt.year
    Name_Daily['Month'] = Name_Daily.DATE.dt.month
    Name_Daily['Day'] = Name_Daily.DATE.dt.day
    year = Name_Daily['Year'].unique()
    year = np.append(year, [2018])
    arr = []
    index = 0
    for y in year:
        numberofDAYS = len(Name_Daily.loc[Name_Daily.Year == y])
        for day in range(0, numberofDAYS):
            arr.append(Name_O[Indicator][index] + (Name_O[Indicator][index+1] - Name_O[Indicator][index]) * day / numberofDAYS)
        index = index + 1

    Name_Daily[Indicator] = arr
    #Name_Daily = Name_Daily.drop(columns=['Year','Month','Day'])
    #out_filename = "Economy_Indicators_Data"+ "\\" + Indicator + '_Daily_Insoo.csv'
    #Name_Daily.to_csv(out_filename, sep=',', encoding='utf-8')
    return Name_Daily

#Convert Yearly to Daily
def Converter_Quartly_Daily(Name_O, Indicator, start_date):
    #How can I use pandas interpolate?
    usb = CustomBusinessDay(calendar = USFederalHolidayCalendar())
    rng = pd.date_range(start=start_date,end="1/1/2018",freq=usb)
    Name_Daily = pd.DataFrame({'DATE': rng})
    Name_Daily['DATE'] = pd.to_datetime(Name_Daily.DATE)
    Name_Daily['Year'] = Name_Daily.DATE.dt.year
    Name_Daily['Month'] = Name_Daily.DATE.dt.month
    Name_Daily['Day'] = Name_Daily.DATE.dt.day
    #Name_Daily = Name_Daily.drop(columns=['DATE'])

    year = Name_Daily['Year'].unique()
    year = np.append(year, [2018])
    arr = []
    index = 0
    for y in year:
        newY = Name_Daily.loc[Name_Daily.Year == y]
        numberofDAYS = len(newY.loc[(newY.Month == 1) | (newY.Month == 2)| (newY.Month == 3)])
        for day in range(0, numberofDAYS):
            arr.append(Name_O[Indicator][index] + (Name_O[Indicator][index+1] - Name_O[Indicator][index]) * day / numberofDAYS)
        index = index + 1
        numberofDAYS = len(newY.loc[(newY.Month == 4) | (newY.Month == 5)| (newY.Month == 6)])
        for day in range(0, numberofDAYS):
            arr.append(Name_O[Indicator][index] + (Name_O[Indicator][index+1] - Name_O[Indicator][index]) * day / numberofDAYS)
        numberofDAYS = len(newY.loc[(newY.Month == 7) | (newY.Month == 8)| (newY.Month == 9)])
        index = index + 1
        for day in range(0, numberofDAYS):
            arr.append(Name_O[Indicator][index] + (Name_O[Indicator][index+1] - Name_O[Indicator][index]) * day / numberofDAYS)
        numberofDAYS = len(newY.loc[(newY.Month == 10) | (newY.Month == 11)| (newY.Month == 12)])
        index = index + 1
        for day in range(0, numberofDAYS):
            arr.append(Name_O[Indicator][index] + (Name_O[Indicator][index+1] - Name_O[Indicator][index]) * day / numberofDAYS)
        index = index + 1

    Name_Daily[Indicator] = arr
    #Name_Daily = Name_Daily.drop(columns=['Year','Month','Day'])
    #out_filename = "Economy_Indicators_Data"+ "\\" + Indicator + '_Daily_Insoo.csv'
    #Name_Daily.to_csv(out_filename, sep=',', encoding='utf-8')
    return Name_Daily

#Convert Monthly to Daily
def Converter_Monthly_Daily(Name_O, Indicator, start_date):
    usb = CustomBusinessDay(calendar = USFederalHolidayCalendar())
    rng = pd.date_range(start=start_date,end="1/1/2018",freq=usb)
    Name_Daily = pd.DataFrame({'DATE': rng})
    Name_Daily['DATE'] = pd.to_datetime(Name_Daily.DATE)
    Name_Daily['Year'] = Name_Daily.DATE.dt.year
    Name_Daily['Month'] = Name_Daily.DATE.dt.month
    Name_Daily['Day'] = Name_Daily.DATE.dt.day
    #Name_Daily = Name_Daily.drop(columns=['DATE'])

    year = Name_Daily['Year'].unique()
    year = np.append(year, [2018])
    arr = []
    index = 0
    for y in year:
        newY = Name_Daily.loc[Name_Daily.Year == y]
        for m in range(1,13):
            numberofDAYS = len(newY.loc[(newY.Month == m)])
            for day in range(0, numberofDAYS):
                arr.append(Name_O[Indicator][index] + (Name_O[Indicator][index+1] - Name_O[Indicator][index]) * day / numberofDAYS)
            index = index + 1

    #Name_Daily = Name_Daily.drop(columns=['Year','Month','Day'])
    Name_Daily[Indicator] = arr
    #out_filename = "Economy_Indicators_Data"+ "\\" + Indicator + '_Daily_Insoo.csv'
    #Name_Daily.to_csv(out_filename,sep=',', encoding='utf-8')
    return Name_Daily

#filter '.' value to previous values
def dotToPrevious(Name_O, Indicator):
    num = Name_O.index[Name_O[Indicator] == '.'].tolist()
    for i in num:
        Name_O[Indicator][i] = Name_O[Indicator][i-1]

    Name_O[Indicator] = pd.to_numeric(Name_O[Indicator]).astype(float)
    return Name_O

DATE_YMD = ['DATE','Year','Month','Day']
SnP = pd.read_csv(r'Economy_Indicators_Data\S&P 500 ^GSPC_Daily.csv',sep=',', date_parser='Date')
SnP = SnP.drop(columns=['High','Low','Close','Adj Close', 'Volume'])
SnP.columns = ['DATE','SnP']
SnP = DATE_DIVIDER(SnP)
SnP = pd.DataFrame({'DATE':SnP.DATE,'Year':SnP.Year,'Month':SnP.Month,'Day':SnP.Day,'SnP':SnP.SnP})

GDP = pd.read_csv(r"Economy_Indicators_Data\Gross Domestic ProductGDP_Quarterly.csv", sep=',', date_parser='DATE')
GDP.columns = ['DATE','GDP']
GDP = DATE_DIVIDER(GDP)
start_date = GDP.DATE[0]

#Convert Quarterly GDP to Daily GDP
GDP_Daily = Converter_Quartly_Daily(GDP, 'GDP', start_date)

#merge GDP Data to the BigFile
data = pd.merge(SnP, GDP_Daily, on=DATE_YMD)

#Import Export data from WTO
#3 features each (Unit value, Volume, value)
Import_Export= pd.read_csv(r"Economy_Indicators_Data\USA_merchandise_indices_annual_dataset.csv", sep=',', date_parser='Year')
Import_Export = Import_Export.drop(columns=['Reporter_code','Reporter_description','Partner_code','Partner_description', 'Indicator_code', 'Indicator_description', 'Flow_Code','Flag','Source_Description','Note'])
Import_Export.columns = ['Flow_Description','Year','Unit','Value']
Import_Export = pd.DataFrame({'Flow_Description':Import_Export.Flow_Description,'Year':Import_Export.Year + 1, 'Unit':Import_Export.Unit,'Month':1, 'Day':1,'Import_Export':Import_Export.Value})
Import_Export['Date'] = pd.to_datetime(Import_Export[['Day','Month','Year']])
# import1
Import_Unit_Value = Import_Export.loc[(Import_Export.Unit == 'Unit value index - Prev. year = 100')&(Import_Export.Flow_Description == 'Imports')]
Import_Unit_Value = Import_Unit_Value.drop(columns=['Unit','Flow_Description','Year','Month','Day'])
Import_Unit_Value.columns = ['Import_Unit_Value','Date']
Import_Unit_Value = Import_Unit_Value.reset_index(drop=True)
Import_Unit_Value = Import_Unit_Value.reindex(columns=['Date','Import_Unit_Value'])
# import2
Import_Volume = Import_Export.loc[(Import_Export.Unit == 'Volume index - Previous year = 100')&(Import_Export.Flow_Description == 'Imports')]
Import_Volume = Import_Volume.drop(columns=['Unit','Flow_Description','Year','Month','Day'])
Import_Volume.columns = ['Import_Volume','Date']
Import_Volume = Import_Volume.reset_index(drop=True)
Import_Volume = Import_Volume.reindex(columns=['Date','Import_Volume'])
# import3
Import_Value = Import_Export.loc[(Import_Export.Unit == 'Value index - Previous year = 100')&(Import_Export.Flow_Description == 'Imports')]
Import_Value = Import_Value.drop(columns=['Unit','Flow_Description','Year','Month','Day'])
Import_Value.columns = ['Import_Value','Date']
Import_Value = Import_Value.reset_index(drop=True)
Import_Value = Import_Value.reindex(columns=['Date','Import_Value'])
# export1
Export_Unit_Value = Import_Export.loc[(Import_Export.Unit == 'Unit value index - Prev. year = 100')&(Import_Export.Flow_Description == 'Imports')]
Export_Unit_Value = Export_Unit_Value.drop(columns=['Unit','Flow_Description','Year','Month','Day'])
Export_Unit_Value.columns = ['Export_Unit_Value','Date']
Export_Unit_Value = Export_Unit_Value.reset_index(drop=True)
Export_Unit_Value = Export_Unit_Value.reindex(columns=['Date','Export_Unit_Value'])
# export2
Export_Volume = Import_Export.loc[(Import_Export.Unit == 'Volume index - Previous year = 100')&(Import_Export.Flow_Description == 'Imports')]
Export_Volume = Export_Volume.drop(columns=['Unit','Flow_Description','Year','Month','Day'])
Export_Volume.columns = ['Export_Volume','Date']
Export_Volume = Export_Volume.reset_index(drop=True)
Export_Volume = Export_Volume.reindex(columns=['Date','Export_Volume'])
# export3
Export_Value = Import_Export.loc[(Import_Export.Unit == 'Value index - Previous year = 100')&(Import_Export.Flow_Description == 'Imports')]
Export_Value = Export_Value.drop(columns=['Unit','Flow_Description','Year','Month','Day'])
Export_Value.columns = ['Export_Value','Date']
Export_Value = Export_Value.reset_index(drop=True)
Export_Value = Export_Value.reindex(columns=['Date','Export_Value'])


start_date = Import_Unit_Value.Date[0]
Import_Unit_Value_Daily = Converter_Yearly_Daily(Import_Unit_Value, 'Import_Unit_Value', start_date)
Import_Volume_Daily = Converter_Yearly_Daily(Import_Volume, 'Import_Volume', start_date)
Import_Value_Daily = Converter_Yearly_Daily(Import_Value, 'Import_Value', start_date)

Export_Unit_Value_Daily = Converter_Yearly_Daily(Export_Unit_Value, 'Export_Unit_Value', start_date)
Export_Volume_Daily = Converter_Yearly_Daily(Export_Volume, 'Export_Volume', start_date)
Export_Value_Daily = Converter_Yearly_Daily(Export_Value, 'Export_Value', start_date)

data = pd.merge(data, Import_Unit_Value_Daily, on=DATE_YMD)
data = pd.merge(data, Import_Volume_Daily, on=DATE_YMD)
data = pd.merge(data, Import_Value_Daily, on=DATE_YMD)
data = pd.merge(data, Export_Unit_Value_Daily, on=DATE_YMD)
data = pd.merge(data, Export_Volume_Daily, on=DATE_YMD)
data = pd.merge(data, Export_Value_Daily, on=DATE_YMD)

#Monthly MoneyBase
MonetaryBase = pd.read_csv(r'Economy_Indicators_Data\Monetary Base; Total BOGMBASE_Monthly.csv', sep=',', date_parser="DATE")
MonetaryBase.columns = ['DATE','MonetaryBase']
MonetaryBase = DATE_DIVIDER(MonetaryBase)
start_date = MonetaryBase.DATE[0]
MonetaryBase_Daily = Converter_Monthly_Daily(MonetaryBase,'MonetaryBase',start_date)
data = pd.merge(data, MonetaryBase_Daily, on=DATE_YMD)

#Comsumer Price Index
CPI = pd.read_csv(r'Economy_Indicators_Data\Consumer Price Index for All Urban Consumers All Items.csv', sep=',', date_parser="DATE")
CPI.columns = ['DATE','CPI']
CPI = DATE_DIVIDER(CPI)
start_date = CPI.DATE[0]
CPI_Daily = Converter_Monthly_Daily(CPI,'CPI',start_date)
data = pd.merge(data, CPI_Daily, on=DATE_YMD)

HomePrice = pd.read_csv(r'Economy_Indicators_Data\All-Transactions House Price Index for the United StatesUSSTHPI.csv', sep=',', date_parser="DATE")
HomePrice.columns = ['DATE','HomePrice']
HomePrice['DATE'] = pd.to_datetime(HomePrice['DATE'])
start_date = HomePrice.DATE[0]
#HomePrice = DATE_DIVIDER(HomePrice)
HomePrice_Daily = Converter_Quartly_Daily(HomePrice, 'HomePrice',start_date)
data = pd.merge(data, HomePrice_Daily, on=DATE_YMD)

Loans = pd.read_csv(r'Economy_Indicators_Data\Commercial and Industrial Loans, Top 100 Banks Ranked by AssetsACILT100.csv', sep=',', date_parser="DATE")
Loans.columns = ['DATE','Loans']
Loans['DATE'] = pd.to_datetime(Loans['DATE'])
start_date = Loans.DATE[0]
#Loans = DATE_DIVIDER(Loans)
Loans_Daily = Converter_Quartly_Daily(Loans, 'Loans',start_date)
data = pd.merge(data, Loans_Daily, on=DATE_YMD)

Employment = pd.read_csv(r'Economy_Indicators_Data\All Employees Total Nonfarm PayrollsPAYNSA.csv', sep=',', date_parser="DATE")
Employment.columns = ['DATE','Employment']
Employment['DATE'] = pd.to_datetime(Employment['DATE'])
start_date = Employment.DATE[0]
#Employment = DATE_DIVIDER(Employment)
Employment_Daily = Converter_Monthly_Daily(Employment,'Employment',start_date)
data = pd.merge(data, Employment_Daily, on=DATE_YMD)

Income = pd.read_csv(r'Economy_Indicators_Data\Personal IncomePI.csv', sep=',', date_parser="DATE")
Income.columns = ['DATE','Income']
Income['DATE'] = pd.to_datetime(Income['DATE'])
start_date = Income.DATE[0]
#Income = DATE_DIVIDER(Income)
Income_Daily = Converter_Monthly_Daily(Income,'Income',start_date)
data = pd.merge(data, Income_Daily, on=DATE_YMD)

ConstructionSpending = pd.read_csv(r'Economy_Indicators_Data\Total Construction Spending.csv', sep=',', date_parser="DATE")
ConstructionSpending.columns = ['DATE', 'ConstructionSpending']
ConstructionSpending = DATE_DIVIDER(ConstructionSpending)
start_date = ConstructionSpending.DATE[0]
ConstructionSpending_Daily = Converter_Monthly_Daily(ConstructionSpending,'ConstructionSpending',start_date)
data = pd.merge(data, ConstructionSpending_Daily, on=DATE_YMD)

FedFundRate_Daily = pd.read_csv(r'Economy_Indicators_Data\Effective Federal Funds Rate_Daily_DFF.csv', sep=',', date_parser="DATE")
FedFundRate_Daily.columns = ['DATE', 'FedFundRate']
FedFundRate_Daily = DATE_DIVIDER(FedFundRate_Daily)
data = pd.merge(data, FedFundRate_Daily, on=DATE_YMD)

USDollar_Daily = pd.read_csv(r'Economy_Indicators_Data\Trade Weighted U.S. Dollar Index BroadDTWEXB.csv', sep=',', date_parser="DATE")
USDollar_Daily.columns = ['DATE', 'USDollar']
USDollar_Daily = DATE_DIVIDER(USDollar_Daily)
USDollar_Daily = dotToPrevious(USDollar_Daily, 'USDollar')
data = pd.merge(data, USDollar_Daily, on=DATE_YMD)

Crude_Oil_Daily = pd.read_csv(r'Economy_Indicators_Data\Crude Oil Prices West Texas Intermediate (WTI)DCOILWTICO.csv', sep=',', date_parser="DATE")
Crude_Oil_Daily.columns = ['DATE', 'CrudeOil']
Crude_Oil_Daily = DATE_DIVIDER(Crude_Oil_Daily)
Crude_Oil_Daily = dotToPrevious(Crude_Oil_Daily, 'CrudeOil')
data = pd.merge(data, Crude_Oil_Daily, on=DATE_YMD)

titles = ['SnP', 'GDP', 'MonetaryBase', 'CPI', 'HomePrice', 'Loans', 'Employment',
         'Income', 'ConstructionSpending', 'FedFundRate', 'USDollar', 'CrudeOil', 'Import_Unit_Value',
          'Import_Volume', 'Import_Value', 'Export_Unit_Value', 'Export_Volume', 'Export_Value' ]

for title in titles:
    data[title + "2"] = data[title].shift(-1)
data = data.dropna()

for title in titles:
    data["pct_" + title] = 1 + (data[title+"2"] - data[title])/data[title]

data['long_short'] = data['pct_SnP'].apply(is_long_short)
out_filename = "Ultimate_Data" + "\\" + "Ultimate_Assortion_pct_Change_Daily_Insoo.csv"
data.to_csv(out_filename, sep=',', encoding='utf-8')

for title in titles:
    data = data.drop(columns=["pct_" + title])

monthly_data = pd.DataFrame()
year = data['Year'].unique()
month = data['Month'].unique()
for y in year:
    for m in month:
        coco = data.loc[(data['Year'] == y) & (data['Month'] == m)]
        monthly_data = monthly_data.append(coco.iloc[0][:])

for title in titles:
    monthly_data["pct_" + title] = 1 + (monthly_data[title+"2"] - monthly_data[title])/monthly_data[title]

monthly_data['long_short'] = monthly_data['pct_SnP'].apply(is_long_short)
out_filename = "Ultimate_Data" + "\\" + "Ultimate_Assortion_pct_Change_Monthly_Insoo.csv"
monthly_data.to_csv(out_filename, sep=',', encoding='utf-8')

quarterly_data = pd.DataFrame()
month = [1,4,7,10]
for y in year:
    for m in month:
        coco = data.loc[(data['Year'] == y) & (data['Month'] == m)]
        quarterly_data = quarterly_data.append(coco.iloc[0][:])

for title in titles:
    quarterly_data["pct_" + title] = 1 + (quarterly_data[title+"2"] - quarterly_data[title])/quarterly_data[title]

quarterly_data['long_short'] = quarterly_data['pct_SnP'].apply(is_long_short)
out_filename = "Ultimate_Data" + "\\" + "Ultimate_Assortion_pct_Change_Quarterly_Insoo.csv"
quarterly_data.to_csv(out_filename, sep=',', encoding='utf-8')

yearly_data = pd.DataFrame()
for y in year:
    coco = data.loc[(data['Year'] == y) & (data['Month'] == m)]
    yearly_data = yearly_data.append(coco.iloc[0][:])

for title in titles:
    yearly_data["pct_" + title] = 1 + (yearly_data[title+"2"] - yearly_data[title])/yearly_data[title]

yearly_data['long_short'] = yearly_data['pct_SnP'].apply(is_long_short)
out_filename = "Ultimate_Data" + "\\" + "Ultimate_Assortion_pct_Change_Yearly_Insoo.csv"
yearly_data.to_csv(out_filename, sep=',', encoding='utf-8')
