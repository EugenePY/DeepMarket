# coding=utf-8

from data import database
import json
import requests
import pandas as pd
import numpy as np


class DailyUpdateFromRemote(database.GetDataServer):
    def __init__(self, db_name, date_list):
        database.GetDataServer.__init__(self, db_name)
        self.date_list = date_list

    def query_data_from_date(self, date="2016/06/01"):
        formdata = {"index": {"fields": ["date"]},
                    "type": "json"}

        response = json.loads(requests.request("POST", self.creditials["url"] +
                                               "/" + self.database + "/_index",
                                               json=formdata).text)
        print response
        formdata = {
            "selector": {"data.date": date},
            "fields": ["_id", "data"]
        }
        process_data = json.loads(requests.request("POST",
                                                   self.creditials["url"] +
                                                   "/" + self.database + "/_find",
                                                   json=formdata).text)
        return process_data['doc']

    def run_update(self, date_list):
        data = []
        for date in date_list:
            data.append(self.query_data_from_date(date))
        return data


class Raw2DataSet(object):
    def __init__(self):
        self.update_price = DailyUpdateFromRemote()
        self.update_weather = DailyUpdateFromRemote()
        self.update_selltalite = DailyUpdateFromRemote()
        self.raw_list = [self.update_price, self.update_weather,
                         self.update_selltalite]

    def merge_raw_by_date(self, raw_data_list):
        # we consider all the raw_data of price as a data instance.
        df_list = []
        for i, raw_data in enumerate(raw_data_list):
            df_list.extend(self.raw_list[i].process(raw_data))
        df = pd.merge(df_list)
        return df

    def get_raw(self, date_list):
        raw_data_list = []
        for raw in self.get_raw:
            raw_data_list.append(raw.run_update(date_list))
        return raw_data_list


class DataFrame2DataSet(object):
    def __init__(self, price_df, weather_df):
        self.price = price_df
        self.weather = weather_df

    def merge(self, weather, price):
        weather['date'] = pd.to_datetime(weather['date'])
        price['日期'] = pd.to_datetime(price['日期'])
        weather = weather.set_index('date')
        m = pd.merge(price, weather, left_on='日期',
                          right_index=True, how='left')
        return m

    def to_numpy(self):
        market = np.array(weather)[:, :-1]
        # invalid sample
        inval_samples = []
        for k, i in enumerate(market.tolist()):
            for z, o in enumerate(i):
                if len(o) != 481:
                    inval_samples.extend(k)
        return np.array(np.delete(market, inval_samples, axis=0).tolist())

    def market_prices(self):
        # sum of all sub markets for a given date
        price = self.price
        price['日期'] = pd.to_datetime(price['日期'])
        def wgt_avg(val_col_name,wt_col_name):
            def inner(group):
                return (group[val_col_name] * group[wt_col_name]).sum() /\
                    group[wt_col_name].sum()
            inner.__name__ = 'wgt_avg'
            return inner

        m =  price.groupby("日期", axis=0)
        # merge the market price by date
        return pd.merge(price, pd.DataFrame(m.apply(
            wgt_avg('平均價', '交易量')), columns=['市場價']), left_on="日期",
             right_index=True, how='left')

    def output_all(self):
        # calcualte the market price and merge with the weather feature
        m = self.market_prices()
        return self.merge(self.weather, m)

if __name__ == "__main__":
    import cPickle as pkl
    weather = pkl.load(open('./data/all_weather.pkl', 'rb'))
    price = pkl.load(open('./data/all_price.pkl', 'rb'))
    m = DataFrame2DataSet(price, weather).output_all()
    pkl.dump(m, open('./data/training_data', 'wb'))
