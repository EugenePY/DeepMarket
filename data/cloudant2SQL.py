# coding=utf-8

from ibmdbpy.base import IdaDataBase
from data import setting
import json
import requests
import pandas as pd
import datetime
import re
# Query cloudant


def test():
    creditials = setting.CREDIT["cloudantdb"]["credentials"]
    formdata = {
        "selector": {"_id": {"$gt": 0}},
        "fields": ["_id", "_rev", "data"],
        "sort": [{"_id": "asc"}]
    }
    # print creditials["url"]

    process_data = json.loads(requests.request("POST", creditials["url"] +
                                               "/price/_find",
                                               json=formdata).text)['docs']

    column_names = reduce(
        lambda x, y: x+y, process_data[1]["data"]["data"][:3][2:] +
        [[u"日期", u"市場"]])
    column_names[-3] = u"平均交易量增減%"

    column_names[-5] = u"平均交易價格增減%"
    f = open('./data/col_name/column_names', 'w')
    f.writelines('\n'.join([name.encode("utf-8") for name in column_names]))
    """
    pattern = re.compile(u"[年月日]")
    credi = json.load(open('./vcap.json', 'rb'))['dashdb']['credentials']
    jdurl = credi['jdbcurl'] + ":user=" + credi['username'] +\
        ";password=" + credi["password"]
    idadb = IdaDataBase(dsn=jdurl)
    # idadb.drop_table("vegetables")
    idadb.as_idadataframe(data, tablename="vegetables")
    # idadf = IdaDataFrame(idadb, "IRIS", indexer="ID")
    """


def string2datetime(string):
    # convert "105年3月7號" into 2016/03/07
    pattern = re.compile(u"[年月日]")
    year, month, day = [int(i) for i in pattern.split(string)[:-1]]
    year = 1911 + year  # convert to YYYY
    return datetime.datetime(year, month, day)


def add_column(doc):
        date = string2datetime(doc["data"]["data"][0][0].split(":")[1])
        market = doc["data"]["data"][0][2].split(":")[1]
        doc = [i + [date, market] for i in doc['data']['data']]
        return doc[3:]


def cloudant2SQL(cloudant_data, column_names):
    m = reduce(lambda x, y: x + y, map(add_column, cloudant_data))
    data = pd.DataFrame(m).convert_objects(convert_numeric=True)
    data.columns = column_names
    return data


def add_column_json(json_list):
    try:
        date = string2datetime(json_list["data"][0][0].split(":")[1])
        market = json_list["data"][0][2].split(":")[1]
        doc = [i + [date, market] for i in json_list['data']]
        return doc[3:]
    except:
        return []


def json2dataframe_price(json_list, column_names):
    m = reduce(lambda x, y: x + y, map(add_column_json, json_list))
    data = pd.DataFrame(m).convert_objects(convert_numeric=True)
    data.columns = column_names
    return data


def json2dataframe_weather(json_list, column_names):
    import os
    import cPickle as pkl
    file_list = os.listdir('./data/data_all')
    weather_list = []
    price_list = []

    for i, file_name in enumerate(file_list):
        print "processing: {0} doc {1}".format(i, file_name)
        kind = file_name.split("_")[0]
        data = json.load(open('./data/data_all/' + file_name, 'rb'))
        if kind == "price":
            price_list.append(data)
        elif kind == "weather":
            m = data['data']
            m['date'] = data['date']
            weather_list.append(m)

    price = pd.DataFrame(weather_list)
    pkl.dump(price, open('./data/all_weather.pkl', 'wb'))

if __name__ == "__main__":
    import numpy as np
    p = np.random.normal(size=(100, 1))
    Qd = 1. - 0.5 * p
    weather = (0.5 + 0.3 * Qd - p) * 0.3
    data1 = pd.DataFrame(np.hstack([p, Qd, weather]),
                         columns=['market_price', 'market_volumn',
                                  'weather_shock_level'])

    p = np.random.normal(loc=1, size=(100, 1))
    weather = (0.5 + 0.3 * Qd - p) * 0.3
    data2 = pd.DataFrame(np.hstack([p, Qd, weather]),
                         columns=['market_price', 'market_volumn',
                                  'weather_shock_level'])

    credi = json.load(open('./data/vcap.json', 'rb'))['dashdb']['credentials']
    jdurl = credi['jdbcurl'] + ":user=" + credi['username'] +\
        ";password=" + credi["password"]
    idadb = IdaDataBase(dsn=jdurl)
    idadb.as_idadataframe(data2, tablename="vegetables_fake_after_shock")
    idadb.as_idadataframe(data1, tablename="vegetables_fake")
