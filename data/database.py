# coding=utf-8
import couchdb

import json
import requests
from ibmdbpy.base import IdaDataBase
from data.cloudant2SQL import cloudant2SQL
from io import BytesIO
from PIL import Image
import numpy as np
from data import setting


class CloudantDB(couchdb.client.Database):

    def __init__(self, name):
        self.db_name = name
        couchdb.client.Database.__init__(self, None)
        self.init()

    def init(self):
        credit = setting.CREDIT["cloudantdb"]["credentials"]
        print "Initial the database"
        couch = couchdb.Server("https://%s.cloudant.com" %
                               credit['username'])
        couch.resource.credentials = (credit['username'], credit['password'])
        try:
            db = couch.create(self.db_name)
            print "Create a new database " + self.db_name
        except:
            db = couch.__getitem__(self.db_name)
            print "Use Data Base " + self.db_name

        print "Create datadase successfully"
        self.__dict__.update(db.__dict__)
        return self
# Query results are treated as iterators, like this:
# print all docs in the database

    def query_massage(self, params):
        params = ''
# params should fit this format
# this is a wrapper of the cochdb map function
# ref: http://www.slideshare.net/okurow/couchdb-mapreduce-13321353
        map_fun = '''
        function(doc) {
        if (doc.id == "message_count") {
            if (doc.message_count.push >= %s ) {
                emit(doc)
                }
            }
        }
        '''.format(str(params['push']))
        return self.query(map_fun)

    def store(self, data):
        self.save(data)


class GetDataServer(object):
    def __init__(self, db_name, time_range=None):
        # time_range: a seq of datetime object which whant to access data

        # Loading the credientials and inital the database
        self.creditials = setting.CREDIT["cloudantdb"]["credentials"]
        credi = json.load(open('./data/vcap.json', 'rb'))['dashdb']['credentials']
        jdurl = credi['jdbcurl'] + ":user=" + credi['username'] + \
            ";password=" + credi["password"]
        self.idadb = IdaDataBase(dsn=jdurl)
        self.database = db_name

    def history(self):
        # TODO add logger functional
        pass

    def query_data_from_date(self, database="price", date="2016/06/01"):
        # Query data from the Cloudant
        # need to check the index exist in the cloudantdb
        # create the field
        formdata = {"index": {"fields": ["date"]},
                    "type": "json"}

        response = json.loads(requests.request("POST", self.creditials["url"] +
                                               "/" + database + "/_index",
                                               json=formdata).text)
        print response
        formdata = {
            "selector": {"date": date},
            "fields": ["_id"]
        }
        process_data = json.loads(requests.request("POST",
                                                   self.creditials["url"] +
                                                   "/" + database + "/_find",
                                                   json=formdata).text)
        return process_data

    def get_image(self, _id_list):
        # get the image from cloudant db and return numpy
        responses = [requests.request("GET", self.creditials["url"] +
                                      "/sattellite/" + i + "/image.jpg"
                                      ) for i in _id_list]
        resps = [BytesIO(i.content) for i in responses]
        pics = [np.array(Image.open(res)) for res in resps]
        return np.array(pics)

    def process(self, process_data):
        # NOTE: can be overrided
        return cloudant2SQL(process_data, self.column_names)

    def save2dashdb(self, process_data, tablename="vegetables"):
        print "Uploading"
        self.idadb.as_idadataframe(process_data, tablename)
        print "Upload Success"

    def run_transfer(self, database="price", date="2016/06/01"):
        process_data = self.query_data_from_date(database, date)
        # process usuage
        idadataframe = self.process(process_data)
        self.save2dashdb(idadataframe)

if __name__ == '__main__':
    pass
