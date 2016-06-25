import couchdb
import cPickle as pkl
from util.log import logger

import requests
import json

import datetime


class CloudantDB(couchdb.client.Database):
    def __init__(self, name):
        self.db_name = name
        couchdb.client.Database.__init__(self, None)
        self.init()

    def init(self):
        logger.debug("Initial the database")

        try:
            CREDIT = json.load(open('./util/vcap.json'))[
                'cloudantDB']['credentials']
            self.server = couchdb.Server(CREDIT['url'])
            self.server.resource.credentials = (CREDIT['username'],
                                                CREDIT['password'])
            try:
                db = self.server.create(self.db_name)
                logger.debug("Create a new database " + self.db_name)
            except:
                db = self.server.__getitem__(self.db_name)
                logger.debug("Use Data Base" + self.db_name)

            logger.debug("Create datadase successfully")
            self.__dict__.update(db.__dict__)
        except:
            print('cannot find the credentials pls bind a CloudantDB Service')

        return self
        # Query results are treated as iterators, like this:
        # print all docs in the database

    def query_massage(self):
        # params should fit this format
        # this is a wrapper of the cochdb map function
        # ref: http://www.slideshare.net/okurow/couchdb-mapreduce-13321353
        map_fun = '''
        function(doc) {
        if (doc.date == "2016/05/12")
            emit(doc);

        }'''
        return self.query(map_fun, descending=True)

    def store(self, data):
        self.save(data)


class DataSet(object):
    def __init__(self, name, batch_size, seq_length):
        self.name = name
        self.data_dir = "/home/eugene/tensorflow_vegetable/data/"
        self.batch_size = batch_size
        self.data_holder = self._load()
        self.seq_length = seq_length
        self.shape = self.data_holder.shape
        self.reset_pointer()

    def _trans_input(self, data, pointer):
        """
        select the sequence seq_length for the training
        """
        batch = data[pointer:
                     pointer + self.batch_size * self.seq_length]
        return batch.reshape((self.batch_size, self.seq_length,
                              self.shape[1], self.shape[2]))

    def __getitem__(self, key):
        return self.data_holder[key]

    def next_batch(self):
        log.logger.info('current batch %i', self.pointer)
        batch_x, batch_y = self._trans_input(self.data_holder, self.pointer),\
            self._trans_input(self.data_holder, self.pointer + 1)
        self.pointer += 1
        return batch_x, batch_y

    def reset_pointer(self):
        self.pointer = 0

    def _load(self):
        data_path = self.data_dir + self.name
        return pkl.load(open(data_path, 'rb'))

    def _database(self):
        pass


if __name__ == "__main__":
    #print DataSet('fake.npy.pkl', 10, 3).next_batch()
    get_url = json.load(open('./util/vcap.json', 'rb'))
    r = requests.post(get_url['cloudantDB']['credentials']['url']+'/sattellite/_find',
                      json={
                                      "selector": {
                                          "date": "2016/05/12"}
                })
