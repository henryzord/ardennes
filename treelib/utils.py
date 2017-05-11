# coding=utf-8

import numpy as np
import sqlite3

from treelib import get_total_nodes
from matplotlib import pyplot as plt

__author__ = 'Henry Cagnini'


# noinspection SqlNoDataSourceInspection
# noinspection SqlDialectInspection
class DatabaseHandler(object):
    # columns return by the pragma table_info() command in sqlite3
    table_info_columns = {
        k: i for i, k in enumerate(['cid', 'name', 'type', 'notnull', 'default_value', 'primary_key'])
    }

    modes = ['holdout', 'cross-validation']

    commit_every = 10  # commits data at every N generations

    def __init__(self, path, dataset_name=None, mode=None, n_runs=None, n_individuals=None,
                 n_iterations=None, tree_height=None, decile=None, random_state=None):

        self.path = path

        self.train_hash = None
        self.test_hash = None
        self.val_hash = None
        self.full_hash = None

        self._run = None  # contextual run
        self._id_run = None  # unique id of run
        self.attributes = None
        self.dataset_name = dataset_name

        # TODO resume evolution!
        # TODO add checkpoints for committing changes!

        self._conn = sqlite3.connect(path)
        cursor = self._conn.cursor()

        cursor.execute("""
          CREATE TABLE IF NOT EXISTS EVALUATION_MODES (
            mode TEXT NOT NULL PRIMARY KEY
          );
        """)

        cursor.execute("""
          CREATE TABLE IF NOT EXISTS ATTRIBUTES (
            dataset_name TEXT NOT NULL,
            attribute TEXT NOT NULL,
            FOREIGN KEY (dataset_name) REFERENCES EVOLUTION(dataset_name),
            CONSTRAINT unique_columns_attributes PRIMARY KEY (dataset_name, attribute)
          );
        """)

        cursor.execute("""
          CREATE TABLE IF NOT EXISTS EVOLUTION (
            id_evolution INTEGER NOT NULL PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            n_runs INTEGER NOT NULL,
            n_individuals INTEGER NOT NULL,
            n_iterations INTEGER NOT NULL,
            max_tree_height INTEGER NOT NULL,
            max_n_nodes INTEGER NOT NULL,
            decile REAL NOT NULL,
            random_state INTEGER DEFAULT NULL,
            CONSTRAINT unique_columns_evolution UNIQUE (
              dataset_name, mode, n_runs, n_individuals, n_iterations,
              max_tree_height, max_n_nodes, decile, random_state
            ),
            FOREIGN KEY(mode) REFERENCES EVALUATION_MODES(mode)
          );
        """)
        cursor.execute("""
          CREATE TABLE IF NOT EXISTS SETS (
            hashkey INTEGER NOT NULL PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            relation_name TEXT NOT NULL,
            n_instances INTEGER NOT NULL,
            n_attributes INTEGER NOT NULL,
            n_classes INTEGER NOT NULL,
            FOREIGN KEY (dataset_name) REFERENCES EVOLUTION (dataset_name)
        );""")
        cursor.execute("""
          CREATE TABLE IF NOT EXISTS RUNS (
            id_run INTEGER NOT NULL PRIMARY KEY,
            run INTEGER NOT NULL,
            train_hashkey INTEGER NOT NULL,
            val_hashkey INTEGER DEFAULT NULL,
            test_hashkey INTEGER DEFAULT NULL,
            FOREIGN KEY(train_hashkey) REFERENCES SETS(hashkey),
            FOREIGN KEY(val_hashkey) REFERENCES SETS(hashkey),
            FOREIGN KEY(test_hashkey) REFERENCES SETS(hashkey),
            CONSTRAINT unique_columns_runs UNIQUE (run, train_hashkey, val_hashkey, test_hashkey)
        );""")
        cursor.execute("""
          CREATE TABLE IF NOT EXISTS POPULATION (
            id_run INTEGER NOT NULL,
            iteration INTEGER NOT NULL,
            individual INTEGER NOT NULL,
            fitness REAL NOT NULL,
            height INTEGER NOT NULL,
            n_nodes INTEGER NOT NULL,
            train_correct INTEGER NOT NULL,
            val_correct INTEGER DEFAULT NULL,
            test_correct INTEGER DEFAULT NULL,
            dot TEXT DEFAULT NULL,
            FOREIGN KEY(id_run) REFERENCES RUNS(id_run),
            CONSTRAINT unique_columns_population UNIQUE (id_run, iteration, individual)
          );
        """)

        cursor.execute("""SELECT COUNT(*) FROM EVOLUTION;""")
        count = cursor.fetchone()[0]
        if count == 0:
            max_n_nodes = get_total_nodes(tree_height - 1)  # max nodes a tree can have
            n_variables = get_total_nodes(tree_height - 2)  # useful variables in the prototype tree

            _prototype_columns_insert = ', '.join(['NODE_%d REAL NOT NULL' % i for i in xrange(n_variables)])
            self._prototype_columns = ', '.join(['NODE_%d' % i for i in xrange(n_variables)])

            for p_mode in DatabaseHandler.modes:
                cursor.execute("""
                  INSERT INTO EVALUATION_MODES (mode) VALUES ('%s')
                """ % p_mode)

            cursor.execute("""
              CREATE TABLE IF NOT EXISTS PROTOTYPE (
                id_run INTEGER NOT NULL,
                iteration INTEGER NOT NULL,
                attribute TEXT NOT NULL,
                %s,
                FOREIGN KEY (id_run) REFERENCES RUNS(id_run),
                FOREIGN KEY (attribute) REFERENCES ATTRIBUTES(attribute),
                CONSTRAINT unique_columns_prototype UNIQUE (id_run, iteration, attribute)
               )
             """ % _prototype_columns_insert)

            evolution_columns = (dataset_name, mode, n_runs, n_individuals, n_iterations, tree_height, max_n_nodes, decile, random_state)
            cursor.execute("""
              INSERT INTO EVOLUTION (id_evolution, dataset_name, mode, n_runs, n_individuals,
                n_iterations, max_tree_height, max_n_nodes, decile, random_state) VALUES (
                %d, '%s', '%s', %d, %d, %d, %d, %d, %f, %s)""" % (
                    hash(tuple(evolution_columns)), dataset_name, mode, n_runs, n_individuals, n_iterations, tree_height, max_n_nodes, decile,
                    random_state if random_state is not None else '\'NULL\''
                )
            )
        else:
            self.dataset_name = cursor.execute("""SELECT DATASET_NAME FROM EVOLUTION;""").fetchone()[0]

            self._id_run = cursor.execute("""SELECT ID_RUN FROM RUNS""").fetchone()[0]
            self.attributes = [x[0] for x in cursor.execute("""SELECT ATTRIBUTE FROM ATTRIBUTES;""").fetchall()]

            cursor.execute("""SELECT RELATION_NAME, HASHKEY FROM SETS;""")
            rows = cursor.fetchall()

            for relation_name, hashkey in rows:
                exec('self.%s_hash = %d' % (relation_name, hashkey))

        self.closed = False

    def set_run(self, run):
        self._run = run

    def get_cursor(self):
        return self._conn.cursor()

    def close(self):
        if self._conn is not None and not self.closed:
            self._conn.commit()
            self._conn.close()
        self.closed = True

    def commit(self):
        if not self.closed:
            self._conn.commit()

    @staticmethod
    def get_hash(dataset):
        return hash(tuple(dataset.apply(lambda x: hash(tuple(x)), axis=1)))

    def write_attributes(self, attributes):
        cursor = self._conn.cursor()

        for attribute in attributes:
            cursor.execute("""
                INSERT INTO ATTRIBUTES (dataset_name, attribute) VALUES ('%s', '%s')
              """ % (self.dataset_name, attribute)
            )

        self.attributes = attributes
        cursor.close()

    def write_sets(self, data):
        cursor = self._conn.cursor()

        for d in data:  # type: dict
            exec('self.%s_hash = %d' % (d['relation_name'], d['hashkey']))

            cursor.execute("""
              INSERT INTO SETS (hashkey, dataset_name, relation_name, n_instances, n_attributes, n_classes) VALUES (
              %d, '%s', '%s', %d, %d, %d
              )""" % (
                    d['hashkey'], self.dataset_name, d['relation_name'],
                    d['n_instances'], d['n_attributes'], d['n_classes']
                )
            )

        self._id_run = hash((self._run, self.train_hash, self.val_hash, self.test_hash))

        cursor.execute("""
          INSERT INTO RUNS (id_run, run, train_hashkey, val_hashkey, test_hashkey) VALUES (%d, %d, %d, %s, %s)
        """ % (
                self._id_run,
                self._run,
                self.train_hash,
                str(self.val_hash) if self.val_hash is not None else 'NULL',
                str(self.test_hash) if self.test_hash is not None else 'NULL'
            )
        )

        self._id_run = cursor.execute("""SELECT ID_RUN FROM RUNS""").fetchone()[0]

        cursor.close()

    def write_population(self, iteration, population):
        cursor = self._conn.cursor()

        for ind in population:
            cursor.execute("""
              INSERT INTO POPULATION (
                id_run, iteration, individual, fitness, height, n_nodes, train_correct, val_correct, test_correct, dot
                ) VALUES (%d, %d, %d, %f, %d, %d, %d, %s, %s, '%s')""" % (
                self._id_run, iteration, ind.ind_id, ind.fitness, ind.height, ind.n_nodes,
                int(ind.train_acc_score * len(ind.y_train_true)),
                str(int(ind.val_acc_score * len(ind.y_val_true))) if self.val_hash is not None else 'NULL',
                str(int(ind.test_acc_score * len(ind.y_test_true))) if self.test_hash is not None else 'NULL',
                ind.to_dot()
                )
            )

        cursor.close()

        if iteration % self.commit_every == 0:
            self._conn.commit()

    def write_prototype(self, iteration, gm):
        """
        :type iteration: int
        :param iteration:
        :type gm: treelib.graphical_model.GraphicalModel
        :param gm:
        :return:
        """
        cursor = self._conn.cursor()

        for attr in self.attributes:
            cursor.execute("""
               INSERT INTO PROTOTYPE (id_run, iteration, attribute, %s) VALUES (
                %d, %d, '%s', %s
               )
            """ % (self._prototype_columns, self._id_run, iteration, attr, ','.join([str(x) for x in gm.attributes.loc[attr]]))
            )
        cursor.close()

        if iteration % self.commit_every == 0:
            self._conn.commit()

    def union(self, db):
        def convert(value, _type):
            if value == None:
                return 'NULL'
            if _type == 'TEXT':
                return str(value).join("''")
            return str(value)

        def __insert__(self_cursor, other_cursor, tables, treat=False):
            for table_name in tables:
                columns = other_cursor.execute('PRAGMA TABLE_INFO(%s)' % table_name).fetchall()

                column_names = ','.join([x[self.table_info_columns['name']] for x in columns])
                column_types = [x[self.table_info_columns['type']] for x in columns]

                other_data = other_cursor.execute("""SELECT %s FROM %s;""" % (column_names, table_name)).fetchall()

                for data in other_data:
                    try:
                        data_str = ','.join([convert(x, column_types[i]) for i, x in enumerate(data)])
                        self_cursor.execute("""INSERT INTO %s (%s) VALUES (%s);""" % (table_name, column_names, data_str))
                    except Exception, e:
                        if treat:
                            pass
                        else:
                            raise e

        if db.closed:
            db = DatabaseHandler(db.path)

        other_cursor = db.get_cursor()
        self_cursor = self._conn.cursor()

        # those values must be unique in the main database, but will repeat in partial databases
        tables = ['evolution', 'attributes', 'sets']
        __insert__(self_cursor, other_cursor, tables, treat=True)

        # those values are also unique, must do not and should not
        # repeat among databases. throw an exception if this happens!
        tables = ['runs', 'population', 'prototype']
        __insert__(self_cursor, other_cursor, tables, treat=False)
        # TODO ignores id but then two other values are not unique!

        other_cursor.close()
        self_cursor.close()

    def plot_population(self):
        plt.figure()
        ax = plt.subplot(111)

        cursor = self._conn.cursor()

        mode = cursor.execute("""SELECT MODE FROM EVOLUTION""").fetchone()[0]

        if mode == 'holdout':
            # TODO group by run!
            # TODO group by run!
            # TODO group by run!
            n_iterations = cursor.execute("""SELECT MAX(ITERATION) FROM POPULATION GROUP BY ID_RUN;""").fetchall()

            medians, means, maxes, mins, max_tests, all_heights, all_n_nodes = [], [], [], [], [], [], []

            cursor.execute("""SELECT N_INSTANCES FROM SETS WHERE RELATION_NAME = 'test';""")
            test_total = cursor.fetchone()[0]

            cursor.execute("""SELECT MAX_TREE_HEIGHT, MAX_N_NODES, DATASET_NAME FROM EVOLUTION;""")
            max_tree_height, max_n_nodes, dataset_name = cursor.fetchone()

            for iteration in n_iterations:
                cursor.execute("""SELECT FITNESS, TEST_CORRECT, HEIGHT, N_NODES
                                          FROM POPULATION
                                          WHERE ITERATION = %d
                                          ORDER BY FITNESS ASC;""" % iteration
                               )
                fitness, test_correct, tree_height, n_nodes = zip(*cursor.fetchall())
                medians += [np.median(fitness)]
                means += [np.mean(fitness)]
                maxes += [np.max(fitness)]
                mins += [np.min(fitness)]
                all_heights += [np.mean(tree_height) / float(max_tree_height)]
                all_n_nodes += [np.mean(n_nodes) / float(max_n_nodes)]
                max_tests += [test_correct[np.argmax(fitness)] / float(test_total)]

            cursor.close()

            plt.plot(medians, label='median fitness', c='green')
            plt.plot(means, label='mean fitness', c='orange')
            plt.plot(maxes, label='max fitness', c='blue')
            plt.plot(mins, label='min fitness', c='pink')
            plt.plot(max_tests, label='best individual\ntest accuracy', c='red')
            plt.plot(all_heights, label='mean height /\n  max height', c='cyan')
            plt.plot(all_n_nodes, label='mean nodes /\n  max nodes', c='magenta')

            plt.title("Population statistics throughout evolution\nfor dataset %s" % dataset_name)

            plt.xlabel('Iteration')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.show()

        elif mode == 'cross-validation':
            # TODO for each run!
            pass  # TODO show other type of data!

    def plot_prototype(self):
        pass


class MetaDataset(object):
    def __init__(self, full):
        self.n_objects, self.n_attributes = full.shape

        self.pred_attr = np.array(full.columns[:-1])  # type: np.ndarray
        self.target_attr = str(full.columns[-1])  # type: str
        self.class_labels = np.sort(full[full.columns[-1]].unique())  # type: np.ndarray

        self.numerical_class_labels = np.arange(len(self.class_labels), dtype=np.int32)  # type: np.ndarray
        self.class_label_index = {k: x for x, k in enumerate(self.class_labels)}  # type: dict
        self.inv_class_label_index = {x: k for x, k in enumerate(self.class_labels)}  # type: dict
        self.attribute_index = {k: x for x, k in enumerate(full.columns)}  # type: dict

        self.column_types = {x: self.raw_type_dict[str(full[x].dtype)] for x in full.columns}  # type: dict

    def to_categorical(self, y):
        """
        Adapted from https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py#L10
        Converts a class vector (integers) to binary class matrix.
        E.g. for use with categorical_crossentropy.
        # Arguments
            y: class vector to be converted into a matrix
                (integers from 0 to nb_classes).
            nb_classes: total number of classes.
        # Returns
            A binary matrix representation of the input.
        """
        nb_classes = self.numerical_class_labels.shape[0]

        y = np.array(y, dtype='int').ravel()
        if not nb_classes:
            nb_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, nb_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    def get_predictive_type(self, dtype):
        """
        Tells whether the attribute is categorical or numerical.

        :type dtype: type
        :param dtype: dtype of an attribute.
        :rtype: str
        :return: Whether this attribute is categorical or numerical.
        """

        raw_type = self.raw_type_dict[str(dtype)]
        mid = self.mid_type_dict[raw_type]
        return mid

    arff_type_dict = {
        'numeric': 'float',
        'real': 'float',
        'continuous': 'float',
    }

    raw_type_dict = {
        'int': 'int',
        'int_': 'int',
        'intc': 'int',
        'intp': 'int',
        'int8': 'int',
        'int16': 'int',
        'int32': 'int',
        'int64': 'int',
        'uint8': 'int',
        'uint16': 'int',
        'uint32': 'int',
        'uint64': 'int',
        'float': 'float',
        'float_': 'float',
        'float16': 'float',
        'float32': 'float',
        'float64': 'float',
        'complex_': 'complex',
        'complex64': 'complex',
        'complex128': 'complex',
        'object': 'object',
        'bool_': 'bool',
        'bool': 'bool',
        'str': 'str',
    }

    mid_type_dict = {
        'object': 'categorical',
        'str': 'categorical',
        'int': 'numerical',
        'float': 'numerical',
        'bool': 'categorical'
    }

    arff_data_types = {
        'date': 'numerical',
        'string': 'categorical',
        'integer': 'numerical',
        'numeric': 'numerical',
        'continuous': 'numerical',
        'real': 'numerical'
    }

    numerical = 'numerical'
    categorical = 'categorical'
