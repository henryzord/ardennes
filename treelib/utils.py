# coding=utf-8

import numpy as np
import sqlite3

from treelib import get_total_nodes
from matplotlib import pyplot as plt

__author__ = 'Henry Cagnini'


# noinspection SqlNoDataSourceInspection
# noinspection SqlDialectInspection
class DatabaseHandler(object):

    def __init__(self, path, **kwargs):
        cursor = None
        self.conn = None

        self.train_hash = None
        self.test_hash = None
        self.val_hash = None
        self.full_hash = None

        n_individuals = kwargs['n_individuals']
        n_iterations = kwargs['n_iterations']
        tree_height = kwargs['tree_height']
        decile = kwargs['decile']
        random_state = kwargs['random_state']
        dataset_name = kwargs['dataset_name']

        max_n_nodes = get_total_nodes(tree_height - 1)  # since the probability of generating the class at D is 100%

        try:
            self.conn = sqlite3.connect(path)
            cursor = self.conn.cursor()

            cursor.execute("""
              CREATE TABLE IF NOT EXISTS EVOLUTION (
                dataset_name TEXT NOT NULL PRIMARY KEY,
                n_individuals INTEGER NOT NULL,
                n_iterations INTEGER NOT NULL,
                max_tree_height INTEGER NOT NULL,
                max_n_nodes INTEGER NOT NULL,
                decile REAL NOT NULL,
                random_state REAL
              )""")

            cursor.execute("""SELECT COUNT(*) FROM EVOLUTION""")
            count = cursor.fetchone()[0]
            if count == 0:
                cursor.execute("""
                  INSERT INTO EVOLUTION VALUES ('%s', %d, %d, %d, %d, %f, %s)""" % (
                        dataset_name, n_individuals, n_iterations, tree_height, max_n_nodes, decile,
                        random_state if random_state is not None else '\'NULL\''
                    )
                )

            cursor.execute("""
              CREATE TABLE IF NOT EXISTS SETS (
                relation_name TEXT NOT NULL PRIMARY KEY,
                fold INTEGER NOT NULL,
                n_instances INTEGER NOT NULL,
                n_attributes INTEGER NOT NULL,
                n_classes INTEGER NOT NULL
              )
            """)

            cursor.execute("""
              CREATE TABLE IF NOT EXISTS POPULATION (
                individual INTEGER NOT NULL,
                iteration INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                fitness REAL NOT NULL,
                height INTEGER NOT NULL,
                n_nodes INTEGER NOT NULL,
                train_correct INTEGER NOT NULL,
                val_correct INTEGER DEFAULT NULL,
                test_correct INTEGER DEFAULT NULL,
                dot TEXT DEFAULT NULL,
                PRIMARY KEY (fold, iteration, individual)
              )
            """)

            n_variables = get_total_nodes(tree_height - 2)  # since the probability of generating the class at D is 100%

            _prototype_columns = '\n'.join(['NODE_%d REAL NOT NULL,' % i for i in xrange(n_variables)])

            cursor.execute("""
              CREATE TABLE IF NOT EXISTS PROTOTYPE (
                iteration INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                %s
                PRIMARY KEY (fold, iteration)
               )
             """ % _prototype_columns)

            self.closed = False
        except Exception, e:
            if self.conn is not None:
                self.conn.close()
            self.closed = True
        finally:
            if cursor is not None:
                cursor.close()

    def write_sets(self, data):
        """

        :type data: list
        :param data:
        :return:
        """

        cursor = None
        try:
            cursor = self.conn.cursor()
            for d in data:  # type: dict
                exec('self.%s_hash = %d' % (d['relation_name'], d['fold']))

                cursor.execute("""INSERT INTO SETS VALUES(
                  '%s', %d, %d, %d, %d
                  )""" % (d['relation_name'], d['fold'], d['n_instances'], d['n_attributes'], d['n_classes'])
                )
        except:
            pass
        finally:
            if cursor is not None:
                cursor.close()

    def write_population(self, iteration, population):
        cursor = None
        try:
            cursor = self.conn.cursor()

            for ind in population:
                cursor.execute(
                    """INSERT INTO POPULATION VALUES (%d, %d, %d, %f, %d, %d, %d, %s, %s, '%s')""" % (
                        ind.ind_id, iteration, self.train_hash, ind.fitness, ind.height, ind.n_nodes,
                        int(ind.train_acc_score * len(ind.y_train_true)),
                        str(int(ind.val_acc_score * len(ind.y_val_true))) if self.val_hash is not None else 'NULL',
                        str(int(ind.test_acc_score * len(ind.y_test_true))) if self.test_hash is not None else 'NULL',
                        ind.to_dot()
                    )
                )
        except:
            pass
        finally:
            if cursor is not None:
                cursor.close()

    def write_prototype(self, iteration, gm):
        """

        :type fold: int
        :param fold:
        :type iteration: int
        :param iteration:
        :type gm: treelib.graphical_model.GraphicalModel
        :param gm:
        :return:
        """
        cursor = None
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
               INSERT INTO PROTOTYPE VALUES (
                %d, %d, %s
               )
            """ % (self.train_hash, iteration, ','.join([gm.attributes.values.ravel()]))
            )
            cursor.close()
        except:
            pass
        finally:
            if cursor is not None:
                cursor.close()

    def plot_population(self):
        plt.figure()
        ax = plt.subplot(111)

        cursor = self.conn.cursor()

        cursor.execute("""SELECT DISTINCT(ITERATION) FROM POPULATION ORDER BY ITERATION ASC;""")
        n_iterations = cursor.fetchall()

        medians = []
        means = []
        maxes = []
        mins = []
        max_tests = []
        all_heights = []
        all_n_nodes = []

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

    def plot_prototype(self):
        pass

    def close(self):
        try:
            if self.conn is not None and not self.closed:
                self.conn.commit()
                self.conn.close()
        except:
            pass
        finally:
            self.closed = True


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
