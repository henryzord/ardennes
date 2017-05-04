# coding=utf-8

import numpy as np
import sqlite3

from treelib import get_total_nodes

__author__ = 'Henry Cagnini'


# noinspection SqlNoDataSourceInspection
# noinspection SqlDialectInspection
class DatabaseHandler(object):

    def __init__(self, path, tree_height):
        self.conn = None
        cursor = None
        try:
            # TODO create table for dataset! fulfill with metadataset data!

            self.conn = sqlite3.connect(path)
            cursor = self.conn.cursor()

            cursor.execute("""
              CREATE TABLE POPULATION (
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
              CREATE TABLE PROTOTYPE (
                iteration INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                %s
                PRIMARY KEY (fold, iteration)
               )
             """ % _prototype_columns)

            self.closed = False
        except:
            if self.conn is not None:
                self.conn.close()
            self.closed = True
        finally:
            if cursor is not None:
                cursor.close()

    def write_population(self, fold, iteration, population, has_val=False, has_test=False):
        cursor = None
        try:
            cursor = self.conn.cursor()

            for ind in population:
                cursor.execute(
                    """INSERT INTO POPULATION VALUES (%d, %d, %d, %f, %d, %d, %d, %s, %s, '%s')""" % (
                        ind.ind_id, iteration, fold, ind.fitness, ind.height, ind.n_nodes,
                        int(ind.train_acc_score * len(ind.y_train_true)),
                        str(int(ind.val_acc_score * len(ind.y_val_true))) if has_val else 'NULL',
                        str(int(ind.test_acc_score) * len(ind.y_test_true)) if has_test else 'NULL',
                        ind.to_dot()
                    )
                )
        except:
            pass
        finally:
            if cursor is not None:
                cursor.close()

        # cPickle.dump(best_individual, open(evo_file.split('.')[0].strip() + '.bin', 'w'))

    def write_prototype(self, fold, iteration, gm):
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
            """ % (fold, iteration, ','.join([gm.attributes.values.ravel()]))
            )
            cursor.close()
        except:
            pass
        finally:
            if cursor is not None:
                cursor.close()

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
