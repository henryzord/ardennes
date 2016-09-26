# coding=utf-8

__author__ = 'Henry Cagnini'


class SetterClass(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    @classmethod
    def set_class_values(cls, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(cls, k, v)


class Session(dict):
    pass
