# coding=utf-8

__author__ = 'Henry Cagnini'


class SetterClass(object):
    @classmethod
    def set_values(cls, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(cls, k, v)
