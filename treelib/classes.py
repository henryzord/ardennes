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


class AbstractTree(object):
    pred_attr = None
    target_attr = None
    class_labels = None
    
    def __init__(self, **kwargs):
        attrs = ['pred_attr', 'target_attr', 'class_labels']
        
        for k in attrs:
            if k in kwargs and getattr(self.__class__.__base__, k) is None:
                setattr(self.__class__.__base__, k, kwargs[k])
            else:
                setattr(self, k, getattr(self.__class__.__base__, k))

    def plot(self):
        pass


def type_check(var, val):
    """
    Checks the type of var (i.e, checks if var is in val). Raises an exception otherwise.
    """
    if type(var) not in val:
        raise TypeError('Variable %s must have one of the following types: %s' % (var, str(val)))


def value_check(var, val):
    """
    Checks the value of var (i.e, checks if var is in val). Raises an exception otherwise.
    """
    if var not in val:
        raise ValueError('Variable %s must have one of the following values: %s' % (var, str(val)))
