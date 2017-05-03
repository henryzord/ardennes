"""
Transforms a .csv dataset into a .arff dataset.
"""

import os
import sys
from collections import Counter

__author__ = 'Henry Cagnini'

def main():
    if len(sys.argv) == 1:
        print 'usage:'
        print '\tpython to_arff.py <csv_file_with_extension>'
        exit(0)

    full_path = sys.argv[1]
    path = '/'.join(full_path.split('/')[:-1])
    name = full_path.split('/')[-1].split('.')[0]

    classes = []

    # counts lines and columns
    with open(full_path, 'r') as f:
        line = f.readline()
        pred_attributes = line.count(',')

        classes += [line.split(',')[0]]
        for line in f:
            classes += [line.split(',')[0]]

    # counts classes
    classes = Counter(classes).keys()

    output_path = os.path.join(path, name + '.arff')

    # converts to arff
    with open(full_path, 'r') as r, open(output_path, 'w') as w:
        w.write('@relation %s\n\n' % name)
        w.write('@attribute class {%s}\n\n' % ','.join([k for k in classes]))

        for i in xrange(pred_attributes):
            w.write('@attribute %d NUMERIC\n' % i)

        w.write('\n@data\n\n')

        for line in r:
            w.write(line)

if __name__ == '__main__':
    main()
