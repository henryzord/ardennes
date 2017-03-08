# coding=utf-8

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

__author__ = 'Henry Cagnini'


def main():
    # Defining the network structure
    model = BayesianModel([('C', 'H'), ('P', 'H')])

    # H: host
    # P: prize
    # C: contestant

    # Defining the CPDs:
    cpd_c = TabularCPD('C', 3, [[0.33, 0.33, 0.33]])
    cpd_p = TabularCPD('P', 3, [[0.33, 0.33, 0.33]])
    cpd_h = TabularCPD('H', 3, [[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5],
                                [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
                                [0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]],
                       evidence=['C', 'P'], evidence_card=[3, 3])

    # Associating the CPDs with the network structure.
    model.add_cpds(cpd_c, cpd_p, cpd_h)

    # Some other methods
    # model.get_cpds()

    # check_model check for the model structure and the associated CPD and
    # returns True if everything is correct otherwise throws an exception
    # print model.check_model()

    # Infering the posterior probability
    infer = VariableElimination(model)
    posterior_p = infer.query(['H'], evidence={'C': 0, 'P': 0})
    print(posterior_p['H'])

if __name__ == '__main__':
    main()
