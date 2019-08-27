from test_generator import *
from test_model import *

from c3po.optimizer.optimizer import Optimizer as Opt
from c3po.simulation.simulator import Simulator as Sim

rechenknecht = Opt()

opt_map = {
    'amp' : [
        (ctrl.get_uuid(), p1.get_uuid()),
        (ctrl.get_uuid(), p2.get_uuid())
        ],
    'T_up' : [
        (ctrl.get_uuid(), p1.get_uuid()),
        (ctrl.get_uuid(), p2.get_uuid())
        ],
    'T_down' : [
        (ctrl.get_uuid(), p1.get_uuid()),
        (ctrl.get_uuid(), p2.get_uuid())
        ],
    'xy_angle': [(ctrl.get_uuid(), p2.get_uuid())],
    'freq_offset': [(ctrl.get_uuid(), p1.get_uuid())]
}

sim = Sim(initial_model, gen, ctrls)


# Goal to drive on qubit 1
indx = initial_model.names.index('Q1')
a_q1 = initial_model.ann_opers[indx]
U_goal = a_q1 + a_q1.dag()

def evaluate_signals(params, opt_params):
    U = sim.propagation(params, opt_params)
    return 1-tf_unitary_overlap(U, U_goal.full())

opt_settings = {}
rechenknecht.optimize_controls(
    controls = ctrls,
    opt_map = opt_map,
    opt = 'open_loop',
    settings = opt_settings,
    calib_name = 'test',
    eval_func = evaluate_signals
    )
