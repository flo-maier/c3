import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import c3.experiment
import c3.optimizers.c1
import c3.libraries.fidelities as fid
import os
from c3.optimizers.c1 import C1
import c3.libraries.algorithms as algorithms
import c3.libraries.fidelities as fidelities
import examples.single_qubit_blackbox_exp


def rx_matrix_3(theta):
    return np.array([[np.cos(theta), -np.sin(theta)*1j, 0],
                     [-np.sin(theta)*1j, np.cos(theta), 0],
                     [0, 0, 1]])


def plot_dynamics(exp, psi_init, seq, goal=-1):
        """
        Plotting code for time-resolved populations.

        Parameters
        ----------
        psi_init: tf.Tensor
            Initial state or density matrix.
        seq: list
            List of operations to apply to the initial state.
        goal: tf.float64
            Value of the goal function, if used.
        debug: boolean
            If true, return a matplotlib figure instead of saving.
        """
        model = exp.pmap.model
        dUs = exp.dUs
        psi_t = psi_init.numpy()
        pop_t = exp.populations(psi_t, model.lindbladian)
        for gate in seq:
            for du in dUs[gate]:
                psi_t = np.matmul(du.numpy(), psi_t)
                pops = exp.populations(psi_t, model.lindbladian)
                pop_t = np.append(pop_t, pops, axis=1)

        fig, axs = plt.subplots(1, 1)
        ts = exp.ts
        dt = ts[1] - ts[0]
        ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
        axs.plot(ts / 1e-9, pop_t.T)
        axs.grid(linestyle="--")
        axs.tick_params(
            direction="in", left=True, right=True, top=True, bottom=True
        )
        axs.set_xlabel('Time [ns]')
        axs.set_ylabel('Population')
        plt.legend(model.state_labels)
        pass


def plot_fidelity(gate: str, channel: str, awg_errortype: str, error_values: np.array ):

    graph = {'errval': error_values, 'fidelity': []}
    fig, ax = plt.subplots()
    log_dir = os.path.join('C:\\c3logs')

    opt_gates = ["RX90p"]
    gateset_opt_map = [
        [
            ("RX90p", "d1", "gauss", "amp"),
        ]
        # ,
        # [
        #   ("RX90p", "d1", "gauss", "freq_offset"),
        # ],
        # [
        #   ("RX90p", "d1", "gauss", "xy_angle"),
        # ],
        # [
        #   ("RX90p", "d1", "gauss", "delta"),
        # ]
        # x90p d1 carrier framechange
    ]

    for errval in error_values:
        exp = examples.single_qubit_blackbox_exp.create_experiment()
        opt = C1(
            dir_path=log_dir,
            fid_func=fidelities.average_infid_set,
            fid_subspace=["Q1"],
            pmap=exp.pmap,
            algorithm=algorithms.lbfgs,
            options={"maxfun": 10},
            run_name="better_RX90"
        )
        exp.pmap.set_opt_map(gateset_opt_map)
        exp.set_opt_gates(opt_gates)
        opt.set_exp(exp)

        setattr(exp.pmap.generator.devices['AWG'], awg_errortype, errval)

        opt.optimize_controls()
        unitaries_after_opt = exp.get_gates()
        val = fid.unitary_infid(unitaries_after_opt, gate, [0], [3], False).numpy()
        graph['fidelity'].append(val)
        ax.scatter(errval, val)


    ax.plot(graph['errval'], graph['fidelity'])
    plt.legend()

    return graph
