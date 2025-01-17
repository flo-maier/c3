"""
Experiment class that models and simulates the whole experiment.

It combines the information about the model of the quantum device, the control stack
and the operations that can be done on the device.

Given this information an experiment run is simulated, returning either processes,
states or populations.
"""

import os
import pickle
import itertools
import hjson
import numpy as np
import tensorflow as tf

from typing import Dict

from c3.generator.generator import Generator
from c3.parametermap import ParameterMap
from c3.signal.gates import Instruction
from c3.system.model import Model
from c3.utils import tf_utils

import matplotlib.pyplot as plt # flo
from operator import add # flo
from scipy.fft import fft, fftfreq # flo
from scipy.signal import blackman # flo
from scipy.signal import kaiser # flo
import c3.signal.pulse as pulse
import c3.libraries.envelopes as envelopes
import c3.signal.gates as gates
from c3.c3objs import Quantity as Qty


def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])


class Experiment:
    """
    It models all of the behaviour of the physical experiment, serving as a
    host for the individual parts making up the experiment.

    Parameters
    ----------
    model: Model
        The underlying physical device.
    generator: Generator
        The infrastructure for generating and sending control signals to the
        device.
    gateset: GateSet
        A gate level description of the operations implemented by control
        pulses.

    """

    def __init__(self, pmap: ParameterMap = None):
        self.pmap = pmap
        self.opt_gates = None
        self.unitaries: dict = {}
        self.dUs: dict = {}
        self.created_by = None
        self.logdir: str = None
        self.psi_init = None

    def set_created_by(self, config):
        """
        Store the config file location used to created this experiment.
        """

        self.created_by = config

    def quick_setup(self, filepath: str) -> None:
        """
        Load a quick setup file and create all necessary components.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read())

        model = Model()
        model.read_config(cfg["model"])
        gen = Generator()
        gen.read_config(cfg["generator"])

        single_gate_time = cfg["single_qubit_gate_time"]
        v2hz = cfg["v2hz"]
        instructions = []
        sideband = cfg.pop("sideband", None)
        for gate_name, props in cfg["single_qubit_gates"].items():
            target_qubit = model.subsystems[props["target_qubit"]]
            instr = Instruction(
                name=gate_name,
                t_start=0.0,
                t_end=single_gate_time,
                channels=[target_qubit.drive_line],
            )
            instr.quick_setup(
                target_qubit.drive_line,
                target_qubit.params["freq"].get_value() / 2 / np.pi,
                single_gate_time,
                v2hz,
                sideband,
            )
            instructions.append(instr)

        for gate_name, props in cfg["two_qubit_gates"].items():
            qubit_1 = model.subsystems[props["qubit_1"]]
            qubit_2 = model.subsystems[props["qubit_2"]]
            instr = Instruction(
                name=gate_name,
                t_start=0.0,
                t_end=props["gate_time"],
                channels=[qubit_1.drive_line, qubit_2.drive_line],
            )
            instr.quick_setup(
                qubit_1.drive_line,
                qubit_1.params["freq"].get_value() / 2 / np.pi,
                props["gate_time"],
                v2hz,
                sideband,
            )
            instr.quick_setup(
                qubit_2.drive_line,
                qubit_2.params["freq"].get_value() / 2 / np.pi,
                props["gate_time"],
                v2hz,
                sideband,
            )
            instructions.append(instr)

        self.pmap = ParameterMap(instructions, generator=gen, model=model)

    def read_config(self, filepath: str) -> None:
        """
        Load a file and parse it to create a Model object.

        Parameters
        ----------
        filepath : str
            Location of the configuration file

        """
        with open(filepath, "r") as cfg_file:
            cfg = hjson.loads(cfg_file.read())
        model = Model()
        model.fromdict(cfg["model"])
        generator = Generator()
        generator.fromdict(cfg["generator"])
        pmap = ParameterMap(model=model, generator=generator)
        pmap.fromdict(cfg["instructions"])
        self.pmap = pmap

    def write_config(self, filepath: str) -> None:
        """
        Write dictionary to a HJSON file.
        """
        with open(filepath, "w") as cfg_file:
            hjson.dump(self.asdict(), cfg_file)

    def asdict(self) -> dict:
        """
        Return a dictionary compatible with config files.
        """
        exp_dict: Dict[str, dict] = {}
        exp_dict["instructions"] = {}
        for name, instr in self.pmap.instructions.items():
            exp_dict["instructions"][name] = instr.asdict()
        exp_dict["model"] = self.pmap.model.asdict()
        exp_dict["generator"] = self.pmap.generator.asdict()
        return exp_dict

    def __str__(self) -> str:
        return hjson.dumps(self.asdict())

    def evaluate(self, seqs, psi_init):
        """
        Compute the population values for a given sequence of operations.

        Parameters
        ----------
        psi_init:
            Initial state vector
        seqs: str list
            A list of control pulses/gates to perform on the device.

        Returns
        -------
        list
            A list of populations

        """
        model = self.pmap.model
        Us = tf_utils.evaluate_sequences(self.unitaries, seqs)

        if psi_init is not None:
            self.psi_init = psi_init
        else:
            psi_init_ground = model.tasks["init_ground"].initialise(
                model.drift_H, model.lindbladian
            )
            self.psi_init = psi_init_ground

        populations = []
        for U in Us:
            psi_final = tf.matmul(U, self.psi_init)
            pops = self.populations(psi_final, model.lindbladian)
            populations.append(pops)
        return populations

    def process(self, populations, labels=None):
        """
        Apply a readout procedure to a population vector. Very specialized
        at the moment.

        Parameters
        ----------
        populations: list
            List of populations from evaluating.

        labels: list
            List of state labels specifying a subspace.

        Returns
        -------
        list
            A list of processed populations.

        """
        model = self.pmap.model
        populations_final = []
        populations_no_rescale = []
        for pops in populations:
            # TODO: Loop over all tasks in a general fashion
            # TODO: Selecting states by label in the case of computational space
            if "conf_matrix" in model.tasks:
                pops = model.tasks["conf_matrix"].confuse(pops)
                if labels is not None:
                    pops_select = 0
                    for label in labels:
                        pops_select += pops[model.comp_state_labels.index(label)]
                    pops = pops_select
                else:
                    pops = tf.reshape(pops, [pops.shape[0]])
            else:
                if labels is not None:
                    pops_select = 0
                    for label in labels:
                        try:
                            pops_select += pops[model.state_labels.index(label)]
                        except ValueError:
                            raise Exception(
                                f"C3:ERROR:State {label} not defined. Available are:\n"
                                f"{model.state_labels}"
                            )
                    pops = pops_select
                else:
                    pops = tf.reshape(pops, [pops.shape[0]])
            if "meas_rescale" in model.tasks:
                populations_no_rescale.append(pops)
                pops = model.tasks["meas_rescale"].rescale(pops)
            populations_final.append(pops)
        return populations_final, populations_no_rescale

    def get_perfect_gates(self, gate_keys: list = None) -> Dict[str, np.array]:
        """Return a perfect gateset for the gate_keys.

        Parameters
        ----------
        gate_keys: list
            (Optional) List of gates to evaluate.

        Returns
        -------
        Dict[str, np.array]
            A dictionary of gate names and np.array representation
            of the corresponding unitary

        Raises
        ------
        Exception
            Raise general exception for undefined gate
        """
        instructions = self.pmap.instructions
        gates = {}
        dims = self.pmap.model.dims
        if gate_keys is None:
            gate_keys = instructions.keys()  # type: ignore
        for gate in gate_keys:
            gates[gate] = perfect_gate(gates_str=gate, dims=dims)

        # TODO parametric gates

        return gates

    def get_gates(self):
        """
        Compute the unitary representation of operations. If no operations are
        specified in self.opt_gates the complete gateset is computed.

        Returns
        -------
        dict
            A dictionary of gate names and their unitary representation.
        """
        model = self.pmap.model
        generator = self.pmap.generator
        instructions = self.pmap.instructions
        gates = {}
        gate_keys = self.opt_gates
        if gate_keys is None:
            gate_keys = instructions.keys()

        for gate in gate_keys:
            try:
                instr = instructions[gate]
            except KeyError:
                raise Exception(
                    f"C3:Error: Gate '{gate}' is not defined."
                    f" Available gates are:\n {list(instructions.keys())}."
                )
            signal = generator.generate_signals(instr)
            U = self.propagation(signal, gate)
            if model.use_FR:
                # TODO change LO freq to at the level of a line
                freqs = {}
                framechanges = {}
                for line, ctrls in instr.comps.items():
                    # TODO calculate properly the average frequency that each qubit sees
                    offset = 0.0
                    for ctrl in ctrls.values():
                        if "freq_offset" in ctrl.params.keys():
                            if ctrl.params["amp"] != 0.0:
                                offset = ctrl.params["freq_offset"].get_value()
                    freqs[line] = tf.cast(
                        ctrls["carrier"].params["freq"].get_value() + offset,
                        tf.complex128,
                    )
                    framechanges[line] = tf.cast(
                        ctrls["carrier"].params["framechange"].get_value(),
                        tf.complex128,
                    )
                t_final = tf.constant(instr.t_end - instr.t_start, dtype=tf.complex128)
                FR = model.get_Frame_Rotation(t_final, freqs, framechanges)
                if model.lindbladian:
                    SFR = tf_utils.tf_super(FR)
                    U = tf.matmul(SFR, U)
                    self.FR = SFR
                else:
                    U = tf.matmul(FR, U)
                    self.FR = FR
            if model.dephasing_strength != 0.0:
                if not model.lindbladian:
                    raise ValueError("Dephasing can only be added when lindblad is on.")
                else:
                    amps = {}
                    for line, ctrls in instr.comps.items():
                        amp, sum = generator.devices["awg"].get_average_amp()
                        amps[line] = tf.cast(amp, tf.complex128)
                    t_final = tf.constant(
                        instr.t_end - instr.t_start, dtype=tf.complex128
                    )
                    dephasing_channel = model.get_dephasing_channel(t_final, amps)
                    U = tf.matmul(dephasing_channel, U)
            gates[gate] = U
            self.unitaries = gates
        return gates

    @staticmethod
    def __find_nearest(array, value):  # needed in plot_signal and measure_fft_peaks
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def plot_signal(self, gate_key: str, lofreq, sideband, channel: str = None):  # flo
        """

        """
        generator = self.pmap.generator
        instructions = self.pmap.instructions

        try:
            instr = instructions[gate_key]
        except KeyError:
            raise Exception(
                f"C3:Error: Gate '{gate_key}' is not defined."
                f" Available gates are:\n {list(instructions.keys())}."
            )

        signal: dict = generator.generate_signals(instr) #, True

        with plt.rc_context({'figure.facecolor': 'white'}):  # 'axes.edgecolor': 'orange', 'xtick.color': 'red', 'ytick.color': 'green',
            fig, axs = plt.subplots(2, 1, dpi=150)
            fig.set_figheight(10)
            fig.set_figwidth(10)

            # TODO messy with no channel
            if channel is None:
                for key in signal.keys():
                    x = signal[key]['ts']
                    y = signal[key]['values']
                    axs[0].plot(x / 1e-9, y)
                plt.legend(list(map(add, ['channel ']*len(signal.keys()), list(signal.keys()))))

            else:
                x = signal[channel]['ts']
                y = signal[channel]['values']
                line1 = axs[0].plot(x / 1e-9, y / 1e-9, label='channel ' + channel)
                iq_signal = generator.devices['AWG'].process(instr, channel)
                ax2 = axs[0].twinx()
                line2 = ax2.plot(iq_signal['ts'] / 1e-9, iq_signal['inphase'], label='inphase')
                line3 = ax2.plot(iq_signal['ts'] / 1e-9, iq_signal['quadrature'], label='quadrature')
                align_yaxis(axs[0], ax2)
                y2 = 0.5*np.cos(2*np.pi*53e6*x)
                #ax2.plot(x / 1e-9, y2, '--')
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                axs[0].legend(lines, labels, loc=1)
                ax2.set_ylabel('Amplitude I & Q [V]')

            for ax in axs:
                ax.grid(linestyle="--")
                ax.tick_params(
                    direction="in", left=True, right=True, top=True, bottom=True
                )
            axs[0].set_xlabel('Time [ns]')
            axs[0].set_ylabel('Amplitude Signal [V]')

            if channel is not None:
                N = len(signal[channel]['values'])
                dt = signal[channel]['ts'][1]-signal[channel]['ts'][0]
                signalf = fft(signal[channel]['values'].numpy())
                w = kaiser(N, beta=14)
                signalwf = fft(signal[channel]['values'].numpy()*w)
                xf = fftfreq(N, dt)[:N//2]
                axs[1].semilogy(xf[1:N // 2]/1e9, 2.0 / N * np.abs(signalf[1:N // 2]), '-b', label='FFT')
                axs[1].semilogy(xf[1:N // 2]/1e9, 2.0 / N * np.abs(signalwf[1:N // 2]), '-r', label='FFT w/ window')
                idx = self.__find_nearest(xf, (lofreq+sideband))
                #axs[1].scatter((lofreq+sideband)/1e9, 2.0 / N * np.abs(signalwf[idx]),s=20,c='r')
                axs[1].axvline(lofreq/1e9, 1e-2, 1e1)
                axs[1].axvline((lofreq+sideband)/1e9, 1e-2, 1e1)
                axs[1].axvline((lofreq-sideband)/1e9, 1e-2, 1e1)
                axs[1].set_xlim([4.5, 5.5])
                axs[1].set_xlabel('Frequency [GHz]')
                axs[1].set_ylabel('Amplitude [V*1e9 Hz/V]')
                axs[1].legend()

        return fig, axs

    def measure_fft_peaks(self, gate_key: str, lofreq, sideband, channel: str, awg_errortype: str, error_values: np.array) -> dict:

        generator = self.pmap.generator
        instructions = self.pmap.instructions

        try:
            instr = instructions[gate_key]
        except KeyError:
            raise Exception(
                f"C3:Error: Gate '{gate_key}' is not defined."
                f" Available gates are:\n {list(instructions.keys())}."
            )

        peaks = dict()
        key = str()

        for errval in error_values:

            print(f'Calculating with {awg_errortype}: {errval}')
            setattr(self.pmap.generator.devices['AWG'], awg_errortype, errval)
            signal: dict = generator.generate_signals(instr)
            N = len(signal[channel]['values'])
            dt = signal[channel]['ts'][1] - signal[channel]['ts'][0]
            w = kaiser(N, beta=14)

            signalwf = fft(signal[channel]['values'].numpy() * w)
            xf = fftfreq(N, dt)[:N // 2]

            for i in np.arange(-3, 4):
                idx = self.__find_nearest(xf, (lofreq + i*sideband))
                if i < 0:
                    key = "{}LSB".format(abs(i))
                elif i > 0:
                    key = "{}RSB".format(i)
                else:
                    key = "LO"
                try:
                    peaks[key].append(2.0 / N * np.abs(signalwf[idx]))
                except KeyError:
                    peaks[key] = [2.0 / N * np.abs(signalwf[idx])]

        peaks[awg_errortype] = error_values

        return peaks

    def get_signal_values(self, gate_key: str, channel: str):#flo
        """
        Returns
        -------
        dict
            A dictionary
        """
        generator = self.pmap.generator
        instructions = self.pmap.instructions

        try:
            instr = instructions[gate_key]
        except KeyError:
            raise Exception(
                f"C3:Error: Gate '{gate_key}' is not defined."
                f" Available gates are:\n {list(instructions.keys())}."
            )
        signal: dict = generator.generate_signals(instr)
        iq_signal = generator.devices['AWG'].process(instr, channel)

        return {"rf": signal, "iq": iq_signal}

    def propagation(self, signal: dict, gate):
        """
        Solve the equation of motion (Lindblad or Schrödinger) for a given control
        signal and Hamiltonians.

        Parameters
        ----------
        signal: dict
            Waveform of the control signal per drive line.
        ts: tf.float64
            Vector of times.
        gate: str
            Identifier for one of the gates.

        Returns
        -------
        unitary
            Matrix representation of the gate.
        """
        model = self.pmap.model
        h0, hctrls = model.get_Hamiltonians()
        signals = []
        hks = []
        for key in signal:
            signals.append(signal[key]["values"])
            ts = signal[key]["ts"]
            hks.append(hctrls[key])
        dt = tf.constant(ts[1].numpy() - ts[0].numpy(), dtype=tf.complex128)

        if model.lindbladian:
            col_ops = model.get_Lindbladians()
            dUs = tf_utils.tf_propagation_lind(h0, hks, col_ops, signals, dt)
        else:
            dUs = tf_utils.tf_propagation_vectorized(h0, hks, signals, dt)
        self.dUs[gate] = dUs
        self.ts = ts
        dUs = tf.cast(dUs, tf.complex128)
        U = tf_utils.tf_matmul_left(dUs)
        self.U = U
        return U

    def set_opt_gates(self, gates):
        """
        Specify a selection of gates to be computed.

        Parameters
        ----------
        opt_gates: Identifiers of the gates of interest. Can contain duplicates.

        """
        self.opt_gates = gates

    def change_pi_amp(self, amp):

        t_final = self.pmap.instructions['Id'].t_end
        sideband = 50e6

        gauss_params_pi = {
            "amp": Qty(value=amp, min_val=amp*0.9, max_val=amp*1.1, unit="V"),
            "t_final": Qty(
                value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
            ),
            "sigma": Qty(
                value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"
            ),
            "xy_angle": Qty(
                value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
            ),
            "freq_offset": Qty(
                value=-sideband - 0.5e6,
                min_val=-60 * 1e6,
                max_val=-40 * 1e6,
                unit="Hz 2pi",
            ),
            "delta": Qty(value=-1, min_val=-5, max_val=3, unit=""),
        }

        gauss_env_pi = pulse.Envelope(
            name="gauss",
            desc="Gaussian comp for single-qubit gates",
            params=gauss_params_pi,
            shape=envelopes.gaussian_nonorm,
        )

        PI01p = gates.Instruction(name="PI01p", t_start=0.0, t_end=t_final, channels=["d1"])
        PI01p.add_component(gauss_env_pi, "d1")
        PI01p.add_component(self.pmap.instructions['PI01p'].comps['d1']['carrier'], "d1")  #reuse carrier

        self.pmap.instructions['PI01p'] = PI01p

        return PI01p

    def change_pi12_amp(self, amp):

        t_final = self.pmap.instructions['Id'].t_end
        sideband = 50e6

        gauss_params_pi = {
            "amp": Qty(value=amp, min_val=amp*0.9, max_val=amp*1.1, unit="V"),
            "t_final": Qty(
                value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
            ),
            "sigma": Qty(
                value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"
            ),
            "xy_angle": Qty(
                value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
            ),
            "freq_offset": Qty(
                value=-sideband - 0.5e6,
                min_val=-60 * 1e6,
                max_val=-40 * 1e6,
                unit="Hz 2pi",
            ),
            "delta": Qty(value=-1, min_val=-5, max_val=3, unit=""),
        }

        gauss_env_pi = pulse.Envelope(
            name="gauss",
            desc="Gaussian comp for single-qubit gates",
            params=gauss_params_pi,
            shape=envelopes.gaussian_nonorm,
        )

        RXp12 = gates.Instruction(name="RXp12", t_start=0.0, t_end=t_final, channels=["d1"])
        RXp12.add_component(gauss_env_pi, "d1")
        RXp12.add_component(self.pmap.instructions['RXp12'].comps['d1']['carrier'], "d1")  #reuse carrier

        self.pmap.instructions['RXp12'] = RXp12

        return RXp12

    def set_opt_gates_seq(self, seqs):
        """
        Specify a selection of gates to be computed.

        Parameters
        ----------
        opt_gates: Identifiers of the gates of interest. Can contain duplicates.

        """
        self.opt_gates = list(set(itertools.chain.from_iterable(seqs)))

    def set_enable_store_unitaries(self, flag, logdir, exist_ok=False):
        """
        Saving of unitary propagators.

        Parameters
        ----------
        flag: boolean
            Enable or disable saving.
        logdir: str
            File path location for the resulting unitaries.
        """
        self.enable_store_unitaries = flag
        self.logdir = logdir
        if self.enable_store_unitaries:
            os.makedirs(self.logdir + "unitaries/", exist_ok=exist_ok)
            self.store_unitaries_counter = 0

    def store_Udict(self, goal):
        """
        Save unitary as text and pickle.

        Parameter
        ---------
        goal: tf.float64
            Value of the goal function, if used.

        """
        folder = (
            self.logdir
            + "unitaries/eval_"
            + str(self.store_unitaries_counter)
            + "_"
            + str(goal)
            + "/"
        )
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(folder + "Us.pickle", "wb+") as file:
            pickle.dump(self.unitaries, file)
        for key, value in self.unitaries.items():
            # Windows is not able to parse ":" as file path
            np.savetxt(folder + key.replace(":", ".") + ".txt", value)

    def populations(self, state, lindbladian):
        """
        Compute populations from a state or density vector.

        Parameters
        ----------
        state: tf.Tensor
            State or densitiy vector.
        lindbladian: boolean
            Specify if conversion to density matrix is needed.

        Returns
        -------
        tf.Tensor
            Vector of populations.
        """
        if lindbladian:
            rho = tf_utils.tf_vec_to_dm(state)
            pops = tf.math.real(tf.linalg.diag_part(rho))
            return tf.reshape(pops, shape=[pops.shape[0], 1])
        else:
            return tf.abs(state) ** 2

    def expect_oper(self, state, lindbladian, oper):
        if lindbladian:
            rho = tf_utils.tf_vec_to_dm(state)
        else:
            rho = tf_utils.tf_state_to_dm(state)
        trace = np.trace(np.matmul(rho, oper))
        return [[np.real(trace)]]  # ,[np.imag(trace)]]
