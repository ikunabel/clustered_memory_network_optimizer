import spikelog
from typing import DefaultDict
import numpy as np
from numpy.core.function_base import linspace
import matplotlib.pyplot as plt
import json
import pickle
import os
import abc
import re
import time
import capacities as cap
from dataclasses import dataclass


class SpikeTrain(object):
    """
    SpikeTrain(spikes_times, t_start=None, t_stop=None)
    This class defines a spike train as a list of times events.

    Event times are given in a list (sparse representation) in milliseconds.

    Inputs:
        spike_times - a list/numpy array of spike times (in milliseconds)
        t_start     - beginning of the SpikeTrain (if not, this is inferred)
        t_stop      - end of the SpikeTrain (if not, this is inferred)

    Examples:
        >> s1 = SpikeTrain([0.0, 0.1, 0.2, 0.5])
        >> s1.isi()
            array([ 0.1,  0.1,  0.3])
        >> s1.mean_rate()
            8.0
        >> s1.cv_isi()
            0.565685424949
    """

    def __init__(self, spike_times, t_start=None, t_stop=None):
        """
        Constructor of the SpikeTrain object
        """

        self.t_start = t_start
        self.t_stop = t_stop
        self.spike_times = np.array(spike_times, np.float32)

        # If t_start is not None, we resize the spike_train keeping only
        # the spikes with t >= t_start
        if self.t_start is not None:
            self.spike_times = np.extract(
                (self.spike_times >= self.t_start), self.spike_times
            )

        # If t_stop is not None, we resize the spike_train keeping only
        # the spikes with t <= t_stop
        if self.t_stop is not None:
            self.spike_times = np.extract(
                (self.spike_times <= self.t_stop), self.spike_times
            )

        # We sort the spike_times. May be slower, but is necessary for quite a
        # lot of methods
        self.spike_times = np.sort(self.spike_times, kind="quicksort")
        # Here we deal with the t_start and t_stop values if the SpikeTrain
        # is empty, with only one element or several elements, if we
        # need to guess t_start and t_stop
        # no element : t_start = 0, t_stop = 0.1
        # 1 element  : t_start = time, t_stop = time + 0.1
        # several    : t_start = min(time), t_stop = max(time)

        size = len(self.spike_times)
        if size == 0:
            if self.t_start is None:
                self.t_start = 0
            if self.t_stop is None:
                self.t_stop = 0.1
        elif size == 1:  # spike list may be empty
            if self.t_start is None:
                self.t_start = self.spike_times[0]
            if self.t_stop is None:
                self.t_stop = self.spike_times[0] + 0.1
        elif size > 1:
            if self.t_start is None:
                self.t_start = np.min(self.spike_times)
            if np.any(self.spike_times < self.t_start):
                raise ValueError("Spike times must not be less than t_start")
            if self.t_stop is None:
                self.t_stop = np.max(self.spike_times)
            if np.any(self.spike_times > self.t_stop):
                raise ValueError("Spike times must not be greater than t_stop")

        if self.t_start >= self.t_stop:
            raise Exception(
                "Incompatible time interval : t_start = %s, t_stop = %s"
                % (self.t_start, self.t_stop)
            )
        if self.t_start < 0:
            raise ValueError("t_start must not be negative")
        if np.any(self.spike_times < 0):
            raise ValueError("Spike times must not be negative")

    def format(self, relative=False, quantized=False):
        """
        Return an array with a new representation of the spike times

        Inputs:
            relative  - if True, spike times are expressed in a relative
                       time compared to the previous one
            quantized - a value to divide spike times with before rounding

        Examples:
            >> st.spikes_times=[0, 2.1, 3.1, 4.4]
            >> st.format(relative=True)
                [0, 2.1, 1, 1.3]
            >> st.format(quantized=2)
                [0, 1, 2, 2]
        """
        spike_times = self.spike_times.copy()

        if relative and len(spike_times) > 0:
            spike_times[1:] = spike_times[1:] - spike_times[:-1]

        if quantized:
            assert (
                quantized > 0
            ), "quantized must either be False or a positive number"
            spike_times = (spike_times / quantized).round().astype("int")

        return spike_times

    def time_axis(self, time_bin=10):
        """
        Return a time axis between t_start and t_stop according to a time_bin

        Inputs:
            time_bin - the bin width

        Examples:
            >> st = SpikeTrain(range(100),0.1,0,100)
            >> st.time_axis(10)
                [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        See also
            time_histogram
        """
        axis = np.arange(self.t_start, self.t_stop + time_bin, time_bin)
        return axis

    def time_histogram(self, time_bin=10, normalized=True, binary=False):
        """
        Bin the spikes with the specified bin width. The first and last bins
        are calculated from `self.t_start` and `self.t_stop`.

        Inputs:
            time_bin   - the bin width for gathering spikes_times
            normalized - if True, the bin values are scaled to represent firing rates
                         in spikes/second, otherwise otherwise it's the number of spikes
                         per bin.
            binary     - if True, a binary matrix of 0/1 is returned

        Examples:
            >> st=SpikeTrain(range(0,100,5),0.1,0,100)
            >> st.time_histogram(10)
                [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
            >> st.time_histogram(10, normalized=False)
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        See also
            time_axis
        """
        bins = self.time_axis(time_bin)
        hist, edges = np.histogram(self.spike_times, bins)
        hist = hist.astype(float)
        if normalized:  # what about normalization if time_bin is a sequence?
            hist *= 1000.0 / float(time_bin)
        if binary:
            hist = hist.astype(bool).astype(int)
        return hist

    def spike_counts(self, dt, normalized=False, binary=False):
        """
        Returns array with all single neuron spike counts
        :param dt:
        :param normalized:
        :param binary:
        :return:
        """
        counts = [
            self.spiketrains[v].time_histogram(
                time_bin=dt, normalized=normalized, binary=binary
            )
            for v in self.id_list
        ]
        return np.array(counts)

    #######################################################################
    ## Analysis methods that can be applied to a SpikeTrain object       ##
    #######################################################################

    def mean_rate(self, t_start=None, t_stop=None):
        """
        Returns the mean firing rate between t_start and t_stop, in spikes/sec

        Inputs:
            t_start - in ms. If not defined, the one of the SpikeTrain object is used
            t_stop  - in ms. If not defined, the one of the SpikeTrain object is used

        Examples:
            >> spk.mean_rate()
                34.2
        """
        if (t_start is None) & (t_stop is None):
            t_start = self.t_start
            t_stop = self.t_stop
            idx = self.spike_times
        else:
            if t_start is None:
                t_start = self.t_start
            else:
                t_start = max(self.t_start, t_start)
            if t_stop is None:
                t_stop = self.t_stop
            else:
                t_stop = min(self.t_stop, t_stop)
            idx = np.where(
                (self.spike_times >= t_start) & (self.spike_times <= t_stop)
            )[0]
        return 1000.0 * len(idx) / (t_stop - t_start)

    def isi(self):
        """
        Return an array with the inter-spike intervals of the SpikeTrain

        Examples:
            >> st.spikes_times=[0, 2.1, 3.1, 4.4]
            >> st.isi()
                [2.1, 1., 1.3]

        See also
            cv_isi
        """
        return np.diff(self.spike_times)

    def mean_rate(self, t_start=None, t_stop=None):
        """
        Returns the mean firing rate between t_start and t_stop, in spikes/sec

        Inputs:
            t_start - in ms. If not defined, the one of the SpikeTrain object is used
            t_stop  - in ms. If not defined, the one of the SpikeTrain object is used

        Examples:
            >> spk.mean_rate()
                34.2
        """
        if (t_start is None) & (t_stop is None):
            t_start = self.t_start
            t_stop = self.t_stop
            idx = self.spike_times
        else:
            if t_start is None:
                t_start = self.t_start
            else:
                t_start = max(self.t_start, t_start)
            if t_stop is None:
                t_stop = self.t_stop
            else:
                t_stop = min(self.t_stop, t_stop)
            idx = np.where(
                (self.spike_times >= t_start) & (self.spike_times <= t_stop)
            )[0]
        return 1000.0 * len(idx) / (t_stop - t_start)

    def cv_isi(self):
        """
        Return the coefficient of variation of the isis.

        cv_isi is the ratio between the standard deviation and the mean of the ISI
          The irregularity of individual spike trains is measured by the squared
        coefficient of variation of the corresponding inter-spike interval (ISI)
        distribution normalized by the square of its mean.
          In point processes, low values reflect more regular spiking, a
        clock-like pattern yields CV2= 0. On the other hand, CV2 = 1 indicates
        Poisson-type behavior. As a measure for irregularity in the network one
        can use the average irregularity across all neurons.

        http://en.wikipedia.org/wiki/Coefficient_of_variation

        See also
            isi, cv_kl

        """
        isi = self.isi()
        if len(isi) > 1:
            return np.std(isi) / np.mean(isi)
        else:
            return np.nan


class SpikeList(object):
    """
    SpikeList(spikes, id_list, t_start=None, t_stop=None, dims=None)

    Return a SpikeList object which will be a list of SpikeTrain objects.

    Inputs:
        spikes  - a list of (id,time) tuples (id being in id_list)
        id_list - the list of the ids of all recorded cells (needed for silent cells)
        t_start - begining of the SpikeList, in ms. If None, will be infered from the data
        t_stop  - end of the SpikeList, in ms. If None, will be infered from the data
        dims    - dimensions of the recorded population, if not 1D population

    t_start and t_stop are shared for all SpikeTrains object within the SpikeList

    Examples:
        >> sl = SpikeList([(0, 0.1), (1, 0.1), (0, 0.2)], range(2))
        >> type( sl[0] )
            <type SpikeTrain>

    See also
        load_spikelist
    """

    def __init__(self, spikes, id_list, t_start=None, t_stop=None, dims=None):
        self.t_start = t_start
        self.t_stop = t_stop
        self.dimensions = dims
        self.spiketrains = {}
        id_list = np.sort(id_list)

        # set dimension explicitly if needed
        if self.dimensions is None:
            self.dimensions = len(id_list)

        if not isinstance(spikes, np.ndarray):
            spikes = np.array(spikes, np.float32)
            # circumvents numpy floating point magic that leads to filtering errors..
        N = len(spikes)
        if N > 0:
            idx = np.argsort(spikes[:, 0])
            spikes = spikes[idx]
            break_points = np.where(np.diff(spikes[:, 0]) > 0)[0] + 1
            break_points = np.concatenate(([0], break_points))
            break_points = np.concatenate((break_points, [N]))
            for idx in range(len(break_points) - 1):
                id = spikes[break_points[idx], 0]
                if id in id_list:
                    self.spiketrains[id] = SpikeTrain(
                        spikes[break_points[idx] : break_points[idx + 1], 1],
                        self.t_start,
                        self.t_stop,
                    )

        if len(self.spiketrains) > 0 and (
            self.t_start is None or self.t_stop is None
        ):
            self.__calc_startstop()
            del spikes

    def __del__(self):
        for id in self.id_list:
            del self.spiketrains[id]

    @property
    def id_list(self):
        """
        Return the list of all the cells ids contained in the
        SpikeList object

        Examples
            >> spklist.id_list
            [0,1,2,3,....,9999]
        """
        return np.array(list(self.spiketrains.keys()))

    def __calc_startstop(self, t_start=None, t_stop=None):
        """
        t_start and t_stop are shared for all neurons, so we take min and max values respectively.
        """
        if len(self) > 0:
            if t_start is not None:
                self.t_start = t_start
                for id in self.spiketrains.keys():
                    self.spiketrains[id].t_start = t_start

            elif self.t_start is None:
                start_times = np.array(
                    [self.spiketrains[idx].t_start for idx in self.id_list],
                    np.float32,
                )
                self.t_start = np.min(start_times)
                for id in self.spiketrains.keys():
                    self.spiketrains[id].t_start = self.t_start
            if t_stop is not None:
                self.t_stop = t_stop
                for id in self.spiketrains.keys():
                    self.spiketrains[id].t_stop = t_stop
            elif self.t_stop is None:
                stop_times = np.array(
                    [self.spiketrains[idx].t_stop for idx in self.id_list],
                    np.float32,
                )
                self.t_stop = np.max(stop_times)
                for id in self.spiketrains.keys():
                    self.spiketrains[id].t_stop = self.t_stop
        else:
            raise Exception("No SpikeTrains")

    def __getitem__(self, id):
        if id in self.id_list:
            return self.spiketrains[id]
        else:
            raise Exception(
                "id %d is not present in the SpikeList. See id_list" % id
            )

    def __getslice__(self, i, j):
        """
        Return a new SpikeList object with all the ids between i and j
        """
        ids = np.where((self.id_list >= i) & (self.id_list < j))[0]
        return self.id_slice(ids)

    def __setitem__(self, id, spktrain):
        assert isinstance(
            spktrain, SpikeTrain
        ), "A SpikeList object can only contain SpikeTrain objects"
        self.spiketrains[id] = spktrain
        self.__calc_startstop()

    def __iter__(self):
        return self.spiketrains.itervalues()

    def __len__(self):
        return len(self.spiketrains)

    def __sub_id_list(self, sub_list=None):
        """
        Internal function used to get a sublist for the Spikelist id list

        Inputs:
            sublist - can be an int (and then N random cells are selected). Otherwise
                    sub_list is a list of cell in self.id_list. If None, id_list is returned

        Examples:
            >> self.__sub_id_list(50)
        """
        if sub_list is None:
            return self.id_list
        elif type(sub_list) == int:
            return np.random.permutation(self.id_list)[0:sub_list]
        else:
            return sub_list

    def __select_with_pairs__(self, nb_pairs, pairs_generator):
        """
        Internal function used to slice two SpikeList according to a list
        of pairs.  Return a list of pairs

        Inputs:
            nb_pairs        - an int specifying the number of cells desired
            pairs_generator - a pairs generator

        Examples:
            >> self.__select_with_pairs__(50, RandomPairs(spk1, spk2))

        See also
            RandomPairs, AutoPairs, CustomPairs
        """
        pairs = pairs_generator.get_pairs(nb_pairs)
        spk1 = pairs_generator.spk1.id_slice(pairs[:, 0])
        spk2 = pairs_generator.spk2.id_slice(pairs[:, 1])
        return spk1, spk2, pairs

    #######################################################################
    # Method to convert the SpikeList into several others format        ##
    #######################################################################
    def convert(self, format="[times, ids]", relative=False, quantized=False):
        """
        Return a new representation of the SpikeList object, in a user designed format.
            format is an expression containing either the keywords times and ids,
            time and id.

        Inputs:
            relative -  a boolean to say if a relative representation of the spikes
                        times compared to t_start is needed
            quantized - a boolean to round the spikes_times.

        Examples:
            >> spk.convert("[times, ids]") will return a list of two elements, the
                first one being the array of all the spikes, the second the array of all the
                corresponding ids
            >> spk.convert("[(time,id)]") will return a list of tuples (time, id)

        See also
            SpikeTrain.format
        """
        is_times = re.compile("times")
        is_ids = re.compile("ids")
        if len(self) > 0:
            times = np.concatenate(
                [
                    st.format(relative, quantized)
                    for st in self.spiketrains.values()
                ]
            )
            ids = np.concatenate(
                [
                    id * np.ones(len(st.spike_times), int)
                    for id, st in self.spiketrains.items()
                ]
            )
        else:
            times = []
            ids = []
        if is_times.search(format):
            if is_ids.search(format):
                return eval(format)
            else:
                raise Exception(
                    "You must have a format with [times, ids] or [time, id]"
                )
        is_times = re.compile("time")
        is_ids = re.compile("id")
        if is_times.search(format):
            if is_ids.search(format):
                result = []
                for id, time in zip(ids, times):
                    result.append(eval(format))
            else:
                raise Exception(
                    "You must have a format with [times, ids] or [time, id]"
                )
            return result

    def raw_data(self):
        """
        Function to return a N by 2 array of all times and ids.

        Examples:
            >> spklist.raw_data()
            >> array([[  1.00000000e+00,   1.00000000e+00],
                      [  1.00000000e+00,   1.00000000e+00],
                      [  2.00000000e+00,   2.00000000e+00],
                         ...,
                      [  2.71530000e+03,   2.76210000e+03]])

        See also:
            convert()
        """
        data = np.array(self.convert("[times, ids]"), np.float32)
        data = np.transpose(data)
        return data

    def time_axis(self, time_bin):
        """
        Return a time axis between t_start and t_stop according to a time_bin

        Inputs:
            time_bin - the bin width

        See also
            spike_histogram
        """
        axis = np.arange(self.t_start, self.t_stop + time_bin, time_bin)
        return axis

    def select_ids(self, criteria):
        """
        Return the list of all the cells in the SpikeList that will match the criteria
        expressed with the following syntax.

        Inputs :
            criteria - a string that can be evaluated on a SpikeTrain object, where the
                       SpikeTrain should be named ``cell''.

        Examples:
            >> spklist.select_ids("cell.mean_rate() > 0") (all the active cells)
            >> spklist.select_ids("cell.mean_rate() == 0") (all the silent cells)
            >> spklist.select_ids("len(cell.spike_times) > 10")
            >> spklist.select_ids("mean(cell.isi()) < 1")
        """
        selected_ids = []
        for id in self.id_list:
            cell = self.spiketrains[id]
            if eval(criteria):
                selected_ids.append(id)
        return selected_ids

    def id_slice(self, id_list):
        """
        Return a new SpikeList obtained by selecting particular ids

        Inputs:
            id_list - Can be an integer (and then N random cells will be selected)
                      or a sublist of the current ids

        The new SpikeList inherits the time parameters (t_start, t_stop)

        Examples:
            >> spklist.id_list
                [830, 1959, 1005, 416, 1011, 1240, 729, 59, 1138, 259]
            >> new_spklist = spklist.id_slice(5)
            >> new_spklist.id_list
                [1011, 729, 1138, 416, 59]

        See also
            time_slice, interval_slice
        """
        new_SpkList = SpikeList([], [], self.t_start, self.t_stop)
        id_list = self.__sub_id_list(id_list)
        new_SpkList.dimensions = len(
            id_list
        )  # update dimension of new spike list

        for i, id_ in enumerate(id_list):

            # try:
            #    new_SpkList.append(i, self.spiketrains[id_])
            # except Exception:
            #    print("id %d is not in the source SpikeList or already in the new one" % id_)
            #
            # new_SpkList.id.append(id)
            try:
                new_SpkList.spiketrains[id_] = self.spiketrains[id_]
            except Exception:
                # print(
                #     "id %d is not in the source SpikeList or already in the new one"
                #     % id_
                # )
                spikelog.spikelist_flag |= True
                spikelog.spikelist_log = "ID XY is not in the source SpikeList or already in the new one"
                spikelog.spikelist_count += 1

        return new_SpkList

    #######################################################################
    ## Analysis methods that can be applied to a SpikeTrain object       ##
    #######################################################################

    def raster_plot(
        self,
        with_rate=False,
        N=50,
        ax=None,
        dt=1.0,
        display=False,
        save=False,
        **kwargs,
    ):
        """
        Plot a simple raster, for a quick check
        """
        if ax is None:
            fig = plt.figure()
            if with_rate:
                ax1 = plt.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
                ax2 = plt.subplot2grid(
                    (30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1
                )
                ax2.set(xlabel="Time [ms]", ylabel="Rate")
                ax1.set(ylabel="Neuron")
            else:
                ax1 = fig.add_subplot(111)
        else:
            if with_rate:
                assert isinstance(ax, list), (
                    "Incompatible properties... (with_rate requires two axes provided or "
                    "None)"
                )
                ax1 = ax[0]
                ax2 = ax[1]
            else:
                ax1 = ax

        subset = self.id_slice(N)
        ax1.plot(
            subset.raw_data()[:, 0], subset.raw_data()[:, 1], ".", **kwargs
        )
        # .plot(self.raw_data()[:, 0], self.raw_data()[:, 1], '.', **kwargs)

        if with_rate:
            time = self.time_axis(dt)[:-1]
            rate = self.firing_rate(dt, average=True)
            ax2.plot(time, rate)
        try:
            ax1.set(
                ylim=[min(self.id_list), max(self.id_list)],
                xlim=[self.t_start, self.t_stop],
            )
        except:
            pass
        if save:
            assert isinstance(save, str), "Please provide filename"
            plt.savefig(save)

        if display:
            plt.show(False)

    def isi(self):
        """
        Return the list of all the isi vectors for all the SpikeTrains objects
        within the SpikeList.

        See also:
            isi_hist
        """
        isis = []
        for id_ in self.id_list:
            isis.append(self.spiketrains[id_].isi())
        return isis

    def cv_isi(self, float_only=False):
        """
        Return the list of all the CV coefficients for each SpikeTrains object
        within the SpikeList. Return NaN when not enough spikes are present

        Inputs:
            float_only - False by default. If true, NaN values are automatically
                         removed

        Examples:
            >> spklist.cv_isi()
                [0.2,0.3,Nan,2.5,Nan,1.,2.5]
            >> spklist.cv_isi(True)
                [0.2,0.3,2.5,1.,2.5]

        See also:
            cv_isi_hist, cv_local, cv_kl, SpikeTrain.cv_isi

        """
        ids = self.id_list
        N = len(ids)
        cvs_isi = np.empty(N)
        for idx in range(N):
            cvs_isi[idx] = self.spiketrains[ids[idx]].cv_isi()

        if float_only:
            cvs_isi = np.extract(np.logical_not(np.isnan(cvs_isi)), cvs_isi)
            cvs_isi = np.extract(np.logical_not(np.isinf(cvs_isi)), cvs_isi)
        return cvs_isi

    def mean_rate(self, t_start=None, t_stop=None):
        """
        Return the mean firing rate averaged across all SpikeTrains between t_start and t_stop.

        Inputs:
            t_start - begining of the selected area to compute mean_rate, in ms
            t_stop  - end of the selected area to compute mean_rate, in ms

        If t_start or t_stop are not defined, those of the SpikeList are used

        Examples:
            >> spklist.mean_rate()
            >> 12.63

        See also
            mean_rates, mean_rate_std
        """
        return np.mean(self.mean_rates(t_start, t_stop))

    def mean_rate_std(self, t_start=None, t_stop=None):
        """
        Standard deviation of the firing rates across all SpikeTrains
        between t_start and t_stop

        Inputs:
            t_start - beginning of the selected area to compute std(mean_rate), in ms
            t_stop  - end of the selected area to compute std(mean_rate), in ms

        If t_start or t_stop are not defined, those of the SpikeList are used

        Examples:
            >> spklist.mean_rate_std()
            >> 13.25

        See also
            mean_rate, mean_rates
        """
        return np.std(self.mean_rates(t_start, t_stop))

    def mean_rates(self, t_start=None, t_stop=None):
        """
        Returns a vector of the size of id_list giving the mean firing rate for each neuron

        Inputs:
            t_start - beginning of the selected area to compute std(mean_rate), in ms
            t_stop  - end of the selected area to compute std(mean_rate), in ms

        If t_start or t_stop are not defined, those of the SpikeList are used

        See also
            mean_rate, mean_rate_std
        """
        rates = []
        for id in self.id_list:
            rates.append(self.spiketrains[id].mean_rate(t_start, t_stop))
        return rates

    def spike_histogram(self, time_bin, normalized=False, binary=False):
        """
        Generate an array with all the spike_histograms of all the SpikeTrains
        objects within the SpikeList.

        Inputs:
            time_bin   - the time bin used to gather the data
            normalized - if True, the histogram are in Hz (spikes/second), otherwise they are
                         in spikes/bin
            binary     - if True, a binary matrix of 0/1 is returned

        See also
            firing_rate, time_axis
        """
        nbins = self.time_axis(time_bin)
        N = len(self)
        M = (
            len(nbins) - 1
        )  # nbins are the bin edges, so M must correct for this...
        if binary:
            spike_hist = np.zeros((N, M), np.int)
        else:
            spike_hist = np.zeros((N, M), np.float32)

        for idx, id in enumerate(self.id_list):
            hist, edges = np.histogram(self.spiketrains[id].spike_times, nbins)
            hist = hist.astype(float)
            if normalized:
                hist *= 1000.0 / float(time_bin)
            if binary:
                hist = hist.astype(bool)
            spike_hist[idx, :] = hist
        return spike_hist

    def firing_rate(self, time_bin, average=False, binary=False):
        """
        Generate an array with all the instantaneous firing rates along time (in Hz)
        of all the SpikeTrains objects within the SpikeList. If average is True, it gives the
        average firing rate over the whole SpikeList

        Inputs:
            time_bin   - the time bin used to gather the data
            average    - If True, return a single vector of the average firing rate over the whole SpikeList
            binary     - If True, a binary matrix with 0/1 is returned.

        See also
            spike_histogram, time_axis
        """
        result = self.spike_histogram(time_bin, normalized=True, binary=binary)
        if average:
            return np.mean(result, axis=0)
        else:
            return result

    def pairwise_cc(
        self, nb_pairs, pairs_generator=None, time_bin=1.0, average=True
    ):
        """
        Function to generate an array of cross correlations computed
        between pairs of cells within the SpikeTrains.

        Inputs:
            nb_pairs        - int specifying the number of pairs
            pairs_generator - The generator that will be used to draw the pairs. If None, a default one is
                              created as RandomPairs(spk, spk, no_silent=False, no_auto=True)
            time_bin        - The time bin used to gather the spikes
            average         - If true, only the averaged CC among all the pairs is returned (less memory needed)

        Examples
            >> a.pairwise_cc(500, time_bin=1, averagec=True)
            >> a.pairwise_cc(100, CustomPairs(a,a,[(i,i+1) for i in xrange(100)]), time_bin=5)

        See also
            pairwise_pearson_corrcoeff, pairwise_cc_zero, RandomPairs, AutoPairs, CustomPairs
        """
        ## We have to extract only the non silent cells, to avoid problems
        if pairs_generator is None:
            pairs_generator = RandomPairs(self, self, True, True)

        # Then we select the pairs of cells
        pairs = pairs_generator.get_pairs(nb_pairs)
        N = len(pairs)
        length = 2 * (len(pairs_generator.spk1.time_axis(time_bin)) - 1)
        if not average:
            results = np.zeros((N, length), float)
        else:
            results = np.zeros(length, float)
        for idx in range(N):
            # We need to avoid empty spike histogram, otherwise the ccf function
            # will give a nan vector
            hist_1 = pairs_generator.spk1[pairs[idx, 0]].time_histogram(
                time_bin
            )
            hist_2 = pairs_generator.spk2[pairs[idx, 1]].time_histogram(
                time_bin
            )
            if not average:
                results[idx, :] = ccf(hist_1, hist_2)
            else:
                results += ccf(hist_1, hist_2)
        if not average:
            return results
        else:
            return results / N

    def pairwise_pearson_corrcoeff(
        self, nb_pairs, pairs_generator=None, time_bin=1.0, all_coef=False
    ):
        """
        Function to return the mean and the variance of the pearson correlation coefficient.
        For more details, see Kumar et al, ....

        Inputs:
            nb_pairs        - int specifying the number of pairs
            pairs_generator - The generator that will be used to draw the pairs. If None, a default one is
                              created as RandomPairs(spk, spk, no_silent=False, no_auto=True)
            time_bin        - The time bin used to gather the spikes
            all_coef        - If True, the whole list of correlation coefficient is returned

        Examples
            >> spk.pairwise_pearson_corrcoeff(50, time_bin=5)
                (0.234, 0.0087)
            >> spk.pairwise_pearson_corrcoeff(100, AutoPairs(spk, spk))
                (1.0, 0.0)

        See also
            pairwise_cc, pairwise_cc_zero, RandomPairs, AutoPairs, CustomPairs
        """
        ## We have to extract only the non silent cells, to avoid problems
        if pairs_generator is None:
            pairs_generator = RandomPairs(self, self, True, True)

        pairs = pairs_generator.get_pairs(nb_pairs)
        N = len(pairs)
        cor = np.zeros(N, float)

        for idx in range(N):
            # get spike counts at the specified bin size
            hist_1 = pairs_generator.spk1[pairs[idx, 0]].time_histogram(
                time_bin
            )
            hist_2 = pairs_generator.spk2[pairs[idx, 1]].time_histogram(
                time_bin
            )

            # count covariance
            cov = np.corrcoef(hist_1, hist_2)[1][0]
            cor[idx] = cov
        if all_coef:
            return cor
        else:
            cor_coef_mean = cor.mean()
            cor_coef_std = cor.std()
        return (cor_coef_mean, cor_coef_std)


def ccf(x, y, axis=None):
    """
    Fast cross correlation function based on fft.

    Computes the cross-correlation function of two series.
    Note that the computations are performed on anomalies (deviations from
    average).
    Returns the values of the cross-correlation at different lags.

    Parameters
    ----------
    x, y : 1D MaskedArrays
        The two input arrays.
    axis : integer, optional
        Axis along which to compute (0 for rows, 1 for cols).
        If `None`, the array is flattened first.

    Examples
    --------
    >> z = np.arange(5)
    >> ccf(z,z)
    array([  3.90798505e-16,  -4.00000000e-01,  -4.00000000e-01,
            -1.00000000e-01,   4.00000000e-01,   1.00000000e+00,
             4.00000000e-01,  -1.00000000e-01,  -4.00000000e-01,
            -4.00000000e-01])
    """
    assert x.ndim == y.ndim, "Inconsistent shape !"
    if axis is None:
        if x.ndim > 1:
            x = x.ravel()
            y = y.ravel()
        npad = x.size + y.size
        xanom = x - x.mean(axis=None)
        yanom = y - y.mean(axis=None)
        Fx = np.fft.fft(
            xanom,
            npad,
        )
        Fy = np.fft.fft(
            yanom,
            npad,
        )
        iFxy = np.fft.ifft(Fx.conj() * Fy).real
        varxy = np.sqrt(np.inner(xanom, xanom) * np.inner(yanom, yanom))
    else:
        npad = x.shape[axis] + y.shape[axis]
        if axis == 1:
            if x.shape[0] != y.shape[0]:
                raise ValueError("Arrays should have the same length!")
            xanom = x - x.mean(axis=1)[:, None]
            yanom = y - y.mean(axis=1)[:, None]
            varxy = np.sqrt((xanom * xanom).sum(1) * (yanom * yanom).sum(1))[
                :, None
            ]
        else:
            if x.shape[1] != y.shape[1]:
                raise ValueError("Arrays should have the same width!")
            xanom = x - x.mean(axis=0)
            yanom = y - y.mean(axis=0)
            varxy = np.sqrt((xanom * xanom).sum(0) * (yanom * yanom).sum(0))
        Fx = np.fft.fft(xanom, npad, axis=axis)
        Fy = np.fft.fft(yanom, npad, axis=axis)
        iFxy = np.fft.ifft(Fx.conj() * Fy, n=npad, axis=axis).real
    # We just turn the lags into correct positions:
    iFxy = np.concatenate(
        (iFxy[int(len(iFxy) / 2) : len(iFxy)], iFxy[0 : int(len(iFxy) / 2)])
    )
    return iFxy / varxy


class PairsGenerator(abc.ABC):
    """
    PairsGenerator(SpikeList, SpikeList, no_silent)
    This class defines the concept of PairsGenerator, that will be used by all
    the functions using pairs of cells. Functions get_pairs() will then be used
    to obtain pairs from the generator.

    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default

    Examples:
        >> p = PairsGenerator(spk1, spk1, True)
        >> p.get_pairs(100)

    See also AutoPairs, RandomPairs, CustomPairs, DistantDependentPairs
    """

    def __init__(self, spk1, spk2, no_silent=False):
        self.spk1 = spk1
        self.spk2 = spk2
        self.no_silent = no_silent
        self._get_id_lists()

    def _get_id_lists(self):
        self.ids_1 = set(self.spk1.id_list)
        self.ids_2 = set(self.spk2.id_list)
        if self.no_silent:
            n1 = set(self.spk1.select_ids("len(cell.spike_times) == 0"))
            n2 = set(self.spk2.select_ids("len(cell.spike_times) == 0"))
            self.ids_1 -= n1
            self.ids_2 -= n2

    @abc.abstractmethod
    def get_pairs(self, nb_pairs):
        """
        Function to obtain a certain number of cells from the generator

        Inputs:
            nb_pairs - int to specify the number of pairs desired

        Examples:
            >> res = p.get_pairs(100)
        """
        raise NotImplementedError()


class RandomPairs(PairsGenerator):
    """
    RandomPairs(SpikeList, SpikeList, no_silent, no_auto). Inherits from PairsGenerator.
    Generator that will return random pairs of elements.

    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. True by default
        no_auto   - Boolean to say if pairs with the same element (id,id) should
                    be removed. True by default, i.e those pairs are discarded

    Examples:
        >> p = RandomPairs(spk1, spk1, True, False)
        >> p.get_pairs(4)
            [[1,3],[2,5],[1,4],[5,5]]
        >> p = RandomPairs(spk1, spk1, True, True)
        >> p.get_pairs(3)
            [[1,3],[2,5],[1,4]]


    See also RandomPairs, CustomPairs, DistantDependentPairs
    """

    def __init__(self, spk1, spk2, no_silent=True, no_auto=True):
        PairsGenerator.__init__(self, spk1, spk2, no_silent)
        self.no_auto = no_auto

    def get_pairs(self, nb_pairs):
        cells1 = np.array(list(self.ids_1), int)
        cells2 = np.array(list(self.ids_2), int)
        pairs = np.zeros((0, 2), int)
        N1 = len(cells1)
        N2 = len(cells2)
        T = min(N1, N2)
        while len(pairs) < nb_pairs:
            N = min(nb_pairs - len(pairs), T)
            tmp_pairs = np.zeros((N, 2), int)
            tmp_pairs[:, 0] = cells1[
                np.floor(np.random.uniform(0, N1, N)).astype(int)
            ]
            tmp_pairs[:, 1] = cells2[
                np.floor(np.random.uniform(0, N2, N)).astype(int)
            ]
            if self.no_auto:
                idx = np.where(tmp_pairs[:, 0] == tmp_pairs[:, 1])[0]
                pairs = np.concatenate(
                    (pairs, np.delete(tmp_pairs, idx, axis=0))
                )
            else:
                pairs = np.concatenate((pairs, tmp_pairs))
        return pairs


###############################################################################
#   functions for loading the data from the data files and proceccing them    #
###############################################################################


def create_plotdir(dirName):
    dirName = "plots/" + dirName
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    return dirName


def load_pickle(data_path, folder):
    # loads either all data to a list or data from one specific file if folder is false
    if folder:
        data = []
        # load all the files from path
        for path in os.listdir(data_path):
            fi = os.path.join(data_path, path)
            with open(fi, "rb") as file:
                data.append(pickle.load(file))
    else:
        with open(data_path, "r") as file:
            data = pickle.load(file)
    return data


def create_SpikeList(data, N, t_start=None, t_stop=None):
    # creates spikelists from data dicts
    if "senders" in data.keys():
        senders = data["senders"]
        times = data["times"]
    elif "res" in data.keys():
        senders = data["res"][0]
        times = data["res"][1]
    if len(senders) > 0:
        return SpikeList(
            list(zip(senders, times)),
            list(range(N + 1)),
            t_start=t_start,
            t_stop=t_stop,
        )
    else:
        print("no spikes")
        pass


def create_nmda_spikelist(raw_data):
    data = find_NMDA(raw_data, -35.0, 29)


def network_nmda_stats(raw_data, params):
    results = {}

    spikelist = create_SpikeList(
        raw_data, N=params["N"], t_start=2000, t_stop=2000 + raw_data["t_sim"]
    )
    ex_spikelist = spikelist.id_slice(raw_data["excitatory_ids"])
    inh_spikelist = spikelist.id_slice(raw_data["inhibitory_ids"])

    results["avg_ex_rate"] = np.round(ex_spikelist.mean_rate(), 3)
    results["avg_inh_rate"] = np.round(inh_spikelist.mean_rate(), 3)

    results["N_silent"] = len(spikelist.select_ids("cell.mean_rate() == 0"))
    N_thresh = len(spikelist.select_ids("cell.mean_rate() >= 20"))
    results["pu20"] = np.round((1 - N_thresh / params["N"]) * 100, 4)

    results["max_avg_rate"] = float(
        np.max(spikelist.firing_rate(5.0, average=True))
    )
    results["max_neuron_rate"] = np.max(spikelist.mean_rates())

    results[
        "avg_cc"
    ] = np.nan  # backup to handle if cv and cc calculation fails
    results["avg_cv"] = np.nan
    results["avg_rate"] = np.nan
    try:
        cv = np.nan
        cv = spikelist.cv_isi()
        cv = cv[np.logical_not(np.isnan(cv))]
        if len(cv) > 0:
            results["cv_min_max"] = "[{},{}]".format(
                np.round(np.min(cv), 2), np.round(np.max(cv), 2)
            )
        results["avg_rate"] = np.round(spikelist.mean_rate(), 3)
        corrcoef = np.nan
        corrcoef = np.round(
            spikelist.pairwise_pearson_corrcoeff(100), 4
        )  ######
        results["avg_cc"] = corrcoef[0]
        results["avg_cv"] = np.round(np.mean(cv), 4)
    except:
        raise ValueError("AIness could not be calculated")

    nmda_data = find_NMDA(raw_data, -35.0, 30)
    results["relative_upstate"] = np.sum(
        [nmda_data[neuron_id]["rel_up"] for neuron_id in nmda_data]
    ) / len(nmda_data)
    results["avg_nmda_length"] = np.sum(
        [nmda_data[neuron_id]["avg_length"] for neuron_id in nmda_data]
    ) / len(nmda_data)

    return results


def net_rate(raw_data, params):
    results = {}

    spikelist = create_SpikeList(
        raw_data, N=params["N"], t_start=2000, t_stop=2000 + raw_data["t_sim"]
    )
    ex_spikelist = spikelist.id_slice(raw_data["excitatory_ids"])
    inh_spikelist = spikelist.id_slice(raw_data["inhibitory_ids"])

    results["avg_rate"] = np.round(spikelist.mean_rate(), 3)
    results["avg_ex_rate"] = np.round(ex_spikelist.mean_rate(), 3)
    results["avg_inh_rate"] = np.round(inh_spikelist.mean_rate(), 3)

    return results


def consecutive_int(nums, min_l):
    # find consecutive integers with a minimum length of min_l from a list
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    intervals = list(zip(edges, edges))
    conditioned_intervals = []
    for inter in intervals:
        if np.abs(inter[0] - inter[1]) >= min_l:
            conditioned_intervals.append(inter)
    return conditioned_intervals


def trace_to_nmda_list(voltage_trace, threshold, l_min):

    return consecutive_int(np.where(voltage_trace >= threshold)[0], l_min)


def data_to_volttrace(data, name):
    """
    takes the output of a multimeter which recordered multiple compartments of multiple nerons

    output: dict{neuron_id: {times : [...], v_comp0 : [...], v_comp1 : [...], ...}
    """
    # only considering the relevant part of the data
    data = data[name]
    neuron_ids = np.unique(data["senders"])
    out = {}
    for neuron_id in neuron_ids:
        int_slice = np.where(data["senders"] == neuron_id)[0].tolist()
        out[f"{neuron_id}"] = {
            f"{lable}": data[f"{lable}".format(lable)][int_slice]
            for lable in ["times", "v_comp0"] + data["v_comps"]
        }
    return out


def find_NMDA(data, threshold, l_min, name="sample_neurons"):
    # transforms voltage trace dict and returns nmda spike locations
    vtrace_data = data_to_volttrace(data, name)  # dict with the voltage traces
    leaf_idxs = data[name]["leaf_idxs"]
    out_data = vtrace_data

    for neuron_id in vtrace_data:
        out_data[neuron_id]["rel_up"] = 0
        out_data[neuron_id]["neuron_id"] = int(neuron_id)
        out_data[neuron_id]["avg_length"] = 0
        for leaf_idx in leaf_idxs:
            nmda_dict = {}

            time_ints = np.where(
                vtrace_data[neuron_id][f"v_comp{leaf_idx}"] >= threshold
            )[0]
            nmda_dict["nmda_pos"] = consecutive_int(time_ints, l_min)
            nmda_dict["n_nmda"] = len(nmda_dict["nmda_pos"])
            nmda_dict["t_up"] = 0
            nmda_dict["avg_length"] = 0
            if nmda_dict["n_nmda"] > 0:
                nmda_dict["t_up"] = np.sum(
                    [itv[1] - itv[0] for itv in nmda_dict["nmda_pos"]]
                )
                nmda_dict["avg_length"] = (
                    nmda_dict["t_up"] / nmda_dict["n_nmda"]
                )
            nmda_dict["rel_up"] = (
                float(
                    nmda_dict["t_up"]
                    / len(vtrace_data[neuron_id][f"v_comp{leaf_idx}"])
                )
                * 100
            )

            out_data[neuron_id][f"nmda_comp{leaf_idx}"] = nmda_dict
            out_data[neuron_id]["rel_up"] += nmda_dict["rel_up"]
            out_data[neuron_id]["avg_length"] += nmda_dict["avg_length"]

        out_data[neuron_id]["rel_up"] = out_data[neuron_id]["rel_up"] / len(
            leaf_idxs
        )
        out_data[neuron_id]["avg_length"] = out_data[neuron_id][
            "avg_length"
        ] / len(leaf_idxs)
    return out_data


def nmda_distribution(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    leaf_idxs = data["sample_neurons"]["leaf_idxs"]
    data = find_NMDA(data, -35.0, 20)
    nmda_dist = {f"comp{leaf_idx}": 0 for leaf_idx in leaf_idxs}
    for neuron_id in data:
        for leaf_idx in leaf_idxs:
            nmda_dist[f"comp{leaf_idx}"] += data[neuron_id][
                f"nmda_comp{leaf_idx}"
            ]["n_nmda"]
    return nmda_dist

    """
    if len(volt_trace_data['times']):
        time_int = int(np.max(volt_trace_data['times']) - np.min(volt_trace_data['times']) + 1)
        volt_trace_data['times'] =  np.array(volt_trace_data['times']).reshape(time_int, n_traces).transpose()[0]
        nmda_list = []
        total_nmda = 0
        for indx, v in enumerate(comps):
            volt_trace_data[v] = np.array(volt_trace_data[v]).reshape(time_int, n_traces).transpose()
            for id in ids:
                time_ints = np.where(volt_trace_data[v][id-1][2000:] >= -35.)[0]
                nmdas = consecutive_int(time_ints, l_min-1) #min lengh of nmdaspike 30ms
                if len(nmdas) > 0:
                    volt_trace = volt_trace_data[v][id-1][2000:][nmdas[0][0] - 5: nmdas[0][0] + 150]
                    nmda_list.append([id, nmdas, v, volt_trace])
                    total_nmda += len(nmdas)
        total_up = total_upstate(nmda_list)
        rel_up = total_up / (time_int*n_traces*len(comps))
    else:
        nmda_list = np.nan
        total_nmda = np.nan
    return np.array(nmda_list, dtype=object), total_nmda, rel_up, volt_trace_data
    """


def pproc_capacity(raw_data):
    spk_list = create_SpikeList(
        raw_data,
        raw_data["N"],
        t_start=2000.0,
        t_stop=2000 + raw_data["steps"] * raw_data["tstep"],
    )
    ex_spk_list = spk_list.id_slice(raw_data["excitatory_ids"])

    state_mat = spklst_to_ratestate(
        ex_spk_list, steps=raw_data["steps"], tstep=raw_data["tstep"]
    )
    seq = raw_data["seq"]
    seq = [[q] for q in seq]
    results = {}
    (
        results["totalcap"],
        results["allcaps"],
        results["numcaps"],
        results["nodes"],
    ) = calc_cap(np.array(seq), np.array(state_mat))

    return results


def calc_cap(seq, state_mat):
    Citer = cap.capacity_iterator()  # verbose = 1)
    totalcap, allcaps, numcaps, nodes = Citer.collect(seq, state_mat)
    print(
        "\nMeasured ",
        numcaps,
        " capacities above threshold.\nTotal capacity = ",
        totalcap,
    )
    print("\n allcaps =", allcaps, "  Nodes =", nodes)
    return totalcap, allcaps, numcaps, nodes


def spklst_to_ratestate(spk_list, steps, tstep, t_start=2000.0):
    state_mat = [
        spk_list.mean_rates(t_start + i * tstep, t_start + (i + 1) * tstep)
        for i in range(steps)
    ]
    state_mat = np.nan_to_num(state_mat)
    print(np.shape(state_mat))
    return state_mat


def total_upstate(nmda_list):
    t_upstate = 0
    for nmdas in nmda_list:
        for st in nmdas[1]:
            t_upstate += st[1] - st[0]
    return t_upstate


if __name__ == "__main__":

    a = 2
