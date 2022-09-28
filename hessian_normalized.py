#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from jax import grad, jit, jacfwd, jacrev
import jax.numpy as jnp
from scipy.signal import argrelextrema


class HessianCircadian:

    def __init__(self):
        self._default_params()


    def hessian(self, u0: jnp.ndarray = jnp.array([0.70,0.0,0.0]), light: jnp.ndarray = jnp.zeros(240*7)):
        params = self.get_parameters_array()
        statefinal = self.step_all_n(u0, light, params, 0.10) 
        def loss(params): 
            states_params = HessianCircadian.step_all_n(u0, light, params, 0.10)
            x1 = states_params[:,0] * jnp.cos(states_params[:,1])
            y1 = states_params[:,0] * jnp.sin(states_params[:,1])
            x2 = statefinal[:,0] * jnp.cos(statefinal[:,1])
            y2 = statefinal[:,0] * jnp.sin(statefinal[:,1])
            return jnp.mean((x1 - x2) ** 2 + (y1 - y2) ** 2)
        H = jacfwd(jacrev(loss)) # hessian matrix, derivatives wrt parameters 
        return H(params)
    def jacobian(self, u0: jnp.ndarray = jnp.array([0.70,0.0,0.0]), light: jnp.ndarray = jnp.zeros(240*7)):
        params = self.get_parameters_array()
        statefinal = self.step_n(u0, light, params, 0.10)
        
        def loss(params): return HessianCircadian.norm(HessianCircadian.step_n(u0, light, params, 0.10), statefinal) 
        J = jacfwd(loss) # hessian matrix, derivatives wrt parameters 
        return J(params)
    def normalized_hessian(self, u0: jnp.ndarray = jnp.array([0.70,0.0,0.0]), light: jnp.ndarray = jnp.zeros(240*7)):
        params = self.get_parameters_array()
        len_param = len(params)
        statefinal = self.step_all_n(u0, light, params, 0.10) 
        def loss(lam: jnp.ndarray = jnp.zeros(len_param)): 
            states_params = HessianCircadian.step_all_n(u0, light, lam*params, 0.10)
            x1 = states_params[:,0] * jnp.cos(states_params[:,1])
            y1 = states_params[:,0] * jnp.sin(states_params[:,1])
            x2 = statefinal[:,0] * jnp.cos(statefinal[:,1])
            y2 = statefinal[:,0] * jnp.sin(statefinal[:,1])
            return jnp.mean((x1 - x2) ** 2 + (y1 - y2) ** 2) 
        H = jacfwd(jacrev(loss)) # hessian matrix, derivatives wrt parameters 
        lam_default = jnp.ones(len_param)#lam_default = 1.0
        return H(lam_default)
    def du_dp(self, u0: jnp.ndarray = jnp.array([0.70,0.0,0.0]), light: jnp.ndarray = jnp.zeros(240*7)):
        params = self.get_parameters_array()
        states_normal = HessianCircadian.step_all_n(u0, light, params, 0.10)
        rcos_psi_normal = states_normal[:,0]*jnp.cos(states_normal[:,1])
        def loss(params):
            states_all_params = HessianCircadian.step_all_n(u0, light, params, 0.10)
            rcos_psi = states_all_params[:,0]*jnp.cos(states_all_params[:,1])
            return jnp.mean(states_all_params[:,0]) #rcos_psi[-1] # last value 
        du_dp_val = jacfwd(loss)
        scaled_sensitivity_ind1 = (params/jnp.mean(states_normal[:,0]))*du_dp_val(params)
        return scaled_sensitivity_ind1 #, du2_dp_val
    
    
    def _default_params(self):
        """
            Load the model parameters, if useFile is False this will search the local directory for a optimalParams.dat file.
            setParameters()
            No return value
        """
        default_params = {'tau': 23.84, 'K': 0.06358, 'gamma': 0.024,
                          'Beta1': -0.09318, 'A1': 0.3855, 'A2': 0.1977,
                          'BetaL1': -0.0026, 'BetaL2': -0.957756, 'sigma': 0.0400692,
                          'G': 33.75, 'alpha_0': 0.05, 'delta': 0.0075,
                          'p': 1.5, 'I0': 9325.0}

        self.set_parameters(default_params)

    def get_parameters_array(self):
        """
            Return a numpy array of the models current parameters
        """
        return jnp.array([self.tau, self.K, self.gamma, self.Beta1, self.A1, self.A2, self.BetaL1, self.BetaL2, self.sigma, self.G, self.alpha_0, self.delta, self.p, self.I0])

    def set_parameters(self, param_dict: dict):
        """
            Update the model parameters using a passed in parameter dictionary. Any parameters not included
            in the dictionary will be set to the default values.
            updateParameters(param_dict)
            Returns null, changes the parameters stored in the class instance
        """

        params = ['tau', 'K', 'gamma', 'Beta1', 'A1', 'A2', 'BetaL1',
                  'BetaL2', 'sigma', 'G', 'alpha_0', 'delta', 'p', 'I0']

        for key, value in param_dict.items():
            setattr(self, key, value)

    @staticmethod
    @jit
    def spmodel(u: jnp.array, light: float, params: jnp.array):

        R, Psi, n = u

        tau, K, gamma, Beta1, A1, A2, BetaL1, BetaL2, sigma, G, alpha_0, delta, p, I0 = params
            

        alpha_0_func = alpha_0 * pow(light, p) / (pow(light, p) + I0)
        Bhat = G * (1.0 - n) * alpha_0_func
        LightAmp = A1 * 0.5 * Bhat * (1.0 - pow(R, 4.0)) * jnp.cos(Psi + BetaL1) + A2 * 0.5 * Bhat * R * (1.0 - pow(R, 8.0)) * jnp.cos(2.0 * Psi + BetaL2)
        LightPhase = sigma * Bhat - A1 * Bhat * 0.5 * (pow(R, 3.0) + 1.0 / R) * jnp.sin(
            Psi + BetaL1) - A2 * Bhat * 0.5 * (1.0 + pow(R, 8.0)) * jnp.sin(2.0 * Psi + BetaL2)

        dR = -1.0 * gamma * R + K * jnp.cos(Beta1) / 2.0 * R * (1.0 - pow(R, 4.0)) + LightAmp
        dPsi = 2 * jnp.pi / tau + K / 2.0 * jnp.sin(Beta1) * (1 + pow(R, 4.0)) + LightPhase
        dn = 60.0 * (alpha_0_func * (1.0 - n) - delta * n)

        du = jnp.array([dR, dPsi, dn])
        return (du)

    @staticmethod
    def spmodel_rk4_step(ustart: jnp.ndarray, light_val: float, params: jnp.array, dt: float):
        """
            Takes a single step forward using the default spmodel
        """

        k1 = HessianCircadian.spmodel(ustart, light_val, params)
        k2 = HessianCircadian.spmodel(ustart + dt / 2 * k1, light_val, params)
        k3 = HessianCircadian.spmodel(ustart + dt / 2 * k2, light_val, params)
        k4 = HessianCircadian.spmodel(ustart + dt * k3, light_val, params)
        return ustart + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4) # runge-kutta 

    @staticmethod
    def step_n(u0: jnp.ndarray, light: jnp.ndarray, params: jnp.ndarray, dt: float):
        for k in range(len(light)):
            u0 = HessianCircadian.spmodel_rk4_step(u0, light[k], params, dt)
        return u0 # only return final value

    @staticmethod
    @jit
    def norm(s1: jnp.ndarray, s2: jnp.ndarray):
        x1 = s1[0] * jnp.cos(s1[1])
        y1 = s1[0] * jnp.sin(s1[1])
        x2 = s2[0] * jnp.cos(s2[1])
        y2 = s2[0] * jnp.sin(s2[1])

        return (x1 - x2) ** 2 + (y1 - y2) ** 2
    
    @staticmethod
    def step_all_n(u0: jnp.ndarray, light: jnp.ndarray, params: jnp.ndarray, dt: float):
        u_all = jnp.zeros([len(light), 3])
        for k in range(len(light)):
            u0 = HessianCircadian.spmodel_rk4_step(u0, light[k], params, dt)
            u_all = u_all.at[k].set(u0)#u_all[k,] = u0
        return u_all # return all values
    
    @staticmethod
    def ics_individual_schedules(final_state_diff, convergence_val, ics, lights, params):
        u0 = ics
        count = 0
        while final_state_diff > convergence_val and count < 50:

            # simulate the model and extract the final time as the initial condition
            count = count + 1
            statesfinal = HessianCircadian.step_n(u0 = u0, light = lights, params = params, dt = 0.10) # final state value
            final_state_diff = abs(statesfinal[0] - u0[0]) + abs(np.mod(statesfinal[1] - u0[1] + np.pi,2*np.pi) - np.pi)
            #print(final_state_diff)
            u0 = statesfinal
        return u0
    
from scipy.ndimage import gaussian_filter1d
class Actogram(object):

    def __init__(self,
                 time_total: np.ndarray,
                 light_vals: np.ndarray,
                 second_zeit: np.ndarray = None,
                 ax=None,
                 threshold=10.0,
                 threshold2=None,
                 smooth=True
                 ):
        """
            Create an actogram of the given marker..... 

            (self, time_total: np.ndarray, light_vals: np.ndarray, ax=None, threshold=10.0) 
        """

        self.time_total = time_total
        self.light_vals = light_vals
        self.num_days = np.ceil((time_total[-1] - time_total[0])/24.0)-1

        self.second_zeit = second_zeit if second_zeit is not None else light_vals

        if smooth:
            self.light_vals=gaussian_filter1d(self.light_vals,sigma=2.0)
            self.second_zeit=gaussian_filter1d(self.second_zeit,sigma=0.1)

        if threshold2 is None:
            threshold2 = threshold

        if ax is not None:
            self.ax = ax
        else:
            #plt.figure(figsize=(18, 12))
            plt.figure()
            ax = plt.gca()
            self.ax = ax

        # Set graphical parameters
        label_scale = int(np.floor(self.num_days/30))
        if label_scale < 1:
            label_scale = 1

        start_day=np.floor(self.time_total[0]/24.0)
        self.ax.set_ylim(start_day, self.num_days+start_day)
        self.ax.set_xlim(0, 48)
        self.ax.set_yticks(np.arange(int(start_day), self.num_days+1+start_day, label_scale))
        ylabels_list = list(range(int(start_day), int(self.num_days+start_day)+1, label_scale))
        # ylabels_list.reverse()

        self.ax.set_yticklabels(ylabels_list)
        self.ax.set_xticks(np.arange(0, 48+3, 3))
        xlabels_list = list(range(0, 27, 3))+list(range(3, 27, 3))
        self.ax.set_xticklabels(xlabels_list)
        self.ax.set_xticks(np.arange(0, 48, 1), minor=True)

        #self.ax.yaxis.grid(False, linewidth=1.0, color='k')
        # self.ax.xaxis.grid(False)

        self.ax.plot(24.0*np.ones(100), np.linspace(0, self.num_days,
                     100), ls='--', lw=2.0, color='black', zorder=9)
        self.ax.set_xlabel("time (hours)")
        self.ax.set_ylabel("day")

        self.addLightSchedule(self.light_vals, threshold, plt_option='left')
        self.addLightSchedule(self.second_zeit, threshold2, plt_option='right')
        self.ax.invert_yaxis()

    def getRectangles(self, timeon, timeoff, colorIn='black'):
        bottom_x = np.fmod(timeon, 24.0)
        bottom_y = int(timeon/24.0)  # -1
        #bottom_y = self.num_days-int(timeon/24.0)-1
        r1 = plt.Rectangle((bottom_x, bottom_y), timeoff -
                           timeon, 0.5, fc=colorIn, zorder=-1)
        r2 = plt.Rectangle((bottom_x+24.0, bottom_y),
                           timeoff-timeon, 0.5, fc=colorIn, zorder=-1)
        return((r1, r2))

    def addRect(self, timeon, timeoff, colorIn='black', plt_option='both'):
        """Used to add a rectangle to the axes"""
        r = self.getRectangles(timeon, timeoff, colorIn)
        if plt_option == 'left':
            self.ax.add_patch(r[0])
            return
        if plt_option == 'right':
            self.ax.add_patch(r[1])
            return
        self.ax.add_patch(r[0])
        self.ax.add_patch(r[1])

    def addLightSchedule(self, zeit: np.ndarray, threshold: float, plt_option: str = 'both', color='white'):
        """
            Add the light schedule as colored rectangles to the axes

        """

        lightdata = zeit
        timedata = self.time_total
        lightsOn = False
        if (lightdata[0] > threshold):
            lightsOn = True
            lightStart = timedata[0]
        else:
            darkOn = True
            darkStart = timedata[0]

        dayCounter = int(timedata[0]/24.0)  # count the days in the data set
        for i in range(1, len(lightdata)):
            currentDay = int(timedata[i]/24.0)
            if (currentDay != dayCounter):
                dayCounter = currentDay
                if (lightsOn == True):
                    self.addRect(
                        lightStart, timedata[i], plt_option=plt_option)
                    if (i+1 < len(timedata)):
                        # reset the light counter to start over the next day
                        lightStart = timedata[i+1]
                else:
                    self.addRect(
                        darkStart, timedata[i], colorIn=color, plt_option=plt_option)
                    if (i+1 < len(timedata)):
                        darkStart = timedata[i+1]

            if (lightdata[i] < threshold and lightsOn == True):
                self.addRect(lightStart, timedata[i-1], plt_option=plt_option)
                lightsOn = False
                darkOn = True
                darkStart = timedata[i]
            if (lightsOn == False and lightdata[i] >= threshold):
                lightsOn = True
                lightStart = timedata[i]
                darkOn = False
                self.addRect(
                    darkStart, timedata[i-1], colorIn=color, plt_option=plt_option)

    def plot_phasemarker(self, phase_marker_times: np.ndarray,
                         error: np.ndarray = None,
                         alpha=1.0,
                         alpha_error=0.30,
                         scatter=False,
                         *args, **kwargs):
        """
        This method takes in a list of times which are assumed to occur at the same 
        circadian phase (e.g. DLMO, CBTmin). These are plotted as points 
        on the actogram.  
            plot_phasemarker(self, phase_marker_times: np.ndarray, *args, **kwargs)
        """

        xvals = deepcopy(phase_marker_times)
        yvals = deepcopy(phase_marker_times)

        xvals = np.fmod(xvals, 24.0)
        yvals = np.floor(yvals / 24.0) + 0.5

        if scatter:
            self.ax.scatter(xvals, yvals, *args, **kwargs)
            self.ax.scatter(xvals+24.0, yvals, *args, **kwargs)

        idx_split = (np.absolute(np.diff(xvals)) > 6.0).nonzero()[0]+1
        xvals_split = np.split(xvals, idx_split)
        yvals_split = np.split(yvals, idx_split)
        if error is not None:
            error_split = np.split(error, idx_split)

        for (idx, xx) in enumerate(xvals_split):
            self.ax.plot(xx, yvals_split[idx], alpha=alpha, *args, **kwargs)
            self.ax.plot(
                xx+24.0, yvals_split[idx], alpha=alpha, *args, **kwargs)
            if error is not None:
                self.ax.fill_betweenx(
                    yvals_split[idx], xx-error_split[idx], xx+error_split[idx], alpha=alpha_error, *args, **kwargs)
                self.ax.fill_betweenx(
                    yvals_split[idx], xx-error_split[idx]+24.0, xx+error_split[idx]+24.0, alpha=alpha_error, *args, **kwargs)

    def plot_phasetimes(self, times: np.ndarray, phases: np.ndarray, error: np.ndarray = None,
                        alpha_error=0.30, alpha=1.0, *args, **kwargs):
        """
            This method takes observations of the phase and times (same length arrays)
            and adds them to the actogram.    

            plot_phasetimes(self, times: np.ndarray, phases: np.ndarray, *args, **kwargs)
        """
        xvals = deepcopy(phases)
        xvals = np.arctan2(np.sin(xvals), np.cos(xvals))
        for i in range(len(xvals)):
            if xvals[i] < 0.0:
                xvals[i] += 2*np.pi

        xvals = np.fmod(xvals, 2*np.pi)
        xvals *= 12.0/np.pi

        xvals = np.fmod(xvals, 24.0)
        yvals = deepcopy(times)
        yvals = np.floor(yvals / 24.0) + 0.5

        idx_split = (np.absolute(np.diff(xvals)) > 6.0).nonzero()[0]+1
        xvals_split = np.split(xvals, idx_split)
        yvals_split = np.split(yvals, idx_split)
        if error is not None:
            error_split = np.split(error, idx_split)

        for (idx, xx) in enumerate(xvals_split):
            self.ax.plot(xx, yvals_split[idx], alpha=alpha, *args, **kwargs)
            self.ax.plot(
                xx+24.0, yvals_split[idx], alpha=alpha, *args, **kwargs)
            if error is not None:
                self.ax.fill_betweenx(
                    yvals_split[idx], xx-error_split[idx], xx+error_split[idx], alpha=alpha_error, *args, **kwargs)
                self.ax.fill_betweenx(
                    yvals_split[idx], xx-error_split[idx]+24.0, xx+error_split[idx]+24.0, alpha=alpha_error, *args, **kwargs)

class ParameterRecovery:
    
    def __init__(self):
        self._default_params()
        
    def _default_params(self):
        """
            Load the model parameters, if useFile is False this will search the local directory for a optimalParams.dat file.
            setParameters()
            No return value
        """
        default_params = {'tau': 23.84, 'K': 0.06358, 'gamma': 0.024,
                          'Beta1': -0.09318, 'A1': 0.3855, 'A2': 0.1977,
                          'BetaL1': -0.0026, 'BetaL2': -0.957756, 'sigma': 0.0400692,
                          'G': 33.75, 'alpha_0': 0.05, 'delta': 0.0075,
                          'p': 1.5, 'I0': 9325.0}

        self.set_parameters(default_params)
        
    def set_parameters(self, param_dict: dict):
        """
            Update the model parameters using a passed in parameter dictionary. Any parameters not included
            in the dictionary will be set to the default values.
            updateParameters(param_dict)
            Returns null, changes the parameters stored in the class instance
        """

        params = ['tau', 'K', 'gamma', 'Beta1', 'A1', 'A2', 'BetaL1',
                  'BetaL2', 'sigma', 'G', 'alpha_0', 'delta', 'p', 'I0']

        for key, value in param_dict.items():
            setattr(self, key, value)
        
    def find_dlmo(model_states_phase, model_states_amp, t_values):
        
        distance_from_cbtmin_phase = (np.mod(model_states_phase,2*np.pi) - np.pi)**2
        minima_indices = argrelextrema(distance_from_cbtmin_phase, np.less)
        close_to_zero_indices = np.where(distance_from_cbtmin_phase < 0.5)
        min_indices = np.intersect1d(minima_indices, close_to_zero_indices)
        min_times = t_values[min_indices]
        predict_phase = np.mod(min_times - 7, 24)
        predict_amps = abs(model_states_amp[min_indices])
        return min_times # predict_phase
    
    def noisy_light(self, mu, sigma, light, ts):
        
        # simulate a schedule with noisy light values
        
        # generate the light schedule
        noise_add = np.random.normal(mu, sigma, len(ts))
        noisy_light = light + noise_add
        noisy_light = noisy_light.at[noisy_light < 0].set(0)
        
        # simulate the model
        sens=HessianCircadian()
        u0 = jnp.array([0.70,0.0,0.0]) # initial condition 1
        params = sens.get_parameters_array()
        u0_new = sens.ics_individual_schedules(100, 10**(-3), u0, noisy_light, params)
        #u0_new = sens.step_n(u0, noisy_light, params, 0.10) 
        model_states = sens.step_all_n(u0_new, noisy_light, params, 0.10)
        return model_states
    
    def noisy_light_params(self, mu, sigma, light, ts, params):
        
        # simulate a schedule with noisy light values
        
        # generate the light schedule
        noise_add = np.random.normal(mu, sigma, len(ts))
        noisy_light = light + noise_add
        noisy_light = noisy_light.at[noisy_light < 0].set(0)
        
        # simulate the model
        sens=HessianCircadian()
        u0 = jnp.array([0.70,0.0,0.0]) # initial condition 1
        u0_new = sens.ics_individual_schedules(100, 10**(-3), u0, noisy_light, params)
        #u0_new = sens.step_n(u0, noisy_light, params, 0.10) 
        model_states = sens.step_all_n(u0_new, noisy_light, params, 0.10)
        return model_states
    
    def perturbed_params(self, u0, params, light, ts):
        
        # simulate a schedule with true light values and perturbed parameters
        sens=HessianCircadian()
        #u0 = jnp.array([0.70,0.0,0.0]) # initial condition 1
        #u0_new = sens.ics_individual_schedules(100, 10**(-3), u0, light, params)
        #u0_new = sens.step_n(u0, light, params, 0.10) # changed: now generate ic with perturbed params
        model_states = sens.step_all_n(u0, light, params, 0.1)
        return model_states
        
    def loss_recovery(self, model_states_true_params, model_states_perturbed_params, ts):
        
        x1 = model_states_perturbed_params[:,0] * jnp.cos(model_states_perturbed_params[:,1])
        y1 = model_states_perturbed_params[:,0] * jnp.sin(model_states_perturbed_params[:,1])
        x2 = model_states_true_params[:,0] * jnp.cos(model_states_true_params[:,1])
        y2 = model_states_true_params[:,0] * jnp.sin(model_states_true_params[:,1])
        norm_diff = np.mean((x1 - x2) ** 2 + (y1 - y2) ** 2)

        return norm_diff

        
        
    def recover_parameter(self, output_sim_scores):
        return np.nanargmin(output_sim_scores)
    
class HessianCircadianNewParams: # excluding K and gamma 

    def __init__(self):
        self._default_params()


    def hessian(self, u0: jnp.ndarray = jnp.array([0.70,0.0,0.0]), light: jnp.ndarray = jnp.zeros(240*7)):
        params = self.get_parameters_array()
        statefinal = self.step_all_n(u0, light, params, 0.10) # so this is variable value after simulate model with all of light
        def loss(params): 
            states_params = HessianCircadianNewParams.step_all_n(u0, light, params, 0.10)
            x1 = states_params[:,0] * jnp.cos(states_params[:,1])
            y1 = states_params[:,0] * jnp.sin(states_params[:,1])
            x2 = statefinal[:,0] * jnp.cos(statefinal[:,1])
            y2 = statefinal[:,0] * jnp.sin(statefinal[:,1])
            return jnp.mean((x1 - x2) ** 2 + (y1 - y2) ** 2)
#         def loss(params): return HessianCircadianNewParams.norm(HessianCircadianNewParams.step_n(u0, light, params, 0.10), statefinal) 
#         H = jacfwd(jacrev(loss)) # hessian matrix, derivatives wrt parameters 
#         return H(params)
        H = jacfwd(jacrev(loss)) # hessian matrix, derivatives wrt parameters 
        return H(params)
    def jacobian(self, u0: jnp.ndarray = jnp.array([0.70,0.0,0.0]), light: jnp.ndarray = jnp.zeros(240*7)):
        params = self.get_parameters_array()
        statefinal = self.step_n(u0, light, params, 0.10) # so this is variable value after simulate model with all of light
        def loss(params): return HessianCircadianNewParams.norm(HessianCircadianNewParams.step_n(u0, light, params, 0.10), statefinal) 
        J = jacfwd(loss) # hessian matrix, derivatives wrt parameters 
        return J(params)
    def _default_params(self):
        """
            Load the model parameters, if useFile is False this will search the local directory for a optimalParams.dat file.
            setParameters()
            No return value
        """
        default_params = {'tau': 23.84,
                          'Beta1': -0.09318, 'A1': 0.3855, 'A2': 0.1977,
                          'BetaL1': -0.0026, 'BetaL2': -0.957756, 'sigma': 0.0400692,
                          'G': 33.75, 'alpha_0': 0.05, 'delta': 0.0075,
                          'p': 1.5, 'I0': 9325.0}

        self.set_parameters(default_params)

    def get_parameters_array(self):
        """
            Return a numpy array of the models current parameters
        """
        return jnp.array([self.tau, self.Beta1, self.A1, self.A2, self.BetaL1, self.BetaL2, self.sigma, self.G, self.alpha_0, self.delta, self.p, self.I0])

    def set_parameters(self, param_dict: dict):
        """
            Update the model parameters using a passed in parameter dictionary. Any parameters not included
            in the dictionary will be set to the default values.
            updateParameters(param_dict)
            Returns null, changes the parameters stored in the class instance
        """

        params = ['tau', 'Beta1', 'A1', 'A2', 'BetaL1',
                  'BetaL2', 'sigma', 'G', 'alpha_0', 'delta', 'p', 'I0']

        for key, value in param_dict.items():
            setattr(self, key, value)

    @staticmethod
    @jit
    def spmodel(u: jnp.array, light: float, params: jnp.array):

        R, Psi, n = u

        tau, Beta1, A1, A2, BetaL1, BetaL2, sigma, G, alpha_0, delta, p, I0 = params
        K =  0.06358
        gamma = 0.024 
            

        alpha_0_func = alpha_0 * pow(light, p) / (pow(light, p) + I0)
        Bhat = G * (1.0 - n) * alpha_0_func
        LightAmp = A1 * 0.5 * Bhat * (1.0 - pow(R, 4.0)) * jnp.cos(Psi + BetaL1) + A2 * 0.5 * Bhat * R * (1.0 - pow(R, 8.0)) * jnp.cos(2.0 * Psi + BetaL2)
        LightPhase = sigma * Bhat - A1 * Bhat * 0.5 * (pow(R, 3.0) + 1.0 / R) * jnp.sin(
            Psi + BetaL1) - A2 * Bhat * 0.5 * (1.0 + pow(R, 8.0)) * jnp.sin(2.0 * Psi + BetaL2)

        dR = -1.0 * gamma * R + K * jnp.cos(Beta1) / 2.0 * R * (1.0 - pow(R, 4.0)) + LightAmp
        dPsi = 2 * jnp.pi / tau + K / 2.0 * jnp.sin(Beta1) * (1 + pow(R, 4.0)) + LightPhase
        dn = 60.0 * (alpha_0_func * (1.0 - n) - delta * n)

        du = jnp.array([dR, dPsi, dn])
        return (du)

    @staticmethod
    def spmodel_rk4_step(ustart: jnp.ndarray, light_val: float, params: jnp.array, dt: float):
        """
            Takes a single step forward using the default spmodel
        """

        k1 = HessianCircadianNewParams.spmodel(ustart, light_val, params)
        k2 = HessianCircadianNewParams.spmodel(ustart + dt / 2 * k1, light_val, params)
        k3 = HessianCircadianNewParams.spmodel(ustart + dt / 2 * k2, light_val, params)
        k4 = HessianCircadianNewParams.spmodel(ustart + dt * k3, light_val, params)
        return ustart + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4) # runge-kutta 

    @staticmethod
    def step_n(u0: jnp.ndarray, light: jnp.ndarray, params: jnp.ndarray, dt: float):
        for k in range(len(light)):
            u0 = HessianCircadianNewParams.spmodel_rk4_step(u0, light[k], params, dt)
        return u0 # only return final value

    @staticmethod
    @jit
    def norm(s1: jnp.ndarray, s2: jnp.ndarray):
        x1 = s1[0] * jnp.cos(s1[1])
        y1 = s1[0] * jnp.sin(s1[1])
        x2 = s2[0] * jnp.cos(s2[1])
        y2 = s2[0] * jnp.sin(s2[1])

        return (x1 - x2) ** 2 + (y1 - y2) ** 2
    
    @staticmethod
    def step_all_n(u0: jnp.ndarray, light: jnp.ndarray, params: jnp.ndarray, dt: float):
        u_all = jnp.zeros([len(light), 3])
        for k in range(len(light)):
            u0 = HessianCircadianNewParams.spmodel_rk4_step(u0, light[k], params, dt)
            u_all = u_all.at[k].set(u0)#u_all[k,] = u0
        return u_all # return all values
    
    @staticmethod
    def ics_individual_schedules(final_state_diff, convergence_val, ics, lights, params):
        u0 = ics
        count = 0
        while final_state_diff > convergence_val and count < 50:

            # simulate the model and extract the final time as the initial condition
            count = count + 1
            statesfinal = HessianCircadianNewParams.step_n(u0 = u0, light = lights, params = params, dt = 0.10) # final state value
            final_state_diff = abs(statesfinal[0] - u0[0]) + abs(np.mod(statesfinal[1] - u0[1] + np.pi,2*np.pi) - np.pi)
            #print(final_state_diff)
            u0 = statesfinal
        return u0