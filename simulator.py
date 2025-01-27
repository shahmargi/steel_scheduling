""" interfacing schedule object and algorithm with inspiration from gym """

"""
This file provides the scheduling environment class Env,
which can be used to load and simulate scheduling-problem instances.
"""
import random
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import gymnasium as gym
import hydra

from omegaconf import DictConfig, OmegaConf
from typing import List, Any

from steel_scheduling.utilities.equipment import ProductionEquipment
from steel_scheduling.utilities.equipment import TransferEquipment
from steel_scheduling.utilities.production_tasks import ProductionTask
from steel_scheduling.utilities.production_tasks import  TransferTask
from steel_scheduling.utilities.stages import Stage


class Steel_plant( ):

    def __init__(self, cfg: DictConfig):
        self.heats = [i for i in range(1, cfg.MILP_args.number_of_heats + 1)]                                           # number of heats - [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.num_heats = cfg.MILP_args.number_of_heats                                                                  # 24
        self.stages = cfg.MILP_args.number_of_stages                                                                    # number of stages - 3
        self.periods = cfg.MILP_args.number_of_hours * 60 / cfg.MILP_args.time_interval                                 # number of discrete periods - 96.0
        self.stage2units = {key: value for key, value in cfg.stage2units.items()}                                       # number of units in each stage - {'1': {'EAF': 2, 'TEAF': 1}, '2': {'AOD': 2, 'TAOD': 1}, '3': {'LF': 2}}
        self.num_of_production_tasks = cfg.MILP_args.number_of_heats                                                    # 24 production tasks to be scheduled
        self.num_of_transfer_tasks =   cfg.MILP_args.number_of_heats                                                    # 24 transfer tasks to be scheduled
        self.number_of_production_units = cfg.MILP_args.number_of_production_units                                      # Total 6 production units
        production_units = [n + 1 for n in range(self.number_of_production_units)]
        self.number_of_transfer_units = cfg.MILP_args.number_of_transfer_units                                          # Total 2 transfer units
        self.parameter=cfg.MILP_args.hyperparameter                                                                     # parameter to tune state space matrix
        self.allocation = 'electricity_cost_and_spinning_reserve_provision'                                                                            # reward definition
        self.processing_times_stage1 = [cfg.Equipment.EAF.processing_time.Heat1,cfg.Equipment.EAF.processing_time.Heat2,
                                        cfg.Equipment.EAF.processing_time.Heat3,cfg.Equipment.EAF.processing_time.Heat4,
                                        cfg.Equipment.EAF.processing_time.Heat5,cfg.Equipment.EAF.processing_time.Heat6,
                                        cfg.Equipment.EAF.processing_time.Heat7,cfg.Equipment.EAF.processing_time.Heat8,
                                        cfg.Equipment.EAF.processing_time.Heat9,cfg.Equipment.EAF.processing_time.Heat10,
                                        cfg.Equipment.EAF.processing_time.Heat11,cfg.Equipment.EAF.processing_time.Heat12,
                                        cfg.Equipment.EAF.processing_time.Heat13,cfg.Equipment.EAF.processing_time.Heat14,
                                        cfg.Equipment.EAF.processing_time.Heat15,cfg.Equipment.EAF.processing_time.Heat16,
                                        cfg.Equipment.EAF.processing_time.Heat17,cfg.Equipment.EAF.processing_time.Heat18,
                                        cfg.Equipment.EAF.processing_time.Heat19,cfg.Equipment.EAF.processing_time.Heat20,
                                        cfg.Equipment.EAF.processing_time.Heat21,cfg.Equipment.EAF.processing_time.Heat22,
                                        cfg.Equipment.EAF.processing_time.Heat23,cfg.Equipment.EAF.processing_time.Heat24
                                        ]
        self.processing_times_stage2 = [cfg.Equipment.AOD.processing_time.Heat1,cfg.Equipment.AOD.processing_time.Heat2,
                                        cfg.Equipment.AOD.processing_time.Heat3,cfg.Equipment.AOD.processing_time.Heat4,
                                        cfg.Equipment.AOD.processing_time.Heat5,cfg.Equipment.AOD.processing_time.Heat6,
                                        cfg.Equipment.AOD.processing_time.Heat7,cfg.Equipment.AOD.processing_time.Heat8,
                                        cfg.Equipment.AOD.processing_time.Heat9,cfg.Equipment.AOD.processing_time.Heat10,
                                        cfg.Equipment.AOD.processing_time.Heat11,cfg.Equipment.AOD.processing_time.Heat12,
                                        cfg.Equipment.AOD.processing_time.Heat13,cfg.Equipment.AOD.processing_time.Heat14,
                                        cfg.Equipment.AOD.processing_time.Heat15,cfg.Equipment.AOD.processing_time.Heat16,
                                        cfg.Equipment.AOD.processing_time.Heat17,cfg.Equipment.AOD.processing_time.Heat18,
                                        cfg.Equipment.AOD.processing_time.Heat19,cfg.Equipment.AOD.processing_time.Heat20,
                                        cfg.Equipment.AOD.processing_time.Heat21,cfg.Equipment.AOD.processing_time.Heat22,
                                        cfg.Equipment.AOD.processing_time.Heat23,cfg.Equipment.AOD.processing_time.Heat24
                                        ]

        self.processing_times_stage3 = [cfg.Equipment.LF.processing_time.Heat1, cfg.Equipment.LF.processing_time.Heat2,
                                        cfg.Equipment.LF.processing_time.Heat3, cfg.Equipment.LF.processing_time.Heat4,
                                        cfg.Equipment.LF.processing_time.Heat5, cfg.Equipment.LF.processing_time.Heat6,
                                        cfg.Equipment.LF.processing_time.Heat7, cfg.Equipment.LF.processing_time.Heat8,
                                        cfg.Equipment.LF.processing_time.Heat9, cfg.Equipment.LF.processing_time.Heat10,
                                        cfg.Equipment.LF.processing_time.Heat11,cfg.Equipment.LF.processing_time.Heat12,
                                        cfg.Equipment.LF.processing_time.Heat13,cfg.Equipment.LF.processing_time.Heat14,
                                        cfg.Equipment.LF.processing_time.Heat15,cfg.Equipment.LF.processing_time.Heat16,
                                        cfg.Equipment.LF.processing_time.Heat17,cfg.Equipment.LF.processing_time.Heat18,
                                        cfg.Equipment.LF.processing_time.Heat19,cfg.Equipment.LF.processing_time.Heat20,
                                        cfg.Equipment.LF.processing_time.Heat21,cfg.Equipment.LF.processing_time.Heat22,
                                        cfg.Equipment.LF.processing_time.Heat23,cfg.Equipment.LF.processing_time.Heat24
                                        ]
        self.duration_stage1 = [cfg.Equipment.EAF.duration.Heat1, cfg.Equipment.EAF.duration.Heat2,
                                cfg.Equipment.EAF.duration.Heat3, cfg.Equipment.EAF.duration.Heat4,
                                cfg.Equipment.EAF.duration.Heat5, cfg.Equipment.EAF.duration.Heat6,
                                cfg.Equipment.EAF.duration.Heat7, cfg.Equipment.EAF.duration.Heat8,
                                cfg.Equipment.EAF.duration.Heat9, cfg.Equipment.EAF.duration.Heat10,
                                cfg.Equipment.EAF.duration.Heat11, cfg.Equipment.EAF.duration.Heat12,
                                cfg.Equipment.EAF.duration.Heat13, cfg.Equipment.EAF.duration.Heat14,
                                cfg.Equipment.EAF.duration.Heat15, cfg.Equipment.EAF.duration.Heat16,
                                cfg.Equipment.EAF.duration.Heat17, cfg.Equipment.EAF.duration.Heat18,
                                cfg.Equipment.EAF.duration.Heat19, cfg.Equipment.EAF.duration.Heat20,
                                cfg.Equipment.EAF.duration.Heat21, cfg.Equipment.EAF.duration.Heat22,
                                cfg.Equipment.EAF.duration.Heat23, cfg.Equipment.EAF.duration.Heat24
                                ]
        self.duration_stage2 = [cfg.Equipment.AOD.duration.Heat1, cfg.Equipment.AOD.duration.Heat2,
                                cfg.Equipment.AOD.duration.Heat3, cfg.Equipment.AOD.duration.Heat4,
                                cfg.Equipment.AOD.duration.Heat5, cfg.Equipment.AOD.duration.Heat6,
                                cfg.Equipment.AOD.duration.Heat7, cfg.Equipment.AOD.duration.Heat8,
                                cfg.Equipment.AOD.duration.Heat9, cfg.Equipment.AOD.duration.Heat10,
                                cfg.Equipment.AOD.duration.Heat11, cfg.Equipment.AOD.duration.Heat12,
                                cfg.Equipment.AOD.duration.Heat13, cfg.Equipment.AOD.duration.Heat14,
                                cfg.Equipment.AOD.duration.Heat15, cfg.Equipment.AOD.duration.Heat16,
                                cfg.Equipment.AOD.duration.Heat17, cfg.Equipment.AOD.duration.Heat18,
                                cfg.Equipment.AOD.duration.Heat19, cfg.Equipment.AOD.duration.Heat20,
                                cfg.Equipment.AOD.duration.Heat21, cfg.Equipment.AOD.duration.Heat22,
                                cfg.Equipment.AOD.duration.Heat23, cfg.Equipment.AOD.duration.Heat24
                                ]
        self.duration_stage3 = [cfg.Equipment.LF.duration.Heat1, cfg.Equipment.LF.duration.Heat2,
                                cfg.Equipment.LF.duration.Heat3, cfg.Equipment.LF.duration.Heat4,
                                cfg.Equipment.LF.duration.Heat5, cfg.Equipment.LF.duration.Heat6,
                                cfg.Equipment.LF.duration.Heat7, cfg.Equipment.LF.duration.Heat8,
                                cfg.Equipment.LF.duration.Heat9, cfg.Equipment.LF.duration.Heat10,
                                cfg.Equipment.LF.duration.Heat11, cfg.Equipment.LF.duration.Heat12,
                                cfg.Equipment.LF.duration.Heat13, cfg.Equipment.LF.duration.Heat14,
                                cfg.Equipment.LF.duration.Heat15, cfg.Equipment.LF.duration.Heat16,
                                cfg.Equipment.LF.duration.Heat17, cfg.Equipment.LF.duration.Heat18,
                                cfg.Equipment.LF.duration.Heat19, cfg.Equipment.LF.duration.Heat20,
                                cfg.Equipment.LF.duration.Heat21, cfg.Equipment.LF.duration.Heat22,
                                cfg.Equipment.LF.duration.Heat23, cfg.Equipment.LF.duration.Heat24
                                ]

        self.min_transfer_times_stage1 = [cfg.TEquipment.TEAF.min_transfer_time.Heat1,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat2,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat3,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat4,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat5,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat6,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat7,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat8,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat9,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat10,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat11,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat12,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat13,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat14,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat15,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat16,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat17,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat18,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat19,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat20,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat21,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat22,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat23,
                                          cfg.TEquipment.TEAF.min_transfer_time.Heat24
                                          ]

        self.max_transfer_times_stage1 = [cfg.TEquipment.TEAF.max_transfer_time.Heat1,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat2,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat3,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat4,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat5,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat6,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat7,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat8,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat9,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat10,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat11,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat12,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat13,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat14,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat15,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat16,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat17,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat18,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat19,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat20,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat21,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat22,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat23,
                                          cfg.TEquipment.TEAF.max_transfer_time.Heat24
                                          ]

        self.min_transfer_times_stage2 = [cfg.TEquipment.TAOD.min_transfer_time.Heat1,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat2,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat3,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat4,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat5,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat6,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat7,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat8,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat9,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat10,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat11,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat12,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat13,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat14,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat15,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat16,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat17,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat18,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat19,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat20,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat21,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat22,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat23,
                                          cfg.TEquipment.TAOD.min_transfer_time.Heat24
                                          ]

        self.max_transfer_times_stage2 = [cfg.TEquipment.TAOD.max_transfer_time.Heat1,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat2,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat3,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat4,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat5,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat6,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat7,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat8,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat9,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat10,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat11,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat12,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat13,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat14,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat15,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat16,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat17,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat18,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat19,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat20,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat21,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat22,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat23,
                                          cfg.TEquipment.TAOD.max_transfer_time.Heat24
                                          ]

############################################# determininstic case ######################################################
        ############################################# electricity price  ###############################################

        energy_price = [33.28, 30.00, 29.15, 28.49, 34.66, 50.01,
                        71.52, 77.94, 81.97, 87.90, 92.76, 94.98,
                        92.31, 90.03, 90.09, 87.44, 85.37, 79.97,
                        79.92, 77.83, 76.28, 65.06, 53.07, 34.16]

        self.electricity_prices = []

        for price in energy_price:
            self.electricity_prices.extend([(price)] * int(60 / 15))


        ################################### SP Prices ##################################################################
        sp_price = [3.28, 3.00, 2.15, 2.49, 3.66, 5.01,
                    7.52, 7.94, 8.97, 8.90, 9.76, 9.98,
                    9.31, 9.03, 9.09, 8.44, 8.37, 7.97,
                    7.92, 7.83, 7.28, 6.06, 5.07, 3.16]

        self.sp_prices = []

        for price in sp_price:
            self.sp_prices.extend([(price)] * int(60 / 15))

        ####################################### generation #############################################################

        self.generation = [7.623491061, 4.517699791, 3.555646106, 2.48438853, 0.391585596, 1.421687974, 1.860133449,
                           0.686599093,
                           2.214638619, 2.942800387, 3.988794377, 6.969082667, 9.23995314, 10.64126725, 10.46401467,
                           10.11317679,
                           10.25090409, 9.420465543, 8.118983344, 8.649518668, 8.125910457, 8.615698059,
                           10.46931187,
                           14.98334437,
                           19.50308155, 17.39479448, 14.5269699, 12.61508684, 14.02984771, 15.76651556, 17.33734019,
                           17.38501503,
                           19.98105231, 21.98991494, 27.85554933, 35.53486477, 36.97855651, 37.74094637, 40,
                           39.9535476,
                           39.66668364, 37.53965263, 29.76783986, 23.27020832, 17.78026792, 14.89940406,
                           11.82376611,
                           10.21626853,
                           9.290887791, 7.289359751, 5.651708858, 6.812203942, 8.832883411, 10.47705394,
                           12.23491061,
                           11.11801559,
                           11.76182957, 14.3696837, 12.55396526, 11.41751133, 11.80461468, 13.22385779, 12.81026843,
                           12.90847043,
                           10.34095655, 9.97952427, 8.952681709, 10.87271431, 8.957571436, 8.596954108, 8.731014109,
                           9.608720012,
                           11.57846483, 12.4961035, 13.05271736, 11.58457699, 10.10910202, 10.91916671, 11.02225844,
                           10.63556257,
                           8.29908827, 7.467427291, 6.326491112, 5.687566852, 4.413793103, 6.339530383, 5.93286813,
                           2.206081597,
                           1.344674782, 2.710130902, 2.766362757, 2.951764886, 3.475373096, 1.560230225,
                           0.514236235, 0]


        ################################### Mean and std of Electricity prices ##########################################
        self.mean_price = np.mean(self.electricity_prices)
        self.std_dev_price = np.std(self.electricity_prices)

        ################################### Mean and std of Generation  ################################################
        self.mean_generation = np.mean(self.generation)
        self.std_dev_generation = np.std(self.generation)

        ################################### Mean and std of SP price  ##################################################
        self.mean_sp_prices = np.mean( self.sp_prices)
        self.std_dev_sp_prices = np.std( self.sp_prices)



        self.total_costs = []                                                                                           # to store total stage cost at each time period

        self.aux = []                                                                                                   # EnergyResource -E[wind]
        self.maxaux = []                                                                                                # max(0,aux)
        self.provision = [ ]
        self.total_cost_electricity = []
        self.total_cost_spinning_reserve = []

        ####################### Initialising stage object and production equipment objects #############################
        ##################################### Stage I ##################################################################

        self.stage1_production_equipments = []
        for equipment_id, norm_mw in cfg.stage1.production_equipment.items():
            production_equipment = ProductionEquipment(equipment_id, norm_mw, cfg.stage1.id, None)
            self.stage1_production_equipments.append(production_equipment)

        self.stage1_transfer_equipments = []
        transfer_equipment = TransferEquipment(cfg.stage1.transfer_equipment.id, cfg.stage1.id, None)
        self.stage1_transfer_equipments.append(transfer_equipment)

        self.stage1 = Stage(cfg.stage1.id, self.stage1_production_equipments, self.stage1_transfer_equipments)

        ##################################### Stage II #################################################################

        self.stage2_production_equipments = []
        for equipment_id, norm_mw in cfg.stage2.production_equipment.items():
            production_equipment = ProductionEquipment(equipment_id, norm_mw, cfg.stage2.id, None)
            self.stage2_production_equipments.append(production_equipment)

        self.stage2_transfer_equipments = []
        transfer_equipment = TransferEquipment(cfg.stage2.transfer_equipment.id, cfg.stage2.id, None)
        self.stage2_transfer_equipments.append(transfer_equipment)


        self.stage2 = Stage(cfg.stage2.id,self.stage2_production_equipments,self.stage2_transfer_equipments)

        ##################################### Stage III ################################################################

        self.stage3_production_equipments = []

        for equipment_id, norm_mw in cfg.stage3.production_equipment.items():
            production_equipment = ProductionEquipment(equipment_id, norm_mw, cfg.stage3.id, None)
            self.stage3_production_equipments.append(production_equipment)

        self.stage3 = Stage(cfg.stage3.id, self.stage3_production_equipments, None)



        ################################Initialising state space matrix for each stage #################################

        self.state_space_stage1_pt =  np.full((cfg.MILP_args.hyperparameter, 5), -1)
        self.state_space_stage2_pt = np.full((cfg.MILP_args.hyperparameter, 5), -1)
        self.state_space_stage3_pt = np.full((cfg.MILP_args.hyperparameter, 5), -1)



    ##################################################### reset method #################################################


    def reset(self):


        # resets episode counters and infos
        self.period = 0                                                                                                 # reset no of steps taken in current episode


        # create a copy of all the initialised objects
        self.stage1_production_equipments_copy = copy.deepcopy(self.stage1_production_equipments)
        self.stage1_transfer_equipments_copy = copy.deepcopy(self.stage1_transfer_equipments)
        self.stage1_copy = copy.deepcopy(self.stage1)


        self.stage2_production_equipments_copy = copy.deepcopy(self.stage2_production_equipments)
        self.stage2_transfer_equipments_copy = copy.deepcopy(self.stage2_transfer_equipments)
        self.stage2_copy = copy.deepcopy(self.stage2)

        self.stage3_production_equipments_copy = copy.deepcopy(self.stage3_production_equipments)
        self.stage3_copy = copy.deepcopy(self.stage3)


        #set state
        self.observe_state()
        self.dynamic_restrict_control()

        state = np.hstack([self.state_space_stage1_pt, self.state_space_stage2_pt,self.state_space_stage3_pt,self.price_info,self.sp_price_info,self.generation_info])
        control_mask = self.feasible_control_set

        return state,control_mask

    def observe_state(self):

        t = self.period

        ################### Price and generation data to be fed in form of state space #################################

        self.price_vector = np.array(self.electricity_prices)
        self.sp_price_vector = np.array(self.sp_prices)
        self.generation_vector = np.array(self.generation)

        if t == 0:  # if t== 0, state space is similar to initialised state space
            self.state_space_stage1_pt = copy.deepcopy(self.state_space_stage1_pt)
            self.state_space_stage2_pt = copy.deepcopy(self.state_space_stage2_pt)
            self.state_space_stage3_pt = copy.deepcopy(self.state_space_stage3_pt)

            self.price_info = np.array(
                [self.price_vector[t + 3], self.price_vector[t + 2], self.price_vector[t + 1], self.price_vector[t]],
                dtype=int)
            self.generation_info = np.array(
                [self.generation_vector[t + 3], self.generation_vector[t + 2], self.generation_vector[t + 1],
                 self.generation_vector[t]], dtype=int)
            self.sp_price_info = np.array(
                [self.sp_price_vector[t + 3], self.sp_price_vector[t + 2], self.sp_price_vector[t + 1],
                 self.sp_price_vector[t]], dtype=int)

            self.price_info = self.price_info.reshape(-1, 1)
            self.sp_price_info = self.sp_price_info.reshape(-1, 1)
            self.generation_info = self.generation_info.reshape(-1, 1)


        elif t == 95:

            self.price_info = np.array([-1, -1, -1, self.price_vector[95]], dtype=int)
            self.generation_info = np.array([-1, -1, -1, self.generation_vector[95]], dtype=int)
            self.sp_price_info = np.array([-1, -1, -1, self.sp_price_vector[95]], dtype=int)

            self.price_info = self.price_info.reshape(-1, 1)
            self.sp_price_info = self.sp_price_info.reshape(-1, 1)
            self.generation_info = self.generation_info.reshape(-1, 1)

            ##################################### State space ##########################################################

            ##################################### Production task matrix ###############################################

            ##################################### Stage I ##############################################################

            all_stage1_production_tasks = []  # However if t > 0; it sees what are the tasks inside the equipment by running step method & gives the 4 most recent tasks from the equipment
            for equipment in self.stage1_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage1_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage1_pt = copy.deepcopy(self.state_space_stage1_pt)

            sorted_tasks = sorted(all_stage1_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]
            for i, task in enumerate(recent_tasks):
                self.state_space_stage1_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,

                                                    task.production_equipment]

            ##################################### Stage II #############################################################

            all_stage2_production_tasks = []
            for equipment in self.stage2_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage2_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage2_pt = copy.deepcopy(self.state_space_stage2_pt)

            sorted_tasks = sorted(all_stage2_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]

            # Create the state_space_stage2_pt array
            for i, task in enumerate(recent_tasks):
                self.state_space_stage2_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage III ############################################################

            all_stage3_production_tasks = []
            for equipment in self.stage3_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage3_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage3_pt = copy.deepcopy(self.state_space_stage3_pt)

            sorted_tasks = sorted(all_stage3_production_tasks, key=lambda task: task.start_time, reverse=True)
            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]
            # Create the state_space_stage3_pt array

            for i, task in enumerate(recent_tasks):
                self.state_space_stage3_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

        elif t == 96:

            self.price_info = np.array([-1, -1, -1, -1], dtype=int)
            self.generation_info = np.array([-1, -1, -1, -1], dtype=int)
            self.sp_price_info = np.array([-1, -1, -1, -1], dtype=int)

            self.price_info = self.price_info.reshape(-1, 1)
            self.sp_price_info = self.sp_price_info.reshape(-1, 1)
            self.generation_info = self.generation_info.reshape(-1, 1)

            ##################################### State space ##########################################################

            ##################################### Production task matrix ###############################################

            ##################################### Stage I ##############################################################

            all_stage1_production_tasks = []
            for equipment in self.stage1_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage1_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage1_pt = copy.deepcopy(self.state_space_stage1_pt)

            sorted_tasks = sorted(all_stage1_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]

            for i, task in enumerate(recent_tasks):
                self.state_space_stage1_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage II #############################################################

            all_stage2_production_tasks = []
            for equipment in self.stage2_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage2_production_tasks.extend(equipment.production_tasks)

                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage2_pt = copy.deepcopy(self.state_space_stage2_pt)

            sorted_tasks = sorted(all_stage2_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]
            # Create the state_space_stage2_pt array

            for i, task in enumerate(recent_tasks):
                self.state_space_stage2_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage III ############################################################

            all_stage3_production_tasks = []
            for equipment in self.stage3_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage3_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage3_pt = copy.deepcopy(self.state_space_stage3_pt)

            sorted_tasks = sorted(all_stage3_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]

            # Create the state_space_stage3_pt array
            for i, task in enumerate(recent_tasks):
                self.state_space_stage3_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,

                                                    task.production_equipment]


        elif t == 93:

            self.price_info = np.array([-1, self.price_vector[t + 2], self.price_vector[t + 1], self.price_vector[t]],
                                       dtype=int)
            self.generation_info = np.array(
                [-1, self.generation_vector[t + 2], self.generation_vector[t + 1], self.generation_vector[t]],
                dtype=int)
            self.sp_price_info = np.array(
                [-1, self.sp_price_vector[t + 2], self.sp_price_vector[t + 1], self.sp_price_vector[t]], dtype=int)

            self.price_info = self.price_info.reshape(-1, 1)
            self.generation_info = self.generation_info.reshape(-1, 1)
            self.sp_price_info = self.sp_price_info.reshape(-1, 1)

            ##################################### State space ##########################################################

            ##################################### Production task matrix ###############################################

            ##################################### Stage I ##############################################################

            all_stage1_production_tasks = []
            for equipment in self.stage1_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage1_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage1_pt = copy.deepcopy(self.state_space_stage1_pt)

            sorted_tasks = sorted(all_stage1_production_tasks, key=lambda task: task.start_time, reverse=True)
            # Take the four most recent tasks

            recent_tasks = sorted_tasks[: self.parameter]
            for i, task in enumerate(recent_tasks):
                self.state_space_stage1_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage II #############################################################

            all_stage2_production_tasks = []
            for equipment in self.stage2_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage2_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage2_pt = copy.deepcopy(self.state_space_stage2_pt)

            sorted_tasks = sorted(all_stage2_production_tasks, key=lambda task: task.start_time, reverse=True)
            # Take the four most recent tasks

            recent_tasks = sorted_tasks[: self.parameter]
            # Create the state_space_stage2_pt array
            for i, task in enumerate(recent_tasks):
                self.state_space_stage2_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage III ############################################################

            all_stage3_production_tasks = []
            for equipment in self.stage3_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage3_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage3_pt = copy.deepcopy(self.state_space_stage3_pt)

            sorted_tasks = sorted(all_stage3_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]
            # Create the state_space_stage3_pt array
            for i, task in enumerate(recent_tasks):
                self.state_space_stage3_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,

                                                    task.production_equipment]

        elif t == 94:

            self.price_info = np.array([-1, -1, self.price_vector[t + 1], self.price_vector[t]], dtype=int)
            self.generation_info = np.array([-1, -1, self.generation_vector[t + 1], self.generation_vector[t]],
                                            dtype=int)
            self.sp_price_info = np.array([-1, -1, self.sp_price_vector[t + 1], self.sp_price_vector[t]], dtype=int)

            self.price_info = self.price_info.reshape(-1, 1)
            self.generation_info = self.generation_info.reshape(-1, 1)
            self.sp_price_info = self.sp_price_info.reshape(-1, 1)

            ##################################### State space ##########################################################

            ##################################### Production task matrix ###############################################

            ##################################### Stage I ##############################################################

            all_stage1_production_tasks = []
            for equipment in self.stage1_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage1_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage1_pt = copy.deepcopy(self.state_space_stage1_pt)

            sorted_tasks = sorted(all_stage1_production_tasks, key=lambda task: task.start_time, reverse=True)
            # Take the four most recent tasks

            recent_tasks = sorted_tasks[: self.parameter]
            for i, task in enumerate(recent_tasks):
                self.state_space_stage1_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage II #############################################################

            all_stage2_production_tasks = []
            for equipment in self.stage2_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage2_production_tasks.extend(equipment.production_tasks)
                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage2_pt = copy.deepcopy(self.state_space_stage2_pt)

            sorted_tasks = sorted(all_stage2_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]

            # Create the state_space_stage2_pt array
            for i, task in enumerate(recent_tasks):
                self.state_space_stage2_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage III ############################################################

            all_stage3_production_tasks = []
            for equipment in self.stage3_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage3_production_tasks.extend(equipment.production_tasks)
                else:

                    # Handle the case when production_tasks is None
                    self.state_space_stage3_pt = copy.deepcopy(self.state_space_stage3_pt)

            sorted_tasks = sorted(all_stage3_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks

            recent_tasks = sorted_tasks[: self.parameter]
            # Create the state_space_stage3_pt array

            for i, task in enumerate(recent_tasks):
                self.state_space_stage3_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]



        else:

            self.price_info = np.array(
                [self.price_vector[t + 3], self.price_vector[t + 2], self.price_vector[t + 1], self.price_vector[t]],
                dtype=int)

            self.generation_info = np.array(
                [self.generation_vector[t + 3], self.generation_vector[t + 2], self.generation_vector[t + 1],
                 self.generation_vector[t]], dtype=int)

            self.sp_price_info = np.array(
                [self.sp_price_vector[t + 3], self.sp_price_vector[t + 2], self.sp_price_vector[t + 1],
                 self.sp_price_vector[t]], dtype=int)

            self.price_info = self.price_info.reshape(-1, 1)
            self.generation_info = self.generation_info.reshape(-1, 1)
            self.sp_price_info = self.sp_price_info.reshape(-1, 1)

            ##################################### State space ##########################################################

            ##################################### Production task matrix ###############################################

            ##################################### Stage I ##############################################################

            all_stage1_production_tasks = []
            for equipment in self.stage1_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage1_production_tasks.extend(equipment.production_tasks)

                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage1_pt = copy.deepcopy(self.state_space_stage1_pt)

            sorted_tasks = sorted(all_stage1_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks

            recent_tasks = sorted_tasks[: self.parameter]
            for i, task in enumerate(recent_tasks):
                self.state_space_stage1_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage II #############################################################

            all_stage2_production_tasks = []
            for equipment in self.stage2_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage2_production_tasks.extend(equipment.production_tasks)

                else:
                    # Handle the case when production_tasks is None
                    self.state_space_stage2_pt = copy.deepcopy(self.state_space_stage2_pt)

            sorted_tasks = sorted(all_stage2_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]

            # Create the state_space_stage2_pt array
            for i, task in enumerate(recent_tasks):
                self.state_space_stage2_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

            ##################################### Stage III ############################################################

            all_stage3_production_tasks = []

            for equipment in self.stage3_copy.production_equipment:
                if equipment.production_tasks is not None:
                    all_stage3_production_tasks.extend(equipment.production_tasks)

                else:

                    # Handle the case when production_tasks is None
                    self.state_space_stage3_pt = copy.deepcopy(self.state_space_stage3_pt)

            sorted_tasks = sorted(all_stage3_production_tasks, key=lambda task: task.start_time, reverse=True)

            # Take the four most recent tasks
            recent_tasks = sorted_tasks[: self.parameter]
            # Create the state_space_stage3_pt array

            for i, task in enumerate(recent_tasks):
                self.state_space_stage3_pt[i, :] = [task.id, task.processingtime, task.start_time, task.end_time,
                                                    task.production_equipment]

        return  np.hstack([self.state_space_stage1_pt, self.state_space_stage2_pt,self.state_space_stage3_pt,self.price_info,self.sp_price_info,self.generation_info])

    ########### MIP #########################################################

    # def dynamic_restrict_control(self):  # to restrict the feasible controls
    #
    #     t = self.period
    #     feasible_control_set = {}
    #
    #     try:
    #         file_path = r'C:\Users\c21084647\OneDrive - Cardiff University\cardiff\Year II\Research Objective II\MIP_Data.xlsx'
    #
    #         df = pd.read_excel(file_path, sheet_name='Sheet3', header=0)
    #
    #         if f't={t}' not in df.columns:
    #             raise ValueError(f"Column 't={t}' not found in the Excel sheet.")
    #
    #         # Get the 't' column as a list for the specified units (i)
    #         feasible_control_set = {
    #             i: [int(df.loc[i, f't={t}'])] for i in range(self.number_of_production_units)
    #         }
    #
    #     except Exception as e:
    #         print("Error reading the Excel file:", str(e))
    #
    #     self.feasible_control_set = feasible_control_set.copy()
    #     return self.feasible_control_set

    def dynamic_restrict_control(self):                                                                                 # to restrict the feasible controls

        t = self.period
        feasible_control_set = {}

        if t == 0:                                                                                                      # At t = 0, in stage 1, i.e in EAF1 and EAF2, all heats are feasible, meaning from 0 to 24
            for i in range(self.number_of_production_units):
                if i in [0, 1]:                                                                                         # i = 0 means EAF1, i=1 means, EAF2
                    feasible_control_set[i] = list( range(self.num_heats + 1))                                          # all heats are feasible, meaning from 0 to 24
                else:                                                                                                   # If i is  2,3,4,5 i.e stage 2 (AOD1 and AOD2) and stage 3(LF1 and LF2)
                    feasible_control_set[i] = [0]                                                                       # no heats are feasible, hence kept 0


        elif t == 1:                                                                                                     # If t > 0:

            for stage in range(1, self.stages + 1):                                                                     # Iterating for all stages

                if stage == 1:                                                                                          # stage 1
                    for equipment in self.stage1_copy.production_equipment:                                             # Iterate over each equipment
                        if equipment.id in self.stage1_copy.feasible_production_equipment_in_stage(t):                  # It checks if the equipment is feasible at time step t based on the feasible_production_equipment_in_stage method.
                            if self.stage1_copy.assigned_production_tasks() == []:                                      # If it's feasible, If there are no assigned production tasks (by checking assigned_production_tasks method)
                                feasible_control_set[equipment.id - 1] = list(range(self.num_heats + 1))                # feasible_control_set[0]-EAF1/feasible_control_set[1]-EAF2 = All heats are feasible for that equipment.
                            else:
                                feasible_control_set[equipment.id - 1] = [num for num in range(self.num_heats + 1) if
                                                                          num not in                                    # feasible_control_set[0]-EAF1/feasible_control_set[1]-EAF2 =Otherwise, the feasible control set is determined by removing the assigned production tasks from the range of all heats.
                                                                          self.stage1_copy.assigned_production_tasks()]
                        else:                                                                                           # If equipment is not feasible
                            feasible_control_set[equipment.id - 1] = [task.id for task in equipment.production_tasks    # feasible_control_set[0]-EAF1/feasible_control_set[1]-EAF2 = There is a task inside equipment which is ongoing, so feasible control will only be that task .
                                                                      if t <= task.end_time]

                if stage == 2:                                                                                          # feasible_control_set[2]-AOD1(id=1)/feasible_control_set[3]-AOD2(id=2)

                    for equipment in self.stage2_copy.production_equipment:                                             # Iterate over each equipment
                        if equipment.id in self.stage2_copy.feasible_production_equipment_in_stage(t):                  # It checks if the equipment is feasible at time step t based on the feasible_production_equipment_in_stage method.
                            if self.stage2_copy.feasible_production_tasks_in_stage(t,self.stage1_copy) == []:           # if there are no feasible_production_tasks_in_stage
                                feasible_control_set[equipment.id + 1] = [0]                                            # feasible control is 0
                            else:
                                feasible_control_set[equipment.id + 1] = \
                                    [0] + self.stage2_copy.feasible_production_tasks_in_stage(t,self.stage1_copy)       # else control is the tasks which are inside feasible_production_tasks_in_stage
                        else:                                                                                           # if equipment is not feasible
                            feasible_control_set[equipment.id + 1] = [task.id for task in equipment.production_tasks
                                                                      if t <= task.end_time]

                if stage == 3:                                                                                          # feasible_control_set[4]-AOD1(id=1)/feasible_control_set[5]-AOD2(id=2)

                    for equipment in self.stage3_copy.production_equipment:                                             # Iterate over each equipment
                        if equipment.id in self.stage3_copy.feasible_production_equipment_in_stage(t):                  # It checks if the equipment is feasible at time step t based on the feasible_production_equipment_in_stage method.
                            if self.stage3_copy.feasible_production_tasks_in_stage(t,self.stage2_copy) == []:           # if there are no feasible_production_tasks_in_stage
                                feasible_control_set[equipment.id + 3] = [0]                                            # feasible control is 0
                            else:
                                feasible_control_set[equipment.id + 3] = \
                                    [0] + self.stage3_copy.feasible_production_tasks_in_stage(t,self.stage2_copy)       # else control is the tasks which are inside feasible_production_tasks_in_stage
                        else:                                                                                           # if equipment is not feasible
                            feasible_control_set[equipment.id + 3] = [task.id for task in equipment.production_tasks
                                                                      if t <= task.end_time]

        elif t >= 2:                                                                                                     # If t > 0:

            for stage in range(1, self.stages + 1):                                                                     # Iterating for all stages

                if stage == 1:                                                                                          # stage 1
                    for equipment in self.stage1_copy.production_equipment:                                             # Iterate over each equipment
                        if equipment.id in self.stage1_copy.feasible_production_equipment_in_stage(t):                  # It checks if the equipment is feasible at time step t based on the feasible_production_equipment_in_stage method.
                            if self.stage1_copy.assigned_production_tasks() == []:                                      # If it's feasible, If there are no assigned production tasks (by checking assigned_production_tasks method)
                                feasible_control_set[equipment.id - 1] = list(range(self.num_heats + 1))                # feasible_control_set[0]-EAF1/feasible_control_set[1]-EAF2 = All heats are feasible for that equipment.
                            else:
                                feasible_control_set[equipment.id - 1] = [num for num in range(self.num_heats + 1) if
                                                                          num not in                                    # feasible_control_set[0]-EAF1/feasible_control_set[1]-EAF2 =Otherwise, the feasible control set is determined by removing the assigned production tasks from the range of all heats.
                                                                          self.stage1_copy.assigned_production_tasks()]
                        else:                                                                                           # If equipment is not feasible
                            feasible_control_set[equipment.id - 1] = [task.id for task in equipment.production_tasks    # feasible_control_set[0]-EAF1/feasible_control_set[1]-EAF2 = There is a task inside equipment which is ongoing, so feasible control will only be that task .
                                                                      if t <= task.end_time]

                if stage == 2:                                                                                          # feasible_control_set[2]-AOD1(id=1)/feasible_control_set[3]-AOD2(id=2)

                    for equipment in self.stage2_copy.production_equipment:                                             # Iterate over each equipment
                        if equipment.id in self.stage2_copy.feasible_production_equipment_in_stage(t):                  # It checks if the equipment is feasible at time step t based on the feasible_production_equipment_in_stage method.
                            if self.stage2_copy.feasible_production_tasks_in_stage(t,self.stage1_copy) == []:           # if there are no feasible_production_tasks_in_stage
                                feasible_control_set[equipment.id + 1] = [0]                                            # feasible control is 0
                            else:
                                feasible_control_set[equipment.id + 1] = \
                                    [0] + self.stage2_copy.feasible_production_tasks_in_stage(t,self.stage1_copy)       # else control is the tasks which are inside feasible_production_tasks_in_stage
                        else:                                                                                           # if equipment is not feasible
                            feasible_control_set[equipment.id + 1] = [task.id for task in equipment.production_tasks
                                                                      if t <= task.end_time]

                if stage == 3:                                                                                          # feasible_control_set[4]-AOD1(id=1)/feasible_control_set[5]-AOD2(id=2)

                    for equipment in self.stage3_copy.production_equipment:                                             # Iterate over each equipment
                        if equipment.id in self.stage3_copy.feasible_production_equipment_in_stage(t):                  # It checks if the equipment is feasible at time step t based on the feasible_production_equipment_in_stage method.
                            if self.stage3_copy.feasible_production_tasks_in_stage(t,self.stage2_copy) == []:           # if there are no feasible_production_tasks_in_stage
                                feasible_control_set[equipment.id + 3] = [0]                                            # feasible control is 0
                            else:
                                feasible_control_set[equipment.id + 3] = \
                                    [0] + self.stage3_copy.feasible_production_tasks_in_stage(t,self.stage2_copy)       # else control is the tasks which are inside feasible_production_tasks_in_stage
                        else:                                                                                           # if equipment is not feasible
                            feasible_control_set[equipment.id + 3] = [task.id for task in equipment.production_tasks
                                                                      if t <= task.end_time]

        self.feasible_control_set = feasible_control_set.copy()

        return self.feasible_control_set


    def assign_transfer_task(self):

        for stage in range(1, self.stages):

            t = self.period

            if stage == 1:
                if self.stage1_copy.feasible_transfer_tasks_in_stage(t) and self.stage1_copy.feasible_transfer_equipment_in_stage(t):

                    if self.stage1_copy.transfer_equipment[0].transfer_tasks is None:

                        self.stage1_copy.transfer_equipment[0].transfer_tasks = []

                        for id in self.stage1_copy.feasible_transfer_tasks_in_stage(t):
                            new_task = TransferTask(id,
                                                    self.min_transfer_times_stage1[id - 1],
                                                    self.max_transfer_times_stage1[id - 1],
                                                    self.period + 1,
                                                    (self.period + 1) + (self.min_transfer_times_stage1[id - 1] - 1),
                                                    self.stage1_copy.transfer_equipment[0],
                                                    self.stage1_copy.id)

                            self.stage1_copy.transfer_equipment[0].transfer_tasks.append(new_task)
                    else:
                        is_recent_task = False                                                                          # Checking is there any recent task, meaning, is there a task inside which is ongoing, meaning, is this a same control as previous
                        for task in self.stage1_copy.transfer_equipment[0].transfer_tasks:

                            if task.end_time-1 >= self.period:
                                is_recent_task = True

                        if not is_recent_task:
                            for id in self.stage1_copy.feasible_transfer_tasks_in_stage(t):                             # if its not, then only create a new task
                                new_task = TransferTask(id,
                                                        self.min_transfer_times_stage1[id - 1],
                                                        self.max_transfer_times_stage1[id - 1],
                                                        self.period + 1,
                                                        (self.period + 1) + (self.min_transfer_times_stage1[id - 1] - 1),
                                                        self.stage1_copy.transfer_equipment[0],
                                                        self.stage1_copy.id)

                                self.stage1_copy.transfer_equipment[0].transfer_tasks.append(new_task)

            if stage == 2:
                if self.stage2_copy.feasible_transfer_tasks_in_stage(t) and self.stage2_copy.feasible_transfer_equipment_in_stage(t):

                    if self.stage2_copy.transfer_equipment[0].transfer_tasks is None:

                        self.stage2_copy.transfer_equipment[0].transfer_tasks = []

                        for id in self.stage2_copy.feasible_transfer_tasks_in_stage(t):
                            new_task = TransferTask(id,
                                                    self.min_transfer_times_stage2[id - 1],
                                                    self.max_transfer_times_stage2[id - 1],
                                                    self.period + 1,
                                                    (self.period + 1) + (self.min_transfer_times_stage2[id - 1] - 1),
                                                    self.stage2_copy.transfer_equipment[0],
                                                    self.stage2_copy.id)

                            self.stage2_copy.transfer_equipment[0].transfer_tasks.append(new_task)
                    else:
                        is_recent_task = False                                                                          # Checking is there any recent task, meaning, is there a task inside which is ongoing, meaning, is this a same control as previous
                        for task in self.stage2_copy.transfer_equipment[0].transfer_tasks:

                            if task.end_time-1 >= self.period:
                                is_recent_task = True

                        if not is_recent_task:                                                                          # if its not, then only create a new task
                            for id in self.stage2_copy.feasible_transfer_tasks_in_stage(t):
                                new_task = TransferTask(id,
                                                        self.min_transfer_times_stage2[id - 1],
                                                        self.max_transfer_times_stage2[id - 1],
                                                        self.period + 1,
                                                        (self.period + 1) + (self.min_transfer_times_stage2[id - 1] - 1),
                                                        self.stage2_copy.transfer_equipment[0],
                                                        self.stage2_copy.id)

                                self.stage2_copy.transfer_equipment[0].transfer_tasks.append(new_task)

        return self.stage1_copy, self.stage2_copy


############################## For penalty #############################################################################
    def spinning_reserve_constraint(self):
        sp_constraint_violations = []
        for stage in range(1, 2):
            if stage == 1:
                for t in [0]:

                    dict_values = {}
                    for t_val in range(t, t + 4):
                        dict_values[t_val] = self.stage1_copy.total_sp_provision_at_time(t_val)

                    if (dict_values[t][t] - dict_values[t + 1][t + 1] == 0 and
                            dict_values[t][t] - dict_values[t + 2][t + 2] == 0 and
                            dict_values[t][t] - dict_values[t + 3][t + 3] == 0):
                        sp_constraint_violations.append(0)
                    else:
                        sp_constraint_violations.append(1)

                for t in (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93):

                    dict_values = {}
                    for t_val in range(t, t + 3):
                        dict_values[t_val] = self.stage1_copy.total_sp_provision_at_time(t_val)

                    if (dict_values[t][t] - dict_values[t + 1][t + 1] == 0 and
                            dict_values[t][t] - dict_values[t + 2][t + 2] == 0):
                        sp_constraint_violations.append(0)
                    else:
                        sp_constraint_violations.append(1)

                for t in (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94):

                    dict_values = {}
                    for t_val in range(t, t + 2):
                        dict_values[t_val] = self.stage1_copy.total_sp_provision_at_time(t_val)

                    if (dict_values[t][t] - dict_values[t + 1][t + 1] == 0):
                        sp_constraint_violations.append(0)

                    else:
                        sp_constraint_violations.append(1)

                for t in (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92):

                    dict_values = {}
                    for t_val in range(t, t + 4):
                        dict_values[t_val] = self.stage1_copy.total_sp_provision_at_time(t_val)

                    if (dict_values[t][t] - dict_values[t + 1][t + 1] == 0 and
                            dict_values[t][t] - dict_values[t + 2][t + 2] == 0 and
                            dict_values[t][t] - dict_values[t + 3][t + 3] == 0):
                        sp_constraint_violations.append(0)

                    else:
                        sp_constraint_violations.append(1)

        return sp_constraint_violations



###################################### Total cost #####################################################################
    def compute_reward(self,control):
        """
        Calculates the reward that will later be returned to the agent.
        return: Reward

        """
        if self.allocation == 'electricity_cost_and_spinning_reserve_provision':

            reward = self.cost(control)
        else:
            raise NotImplementedError(f'The reward strategy {self.allocation} has not been implemented.')

        return reward

    def cost(self,control):

        t = self.period

        self.consumption = self.stage1_copy.total_energy_consumption(t)[t] + \
                          self.stage2_copy.total_energy_consumption(t)[t] + \
                          self.stage3_copy.total_energy_consumption(t)[t]

        self.spinning_reserve_provision = self.stage1_copy.total_sp_provision_at_time(t)[t]
       # print(self.spinning_reserve_provision)

        self.difference = self.consumption - self.generation_vector[t]
        self.max = max(0, self.difference)                                                                               # Just for plotting purpose
        self.LHS_term = (self.price_vector[t] * max(0, self.difference))
        self.RHS_term = (self.sp_price_vector[t] * self.spinning_reserve_provision)
        self.electricity_cost = (self.price_vector[t] * max(0, self.difference))-(self.sp_price_vector[t] * self.spinning_reserve_provision )


        return self.electricity_cost



    ##################################################### step method ##################################################

    def step(self, control):                                                                                            # feed the control which is the output of sample_action method # def step(self,cfg, control):


        t = self.period                                                                                                 # what is current period? initially it will be 0, step is executed at t=1



        for i, value in enumerate(control):

            if i == 0:                                                                                                   # for EAF1
                if self.stage1_copy.production_equipment[0].production_tasks is None:                                    # If there are None production tasks
                                                                                                                         # make the production_tasks as empty list
                    if value != 0:
                        self.stage1_copy.production_equipment[0].production_tasks = []                                   # if value at i is not 0,meaning there is some control
                        new_task = ProductionTask(value,                                                                 # create a new task
                                                  self.processing_times_stage1[value - 1],
                                                  self.duration_stage1[value - 1],
                                                  self.period,
                                                  self.period + (self.duration_stage1[value - 1] - 1),
                                                  self.stage1_copy.production_equipment[0].id,
                                                  self.stage1_copy.id)
                        self.stage1_copy.assign_production_task(new_task,                                                 # assign this task to EAF1
                                                                self.stage1_copy.production_equipment[0],
                                                                self.period)

                else:                                                                                                     # If there are not None production tasks
                    is_recent_task = False                                                                                # Checking is there any recent task, meaning, is there a task inside which is ongoing, meaning, is this a same control as previous
                    for task in self.stage1_copy.production_equipment[0].production_tasks:

                        if task.end_time >= self.period:
                            is_recent_task = True

                    if not is_recent_task:                                                                                # if its not, then only create a new task
                        if value != 0:
                            new_task = ProductionTask(value,
                                                      self.processing_times_stage1[value - 1],
                                                      self.duration_stage1[value - 1],
                                                      self.period,
                                                      self.period + (self.duration_stage1[value - 1] - 1),
                                                      self.stage1_copy.production_equipment[0].id,
                                                      self.stage1_copy.id)

                            self.stage1_copy.assign_production_task(new_task,                                            # assign new task to EAF1
                                                                    self.stage1_copy.production_equipment[0],
                                                                    self.period)

            if i == 1:                                                                                                   # for EAF2
                if self.stage1_copy.production_equipment[1].production_tasks is None:                                    # If there are None production tasks
                                                                                                                         # make the production_tasks as empty list
                    if value != 0:
                        self.stage1_copy.production_equipment[1].production_tasks = []                                   # if value at i is not 0,meaning there is some control
                        new_task = ProductionTask(value,                                                                 # create a new task
                                                  self.processing_times_stage1[value - 1],
                                                  self.duration_stage1[value - 1],
                                                  self.period,
                                                  self.period + (self.duration_stage1[value - 1] - 1),
                                                  self.stage1_copy.production_equipment[1].id,
                                                  self.stage1_copy.id)
                        self.stage1_copy.assign_production_task(new_task,                                                 # assign this task to EAF2
                                                                self.stage1_copy.production_equipment[1],
                                                                self.period)

                else:                                                                                                     # If there are not None production tasks
                    is_recent_task = False                                                                                # Checking is there any recent task, meaning, is there a task inside which is ongoing, meaning, is this a same control as previous
                    for task in self.stage1_copy.production_equipment[1].production_tasks:

                        if task.end_time >= self.period:
                            is_recent_task = True

                    if not is_recent_task:                                                                                # if its not, then only create a new task
                        if value != 0:
                            new_task = ProductionTask(value,
                                                      self.processing_times_stage1[value - 1],
                                                      self.duration_stage1[value - 1],
                                                      self.period,
                                                      self.period + (self.duration_stage1[value - 1] - 1),
                                                      self.stage1_copy.production_equipment[1].id,
                                                      self.stage1_copy.id)

                            self.stage1_copy.assign_production_task(new_task,                                             # assign new task to EAF2
                                                                    self.stage1_copy.production_equipment[1],
                                                                    self.period)


            if i == 2:                                                                                                    # for AOD1
                if self.stage2_copy.production_equipment[0].production_tasks is None:                                     # If there are None production tasks
                                                                                                                          # make the production_tasks as empty list
                   if value != 0:
                        self.stage2_copy.production_equipment[0].production_tasks = []                                   # if value at i is not 0,meaning there is some control
                        new_task = ProductionTask(value,                                                                 # create a new task
                                                  self.processing_times_stage2[value - 1],
                                                  self.duration_stage2[value - 1],
                                                  self.period,
                                                  self.period + (self.duration_stage2[value - 1]-1),
                                                  self.stage2_copy.production_equipment[0].id,
                                                  self.stage2_copy.id)


                        self.stage2_copy.assign_production_task(new_task,                                                 # assign this task to AOD1
                                                            self.stage2_copy.production_equipment[0],
                                                            self.period,self.stage1_copy)


                else:                                                                                                     # If there are not None production tasks
                    is_recent_task = False                                                                                # Checking is there any recent task, meaning, is there a task inside which is ongoing, meaning, is this a same control as previous
                    for task in self.stage2_copy.production_equipment[0].production_tasks:

                        if task.end_time >= self.period:
                            is_recent_task = True

                    if not is_recent_task:                                                                               # if its not, then only create a new task
                        if value != 0:
                            new_task = ProductionTask(value,
                                                      self.processing_times_stage2[value - 1],
                                                      self.duration_stage2[value - 1],
                                                      self.period,
                                                      self.period + (self.duration_stage2[value - 1]-1),
                                                      self.stage2_copy.production_equipment[0].id,
                                                      self.stage2_copy.id)



                            self.stage2_copy.assign_production_task(new_task,                                            # assign new task to AOD1
                                                                self.stage2_copy.production_equipment[0],
                                                                self.period,self.stage1_copy)


            if i == 3:                                                                                                    # for AOD2
                if self.stage2_copy.production_equipment[1].production_tasks is None:                                     # If there are None production tasks
                                                                                                                          # make the production_tasks as empty list
                   if value != 0:
                        self.stage2_copy.production_equipment[1].production_tasks = []
                        new_task = ProductionTask(value,                                                                  # create a new task
                                                  self.processing_times_stage2[value - 1],
                                                  self.duration_stage2[value - 1],
                                                  self.period,
                                                  self.period + (self.duration_stage2[value - 1]-1),
                                                  self.stage2_copy.production_equipment[1].id,
                                                  self.stage2_copy.id)


                        self.stage2_copy.assign_production_task(new_task,                                                 # assign this task to AOD2
                                                            self.stage2_copy.production_equipment[1],
                                                            self.period,self.stage1_copy)

                else:                                                                                                     # If there are not None production tasks
                    is_recent_task = False                                                                                # Checking is there any recent task, meaning, is there a task inside which is ongoing, meaning, is this a same control as previous
                    for task in self.stage2_copy.production_equipment[1].production_tasks:

                        if task.end_time >= self.period:
                            is_recent_task = True

                    if not is_recent_task:                                                                                # if its not, then only create a new task
                        if value != 0:
                            new_task = ProductionTask(value,
                                                      self.processing_times_stage2[value - 1],
                                                      self.duration_stage2[value - 1],
                                                      self.period,
                                                      self.period + (self.duration_stage2[value - 1]-1),
                                                      self.stage2_copy.production_equipment[1].id,
                                                      self.stage2_copy.id)

                            self.stage2_copy.assign_production_task(new_task,                                             # assign this task to AOD2
                                                                    self.stage2_copy.production_equipment[1],
                                                                    self.period, self.stage1_copy)


            if i == 4:                                                                                                    # for LF1
                if self.stage3_copy.production_equipment[0].production_tasks is None:                                     # If there are None production tasks
                                                                                                                          # make the production_tasks as empty list
                   if value != 0:
                        self.stage3_copy.production_equipment[0].production_tasks = []                                    # if value at i is not 0,meaning there is some control
                        new_task = ProductionTask(value,                                                                  # create a new task
                                                  self.processing_times_stage3[value - 1],
                                                  self.duration_stage3[value - 1],
                                                  self.period,
                                                  self.period + (self.duration_stage3[value - 1]-1),
                                                  self.stage3_copy.production_equipment[0].id,
                                                  self.stage3_copy.id)

                        self.stage3_copy.assign_production_task(new_task,                                                 # assign this task to LF1
                                                            self.stage3_copy.production_equipment[0],
                                                            self.period,self.stage2_copy)

                else:                                                                                                     # If there are not None production tasks
                    is_recent_task = False                                                                                # Checking is there any recent task, meaning, is there a task inside which is ongoing, meaning, is this a same control as previous
                    for task in self.stage3_copy.production_equipment[0].production_tasks:

                        if task.end_time >= self.period:
                            is_recent_task = True

                    if not is_recent_task:                                                                                # if its not, then only create a new task
                        if value != 0:
                            new_task = ProductionTask(value,
                                                      self.processing_times_stage3[value - 1],
                                                      self.duration_stage3[value - 1],
                                                      self.period,
                                                      self.period + (self.duration_stage3[value - 1]-1),
                                                      self.stage3_copy.production_equipment[0].id,
                                                      self.stage3_copy.id)

                            self.stage3_copy.assign_production_task(new_task,                                             # assign new task to LF1
                                                                self.stage3_copy.production_equipment[0],
                                                                self.period,self.stage2_copy)


            if i == 5:                                                                                                     # for LF2
                if self.stage3_copy.production_equipment[1].production_tasks is None:                                      # If there are None production tasks
                                                                                                                           # make the production_tasks as empty list
                    if value != 0:
                        self.stage3_copy.production_equipment[1].production_tasks = []                                     # if value at i is not 0,meaning there is some control
                        new_task = ProductionTask(value,                                                                   # create a new task
                                                  self.processing_times_stage3[value - 1],
                                                  self.duration_stage3[value - 1],
                                                  self.period,
                                                  self.period + (self.duration_stage3[value - 1]-1),
                                                  self.stage3_copy.production_equipment[1].id,
                                                  self.stage3_copy.id)

                        self.stage3_copy.assign_production_task(new_task,                                                 # assign this task to LF2
                                                                self.stage3_copy.production_equipment[1],
                                                                self.period,self.stage2_copy)

                else:                                                                                                     # If there are not None production tasks
                    is_recent_task = False                                                                                # Checking is there any recent task, meaning, is there a task inside which is ongoing, meaning, is this a same control as previous
                    for task in self.stage3_copy.production_equipment[1].production_tasks:

                        if task.end_time >= self.period:
                            is_recent_task = True

                    if not is_recent_task:                                                                                # if its not, then only create a new task
                        if value != 0:
                            new_task = ProductionTask(value,
                                                      self.processing_times_stage3[value - 1],
                                                      self.duration_stage3[value - 1],
                                                      self.period,
                                                      self.period + (self.duration_stage3[value - 1]-1),
                                                      self.stage3_copy.production_equipment[1].id,
                                                      self.stage3_copy.id)

                            self.stage3_copy.assign_production_task(new_task,                                             # assign new task to LF2
                                                                self.stage3_copy.production_equipment[1],
                                                                self.period,self.stage2_copy)


        self.assign_transfer_task()                                                                                      # Assignment of transfer task


        current_electricity_cost = self.compute_reward(control)                                                          # calculating cost for each time period for all stages



        self.total_costs.append(current_electricity_cost)                                                                # Electricity cost at each period

        #energy consumption
        self.aux.append(self.difference)                                                                                # Energyresource-E , for calculating maxaux
        self.maxaux.append(self.max)                                                                                    # max(0,self.aux),for plotting consumption plot, total consumed power at each time slot


        self.provision.append(self.spinning_reserve_provision)
        self.total_cost_electricity.append(self.LHS_term)
        self.total_cost_spinning_reserve.append(self.RHS_term)


        all_heats_finished, done = self.check_done()                                                                     # Determine if the simulation should terminate



        # Step period on and update state
        self.period += 1                                                                                                 # increment the period by 1
        t = self.period


        # Update state
        self.observe_state()                                                                                            # after assigning task, call observe _state method to see the state


        # Feasible controls
        self.dynamic_restrict_control()                                                                                 # then restrict the control space for the policy to make the next decision at next time step


        info = {
                'all heats finished': self.all_heats_finished
               }


        state = np.hstack([self.state_space_stage1_pt, self.state_space_stage2_pt,self.state_space_stage3_pt,self.price_info,self.sp_price_info,self.generation_info])

        control_mask = self.feasible_control_set

        return state, current_electricity_cost, control_mask, done, info                                                  # state, cost, done, info





    def check_done(self) -> bool:  

        t = self.period  

        for stage in range(1, self.stages + 1):

            if stage == 1:  # for 1st stage
                complete_1 = 0
                self.task_status = self.stage1_copy.is_tasks_ended(t)
                if len((self.task_status[1].keys())) + len((self.task_status[2].keys())) == self.num_heats:
                    if all(self.task_status[1].values()) and all(self.task_status[2].values()) == True:
                        complete_1 = complete_1 + 1

                self.incomplete_tasks_1 = [num for num in range(1, 25) if
                                           num not in self.task_status[1] and num not in self.task_status[2]]

            if stage == 2:
                complete_2 = 0
                self.task_status = self.stage2_copy.is_tasks_ended(t)
                if len((self.task_status[1].keys())) + len((self.task_status[2].keys())) == self.num_heats:
                    if all(self.task_status[1].values()) and all(self.task_status[2].values()) == True:
                        complete_2 = complete_2 + 1

                self.incomplete_tasks_2 = [num for num in range(1, 25) if
                                           num not in self.task_status[1] and num not in self.task_status[2]]

            if stage == 3:
                complete_3 = 0
                self.task_status = self.stage3_copy.is_tasks_ended(t)
                if len((self.task_status[1].keys())) + len((self.task_status[2].keys())) == self.num_heats:
                    if all(self.task_status[1].values()) and all(self.task_status[2].values()) == True:
                        complete_3 = complete_3 + 1

                self.incomplete_tasks_3 = [num for num in range(1, 25) if
                                           num not in self.task_status[1] and num not in self.task_status[2]]

        total = complete_1 + complete_2 + complete_3

        self.incomplete_tasks = []
        for lst in [self.incomplete_tasks_1, self.incomplete_tasks_2, self.incomplete_tasks_3]:
            for item in lst:
                if item not in self.incomplete_tasks:
                    self.incomplete_tasks.append(item)

        if total == self.stages:
            self.all_heats_finished = 0
        else:
            self.all_heats_finished = 2

        done = self.all_heats_finished == 0 and self.period < int(self.periods)
        
        return self.all_heats_finished, done

    def input_data(self):
        periods = range(1, len(self.electricity_prices) + 1)
        prices = self.electricity_prices
        generation = self.generation

        # Create the primary y-axis for prices
        fig, ax1 = plt.subplots()

        # Plot the prices on the primary y-axis
        # ax1.plot(periods, prices, marker='o', linestyle='-', color='b', label='Wholesale market price')
        ax1.step(periods, prices, linestyle='-', color='b', label='Wholesale market price')
        ax1.set_xlabel('Time slots')
        ax1.set_ylabel('Price (£/MWh)')
        ax1.grid(True)
        # ax1.legend()

        # Set x-axis ticks at intervals of 4 time slots
        x_ticks_interval = 4
        ax1.set_xticks(np.arange(1, len(self.electricity_prices) + 1, x_ticks_interval))

        # Create the secondary y-axis for generation
        ax2 = ax1.twinx()

        # Plot the generation on the secondary y-axis
        # ax2.plot(periods, generation, marker='x', linestyle='-', color='r', label='RES Generation')
        ax2.step(periods, generation, linestyle='-', color='r', label='RES Generation')
        ax2.set_ylabel('Generation (MW)')

        # Combine the legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.show()
        plt.savefig('input_data.svg', bbox_inches='tight')
        plt.close()

    def generate_gantt_chart(self):
        # Create a dictionary to store the colors for each task ID
        task_colors = {}

        EAF1 = self.stage1_copy.production_equipment[0]
        EAF2 = self.stage1_copy.production_equipment[1]
        AOD1 = self.stage2_copy.production_equipment[0]
        AOD2 = self.stage2_copy.production_equipment[1]
        LF1 = self.stage3_copy.production_equipment[0]
        LF2 = self.stage3_copy.production_equipment[1]

        # Assign colors to each unique task ID
        task_ids = range(1, 25)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray', 'aquamarine',
                  'skyblue', 'navy', 'orange', 'purple', 'lime', 'pink', 'teal', 'gold',
                  'silver', 'olive', 'maroon', 'salmon', 'peru', 'plum', 'darkgreen']

        for i, task_id in enumerate(task_ids):
            color = colors[i % len(colors)]
            task_colors[task_id] = color

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_xlim(0, int(self.periods))
        ax.set_xticks(np.arange(0, int(self.periods) + 1, 4))

        # Set the y-axis labels
        equipment_labels = ['EAF1', 'EAF2', 'AOD1', 'AOD2', 'LF1', 'LF2']
        ax.set_yticks(range(len(equipment_labels)))
        ax.set_yticklabels(equipment_labels)
        ax.set_ylim(-0.5, len(equipment_labels) - 0.5)

        ax.set_xticklabels(range(0, int(self.periods) + 1, 4))

        # Plot the tasks for each equipment
        legend_patches = []  # List to store legend patches
        processed_task_ids = set()

        for i, equipment in enumerate([EAF1, EAF2, AOD1, AOD2, LF1, LF2]):
            if equipment.production_tasks is not None:
                for task in equipment.production_tasks:
                    color = task_colors.get(task.id, 'gray')  # Default to 'gray' if color is not found
                    start_time = task.start_time
                    end_time = task.end_time

                    # Plot a horizontal bar for each task
                    ax.barh(i, end_time - start_time, left=start_time, height=0.2, color=color)

                    # Add legend patch for the task if it doesn't exist in the list
                    if task.id not in processed_task_ids:
                        legend_patch = mpatches.Patch(color=color, label=f'Heat {task.id}')
                        legend_patches.append(legend_patch)
                        processed_task_ids.add(task.id)

        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()  # Adjusts the subplot parameters to fit the legend properly

        # Set the chart title and labels
        plt.title('Production schedule')
        plt.xlabel('Time Slots')

        # Rotate the y-axis labels for better readability
        plt.tick_params(axis='y', rotation=0)

        # Remove grid lines
        ax.grid(False)

        plt.show()
        plt.savefig('gantt_chart.svg', bbox_inches='tight')
        plt.close()

    def consumption_plot(self):
        periods = range(1, len(self.maxaux) + 1)  # Use the total number of time steps (length of self.total_costs)
        consumption = self.maxaux  # Use the stored total costs for each time step

        # Plotting the graph
        plt.step(periods, consumption, linestyle='-', color='r', label='Consumption')
        # plt.bar(periods, consumption, width=0.8, color='r', label='Cost')
        plt.xlabel('Period')
        plt.ylabel('Consumption')
        plt.title('Electricity Consumption')
        plt.grid(True)
        plt.legend()

        # Set x-axis ticks at intervals of 4 time slots
        x_ticks_interval = 4
        plt.xticks(np.arange(1, len(self.maxaux) + 1, x_ticks_interval))

        plt.show()
        plt.savefig('consumption_plot.svg', bbox_inches='tight')
        plt.close()

    def provision_plot(self):
        # print(self.provision)

        periods = range(1, len(self.provision) + 1)  # Use the total number of time steps (length of self.total_costs)
        provision = self.provision  # Use the stored total costs for each time step

        # Plotting the graph
        plt.step(periods, provision,marker ='o', linestyle='-', color='b', label='Provision')
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to 'equal'
        # plt.bar(periods, consumption, width=0.8, color='r', label='Cost')
        plt.xlabel('Period')
        plt.ylabel('Provision')
        plt.title('Spinning reserve provision')
        plt.grid(True)
        plt.legend()

        # Set x-axis ticks at intervals of 4 time slots
        x_ticks_interval = 4
        plt.xticks(np.arange(1, len(self.provision) + 1, x_ticks_interval))

        plt.show()
        plt.savefig('provisionplot.svg', bbox_inches='tight')
        plt.close()

    def cost_plot(self):
        periods = range(1, len(self.total_costs) + 1)  # Use the total number of time steps (length of self.total_costs)
        costs = self.total_costs  # Use the stored total costs for each time step

        # Plotting the graph
        plt.step(periods, costs, linestyle='-', color='b', label='Cost')
        plt.xlabel('Period')
        plt.ylabel('Cost')
        plt.title('Electricity Cost')
        plt.grid(True)
        plt.legend()

        # Set x-axis ticks at intervals of 4 time slots
        x_ticks_interval = 4
        plt.xticks(np.arange(1, len(range(1, int(self.periods) + 1)) + 1, x_ticks_interval))

        plt.show()
        plt.savefig('cost_plot.svg', bbox_inches='tight')
        plt.close()

    def consumption_and_prices_plot(self):
        periods = range(1, len(self.maxaux) + 1)

        #To adjust prices size accordingly to maxaux size
        size=len(periods)
        self.electricity_prices = self.electricity_prices[:size]

        # Plotting consumption on the left y-axis
        fig, ax1 = plt.subplots()
        ax1.step(periods, self.maxaux, linestyle='-', color='r', label='Electricity Consumption')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Electricty Consumption (MWh)', color='r')
        ax1.tick_params('y', colors='r')
        ax1.grid(True)

        # Set x-axis ticks at intervals of 4 time slots
        x_ticks_interval = 4
        ax1.set_xticks(np.arange(1, len(self.maxaux) + 1, x_ticks_interval))

        # Creating the secondary y-axis for prices
        ax2 = ax1.twinx()
        ax2.step(periods, self.electricity_prices, linestyle='-', color='g', label='Electricity Prices')
        ax2.set_ylabel('Electricity Price (£/MWh)', color='g')
        ax2.tick_params('y', colors='g')

        # Combine the legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        # plt.title('Electricity Consumption and Prices')
        plt.show()
        plt.savefig('consumption_and_prices_plot.svg', bbox_inches='tight')
        plt.close()

    def provision_plot(self):
        periods = range(1, len(self.provision) + 1)  # Use the total number of time steps (length of self.total_costs)
        provision = self.provision  # Use the stored total costs for each time step

        # Plotting the graph
        plt.step(periods, provision, linestyle='-', color='b', label='Provision')
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to 'equal'
        # plt.bar(periods, consumption, width=0.8, color='r', label='Cost')
        plt.xlabel('Period')
        plt.ylabel('Provision')
        plt.title('Spinning reserve provision')
        plt.grid(True)
        plt.legend()

        # Set x-axis ticks at intervals of 4 time slots
        x_ticks_interval = 4
        plt.xticks(np.arange(1, len(self.provision) + 1, x_ticks_interval))

        plt.show()
        plt.savefig('provisionplot.svg', bbox_inches='tight')
        plt.close()


    def render(self):

        self.input_data()
        self.generate_gantt_chart()
        self.consumption_plot()
        self.cost_plot()
        self.consumption_and_prices_plot()
        self.provision_plot()

        return


