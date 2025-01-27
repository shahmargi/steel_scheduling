from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Any
from omegaconf import DictConfig, OmegaConf

from steel_scheduling.utilities.production_tasks import ProductionTask
from steel_scheduling.utilities.equipment import ProductionEquipment
from steel_scheduling.utilities.equipment import TransferEquipment


@dataclass
class Stage:
    id: int
    production_equipment: List[ProductionEquipment] | None = None
    transfer_equipment: List[TransferEquipment] | None = None

    ###########################   ASSIGNMENT DECISIONS FOR A PRODUCTION TASK   #########################################

    ################  To check if the production task is not assigned in the time horizon   ############################
    def assigned_production_tasks(self):
        """
        Input: Stage itself
        Method: Operates in each stage --> Checks each ProductionEquipment --> Checks the tasks inside both production
                equipments --> Appends the task id to a list
        Output: List of ProductionTask's id which are in the production equipment of a stage
      """
        assigned_production_task_list = []

        for equipment in self.production_equipment:

            if equipment.production_tasks is not None:

                for task in equipment.production_tasks:
                    assigned_production_task_list.append(task.id)

        return assigned_production_task_list

    #####################  To check if the transfer task has passed minimum time duration   ############################

    def completed_transfer_tasks(self, time: int):
        """
        Input: Stage itself and time int
        Method: Operates in each stage --> Checks each TransferEquipment --> Checks each TransferTask --> Checks
                 current time and its start time if greater than or equal to minimum transfer duration --> Appends
                  its id to a list
        Output: List of TransferTask's id which has passed minimum transfer duration
        """

        self._time = time

        if self.transfer_equipment is not None:

            completed_transfer_task_list = []

            for equipment in self.transfer_equipment:

                if equipment.transfer_tasks is not None:

                    for task in equipment.transfer_tasks:

                        if time - task.start_time > (task.min_transfer_duration - 1):
                            completed_transfer_task_list.append(task.id)

            return completed_transfer_task_list

        #####################  To check if the transfer task has passed minimum time duration   ############################

    def completed_transfer_tasks_policy(self, time: int):
        """
        Input: Stage itself and time int
        Method: Operates in each stage --> Checks each TransferEquipment --> Checks each TransferTask --> Checks
                 current time and its start time if greater than or equal to minimum transfer duration --> Appends
                  its id to a list
        Output: List of TransferTask's id which has passed minimum transfer duration
        """

        self._time = time

        if self.transfer_equipment is not None:

            completed_transfer_task_list = []

            for equipment in self.transfer_equipment:

                if equipment.transfer_tasks is not None:

                    for task in equipment.transfer_tasks:

                        if time - task.start_time > (task.min_transfer_duration - 1):
                            completed_transfer_task_list.append(task)

            return completed_transfer_task_list

    #####################  To check if the transfer task has passed maximum time duration   ############################

    def violated_transfer_tasks(self, time: int):
        """
        Input: Stage itself and time int
        Method: Operates in each stage --> Checks each TransferEquipment --> Checks each TransferTask --> Checks if
                  the task has passed  its maximum transfer duration --> Append its id to a list
        Output: List of TransferTask's id which has passed maximum transfer duration
        """

        self._time = time

        if self.transfer_equipment is not None:

            violated_transfer_task_list = []

            for equipment in self.transfer_equipment:

                if equipment.transfer_tasks is not None:

                    for task in equipment.transfer_tasks:

                        if not (time - task.start_time <= (task.max_transfer_duration)):
                            violated_transfer_task_list.append(task.id)

            return violated_transfer_task_list

    ## To check if the transfer task has passed minimum time duration and is waiting before maximum transfer duration ##

    def waiting_transfer_tasks(self, time: int):
        """
        Input: Stage itself and time int
        Method: Operates in each stage --> Checks each TransferEquipment --> Checks each TransferTask --> Checks if
                the task has passed its minimum duration, but it is inside its maximum duration --> Appends it to
                 a list
        Output: List of TransferTask's id which has passed minimum transfer duration, but it is inside its
                 maximum duration
        """
        self._time = time

        if self.transfer_equipment is not None:

            waiting_transfer_task_list = []

            for equipment in self.transfer_equipment:

                if equipment.transfer_tasks is not None:

                    for task in equipment.transfer_tasks:

                        if (time - task.start_time > task.min_transfer_duration) and (time - task.start_time <=
                                                                                      task.max_transfer_duration):
                            waiting_transfer_task_list.append(task.id)

            return waiting_transfer_task_list

    ############################ To check if the production equipment is available #####################################

    def production_equipment_available(self, time):
        """
        Input: Stage itself and time int
        Method: Operates in each stage --> Checks each ProductionEquipment --> Checks if all production tasks inside
                equipment has end time less than current time
        Output: Dictionary of ProductionEquipment id's with boolean if available or not
        """

        equipment_available_dict = {}

        for equipment in self.production_equipment:

            if equipment.production_tasks is not None:

                if all((task.end_time < time for task in equipment.production_tasks)):

                    available = True

                else:

                    available = False

                equipment_available_dict[equipment.id] = available
            else:
                equipment_available_dict[equipment.id] = True

        return equipment_available_dict

    def feasible_production_equipment_in_stage(self, time):
        """
        Input: Stage itself
        Method: Checks each equipment in stage if its id is in the production_equipment_available method and if it's
                true.
        Output: List of feasible ProductionEquipment's id in each stage
        """
        feasible_equipment = []

        equipment_avaibility = self.production_equipment_available(time)

        for equipment in self.production_equipment:

            if equipment.id in equipment_avaibility and equipment_avaibility[equipment.id]:
                feasible_equipment.append(equipment.id)

        return feasible_equipment

        ############################ To check what are the feasible production tasks in a stage ############################

    def feasible_production_tasks_in_stage(self, time, stage):
        """
        Input: Stage itself,Previous stage object and time
        Method: [Checks if the task is in completed_transfer_tasks & not in violated_transfer_tasks &
                 not in assigned_production_tasks ] OR
                [Checks if the task is in completed_transfer_tasks & in waiting_transfer_tasks &
                 not in violated_transfer_tasks & not in assigned_production_tasks]
        Output: List of feasible ProductionTask's id in each stage.
       """
        self._time = time

        production_tasks = self.assigned_production_tasks()
        transfer_tasks_completed = stage.completed_transfer_tasks(self._time)
        transfer_tasks_violated = stage.violated_transfer_tasks(self._time)
        transfer_tasks_waiting = stage.waiting_transfer_tasks(self._time)

        feasible_production_tasks = []

        for task_id in transfer_tasks_completed:

            if (task_id not in transfer_tasks_violated and task_id not in production_tasks) \
                    or (
                    task_id in transfer_tasks_waiting and task_id not in transfer_tasks_violated and task_id not in
                    production_tasks):
                feasible_production_tasks.append(task_id)

        return feasible_production_tasks

    ################################### To assign a production task in a stage #########################################

    def assign_production_task(self, task: ProductionTask, equipment: ProductionEquipment, time: int, stage=None):
        """
        Input: Stage itself, a ProductionTask class ,e ProductionEquipment class chosen by policy,time, previous
                stage object
        Method: For stage 1 --> previous stage object input is None by default --> Checks if the task id given in
                input is not in the list of assigned production tasks list and if equipment id is in list of
                feasible_production_equipment_in_stage -->  appends that task into equipment of that stage
                 For other stages --> previous stage object input is given in input --> Checks if the task id given
                 in input is in list of feasible production tasks (by checking completed transfer tasks of previous
                 object)--> also checking if equipment id is in list of feasible_production_equipment_in_stage -->
                appends that task into equipment of that stage
        Output: Equipment class
       """
        if self.id == 1:

            assigned_production_ids = self.assigned_production_tasks()
            feasible_equipment_ids = self.feasible_production_equipment_in_stage(time)

            if task.id not in assigned_production_ids and equipment.id in feasible_equipment_ids:

                equipment.production_tasks.append(task)

                warning_flag = False

            else:
                warning_flag = True

                raise Warning(f'Could not assign production task')

        else:

            feasible_production_ids = self.feasible_production_tasks_in_stage(time, stage)
            feasible_equipment_ids = self.feasible_production_equipment_in_stage(time)

            if task.id in feasible_production_ids and equipment.id in feasible_equipment_ids:

                equipment.production_tasks.append(task)

                warning_flag = False

            else:

                warning_flag = True

                raise Warning(f'Could not assign production task')

        return warning_flag

    ###########################   ASSIGNMENT DECISIONS FOR A TRANSFER TASK   ###########################################

    ################  To check if the transfer task is not assigned in the time horizon   #############################
    def assigned_transfer_tasks(self):
        """
        Input: Stage itself
        Method: Operates in each stage --> Checks each TransferEquipment --> Checks the tasks inside both transfer
                equipments --> Appends the task id to a list
        Output: List of TransferTask's id which are in the transfer equipment of a stage
        """
        assigned_transfer_task_list = []

        for equipment in self.transfer_equipment:

            if equipment.transfer_tasks is not None:

                for task in equipment.transfer_tasks:
                    assigned_transfer_task_list.append(task)

        return assigned_transfer_task_list

    ########################### To check which are the production tasks in a stage which has completed  ################

    def completed_production_tasks(self, time: int):
        """
        Input: Stage itself and time int
        Method: Operates in each stage --> Checks each ProductionEquipment --> Checks each ProductionTask --> Checks
                current time and its start time if equal to task duration
        Output: List of Production task's id which has passed minimum transfer duration
       """
        completed_production_task_list = []

        for equipment in self.production_equipment:

            if equipment.production_tasks is not None:

                for task in equipment.production_tasks:

                    if time - task.start_time >= (task.duration - 1):
                        completed_production_task_list.append(task)

        return completed_production_task_list

    def feasible_transfer_tasks_in_stage(self, time):
        """
        Input: Stage itself and time
        Method: Checks if the task has passed task duration in the production equipment & current time is
                its end time --> Check if that task is not in assigned_transfer_tasks --> Append those tasks in a list
        Output: List of feasible Transfer Task class in each stage.
       """
        transfer_tasks = self.assigned_transfer_tasks()
        production_tasks = self.completed_production_tasks(time)

        feasible_transfer_tasks = []

        for task in production_tasks:

            if task.end_time == time:

                if task not in transfer_tasks:
                    feasible_transfer_tasks.append(task.id)

        return feasible_transfer_tasks

    ############################ To check if the transfer equipment is available #######################################
    def transfer_equipment_available(self, time):
        """
        Input: Stage itself and time int
        Method: Operates in each stage --> Checks each TransferEquipment --> Checks if all transfer tasks has passed
                its minimum transfer time duration
        Output: Dictionary of TransferEquipment id's with boolean if available or not
       """
        equipment_available_dict = {}

        if self.transfer_equipment is not None:

            for equipment in self.transfer_equipment:

                if equipment.transfer_tasks is not None:

                    if all(((time - task.start_time) >= (task.min_transfer_duration - 1) for task in
                            equipment.transfer_tasks)):

                        available = True

                    else:

                        available = False

                    equipment_available_dict[equipment.id] = available
                else:
                    equipment_available_dict[equipment.id] = True

            return equipment_available_dict

    def feasible_transfer_equipment_in_stage(self, time):
        """
        Input: Stage itself
        Method: Checks each equipment in stage if its id is in the transfer_equipment_available method and if it's
                true.
        Output: List of feasible TransferEquipment's id in each stage
      """
        feasible_equipment = []

        equipment_avaibility = self.transfer_equipment_available(time)

        for equipment in self.transfer_equipment:

            if equipment.id in equipment_avaibility and equipment_avaibility[equipment.id]:
                feasible_equipment.append(equipment.id)

        return feasible_equipment

    ################################### To calculate energy consumption of each task ###################################
    def energy_consumption(self):
        """
        Input: Stage itself
        Method: Looks inside list of production tasks inside each production equipment --> Sees those tasks whose end
                 time == current time --> Then it calculates that task's total energy consumption --> Then there is a
                 loop that will calculate its energy consumption at each time slot --> Finally we have a dictionary
                 with key as task id and value as a list of its consumption in each time slot
        Output: A dictionary having keys as task id and values as list of energy consumption at each time slot of its
                 task duration
      """
        energy_consumption_dict = {}

        time_interval = 15  # step size

        for equipment in self.production_equipment:

            if equipment.production_tasks is not None:

                for task in equipment.production_tasks:

                    total_energy_consumed = (task.processingtime * equipment.norm_mw) / 60

                    energy_profile = []

                    for i in range(task.duration - 1):
                        energy_profile.append((time_interval * equipment.norm_mw) / 60)

                    energy_profile = energy_profile + [total_energy_consumed - sum(energy_profile)]

                    energy_consumption_dict[task.id] = energy_profile

        return energy_consumption_dict

    ################################### To calculate total energy consumption in each time period ######################

    def total_energy_consumption(self, time: int):

        """
        # Input:
        # Method:
        # Output: Gives total energy consumption at each time step
       """
        total_energy_consumption_at_time = {}

        total_consumption = 0

        for equipment in self.production_equipment:

            energy_profile_of_tasks = self.energy_consumption()

            if equipment.production_tasks is not None:

                for task in equipment.production_tasks:

                    if time <= task.end_time:

                        task_time_slot = (time - task.start_time) + 1

                        if task.id in energy_profile_of_tasks:
                            consumption = energy_profile_of_tasks[task.id][task_time_slot - 1]

                            total_consumption = total_consumption + consumption

        total_energy_consumption_at_time[time] = total_consumption

        return total_energy_consumption_at_time

    ####################################### To check if the task is done in a stage ####################################

    def is_tasks_ended(self, time):
        is_task_ended_dict = {}

        for equipment in self.production_equipment:

            task_statuses = {}

            if equipment.production_tasks is not None:

                for task in equipment.production_tasks:

                    if task in equipment.production_tasks:

                        if time >= task.end_time:

                            task_statuses[task.id] = True
                        else:
                            task_statuses[task.id] = False

            is_task_ended_dict[equipment.id] = task_statuses

        return is_task_ended_dict

    ################################### To calculate SP Provision of each task ###################################
    def sp_provision(self):

        sp_provision_dict = {}

        time_interval = 15  # step size

        for equipment in self.production_equipment:

            energy_profile_of_tasks = self.energy_consumption()

            if equipment.production_tasks is not None:

                for task in equipment.production_tasks:

                    sp_profile = []

                    for i in range(len(energy_profile_of_tasks[task.id])):
                        sp_profile.append(34)

                    sp_profile = sp_profile + [0]

                    sp_provision_dict[task.id] = sp_profile

        return sp_provision_dict

    def total_sp_provision_at_time(self, time: int):

        sp_provision_at_time = {}
        total_provision = 0

        for equipment in self.production_equipment:

            sp_profile_of_tasks = self.sp_provision()

            if equipment.production_tasks is not None:

                for task in equipment.production_tasks:

                    if task.start_time <= time <= task.end_time:

                    #if time <= task.end_time:

                        task_time_slot = (time - task.start_time) + 1

                        if task.id in sp_profile_of_tasks:
                            provision = sp_profile_of_tasks[task.id][task_time_slot - 1]

                            total_provision = total_provision + provision

        sp_provision_at_time[time] = total_provision

        return sp_provision_at_time

    def sp_provision_at_time(self, time: int):
        sp_provision_at_time = {}

        for equipment in self.production_equipment:
            total_provision = 0
            sp_profile_of_tasks = self.sp_provision()

            if equipment.production_tasks is not None:
                for task in equipment.production_tasks:
                    if task.start_time <= time <= task.end_time:
                        task_time_slot = (time - task.start_time) + 1
                        if task.id in sp_profile_of_tasks:
                            provision = sp_profile_of_tasks[task.id][task_time_slot - 1]
                            total_provision += provision

            sp_provision_at_time[equipment.id] = total_provision

        return sp_provision_at_time





################################## check methods ###################################################################


# if __name__ == '__main__':
#     def main():
#         stage1 = Stage(1, [ EAF2])
#         y = stage1.sp_provision()
#         #x = stage1.sp_provision_at_time(84)
#         z = stage1.total_sp_provision_at_time(84)
#         print(y)
#         #print(x)
#         print(z)
#
#
# main()
