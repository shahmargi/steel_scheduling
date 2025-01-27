" production tasks"

from dataclasses import dataclass, field
from typing import List, Any

#################################################################### Production Task ###################################
"""
    This ProductionTask class can be used to model tasks of a scheduling problem.
    

    :param id: index of the task
    :param processingtime: processingtime of the task in minutes
    :param duration: duration of the task in time slots 
    :param start_time: time the task started at in the schedule
    :param end_time: time the task finished at in the schedule
    :param equipment_item: selected equipment item from the equipments list 
    :param stage: selected stage item from the equipments list 
    
 """

UNASSIGNED_FLAG = -1


@dataclass
class ProductionTask:
    id: int
    processingtime: int
    duration: float
    start_time: int = UNASSIGNED_FLAG
    end_time: int = UNASSIGNED_FLAG
    production_equipment: Any = UNASSIGNED_FLAG
    stage: Any = UNASSIGNED_FLAG

    def is_assigned_production_task(self):
        "A method to determine if the production task has been assigned"
        flags = (
            prop == UNASSIGNED_FLAG
            for prop in [
            self.production_equipment,
            self.stage,
            self.end_time,
            self.start_time
        ]
        )
        if all(flags):
            return False
        elif any(flags):
            raise RuntimeError("This production task has a mixed unassigned state.")
        else:
            return True


#################################################################### Transfer Task #####################################
"""
    This TransferTask class can be used to model tasks of a scheduling problem.
    

    :param id: index of the task
    :param min_transfer_duration: min_transfer_duration of the task in time slots
    :param max_transfer_duration: max_transfer_duration of the task in time slots 
    :param start_time: time the task started at in the schedule
    :param end_time: time the task finished at in the schedule
    :param equipment_item: selected equipment item from the equipments list 
    :param stage: selected stage item from the equipments list 
    
 """


@dataclass
class TransferTask:
    id: int
    min_transfer_duration: float
    max_transfer_duration: float
    start_time: int = UNASSIGNED_FLAG
    end_time: int = UNASSIGNED_FLAG
    transfer_equipment: Any = UNASSIGNED_FLAG
    stage: Any = UNASSIGNED_FLAG

    def is_assigned_transfer_task(self):
        "A method to determine if the production task has been assigned"
        flags = (
            prop == UNASSIGNED_FLAG
            for prop in [
            self.transfer_equipment,
            self.stage,
            self.start_time
        ]
        )
        if all(flags):
            return False
        elif any(flags):
            raise RuntimeError("This production task has a mixed unassigned state.")
        else:
            return True

    # def __setattr__(self, name: str, value: Any) -> None:
    #     """This emulates an immutable class if the end time of a task is not an unassigned flag.
    #     The idea is to handle a distinction between a mutable production task that
    #     can be reassigned or modified during planning modifications,
    #     while allowing a production task once it has been realized in time during a simulation.
    #     """
    #     if self.end_time is not UNASSIGNED_FLAG:
    #         raise AttributeError("This production task is finished, it cannot be updated.")
    #     else:
    #         super().__setattr__(name, value)


# Heat3 = ProductionTask(3, 90, 6, 24, 29, 1, 1)
# Heat7 = ProductionTask(7, 85, 6, 30, 35, 1, 1)
# # Heat24 = ProductionTask(24, 80, 6, 12, 17, 1, 1)
# # Heat2 = ProductionTask(2, 80, 6, 18, 23, 1, 1)
# # Heat11 = ProductionTask(11, 90, 6, 0, 5, 2, 1)
# Heat22 = ProductionTask(22, 80, 6, 24, 29, 2, 1)
# Heat9 = ProductionTask(9, 90, 6, 13, 18, 2, 1)
# Heat5 = ProductionTask(5, 85, 6, 19, 24, 2, 1)


