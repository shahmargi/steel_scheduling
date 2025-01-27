from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Any

from steel_scheduling.utilities.production_tasks import ProductionTask
from steel_scheduling.utilities.production_tasks import TransferTask


UNASSIGNED_FLAG = -1

"""
    This ProductionEquipment class can be used to model production equipment of a scheduling problem.


    :param id: index of the production equipment
    :param norm_mw: power rating of production equipment
    :param stage: id of the stage to which this production equipment belongs 
    :param production_tasks: list of production tasks assigned to this production equipment

 """


@dataclass
class ProductionEquipment:
    id: int
    norm_mw: Any
    stage: int
    production_tasks: List[ProductionTask] | None = None


"""
    This TransferEquipment class can be used to model transfer equipment of a scheduling problem.


    :param id: index of the transfer equipment
    :param stage: id of the stage to which this transfer equipment belongs 
    :param production_tasks: list of transfer tasks assigned to this transfer equipment

 """


@dataclass
class TransferEquipment:
    id: int
    stage: int
    transfer_tasks: List[TransferTask] | None = None


























