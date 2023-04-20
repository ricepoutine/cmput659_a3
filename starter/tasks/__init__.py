from .task import Task
from .stair_climber import StairClimber
from .maze import Maze
from .maze_sparse import MazeSparse

def get_task_cls(task_cls_name: str) -> type[Task]:
    task_cls = globals()[task_cls_name]    
    assert issubclass(task_cls, Task)
    return task_cls