from typing import Type, Callable, Dict, Any
from mesa.agent import Agent
from mesa.model import Model
from collections import defaultdict

# Alternative import method
import mesa

# Try different import methods depending on Mesa version
try:
    from mesa.time import SimultaneousActivation
except ImportError:
    try:
        SimultaneousActivation = mesa.time.SimultaneousActivation
    except AttributeError:
        import mesa.time as mesa_time
        SimultaneousActivation = mesa_time.SimultaneousActivation


class SimultaneousActivationByTypeFiltered(SimultaneousActivation):
    """
    A scheduler that overrides the get_type_count method to allow for filtering
    of agents by a function before counting.

    Example:
    >>> scheduler = SimultaneousActivationByTypeFiltered(model)
    >>> scheduler.get_type_count(AgentA, lambda agent: agent.some_attribute > 10)
    """

    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.agents_by_type: Dict[Type[Agent], Dict[int, Agent]] = defaultdict(dict)

    def add(self, agent: Agent) -> None:
        """
        Add an Agent object to the schedule

        Args:
            agent: An Agent to be added to the schedule.
        """
        super().add(agent)
        agent_class = type(agent)
        self.agents_by_type[agent_class][agent.unique_id] = agent

    def remove(self, agent: Agent) -> None:
        """
        Remove all instances of a given agent from the schedule.
        """
        super().remove(agent)  # Use super().remove() instead of direct deletion
        agent_class = type(agent)
        if agent.unique_id in self.agents_by_type[agent_class]:
            del self.agents_by_type[agent_class][agent.unique_id]

    def get_type_count(
        self,
        type_class: Type[Agent],
        filter_func: Callable[[Agent], bool] = None,
    ) -> int:
        """
        Returns the current number of agents of certain type in the queue 
        that satisfy the filter function.
        """
        count = 0
        for agent in self.agents_by_type[type_class].values():
            if filter_func is None or filter_func(agent):
                count += 1
        return count

    def get_agents_of_type(self, type_class: Type[Agent]):
        """
        Returns all agents of a given type.
        """
        return list(self.agents_by_type[type_class].values())