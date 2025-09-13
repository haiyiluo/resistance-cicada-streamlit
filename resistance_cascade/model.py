import mesa
import math
import logging as log
import numpy as np

# Import scheduler with compatibility handling
try:
    from resistance_cascade.scheduler import SimultaneousActivationByTypeFiltered
except ImportError:
    from .scheduler import SimultaneousActivationByTypeFiltered

# Import agents
try:
    from resistance_cascade.agent import Citizen, Security
except ImportError:
    from .agent import Citizen, Security

# Fix Mesa space import for different versions
try:
    from mesa.space import MultiGrid
except ImportError:
    try:
        MultiGrid = mesa.space.MultiGrid
    except AttributeError:
        from mesa import space
        MultiGrid = space.MultiGrid

# Fix Mesa DataCollector import for different versions
try:
    from mesa.datacollection import DataCollector
except ImportError:
    try:
        DataCollector = mesa.DataCollector
    except AttributeError:
        from mesa import datacollection
        DataCollector = datacollection.DataCollector


class ResistanceCascade(mesa.Model):
    """
    The model consists of two types of agent---Citizen agents which are the 
    primary agents of question and are the subject of most macro-scale model 
    measures, and Security agents.
    """

    def __init__(
        self,
        width=40,
        height=40,
        citizen_vision=7,
        citizen_density=0.7,
        security_density=0.00,
        security_vision=7,
        max_jail_term=100,
        movement=True,
        multiple_agents_per_cell=True,
        private_preference_distribution_mean=0,
        standard_deviation=1,
        epsilon=0.5,
        max_iters=1000,
        threshold=3.66356,
        seed=None,
        random_seed=False,
    ):
        super().__init__()
        
        # Initialize time tracking
        self.time = 0
        self.iteration = 0
        
        # Set up random seed
        if seed is None and not random_seed:
            seed = 42  # Default seed
            
        if random_seed:
            seed = np.random.randint(0, 1000000)
            
        # Store seed for reporting
        self._seed = seed
        # Use Mesa's built-in random seeding
        self.reset_randomizer(seed)
        
        print(f"Running ResistanceCascade with seed {self._seed}")
        log.info(f"Running ResistanceCascade with seed {self._seed}")
        
        # Initialize scheduler
        self.schedule = SimultaneousActivationByTypeFiltered(self)
        
        # Model parameters
        self.width = width
        self.height = height

        # Model boolean constants
        self.movement = movement
        self.multiple_agents_per_cell = multiple_agents_per_cell

        # Agent level constants
        self.citizen_density = citizen_density
        self.citizen_vision = citizen_vision
        self.private_preference_distribution_mean = private_preference_distribution_mean
        self.standard_deviation = standard_deviation
        self.epsilon = epsilon
        self.threshold = threshold
        self.threshold_constant_sigmoid = self.sigmoid(self.threshold)
        self.security_density = security_density
        self.security_vision = security_vision

        # Model level constants
        self.max_jail_term = max_jail_term
        self.citizen_count = round(self.width * self.height * self.citizen_density)
        self.security_count = round(self.width * self.height * self.security_density)

        # Model setup
        self.max_iters = max_iters
        self.random_seed = random_seed
        
        # Initialize grid
        self.grid = MultiGrid(self.width, self.height, torus=True)

        # Agent counts
        self.support_count = 0
        self.active_count = 0
        self.oppose_count = 0

        # Revolution tracking
        self.revolution = False

        # Create citizens
        for i in range(self.citizen_count):
            pos = None
            if not self.multiple_agents_per_cell and len(self.grid.empties) > 0:
                pos = self.random.choice(list(self.grid.empties))
            else:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                pos = (x, y)
                
            # Normal distribution of private regime preference
            private_preference = self.random.gauss(
                self.private_preference_distribution_mean, self.standard_deviation
            )
            # Error term for information controlled society
            epsilon = self.random.gauss(0, self.epsilon)
            # Epsilon error term sigmoid value
            epsilon_probability = self.sigmoid(epsilon)
            # Threshold calculations
            thresholds = [self.random.gauss(self.threshold, epsilon) for _ in range(0, 2)]
            # Threshold for opposition
            oppose_threshold = min(thresholds)
            # Threshold for activation
            active_threshold = max(thresholds)
            
            citizen = Citizen(
                self.next_id(),
                self,
                pos,
                self.citizen_vision,
                private_preference,
                epsilon,
                epsilon_probability,
                oppose_threshold,
                active_threshold,
            )
            self.grid.place_agent(citizen, pos)
            self.schedule.add(citizen)

        # Create Security agents
        for i in range(self.security_count):
            pos = None
            if not self.multiple_agents_per_cell and len(self.grid.empties) > 0:
                pos = self.random.choice(list(self.grid.empties))
            else:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                pos = (x, y)

            # Normal distribution of private regime preference
            private_preference = self.random.gauss(
                self.private_preference_distribution_mean, self.standard_deviation
            )

            security = Security(
                self.next_id(),
                self,
                pos,
                self.security_vision,
                private_preference,
            )
            self.grid.place_agent(security, pos)
            self.schedule.add(security)

        # Set up data collection
        model_reporters = {
            "Seed": self.report_seed,
            "Citizen Count": self.count_citizen,
            "Active Count": self.count_active,
            "Support Count": self.count_support,
            "Oppose Count": self.count_oppose,
            "Jail Count": self.count_jail,
            "Speed of Spread": self.speed_of_spread,
            "Security Density": self.report_security_density,
            "Private Preference": self.report_private_preference,
            "Epsilon": self.report_epsilon,
            "Threshold": self.report_threshold,
            "Revolution": self.report_revolution,
        }
        
        agent_reporters = {
            "pos": lambda a: getattr(a, "pos", None),
            "condition": lambda a: getattr(a, "condition", None),
            "opinion": lambda a: getattr(a, "opinion", None),
            "activation": lambda a: getattr(a, "activation", None),
            "private_preference": lambda a: getattr(a, "private_preference", None),
            "epsilon": lambda a: getattr(a, "epsilon", None),
            "oppose_threshold": lambda a: getattr(a, "oppose_threshold", None),
            "active_threshold": lambda a: getattr(a, "active_threshold", None),
            "jail_sentence": lambda a: getattr(a, "jail_sentence", None),
            "actives_in_vision": lambda a: getattr(a, "actives_in_vision", None),
            "opposed_in_vision": lambda a: getattr(a, "opposed_in_vision", None),
            "support_in_vision": lambda a: getattr(a, "support_in_vision", None),
            "security_in_vision": lambda a: getattr(a, "security_in_vision", None),
            "perception": lambda a: getattr(a, "perception", None),
            "arrest_prob": lambda a: getattr(a, "arrest_prob", None),
            "active_level": lambda a: getattr(a, "active_level", None),
            "oppose_level": lambda a: getattr(a, "oppose_level", None),
            "flip": lambda a: getattr(a, "flip", None),
            "ever_flipped": lambda a: getattr(a, "ever_flipped", None),
            "model_seed": lambda a: getattr(a, "dc_seed", None),
            "model_security_density": lambda a: getattr(a, "dc_security_density", None),
            "model_private_preference": lambda a: getattr(a, "dc_private_preference", None),
            "model_epsilon": lambda a: getattr(a, "dc_epsilon", None),
            "model_threshold": lambda a: getattr(a, "dc_threshold", None),
        }
        
        self.datacollector = DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

        # Set citizen states prior to first step
        for agent in self.schedule.agents_by_type[Citizen].values():
            agent.neighborhood = agent.update_neighbors()
            agent.determine_condition()

        # Start the model
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        self.schedule.step()
        self.time += 1
        self.iteration += 1
        
        # Check revolution condition
        active_or_jailed_agents = sum(
            1 for agent in self.schedule.agents 
            if type(agent) is Citizen and (agent.condition == "Active" or agent.condition == "Jailed")
        )
        proportion_active_or_jailed = active_or_jailed_agents / self.citizen_count

        # Check stop condition
        if proportion_active_or_jailed >= 0.95:
            log.debug(f"Stop condition met at iteration {self.iteration}, Viva la Revolucion!")
            print(f"Stop condition met at iteration {self.iteration}, Viva la Revolucion!")
            self.revolution = True
            self.running = False

        # Collect data
        self.datacollector.collect(self)

        # Update agent counts
        self.active_count = self.count_active(self)
        self.support_count = self.count_support(self)
        self.oppose_count = self.count_oppose(self)

        # Check max iterations
        if self.iteration > self.max_iters:
            self.running = False

    def distance_calculation(self, agent1, agent2):
        """Helper method to calculate distance between two agents."""
        return math.sqrt(
            (agent1.pos[0] - agent2.pos[0]) ** 2 + (agent1.pos[1] - agent2.pos[1]) ** 2
        )

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + math.exp(-x))

    # Static reporter methods
    @staticmethod
    def report_seed(model):
        """Helper method to report the seed."""
        return model._seed

    @staticmethod
    def count_citizen(model):
        """Helper method to report the citizen count."""
        return model.citizen_count

    @staticmethod
    def speed_of_spread(model):
        """Calculates the speed of transmission of the rebellion."""
        return (
            len([
                agent for agent in model.schedule.agents_by_type[Citizen].values()
                if agent.flip is True
            ]) / model.citizen_count
        )

    @staticmethod
    def count_active(model):
        """Helper method to count active agents."""
        return len([
            agent for agent in model.schedule.agents_by_type[Citizen].values()
            if agent.condition == "Active"
        ])

    @staticmethod
    def count_oppose(model):
        """Helper method to count publicly opposing agents."""
        return len([
            agent for agent in model.schedule.agents_by_type[Citizen].values()
            if agent.condition == "Oppose"
        ])

    @staticmethod
    def count_support(model):
        """Helper method to count publicly supporting agents."""
        return len([
            agent for agent in model.schedule.agents_by_type[Citizen].values()
            if agent.condition == "Support"
        ])

    @staticmethod
    def count_jail(model):
        """Helper method to count jailed agents."""
        return len([
            agent for agent in model.schedule.agents_by_type[Citizen].values()
            if agent.condition == "Jailed"
        ])

    @staticmethod
    def report_security_density(model):
        """Helper method to count security density."""
        return model.security_density

    @staticmethod
    def report_private_preference(model):
        """Helper method to count private preference distribution mean."""
        return model.private_preference_distribution_mean

    @staticmethod
    def report_epsilon(model):
        """Helper method to count epsilon."""
        return model.epsilon

    @staticmethod
    def report_threshold(model):
        """Helper method to count threshold."""
        return model.threshold

    @staticmethod
    def report_revolution(model):
        """Helper method to count revolutions."""
        return model.revolution