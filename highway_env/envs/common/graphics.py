import os
from typing import TYPE_CHECKING, Callable, List, Optional
import numpy as np
import pygame

from highway_env.envs.common.action import ActionType, DiscreteMetaAction, ContinuousAction
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv
    from highway_env.envs.common.abstract import Action


class EnvViewer(object):

    """A viewer to render a highway driving environment."""

    SAVE_IMAGES = False
    agent_display = None

    def __init__(self, env: 'AbstractEnv', policy: str, config: Optional[dict] = None) -> None:
        self.env = env
        self.config = config or env.config
        self.offscreen = self.config["offscreen_rendering"]
        self.observer_vehicle = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None
        self.policy = policy

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.config["screen_width"], self.config["screen_height"])

        # A display is not mandatory to draw things. Ignoring the display.set_mode()
        # instruction allows the drawing to be done on surfaces without
        # handling a screen display, useful for e.g. cloud computing
        if not self.offscreen:
            self.screen = pygame.display.set_mode([self.config["screen_width"], self.config["screen_height"]])
        if self.agent_display:
            self.extend_display()
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = self.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False
    
    def set_policy(self, policy: str) -> None:
        self.policy = policy

    def set_agent_display(self, agent_display: Callable) -> None:
        """
        Set a display callback provided by an agent

        So that they can render their behaviour on a dedicated agent surface, or even on the simulation surface.

        :param agent_display: a callback provided by the agent to display on surfaces
        """
        if EnvViewer.agent_display is None:
            self.extend_display()
        EnvViewer.agent_display = agent_display

    def extend_display(self) -> None:
        if not self.offscreen:
            if self.config["screen_width"] > self.config["screen_height"]:
                self.screen = pygame.display.set_mode((self.config["screen_width"],
                                                       2 * self.config["screen_height"]))
            else:
                self.screen = pygame.display.set_mode((2 * self.config["screen_width"],
                                                       self.config["screen_height"]))
        self.agent_surface = pygame.Surface((self.config["screen_width"], self.config["screen_height"]))

    def set_agent_action_sequence(self, actions: List['Action']) -> None:
        """
        Set the sequence of actions chosen by the agent, so that it can be displayed

        :param actions: list of action, following the env's action space specification
        """
        if isinstance(self.env.action_type, DiscreteMetaAction):
            actions = [self.env.action_type.actions[a] for a in actions]
        elif isinstance(self.env.action_type, ContinuousAction):
            actions = [self.env.action_type.get_action(a) for a in actions]
        if len(actions) > 1:
            self.vehicle_trajectory = self.env.vehicle.predict_trajectory(actions,
                                                                          1 / self.env.config["policy_frequency"],
                                                                          1 / 3 / self.env.config["policy_frequency"],
                                                                          1 / self.env.config["simulation_frequency"])

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler.handle_event(self.env.action_type, event)

    def display(self) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if EnvViewer.agent_display:
            EnvViewer.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            self.policy,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1

    def render_info_overlay(self, render_infos: dict):
        pygame.font.init()
        my_font = pygame.font.SysFont('arial', 25)

        timestep_surface = my_font.render(f"Timestep: {render_infos['timestep']:d}", False, (0, 0, 0))
        self.sim_surface.blit(timestep_surface, (2, 0))
        episode_surface = my_font.render(f"Episode:  {render_infos['episode']:d}", False, (0, 0, 0))
        self.sim_surface.blit(episode_surface, (2, 30))

        forward_surface = my_font.render(f"Fwd Speed: {render_infos['forward speed']:.2f}", False, (0, 0, 0))
        self.sim_surface.blit(forward_surface, (300, 0))
        lateral_surface = my_font.render(f"Lat Speed:    {render_infos['lateral speed']:.2f}", False, (0, 0, 0))
        self.sim_surface.blit(lateral_surface, (300, 30))
        basic_surface = my_font.render(f"Basic Rwd:   {render_infos['basic reward']:.2f}", False, (0, 0, 0))
        self.sim_surface.blit(basic_surface, (300, 60))
        added_surface = my_font.render(f"Added Rwd: {render_infos['added reward']:.2f}", False, (0, 0, 0))
        self.sim_surface.blit(added_surface, (300, 90))

        takeover = render_infos['takeover']
        change_flag = render_infos['change_flag']
        policy_color = (0, 0, 0)
        if not takeover:
            policy_color = (50, 200, 0)
            if not change_flag:  # prior no change
                text_surface = my_font.render("Press 'Enter' to engage", False, (100, 200, 255))
                self.sim_surface.blit(text_surface, (2, 300))
            else: # human to prior
                text_surface = my_font.render("Disengaged!", False, (255, 0, 0))
                self.sim_surface.blit(text_surface, (2, 300))
        else:
            text_surface = my_font.render("Select actions", False, (0, 255, 0))
            self.sim_surface.blit(text_surface, (2, 300))
            text_surface = my_font.render("'Up' to FASTER", False, (0, 0, 0))
            self.sim_surface.blit(text_surface, (300, 300))
            text_surface = my_font.render("'Space' to IDLE", False, (0, 0, 0))
            self.sim_surface.blit(text_surface, (300, 330))
            text_surface = my_font.render("'Down' to SLOWER", False, (0, 0, 0))
            self.sim_surface.blit(text_surface, (300, 360))
            text_surface = my_font.render("'Left' to LANE_LEFT", False, (0, 0, 0))
            self.sim_surface.blit(text_surface, (300, 390))
            text_surface = my_font.render("'Right' to LANE_RIGHT", False, (0, 0, 0))
            self.sim_surface.blit(text_surface, (300, 420))
            text_surface = my_font.render("'Esc' to Disengage", False, (0, 0, 0))
            self.sim_surface.blit(text_surface, (300, 450))
            if change_flag: # prior to human
                policy_color = (50, 200, 0)
            else: # human no change
                policy_color = (200, 200, 0)

        policy_surface = my_font.render(f"Last Policy: {render_infos['policy']}", False, policy_color)
        self.sim_surface.blit(policy_surface, (2, 60))
        action_surface = my_font.render(f"Last Action: {render_infos['action']}", False, policy_color)
        self.sim_surface.blit(action_surface, (2, 90))

    def display_info(self, render_infos: dict) -> None:
        """Display the road and vehicles on a pygame window with info overlay."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            self.policy,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)

        # overlay information
        if render_infos:
            self.render_info_overlay(render_infos=render_infos)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1

    def get_image(self) -> np.ndarray:
        """
        The rendered image as a rgb array.

        Gymnasium's channel convention is H x W x C
        """
        surface = self.screen if self.config["render_agent"] and not self.offscreen else self.sim_surface
        data = pygame.surfarray.array3d(surface)  # in W x H x C channel convention
        return np.moveaxis(data, 0, 1)

    def window_position(self) -> np.ndarray:
        """the world position of the center of the displayed window."""
        if self.observer_vehicle:
            return self.observer_vehicle.position
        elif self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self) -> None:
        """Close the pygame window."""
        pygame.quit()


class EventHandler(object):
    @classmethod
    def handle_event(cls, action_type: ActionType, event: pygame.event.EventType) -> None:
        """
        Map the pygame keyboard events to control decisions

        :param action_type: the ActionType that defines how the vehicle is controlled
        :param event: the pygame event
        """
        if isinstance(action_type, DiscreteMetaAction):
            cls.handle_discrete_action_event(action_type, event)
        elif action_type.__class__ == ContinuousAction:
            cls.handle_continuous_action_event(action_type, event)

    @classmethod
    def handle_discrete_action_event(cls, action_type: DiscreteMetaAction, event: pygame.event.EventType) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["FASTER"])
            if event.key == pygame.K_LEFT and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["SLOWER"])
            if event.key == pygame.K_DOWN and action_type.lateral:
                action_type.act(action_type.actions_indexes["LANE_RIGHT"])
            if event.key == pygame.K_UP:
                action_type.act(action_type.actions_indexes["LANE_LEFT"])

    @classmethod
    def handle_continuous_action_event(cls, action_type: ContinuousAction, event: pygame.event.EventType) -> None:
        action = action_type.last_action.copy()
        steering_index = action_type.space().shape[0] - 1
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0.7
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = -0.7
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = -0.7
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0.7
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_LEFT and action_type.lateral:
                action[steering_index] = 0
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action[0] = 0
            if event.key == pygame.K_UP and action_type.longitudinal:
                action[0] = 0
        action_type.act(action)


class ObservationGraphics(object):
    COLOR = (0, 0, 0)

    @classmethod
    def display(cls, obs, sim_surface):
        from highway_env.envs.common.observation import LidarObservation
        if isinstance(obs, LidarObservation):
            cls.display_grid(obs, sim_surface)

    @classmethod
    def display_grid(cls, lidar_observation, surface):
        psi = np.repeat(np.arange(-lidar_observation.angle/2,
                                  2 * np.pi - lidar_observation.angle/2,
                                  2 * np.pi / lidar_observation.grid.shape[0]), 2)
        psi = np.hstack((psi[1:], [psi[0]]))
        r = np.repeat(np.minimum(lidar_observation.grid[:, 0], lidar_observation.maximum_range), 2)
        points = [(surface.pos2pix(lidar_observation.origin[0] + r[i] * np.cos(psi[i]),
                                   lidar_observation.origin[1] + r[i] * np.sin(psi[i])))
                  for i in range(np.size(psi))]
        pygame.draw.lines(surface, ObservationGraphics.COLOR, True, points, 1)
