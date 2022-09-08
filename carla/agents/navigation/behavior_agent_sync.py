# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import math
import random
import numpy as np
import carla
from agents.navigation.agent import Agent
from agents.navigation.local_planner_behavior import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.types_behavior import Cautious, Aggressive, Normal
from agents.navigation.controllerPID import PIDController

from agents.tools.misc import get_speed, positive

class BehaviorAgent(Agent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment,
    such as overtaking or tailgating avoidance. Adding to these are possible
    behaviors, the agent can also keep safety distance from a car in front of it
    by tracking the instantaneous time to collision and keeping it in a certain range.
    Finally, different sets of behaviors are encoded in the agent, from cautious
    to a more aggressive ones.
    """

    def __init__(self, vehicle, ignore_traffic_light=False, behavior='normal'):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param ignore_traffic_light: boolean to ignore any traffic light
            :param behavior: type of agent to apply
        """

        super(BehaviorAgent, self).__init__(vehicle)
        self.vehicle = vehicle
        self.ignore_traffic_light = ignore_traffic_light
        self._local_planner = LocalPlanner(self)
        self._grp = None
        self.look_ahead_steps = 0

        # Vehicle information
        self.speed = 0
        self.speed_limit = 0
        self.direction = None
        self.incoming_direction = None
        self.incoming_waypoint = None
        self.start_waypoint = None
        self.end_waypoint = None
        self.is_at_traffic_light = 0
        self.light_state = "Green"
        self.light_id_to_ignore = -1
        self.min_speed = 5
        self.behavior = None
        self._sampling_resolution = 5  # 2 backup
        self.aux_bool = True
        self.frame_rate = 20
        self._speed_controller_extrapolation = PIDController(K_P=0.15, K_I=0.07, K_D=0.05, n=40)
        self._lateral_controller_extrapolation = PIDController(K_P=0.58, K_I=0.5, K_D=0.02, n=40)
        # self._lateral_controller_extrapolation = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        # self._speed_controller_extrapolation = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self.clip_delta = 0.25  # Max angular error for turn controller
        self.clip_throttle = 0.75  # Max throttle (0-1)
        self.angle_search_range = 0  # Number of future waypoints to consider in angle search

        # Parameters for agent behavior
        if behavior == 'cautious':
            self.behavior = Cautious()

        elif behavior == 'normal':
            self.behavior = Normal()

        elif behavior == 'aggressive':
            self.behavior = Aggressive()

    def update_information(self, vehicle):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.

            :param world: carla.world object
        """

        aux_direction = self.direction
        self.speed = get_speed(self.vehicle)
        # self.speed_limit = world.player.get_speed_limit()
        self.speed_limit = vehicle.get_speed_limit()
        self._local_planner.set_speed(self.speed_limit)
        self.direction = self._local_planner.target_road_option

        if self.direction is None:
            self.direction = RoadOption.LANEFOLLOW

        self.look_ahead_steps = int((self.speed_limit) / 10)

        self.incoming_waypoint, self.incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self.look_ahead_steps)
        if self.incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW

        if self.incoming_waypoint is None:
            print(self.incoming_waypoint)

        # self.is_at_traffic_light = world.player.is_at_traffic_light()
        self.is_at_traffic_light = vehicle.is_at_traffic_light()
        if self.ignore_traffic_light:
            self.light_state = "Green"
        else:
            # This method also includes stop signs and intersections.
            self.light_state = str(self.vehicle.get_traffic_light_state())

        # print("Incoming waypoint: ", self.incoming_waypoint.transform, "Incoming direction: ", self.incoming_direction)
        if aux_direction != self.direction:
            print("Maneuver: ", self.direction)

    def set_destination(self, start_location, end_location, clean=False):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router.

            :param start_location: initial position
            :param end_location: final position
            :param clean: boolean to clean the waypoint queue
        """
        if clean:
            self._local_planner.waypoints_queue.clear()
        self.start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self._trace_route(self.start_waypoint, self.end_waypoint)

        self._local_planner.set_global_plan(route_trace, clean)

    def reroute(self, spawn_points):
        """
        This method implements re-routing for vehicles approaching its destination.
        It finds a new target and computes another path to reach it.

            :param spawn_points: list of possible destinations for the agent
        """

        print("Target almost reached, setting new destination...")
        random.shuffle(spawn_points)
        new_start = self._local_planner.waypoints_queue[-1][0].transform.location
        destination = spawn_points[0].location if spawn_points[0].location != new_start else spawn_points[1].location
        print("New destination: " + str(destination))

        self.set_destination(new_start, destination)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the
        optimal route from start_waypoint to end_waypoint.

            :param start_waypoint: initial position
            :param end_waypoint: final position
        """
        # Setting up global router
        if self._grp is None:
            wld = self.vehicle.get_world()
            dao = GlobalRoutePlannerDAO(
                wld.get_map(), sampling_resolution=self._sampling_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def traffic_light_manager(self, waypoint):
        """
        This method is in charge of behaviors for red lights and stops.

        WARNING: What follows is a proxy to avoid having a car brake after running a yellow light.
        This happens because the car is still under the influence of the semaphore,
        even after passing it. So, the semaphore id is temporarely saved to
        ignore it and go around this issue, until the car is near a new one.

            :param waypoint: current waypoint of the agent
        """

        light_id = self.vehicle.get_traffic_light().id if self.vehicle.get_traffic_light() is not None else -1

        if self.light_state == "Red":
            if not waypoint.is_junction and (self.light_id_to_ignore != light_id or light_id == -1):
                return 1
            elif waypoint.is_junction and light_id != -1:
                self.light_id_to_ignore = light_id
        if self.light_id_to_ignore != light_id:
            self.light_id_to_ignore = -1
        return 0

    def _overtake(self, location, waypoint, vehicle_list):
        """
        This method is in charge of overtaking behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        if (left_turn == carla.LaneChange.Left or left_turn ==
                carla.LaneChange.Both) and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
            new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=-1)
            if not new_vehicle_state:
                print("Overtaking to the left!")
                self.behavior.overtake_counter = 50
                self.set_destination(left_wpt.transform.location,
                                     self.end_waypoint.transform.location, clean=True)
        elif right_turn == carla.LaneChange.Right and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
            new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=180, lane_offset=1)
            if not new_vehicle_state:
                print("Overtaking to the right!")
                self.behavior.overtake_counter = 50
                self.set_destination(right_wpt.transform.location,
                                     self.end_waypoint.transform.location, clean=True)

    def _tailgating(self, location, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
            self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self.speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    self.behavior.tailgate_counter = 200
                    self.set_destination(right_wpt.transform.location,
                                         self.end_waypoint.transform.location, clean=True)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._bh_is_vehicle_hazard(waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    self.behavior.tailgate_counter = 200
                    self.set_destination(left_wpt.transform.location,
                                         self.end_waypoint.transform.location, clean=True)

    def collision_and_car_avoid_manager(self, location, waypoint, world):
        """
        This module is in charge of warning in case of a collision
        and managing possible overtaking or tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        # vehicle_list = [v for v in vehicle_list if dist(v) < 145 and v.id != self.vehicle.id]
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self.vehicle.id]

        if self.direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                waypoint, location, vehicle_list, max(
                    self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=30)

            # Check for overtaking

            if vehicle_state and self.direction == RoadOption.LANEFOLLOW and \
                    not waypoint.is_junction and self.speed > 1 \
                    and self.behavior.overtake_counter == 0 and self.speed > get_speed(vehicle):
            # if vehicle_state and self.direction == RoadOption.LANEFOLLOW and \
            #         not waypoint.is_junction and \
            #         self.behavior.overtake_counter == 0:
                self._overtake(location, waypoint, vehicle_list)
                print("Overtaking ONNN without speed - debug")

            # Check for tailgating

            elif not vehicle_state and self.direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self.speed > 10 \
                    and self.behavior.tailgate_counter == 0:
                self._tailgating(location, waypoint, vehicle_list)

        if True:
            control = carla.VehicleControl()
            self.ego_model = self.EgoModel(dt=(1.0 / self.frame_rate))
            # speed = self._get_forward_speed()
            speed_vec3 = self.vehicle.get_velocity()
            speed3 = (speed_vec3.x ** 2 + speed_vec3.y ** 2 + speed_vec3.z ** 2) ** 0.5
            vehicle_transform = self.vehicle.get_transform()
            next_loc_no_brake = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
            next_yaw_no_brake = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
            next_speed_no_brake = np.array([speed3])
            # NOTE intentionally set ego vehicle to move at the target speed (we want to know if there is an intersection if we would not brake)
            # throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)
            action_no_brake = np.array(np.stack([control.steer, control.throttle, 0.0], axis=-1))
            # action_no_brake = np.array(np.stack([control.steer, 0.5, 0.0], axis=-1))

            # if vehicle_list:
            if not True:
                """
                # obstacle_vehicle
                obstacle_control = vehicle_list[0].get_control()
                self.vehicle_obstacle_model = self.EgoModel(dt=(1.0 / self.frame_rate))
                # speed = self._get_forward_speed()
                speed_vec = vehicle_list[0].get_velocity()
                speed2 = (speed_vec.x**2 + speed_vec.y**2 + speed_vec.z**2)**0.5
                # print(speed2)
                vehicle_transform_obs = vehicle_list[0].get_transform()
                next_loc = np.array([vehicle_transform_obs.location.x, vehicle_transform_obs.location.y])
                next_yaw = np.array([vehicle_transform_obs.rotation.yaw / 180.0 * np.pi])
                next_speed = np.array([speed2])

                action = np.array(np.stack([obstacle_control.steer, obstacle_control.throttle, obstacle_control.brake], axis=-1))
                # action = np.array(np.stack([0, 0, 1], axis=-1))
                # future_seconds = 3
                # frame_rate = 10
                # future_frames = future_seconds * frame_rate
                """
                future_frames = 30
                # aux_next_loc_no_break = next_loc_no_brake
                # aux_next_yaw_no_break = next_yaw_no_brake
                vehicle_transform_prediction = self.vehicle.get_transform()
                carla_loca = self.vehicle.get_location()
                target_speed = min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)
                for k in range(future_frames):
                    world.debug.draw_line(carla_loca, vehicle_transform_prediction.location, thickness=0.1,
                                          color=carla.Color(r=0, g=255, b=0),
                                          life_time=0.5)
                    next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake = self.ego_model.forward(
                        next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake, action_no_brake)
                    # next_loc, next_yaw, next_speed = self.vehicle_obstacle_model.forward(next_loc, next_yaw, next_speed, action)
                    temp_waypoint = self._map.get_waypoint(carla_loca)
                    carla_loca.z = 0.5
                    carla_loca.x = next_loc_no_brake[0]
                    carla_loca.y = next_loc_no_brake[1]
                    waypoint_route_extrapolation = self._trace_route(temp_waypoint, self.end_waypoint)
                    # waypoint_route_extrapolation = self._waypoint_planner_extrapolation.run_step(
                    #     next_loc_no_brake_temp)
                    # vehicle_speed = get_speed(self.vehicle)
                    # target_speed = min(max(self.min_speed, vehicle_speed),
                    #              min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist))/3.6
                    # target_speed = min(self.behavior.max_speed, self.speed_limit - 5)
                    steer_extrapolation = self._get_steer_extrapolation(waypoint_route_extrapolation, next_loc_no_brake, next_yaw_no_brake, 3.6*next_speed_no_brake, restore=False)
                    throttle_extrapolation = self._get_throttle_extrapolation(target_speed, 3.6*next_speed_no_brake, restore=False)
                    # print("Target speed: ", "%.4f" % target_speed, "Predicted speed: ", "%.4f" % next_speed_no_brake, "Actual speed: ", "%.4f" % (vehicle_speed/3.6))
                    # print("Throttle extrapolation: ", "%.4f" % throttle_extrapolation, "Actual throttle:", "%.4f" % self.vehicle.get_control().throttle)
                    # print("Steer extrapolation: ", "%.4f" % steer_extrapolation, "Actual steer: ", "%.4f" % self.vehicle.get_control().steer)
                    brake_extrapolation = 0
                    action_no_brake = np.array(np.stack([steer_extrapolation, float(throttle_extrapolation), brake_extrapolation], axis=-1))
                    """
                    # delta_yaws_no_brake = next_yaw_no_brake.item() * 180.0 / np.pi
                # next_loc_no_brake = aux_next_loc_no_break
                # next_yaw_no_brake = next_yaw_no_brake - aux_next_yaw_no_break
                    distance_collision_pred = (next_loc_no_brake-next_loc)  #  collision between future ego pos and predicted obst
                    distance_collision_actual = (self.vehicle.get_location() - vehicle_list[0].get_location())  #actual collision checker (no prediction)
                    dist_colli_egopred_actualobs = (next_loc_no_brake - [vehicle_list[0].get_location().x, vehicle_list[0].get_location().y])  # collision between future ego path and an actual obstacle
                    dist_colli_egopred_actualobs_norm = math.sqrt(dist_colli_egopred_actualobs[0]**2+dist_colli_egopred_actualobs[1]**2)
                    distance_collision_pred_norm = math.sqrt(distance_collision_pred[0]**2+distance_collision_pred[1]**2)
                    distance_collision_actual_norm = math.sqrt(distance_collision_actual.x**2+distance_collision_actual.y**2)
                    dist_list = [distance_collision_pred_norm,distance_collision_actual_norm]
                    distance_collision = min(dist_list)
                    dist_list_index = dist_list.index(distance_collision)
                    # distance_collision = min(math.sqrt(distance_collision_pred[0]**2+distance_collision_pred[1]**2)
                    #                         ,math.sqrt(distance_collision_actual.x**2+distance_collision_actual.y**2))
                    """
                    #better distance computation
                # vehicle_state, vehicle, distance = self._bh_is_vehicle_hazard(
                #     waypoint, location, vehicle_list, max(
                #         self.behavior.min_proximity_threshold+10, self.speed_limit / 3), up_angle_th=30)
                    # if distance_collision < 2:
                    #     print("Less than 2")
                # print(next_loc)
                # Bounding box
                # print("Predicted location: ", carla_loca.x, carla_loca.y, "Actual location: ", np.array([vehicle_transform.location.x, vehicle_transform.location.y]))
                # print("Steer extrapolation: ", steer_extrapolation, "Throttle extra: ", throttle_extrapolation)
                vehicle_transform_prediction.location.x = next_loc_no_brake[0]
                vehicle_transform_prediction.location.y = next_loc_no_brake[1]
                # obs_transform_prediction = vehicle_list[0].get_transform()
                # obs_transform_prediction.location.x = 231 # to test model and carla simulator
                # obs_transform_prediction.location.y = -5 # to test model and carla simulator
                # obs_transform_prediction.location.x = next_loc[0]
                # obs_transform_prediction.location.y = next_loc[1]
                # if True:
                """
                    braking_distance = (3.6*speed3/10)**2*2+10
                    # braking_distance = 15
                    print("hazard distance: ", distance, "CAUTION, distance is: ", distance_collision, "breakig distance is: ", braking_distance, "index_colli", dist_list_index)
                    # if distance_collision < 1 and dist_list_index==0:
                    if distance_collision_pred_norm < 2 or dist_colli_egopred_actualobs_norm < 2:
                        self._draw_bbox_obstacle(world, self.vehicle, vehicle_transform_prediction, 2, carla.Color(r=255, g=0, b=0))  # print prediction debug
                        self._draw_bbox_obstacle(world, vehicle_list[0], obs_transform_prediction, 2, carla.Color(r=255, g=0, b=0))  # print prediction debug
                        return True, vehicle_list[0], distance_collision
                    elif distance_collision_actual_norm < braking_distance:
                        return True, vehicle_list[0], distance_collision
                    else:
                        self._draw_bbox_obstacle(world, self.vehicle, vehicle_transform_prediction,
                                                 2)  # print prediction debug
                        self._draw_bbox_obstacle(world, vehicle_list[0], obs_transform_prediction,
                                                 2)  # print prediction debug
                """
                # self._draw_bbox_obstacle(world, self.vehicle, vehicle_transform_prediction, 1)

            # else:
                # distance_collision_actual = (self.vehicle.get_location() - vehicle_list[0].get_location())
                # distance_collision = math.sqrt(distance_collision_actual.x ** 2 + distance_collision_actual.y ** 2)
                # print("no dangerous vehicle ahead, distance: ", distance_collision)
        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, location, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self.direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self.direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._bh_is_vehicle_hazard(waypoint, location, walker_list, max(
                self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self.speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self.behavior.safety_time > ttc > 0.0:
            control = self._local_planner.run_step(
                target_speed=min(positive(vehicle_speed - self.behavior.speed_decrease),
                                 min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)), debug=debug)
        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self.behavior.safety_time > ttc >= self.behavior.safety_time:
            control = self._local_planner.run_step(
                target_speed=min(max(self.min_speed, vehicle_speed),
                                 min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)), debug=debug)
        # Normal behavior.
        else:
            control = self._local_planner.run_step(
                target_speed= min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)

        return control

    def run_step(self, world, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        control = None
        if self.behavior.tailgate_counter > 0:
            self.behavior.tailgate_counter -= 1
        if self.behavior.overtake_counter > 0:
            self.behavior.overtake_counter -= 1

        ego_vehicle_loc = self.vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior

        if self.traffic_light_manager(ego_vehicle_wp) != 0:
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors

        walker_state, walker, w_distance = self.pedestrian_avoid_manager(
            ego_vehicle_loc, ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self.vehicle.bounding_box.extent.y, self.vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self.behavior.braking_distance:
                return self.emergency_stop()

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(
            ego_vehicle_loc, ego_vehicle_wp, world)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self.vehicle.bounding_box.extent.y, self.vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self.behavior.braking_distance:
                print("emergency control activated")
                return self.emergency_stop()
            else:
                print("Not emergency braking, distance is: ", distance)
                control = self.car_following_manager(vehicle, distance)

        # 4: Intersection behavior

        # Checking if there's a junction nearby to slow down
        elif self.incoming_waypoint.is_junction and (self.incoming_direction == RoadOption.LEFT or self.incoming_direction == RoadOption.RIGHT):
            control = self._local_planner.run_step(
                target_speed=min(self.behavior.max_speed, self.speed_limit - 5), debug=debug)

        # 5: Normal behavior

        # Calculate controller based on no turn, tsraffic light or vehicle in front
        else:
            control = self._local_planner.run_step(
                target_speed=min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist), debug=debug)

        if self.aux_bool:
            # DEBUG: Draw path forward
            self._draw_path(world, life_time=100.0, skip=0)
            # self._draw_path(world, life_time=100.0, skip=0)
            self.aux_bool = False
        if not True:
            # Bounding red box
            # self._draw_bbox(world, self.vehicle, 60)

            # DEBUG: Draw current waypoint backward
            # world.world.debug.draw_point(ego_vehicle_wp.transform.location, color=carla.Color(0, 255, 0), life_time=5.0)
            world.debug.draw_string(ego_vehicle_wp.transform.location, text='O', draw_shadow=False,
                                      color=carla.Color(r=0, g=255, b=255), life_time=20, persistent_lines=True)
            # incoming waypoint
            world.debug.draw_string(self.incoming_waypoint.transform.location, text='O', draw_shadow=False,
                                          color=carla.Color(r=255, g=248, b=220), life_time=20, persistent_lines=True)
        return control

    def _draw_path(self, world, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        route_waypoints = self._trace_route(self.start_waypoint, self.end_waypoint)
        for i in range(0, len(route_waypoints)-1, skip+1):
            w0 = route_waypoints[i][0]
            w1 = route_waypoints[i+1][0]
            world.debug.draw_line(
                w0.transform.location + carla.Location(z=0.25),
                w1.transform.location + carla.Location(z=0.25),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=True)
            world.debug.draw_point(
                w0.transform.location + carla.Location(z=0.25), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        world.debug.draw_string(route_waypoints[-1][0].transform.location, text='X', draw_shadow=False,
                                color=carla.Color(r=0, g=0, b=255), life_time=40, persistent_lines=True)
        # world.debug.draw_point(
        #     route_waypoints[-1][0].transform.location + carla.Location(z=0.25), 0.1,
        #     carla.Color(0, 0, 255),
        #     life_time, False)

    def _draw_bbox(self, world, actor, life_time=15.0):
        color = carla.Color(r=255, g=0, b=0)
        # get the current transform (location + rotation)
        transform = actor.get_transform()
        # bounding box is relative to the actor
        bounding_box = actor.bounding_box
        bounding_box.location += transform.location  # from relative to world
        world.debug.draw_box(bounding_box, transform.rotation,
                             color=color, life_time=life_time)

    def _draw_bbox_obstacle(self, world, actor, transform, life_time=15.0, color=carla.Color(r=0, g=200, b=0)):
        # color = carla.Color(r=0, g=200, b=0)
        # get the current transform (location + rotation)
        # transform = actor.get_transform()
        # bounding box is relative to the actor
        bounding_box = actor.bounding_box
        # bounding_box.location += transform.location  # from relative to world
        bounding_box.location += transform.location  # from relative to world
        world.debug.draw_box(bounding_box, transform.rotation,
                             color=color, life_time=life_time)

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self.vehicle.get_velocity()
        if not transform:
            transform = self.vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def _get_throttle_extrapolation(self, target_speed, speed, restore=True):
        # if self._waypoint_planner_extrapolation.is_last:  # end of route
        #     target_speed = 0.0

        # delta = np.clip(target_speed - speed, 0.0, self.clip_delta)
        delta = target_speed - speed
        # print("predicted error: ", delta)
        if restore: self._speed_controller_extrapolation.load()
        throttle = self._speed_controller_extrapolation.step(delta)
        if restore: self._speed_controller_extrapolation.save()

        throttle = np.clip(throttle, 0.0, self.clip_throttle)

        return throttle

    def _get_steer_extrapolation(self, route, pos, theta, speed, restore=True):
        # if self._trace_route.is_last:  # end of route
        #     angle = 0.0

        if len(route) == 1:
            target = route[0][0]
            target = np.array([target.transform.location.x, target.transform.location.y])
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        elif self.angle_search_range <= 2:
            target = route[1][0]
            target = np.array([target.transform.location.x,target.transform.location.y])
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        else:
            search_range = min([len(route), self.angle_search_range])
            for i in range(1, search_range):
                target = route[i][0]
                target = np.array([target.transform.location.x, target.transform.location.y])
                angle_unnorm = self._get_angle_to(pos, theta, target)
                angle_new = angle_unnorm / 90
                if i == 1:
                    angle = angle_new
                if np.abs(angle_new) < np.abs(angle):
                    angle = angle_new

        self.angle = angle

        if restore: self._lateral_controller_extrapolation.load()
        steer = self._lateral_controller_extrapolation.step(angle)
        if restore: self._lateral_controller_extrapolation.save()

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer

    def _get_angle_to(self, pos, theta, target):  # 2 - 3 mu
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        diff = target - pos
        aim_0 = (cos_theta * diff[0] + sin_theta * diff[1])
        aim_1 = (-sin_theta * diff[0] + cos_theta * diff[1])

        angle = -math.degrees(math.atan2(-aim_1, aim_0))
        angle = np.float_(angle)  # So that the optimized function has the same datatype as output.
        return angle


    class EgoModel():
        def __init__(self, dt=1. / 4):
            self.dt = dt

            # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
            self.front_wb = -0.090769015
            self.rear_wb = 1.4178275

            self.steer_gain = 0.36848336
            self.brake_accel = -4.952399
            self.throt_accel = 0.5633837

        def forward(self, locs, yaws, spds, acts):
            # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
            steer = acts[..., 0:1].item()
            throt = acts[..., 1:2].item()
            brake = acts[..., 2:3].astype(np.uint8)

            if brake:
                accel = self.brake_accel
            else:
                accel = self.throt_accel * throt

            wheel = self.steer_gain * steer

            beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
            yaws = yaws.item()
            spds = spds.item()
            next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
            next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
            next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
            next_spds = spds + accel * self.dt
            next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

            next_locs = np.array([next_locs_0, next_locs_1])
            next_yaws = np.array(next_yaws)
            next_spds = np.array(next_spds)

            return next_locs, next_yaws, next_spds



