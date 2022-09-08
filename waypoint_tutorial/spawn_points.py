import glob
import os
import sys
# import controller

# sys.path.insert(0, '/home/carla/PythonAPI/carla')
# from controller import VehiclePIDController

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# try:
#     sys.path.append("/home/tiagorochag/carla/PythonAPI/")
# except IndexError:
#     pass


# from agents.navigation.controller import VehiclePIDController
from agents.navigation.controller import VehiclePIDController

import carla
import random
import time



actor_list = []
try:

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    # spawn_point2 = random.choice(spawn_points) if spawn_points else carla.Transform()
    waypoints2 = world.get_map().generate_waypoints(distance=2.0)
    counter = 0
    for waypoint in spawn_points:
        world.debug.draw_string(waypoint.location, text='*'+str(counter), draw_shadow=False,
                                color=carla.Color(r=0, g=255, b=0), life_time=20, persistent_lines=True)
        counter += 1
    print(counter)
    """""
    waypoints = world.get_map().generate_waypoints(distance=2.0)
    draw_waypoints(waypoints, road_id=7, life_time=20)
ww
    # Change weather parameter to night (facilitate waypoint visualization)
    weather = carla.WeatherParameters(sun_altitude_angle=0, sun_azimuth_angle=-90)
    world.set_weather(weather)
    # Find inside the blueprint library our car model3

    vehicle_blueprint = world.get_blueprint_library().filter('model3')[0]
    # List of all waypoints on road id
    filtered_waypoints = []
    for waypoint in waypoints:
        if(waypoint.road_id == 7):
            filtered_waypoints.append(waypoint)

    spawn_point = filtered_waypoints[0].transform
    spawn_point.location.z +=2
    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
    actor_list.append(vehicle)


    # PID control for longitudinal and lateral control
    custom_controller = VehiclePIDController(vehicle, args_lateral={'K_P': 0, 'K_D': 0.0, 'K_I': 0},
                                             args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})

    # Chose the target waypoint and color it red
    target_waypoint = filtered_waypoints[36]
    world.debug.draw_string(target_waypoint.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=20,
                                         persistent_lines=True)

    ticks_to_track = 20
    for i in range(ticks_to_track):
        control_signal = custom_controller.run_step(1, target_waypoint)
        print(control_signal)
        vehicle.apply_control(control_signal)


    time.sleep(8)
    """""
finally:

    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')