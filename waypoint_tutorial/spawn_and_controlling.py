import glob
import os
# os.environ["PYTHONPATH"] = "/home/tiagorochag/carla/PythonAPI/carla"
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

try:
    sys.path.append("/home/tiagorochag/carla/PythonAPI/carla")
except IndexError:
    pass

from agents.navigation.controller import VehiclePIDController
# from PythonAPI.carla.agents.navigation.controller import VehiclePIDController

import carla
import random
import time



def main():
    t = time.time()
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        ego_col = None
        count = 0
        # print(str(count))
        #draw waypoints correspondent to the road_id. Many waypoints might lie in the same road_id
        def draw_waypoints(waypoints, road_id=None, life_time=50.0):
            count = 0
            for waypoint in waypoints:
                if (waypoint.road_id == road_id):
                    count += 1
                    world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                                 color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                                 persistent_lines=True)
                    # print(waypoint.road_id)
            # print(count)

        world = client.get_world()
        # Change weather parameter to night (facilitate waypoint visualization)
        weather = carla.WeatherParameters(sun_altitude_angle=0, sun_azimuth_angle=-90)
        world.set_weather(weather)

        # spawn_points = world.get_map().get_spawn_points()
        map = world.get_map()
        waypoints = map.generate_waypoints(distance=2.0)
        spawn_all_points = map.get_spawn_points()
        draw_waypoints(waypoints, road_id=7, life_time=20)

        # Find inside the blueprint library our car model3 and an obstacle prius
        vehicle_blueprint = world.get_blueprint_library().filter('model3')[0]
        vehicle_obstacle_bp = world.get_blueprint_library().filter('prius')[0]



        # List of all waypoints on road id
        filtered_waypoints = []
        for waypoint in waypoints:
            if(waypoint.road_id == 7):
                filtered_waypoints.append(waypoint)

        spawn_point = filtered_waypoints[0].transform
        spawn_point.location.z +=2

        # Spawn the vehicle
        ego_vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
        actor_list.append(ego_vehicle)
        # Spawn the obstacle
        id_spawn = 146
        vehicle2 = world.spawn_actor(vehicle_obstacle_bp, spawn_all_points[id_spawn])
        actor_list.append(vehicle2)

        spectator = world.get_spectator()
        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-10, z=2.5)),
                                    ego_vehicle.get_transform().rotation)
        spectator.set_transform(transform)


        # Adding collision sensor to model 3 (ego)
        # vehicle_blueprint.set_attribute('role_name', 'ego')
        col_bp = world.get_blueprint_library().find('sensor.other.collision')
        col_location = carla.Location(0, 0, 0)
        col_rotation = carla.Rotation(0, 0, 0)
        col_transform = carla.Transform(col_location, col_rotation)
        ego_col = world.spawn_actor(col_bp, col_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)

        def col_callback(colli):
            print("Collision detected: \n" + str(colli)+'\n')
        ego_col.listen(lambda colli: col_callback(colli))

        # --------------
        # Add IMU sensor to ego vehicle.
        # --------------

        imu_bp = world.get_blueprint_library().find('sensor.other.imu')
        imu_location = carla.Location(0, 0, 0)
        imu_rotation = carla.Rotation(0, 0, 0)
        imu_transform = carla.Transform(imu_location, imu_rotation)
        imu_bp.set_attribute("sensor_tick", str(3.0))
        ego_imu = world.spawn_actor(imu_bp, imu_transform, attach_to=ego_vehicle,
                                    attachment_type=carla.AttachmentType.Rigid)

        def imu_callback(imu):
            print("IMU measure:\n" + str(imu) + '\n')
        ego_imu.listen(lambda imu: imu_callback(imu))


        def acc_callback(acc):
            acc = ego_vehicle.get_acceleration()
            print("\nAcceleration x: " + str(acc.x))
            print("\nAcceleration y: " + str(acc.y))
            print("\nAcceleration z: " + str(acc.z))
        # ego_vehicle.get_control(lambda acc: acc_callback(acc))


        """""
        for actor in world.get_actors():
        if actor.attributes.get('role_name') == 'hero':
            player = actor
            break
        """""

        # PID control for longitudinal and lateral control
        custom_controller = VehiclePIDController(ego_vehicle, args_lateral={'K_P': 0, 'K_D': 0.0, 'K_I': 0},
                                                 args_longitudinal={'K_P': 1, 'K_D': 0.0, 'K_I': 0.0})

        # Chose the target waypoint and color it red
        target_waypoint = filtered_waypoints[36]
        world.debug.draw_string(target_waypoint.transform.location, 'O', draw_shadow=False,
                                            color=carla.Color(r=255, g=0, b=0), life_time=20,
                                             persistent_lines=True)
        """
        ticks_to_track = 20
        for i in range(ticks_to_track):
            control_signal = custom_controller.run_step(1, target_waypoint)
            control_signal.throttle=0.5
            ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
            # ego_vehicle.apply_control(control_signal)
            print(control_signal)
        """
        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
        print("\n Get control: " + str(ego_vehicle.get_control()))
        """
        SendNonPlayerAgentsInfo = True
        measurements, sensor_data = client.read_data()
        for agent in measurements.non_player_agents:
            print(agent.id)  # unique id of the agent
            if agent.HasField('vehicle'):
                print(agent.vehicle.acceleration)
                # agent.vehicle.forward_speed
                # agent.vehicle.transform
                # agent.vehicle.bounding_box
        """

        # world_snapshot = world.wait_for_tick()
        # time.sleep(11)
        # ego_vehicle.apply_control(VehicleControl(throttle=1.0))
        # vehicle_obstacle_bp.apply_control(Veh)
        # ego_vehicle.set_autopilot(True)
        vehicle2.set_autopilot(False)
        count += 1
        print(str(count))
        print("\n Get control: " + str(ego_vehicle.get_control()))

        while time.sleep(10):
            world_snapshot = world.wait_for_tick()


    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        ego_col.destroy()
        ego_imu.destroy()
        print('done.')
        print("count final: "+str(count))
        elapsed = time.time() - t
        print("elapsed time: " + str(elapsed))


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone')

    # actors = world.get_actors()
    # client.apply_batch([carla.command.DestroyActor(actors[0])])
