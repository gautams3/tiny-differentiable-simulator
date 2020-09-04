#include <assert.h>
#include <stdio.h>

#include <chrono>
#include <thread>

#include "Utils/b3Clock.h"
#include "pybullet_urdf_import.h"
#include "pybullet_visualizer_api.h"
#include "tiny_double_utils.h"
#include "utils/file_utils.hpp"
#include "tiny_inverse_kinematics.h"
#include "tiny_mb_constraint_solver_spring.h"
#include "tiny_pd_control.h"
#include "tiny_urdf_to_multi_body.h"

typedef PyBulletVisualizerAPI VisualizerAPI;

int make_sphere(VisualizerAPI* sim, float r = 1, float g = 0.6f, float b = 0,
                float a = 0.8f) {
  int sphere_id = sim->loadURDF("sphere_small.urdf");
  b3RobotSimulatorChangeVisualShapeArgs vargs;
  vargs.m_objectUniqueId = sphere_id;
  vargs.m_hasRgbaColor = true;
  vargs.m_rgbaColor = btVector4(r, g, b, a);
  sim->changeVisualShape(vargs);
  return sphere_id;
}

void print(const std::vector<double>& v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    printf("%.3f", v[i]);
    if (i < v.size() - 1) {
      printf(", ");
    }
  }
  printf("\n");
}

void update_position(VisualizerAPI* sim, int object_id, double x, double y,
                     double z) {
  btVector3 pos(x, y, z);
  btQuaternion orn;
  sim->resetBasePositionAndOrientation(object_id, pos, orn);
}

template <typename Scalar, typename Util>
void update_position(VisualizerAPI* sim, int object_id,
                     const TinyVector3<Scalar, Util>& pos) {
  update_position(sim, object_id, Util::getDouble(pos.m_x),
                  Util::getDouble(pos.m_y), Util::getDouble(pos.m_z));
}

int main(int argc, char* argv[]) {
  double dt = 1. / 1000;

  // btVector3 laikago_initial_pos(0, 0.2, 1.65);
  btVector3 laikago_initial_pos = btVector3(0, 0, .55);
  // btVector3 laikago_initial_pos(-3, 2, .65);
  // btVector3 laikago_initial_pos(2, 0, .65);
  btQuaternion laikago_initial_orn(0, 0, 0, 1);
  // laikago_initial_orn.setEulerZYX(-0.1, 0.1, 0);
  // laikago_initial_orn.setEulerZYX(0.7, 0, 0);
  double initialXvel = 0;
  btVector3 initialAngVel(0, 0, 0);
  double knee_angle = -0.5;
  double abduction_angle = 0.2;
  double initial_poses[] = {
      abduction_angle, 0., knee_angle, abduction_angle, 0., knee_angle,
      abduction_angle, 0., knee_angle, abduction_angle, 0., knee_angle,
  };

  std::string connection_mode = "gui";

  std::string laikago_filename;
  TinyFileUtils::find_file("laikago/laikago_toes_zup.urdf", laikago_filename);

  std::string plane_filename;
  TinyFileUtils::find_file("plane_implicit.urdf", plane_filename);

  char path[TINY_MAX_EXE_PATH_LEN];
  TinyFileUtils::extract_path(plane_filename.c_str(), path,
                              TINY_MAX_EXE_PATH_LEN);
  std::string search_path = path;

  printf("search_path=%s\n", search_path.c_str());
  VisualizerAPI* sim2 = new VisualizerAPI();
  bool isConnected2 =
      sim2->connect(eCONNECT_DIRECT);  // eCONNECT_SHARED_MEMORY);
  sim2->setAdditionalSearchPath(search_path.c_str());

  VisualizerAPI* sim = new VisualizerAPI();
  printf("mode=%s\n", connection_mode.c_str());
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "shared_memory") mode = eCONNECT_SHARED_MEMORY;

  bool isConnected = sim->connect(mode);
  sim->setTimeOut(1e30);
  sim2->setTimeOut(1e30);
  sim->setAdditionalSearchPath(search_path.c_str());
  int logId = sim->startStateLogging(STATE_LOGGING_PROFILE_TIMINGS,
                                     "/tmp/laikago_timing.json");

  if (!isConnected || !isConnected2) {
    printf("Cannot connect\n");
    return -1;
  }

  sim->setTimeOut(1e30);
  sim->resetSimulation();

  b3Clock clock;

  int rotateCamera = 0;

  TinyWorld<double, DoubleUtils> world;
  typedef ::TinyRigidBody<double, DoubleUtils> TinyRigidBodyDouble;
  typedef ::TinyVector3<double, DoubleUtils> TinyVector3;
  typedef ::TinySpatialTransform<double, DoubleUtils> TinySpatialTransform;

  std::vector<TinyRigidBody<double, DoubleUtils>*> bodies;
  std::vector<int> visuals;

  std::vector<TinyMultiBody<double, DoubleUtils>*> mbbodies;
  std::vector<int> paramUids;

  int grav_id = sim->addUserDebugParameter("gravity", -10, 0, -9.8);
  int kp_id = sim->addUserDebugParameter("kp", 0, 2000, 550);
  int kd_id = sim->addUserDebugParameter("kd", 0, 100, 3.);
  int force_id = sim->addUserDebugParameter("max force", 0, 15000, 550);

  {
    TinyMultiBody<double, DoubleUtils>* mb = world.create_multi_body();
    int robotId = sim->loadURDF(plane_filename);
    TinyUrdfStructures<double, DoubleUtils> urdf_data;
    PyBulletUrdfImport<double, DoubleUtils>::extract_urdf_structs(
        urdf_data, robotId, *sim, *sim);
    TinyUrdfToMultiBody<double, DoubleUtils>::convert_to_multi_body(urdf_data,
                                                                    world, *mb);
    mb->initialize();
    sim->removeBody(robotId);
  }

  TinyMultiBody<double, DoubleUtils>* mb = world.create_multi_body();
  {
    b3RobotSimulatorLoadUrdfFileArgs args;
    args.m_flags |= URDF_MERGE_FIXED_LINKS;
    int robotId = sim->loadURDF(laikago_filename, args);

    TinyUrdfStructures<double, DoubleUtils> urdf_data;
    PyBulletUrdfImport<double, DoubleUtils>::extract_urdf_structs(
        urdf_data, robotId, *sim, *sim);
    TinyUrdfToMultiBody<double, DoubleUtils>::convert_to_multi_body(urdf_data,
                                                                    world, *mb);

    mbbodies.push_back(mb);
    mb->m_isFloating = true;
    mb->initialize();
    sim->removeBody(robotId);
    // mb->m_q[5] = 3;
    int start_index = 0;
    start_index = 7;
    mb->m_q[0] = laikago_initial_orn[0];
    mb->m_q[1] = laikago_initial_orn[1];
    mb->m_q[2] = laikago_initial_orn[2];
    mb->m_q[3] = laikago_initial_orn[3];

    mb->m_q[4] = laikago_initial_pos[0];
    mb->m_q[5] = laikago_initial_pos[1];
    mb->m_q[6] = laikago_initial_pos[2];

    mb->m_qd[0] = initialAngVel[0];
    mb->m_qd[1] = initialAngVel[1];
    mb->m_qd[2] = initialAngVel[2];
    mb->m_qd[3] = initialXvel;
    if (mb->m_q.size() >= 12) {
      for (int cc = 0; cc < 12; cc++) {
        mb->m_q[start_index + cc] = initial_poses[cc];
      }
    }
  }
  world.default_friction = 1.2;
  auto* contact_model =
      new TinyMultiBodyConstraintSolverSpring<double, DoubleUtils>;
  world.m_mb_constraint_solver = contact_model;
  contact_model->spring_k = 80000;
  contact_model->damper_d = 5000;


  // contact_model->spring_k = 80000;
  // contact_model->damper_d = 14000;

  printf("Initial state:\n");
  mb->print_state();

  std::vector<double> q_target = mb->m_q;
  // body indices of feet
  const int foot_fr = 2;
  const int foot_fl = 5;
  const int foot_br = 8;
  const int foot_bl = 11;
  const TinyVector3 foot_offset(0, -0.24, -0.02);

  auto* servo = new TinyServoActuator<double, DoubleUtils>(
      mb->dof_actuated(), 150., 30., -500., 500.);
  mb->m_actuator = servo;

  std::vector<std::vector<double>> q_targets;

  double jointOffsets[] = {0, -0.7, 0.7, 0, -0.7, 0.7,
                           0, -0.7, 0.7, 0, -0.7, 0.7};
  double jointDirections[] = {-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1};

  bool load_data1 = false;

  std::string data_filename;
  if (load_data1) {
    TinyFileUtils::find_file("laikago/data1.txt", data_filename);
    std::ifstream data_file(data_filename);
    std::string data_line;
    while (std::getline(data_file, data_line)) {
      std::istringstream iss(data_line);
      int row, time;
      iss >> row;
      iss.get();
      iss >> time;
      iss.get();
      std::vector<double> q_target(12);
      for (int i = 0; i < 12; ++i) {
        iss >> q_target[i];
        if (i < 11) {
          iss.get();
        }
      }
      for (int i = 0; i < 12; ++i) {
        q_target[i] = jointDirections[i] * q_target[i] + jointOffsets[i];
      }
      q_targets.push_back(q_target);
    }
  } else {
    TinyFileUtils::find_file("laikago/mpc_walk.txt", data_filename);
    std::ifstream data_file(data_filename);
    std::string data_line;
    while (std::getline(data_file, data_line)) {
      std::istringstream iss(data_line);
      int row;
      iss >> row;
      iss.get();
      double time;
      iss >> time;
      iss.get();
      std::vector<double> q_target(12);
      for (int i = 0; i < 12; ++i) {
        iss >> q_target[i];
        if (i < 11) {
          iss.get();
        }
      }
      q_targets.push_back(q_target);
    }
  }

  sim->setTimeStep(dt);
  double time = 0;

  int substeps = 4;

  std::vector<double> control(mb->dof_actuated());
  std::vector<double> old_tau(mb->dof_actuated());
  for (int step = 0; sim->isConnected(); ++step) {
    sim->submitProfileTiming("loop");
    {
      sim->submitProfileTiming("sleep_for");
      std::this_thread::sleep_for(std::chrono::duration<double>(dt));
      sim->submitProfileTiming("");
    }
    double gravZ = sim->readUserDebugParameter(grav_id);
    sim->setGravity(btVector3(0, 0, gravZ));

    {
      double gravZ = sim->readUserDebugParameter(grav_id);
      world.set_gravity(TinyVector3(0, 0, gravZ));

      mb->print_state();
      {
        sim->submitProfileTiming("forward_kinematics");
        mb->forward_kinematics();
        sim->submitProfileTiming("");
      }

      {
        TinySpatialTransform base_X_world;
        std::vector<TinySpatialTransform> links_X_world;
        mb->forward_kinematics_q(mb->m_q, &base_X_world, &links_X_world);
      }

      {
        mb->forward_kinematics();
        servo->kp = sim->readUserDebugParameter(kp_id);
        servo->kd = sim->readUserDebugParameter(kd_id);
        servo->max_force = sim->readUserDebugParameter(force_id);
        servo->min_force = -servo->max_force;

        int skip_step = load_data1 ? 4 : 1;

        const std::vector<double>& q_target =
            q_targets[(step / skip_step) % q_targets.size()];
        mb->control(dt, q_target);
      }

      {
        sim->submitProfileTiming("forwardDynamics");
        mb->forward_dynamics(world.get_gravity());
        sim->submitProfileTiming("");
        mb->clear_forces();
      }

      {
        sim->submitProfileTiming("integrate_q");
        mb->integrate_q(dt);  //??
        sim->submitProfileTiming("");
      }

      {
        if (step % 1000 == 0) {
          printf("Step %06d \t Time: %.3f\n", step, time);
        }
        sim->submitProfileTiming("world_step");
        // mb->print_state();
        world.step(dt);
        // mb->print_state();

        sim->submitProfileTiming("");
        time += dt;
      }

      {
        sim->submitProfileTiming("integrate");
        mb->integrate(dt);
        sim->submitProfileTiming("");
      }
      if (1) {
        sim->submitProfileTiming("sync graphics");

        // sync physics to visual transforms
        {
          for (int b = 0; b < mbbodies.size(); b++) {
            const TinyMultiBody<double, DoubleUtils>* body = mbbodies[b];
            PyBulletUrdfImport<double, DoubleUtils>::sync_graphics_transforms(
                body, *sim);
          }
        }
        sim->submitProfileTiming("");
      }
    }
    {
      sim->submitProfileTiming("get_keyboard_events");
      b3KeyboardEventsData keyEvents;
      sim->getKeyboardEvents(&keyEvents);
      if (keyEvents.m_numKeyboardEvents) {
        for (int i = 0; i < keyEvents.m_numKeyboardEvents; i++) {
          b3KeyboardEvent& e = keyEvents.m_keyboardEvents[i];

          if (e.m_keyCode == 'r' && e.m_keyState & eButtonTriggered) {
            rotateCamera = 1 - rotateCamera;
          }
        }
      }
      sim->submitProfileTiming("");
    }

    if (rotateCamera) {
      sim->submitProfileTiming("rotateCamera");
      static double yaw = 0;
      double distance = 1;
      yaw += 0.1;
      btVector3 basePos(0, 0, 0);
      btQuaternion baseOrn(0, 0, 0, 1);
      sim->resetDebugVisualizerCamera(distance, -20, yaw, basePos);
      sim->submitProfileTiming("");
    }
    sim->submitProfileTiming("");
  }

  sim->stopStateLogging(logId);
  printf("sim->disconnect\n");

  sim->disconnect();

  printf("delete sim\n");
  delete sim;

  printf("exit\n");
}

#pragma clang diagnostic pop