#include <assert.h>
#include <stdio.h>

#include <chrono>
#include <thread>

#include "Utils/b3Clock.h"
#include "motion_import.h"
#include "pybullet_urdf_import.h"
#include "pybullet_visualizer_api.h"
#include "tiny_double_utils.h"
#include "tiny_file_utils.h"
#include "tiny_inverse_kinematics.h"
#include "tiny_mb_constraint_solver_spring.h"
#include "tiny_pd_control.h"
#include "tiny_urdf_to_multi_body.h"
#include "tiny_ceres_estimator.h"

typedef PyBulletVisualizerAPI VisualizerAPI;

// whether to use Parallel Basin Hopping
#define USE_PBH true
#define USE_NEURAL_AUGMENTATION true
#define RESIDUAL_PHYSICS true
const int state_dim = 3;
const ResidualMode res_mode = RES_MODE_1D;
const bool assign_analytical_params = false;
const std::size_t analytical_param_dim = 3;
const int num_files = 50;  // 100;
const int param_dim = 2;

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

struct Estimator 
    : public TinyCeresEstimator<param_dim, state_dim, res_mode> {
  typedef TinyCeresEstimator<param_dim, state_dim, res_mode> CeresEstimator;
  using CeresEstimator::kStateDim, CeresEstimator::kParameterDim;
  using CeresEstimator::parameters, CeresEstimator::dt;
  using typename CeresEstimator::ADScalar;

  std::vector<double> initial_params;

  // stores trajectories per surface-shape combination
  std::vector<PushData> trajectories;

  mutable std::map<std::string, Laboratory<NDScalar, NDUtils> *> labs_double;
  mutable std::map<std::string, Laboratory<NAScalar, NAUtils> *> labs_ad;

  template <typename Scalar, typename Utils>
void rollout(
      const std::vector<Scalar> &params,
      std::vector<std::vector<Scalar>> &output_states, double &dt,
      std::size_t ref_id) const {  //}, VisualizerAPI *sim = nullptr) const {
    // printf("Rollout with parameters:");
    // for (const Scalar &p : params) {
    //   printf("\t%.5f", Utils::getDouble(p));
    // }
    // printf("\n");
    if constexpr (is_neural_scalar<Scalar, Utils>::value) {
      typedef typename Scalar::InnerScalarType IScalar;
      typedef typename Scalar::InnerUtilsType IUtils;
      std::vector<IScalar> iparams(params.size());
      for (std::size_t i = 0; i < params.size(); ++i) {
        iparams[i] = params[i].evaluate();
      }
      neural_augmentation.template instantiate<IScalar, IUtils>(
          iparams, analytical_param_dim);
    }

    const auto &data = trajectories[ref_id];
    dt = data.dt * skip_steps;
    this->dt = dt;

    Laboratory<Scalar, Utils> &lab =
        this->template get_lab<Scalar, Utils>(data.lab_name);
    auto *tip = lab.tip;
    tip->initialize();

    if (assign_analytical_params) {
      lab.world_ground.default_friction = params[0];
      lab.contact_ground->mu_static = params[1];
      lab.contact_ground->v_transition = params[2];
    }

    auto *object = lab.object;
    object->initialize();
    object->m_q[0] = Utils::scalar_from_double(data.object_x[0]);
    object->m_q[1] = Utils::scalar_from_double(data.object_y[0]);
    object->m_q[2] = Utils::scalar_from_double(0.005);  // initial object height
    object->m_q[3] = Utils::scalar_from_double(data.object_yaw[0]);
    object->forward_kinematics();

    auto &world_ground = lab.world_ground;

    double sim_dt = data.dt;
    const Scalar sdt = Utils::scalar_from_double(sim_dt);

    for (std::size_t i = 0; i < data.time.size(); ++i) {
      tip->m_q[0] = Utils::scalar_from_double(data.tip_x[i]);
      tip->m_q[1] = Utils::scalar_from_double(data.tip_y[i]);
      if (i > 0) {
        tip->m_qd[0] = Utils::scalar_from_double(
            (data.tip_x[i] - data.tip_x[i - 1]) / sim_dt);
        tip->m_qd[1] = Utils::scalar_from_double(
            (data.tip_y[i] - data.tip_y[i - 1]) / sim_dt);
      }
      tip->forward_kinematics();

      object->forward_dynamics(world_ground.get_gravity());
      object->clear_forces();
      object->integrate_q(sdt);

      transform_points<Scalar, Utils>(lab.object_exterior,
                                      lab.tf_object_exterior, object->m_q[0],
                                      object->m_q[1], object->m_q[3]);
      auto tip_contact =
          compute_contact<Scalar, Utils>(tip, object, lab.tf_object_exterior);

      // XXX friction between tip and object
      tip_contact.m_friction =
          Utils::scalar_from_double(0.25);  // Utils::scalar_from_double(1);
      lab.tip_contact_model.erp = Utils::scalar_from_double(0.001);
      lab.tip_contact_model.cfm = Utils::scalar_from_double(0.000001);

      lab.tip_contact_model.resolveCollision({tip_contact}, sdt);

      world_ground.step(sdt);
      object->integrate(sdt);
      // object->print_state();

      if (i % skip_steps == 0) {
        const Scalar &object_x = object->m_q[0];
        const Scalar &object_y = object->m_q[1];
        const Scalar &object_yaw = object->m_q[3];

        Scalar out_object_x = object->m_q[0];
        Scalar out_object_y = object->m_q[1];
        Scalar out_object_yaw = object->m_q[3];

#if RESIDUAL_PHYSICS
        if constexpr (is_neural_scalar<Scalar, Utils>::value) {
          Scalar tip_force_x = Utils::scalar_from_double(data.force_x[i]);
          Scalar tip_force_y = Utils::scalar_from_double(data.force_y[i]);
          Scalar tip_force_yaw = Utils::scalar_from_double(data.force_yaw[i]);
          tip_force_x.assign("tip_force_x");
          tip_force_y.assign("tip_force_y");
          tip_force_yaw.assign("tip_force_yaw");

          object_x.assign("in_object_x");
          object_y.assign("in_object_y");
          object_yaw.assign("in_object_yaw");

          out_object_x.assign("out_object_x");
          out_object_y.assign("out_object_y");
          out_object_yaw.assign("out_object_yaw");

          out_object_x.evaluate();
          out_object_y.evaluate();
          out_object_yaw.evaluate();
        }
#endif

        output_states.push_back({out_object_x, out_object_y, out_object_yaw});

        if (sim) {
          object->m_q[0] = out_object_x;
          object->m_q[1] = out_object_y;
          object->m_q[3] = out_object_yaw;
          object->forward_kinematics();

          // if constexpr (is_neural_scalar<Scalar, Utils>::value) {
          //   if (i == 20) {Scalar::print_neural_networks();}
          // }

          // output_states.push_back(
          //     {object->m_q[0], object->m_q[1], object->m_q[3]});

          TinyMultiBody<Scalar, Utils> *true_object = lab.true_object;
          true_object->m_q[0] = Utils::scalar_from_double(data.object_x[i]);
          true_object->m_q[1] = Utils::scalar_from_double(data.object_y[i]);
          true_object->m_q[3] = Utils::scalar_from_double(data.object_yaw[i]);
          true_object->forward_kinematics();
          object->forward_kinematics();

          PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(tip,
                                                                      *sim);
          PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(object,
                                                                      *sim);
          PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(
              true_object, *sim);
          std::this_thread::sleep_for(std::chrono::duration<double>(sim_dt));
          object->m_q[0] = object_x;
          object->m_q[1] = object_y;
          object->m_q[3] = object_yaw;
        }
      }
    }
  }
};

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
  int kp_id = sim->addUserDebugParameter("kp", 0, 400, 180);
  int kd_id = sim->addUserDebugParameter("kd", 0, 13, 3.);
  int force_id = sim->addUserDebugParameter("max force", 0, 1500, 550);

  Motion reference;
  std::string motion_filename;
  TinyFileUtils::find_file("laikago_dance_sidestep0.txt", motion_filename);
  bool load_success = Motion::load_from_file(motion_filename, &reference);

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
  world.default_friction = 1.;
  auto* contact_model =
      new TinyMultiBodyConstraintSolverSpring<double, DoubleUtils>;
  world.m_mb_constraint_solver = contact_model;
  contact_model->spring_k = 70000;
  contact_model->damper_d = 5000;

  printf("Initial state:\n");
  mb->print_state();

  std::vector<double> q_target = mb->m_q;
  // body indices of feet
  const int foot_fr = 2;
  const int foot_fl = 5;
  const int foot_br = 8;
  const int foot_bl = 11;
  const TinyVector3 foot_offset(0, -0.24, -0.02);

  int walking_start = 500;  // step number when to start walking (first settle)

  auto* servo = new TinyServoActuator<double, DoubleUtils>(
      mb->dof_actuated(), 150., 3., -500., 500.);
  mb->m_actuator = servo;

  sim->setTimeStep(dt);
  double time = 0;

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
      {
        sim->submitProfileTiming("forward_kinematics");
        mb->forward_kinematics();
        sim->submitProfileTiming("");
      }

      {
        TinySpatialTransform base_X_world;
        std::vector<TinySpatialTransform> links_X_world;
        mb->forward_kinematics_q(mb->m_q, &base_X_world, &links_X_world);

        if (step > walking_start) {
          q_target = reference.calculate_frame(time);
        }
      }

      {
        mb->forward_kinematics();
        servo->kp = sim->readUserDebugParameter(kp_id);
        servo->kd = sim->readUserDebugParameter(kd_id);
        servo->max_force = sim->readUserDebugParameter(force_id);
        servo->min_force = -servo->max_force;

        for (int i = 0; i < mb->dof_actuated(); ++i) {
          control[i] = q_target[i + 7];
        }

        // if (time > 3) {
        //   for (int i = 0; i < mb->dof_actuated(); ++i) {
        //     mb->m_tau[i] = old_tau[i];
        //   }
        // } else {
        mb->control(dt, control);
        for (int i = 0; i < mb->dof_actuated(); ++i) {
          old_tau[i] = mb->m_tau[i];
        }
        // }
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