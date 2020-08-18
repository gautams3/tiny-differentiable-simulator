#include <fstream>

#include "opengl_window/tiny_opengl3_app.h"
#include "pendulum.h"
#include "tiny_ceres_estimator.h"
#include "tiny_file_utils.h"
#include "tiny_mb_constraint_solver_spring.h"
#include "tiny_multi_body.h"
#include "tiny_system_constructor.h"
#include "tiny_world.h"

// whether the state consists of [q qd] or just q
#define STATE_INCLUDES_QD true
std::vector<double> start_state;
const int analytical_param_dim = 2;
const int param_dim = analytical_param_dim;

#ifdef USE_MATPLOTLIB
template <typename T>
void plot_trajectory(const std::vector<std::vector<T>> &states) {
  typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils,
                             CeresUtils<param_dim>>
      Utils;
  for (int i = 0; i < static_cast<int>(states[0].size()); ++i) {
    std::vector<double> traj(states.size());
    for (int t = 0; t < static_cast<int>(states.size()); ++t) {
      traj[t] = Utils::getDouble(states[t][i]);
    }
    plt::named_plot("state[" + std::to_string(i) + "]", traj);
  }
  plt::legend();
  plt::show();
}
#endif

template <typename Scalar, typename Utils>
std::vector<std::vector<double>> to_double_states(
    const std::vector<std::vector<Scalar>> &states) {
  if constexpr (std::is_same_v<Scalar, double>) {
    return states;
  }
  std::vector<std::vector<double>> double_states(states.size());
  for (int t = 0; t < static_cast<int>(states.size()); ++t) {
    std::vector<double> double_state(states[t].size());
    for (int i = 0; i < static_cast<int>(states[t].size()); ++i) {
      double_state[i] = Utils::getDouble(states[t][i]);
    }
    double_states[t] = double_state;
  }
  return double_states;
}

template <typename Scalar1 = double, typename Utils1 = DoubleUtils,
          typename Scalar2 = double, typename Utils2 = DoubleUtils>
void visualize_traces(const std::vector<std::vector<Scalar1>> &our_states_raw,
                      const std::vector<std::vector<Scalar1>> &ini_states_raw,
                      const std::vector<std::vector<Scalar2>> &ref_states_raw,
                      int num_states = 10) {
  auto our_states = to_double_states<Scalar1, Utils1>(our_states_raw);
  auto ini_states = to_double_states<Scalar1, Utils1>(ini_states_raw);
  auto ref_states = to_double_states<Scalar2, Utils2>(ref_states_raw);

  TinyOpenGL3App app(
      "Trajectories (green = reference, blue = initial, orange = ours)", 1024,
      768);
  app.m_renderer->init();
  app.set_up_axis(2);
  app.m_renderer->get_active_camera()->set_camera_distance(4);
  app.m_renderer->get_active_camera()->set_camera_pitch(-30);
  app.m_renderer->get_active_camera()->set_camera_target_position(0, 0, 0);

  TinyWorld<double, DoubleUtils> world;
  TinyMultiBody<double, DoubleUtils> *system = world.create_multi_body();
  system->m_isFloating = true;
  system->initialize();

  int cube_shape = app.register_cube_shape(1.f, 1.f, 1.f);
  TinyVector3f ref_color(0.3, 0.8, 0.);
  TinyVector3f our_color(1.0, 0.6, 0.);
  TinyVector3f ini_color(0.1, 0.6, 1.);
  float opacity = 0.5f;
  float scale = 0.5f;
  TinyVector3f scaling(scale, scale, scale);

  // show our states
  int i = 0;
  int show_every = static_cast<int>(our_states.size()) / num_states + 1;
  for (const std::vector<double> &state : our_states) {
    if (i++ % show_every != 0) {
      continue;
    }
    TinyVector3f pos(state[4], state[5], state[6]);
    TinyQuaternionf orn(state[0], state[1], state[2], state[3]);
    app.m_renderer->register_graphics_instance(cube_shape, pos, orn, our_color,
                                               scaling, opacity);
  }

  // show reference states
  i = 0;
  for (const std::vector<double> &state : ref_states) {
    if (i++ % show_every != 0) {
      continue;
    }
    TinyVector3f pos(state[4], state[5], state[6]);
    TinyQuaternionf orn(state[0], state[1], state[2], state[3]);
    app.m_renderer->register_graphics_instance(cube_shape, pos, orn, ref_color,
                                               scaling, opacity);
  }

  // show initial states
  i = 0;
  for (const std::vector<double> &state : ini_states) {
    if (i++ % show_every != 0) {
      continue;
    }
    TinyVector3f pos(state[4], state[5], state[6]);
    TinyQuaternionf orn(state[0], state[1], state[2], state[3]);
    app.m_renderer->register_graphics_instance(cube_shape, pos, orn, ini_color,
                                               scaling, opacity);
  }

  while (!app.m_window->requested_exit()) {
    app.m_renderer->update_camera(2);
    DrawGridData data;
    data.upAxis = 2;
    app.draw_grid(data);

    app.m_renderer->render_scene();
    app.m_renderer->write_transforms();
    app.swap_buffer();
  }
}

template <typename Scalar = double, typename Utils = DoubleUtils>
void visualize_trajectory(const std::vector<std::vector<Scalar>> &states_raw,
                          const Scalar &dt, const char *window_title) {
  auto states = to_double_states<Scalar, Utils>(states_raw);

  TinyOpenGL3App app(window_title, 1024, 768);
  app.m_renderer->init();
  app.set_up_axis(2);
  app.m_renderer->get_active_camera()->set_camera_distance(4);
  app.m_renderer->get_active_camera()->set_camera_pitch(-30);
  app.m_renderer->get_active_camera()->set_camera_target_position(0, 0, 0);

  TinyWorld<double, DoubleUtils> world;
  TinyMultiBody<double, DoubleUtils> *system = world.create_multi_body();
  system->m_isFloating = true;
  system->initialize();

  int cube_shape = app.register_cube_shape(1.f, 1.f, 1.f);

  const std::vector<double> &state0 = states[0];
  TinyVector3f pos(0, 0, 0);
  TinyQuaternionf orn(0, 0, 0, 1);
  TinyVector3f color(0.2, 0.6, 1);
  float scale = 0.5f;
  TinyVector3f scaling(scale, scale, scale);
  int cube_id = app.m_renderer->register_graphics_instance(cube_shape, pos, orn,
                                                           color, scaling);
  for (const std::vector<double> &state : states) {
    app.m_renderer->update_camera(2);
    DrawGridData data;
    data.upAxis = 2;
    app.draw_grid(data);
    for (int i = 0; i < 7; ++i) {
      system->m_q[i] = state[i];
    }
    system->forward_kinematics();
    // system->print_state();

    std::this_thread::sleep_for(
        std::chrono::duration<double>(Utils::getDouble(dt)));
    // sync transform
    TinyQuaternion<double, DoubleUtils> rot;
    const TinySpatialTransform<double, DoubleUtils> &geom_X_world =
        system->m_base_X_world;
    TinyVector3f base_pos(geom_X_world.m_translation.getX(),
                          geom_X_world.m_translation.getY(),
                          geom_X_world.m_translation.getZ());
    geom_X_world.m_rotation.getRotation(rot);
    TinyQuaternionf base_orn(rot.getX(), rot.getY(), rot.getZ(), rot.getW());
    app.m_renderer->write_single_instance_transform_to_cpu(base_pos, base_orn,
                                                           cube_id);
    app.m_renderer->render_scene();
    app.m_renderer->write_transforms();
    app.swap_buffer();
  }
}

struct rollout_dynamics {
  int call_counter{0};

  TinySystemConstructor<> constructor;
  rollout_dynamics(const std::string &urdf_filename,
                   const std::string &plane_filename)
      : constructor(urdf_filename, plane_filename) {
    constructor.m_is_floating = true;
  }

  /**
   * Roll-out cube contact dynamics, and compute states [q, qd].
   */
  template <typename Scalar = double, typename Utils = DoubleUtils>
  void operator()(const std::vector<Scalar> &params,
                  std::vector<std::vector<Scalar>> &output_states,
                  int time_steps, double dt) {
    TinyVector3<Scalar, Utils> gravity(Utils::zero(), Utils::zero(),
                                       Utils::fraction(-981, 100));
    output_states.resize(time_steps);
    TinyWorld<Scalar, Utils> world;
    TinyMultiBody<Scalar, Utils> *mb;
    constructor(world, &mb);

    world.set_gravity(gravity);
    world.default_friction = Utils::zero();

    int x0_size = static_cast<int>(start_state.size());
    if (x0_size >= mb->dof()) {
      for (int i = 0; i < mb->dof(); ++i) {
        mb->m_q[i] = Utils::scalar_from_double(start_state[i]);
      }
      if (x0_size >= mb->dof() + mb->dof_qd()) {
        for (int i = 0; i < mb->dof_qd(); ++i) {
          mb->m_qd[i] = Utils::scalar_from_double(start_state[i + mb->dof()]);
        }
      }
    }
    for (int t = 0; t < time_steps; ++t) {
#if STATE_INCLUDES_QD
      output_states[t].resize(mb->dof() + mb->dof_qd());
#else
      output_states[t].resize(mb->dof());
#endif
      for (int i = 0; i < mb->dof(); ++i) {
        output_states[t][i] = mb->m_q[i];
      }
#if STATE_INCLUDES_QD
      for (int i = 0; i < mb->dof_qd(); ++i) {
        output_states[t][i + mb->dof()] = mb->m_qd[i];
      }
#endif
      mb->forward_dynamics(gravity);
      mb->clear_forces();
      mb->integrate_q(Utils::scalar_from_double(dt));
      world.step(Utils::scalar_from_double(dt));
      mb->integrate(Utils::scalar_from_double(dt));
      // mb->print_state();
    }
  }
};

template <ResidualMode ResMode>
class ContactEstimator
    : public TinyCeresEstimator<param_dim, 7 + STATE_INCLUDES_QD * 6, ResMode> {
 public:
  typedef TinyCeresEstimator<param_dim, 7 + STATE_INCLUDES_QD * 6, ResMode>
      CeresEstimator;
  using CeresEstimator::kStateDim, CeresEstimator::kParameterDim;
  using CeresEstimator::parameters;
  using typename CeresEstimator::ADScalar;

  std::vector<double> initial_params;

  int time_steps;

  rollout_dynamics *sampler;

  ContactEstimator(int time_steps, double dt)
      : CeresEstimator(dt), time_steps(time_steps) {
    std::string urdf_filename, plane_filename;
    TinyFileUtils::find_file("sphere8cube.urdf", urdf_filename);
    TinyFileUtils::find_file("plane_implicit.urdf", plane_filename);
    sampler = new rollout_dynamics(urdf_filename, plane_filename);
    parameters[0] = {"spring_k", 0., 0., 20000.};
    parameters[1] = {"damper_d", 0., 0., 20000.};
    for (const auto &p : parameters) {
      initial_params.push_back(p.value);
    }
  }
};

template <typename Scalar = double, typename Utils = DoubleUtils>
void print_states(const std::vector<std::vector<Scalar>> &states_raw) {
  auto states = to_double_states<Scalar, Utils>(states_raw);
  for (const auto &s : states) {
    for (double d : s) {
      printf("%.2f ", d);
    }
    printf("\n");
  }
}

template <typename Scalar = double, typename Utils = DoubleUtils>
void save_states(const std::string &filename,
                 const std::vector<std::vector<Scalar>> &states_raw,
                 double dt) {
  auto states = to_double_states<Scalar, Utils>(states_raw);
  std::ofstream traj_file(filename);
  int time_steps = static_cast<int>(states.size());
  for (int t = 0; t < time_steps; ++t) {
    traj_file << (t * dt);
    for (double v : states[t]) {
      traj_file << "\t" << v;
    }
    traj_file << "\n";
  }
  traj_file.close();
  printf("Saved %i states to %s.\n", time_steps, filename.c_str());
}

int main(int argc, char *argv[]) {
  typedef ContactEstimator<RES_MODE_1D> Estimator;

  const double dt = 1. / 100;
  const double time_limit = 3;
  const int time_steps = time_limit / dt;
  const double initial_height = 0.4; //box on floor
  const TinyVector3<double, DoubleUtils> initial_velocity(3., 0., 0.);
  srand(123);

  google::InitGoogleLogging(argv[0]);

  // keep the target_times empty, since target time steps match the model
  std::vector<double> target_times;
  std::vector<std::vector<double>> target_states;
  // rollout pendulum with damping
  std::vector<double> empty_params;
  std::string urdf_filename, plane_filename;
  TinyFileUtils::find_file("sphere8cube.urdf", urdf_filename);
  TinyFileUtils::find_file("plane_implicit.urdf", plane_filename);
  rollout_dynamics sampler(urdf_filename, plane_filename);
  TinyQuaternion<double, DoubleUtils> start_rot;
  start_rot.set_euler_rpy(TinyVector3<double, DoubleUtils>(0., 0., 0.));
  start_state = {start_rot.x(),
                 start_rot.y(),
                 start_rot.z(),
                 start_rot.w(),
                 0.,
                 0.,
                 initial_height,
                 0.,
                 0.,
                 0.,
                 initial_velocity.x(),
                 initial_velocity.y(),
                 initial_velocity.z()};
  sampler(empty_params, target_states, time_steps, dt);
  save_states("simple_box_ref.csv", target_states, dt);
  visualize_trajectory(target_states, dt, "Reference trajectory");
  const std::vector<std::vector<double>> ref_states = target_states;
  return EXIT_SUCCESS;
}