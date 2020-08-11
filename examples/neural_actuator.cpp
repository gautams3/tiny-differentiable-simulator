#include <assert.h>
#include <stdio.h>

#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <thread>

#include "motion_import.h"
#include "neural_augmentation.h"
#include "pybullet_urdf_import.h"
#include "pybullet_visualizer_api.h"
#include "tiny_ceres_estimator.h"
#include "tiny_double_utils.h"
#include "tiny_file_utils.h"
#include "tiny_inverse_kinematics.h"
#include "tiny_mb_constraint_solver_spring.h"
#include "tiny_pd_control.h"
#include "tiny_system_constructor.h"
#include "tiny_urdf_to_multi_body.h"

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
constexpr double kDT = 1e-3;

typedef ceres::Jet<double, param_dim> ADScalar;
typedef CeresUtils<param_dim> ADUtils;

#if USE_NEURAL_AUGMENTATION
typedef NeuralScalar<double, DoubleUtils> NDScalar;
typedef NeuralScalarUtils<double, DoubleUtils> NDUtils;
typedef NeuralScalar<ADScalar, ADUtils> NAScalar;
typedef NeuralScalarUtils<ADScalar, ADUtils> NAUtils;
#else
typedef double NDScalar;
typedef DoubleUtils NDUtils;
typedef ADScalar NAScalar;
typedef ADUtils NAUtils;
#endif

void print(const std::vector<double>& v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    printf("%.3f", v[i]);
    if (i < v.size() - 1) {
      printf(", ");
    }
  }
  printf("\n");
}

template <typename Scalar, typename Utils>
struct Laboratory {
  TinyWorld<Scalar, Utils> world;
  TinyMultiBody<Scalar, Utils>* mb;
  TinyMultiBody<Scalar, Utils>* ground{nullptr};

  Laboratory(TinyUrdfCache<Scalar, Utils>& urdf_cache,
             const std::string& plane_urdf_filename,
             const std::string& laikago_urdf_filename, VisualizerAPI* sim,
             VisualizerAPI* sim2) {
    bool ignore_cache = true;

    bool is_floating = true;
    mb = urdf_cache.construct(laikago_urdf_filename, world, sim2, sim,
                              ignore_cache, is_floating);
    ground = urdf_cache.construct(plane_urdf_filename, world, sim2, sim,
                                  ignore_cache);

    world.default_friction = Utils::one();
    auto* contact_model =
        new TinyMultiBodyConstraintSolverSpring<Scalar, Utils>;
    world.m_mb_constraint_solver = contact_model;
    contact_model->spring_k = Utils::scalar_from_double(70000);
    contact_model->damper_d = Utils::scalar_from_double(5000);
  }
};

struct Estimator : public TinyCeresEstimator<param_dim, state_dim, res_mode> {
  typedef TinyCeresEstimator<param_dim, state_dim, res_mode> CeresEstimator;
  using CeresEstimator::kStateDim, CeresEstimator::kParameterDim;
  using CeresEstimator::parameters;
  using typename CeresEstimator::ADScalar;

  std::vector<double> initial_params;

  NeuralAugmentation neural_augmentation;

  TinyUrdfCache<NDScalar, NDUtils> urdf_cache_double;
  TinyUrdfCache<NAScalar, NAUtils> urdf_cache_ad;

  VisualizerAPI* sim{nullptr};

  Motion reference;

  mutable Laboratory<NDScalar, NDUtils> lab_double;
  mutable Laboratory<NAScalar, NAUtils> lab_ad;

  int num_time_steps = 8 / kDT;

  Estimator(const std::string& plane_urdf_filename,
            const std::string& laikago_urdf_filename, VisualizerAPI* sim,
            VisualizerAPI* sim2)
      : CeresEstimator(kDT),
        lab_double(urdf_cache_double, plane_urdf_filename,
                   laikago_urdf_filename, sim, sim2),
        lab_ad(urdf_cache_ad, plane_urdf_filename, laikago_urdf_filename, sim,
               sim2),
        sim(sim) {}

  template <typename Scalar, typename Utils>
  constexpr Laboratory<Scalar, Utils>& get_lab(
      const std::string& lab_name) const {
    if constexpr (std::is_same_v<Scalar, NDScalar>) {
      return lab_double;
    } else {
      return lab_ad;
    }
  }

  template <typename Scalar, typename Utils>
  void rollout(const std::vector<Scalar>& params,
               std::vector<std::vector<Scalar>>& output_states, double&,
               std::size_t ref_id) const {
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

    const Scalar dt = Utils::scalar_from_double(kDT);

    Laboratory<Scalar, Utils>& lab = this->template get_lab<Scalar, Utils>();
    TinyMultiBody<Scalar, Utils>* mb = lab.mb;

    double knee_angle = -0.5;
    double abduction_angle = 0.2;
    double initial_poses[] = {
        abduction_angle, 0., knee_angle, abduction_angle, 0., knee_angle,
        abduction_angle, 0., knee_angle, abduction_angle, 0., knee_angle,
    };

    // mb->m_q[5] = 3;
    int start_index = 0;
    start_index = 7;
    mb->initialize();

    mb->m_q[4] = Utils::zero();
    mb->m_q[5] = Utils::zero();
    mb->m_q[6] = Utils::scalar_from_double(0.55);

    if (mb->m_q.size() >= 12) {
      for (int cc = 0; cc < 12; cc++) {
        mb->m_q[start_index + cc] =
            Utils::scalar_from_double(initial_poses[cc]);
      }
    }

    printf("Initial state:\n");
    mb->print_state();

    // step number when to start walking (first settle)
    int walking_start = 500;
    std::vector<Scalar> q_target = mb->m_q;

    if constexpr (std::is_same_v<Scalar, double>) {
      // ground-truth actuator
      auto* servo = new TinyServoActuator<double, DoubleUtils>(
          mb->dof_actuated(), 150., 3., -500., 500.);
      servo->kp = 180;
      servo->kd = 3;
      servo->max_force = 550;
      servo->min_force = -550;
      mb->m_actuator = servo;
    }

    Scalar time = Utils::zero();

    std::vector<Scalar> control(mb->dof_actuated());
    for (int step = 0; num_time_steps; ++step) {
      mb->forward_kinematics();

      if (step > walking_start) {
        q_target = reference.calculate_frame(time);
      }
      mb->forward_kinematics();
      for (int i = 0; i < mb->dof_actuated(); ++i) {
        control[i] = q_target[i + 7];
      }
      mb->forward_dynamics(world.get_gravity());
      mb->clear_forces();

      mb->integrate_q(dt);  //??
      world.step(dt);

      time += dt;

      mb->integrate(dt);

      if (sim) {
        PyBulletUrdfImport<Scalar, Utils>::sync_graphics_transforms(mb, *sim);
        std::this_thread::sleep_for(std::chrono::duration<double>(kDT));
      }
    }

    if (i % skip_steps == 0) {
      const Scalar& object_x = object->m_q[0];
      const Scalar& object_y = object->m_q[1];
      const Scalar& object_yaw = object->m_q[3];

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

        TinyMultiBody<Scalar, Utils>* true_object = lab.true_object;
        true_object->m_q[0] = Utils::scalar_from_double(data.object_x[i]);
        true_object->m_q[1] = Utils::scalar_from_double(data.object_y[i]);
        true_object->m_q[3] = Utils::scalar_from_double(data.object_yaw[i]);
        true_object->forward_kinematics();
        object->forward_kinematics();

        object->m_q[0] = object_x;
        object->m_q[1] = object_y;
        object->m_q[3] = object_yaw;
      }
    }
  }

  void rollout(const std::vector<ADScalar>& params,
               std::vector<std::vector<ADScalar>>& output_states, double& dt,
               std::size_t ref_id) const override {
    printf("rollout AD\n");
#if USE_NEURAL_AUGMENTATION
    typedef NeuralScalar<ADScalar, ADUtils> NScalar;
    typedef NeuralScalarUtils<ADScalar, ADUtils> NUtils;
    auto n_params = NUtils::to_neural(params);
    std::vector<std::vector<NScalar>> n_output_states;
    this->template rollout<NScalar, NUtils>(n_params, n_output_states, dt,
                                            ref_id);
    for (const auto& state : n_output_states) {
      output_states.push_back(NUtils::from_neural(state));
    }
#else
    this->template rollout<ADScalar, ADUtils>(params, output_states, dt,
                                              ref_id);
#endif
  }
  void rollout(const std::vector<double>& params,
               std::vector<std::vector<double>>& output_states, double& dt,
               std::size_t ref_id) const override {
    printf("rollout DOUBLE\n");
#if USE_NEURAL_AUGMENTATION
    typedef NeuralScalar<double, DoubleUtils> NScalar;
    typedef NeuralScalarUtils<double, DoubleUtils> NUtils;
    auto n_params = NUtils::to_neural(params);
    std::vector<std::vector<NScalar>> n_output_states;
    this->template rollout<NScalar, NUtils>(n_params, n_output_states, dt,
                                            ref_id);
    for (const auto& state : n_output_states) {
      output_states.push_back(NUtils::from_neural(state));
    }
#else
    this->template rollout<double, DoubleUtils>(params, output_states, dt,
                                                ref_id);
#endif
  }
};

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);

  cxxopts::Options options("neural_push", "Learn contact friction model");
  options.add_options()("connection_mode", "Connection mode",
                        cxxopts::value<std::string>()->default_value("gui"));

  auto cli_args = options.parse(argc, argv);
  std::string connection_mode = cli_args["connection_mode"].as<std::string>();

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

  // create estimator for single-thread optimization (without PBH) and
  // visualization
  Estimator frontend_estimator;

  frontend_estimator.use_finite_diff = false;
  frontend_estimator.minibatch_size = num_files;  // 50;

  frontend_estimator.sim = sim;
  // frontend_estimator.options.line_search_direction_type =
  // ceres::LineSearchDirectionType::STEEPEST_DESCENT;
  // frontend_estimator.options.line_search_type = ceres::LineSearchType::WOLFE;
  frontend_estimator.options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
  frontend_estimator.set_bounds = false;  // true; //true;
  frontend_estimator.neural_augmentation.weight_limit = 0.05;
  frontend_estimator.neural_augmentation.bias_limit = 0.00001;
  frontend_estimator.neural_augmentation.input_lasso_regularization = 0;
  frontend_estimator.neural_augmentation.upper_l2_regularization = 0;
  frontend_estimator.neural_augmentation.default_hidden_layers = 2;
  frontend_estimator.options.max_num_iterations = 300;

  std::string motion_filename;
  TinyFileUtils::find_file("laikago_dance_sidestep0.txt", motion_filename);
  bool load_success =
      Motion::load_from_file(motion_filename, &frontend_estimator.reference);

  printf("delete sim\n");
  delete sim;
}
