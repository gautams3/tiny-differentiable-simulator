#include <fenv.h>
#include <stdio.h>

#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <thread>

#include "neural_augmentation.h"
#include "neural_push_utils.h"
#include "pybullet_visualizer_api.h"
#include "utils/ceres_estimator.hpp"
#include "tiny_dataset.h"
#include "tiny_double_utils.h"
#include "utils/file_utils.hpp"
#include "multi_body.hpp
#include "tiny_system_constructor.h"

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

#if USE_NEURAL_AUGMENTATION
#if RESIDUAL_PHYSICS
const int param_dim = 86;
#else
const int param_dim = 75;
#endif
#else
const int param_dim = 3;
#endif

const int residual_dim =
    TinyCeresEstimator<param_dim, state_dim, res_mode>::kResidualDim;

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

template <typename Scalar, typename Utils>
struct Laboratory {
  // create duplicate world object to not get any automatic collision response
  // between object and tip (we create our own contact point for this
  // interaction)
  TinyWorld<Scalar, Utils> world_ground, world_tip;
  TinyMultiBody<Scalar, Utils> *ground{nullptr};
  TinyMultiBody<Scalar, Utils> *tip{nullptr};

  TinyMultiBody<Scalar, Utils> *true_object{nullptr};
  TinyMultiBody<Scalar, Utils> *object{nullptr};
  Dataset<double, 2> object_exterior;

  // object exterior transformed by object pose
  Dataset<Scalar, 2> tf_object_exterior;

  TinyMultiBodyConstraintSolverSpring<Scalar, Utils> *contact_ground{nullptr};
  TinyMultiBodyConstraintSolver<Scalar, Utils> tip_contact_model;

  Laboratory(TinyUrdfCache<Scalar, Utils> &urdf_cache,
             const std::string &shape_urdf_filename,
             const std::string &surface_urdf_filename,
             const std::string &tip_urdf_filename,
             const std::string &exterior_filename, VisualizerAPI *sim,
             VisualizerAPI *sim2) {
    bool ignore_cache = true;
    ground = urdf_cache.construct(surface_urdf_filename, world_ground, sim2,
                                  sim, ignore_cache);

    true_object = urdf_cache.construct(shape_urdf_filename, world_tip, sim2,
                                       sim, ignore_cache);
    object = urdf_cache.construct(shape_urdf_filename, world_ground, sim2, sim,
                                  ignore_cache);

    // make groundtruth object green
    for (auto &link : true_object->m_links) {
      for (auto visual_id : link.m_visual_uids1) {
        b3RobotSimulatorChangeVisualShapeArgs vargs;
        vargs.m_objectUniqueId = visual_id;
        vargs.m_hasRgbaColor = true;
        vargs.m_rgbaColor = btVector4(0.1, 0.6, 0, 0.7);
        sim->changeVisualShape(vargs);
      }
    }

    NumpyReader<double, 2> npy_reader;
    bool npy_success = npy_reader.Open(exterior_filename);
    assert(npy_success);
    object_exterior = npy_reader.Read();
    tf_object_exterior.Resize(object_exterior.Shape());

    tip = urdf_cache.construct(tip_urdf_filename, world_tip, sim2, sim);

    {
      // set up contact (friction) model for surface <-> object contact
      delete world_ground.m_mb_constraint_solver;
      contact_ground = new TinyMultiBodyConstraintSolverSpring<Scalar, Utils>;
      world_ground.m_mb_constraint_solver = contact_ground;

      // use some sensible initial settings
      contact_ground->spring_k = Utils::scalar_from_double(5);
      contact_ground->damper_d = Utils::scalar_from_double(0.2);
      contact_ground->mu_static = Utils::scalar_from_double(0.1);
      // contact_ground->friction_model = FRICTION_NONE;
      world_ground.default_friction = Utils::scalar_from_double(0.5);
    }
  }

  ~Laboratory() { delete contact_ground; }
};

class PushEstimator
    : public TinyCeresEstimator<param_dim, state_dim, res_mode> {
 public:
  typedef TinyCeresEstimator<param_dim, state_dim, res_mode> CeresEstimator;
  using CeresEstimator::kStateDim, CeresEstimator::kParameterDim;
  using CeresEstimator::parameters, CeresEstimator::dt;
  using typename CeresEstimator::ADScalar;

  std::vector<double> initial_params;

  // stores trajectories per surface-shape combination
  std::vector<PushData> trajectories;

  mutable std::map<std::string, Laboratory<NDScalar, NDUtils> *> labs_double;
  mutable std::map<std::string, Laboratory<NAScalar, NAUtils> *> labs_ad;

  std::size_t skip_steps = 1;

  NeuralAugmentation neural_augmentation;

  VisualizerAPI *sim{nullptr};

  PushEstimator() : CeresEstimator(0.0) {}

  ~PushEstimator() {
    for (auto &entry : labs_ad) {
      delete entry.second;
    }
    for (auto &entry : labs_double) {
      delete entry.second;
    }
  }

  ceres::Problem &setup(ceres::LossFunction *loss_function = nullptr) override {
    parameters[0] = {"mu_kinetic", 0.5, 0.1, 1.5};
    parameters[1] = {"mu_static", 0.5, 0.1, 1.5};
    parameters[2] = {"v_transition", 0.01, 0.0001, 0.2};

#if USE_NEURAL_AUGMENTATION
    neural_augmentation.assign_estimation_parameters(
        parameters, analytical_param_dim);  //, NN_INIT_ZERO);
    for (int i = 3; i < param_dim; ++i) {
      parameters[i].value *= 0.01;
      // parameters[i].value = 0.0;
      // if (i > 19) {
      //   parameters[i].value += 0.001; // nonzero bias
      // }
    }
#endif
    for (const auto &p : parameters) {
      initial_params.push_back(p.value);
    }

    return CeresEstimator::setup(loss_function);
  }

  void add_training_dataset(const std::string &h5_filename, VisualizerAPI *sim,
                            VisualizerAPI *sim2) {
    PushData data(h5_filename);
    trajectories.push_back(data);
    // target_times.push_back({});
    target_times.push_back(data.time);
    target_trajectories.push_back(data.states);

    add_laboratory(data, sim, sim2);
  }

  void add_laboratory(const PushData &data, VisualizerAPI *sim,
                      VisualizerAPI *sim2) {
    std::string shape_urdf_filename, surface_urdf_filename, tip_urdf_filename,
        exterior_filename;

    TinyFileUtils::find_file("mit-push/obj/" + data.shape_name + ".urdf",
                             shape_urdf_filename);
    TinyFileUtils::find_file("mit-push/obj/" + data.surface_name + ".urdf",
                             surface_urdf_filename);
    TinyFileUtils::find_file("mit-push/obj/tip.urdf", tip_urdf_filename);
    TinyFileUtils::find_file("mit-push/obj/" + data.shape_name + "_ext.npy",
                             exterior_filename);

    add_laboratory<NDScalar, NDUtils>(data.lab_name, shape_urdf_filename,
                                      surface_urdf_filename, tip_urdf_filename,
                                      exterior_filename, urdf_cache_double,
                                      &labs_double, sim, sim2);
    add_laboratory<NAScalar, NAUtils>(data.lab_name, shape_urdf_filename,
                                      surface_urdf_filename, tip_urdf_filename,
                                      exterior_filename, urdf_cache_ad,
                                      &labs_ad, sim, sim2);
  }

  void load_laboratories(VisualizerAPI *sim, VisualizerAPI *sim2) {
    for (const auto &data : trajectories) {
      if (labs_double.find(data.lab_name) == labs_double.end()) {
        add_laboratory(data, sim, sim2);
      }
    }
  }

  void rollout(const std::vector<ADScalar> &params,
               std::vector<std::vector<ADScalar>> &output_states, double &dt,
               std::size_t ref_id) const override {
    printf("rollout AD\n");
#if USE_NEURAL_AUGMENTATION
    typedef NeuralScalar<ADScalar, ADUtils> NScalar;
    typedef NeuralScalarUtils<ADScalar, ADUtils> NUtils;
    auto n_params = NUtils::to_neural(params);
    std::vector<std::vector<NScalar>> n_output_states;
    this->template rollout<NScalar, NUtils>(n_params, n_output_states, dt,
                                            ref_id);
    for (const auto &state : n_output_states) {
      output_states.push_back(NUtils::from_neural(state));
    }
#else
    this->template rollout<ADScalar, ADUtils>(params, output_states, dt,
                                              ref_id);
#endif
  }
  void rollout(const std::vector<double> &params,
               std::vector<std::vector<double>> &output_states, double &dt,
               std::size_t ref_id) const override {
    printf("rollout DOUBLE\n");
#if USE_NEURAL_AUGMENTATION
    typedef NeuralScalar<double, DoubleUtils> NScalar;
    typedef NeuralScalarUtils<double, DoubleUtils> NUtils;
    auto n_params = NUtils::to_neural(params);
    std::vector<std::vector<NScalar>> n_output_states;
    this->template rollout<NScalar, NUtils>(n_params, n_output_states, dt,
                                            ref_id);
    for (const auto &state : n_output_states) {
      output_states.push_back(NUtils::from_neural(state));
    }
#else
    this->template rollout<double, DoubleUtils>(params, output_states, dt,
                                                ref_id);
#endif
  }

 private:
  template <typename Scalar, typename Utils>
  static void add_laboratory(
      const std::string &lab_name, const std::string &shape_urdf_filename,
      const std::string &surface_urdf_filename,
      const std::string &tip_urdf_filename,
      const std::string &exterior_filename,
      TinyUrdfCache<Scalar, Utils> &urdf_cache,
      std::map<std::string, Laboratory<Scalar, Utils> *> *labs,
      VisualizerAPI *sim, VisualizerAPI *sim2) {
    // only add a new "laboratory" for a novel shape-surface combination
    if (labs->find(lab_name) != labs->end()) {
      return;
    }
    (*labs)[lab_name] = new Laboratory<Scalar, Utils>(
        urdf_cache, shape_urdf_filename, surface_urdf_filename,
        tip_urdf_filename, exterior_filename, sim, sim2);
  }

  template <typename Scalar, typename Utils>
  constexpr Laboratory<Scalar, Utils> &get_lab(
      const std::string &lab_name) const {
    if constexpr (std::is_same_v<Scalar, NDScalar>) {
      return *(labs_double[lab_name]);
    } else {
      return *(labs_ad[lab_name]);
    }
  }

  TinyUrdfCache<NDScalar, NDUtils> urdf_cache_double;
  TinyUrdfCache<NAScalar, NAUtils> urdf_cache_ad;

  template <typename Scalar>
  constexpr auto &get_cache() {
    if constexpr (std::is_same_v<Scalar, NDScalar>) {
      return urdf_cache_double;
    } else {
      return urdf_cache_ad;
    }
  }

 public:
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

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  cxxopts::Options options("neural_push", "Learn contact friction model");
  options.add_options()("connection_mode", "Connection mode",
                        cxxopts::value<std::string>()->default_value("gui"));

  auto cli_args = options.parse(argc, argv);

  typedef double Scalar;
  typedef DoubleUtils Utils;

  std::string connection_mode = cli_args["connection_mode"].as<std::string>();

  std::string shape = "butter";

  std::string object_filename;
  TinyFileUtils::find_file("mit-push/obj/" + shape + ".urdf", object_filename);
  std::string tip_filename;
  TinyFileUtils::find_file("mit-push/obj/tip.urdf", tip_filename);
  std::string ground_filename;
  TinyFileUtils::find_file("mit-push/obj/plywood.urdf", ground_filename);

  std::string exterior_filename;
  TinyFileUtils::find_file("mit-push/obj/" + shape + "_ext.npy",
                           exterior_filename);
  NumpyReader<Scalar, 2> npy_reader;
  bool npy_success = npy_reader.Open(exterior_filename);
  assert(npy_success);
  auto exterior = npy_reader.Read();

  std::string push_filename;
  TinyFileUtils::find_file("mit-push/abs/" + shape + "/" + shape +
                               "_h5/"
                               "motion_surface=abs_shape=" +
                               shape + "_a=0_v=10_i=0.000_s=0.000_t=0.000.h5",
                           push_filename);

  PushData data(push_filename);

  if (argc > 1) object_filename = std::string(argv[1]);
  bool floating_base = true;

  // Set NaN trap
  // feenableexcept(FE_INVALID | FE_OVERFLOW);

  printf("floating_base=%d\n", floating_base);
  printf("object_filename=%s\n", object_filename.c_str());
  printf("tip_filename=%s\n", tip_filename.c_str());
  VisualizerAPI *sim2 = new VisualizerAPI();
  bool isConnected2 = sim2->connect(eCONNECT_DIRECT);

  VisualizerAPI *sim = new VisualizerAPI();

  printf("connection_mode=%s\n", connection_mode.c_str());
  int mode = eCONNECT_GUI;
  if (connection_mode == "direct") mode = eCONNECT_DIRECT;
  if (connection_mode == "gui") mode = eCONNECT_GUI;
  if (connection_mode == "shared_memory") mode = eCONNECT_SHARED_MEMORY;

  bool isConnected = sim->connect(mode);
  if (!isConnected) {
    printf("Cannot connect\n");
    return -1;
  }

  sim->resetSimulation();
  sim->setTimeOut(10);

  // create estimator for single-thread optimization (without PBH) and
  // visualization
  PushEstimator frontend_estimator;
  // std::string path = std::filesystem::path(push_filename).parent_path();
  std::string path =
      "/home/eric/tiny-differentiable-simulator/data/mit-push/abs/train";
  for (const auto &entry : std::filesystem::directory_iterator(path)) {
    if (entry.path().string().find("_a=0_") == std::string::npos) {
      // skip files with nonzero acceleration
      continue;
    }
    std::cout << "Loading " << entry.path() << std::endl;
    frontend_estimator.add_training_dataset(entry.path(), sim, sim2);
    if (frontend_estimator.trajectories.size() >= num_files) {
      break;
    }
  }
  // frontend_estimator.add_training_dataset(push_filename, sim, sim2);
  frontend_estimator.use_finite_diff = false;
  frontend_estimator.minibatch_size = num_files;  // 50;

  // frontend_estimator.sim = sim;
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
// frontend_estimator.neural_augmentation.add_wiring(
//       std::vector<std::string>{"tau_0", "tau_1", "tau_3"},
//       std::vector<std::string>{"q_0", "q_1", "q_3"});
#if RESIDUAL_PHYSICS
  frontend_estimator.neural_augmentation.add_wiring(
      std::vector<std::string>{"out_object_x", "out_object_y",
                               "out_object_yaw"},
      std::vector<std::string>{"in_object_x", "in_object_y", "in_object_yaw",
                               "tip_force_x", "tip_force_y", "tip_force_yaw"});
#else
  frontend_estimator.neural_augmentation.add_wiring(
      std::vector<std::string>{"friction/fr_vec.x", "friction/fr_vec.y"},
      std::vector<std::string>{"friction/fn", "friction/point.x",
                               "friction/point.y", "friction/rel_vel.x",
                               "friction/rel_vel.y"});
#endif
  frontend_estimator.setup();

  std::function<std::unique_ptr<PushEstimator>()> construct_estimator =
      [&push_filename, &frontend_estimator]() {
        auto estimator = std::make_unique<PushEstimator>();
        estimator->options.minimizer_progress_to_stdout = !USE_PBH;
        estimator->options.max_num_consecutive_invalid_steps = 20;
        // estimator->options.max_num_iterations = 100;
        // divide each cost term by integer time step ^ 2 to reduce gradient
        // explosion
        estimator->divide_cost_by_time_factor = 0.;
        // estimator->divide_cost_by_time_exponent = 1.2;

        VisualizerAPI *sim_direct = new VisualizerAPI();
        bool isConnected2 = sim_direct->connect(eCONNECT_DIRECT);
        // estimator->add_training_dataset(push_filename, sim_direct,
        // sim_direct);
        estimator->trajectories = frontend_estimator.trajectories;
        estimator->target_trajectories = frontend_estimator.target_trajectories;
        estimator->target_times = frontend_estimator.target_times;
        estimator->load_laboratories(sim_direct, sim_direct);
        estimator->use_finite_diff = frontend_estimator.use_finite_diff;
        estimator->minibatch_size = frontend_estimator.minibatch_size;
        estimator->options = frontend_estimator.options;
        estimator->set_bounds = frontend_estimator.set_bounds;
        estimator->neural_augmentation = frontend_estimator.neural_augmentation;
        return estimator;
      };

  std::vector<double> best_params;

  // best_params = {0.1000022164, 0.9283739043, 0.1999897444};
  // best_params = {0.5, 0.5, 0.01};

#if USE_PBH
  std::array<double, param_dim> initial_guess;
  for (int i = 0; i < param_dim; ++i) {
    initial_guess[i] = frontend_estimator.parameters[i].value;
  }
  BasinHoppingEstimator<param_dim, PushEstimator> bhe(construct_estimator,
                                                      initial_guess);
  bhe.time_limit = 60 * 60 * 4;  // 4 hours
  bhe.run();

  printf("Optimized parameters:");
  for (int i = 0; i < param_dim; ++i) {
    printf(" %.8f", bhe.params[i]);
  }
  printf("\n");

  printf("Best cost: %f\n", bhe.best_cost());

  for (const auto &p : bhe.params) {
    best_params.push_back(p);
  }
#else
  double cost = 0;
  double gradient[param_dim] = {0, 0, 0};

  // double vars[] = {
  //     0.8419394510,  0.2068648047,  0.2000000000,  -0.0282036139,
  //     -0.0474505852, -0.0491347716, 0.0159892392,  0.0223616310,
  //     -0.0474748849, 0.0446727013, 0.0147376202,  -0.0215294212,
  //     -0.0247493320, -0.0365737791, -0.0552035766, -0.0346462365,
  //     -0.0163709386, -0.0166282401, -0.0129492093, -0.0546531948,
  //     0.0140682806,  0.0407330973,  0.0210522115,  0.0207751488,
  //     -0.0143962013, 0.0417694072,  -0.0180367930, 0.0059515549,
  //     -0.0497568537, -0.0580535963, -0.0113478764, -0.0111573448,
  //     -0.0624960850, 0.0160493265,  -0.0618908926, 0.0348041017,
  //     0.0364463975,  -0.0539135484, -0.0003163419, -0.0005941433,
  //     0.0008916415,  0.0013013567,  0.0016905169,  -0.0035324497,
  //     0.0184809508};

  //     best_params = std::vector<double>(vars, vars+param_dim);
  // frontend_estimator.compute_loss(vars, &cost, gradient);

  // frontend_estimator.compute_loss(frontend_estimator.vars(), &cost,
  // gradient);
  // frontend_estimator.compute_loss(vars, &cost, gradient);
  // printf("Cost: %.6f\n", cost);
  // printf("Gradient:  ");
  // for (int i = 0; i < param_dim; ++i) {
  //   printf("%.6f  ", gradient[i]);
  // }
  // printf("\n\n");
  // for (const auto &p : frontend_estimator.parameters) {
  //   printf("\t%s:  %.8f\n", p.name.c_str(), p.value);
  // }
  // return 0;

  // frontend_estimator.gradient_descent(0.001, 20);

  auto summary = frontend_estimator.solve();
  std::cout << summary.FullReport() << std::endl;
  std::cout << "Final cost: " << summary.final_cost << "\n";
  std::cout << "Best cost: " << frontend_estimator.best_cost() << "\n";

  for (const auto &p : frontend_estimator.parameters) {
    best_params.push_back(p.value);
  }

  std::ofstream file("param_evolution.txt");
  for (const auto &params : frontend_estimator.parameter_evolution()) {
    for (int i = 0; i < static_cast<int>(params.size()); ++i) {
      file << params[i];
      if (i < static_cast<int>(params.size()) - 1) file << "\t";
    }
    file << "\n";
  }
  file.close();
#endif

  for (int i = 0; i < param_dim; ++i) {
    const auto &p = frontend_estimator.parameters[i];
    printf("%s: %.10f\n", p.name.c_str(), best_params[i]);
  }

  printf("\n\n");
  fflush(stdout);

  std::vector<NDScalar> nparams(best_params.size());
  for (std::size_t i = 0; i < best_params.size(); ++i) {
    nparams[i] = NDUtils::scalar_from_double(best_params[i]);
  }
  frontend_estimator.sim = sim;
  while (connection_mode == "gui") {
    std::size_t ref_id =
        std::size_t(rand()) % frontend_estimator.trajectories.size();
    // ref_id = 0;
    std::cout << "Simulating trajectory #" << ref_id << "\t"
              << frontend_estimator.trajectories[ref_id].filename << "\n";
    std::vector<std::vector<NDScalar>> output_states;
    frontend_estimator.template rollout<NDScalar, NDUtils>(
        nparams, output_states, data.dt, ref_id);  //, sim);
    std::this_thread::sleep_for(std::chrono::duration<double>(5.));
  }

  return EXIT_SUCCESS;
}
