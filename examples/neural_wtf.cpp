#include <fstream>

#include "neural_augmentation.h"
#include "neural_scalar.h"
#include "pendulum.h"
#include "pybullet_visualizer_api.h"
#include "tiny_ceres_estimator.h"
#include "tiny_file_utils.h"
#include "tiny_multi_body.h"
#include "tiny_world.h"

// whether to use Parallel Basin Hopping
#define USE_PBH false
// whether the state consists of [q qd] or just q
#define STATE_INCLUDES_QD false
std::vector<double> start_state;
const int param_dim = 62;
const int state_dim = 2;
NeuralAugmentation augmentation;

/**
 * Roll-out pendulum dynamics, and compute states [q, qd].
 */
template <typename Scalar = double, typename Utils = DoubleUtils>
void rollout_pendulum(const std::vector<Scalar> &params,
                      std::vector<std::vector<Scalar>> &output_states,
                      int time_steps, double dt,
                      const std::array<double, 2> &damping = {0., 0.},
                      std::vector<Scalar> *kin_energy = nullptr) {
  output_states.resize(time_steps);

  if constexpr (std::is_same_v<Scalar, double>) {
    for (int t = 0; t < time_steps; ++t) {
      output_states[t].resize(state_dim);
      for (int i = 0; i < state_dim; ++i) {
        output_states[t][i] = (i + t) / 10.0;
      }
    }
    return;
  }

  if constexpr (is_neural_scalar<Scalar, Utils>::value) {
    if (!params.empty()) {
      typedef typename Scalar::InnerScalarType IScalar;
      typedef typename Scalar::InnerUtilsType IUtils;
      std::vector<IScalar> iparams(params.size());
      for (std::size_t i = 0; i < params.size(); ++i) {
        iparams[i] = params[i].evaluate();
      }
      augmentation.template instantiate<IScalar, IUtils>(iparams);
    }
    for (int t = 0; t < time_steps; ++t) {

    Scalar time = Utils::scalar_from_double(t);
    time.assign("time");

      output_states[t].resize(state_dim);
      std::vector<Scalar> state(state_dim);
      for (int i = 0; i < state_dim; ++i) {
        state[i] = Utils::zero();
        state[i].assign(std::string("x_in_") + std::to_string(i));
      }

      for (int i = 0; i < state_dim; ++i) {
        Scalar x_out = Utils::zero();
        x_out.assign(std::string("x_out_") + std::to_string(i));

        x_out.evaluate();

        output_states[t][i] = x_out;
      }
    }
  }
}

template <ResidualMode ResMode>
class PendulumEstimator
    : public TinyCeresEstimator<param_dim, state_dim, ResMode> {
 public:
  typedef TinyCeresEstimator<param_dim, state_dim, ResMode> CeresEstimator;
  using CeresEstimator::kStateDim, CeresEstimator::kParameterDim;
  using CeresEstimator::parameters;
  using typename CeresEstimator::ADScalar;

  int time_steps;

  PendulumEstimator(int time_steps, double dt)
      : CeresEstimator(dt), time_steps(time_steps) {
    augmentation.assign_estimation_parameters(parameters);
  }

  void rollout(const std::vector<ADScalar> &params,
               std::vector<std::vector<ADScalar>> &output_states, double &dt,
               std::size_t ref_id) const override {
    typedef CeresUtils<kParameterDim> ADUtils;
    typedef NeuralScalar<ADScalar, ADUtils> NScalar;
    typedef NeuralScalarUtils<ADScalar, ADUtils> NUtils;
    auto n_params = NUtils::to_neural(params);
    std::vector<std::vector<NScalar>> n_output_states;
    rollout_pendulum<NScalar, NUtils>(n_params, n_output_states, time_steps,
                                      dt);
    for (const auto &state : n_output_states) {
      output_states.push_back(NUtils::from_neural(state));
    }
  }
  void rollout(const std::vector<double> &params,
               std::vector<std::vector<double>> &output_states, double &dt,
               std::size_t ref_id) const override {
    typedef NeuralScalar<double, DoubleUtils> NScalar;
    typedef NeuralScalarUtils<double, DoubleUtils> NUtils;
    auto n_params = NUtils::to_neural(params);
    std::vector<std::vector<NScalar>> n_output_states;
    rollout_pendulum<NScalar, NUtils>(n_params, n_output_states, time_steps,
                                      dt);
    for (const auto &state : n_output_states) {
      output_states.push_back(NUtils::from_neural(state));
    }
  }
};

void print_states(const std::vector<std::vector<double>> &states) {
  for (const auto &s : states) {
    for (double d : s) {
      printf("%.2f ", d);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  typedef PendulumEstimator<RES_MODE_1D> Estimator;

  const double dt = 1. / 500;
  const double time_limit = 5;
  const int time_steps = 10;

  google::InitGoogleLogging(argv[0]);

  // keep the target_times empty, since target time steps match the model
  std::vector<double> target_times;
  std::vector<std::vector<double>> target_states;
  // rollout pendulum with damping
  std::array<double, 2> true_damping{0.2, 0.0};
  rollout_pendulum({}, target_states, time_steps, dt, true_damping);
  start_state = target_states[0];

  std::vector<std::string> outputs, inputs{"time"};
  for (int i = 0; i < state_dim; ++i) {
    outputs.push_back(std::string("x_out_") + std::to_string(i));
    inputs.push_back(std::string("x_in_") + std::to_string(i));
  }
  augmentation.add_wiring(outputs, inputs);

  // for (std::size_t i = 0; i < augmentation.specs.size(); ++i) {
  //   augmentation.specs[i].template save_graphviz<double, DoubleUtils>(
  //       "net_" + std::to_string(i) + ".dot",
  //       augmentation.output_inputs[i].second,
  //       {augmentation.output_inputs[i].first});
  // }

  std::function<std::unique_ptr<Estimator>()> construct_estimator =
      [&target_times, &target_states, &time_steps, &dt]() {
        auto estimator = std::make_unique<Estimator>(time_steps, dt);
        estimator->use_finite_diff = false;
        estimator->target_times = {target_times};
        estimator->target_trajectories = {target_states};
        estimator->options.minimizer_progress_to_stdout = !USE_PBH;
        estimator->options.max_num_consecutive_invalid_steps = 100;
        estimator->set_bounds = false;
        estimator->options.minimizer_type =
        ceres::MinimizerType::LINE_SEARCH;
        // estimator->options.line_search_direction_type =
        //     ceres::LineSearchDirectionType::STEEPEST_DESCENT;
        // divide each cost term by integer time step ^ 2 to reduce gradient
        // explosion
        estimator->divide_cost_by_time_factor = 10.;
        estimator->divide_cost_by_time_exponent = 1.2;
        estimator->options.max_num_iterations = 5000;
        return estimator;
      };

#if USE_PBH
  std::array<double, param_dim> initial_guess;
  for (int i = 0; i < param_dim; ++i) {
    initial_guess[i] = 0.0;
  }
  BasinHoppingEstimator<param_dim, Estimator> bhe(construct_estimator,
                                                  initial_guess);
  bhe.time_limit = 600;
  bhe.run();

  printf("Optimized parameters:");
  for (int i = 0; i < param_dim; ++i) {
    printf(" %.8f", bhe.params[i]);
  }
  printf("\n");

  printf("Best cost: %f\n", bhe.best_cost());

  std::vector<double> best_params;
  for (const auto &p : bhe.params) {
    best_params.push_back(p);
  }
  target_states.clear();
#else
  std::unique_ptr<Estimator> estimator = construct_estimator();
  estimator->setup(nullptr);

  // for (const auto &p : estimator->parameters) {
  //   printf("%s: %.3f\t", p.name.c_str(), p.value);
  // }
  // printf("\n\n");
  // for (int i = 0; i < param_dim; ++i) {
  //   printf("%s: %.3f\t", estimator->parameters[i].name.c_str(),
  //          estimator->vars()[i]);
  // }
  // printf("\n\n");
  // double cost;
  // double gradient[param_dim];
  // estimator->compute_loss(estimator->vars(), &cost, gradient);
  // printf("gradient:   ");
  // for (int i = 0; i < param_dim; ++i) {
  //   printf("%.4f  ", gradient[i]);
  // }
  // printf("\n");

  // XXX verify cost is zero for the true network weights
  // double cost;
  // double gradient[4];
  // double my_params[] = {-true_damping[0], -true_damping[1]};
  // estimator->compute_loss(my_params, &cost, gradient);
  // std::cout << "Gradient: " << gradient[0] << "  " << gradient[1] << "  "
  //           << gradient[2] << "  " << gradient[3] << "  \n";
  // std::cout << "Cost: " << cost << "\n";
  // assert(cost < 1e-4);

  // return 0;

//   estimator->gradient_descent(3e-2, 1000);

  auto summary = estimator->solve();
  std::cout << summary.FullReport() << std::endl;
  std::cout << "Final cost: " << summary.final_cost << "\n";

  std::vector<double> best_params;
  for (const auto &p : estimator->parameters) {
    printf("%s: %.3f\n", p.name.c_str(), p.value);
    best_params.push_back(p.value);
  }

  std::ofstream file("param_evolution.txt");
  for (const auto &params : estimator->parameter_evolution()) {
    for (int i = 0; i < static_cast<int>(params.size()); ++i) {
      file << params[i];
      if (i < static_cast<int>(params.size()) - 1) file << "\t";
    }
    file << "\n";
  }
  file.close();
#endif

  rollout_pendulum<double, DoubleUtils>(best_params, target_states, time_steps,
                                        dt);
  std::ofstream traj_file("estimated_neural_trajectory.csv");
  for (int t = 0; t < time_steps; ++t) {
    traj_file << (t * dt);
    for (double v : target_states[t]) {
      traj_file << "\t" << v;
    }
    traj_file << "\n";
  }
  traj_file.close();

  augmentation.save_graphviz(best_params);

  return EXIT_SUCCESS;
}
