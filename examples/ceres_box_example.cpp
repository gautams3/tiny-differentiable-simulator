#include <fstream>
#include "math.h"
#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

#define USE_MATPLOTLIB 1

template<typename T> T sign(const T& x) {
  if (x > T(0)) {
    return T(1);
  } else if (x < T(0)) {
    return T(-1);
  } else {
    return T(0);
  }
}

#ifdef USE_MATPLOTLIB
#include "third_party/matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <typename T>
void plot_trajectory(const T states) {
  int state_dim = 2;
  int traj_length = 10;
  for (int i = 0; i < state_dim; ++i) {
    std::vector<double> traj(traj_length);
    for (int t = 0; t < traj_length; ++t) {
      traj[t] = static_cast<double>(states[t*state_dim + i]);
    }
    plt::named_plot("state[" + std::to_string(i) + "]", traj);
  }
  plt::legend();
  plt::show();
}
#endif

template <typename T>
T rollout(const T* inputs, T* states, double dt, bool doprint = false) {
  size_t state_dim = 2; // x, xdot
  int N = ceil(1.0/dt);
  double m = 1.0; // unit mass
  double mu = 0.0;
  double g = 9.81; // m/s^2
  T goal_position = T(1.0);
  T goal_velocity = T(0.0);
  states[0] = T(0.0); // start from 0 position
  states[1] = T(0.0); // start with 0 velocity
  T friction = T(0.0);
  T reg_input = T(0.0); //input regularization

  for (size_t i = 1; i < N; i++) {
    reg_input = reg_input + pow(inputs[i-1], 2.0);
    size_t idx = i * state_dim, prev_idx = (i-1) * state_dim;
    T v_prev = states[prev_idx + 1], x_prev = states[prev_idx + 0];
    if (v_prev == T(0)) { //static friction: opposite to force
      friction = -sign(inputs[i-1]) * std::min(abs(inputs[i-1]), T(mu * m * g));
    } else {  //kinetic friction: opposite to motion
      friction = -sign(v_prev) * T(mu * m * g);
    }
    states[idx + 0] = x_prev + v_prev * T(dt);
    states[idx + 1] = v_prev + (inputs[i-1] + friction)/m * T(dt);

    //Account for friction having a larger effect due to time discretization
    if (abs(friction) > abs(inputs[i-1])) {
      if(doprint && (states[idx + 1] < T(0))) {
        printf("Friction moving body in opposite direction! Clip velocity %.3f below to 0\n", states[idx + 1]);
      }
      states[idx + 1] = std::max(states[idx + 1], T(0));
    }

    //Logging
    if (doprint) {
      printf("%lu: x %.3f, xdot %.3f, accn %.3f, u %.3f\t", i, states[idx+0], states[idx+1], (inputs[i-1] + friction)/m, inputs[i]);
      printf("friction = %.3f\n", friction);
    }
  }

  T error_pos = pow(states[(N-1) * state_dim + 0] - goal_position, 2.0);
  T error_vel = pow(states[(N-1) * state_dim + 1] - goal_velocity, 2.0);
  T error = T(0.0);
  error += error_pos; 
  error += T(0.02) * error_vel;
  error += T(0.0002) * reg_input;

  return error;
}

void print_trajectory(const double states[], const double inputs[], const int N) {
  //N = timesteps
  double dt = 1.0/ static_cast<double>(N); //dt = 1/N
  const int state_dim = 2; //state = x, xdot

  for (size_t i = 0; i < N; i++) {
    size_t idx = i * state_dim;
    printf("%lu: x %.3f, xdot %.3f, u %.3f\n", i, states[idx+0], states[idx+1], inputs[i]);
  }
}

struct CeresFunctional
{
  template <typename T>
  bool operator()(const  T* const inputs, T* residual) const {
    T dummy_states[20];
    residual[0] = rollout<T>(inputs, dummy_states, 0.1);
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]); // TODO: what's the API for glog? using printf for now

  Problem problem;
  const int N = 10;
  double dt = 1/N;
  const int state_dim = 2;
  double states[state_dim*N];
  double inputs[N];

  // INITIALIZE
  double init_x = 0.0;
  double init_xdot = 0.0;
  double init_u = 0.0;

  for (size_t i = 0; i < N; i++) {
    size_t idx = i * state_dim;
    states[idx + 0] = init_x;
    states[idx + 1] = init_xdot;
    inputs[i] = init_u;
  }
  printf("Initialize Trajectory\n");
  print_trajectory(states, inputs, N);

  CostFunction* cost_function = new AutoDiffCostFunction<CeresFunctional, 1, N>(new CeresFunctional);

  problem.AddResidualBlock(cost_function, NULL, inputs);

  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 300;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  rollout<double>(inputs, states, 0.1, false);
  print_trajectory(states, inputs, N);
  plot_trajectory(states);

  return 0;
}
