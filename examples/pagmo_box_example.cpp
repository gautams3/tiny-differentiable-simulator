/**
 * 1D box pushing experiment using PAGMO
 */

#include <fstream>

// #include "ceres/ceres.h"
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/population.hpp>
// #include "glog/logging.h"
#include "math.h"

using namespace pagmo;

constexpr std::size_t kTimesteps = 20;
constexpr double dt = 1.0/ kTimesteps;
constexpr std::size_t state_dim = 2; // x, xdot
constexpr double m = 1.0; // unit mass
constexpr double mu = 0.5;
constexpr double g = 9.81; // m/s^2

constexpr double init_x = 0.0;
constexpr double init_xdot = 0.0;
constexpr double init_u = 0.0;

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
std::string colours[] = {"b","orange","y","g"};
size_t colours_size = 4;

template <typename T>
void plot_trajectory(const std::vector<std::vector<T>> states) {
  int traj_length = states.size();
  int state_dim = states[0].size();
  for (int i = 0; i < state_dim; ++i) {
    std::vector<double> traj(traj_length);
    std::vector<double> time_s(traj_length);
    for (int t = 0; t < traj_length; ++t) {
      traj[t] = static_cast<double>(states[t][i]);
      time_s[t] = t*dt;
    }
    plt::named_plot("state[" + std::to_string(i) + "]", time_s, traj, colours[i % colours_size]);
  }
  plt::xlabel("Time (s)");
  plt::title("pos(0) and vel(1)");
  plt::legend();
  plt::show();
}
#endif

template <typename T>
T rollout(const std::vector<T> &inputs, std::vector<std::vector<T>> &states, const bool doprint = false) {
  const std::size_t traj_length = states.size();
  const T goal_position = T(1.0);
  const T goal_velocity = T(0.0);
  std::vector<T> init_state = {T(init_x), T(init_xdot)};
  states[0] = init_state;
  T friction = T(0.0);
  T reg_input = T(0.0); //input regularization

  for (size_t i = 1; i < kTimesteps; i++) {
    reg_input = reg_input + pow(inputs[i-1], 2.0);
    size_t idx = i * state_dim, prev_idx = (i-1) * state_dim;
    const T v_prev = states[i-1][1], x_prev = states[i-1][0];
    if (v_prev == T(0)) { //static friction: opposite to force
      friction = -sign(inputs[i-1]) * std::min(abs(inputs[i-1]), T(mu * m * g));
    } else {  //kinetic friction: opposite to motion
      friction = -sign(v_prev) * T(mu * m * g);
    }
    states[i][0] = x_prev + v_prev * T(dt);
    states[i][1] = v_prev + (inputs[i-1] + friction)/m * T(dt);

    //Account for friction having a larger effect due to time discretization
    if (abs(friction) > abs(inputs[i-1])) {
      if(doprint && (states[i][1] < T(0))) {
        printf("Friction moving body in opposite direction! Clip velocity %.3f below to 0\n", states[i][1]);
      }
      states[i][1] = std::max(states[i][1], T(0));
    }

    //Logging
    if (doprint) {
      printf("%lu: x %.3f, xdot %.3f, accn %.3f, u %.3f\t", i, states[i][0], states[i][1], (inputs[i-1] + friction)/m, inputs[i]);
      printf("friction = %.3f\n", friction);
    }
  }

  T error_pos = pow(states[kTimesteps-1][0] - goal_position, 2.0);
  T error_vel = pow(states[kTimesteps-1][1] - goal_velocity, 2.0);
  T error = T(0.0);
  error += error_pos; 
  error += T(0.00006) * error_vel;
  error += T(0.00006) * reg_input;

  return error;
}

void print_trajectory(const std::vector<std::vector<double>> &states, const std::vector<double> &inputs) {
  for (size_t i = 0; i < kTimesteps; i++) {
    printf("%lu: x %.3f, xdot %.3f, u %.3f\n", i, states[i][0], states[i][1], inputs[i]);
  }
}

struct box_problem
{
  //objective function. & constraints (if any)
  vector_double fitness(const vector_double &inputs) const {
    std::vector<double>input_vec(inputs); //vector from input array
    std::vector<double> init_traj(state_dim, 0.0); //initialize dummy_states with 0s
    std::vector<std::vector<double>> dummy_states(kTimesteps, init_traj);
    double residual = rollout<double>(input_vec, dummy_states);
    vector_double output = {residual};
    return output;
  }

  // Implementation of the box bounds.
  std::pair<vector_double, vector_double> get_bounds() const {
    const double max_force = m * g;
    double u_lb = -max_force;
    double u_hb = +max_force;
    vector_double lb(kTimesteps, u_lb);
    vector_double hb(kTimesteps, u_hb);
    return {lb, hb};
  }
};

int main(int argc, char** argv) {
  pagmo::problem prob{box_problem{}};

  std::vector<double> init_traj = {init_x, init_xdot};
  std::vector<std::vector<double>> states(kTimesteps, init_traj); //initialize traj
  std::vector<double> inputs(kTimesteps, init_u);
  printf("Initialize Trajectory\n");
  print_trajectory(states, inputs);

  std::cout << "Value of the objfun in (u = init_u): " << prob.fitness(inputs)[0] << '\n';

  pagmo::algorithm algo{pagmo::sga(100)};

  pagmo::population pop{prob, 24};

  pop = algo.evolve(pop);

  std::cout << "The population: \n" << pop;

  inputs = pop.champion_x();
  // std::cout<<"Champion inputs\t"<<inputs<<std::endl;
  rollout<double>(inputs, states, false);
  print_trajectory(states, inputs);
  plot_trajectory(states);

  return 0;
}
