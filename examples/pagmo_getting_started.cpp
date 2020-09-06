	

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <utility>

#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

using namespace pagmo;

// Our simple example problem, version 0.
struct problem_v0 {
    // Implementation of the objective function.
    vector_double fitness(const vector_double &dv) const
    {
        return {dv[0] * dv[3] * (dv[0] + dv[1] + dv[2]) + dv[2]};
    }
    // Implementation of the box bounds.
    std::pair<vector_double, vector_double> get_bounds() const
    {
        return {{1., 1., 1., 1.}, {5., 5., 5., 5.}};
    }
};

int main()
{
    // Construct a pagmo::problem from our example problem.
    problem p{problem_v0{}};

    // Compute the value of the objective function
    // in the point (1, 2, 3, 4).
    std::cout << "Value of the objfun in (1, 2, 3, 4): " << p.fitness({1, 2, 3, 4})[0] << '\n';

    // Fetch the lower/upper bounds for the first variable.
    std::cout << "Lower bounds: [" << p.get_lb()[0] << "]\n";
    std::cout << "Upper bounds: [" << p.get_ub()[0] << "]\n\n";

    // Print p to screen.
    std::cout << p << '\n';
}


// #include <iostream>

// #include <pagmo/algorithm.hpp>
// #include <pagmo/algorithms/sade.hpp>
// #include <pagmo/archipelago.hpp>
// #include <pagmo/problem.hpp>
// #include <pagmo/problems/schwefel.hpp>

// using namespace pagmo;

// int main()
// {
//     // 1 - Instantiate a pagmo problem constructing it from a UDP
//     // (i.e., a user-defined problem, in this case the 30-dimensional
//     // generalised Schwefel test function).
//     problem prob{schwefel(30)};

//     // 2 - Instantiate a pagmo algorithm (self-adaptive differential
//     // evolution, 100 generations).
//     algorithm algo{sade(100)};

//     // 3 - Instantiate an archipelago with 16 islands having each 20 individuals.
//     archipelago archi{16u, algo, prob, 20u};

//     // 4 - Run the evolution in parallel on the 16 separate islands 10 times.
//     archi.evolve(10);

//     // 5 - Wait for the evolutions to finish.
//     archi.wait_check();

//     // 6 - Print the fitness of the best solution in each island.
//     for (const auto &isl : archi) {
//         std::cout << isl.get_population().champion_f()[0] << '\n';
//     }
// }
