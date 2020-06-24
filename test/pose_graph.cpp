#include <types.h>

#include <pose_graph_3d_error_term.h>
#include <read_g2o.h>
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(input, "", "The pose graph definition filename in g2o format.");


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // CERES_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(FLAGS_input != "") << "Need to specify the filename to read.";

  pose_graph::MapOfPoses poses;
  pose_graph::VectorOfConstraints constraints;
  pose_graph::MapOfPoses_SE3 poses_SE3;
  pose_graph::VectorOfConstraints_SE3 constraints_SE3;

  CHECK(pose_graph::ReadG2oFile(FLAGS_input, &poses, &constraints))
      << "Error reading the file: " << FLAGS_input;

  CHECK(pose_graph::ReadG2oFile(FLAGS_input, &poses_SE3, &constraints_SE3))
      << "Error reading the file: " << FLAGS_input;

  std::cout << "Number of poses: " << poses.size() << '\n';
  std::cout << "Number of constraints: " << constraints.size() << '\n';

  CHECK(pose_graph::OutputPoses("poses_original.txt", poses))
      << "Error outputting to poses_original.txt";

  ceres::Problem problem_1, problem_2;

  // Use Eigen Vector and Quarternion
  pose_graph::BuildOptimizationProblem(constraints, &poses, &problem_1);
  CHECK(pose_graph::SolveOptimizationProblem(&problem_1))
      << "The solve was not successful, exiting.";

  // Use sophus SE3
  pose_graph::BuildOptimizationProblem_SE3(constraints_SE3,
                                           &poses_SE3,
                                           &problem_2);
  CHECK(pose_graph::SolveOptimizationProblem(&problem_2))
      << "The solve was not successful, exiting.";

  CHECK(pose_graph::OutputPoses_SE3("poses_optimized.txt", poses_SE3))
      << "Error outputting to poses_original.txt";

  return 0;
}
