#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("EulerStep2DEpileptor")
    .Input("theta: float")
    .Input("current_state: float")
    .Input("connectome: float")
    .Output("next_state: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
                  c->set_output(0, c->input(1));
                  return Status::OK();
                });

class EulerStep2DEpileptorOp : public OpKernel
{
public:
  explicit EulerStep2DEpileptorOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override
  {
    // Grab the input tensor
    const Tensor &theta_tnsr = context->input(0);
    const Tensor &current_state_tnsr = context->input(1);
    const Tensor &SC_tnsr = context->input(2);
    const auto theta = theta_tnsr.flat<float>();
    const auto current_state = current_state_tnsr.flat<float>();
    const auto SC = SC_tnsr.matrix<float>();

    // Create an output tensor
    Tensor *next_state_tnsr = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, current_state_tnsr.shape(),
                                                     &next_state_tnsr));
    auto next_state = next_state_tnsr->flat<float>();

    // Define temporary variables and any constants in the 2D Epileptor model
    const float I1 = 4.1;
    const float dt = 0.1;
    const int nn = current_state.size() / 2;

    const auto x0 = theta.slice(Eigen::array<int, 1>({0}), Eigen::array<int, 1>({nn}));
    const auto tau = theta(nn);
    const auto K = theta(nn+1);

    // Compute derivatives
    const auto x = current_state.slice(Eigen::array<int, 1>({0}), Eigen::array<int, 1>({nn}));
    const auto z = current_state.slice(Eigen::array<int, 1>({nn}), Eigen::array<int, 1>({nn}));
    const auto x_diff = x.reshape(Eigen::array<int, 2>({1, nn})).broadcast(Eigen::array<int, 2>({nn, 1})) -
                        x.reshape(Eigen::array<int, 2>({nn, 1})).broadcast(Eigen::array<int, 2>({1, nn}));
    const auto dx = 1.0 - x.pow(3) - 2 * x.pow(2) - z + I1;
    const auto dz = (1.0 / tau) * (4 * (x - x0) - z - K * (SC * x_diff).sum(Eigen::array<int, 1>({1})));

    // Compute next state using Euler method
    next_state.slice(Eigen::array<int, 1>({0}), Eigen::array<int, 1>({nn})) = x + dt * dx;
    next_state.slice(Eigen::array<int, 1>({nn}), Eigen::array<int, 1>({nn})) = z + dt * dz;
  }
};

// Register kernel
REGISTER_KERNEL_BUILDER(Name("EulerStep2DEpileptor").Device(DEVICE_CPU), EulerStep2DEpileptorOp);
