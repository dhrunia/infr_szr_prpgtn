#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("EulerStep2DEpileptorSingle")
    .Input("theta: float")
    .Input("current_state: float")
    .Output("next_state: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

class EulerStep2DEpileptorSingleOp : public OpKernel
{
public:
  explicit EulerStep2DEpileptorSingleOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override
  {
    // Grab the input tensor
    const Tensor &theta_tnsr = context->input(0);
    const Tensor &current_state_tnsr = context->input(1);
    auto theta = theta_tnsr.flat<float>();
    auto current_state = current_state_tnsr.flat<float>();

    // Create an output tensor
    Tensor *next_state_tnsr = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, current_state_tnsr.shape(),
                                                     &next_state_tnsr));
    auto next_state = next_state_tnsr->flat<float>();

    // Define temporary variables and any constants in the 2D Epileptor model
    const float I1 = 4.1;
    const float dt = 0.1;
    const float tau = 50.0;
    float dx;
    float dz;

    // Compute derivatives
    dx = 1.0 - std::pow(current_state(0), 3) - 2 * std::pow(current_state(0), 2) - current_state(1) + I1;
    dz = (1.0 / tau) * (4 * (current_state(0) - theta(0)) - current_state(1));

    // Compute next state using Euler method
    next_state(0) = current_state(0) + dt * dx;
    next_state(1) = current_state(1) + dt * dz;
  }
};

// Register kernel
REGISTER_KERNEL_BUILDER(Name("EulerStep2DEpileptorSingle").Device(DEVICE_CPU), EulerStep2DEpileptorSingleOp);
