#include "pch.h"
#include "Tensor.h"

// Whether the tensor's gradient must be calculated during the backward pass.
bool Tensor::inference = false;

// Whether the engine is operating in inference mode (disables autograd graph construction).
bool Tensor::log_debug = false;

// Whether the engine should log debug data.
int Tensor::_forward_gen = 0;