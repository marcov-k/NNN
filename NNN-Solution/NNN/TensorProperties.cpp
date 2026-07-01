#include "pch.h"
#include "Tensor.h"

bool Tensor::inference = false;
bool Tensor::log_debug = false;
int Tensor::_forward_gen = 0;