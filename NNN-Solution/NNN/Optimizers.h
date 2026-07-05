#pragma once

#include <cmath>
#include <memory>
#include <span>

#include "Tensor.h"

// Optimizer step function implementations.
class Optimizers
{
public:
	Optimizers() = delete;

	// Applies a Stochastic Gradient Descent optimizer step to the given parameter.
	static void sgd(const std::shared_ptr<Tensor>& para, double lr);

	// Applies an Adam optimizer step to the given parameter.
	static void adam(const std::shared_ptr<Tensor>& para, double lr, int iter, std::span<double> m, std::span<double> v,
		double beta1, double one_minus_beta1, double beta2, double one_minus_beta2, double epsilon, double weight_decay);
};