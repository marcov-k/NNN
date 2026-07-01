#pragma once

#include <cmath>
#include <memory>
#include <span>
#include <vector>

#include "Tensor.h"

class Optimizers
{
public:
	Optimizers() = delete;

	static void clip_gradients(const std::vector<std::shared_ptr<Tensor>*>& paras, double max_norm);

	static void sgd(const std::shared_ptr<Tensor>& para, double lr);

	static void adam(const std::shared_ptr<Tensor>& para, double lr, int iter, std::span<double> m, std::span<double> v,
		double beta1, double one_minus_beta1, double beta2, double one_minus_beta2, double epsilon, double weight_decay);
};