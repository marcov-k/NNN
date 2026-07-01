#include "pch.h"
#include "Optimizers.h"
#include "Tensor.h"
#include "MathUtils.h"

void Optimizers::clip_gradients(const std::vector<std::shared_ptr<Tensor>*>& paras, double max_norm)
{
	thread_local std::vector<double> scratch1;

	double total_norm = 0.0;
	for (std::shared_ptr<Tensor>* para_ptr : paras)
	{
		auto& para = *para_ptr;
		scratch1.resize(para->element_count());
		MathUtils::vector_sq(para->grad(), scratch1);
		total_norm += MathUtils::vector_sum(scratch1);
	}
	total_norm = std::sqrt(total_norm);

	if (total_norm > max_norm)
	{
		const double scale = max_norm / (total_norm + 1e-8);

		for (std::shared_ptr<Tensor>* para_ptr : paras)
		{
			MathUtils::vector_mul((*para_ptr)->mutable_grad(), scale);
		}
	}
}

void Optimizers::sgd(const std::shared_ptr<Tensor>& para, double lr)
{
	MathUtils::vector_fnmadd(para->mutable_data(), para->grad(), lr);
}

void Optimizers::adam(const std::shared_ptr<Tensor>& para, double lr, int iter, std::span<double> m, std::span<double> v,
	double beta1, double one_minus_beta1, double beta2, double one_minus_beta2, double epsilon, double weight_decay)
{
	const double bias_corr1 = 1.0 - std::pow(beta1, iter + 1);
	const double bias_corr2 = 1.0 - std::pow(beta2, iter + 1);

	thread_local std::vector<double> scratch1;
	thread_local std::vector<double> scratch2;

	const int param_count = para->element_count();
	scratch1.resize(param_count);
	scratch2.resize(param_count);

	MathUtils::vector_mul(m, beta1, scratch1);
	MathUtils::vector_fmadd(scratch1, para->grad(), one_minus_beta1, m);

	MathUtils::vector_mul(v, beta2, scratch1);
	MathUtils::vector_sq(para->grad(), scratch2);
	MathUtils::vector_fmadd(scratch1, scratch2, one_minus_beta2, v);

	MathUtils::vector_div(m, bias_corr1, scratch1);
	MathUtils::vector_div(v, bias_corr2, scratch2);

	MathUtils::vector_sqrt(scratch2);
	MathUtils::vector_add(scratch2, epsilon);
	MathUtils::vector_div(lr, scratch2);
	MathUtils::vector_fnmadd(para->mutable_data(), scratch2, scratch1);

	if (weight_decay > 0.0)
	{
		const double coeff = lr * weight_decay;
		MathUtils::vector_mul(para->data(), coeff, scratch1);
		MathUtils::vector_sub(para->mutable_data(), scratch1);
	}
}