#include "pch.h"
#include "Optimizers.h"
#include "Tensor.h"
#include "MathUtils.h"

// Applies a Stochastic Gradient Descent optimizer step to the given parameter.
void Optimizers::sgd(const std::shared_ptr<Tensor>& para, float lr)
{
	MathUtils::vector_fnmadd(para->mutable_data(), para->grad(), lr); // p -= grad * lr
}

void Optimizers::adam(const std::shared_ptr<Tensor>& para, float lr, int iter, std::span<float> m, std::span<float> v,
	float beta1, float one_minus_beta1, float beta2, float one_minus_beta2, float epsilon, float weight_decay)
{
	// Calculate bias corrections
	const float bias_corr1 = 1.0f - std::pow(beta1, (float)(iter + 1));
	const float bias_corr2 = 1.0f - std::pow(beta2, (float)(iter + 1));

	// Initialize persistent scratch buffers
	thread_local std::vector<float> scratch1;
	thread_local std::vector<float> scratch2;

	const int param_count = para->element_count();
	scratch1.resize(param_count);
	scratch2.resize(param_count);

	// Update first moment -> m_t = b_1 * m_t-1 + (1 - b_1)g_t
	MathUtils::vector_mul(m, beta1, scratch1);
	MathUtils::vector_fmadd(scratch1, para->grad(), one_minus_beta1, m);

	// Update second moment -> v_t = b_2 * v_t-1 + (1 - b_2)g_t^2
	MathUtils::vector_mul(v, beta2, scratch1);
	MathUtils::vector_sq(para->grad(), scratch2);
	MathUtils::vector_fmadd(scratch1, scratch2, one_minus_beta2, v);

	// Apply bias correction
	MathUtils::vector_div(m, bias_corr1, scratch1);
	MathUtils::vector_div(v, bias_corr2, scratch2);

	// Update parameter -> p_t = p_t-1 - (lr / (sqrt(v_hat_t) + eps)m_hat_t
	MathUtils::vector_sqrt(scratch2);
	MathUtils::vector_add(scratch2, epsilon);
	MathUtils::vector_div(lr, scratch2);
	MathUtils::vector_fnmadd(para->mutable_data(), scratch2, scratch1);

	// Apply AdamW weight decay if needed -> p_t = w_t-1 - lr*wd*w_t-1 - lr(m_hat_t / (sqrt(v_hat_t) + eps))
	if (weight_decay > 0.0f)
	{
		const float coeff = lr * weight_decay;
		MathUtils::vector_mul(para->data(), coeff, scratch1);
		MathUtils::vector_sub(para->mutable_data(), scratch1);
	}
}