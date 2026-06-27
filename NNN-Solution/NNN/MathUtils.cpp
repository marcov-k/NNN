#include "pch.h"
#include "MathUtils.h"

double MathUtils::sum_m256d(__m256d v)
{
	__m128d hi = _mm256_extractf128_pd(v, 1);
	__m128d lo = _mm256_castpd256_pd128(v);
	__m128d sum128 = _mm_add_pd(hi, lo);
	__m128d permuted = _mm_permute_pd(sum128, 1);
	__m128d total = _mm_add_sd(sum128, permuted);
	return _mm_cvtsd_f64(total);
}

double MathUtils::max_m256d(__m256d v)
{
	__m128d hi = _mm256_extractf128_pd(v, 1);
	__m128d lo = _mm256_castpd256_pd128(v);
	__m128d max128 = _mm_max_pd(lo, hi);
	__m128d permuted = _mm_permute_pd(max128, 1);
	__m128d max64 = _mm_max_pd(max128, permuted);
	return _mm_cvtsd_f64(max64);
}

void MathUtils::vector_add(std::span<const double> a, std::span<const double> b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d sum0 = _mm256_add_pd(reg_a0, reg_b0);
		__m256d sum1 = _mm256_add_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_c[i], sum0);
		_mm256_storeu_pd(&p_c[i + 4], sum1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);

		__m256d sum = _mm256_add_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], sum);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] + p_b[i];
	}
}

void MathUtils::vector_add(std::span<double> a, std::span<const double> b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d sum0 = _mm256_add_pd(reg_a0, reg_b0);
		__m256d sum1 = _mm256_add_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_a[i], sum0);
		_mm256_storeu_pd(&p_a[i + 4], sum1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);

		__m256d sum = _mm256_add_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], sum);
	}

	for (; i < n; ++i)
	{
		p_a[i] += p_b[i];
	}
}

void MathUtils::vector_add(std::span<const double> a, double b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d sum0 = _mm256_add_pd(reg_a0, reg_b);
		__m256d sum1 = _mm256_add_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_c[i], sum0);
		_mm256_storeu_pd(&p_c[i + 4], sum1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d sum = _mm256_add_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], sum);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] + b;
	}
}

void MathUtils::vector_add(std::span<double> a, double b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d sum0 = _mm256_add_pd(reg_a0, reg_b);
		__m256d sum1 = _mm256_add_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_a[i], sum0);
		_mm256_storeu_pd(&p_a[i + 4], sum1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d sum = _mm256_add_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], sum);
	}

	for (; i < n; ++i)
	{
		p_a[i] += b;
	}
}

void MathUtils::vector_sub(std::span<const double> a, std::span<const double> b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a0, reg_b0);
		__m256d dif1 = _mm256_sub_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_c[i], dif0);
		_mm256_storeu_pd(&p_c[i + 4], dif1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], dif);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] - p_b[i];
	}
}

void MathUtils::vector_sub(std::span<double> a, std::span<const double> b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a0, reg_b0);
		__m256d dif1 = _mm256_sub_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_a[i], dif0);
		_mm256_storeu_pd(&p_a[i + 4], dif1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], dif);
	}

	for (; i < n; ++i)
	{
		p_a[i] -= p_b[i];
	}
}

void MathUtils::vector_sub(std::span<const double> a, double b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a0, reg_b);
		__m256d dif1 = _mm256_sub_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_c[i], dif0);
		_mm256_storeu_pd(&p_c[i + 4], dif1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], dif);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] - b;
	}
}

void MathUtils::vector_sub(std::span<double> a, double b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a0, reg_b);
		__m256d dif1 = _mm256_sub_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_a[i], dif0);
		_mm256_storeu_pd(&p_a[i + 4], dif1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], dif);
	}

	for (; i < n; ++i)
	{
		p_a[i] -= b;
	}
}

void MathUtils::vector_sub(double a, std::span<const double> b, std::span<double> c)
{
	const int n = (int)b.size();

	const __m256d reg_a = _mm256_set1_pd(a);
	const double* const __restrict p_b = b.data();
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a, reg_b0);
		__m256d dif1 = _mm256_sub_pd(reg_a, reg_b1);

		_mm256_storeu_pd(&p_c[i], dif0);
		_mm256_storeu_pd(&p_c[i + 4], dif1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], dif);
	}

	for (; i < n; ++i)
	{
		p_c[i] = a - p_b[i];
	}
}

void MathUtils::vector_mul(std::span<const double> a, std::span<const double> b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d pro0 = _mm256_mul_pd(reg_a0, reg_b0);
		__m256d pro1 = _mm256_mul_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_c[i], pro0);
		_mm256_storeu_pd(&p_c[i + 4], pro1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d pro = _mm256_mul_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], pro);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] * p_b[i];
	}
}

void MathUtils::vector_mul(std::span<double> a, std::span<const double> b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d pro0 = _mm256_mul_pd(reg_a0, reg_b0);
		__m256d pro1 = _mm256_mul_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_a[i], pro0);
		_mm256_storeu_pd(&p_a[i + 4], pro1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d pro = _mm256_mul_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], pro);
	}

	for (; i < n; ++i)
	{
		p_a[i] *= p_b[i];
	}
}

void MathUtils::vector_mul(std::span<const double> a, double b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d pro0 = _mm256_mul_pd(reg_a0, reg_b);
		__m256d pro1 = _mm256_mul_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_c[i], pro0);
		_mm256_storeu_pd(&p_c[i + 4], pro1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d pro = _mm256_mul_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], pro);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] * b;
	}
}

void MathUtils::vector_mul(std::span<double> a, double b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d pro0 = _mm256_mul_pd(reg_a0, reg_b);
		__m256d pro1 = _mm256_mul_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_a[i], pro0);
		_mm256_storeu_pd(&p_a[i + 4], pro1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d pro = _mm256_mul_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], pro);
	}

	for (; i < n; ++i)
	{
		p_a[i] *= b;
	}
}

void MathUtils::vector_div(std::span<const double> a, std::span<const double> b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a0, reg_b0);
		__m256d quo1 = _mm256_div_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_c[i], quo0);
		_mm256_storeu_pd(&p_c[i + 4], quo1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], quo);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] / p_b[i];
	}
}

void MathUtils::vector_div(std::span<double> a, std::span<const double> b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a0, reg_b0);
		__m256d quo1 = _mm256_div_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_a[i], quo0);
		_mm256_storeu_pd(&p_a[i + 4], quo1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], quo);
	}

	for (; i < n; ++i)
	{
		p_a[i] /= p_b[i];
	}
}

void MathUtils::vector_div(std::span<const double> a, double b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double recip_b = 1.0 / b;
	const __m256d reg_recip_b = _mm256_set1_pd(recip_b);
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d quo0 = _mm256_mul_pd(reg_a0, reg_recip_b);
		__m256d quo1 = _mm256_mul_pd(reg_a1, reg_recip_b);

		_mm256_storeu_pd(&p_c[i], quo0);
		_mm256_storeu_pd(&p_c[i + 4], quo1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d quo = _mm256_mul_pd(reg_a, reg_recip_b);
		_mm256_storeu_pd(&p_c[i], quo);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] * recip_b;
	}
}

void MathUtils::vector_div(std::span<double> a, double b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double recip_b = 1.0 / b;
	const __m256d reg_recip_b = _mm256_set1_pd(recip_b);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d quo0 = _mm256_mul_pd(reg_a0, reg_recip_b);
		__m256d quo1 = _mm256_mul_pd(reg_a1, reg_recip_b);

		_mm256_storeu_pd(&p_a[i], quo0);
		_mm256_storeu_pd(&p_a[i + 4], quo1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d quo = _mm256_mul_pd(reg_a, reg_recip_b);
		_mm256_storeu_pd(&p_a[i], quo);
	}

	for (; i < n; ++i)
	{
		p_a[i] *= recip_b;
	}
}

void MathUtils::vector_div(double a, std::span<const double> b, std::span<double> c)
{
	const int n = (int)b.size();

	const __m256d reg_a = _mm256_set1_pd(a);
	const double* const __restrict p_b = b.data();
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a, reg_b0);
		__m256d quo1 = _mm256_div_pd(reg_a, reg_b1);

		_mm256_storeu_pd(&p_c[i], quo0);
		_mm256_storeu_pd(&p_c[i + 4], quo1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], quo);
	}

	for (; i < n; ++i)
	{
		p_c[i] = a / p_b[i];
	}
}

void MathUtils::vector_pow(std::span<const double> a, std::span<const double> b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a0, reg_b0);
		__m256d exp1 = _mm256_pow_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_c[i], exp0);
		_mm256_storeu_pd(&p_c[i + 4], exp1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], exp);
	}

	for (; i < n; ++i)
	{
		p_c[i] = std::pow(p_a[i], p_b[i]);
	}
}

void MathUtils::vector_pow(std::span<double> a, std::span<const double> b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a0, reg_b0);
		__m256d exp1 = _mm256_pow_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&p_a[i], exp0);
		_mm256_storeu_pd(&p_a[i + 4], exp1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], exp);
	}

	for (; i < n; ++i)
	{
		p_a[i] = std::pow(p_a[i], p_b[i]);
	}
}

void MathUtils::vector_pow(std::span<const double> a, double b, std::span<double> c)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a0, reg_b);
		__m256d exp1 = _mm256_pow_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_c[i], exp0);
		_mm256_storeu_pd(&p_c[i + 4], exp1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], exp);
	}

	for (; i < n; ++i)
	{
		p_c[i] = std::pow(p_a[i], b);
	}
}

void MathUtils::vector_pow(std::span<double> a, double b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a0, reg_b);
		__m256d exp1 = _mm256_pow_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_a[i], exp0);
		_mm256_storeu_pd(&p_a[i + 4], exp1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], exp);
	}

	for (; i < n; ++i)
	{
		p_a[i] = std::pow(p_a[i], b);
	}
}

void MathUtils::vector_pow(double a, std::span<const double> b, std::span<double> c)
{
	const int n = (int)b.size();

	const __m256d reg_a = _mm256_set1_pd(a);
	const double* const __restrict p_b = b.data();
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a, reg_b0);
		__m256d exp1 = _mm256_pow_pd(reg_a, reg_b1);

		_mm256_storeu_pd(&p_c[i], exp0);
		_mm256_storeu_pd(&p_c[i + 4], exp1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], exp);
	}

	for (; i < n; ++i)
	{
		p_c[i] = std::pow(a, p_b[i]);
	}
}

void MathUtils::vector_log(std::span<const double> arg, std::span<const double> log_base, std::span<double> r)
{
	const int n = (int)arg.size();

	const double* const __restrict p_arg = arg.data();
	const double* const __restrict p_base = log_base.data();
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_arg0 = _mm256_loadu_pd(&p_arg[i]);
		__m256d reg_arg1 = _mm256_loadu_pd(&p_arg[i + 4]);

		__m256d reg_base0 = _mm256_loadu_pd(&p_base[i]);
		__m256d reg_base1 = _mm256_loadu_pd(&p_base[i + 4]);

		__m256d ln_arg0 = _mm256_log_pd(reg_arg0);
		__m256d ln_arg1 = _mm256_log_pd(reg_arg1);

		__m256d ln_base0 = _mm256_log_pd(reg_base0);
		__m256d ln_base1 = _mm256_log_pd(reg_base1);

		__m256d log0 = _mm256_div_pd(ln_arg0, ln_base0);
		__m256d log1 = _mm256_div_pd(ln_arg1, ln_base1);

		_mm256_storeu_pd(&p_r[i], log0);
		_mm256_storeu_pd(&p_r[i + 4], log1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_arg = _mm256_loadu_pd(&p_arg[i]);
		__m256d reg_base = _mm256_loadu_pd(&p_base[i]);
		__m256d ln_arg = _mm256_log_pd(reg_arg);
		__m256d ln_base = _mm256_log_pd(reg_base);
		__m256d log = _mm256_div_pd(ln_arg, ln_base);
		_mm256_storeu_pd(&p_r[i], log);
	}

	for (; i < n; ++i)
	{
		p_r[i] = std::log(p_arg[i]) / std::log(p_base[i]);
	}
}

void MathUtils::vector_log(std::span<double> arg, std::span<const double> log_base)
{
	const int n = (int)arg.size();

	double* const __restrict p_arg = arg.data();
	const double* const __restrict p_base = log_base.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_arg0 = _mm256_loadu_pd(&p_arg[i]);
		__m256d reg_arg1 = _mm256_loadu_pd(&p_arg[i + 4]);

		__m256d reg_base0 = _mm256_loadu_pd(&p_base[i]);
		__m256d reg_base1 = _mm256_loadu_pd(&p_base[i + 4]);

		__m256d ln_arg0 = _mm256_log_pd(reg_arg0);
		__m256d ln_arg1 = _mm256_log_pd(reg_arg1);

		__m256d ln_base0 = _mm256_log_pd(reg_base0);
		__m256d ln_base1 = _mm256_log_pd(reg_base1);

		__m256d log0 = _mm256_div_pd(ln_arg0, ln_base0);
		__m256d log1 = _mm256_div_pd(ln_arg1, ln_base1);

		_mm256_storeu_pd(&p_arg[i], log0);
		_mm256_storeu_pd(&p_arg[i + 4], log1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_arg = _mm256_loadu_pd(&p_arg[i]);
		__m256d reg_base = _mm256_loadu_pd(&p_base[i]);
		__m256d ln_arg = _mm256_log_pd(reg_arg);
		__m256d ln_base = _mm256_log_pd(reg_base);
		__m256d log = _mm256_div_pd(ln_arg, ln_base);
		_mm256_storeu_pd(&p_arg[i], log);
	}

	for (; i < n; ++i)
	{
		p_arg[i] = std::log(p_arg[i]) / std::log(p_base[i]);
	}
}

void MathUtils::vector_log(std::span<const double> arg, double log_base, std::span<double> r)
{
	const int n = (int)arg.size();

	const double* const __restrict p_arg = arg.data();
	const double ln_base = 1.0 / std::log(log_base);
	const __m256d reg_ln_base = _mm256_set1_pd(ln_base);
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_arg0 = _mm256_loadu_pd(&p_arg[i]);
		__m256d reg_arg1 = _mm256_loadu_pd(&p_arg[i + 4]);

		__m256d ln_arg0 = _mm256_log_pd(reg_arg0);
		__m256d ln_arg1 = _mm256_log_pd(reg_arg1);

		__m256d log0 = _mm256_mul_pd(ln_arg0, reg_ln_base);
		__m256d log1 = _mm256_mul_pd(ln_arg1, reg_ln_base);

		_mm256_storeu_pd(&p_r[i], log0);
		_mm256_storeu_pd(&p_r[i + 4], log1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_arg = _mm256_loadu_pd(&p_arg[i]);
		__m256d ln_arg = _mm256_log_pd(reg_arg);
		__m256d log = _mm256_mul_pd(ln_arg, reg_ln_base);
		_mm256_storeu_pd(&p_r[i], log);
	}

	for (; i < n; ++i)
	{
		p_r[i] = std::log(p_arg[i]) * ln_base;
	}
}

void MathUtils::vector_log(std::span<double> arg, double log_base)
{
	const int n = (int)arg.size();

	double* const __restrict p_arg = arg.data();
	const double ln_base = 1.0 / std::log(log_base);
	const __m256d reg_ln_base = _mm256_set1_pd(ln_base);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_arg0 = _mm256_loadu_pd(&p_arg[i]);
		__m256d reg_arg1 = _mm256_loadu_pd(&p_arg[i + 4]);

		__m256d ln_arg0 = _mm256_log_pd(reg_arg0);
		__m256d ln_arg1 = _mm256_log_pd(reg_arg1);

		__m256d log0 = _mm256_mul_pd(ln_arg0, reg_ln_base);
		__m256d log1 = _mm256_mul_pd(ln_arg1, reg_ln_base);

		_mm256_storeu_pd(&p_arg[i], log0);
		_mm256_storeu_pd(&p_arg[i + 4], log1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_arg = _mm256_loadu_pd(&p_arg[i]);
		__m256d ln_arg = _mm256_log_pd(reg_arg);
		__m256d log = _mm256_mul_pd(ln_arg, reg_ln_base);
		_mm256_storeu_pd(&p_arg[i], log);
	}

	for (; i < n; ++i)
	{
		p_arg[i] = std::log(p_arg[i]) * ln_base;
	}
}

void MathUtils::vector_log(double arg, std::span<const double> log_base, std::span<double> r)
{
	const int n = (int)log_base.size();

	const double ln_arg = std::log(arg);
	const __m256d reg_ln_arg = _mm256_set1_pd(ln_arg);
	const double* const __restrict p_base = log_base.data();
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_base0 = _mm256_loadu_pd(&p_base[i]);
		__m256d reg_base1 = _mm256_loadu_pd(&p_base[i + 4]);

		__m256d ln_base0 = _mm256_log_pd(reg_base0);
		__m256d ln_base1 = _mm256_log_pd(reg_base1);

		__m256d log0 = _mm256_div_pd(reg_ln_arg, ln_base0);
		__m256d log1 = _mm256_div_pd(reg_ln_arg, ln_base1);

		_mm256_storeu_pd(&p_r[i], log0);
		_mm256_storeu_pd(&p_r[i + 4], log1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_base = _mm256_loadu_pd(&p_base[i]);
		__m256d ln_base = _mm256_log_pd(reg_base);
		__m256d log = _mm256_div_pd(reg_ln_arg, ln_base);
		_mm256_storeu_pd(&p_r[i], log);
	}

	for (; i < n; ++i)
	{
		p_r[i] = ln_arg / std::log(p_base[i]);
	}
}

void MathUtils::vector_fmadd(std::span<const double> a, std::span<const double> b, std::span<const double> c, std::span<double> r)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	const double* const __restrict p_c = c.data();
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d reg_c0 = _mm256_loadu_pd(&p_c[i]);
		__m256d reg_c1 = _mm256_loadu_pd(&p_c[i + 4]);

		__m256d fmadd0 = _mm256_fmadd_pd(reg_b0, reg_c0, reg_a0);
		__m256d fmadd1 = _mm256_fmadd_pd(reg_b1, reg_c1, reg_a1);

		_mm256_storeu_pd(&p_r[i], fmadd0);
		_mm256_storeu_pd(&p_r[i + 4], fmadd0);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_c = _mm256_loadu_pd(&p_c[i]);
		__m256d fmadd = _mm256_fmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&p_r[i], fmadd);
	}

	for (; i < n; ++i)
	{
		p_r[i] = p_a[i] + p_b[i] * p_c[i];
	}
}

void MathUtils::vector_fmadd(std::span<double> a, std::span<const double> b, std::span<const double> c)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	const double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d reg_c0 = _mm256_loadu_pd(&p_c[i]);
		__m256d reg_c1 = _mm256_loadu_pd(&p_c[i + 4]);

		__m256d fmadd0 = _mm256_fmadd_pd(reg_b0, reg_c0, reg_a0);
		__m256d fmadd1 = _mm256_fmadd_pd(reg_b1, reg_c1, reg_a1);

		_mm256_storeu_pd(&p_a[i], fmadd0);
		_mm256_storeu_pd(&p_a[i + 4], fmadd1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_c = _mm256_loadu_pd(&p_c[i]);
		__m256d fmadd = _mm256_fmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&p_a[i], fmadd);
	}

	for (; i < n; ++i)
	{
		p_a[i] += p_b[i] * p_c[i];
	}
}

void MathUtils::vector_fmadd(std::span<const double> a, std::span<const double> b, double c, std::span<double> r)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	const __m256d reg_c = _mm256_set1_pd(c);
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d fmadd0 = _mm256_fmadd_pd(reg_b0, reg_c, reg_a0);
		__m256d fmadd1 = _mm256_fmadd_pd(reg_b1, reg_c, reg_a1);

		_mm256_storeu_pd(&p_r[i], fmadd0);
		_mm256_storeu_pd(&p_r[i + 4], fmadd1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d fmadd = _mm256_fmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&p_r[i], fmadd);
	}

	for (; i < n; ++i)
	{
		p_r[i] = p_a[i] + p_b[i] * c;
	}
}

void MathUtils::vector_fmadd(std::span<double> a, std::span<const double> b, double c)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	const __m256d reg_c = _mm256_set1_pd(c);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d fmadd0 = _mm256_fmadd_pd(reg_b0, reg_c, reg_a0);
		__m256d fmadd1 = _mm256_fmadd_pd(reg_b1, reg_c, reg_a1);

		_mm256_storeu_pd(&p_a[i], fmadd0);
		_mm256_storeu_pd(&p_a[i + 4], fmadd1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d fmadd = _mm256_fmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&p_a[i], fmadd);
	}

	for (; i < n; ++i)
	{
		p_a[i] += p_b[i] * c;
	}
}

void MathUtils::vector_fnmadd(std::span<const double> a, std::span<const double> b, std::span<const double> c, std::span<double> r)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	const double* const __restrict p_c = c.data();
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d reg_c0 = _mm256_loadu_pd(&p_c[i]);
		__m256d reg_c1 = _mm256_loadu_pd(&p_c[i + 4]);

		__m256d fnmadd0 = _mm256_fnmadd_pd(reg_b0, reg_c0, reg_a0);
		__m256d fnmadd1 = _mm256_fnmadd_pd(reg_b1, reg_c1, reg_a1);

		_mm256_storeu_pd(&p_r[i], fnmadd0);
		_mm256_storeu_pd(&p_r[i + 4], fnmadd1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_c = _mm256_loadu_pd(&p_c[i]);
		__m256d fnmadd = _mm256_fnmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&p_r[i], fnmadd);
	}

	for (; i < n; ++i)
	{
		p_r[i] = p_a[i] - p_b[i] * p_c[i];
	}
}

void MathUtils::vector_fnmadd(std::span<double> a, std::span<const double> b, std::span<const double> c)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	const double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d reg_c0 = _mm256_loadu_pd(&p_c[i]);
		__m256d reg_c1 = _mm256_loadu_pd(&p_c[i + 4]);

		__m256d fnmadd0 = _mm256_fnmadd_pd(reg_b0, reg_c0, reg_a0);
		__m256d fnmadd1 = _mm256_fnmadd_pd(reg_b1, reg_c1, reg_a1);

		_mm256_storeu_pd(&p_a[i], fnmadd0);
		_mm256_storeu_pd(&p_a[i + 4], fnmadd1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_c = _mm256_loadu_pd(&p_c[i]);
		__m256d fnmadd = _mm256_fnmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&p_a[i], fnmadd);
	}

	for (; i < n; ++i)
	{
		p_a[i] -= p_b[i] * p_c[i];
	}
}

void MathUtils::vector_fnmadd(std::span<const double> a, std::span<const double> b, double c, std::span<double> r)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	const __m256d reg_c = _mm256_set1_pd(c);
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d fnmadd0 = _mm256_fnmadd_pd(reg_b0, reg_c, reg_a0);
		__m256d fnmadd1 = _mm256_fnmadd_pd(reg_b1, reg_c, reg_a1);

		_mm256_storeu_pd(&p_r[i], fnmadd0);
		_mm256_storeu_pd(&p_r[i + 4], fnmadd1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d fnmadd = _mm256_fnmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&p_r[i], fnmadd);
	}

	for (; i < n; ++i)
	{
		p_r[i] = p_a[i] - p_b[i] * c;
	}
}

void MathUtils::vector_fnmadd(std::span<double> a, std::span<const double> b, double c)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();
	const __m256d reg_c = _mm256_set1_pd(c);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d fnmadd0 = _mm256_fnmadd_pd(reg_b0, reg_c, reg_a0);
		__m256d fnmadd1 = _mm256_fnmadd_pd(reg_b1, reg_c, reg_a1);

		_mm256_storeu_pd(&p_a[i], fnmadd0);
		_mm256_storeu_pd(&p_a[i + 4], fnmadd1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d fnmadd = _mm256_fnmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&p_a[i], fnmadd);
	}

	for (; i < n; ++i)
	{
		p_a[i] -= p_b[i] * c;
	}
}

void MathUtils::vector_sq(std::span<const double> a, std::span<double> r)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d sq0 = _mm256_mul_pd(reg_a0, reg_a0);
		__m256d sq1 = _mm256_mul_pd(reg_a1, reg_a1);

		_mm256_storeu_pd(&p_r[i], sq0);
		_mm256_storeu_pd(&p_r[i + 4], sq1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d sq = _mm256_mul_pd(reg_a, reg_a);
		_mm256_storeu_pd(&p_r[i], sq);
	}

	for (; i < n; ++i)
	{
		p_r[i] = p_a[i] * p_a[i];
	}
}

void MathUtils::vector_sq(std::span<double> a)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d sq0 = _mm256_mul_pd(reg_a0, reg_a0);
		__m256d sq1 = _mm256_mul_pd(reg_a1, reg_a1);

		_mm256_storeu_pd(&p_a[i], sq0);
		_mm256_storeu_pd(&p_a[i + 4], sq1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d sq = _mm256_mul_pd(reg_a, reg_a);
		_mm256_storeu_pd(&p_a[i], sq);
	}

	for (; i < n; ++i)
	{
		p_a[i] *= p_a[i];
	}
}

void MathUtils::vector_sqrt(std::span<const double> a, std::span<double> r)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d sqrt0 = _mm256_sqrt_pd(reg_a0);
		__m256d sqrt1 = _mm256_sqrt_pd(reg_a1);

		_mm256_storeu_pd(&p_r[i], sqrt0);
		_mm256_storeu_pd(&p_r[i + 4], sqrt1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d sqrt = _mm256_sqrt_pd(reg_a);
		_mm256_storeu_pd(&p_r[i], sqrt);
	}

	for (; i < n; ++i)
	{
		p_r[i] = std::sqrt(p_a[i]);
	}
}

void MathUtils::vector_sqrt(std::span<double> a)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d sqrt0 = _mm256_sqrt_pd(reg_a0);
		__m256d sqrt1 = _mm256_sqrt_pd(reg_a1);

		_mm256_storeu_pd(&p_a[i], sqrt0);
		_mm256_storeu_pd(&p_a[i + 4], sqrt1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d sqrt = _mm256_sqrt_pd(reg_a);
		_mm256_storeu_pd(&p_a[i], sqrt);
	}

	for (; i < n; ++i)
	{
		p_a[i] = std::sqrt(p_a[i]);
	}
}

void MathUtils::vector_exp(std::span<const double> a, std::span<double> r)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d exp0 = _mm256_exp_pd(reg_a0);
		__m256d exp1 = _mm256_exp_pd(reg_a1);

		_mm256_storeu_pd(&p_r[i], exp0);
		_mm256_storeu_pd(&p_r[i + 4], exp1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d exp = _mm256_exp_pd(reg_a);
		_mm256_storeu_pd(&p_r[i], exp);
	}

	for (; i < n; ++i)
	{
		p_r[i] = std::exp(p_a[i]);
	}
}

void MathUtils::vector_exp(std::span<double> a)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d exp0 = _mm256_exp_pd(reg_a0);
		__m256d exp1 = _mm256_exp_pd(reg_a1);

		_mm256_storeu_pd(&p_a[i], exp0);
		_mm256_storeu_pd(&p_a[i + 4], exp1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d exp = _mm256_exp_pd(reg_a);
		_mm256_storeu_pd(&p_a[i], exp);
	}

	for (; i < n; ++i)
	{
		p_a[i] = std::exp(p_a[i]);
	}
}

void MathUtils::vector_ln(std::span<const double> a, std::span<double> r)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	double* const __restrict p_r = r.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d ln0 = _mm256_log_pd(reg_a0);
		__m256d ln1 = _mm256_log_pd(reg_a1);

		_mm256_storeu_pd(&p_r[i], ln0);
		_mm256_storeu_pd(&p_r[i + 4], ln1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d ln = _mm256_log_pd(reg_a);
		_mm256_storeu_pd(&p_r[i], ln);
	}

	for (; i < n; ++i)
	{
		p_r[i] = std::log(p_a[i]);
	}
}

void MathUtils::vector_ln(std::span<double> a)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d ln0 = _mm256_log_pd(reg_a0);
		__m256d ln1 = _mm256_log_pd(reg_a1);

		_mm256_storeu_pd(&p_a[i], ln0);
		_mm256_storeu_pd(&p_a[i + 4], ln1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d ln = _mm256_log_pd(reg_a);
		_mm256_storeu_pd(&p_a[i], ln);
	}

	for (; i < n; ++i)
	{
		p_a[i] = std::log(p_a[i]);
	}
}

double MathUtils::vector_sum(std::span<const double> a)
{
	const int n = (int)a.size();
	const double* const __restrict p_a = a.data();

	__m256d acc0 = _mm256_setzero_pd();
	__m256d acc1 = _mm256_setzero_pd();

	int i = 0;

	for (; i <= n - 8; i += 8)
	{
		__m256d reg0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg1 = _mm256_loadu_pd(&p_a[i + 4]);

		acc0 = _mm256_add_pd(acc0, reg0);
		acc1 = _mm256_add_pd(acc1, reg1);
	}

	__m256d total_acc = _mm256_add_pd(acc0, acc1);

	for (; i <= n - 4; i += 4)
	{
		__m256d reg = _mm256_loadu_pd(&p_a[i]);
		total_acc = _mm256_add_pd(total_acc, reg);
	}

	double sum = sum_m256d(total_acc);

	for (; i < n; ++i)
	{
		sum += p_a[i];
	}

	return sum;
}

double MathUtils::vector_max(std::span<const double> a)
{
	const int n = (int)a.size();
	const double* const __restrict p_a = a.data();

	__m256d max0 = _mm256_set1_pd(std::numeric_limits<double>::lowest());
	__m256d max1 = _mm256_set1_pd(std::numeric_limits<double>::lowest());

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg1 = _mm256_loadu_pd(&p_a[i + 4]);

		max0 = _mm256_max_pd(max0, reg0);
		max1 = _mm256_max_pd(max1, reg1);
	}

	__m256d total_max = _mm256_max_pd(max0, max1);

	for (; i <= n - 4; i += 4)
	{
		__m256d reg = _mm256_loadu_pd(&p_a[i]);
		total_max = _mm256_max_pd(total_max, reg);
	}

	double max = max_m256d(total_max);

	for (; i < n; ++i)
	{
		if (p_a[i] > max) max = p_a[i];
	}

	return max;
}

double MathUtils::vector_dot(std::span<const double> a, std::span<const double> b)
{
	const int n = (int)a.size();

	const double* const __restrict p_a = a.data();
	const double* const __restrict p_b = b.data();

	__m256d acc0 = _mm256_setzero_pd();
	__m256d acc1 = _mm256_setzero_pd();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d mul0 = _mm256_mul_pd(reg_a0, reg_b0);
		__m256d mul1 = _mm256_mul_pd(reg_a1, reg_b1);

		acc0 = _mm256_add_pd(acc0, mul0);
		acc1 = _mm256_add_pd(acc1, mul1);
	}

	__m256d total_acc = _mm256_add_pd(acc0, acc1);

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d mul = _mm256_mul_pd(reg_a, reg_b);
		total_acc = _mm256_add_pd(total_acc, mul);
	}

	double dot = sum_m256d(total_acc);

	for (; i < n; ++i)
	{
		dot += p_a[i] * p_b[i];
	}

	return dot;
}

double MathUtils::vector_dot(const double* __restrict a, const double* __restrict b, int a_off, int b_off, int n)
{
	const double* const __restrict p_a = &a[a_off];
	const double* const __restrict p_b = &b[b_off];

	__m256d acc0 = _mm256_setzero_pd();
	__m256d acc1 = _mm256_setzero_pd();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&p_b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&p_b[i + 4]);

		__m256d mul0 = _mm256_mul_pd(reg_a0, reg_b0);
		__m256d mul1 = _mm256_mul_pd(reg_a1, reg_b1);

		acc0 = _mm256_add_pd(acc0, mul0);
		acc1 = _mm256_add_pd(acc1, mul1);
	}

	__m256d total_acc = _mm256_add_pd(acc0, acc1);

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_b = _mm256_loadu_pd(&p_b[i]);
		__m256d mul = _mm256_mul_pd(reg_a, reg_b);
		total_acc = _mm256_add_pd(total_acc, mul);
	}

	double dot = sum_m256d(total_acc);

	for (; i < n; ++i)
	{
		dot += p_a[i] * p_b[i];
	}

	return dot;
}