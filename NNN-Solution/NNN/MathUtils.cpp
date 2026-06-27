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

	for (; i < n - 4; i += 4)
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
		_mm256_storeu_pd(&p_a[i], dif1);
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
		_mm256_storeu_pd(&p_a[i], quo1);
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
	const __m256d reg_b = _mm256_set1_pd(b);
	double* const __restrict p_c = c.data();

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a0, reg_b);
		__m256d quo1 = _mm256_div_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_c[i], quo0);
		_mm256_storeu_pd(&p_c[i + 4], quo1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_c[i], quo);
	}

	for (; i < n; ++i)
	{
		p_c[i] = p_a[i] / b;
	}
}

void MathUtils::vector_div(std::span<double> a, double b)
{
	const int n = (int)a.size();

	double* const __restrict p_a = a.data();
	const __m256d reg_b = _mm256_set1_pd(b);

	int i = 0;
	for (; i <= n - 8; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&p_a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&p_a[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a0, reg_b);
		__m256d quo1 = _mm256_div_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&p_a[i], quo0);
		_mm256_storeu_pd(&p_a[i + 4], quo1);
	}

	for (; i <= n - 4; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&p_a[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&p_a[i], quo);
	}

	for (; i < n; ++i)
	{
		p_a[i] /= b;
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
		acc0 = _mm256_add_pd(acc1, mul1);
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