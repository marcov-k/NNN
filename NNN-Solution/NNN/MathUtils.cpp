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

void MathUtils::vector_add(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c)
{
	int n = static_cast<int>(a.size());

	if (c.size() < a.size())
	{
		c.resize(n);
	}

	const double* __restrict p_a = a.data();
	const double* __restrict p_b = b.data();
	double* __restrict p_c = c.data();

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

	for (; i < n; i++)
	{
		p_c[i] = p_a[i] + p_b[i];
	}
}

void MathUtils::vector_add(std::vector<double>& a, const std::vector<double>& b)
{
	int n = static_cast<int>(a.size());

	double* __restrict p_a = a.data();
	const double* __restrict p_b = b.data();

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

	for (; i < n; i++)
	{
		p_a[i] += p_b[i];
	}
}

void MathUtils::vector_add(const std::vector<double>& a, double b, std::vector<double>& c)
{
	int n = static_cast<int>(a.size());

	if (c.size() < a.size())
	{
		c.resize(n);
	}

	const double* __restrict p_a = a.data();
	__m256d reg_b = _mm256_set1_pd(b);
	double* __restrict p_c = c.data();

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

	for (; i < n; i++)
	{
		p_c[i] = p_a[i] + b;
	}
}

void MathUtils::vector_add(std::vector<double>& a, double b)
{
	int n = static_cast<int>(a.size());

	double* __restrict p_a = a.data();
	__m256d reg_b = _mm256_set1_pd(b);

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

	for (; i < n; i++)
	{
		p_a[i] += b;
	}
}

double MathUtils::vector_sum(const double* __restrict a, int n)
{
	__m256d acc0 = _mm256_setzero_pd();
	__m256d acc1 = _mm256_setzero_pd();

	int i = 0;

	for (; i <= n - 8; i += 8)
	{
		__m256d reg0 = _mm256_loadu_pd(&a[i]);
		__m256d reg1 = _mm256_loadu_pd(&a[i + 4]);

		acc0 = _mm256_add_pd(acc0, reg0);
		acc1 = _mm256_add_pd(acc1, reg1);
	}

	__m256d total_acc = _mm256_add_pd(acc0, acc1);

	for (; i <= n - 4; i += 4)
	{
		__m256d reg = _mm256_loadu_pd(&a[i]);
		total_acc = _mm256_add_pd(total_acc, reg);
	}

	double sum = sum_m256d(total_acc);

	for (; i < n; i++)
	{
		sum += a[i];
	}

	return sum;
}