#include "pch.h"
#include "MathUtils.h"

// Collection of various math and vectorization utility functions.

/* Register operations */

// Computes the sum of a 256-bit register of doubles.
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

/* Vector addition */

// Vectorizes the addition of two vectors and writes the result into the provided vector -> c = a + b
void MathUtils::vector_add(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d sum0 = _mm256_add_pd(reg_a0, reg_b0);
		__m256d sum1 = _mm256_add_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&c[i], sum0);
		_mm256_storeu_pd(&c[i + 4], sum1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);

		__m256d sum = _mm256_add_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], sum);
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] + b[i];
	}
}

void MathUtils::vector_add(double* const __restrict a, const double* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d sum0 = _mm256_add_pd(reg_a0, reg_b0);
		__m256d sum1 = _mm256_add_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&a[i], sum0);
		_mm256_storeu_pd(&a[i + 4], sum1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);

		__m256d sum = _mm256_add_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], sum);
	}

	for (; i < n; ++i)
	{
		a[i] += b[i];
	}
}

void MathUtils::vector_add(const double* const __restrict a, double b, double* const __restrict c, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d sum0 = _mm256_add_pd(reg_a0, reg_b);
		__m256d sum1 = _mm256_add_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&c[i], sum0);
		_mm256_storeu_pd(&c[i + 4], sum1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d sum = _mm256_add_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], sum);
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] + b;
	}
}

void MathUtils::vector_add(double* const __restrict a, double b, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d sum0 = _mm256_add_pd(reg_a0, reg_b);
		__m256d sum1 = _mm256_add_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&a[i], sum0);
		_mm256_storeu_pd(&a[i + 4], sum1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d sum = _mm256_add_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], sum);
	}

	for (; i < n; ++i)
	{
		a[i] += b;
	}
}

/* Vector subtraction */

// Vectorizes the subtraction of two vectors and writes the result into the provided vector -> c = a - b
void MathUtils::vector_sub(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a0, reg_b0);
		__m256d dif1 = _mm256_sub_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&c[i], dif0);
		_mm256_storeu_pd(&c[i + 4], dif1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], dif);
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] - b[i];
	}
}

void MathUtils::vector_sub(double* const __restrict a, const double* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a0, reg_b0);
		__m256d dif1 = _mm256_sub_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&a[i], dif0);
		_mm256_storeu_pd(&a[i + 4], dif1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], dif);
	}

	for (; i < n; ++i)
	{
		a[i] -= b[i];
	}
}

void MathUtils::vector_sub(const double* const __restrict a, double b, double* const __restrict c, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a0, reg_b);
		__m256d dif1 = _mm256_sub_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&c[i], dif0);
		_mm256_storeu_pd(&c[i + 4], dif1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], dif);
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] - b;
	}
}

void MathUtils::vector_sub(double* const __restrict a, double b, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a0, reg_b);
		__m256d dif1 = _mm256_sub_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&a[i], dif0);
		_mm256_storeu_pd(&a[i + 4], dif1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], dif);
	}

	for (; i < n; ++i)
	{
		a[i] -= b;
	}
}

void MathUtils::vector_sub(double a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	const __m256d reg_a = _mm256_set1_pd(a);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a, reg_b0);
		__m256d dif1 = _mm256_sub_pd(reg_a, reg_b1);

		_mm256_storeu_pd(&c[i], dif0);
		_mm256_storeu_pd(&c[i + 4], dif1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], dif);
	}

	for (; i < n; ++i)
	{
		c[i] = a - b[i];
	}
}

void MathUtils::vector_sub(double a, double* const __restrict b, size_t n)
{
	const __m256d reg_a = _mm256_set1_pd(a);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d dif0 = _mm256_sub_pd(reg_a, reg_b0);
		__m256d dif1 = _mm256_sub_pd(reg_a, reg_b1);

		_mm256_storeu_pd(&b[i], dif0);
		_mm256_storeu_pd(&b[i + 4], dif1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d dif = _mm256_sub_pd(reg_a, reg_b);
		_mm256_storeu_pd(&b[i], dif);
	}

	for (; i < n; ++i)
	{
		b[i] = a - b[i];
	}
}

/* Vector multiplication */

// Vectorizes the multiplication of two vectors and writes the result into the provided vector -> c = a * b
void MathUtils::vector_mul(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d pro0 = _mm256_mul_pd(reg_a0, reg_b0);
		__m256d pro1 = _mm256_mul_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&c[i], pro0);
		_mm256_storeu_pd(&c[i + 4], pro1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d pro = _mm256_mul_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], pro);
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] * b[i];
	}
}

void MathUtils::vector_mul(double* const __restrict a, const double* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d pro0 = _mm256_mul_pd(reg_a0, reg_b0);
		__m256d pro1 = _mm256_mul_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&a[i], pro0);
		_mm256_storeu_pd(&a[i + 4], pro1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d pro = _mm256_mul_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], pro);
	}

	for (; i < n; ++i)
	{
		a[i] *= b[i];
	}
}

void MathUtils::vector_mul(const double* const __restrict a, double b, double* const __restrict c, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d pro0 = _mm256_mul_pd(reg_a0, reg_b);
		__m256d pro1 = _mm256_mul_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&c[i], pro0);
		_mm256_storeu_pd(&c[i + 4], pro1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d pro = _mm256_mul_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], pro);
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] * b;
	}
}

void MathUtils::vector_mul(double* const __restrict a, double b, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d pro0 = _mm256_mul_pd(reg_a0, reg_b);
		__m256d pro1 = _mm256_mul_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&a[i], pro0);
		_mm256_storeu_pd(&a[i + 4], pro1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d pro = _mm256_mul_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], pro);
	}

	for (; i < n; ++i)
	{
		a[i] *= b;
	}
}

/* Vector division */

// Vectorizes the division of two vectors and writes the result into the provided vector -> c = a / b
void MathUtils::vector_div(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a0, reg_b0);
		__m256d quo1 = _mm256_div_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&c[i], quo0);
		_mm256_storeu_pd(&c[i + 4], quo1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], quo);
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] / b[i];
	}
}

void MathUtils::vector_div(double* const __restrict a, const double* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a0, reg_b0);
		__m256d quo1 = _mm256_div_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&a[i], quo0);
		_mm256_storeu_pd(&a[i + 4], quo1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], quo);
	}

	for (; i < n; ++i)
	{
		a[i] /= b[i];
	}
}

void MathUtils::vector_div(const double* const __restrict a, double b, double* const __restrict c, size_t n)
{
	const double recip_b = 1.0 / b;
	const __m256d reg_recip_b = _mm256_set1_pd(recip_b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d quo0 = _mm256_mul_pd(reg_a0, reg_recip_b);
		__m256d quo1 = _mm256_mul_pd(reg_a1, reg_recip_b);

		_mm256_storeu_pd(&c[i], quo0);
		_mm256_storeu_pd(&c[i + 4], quo1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d quo = _mm256_mul_pd(reg_a, reg_recip_b);
		_mm256_storeu_pd(&c[i], quo);
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] * recip_b;
	}
}

void MathUtils::vector_div(double* const __restrict a, double b, size_t n)
{
	const double recip_b = 1.0 / b;
	const __m256d reg_recip_b = _mm256_set1_pd(recip_b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d quo0 = _mm256_mul_pd(reg_a0, reg_recip_b);
		__m256d quo1 = _mm256_mul_pd(reg_a1, reg_recip_b);

		_mm256_storeu_pd(&a[i], quo0);
		_mm256_storeu_pd(&a[i + 4], quo1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d quo = _mm256_mul_pd(reg_a, reg_recip_b);
		_mm256_storeu_pd(&a[i], quo);
	}

	for (; i < n; ++i)
	{
		a[i] *= recip_b;
	}
}

void MathUtils::vector_div(double a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	const __m256d reg_a = _mm256_set1_pd(a);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a, reg_b0);
		__m256d quo1 = _mm256_div_pd(reg_a, reg_b1);

		_mm256_storeu_pd(&c[i], quo0);
		_mm256_storeu_pd(&c[i + 4], quo1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], quo);
	}

	for (; i < n; ++i)
	{
		c[i] = a / b[i];
	}
}

void MathUtils::vector_div(double a, double* const __restrict b, size_t n)
{
	const __m256d reg_a = _mm256_set1_pd(a);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d quo0 = _mm256_div_pd(reg_a, reg_b0);
		__m256d quo1 = _mm256_div_pd(reg_a, reg_b1);

		_mm256_storeu_pd(&b[i], quo0);
		_mm256_storeu_pd(&b[i + 4], quo1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d quo = _mm256_div_pd(reg_a, reg_b);
		_mm256_storeu_pd(&b[i], quo);
	}

	for (; i < n; ++i)
	{
		b[i] = a / b[i];
	}
}

/* Vector exponentiation */

// Vectorizes the exponentiation of two vectors and writes the result into the provided vector -> c = a ^ b
void MathUtils::vector_pow(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a0, reg_b0);
		__m256d exp1 = _mm256_pow_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&c[i], exp0);
		_mm256_storeu_pd(&c[i + 4], exp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], exp);
	}

	for (; i < n; ++i)
	{
		c[i] = std::pow(a[i], b[i]);
	}
}

void MathUtils::vector_pow(double* const __restrict a, const double* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a0, reg_b0);
		__m256d exp1 = _mm256_pow_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&a[i], exp0);
		_mm256_storeu_pd(&a[i + 4], exp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], exp);
	}

	for (; i < n; ++i)
	{
		a[i] = std::pow(a[i], b[i]);
	}
}

void MathUtils::vector_pow(const double* const __restrict a, double b, double* const __restrict c, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a0, reg_b);
		__m256d exp1 = _mm256_pow_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&c[i], exp0);
		_mm256_storeu_pd(&c[i + 4], exp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], exp);
	}

	for (; i < n; ++i)
	{
		c[i] = std::pow(a[i], b);
	}
}

void MathUtils::vector_pow(double* const __restrict a, double b, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a0, reg_b);
		__m256d exp1 = _mm256_pow_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&a[i], exp0);
		_mm256_storeu_pd(&a[i + 4], exp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], exp);
	}

	for (; i < n; ++i)
	{
		a[i] = std::pow(a[i], b);
	}
}

void MathUtils::vector_pow(double a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	const __m256d reg_a = _mm256_set1_pd(a);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d exp0 = _mm256_pow_pd(reg_a, reg_b0);
		__m256d exp1 = _mm256_pow_pd(reg_a, reg_b1);

		_mm256_storeu_pd(&c[i], exp0);
		_mm256_storeu_pd(&c[i + 4], exp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d exp = _mm256_pow_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], exp);
	}

	for (; i < n; ++i)
	{
		c[i] = std::pow(a, b[i]);
	}
}

/* Vector logarithm */

// Vectorizes the logarithm of a vector argument and base and writes the result into the provided vector -> r = log_base(arg)
void MathUtils::vector_log(const double* const __restrict arg, const double* const __restrict log_base, double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_arg0 = _mm256_loadu_pd(&arg[i]);
		__m256d reg_arg1 = _mm256_loadu_pd(&arg[i + 4]);

		__m256d reg_base0 = _mm256_loadu_pd(&log_base[i]);
		__m256d reg_base1 = _mm256_loadu_pd(&log_base[i + 4]);

		__m256d ln_arg0 = _mm256_log_pd(reg_arg0);
		__m256d ln_arg1 = _mm256_log_pd(reg_arg1);

		__m256d ln_base0 = _mm256_log_pd(reg_base0);
		__m256d ln_base1 = _mm256_log_pd(reg_base1);

		__m256d log0 = _mm256_div_pd(ln_arg0, ln_base0);
		__m256d log1 = _mm256_div_pd(ln_arg1, ln_base1);

		_mm256_storeu_pd(&r[i], log0);
		_mm256_storeu_pd(&r[i + 4], log1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_arg = _mm256_loadu_pd(&arg[i]);
		__m256d reg_base = _mm256_loadu_pd(&log_base[i]);
		__m256d ln_arg = _mm256_log_pd(reg_arg);
		__m256d ln_base = _mm256_log_pd(reg_base);
		__m256d log = _mm256_div_pd(ln_arg, ln_base);
		_mm256_storeu_pd(&r[i], log);
	}

	for (; i < n; ++i)
	{
		r[i] = std::log(arg[i]) / std::log(log_base[i]);
	}
}

void MathUtils::vector_log(double* const __restrict arg, const double* const __restrict log_base, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_arg0 = _mm256_loadu_pd(&arg[i]);
		__m256d reg_arg1 = _mm256_loadu_pd(&arg[i + 4]);

		__m256d reg_base0 = _mm256_loadu_pd(&log_base[i]);
		__m256d reg_base1 = _mm256_loadu_pd(&log_base[i + 4]);

		__m256d ln_arg0 = _mm256_log_pd(reg_arg0);
		__m256d ln_arg1 = _mm256_log_pd(reg_arg1);

		__m256d ln_base0 = _mm256_log_pd(reg_base0);
		__m256d ln_base1 = _mm256_log_pd(reg_base1);

		__m256d log0 = _mm256_div_pd(ln_arg0, ln_base0);
		__m256d log1 = _mm256_div_pd(ln_arg1, ln_base1);

		_mm256_storeu_pd(&arg[i], log0);
		_mm256_storeu_pd(&arg[i + 4], log1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_arg = _mm256_loadu_pd(&arg[i]);
		__m256d reg_base = _mm256_loadu_pd(&log_base[i]);
		__m256d ln_arg = _mm256_log_pd(reg_arg);
		__m256d ln_base = _mm256_log_pd(reg_base);
		__m256d log = _mm256_div_pd(ln_arg, ln_base);
		_mm256_storeu_pd(&arg[i], log);
	}

	for (; i < n; ++i)
	{
		arg[i] = std::log(arg[i]) / std::log(log_base[i]);
	}
}

void MathUtils::vector_log(const double* const __restrict arg, double log_base, double* const __restrict r, size_t n)
{
	const double ln_base = 1.0 / std::log(log_base);
	const __m256d reg_ln_base = _mm256_set1_pd(ln_base);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_arg0 = _mm256_loadu_pd(&arg[i]);
		__m256d reg_arg1 = _mm256_loadu_pd(&arg[i + 4]);

		__m256d ln_arg0 = _mm256_log_pd(reg_arg0);
		__m256d ln_arg1 = _mm256_log_pd(reg_arg1);

		__m256d log0 = _mm256_mul_pd(ln_arg0, reg_ln_base);
		__m256d log1 = _mm256_mul_pd(ln_arg1, reg_ln_base);

		_mm256_storeu_pd(&r[i], log0);
		_mm256_storeu_pd(&r[i + 4], log1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_arg = _mm256_loadu_pd(&arg[i]);
		__m256d ln_arg = _mm256_log_pd(reg_arg);
		__m256d log = _mm256_mul_pd(ln_arg, reg_ln_base);
		_mm256_storeu_pd(&r[i], log);
	}

	for (; i < n; ++i)
	{
		r[i] = std::log(arg[i]) * ln_base;
	}
}

void MathUtils::vector_log(double* const __restrict arg, double log_base, size_t n)
{
	const double ln_base = 1.0 / std::log(log_base);
	const __m256d reg_ln_base = _mm256_set1_pd(ln_base);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_arg0 = _mm256_loadu_pd(&arg[i]);
		__m256d reg_arg1 = _mm256_loadu_pd(&arg[i + 4]);

		__m256d ln_arg0 = _mm256_log_pd(reg_arg0);
		__m256d ln_arg1 = _mm256_log_pd(reg_arg1);

		__m256d log0 = _mm256_mul_pd(ln_arg0, reg_ln_base);
		__m256d log1 = _mm256_mul_pd(ln_arg1, reg_ln_base);

		_mm256_storeu_pd(&arg[i], log0);
		_mm256_storeu_pd(&arg[i + 4], log1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_arg = _mm256_loadu_pd(&arg[i]);
		__m256d ln_arg = _mm256_log_pd(reg_arg);
		__m256d log = _mm256_mul_pd(ln_arg, reg_ln_base);
		_mm256_storeu_pd(&arg[i], log);
	}

	for (; i < n; ++i)
	{
		arg[i] = std::log(arg[i]) * ln_base;
	}
}

void MathUtils::vector_log(double arg, const double* const __restrict log_base, double* const __restrict r, size_t n)
{
	const double ln_arg = std::log(arg);
	const __m256d reg_ln_arg = _mm256_set1_pd(ln_arg);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_base0 = _mm256_loadu_pd(&log_base[i]);
		__m256d reg_base1 = _mm256_loadu_pd(&log_base[i + 4]);

		__m256d ln_base0 = _mm256_log_pd(reg_base0);
		__m256d ln_base1 = _mm256_log_pd(reg_base1);

		__m256d log0 = _mm256_div_pd(reg_ln_arg, ln_base0);
		__m256d log1 = _mm256_div_pd(reg_ln_arg, ln_base1);

		_mm256_storeu_pd(&r[i], log0);
		_mm256_storeu_pd(&r[i + 4], log1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_base = _mm256_loadu_pd(&log_base[i]);
		__m256d ln_base = _mm256_log_pd(reg_base);
		__m256d log = _mm256_div_pd(reg_ln_arg, ln_base);
		_mm256_storeu_pd(&r[i], log);
	}

	for (; i < n; ++i)
	{
		r[i] = ln_arg / std::log(log_base[i]);
	}
}

/* Vector fused multiply addition */

// Vectorizes the fused multiply addition of three vectors and writes the result into the provided vector -> r = a + b * c
void MathUtils::vector_fmadd(const double* const __restrict a, const double* const __restrict b, const double* const __restrict c,
	double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d reg_c0 = _mm256_loadu_pd(&c[i]);
		__m256d reg_c1 = _mm256_loadu_pd(&c[i + 4]);

		__m256d fmadd0 = _mm256_fmadd_pd(reg_b0, reg_c0, reg_a0);
		__m256d fmadd1 = _mm256_fmadd_pd(reg_b1, reg_c1, reg_a1);

		_mm256_storeu_pd(&r[i], fmadd0);
		_mm256_storeu_pd(&r[i + 4], fmadd1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d reg_c = _mm256_loadu_pd(&c[i]);
		__m256d fmadd = _mm256_fmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&r[i], fmadd);
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] + b[i] * c[i];
	}
}

void MathUtils::vector_fmadd(double* const __restrict a, const double* const __restrict b, const double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d reg_c0 = _mm256_loadu_pd(&c[i]);
		__m256d reg_c1 = _mm256_loadu_pd(&c[i + 4]);

		__m256d fmadd0 = _mm256_fmadd_pd(reg_b0, reg_c0, reg_a0);
		__m256d fmadd1 = _mm256_fmadd_pd(reg_b1, reg_c1, reg_a1);

		_mm256_storeu_pd(&a[i], fmadd0);
		_mm256_storeu_pd(&a[i + 4], fmadd1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d reg_c = _mm256_loadu_pd(&c[i]);
		__m256d fmadd = _mm256_fmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&a[i], fmadd);
	}

	for (; i < n; ++i)
	{
		a[i] += b[i] * c[i];
	}
}

void MathUtils::vector_fmadd(const double* const __restrict a, const double* const __restrict b, double c,
	double* const __restrict r, size_t n)
{
	const __m256d reg_c = _mm256_set1_pd(c);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d fmadd0 = _mm256_fmadd_pd(reg_b0, reg_c, reg_a0);
		__m256d fmadd1 = _mm256_fmadd_pd(reg_b1, reg_c, reg_a1);

		_mm256_storeu_pd(&r[i], fmadd0);
		_mm256_storeu_pd(&r[i + 4], fmadd1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d fmadd = _mm256_fmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&r[i], fmadd);
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] + b[i] * c;
	}
}

void MathUtils::vector_fmadd(double* const __restrict a, const double* const __restrict b, double c, size_t n)
{
	const __m256d reg_c = _mm256_set1_pd(c);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d fmadd0 = _mm256_fmadd_pd(reg_b0, reg_c, reg_a0);
		__m256d fmadd1 = _mm256_fmadd_pd(reg_b1, reg_c, reg_a1);

		_mm256_storeu_pd(&a[i], fmadd0);
		_mm256_storeu_pd(&a[i + 4], fmadd1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d fmadd = _mm256_fmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&a[i], fmadd);
	}

	for (; i < n; ++i)
	{
		a[i] += b[i] * c;
	}
}

/* Vector fused negative multiply addition */

// Vectorizes the fused negative multiply addition of three vectors and writes the result into the provided vector -> r = a - b * c
void MathUtils::vector_fnmadd(const double* const __restrict a, const double* const __restrict b, const double* const __restrict c,
	double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d reg_c0 = _mm256_loadu_pd(&c[i]);
		__m256d reg_c1 = _mm256_loadu_pd(&c[i + 4]);

		__m256d fnmadd0 = _mm256_fnmadd_pd(reg_b0, reg_c0, reg_a0);
		__m256d fnmadd1 = _mm256_fnmadd_pd(reg_b1, reg_c1, reg_a1);

		_mm256_storeu_pd(&r[i], fnmadd0);
		_mm256_storeu_pd(&r[i + 4], fnmadd1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d reg_c = _mm256_loadu_pd(&c[i]);
		__m256d fnmadd = _mm256_fnmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&r[i], fnmadd);
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] - b[i] * c[i];
	}
}

void MathUtils::vector_fnmadd(double* const __restrict a, const double* const __restrict b, const double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d reg_c0 = _mm256_loadu_pd(&c[i]);
		__m256d reg_c1 = _mm256_loadu_pd(&c[i + 4]);

		__m256d fnmadd0 = _mm256_fnmadd_pd(reg_b0, reg_c0, reg_a0);
		__m256d fnmadd1 = _mm256_fnmadd_pd(reg_b1, reg_c1, reg_a1);

		_mm256_storeu_pd(&a[i], fnmadd0);
		_mm256_storeu_pd(&a[i + 4], fnmadd1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d reg_c = _mm256_loadu_pd(&c[i]);
		__m256d fnmadd = _mm256_fnmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&a[i], fnmadd);
	}

	for (; i < n; ++i)
	{
		a[i] -= b[i] * c[i];
	}
}

void MathUtils::vector_fnmadd(const double* const __restrict a, const double* const __restrict b, double c,
	double* const __restrict r, size_t n)
{
	const __m256d reg_c = _mm256_set1_pd(c);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d fnmadd0 = _mm256_fnmadd_pd(reg_b0, reg_c, reg_a0);
		__m256d fnmadd1 = _mm256_fnmadd_pd(reg_b1, reg_c, reg_a1);

		_mm256_storeu_pd(&r[i], fnmadd0);
		_mm256_storeu_pd(&r[i + 4], fnmadd1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d fnmadd = _mm256_fnmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&r[i], fnmadd);
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] - b[i] * c;
	}
}

void MathUtils::vector_fnmadd(double* const __restrict a, const double* const __restrict b, double c, size_t n)
{
	const __m256d reg_c = _mm256_set1_pd(c);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d fnmadd0 = _mm256_fnmadd_pd(reg_b0, reg_c, reg_a0);
		__m256d fnmadd1 = _mm256_fnmadd_pd(reg_b1, reg_c, reg_a1);

		_mm256_storeu_pd(&a[i], fnmadd0);
		_mm256_storeu_pd(&a[i + 4], fnmadd1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d fnmadd = _mm256_fnmadd_pd(reg_b, reg_c, reg_a);
		_mm256_storeu_pd(&a[i], fnmadd);
	}

	for (; i < n; ++i)
	{
		a[i] -= b[i] * c;
	}
}

/* Vector square */

// Vectorizes the square of a vector and writes the result into the provided vector -> r = a ^ 2
void MathUtils::vector_sq(const double* const __restrict a, double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d sq0 = _mm256_mul_pd(reg_a0, reg_a0);
		__m256d sq1 = _mm256_mul_pd(reg_a1, reg_a1);

		_mm256_storeu_pd(&r[i], sq0);
		_mm256_storeu_pd(&r[i + 4], sq1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d sq = _mm256_mul_pd(reg_a, reg_a);
		_mm256_storeu_pd(&r[i], sq);
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] * a[i];
	}
}

void MathUtils::vector_sq(double* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d sq0 = _mm256_mul_pd(reg_a0, reg_a0);
		__m256d sq1 = _mm256_mul_pd(reg_a1, reg_a1);

		_mm256_storeu_pd(&a[i], sq0);
		_mm256_storeu_pd(&a[i + 4], sq1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d sq = _mm256_mul_pd(reg_a, reg_a);
		_mm256_storeu_pd(&a[i], sq);
	}

	for (; i < n; ++i)
	{
		a[i] *= a[i];
	}
}

/* Vector square root */

// Vectorizes the square root of a vector and writes the result into the provided vector -> r = sqrt(a)
void MathUtils::vector_sqrt(const double* const __restrict a, double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d sqrt0 = _mm256_sqrt_pd(reg_a0);
		__m256d sqrt1 = _mm256_sqrt_pd(reg_a1);

		_mm256_storeu_pd(&r[i], sqrt0);
		_mm256_storeu_pd(&r[i + 4], sqrt1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d sqrt = _mm256_sqrt_pd(reg_a);
		_mm256_storeu_pd(&r[i], sqrt);
	}

	for (; i < n; ++i)
	{
		r[i] = std::sqrt(a[i]);
	}
}

void MathUtils::vector_sqrt(double* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d sqrt0 = _mm256_sqrt_pd(reg_a0);
		__m256d sqrt1 = _mm256_sqrt_pd(reg_a1);

		_mm256_storeu_pd(&a[i], sqrt0);
		_mm256_storeu_pd(&a[i + 4], sqrt1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d sqrt = _mm256_sqrt_pd(reg_a);
		_mm256_storeu_pd(&a[i], sqrt);
	}

	for (; i < n; ++i)
	{
		a[i] = std::sqrt(a[i]);
	}
}

/* Vector natural exponentiation */

// Vectorizes the natural exponentiation of a vector and writes the result into the provided vector -> r = e ^ a
void MathUtils::vector_exp(const double* const __restrict a, double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d exp0 = _mm256_exp_pd(reg_a0);
		__m256d exp1 = _mm256_exp_pd(reg_a1);

		_mm256_storeu_pd(&r[i], exp0);
		_mm256_storeu_pd(&r[i + 4], exp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d exp = _mm256_exp_pd(reg_a);
		_mm256_storeu_pd(&r[i], exp);
	}

	for (; i < n; ++i)
	{
		r[i] = std::exp(a[i]);
	}
}

void MathUtils::vector_exp(double* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d exp0 = _mm256_exp_pd(reg_a0);
		__m256d exp1 = _mm256_exp_pd(reg_a1);

		_mm256_storeu_pd(&a[i], exp0);
		_mm256_storeu_pd(&a[i + 4], exp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d exp = _mm256_exp_pd(reg_a);
		_mm256_storeu_pd(&a[i], exp);
	}

	for (; i < n; ++i)
	{
		a[i] = std::exp(a[i]);
	}
}

/* Vector natural logarithm */

// Vectorizes the natural logarithm of a vector and writes the result into the provided vector -> r = ln(a)
void MathUtils::vector_ln(const double* const __restrict a, double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d ln0 = _mm256_log_pd(reg_a0);
		__m256d ln1 = _mm256_log_pd(reg_a1);

		_mm256_storeu_pd(&r[i], ln0);
		_mm256_storeu_pd(&r[i + 4], ln1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d ln = _mm256_log_pd(reg_a);
		_mm256_storeu_pd(&r[i], ln);
	}

	for (; i < n; ++i)
	{
		r[i] = std::log(a[i]);
	}
}

void MathUtils::vector_ln(double* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d ln0 = _mm256_log_pd(reg_a0);
		__m256d ln1 = _mm256_log_pd(reg_a1);

		_mm256_storeu_pd(&a[i], ln0);
		_mm256_storeu_pd(&a[i + 4], ln1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d ln = _mm256_log_pd(reg_a);
		_mm256_storeu_pd(&a[i], ln);
	}

	for (; i < n; ++i)
	{
		a[i] = std::log(a[i]);
	}
}

/* Vector operations */

// Vectorizes the sum of a vector.
double MathUtils::vector_sum(const double* const __restrict a, size_t n)
{
	__m256d acc0 = _mm256_setzero_pd();
	__m256d acc1 = _mm256_setzero_pd();

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg0 = _mm256_loadu_pd(&a[i]);
		__m256d reg1 = _mm256_loadu_pd(&a[i + 4]);

		acc0 = _mm256_add_pd(acc0, reg0);
		acc1 = _mm256_add_pd(acc1, reg1);
	}

	__m256d total_acc = _mm256_add_pd(acc0, acc1);

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg = _mm256_loadu_pd(&a[i]);
		total_acc = _mm256_add_pd(total_acc, reg);
	}

	double sum = sum_m256d(total_acc);

	for (; i < n; ++i)
	{
		sum += a[i];
	}

	return sum;
}

double MathUtils::vector_max(const double* const __restrict a, size_t n)
{
	__m256d max0 = _mm256_set1_pd(std::numeric_limits<double>::lowest());
	__m256d max1 = _mm256_set1_pd(std::numeric_limits<double>::lowest());

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg0 = _mm256_loadu_pd(&a[i]);
		__m256d reg1 = _mm256_loadu_pd(&a[i + 4]);

		max0 = _mm256_max_pd(max0, reg0);
		max1 = _mm256_max_pd(max1, reg1);
	}

	__m256d total_max = _mm256_max_pd(max0, max1);

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg = _mm256_loadu_pd(&a[i]);
		total_max = _mm256_max_pd(total_max, reg);
	}

	double max = max_m256d(total_max);

	for (; i < n; ++i)
	{
		if (a[i] > max) max = a[i];
	}

	return max;
}

double MathUtils::vector_dot(const double* const __restrict a, const double* const __restrict b, size_t n)
{
	__m256d acc0 = _mm256_setzero_pd();
	__m256d acc1 = _mm256_setzero_pd();

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d mul0 = _mm256_mul_pd(reg_a0, reg_b0);
		__m256d mul1 = _mm256_mul_pd(reg_a1, reg_b1);

		acc0 = _mm256_add_pd(acc0, mul0);
		acc1 = _mm256_add_pd(acc1, mul1);
	}

	__m256d total_acc = _mm256_add_pd(acc0, acc1);

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d mul = _mm256_mul_pd(reg_a, reg_b);
		total_acc = _mm256_add_pd(total_acc, mul);
	}

	double dot = sum_m256d(total_acc);

	for (; i < n; ++i)
	{
		dot += a[i] * b[i];
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
	for (; i + 8 <= n; i += 8)
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

	for (; i + 4 <= n; i += 4)
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

/* Vector limiting functions */

// Vectorizes the max of two vectors and writes the result into the provided vector -> c = max(a, b)
void MathUtils::vector_max(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);
		
		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d max0 = _mm256_max_pd(reg_a0, reg_b0);
		__m256d max1 = _mm256_max_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&c[i], max0);
		_mm256_storeu_pd(&c[i + 4], max1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d max = _mm256_max_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], max);
	}

	for (; i < n; ++i)
	{
		c[i] = max(a[i], b[i]);
	}
}

void MathUtils::vector_max(double* const __restrict a, const double* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d max0 = _mm256_max_pd(reg_a0, reg_b0);
		__m256d max1 = _mm256_max_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&a[i], max0);
		_mm256_storeu_pd(&a[i + 4], max1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d max = _mm256_max_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], max);
	}

	for (; i < n; ++i)
	{
		if (b[i] > a[i]) a[i] = b[i];
	}
}

void MathUtils::vector_max(const double* const __restrict a, double b, double* const __restrict c, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d max0 = _mm256_max_pd(reg_a0, reg_b);
		__m256d max1 = _mm256_max_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&c[i], max0);
		_mm256_storeu_pd(&c[i + 4], max1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d max = _mm256_max_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], max);
	}

	for (; i < n; ++i)
	{
		c[i] = max(a[i], b);
	}
}

void MathUtils::vector_max(double* const __restrict a, double b, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d max0 = _mm256_max_pd(reg_a0, reg_b);
		__m256d max1 = _mm256_max_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&a[i], max0);
		_mm256_storeu_pd(&a[i + 4], max1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d max = _mm256_max_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], max);
	}

	for (; i < n; ++i)
	{
		if (b > a[i]) a[i] = b;
	}
}

void MathUtils::vector_min(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d min0 = _mm256_min_pd(reg_a0, reg_b0);
		__m256d min1 = _mm256_min_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&c[i], min0);
		_mm256_storeu_pd(&c[i + 4], min1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d min = _mm256_min_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], min);
	}

	for (; i < n; ++i)
	{
		c[i] = min(a[i], b[i]);
	}
}

void MathUtils::vector_min(double* const __restrict a, const double* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_b0 = _mm256_loadu_pd(&b[i]);
		__m256d reg_b1 = _mm256_loadu_pd(&b[i + 4]);

		__m256d min0 = _mm256_min_pd(reg_a0, reg_b0);
		__m256d min1 = _mm256_min_pd(reg_a1, reg_b1);

		_mm256_storeu_pd(&a[i], min0);
		_mm256_storeu_pd(&a[i + 4], min1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_b = _mm256_loadu_pd(&b[i]);
		__m256d min = _mm256_min_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], min);
	}

	for (; i < n; ++i)
	{
		if (b[i] < a[i]) a[i] = b[i];
	}
}

void MathUtils::vector_min(const double* const __restrict a, double b, double* const __restrict c, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d min0 = _mm256_min_pd(reg_a0, reg_b);
		__m256d min1 = _mm256_min_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&c[i], min0);
		_mm256_storeu_pd(&c[i + 4], min1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d min = _mm256_min_pd(reg_a, reg_b);
		_mm256_storeu_pd(&c[i], min);
	}

	for (; i < n; ++i)
	{
		c[i] = min(a[i], b);
	}
}

void MathUtils::vector_min(double* const __restrict a, double b, size_t n)
{
	const __m256d reg_b = _mm256_set1_pd(b);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d min0 = _mm256_min_pd(reg_a0, reg_b);
		__m256d min1 = _mm256_min_pd(reg_a1, reg_b);

		_mm256_storeu_pd(&a[i], min0);
		_mm256_storeu_pd(&a[i + 4], min1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d min = _mm256_min_pd(reg_a, reg_b);
		_mm256_storeu_pd(&a[i], min);
	}

	for (; i < n; ++i)
	{
		if (b < a[i]) a[i] = b;
	}
}

void MathUtils::vector_clamp(const double* const __restrict a, const double* const __restrict min, const double* const __restrict max,
	double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_min0 = _mm256_loadu_pd(&min[i]);
		__m256d reg_min1 = _mm256_loadu_pd(&min[i + 4]);

		__m256d reg_max0 = _mm256_loadu_pd(&max[i]);
		__m256d reg_max1 = _mm256_loadu_pd(&max[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_a0, reg_max0), reg_min0);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_a1, reg_max1), reg_min1);

		_mm256_storeu_pd(&r[i], clamp0);
		_mm256_storeu_pd(&r[i + 4], clamp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_min = _mm256_loadu_pd(&min[i]);
		__m256d reg_max = _mm256_loadu_pd(&max[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_a, reg_max), reg_min);
		_mm256_storeu_pd(&r[i], clamp);
	}

	for (; i < n; ++i)
	{
		r[i] = std::clamp(a[i], min[i], max[i]);
	}
}

void MathUtils::vector_clamp(double* const __restrict a, const double* const __restrict min, const double* const __restrict max, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_min0 = _mm256_loadu_pd(&min[i]);
		__m256d reg_min1 = _mm256_loadu_pd(&min[i + 4]);

		__m256d reg_max0 = _mm256_loadu_pd(&max[i]);
		__m256d reg_max1 = _mm256_loadu_pd(&max[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_a0, reg_max0), reg_min0);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_a1, reg_max1), reg_min1);

		_mm256_storeu_pd(&a[i], clamp0);
		_mm256_storeu_pd(&a[i + 4], clamp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_min = _mm256_loadu_pd(&min[i]);
		__m256d reg_max = _mm256_loadu_pd(&max[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_a, reg_max), reg_min);
		_mm256_storeu_pd(&a[i], clamp);
	}

	for (; i < n; ++i)
	{
		a[i] = std::clamp(a[i], min[i], max[i]);
	}
}

void MathUtils::vector_clamp(const double* const __restrict a, double min, const double* const __restrict max,
	double* const __restrict r, size_t n)
{
	const __m256d reg_min = _mm256_set1_pd(min);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_max0 = _mm256_loadu_pd(&max[i]);
		__m256d reg_max1 = _mm256_loadu_pd(&max[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_a0, reg_max0), reg_min);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_a1, reg_max1), reg_min);

		_mm256_storeu_pd(&r[i], clamp0);
		_mm256_storeu_pd(&r[i + 4], clamp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_max = _mm256_loadu_pd(&max[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_a, reg_max), reg_min);
		_mm256_storeu_pd(&r[i], clamp);
	}

	for (; i < n; ++i)
	{
		r[i] = std::clamp(a[i], min, max[i]);
	}
}

void MathUtils::vector_clamp(double* const __restrict a, double min, const double* const __restrict max, size_t n)
{
	const __m256d reg_min = _mm256_set1_pd(min);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_max0 = _mm256_loadu_pd(&max[i]);
		__m256d reg_max1 = _mm256_loadu_pd(&max[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_a0, reg_max0), reg_min);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_a1, reg_max1), reg_min);

		_mm256_storeu_pd(&a[i], clamp0);
		_mm256_storeu_pd(&a[i + 4], clamp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_max = _mm256_loadu_pd(&max[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_a, reg_max), reg_min);
		_mm256_storeu_pd(&a[i], clamp);
	}

	for (; i < n; ++i)
	{
		a[i] = std::clamp(a[i], min, max[i]);
	}
}

void MathUtils::vector_clamp(const double* const __restrict a, const double* const __restrict min, double max,
	double* const __restrict r, size_t n)
{
	const __m256d reg_max = _mm256_set1_pd(max);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_min0 = _mm256_loadu_pd(&min[i]);
		__m256d reg_min1 = _mm256_loadu_pd(&min[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_a0, reg_max), reg_min0);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_a1, reg_max), reg_min1);

		_mm256_storeu_pd(&r[i], clamp0);
		_mm256_storeu_pd(&r[i + 4], clamp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_min = _mm256_loadu_pd(&min[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_a, reg_max), reg_min);
		_mm256_storeu_pd(&r[i], clamp);
	}

	for (; i < n; ++i)
	{
		r[i] = std::clamp(a[i], min[i], max);
	}
}

void MathUtils::vector_clamp(double* const __restrict a, const double* const __restrict min, double max, size_t n)
{
	const __m256d reg_max = _mm256_set1_pd(max);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d reg_min0 = _mm256_loadu_pd(&min[i]);
		__m256d reg_min1 = _mm256_loadu_pd(&min[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_a0, reg_max), reg_min0);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_a1, reg_max), reg_min1);

		_mm256_storeu_pd(&a[i], clamp0);
		_mm256_storeu_pd(&a[i + 4], clamp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d reg_min = _mm256_loadu_pd(&min[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_a, reg_max), reg_min);
		_mm256_storeu_pd(&a[i], clamp);
	}

	for (; i < n; ++i)
	{
		a[i] = std::clamp(a[i], min[i], max);
	}
}

void MathUtils::vector_clamp(const double* const __restrict a, double min, double max, double* const __restrict r, size_t n)
{
	const __m256d reg_min = _mm256_set1_pd(min);
	const __m256d reg_max = _mm256_set1_pd(max);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_a0, reg_max), reg_min);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_a1, reg_max), reg_min);

		_mm256_storeu_pd(&r[i], clamp0);
		_mm256_storeu_pd(&r[i + 4], clamp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_a, reg_max), reg_min);
		_mm256_storeu_pd(&r[i], clamp);
	}

	for (; i < n; ++i)
	{
		r[i] = std::clamp(a[i], min, max);
	}
}

void MathUtils::vector_clamp(double* const __restrict a, double min, double max, size_t n)
{
	const __m256d reg_min = _mm256_set1_pd(min);
	const __m256d reg_max = _mm256_set1_pd(max);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d clamp0 = _mm256_max_pd(_mm256_min_pd(reg_a0, reg_max), reg_min);
		__m256d clamp1 = _mm256_max_pd(_mm256_min_pd(reg_a1, reg_max), reg_min);

		_mm256_storeu_pd(&a[i], clamp0);
		_mm256_storeu_pd(&a[i + 4], clamp1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d clamp = _mm256_max_pd(_mm256_min_pd(reg_a, reg_max), reg_min);
		_mm256_storeu_pd(&a[i], clamp);
	}

	for (; i < n; ++i)
	{
		a[i] = std::clamp(a[i], min, max);
	}
}

/* Vector activation functions */

// Vectorizes the sigmoid function applied to a vector and writes the result into the provided vector -> r = sigmoid(a)
void MathUtils::vector_sigmoid(const double* const __restrict a, double* const __restrict r, size_t n)
{
	const __m256d neg_mask = _mm256_set1_pd(-0.0);
	const __m256d reg_one = _mm256_set1_pd(1.0);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d neg0 = _mm256_xor_pd(reg_a0, neg_mask);
		__m256d neg1 = _mm256_xor_pd(reg_a1, neg_mask);

		__m256d exp0 = _mm256_exp_pd(neg0);
		__m256d exp1 = _mm256_exp_pd(neg1);

		__m256d denom0 = _mm256_add_pd(exp0, reg_one);
		__m256d denom1 = _mm256_add_pd(exp1, reg_one);

		__m256d sig0 = _mm256_div_pd(reg_one, denom0);
		__m256d sig1 = _mm256_div_pd(reg_one, denom1);

		_mm256_storeu_pd(&r[i], sig0);
		_mm256_storeu_pd(&r[i + 4], sig1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d neg = _mm256_xor_pd(reg_a, neg_mask);
		__m256d exp = _mm256_exp_pd(neg);
		__m256d denom = _mm256_add_pd(exp, reg_one);
		__m256d sig = _mm256_div_pd(reg_one, denom);
		_mm256_storeu_pd(&r[i], sig);
	}

	for (; i < n; ++i)
	{
		r[i] = 1.0 / (1.0 + std::exp(-a[i]));
	}
}

void MathUtils::vector_sigmoid(double* const __restrict a, size_t n)
{
	const __m256d neg_mask = _mm256_set1_pd(-0.0);
	const __m256d reg_one = _mm256_set1_pd(1.0);

	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d neg0 = _mm256_xor_pd(reg_a0, neg_mask);
		__m256d neg1 = _mm256_xor_pd(reg_a1, neg_mask);

		__m256d exp0 = _mm256_exp_pd(neg0);
		__m256d exp1 = _mm256_exp_pd(neg1);

		__m256d denom0 = _mm256_add_pd(exp0, reg_one);
		__m256d denom1 = _mm256_add_pd(exp1, reg_one);

		__m256d sig0 = _mm256_div_pd(reg_one, denom0);
		__m256d sig1 = _mm256_div_pd(reg_one, denom1);

		_mm256_storeu_pd(&a[i], sig0);
		_mm256_storeu_pd(&a[i + 4], sig1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d neg = _mm256_xor_pd(reg_a, neg_mask);
		__m256d exp = _mm256_exp_pd(neg);
		__m256d denom = _mm256_add_pd(exp, reg_one);
		__m256d sig = _mm256_div_pd(reg_one, denom);
		_mm256_storeu_pd(&a[i], sig);
	}

	for (; i < n; ++i)
	{
		a[i] = 1.0 / (1.0 + std::exp(-a[i]));
	}
}

void MathUtils::vector_tanh(const double* const __restrict a, double* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d tanh0 = _mm256_tanh_pd(reg_a0);
		__m256d tanh1 = _mm256_tanh_pd(reg_a1);

		_mm256_storeu_pd(&r[i], tanh0);
		_mm256_storeu_pd(&r[i + 4], tanh1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d tanh = _mm256_tanh_pd(reg_a);
		_mm256_storeu_pd(&r[i], tanh);
	}

	for (; i < n; ++i)
	{
		r[i] = std::tanh(a[i]);
	}
}

void MathUtils::vector_tanh(double* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 8 <= n; i += 8)
	{
		__m256d reg_a0 = _mm256_loadu_pd(&a[i]);
		__m256d reg_a1 = _mm256_loadu_pd(&a[i + 4]);

		__m256d tanh0 = _mm256_tanh_pd(reg_a0);
		__m256d tanh1 = _mm256_tanh_pd(reg_a1);

		_mm256_storeu_pd(&a[i], tanh0);
		_mm256_storeu_pd(&a[i + 4], tanh1);
	}

	for (; i + 4 <= n; i += 4)
	{
		__m256d reg_a = _mm256_loadu_pd(&a[i]);
		__m256d tanh = _mm256_tanh_pd(reg_a);
		_mm256_storeu_pd(&a[i], tanh);
	}

	for (; i < n; ++i)
	{
		a[i] = std::tanh(a[i]);
	}
}