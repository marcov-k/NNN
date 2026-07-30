#include "pch.h"
#include "MathUtils.h"

// Collection of various math and AVX2 SIMD vectorization utility functions.

/* Register operations */

// Computes the sum of a 256-bit register of floats.
float MathUtils::sum_m256(__m256 v)
{
	const __m128 hi = _mm256_extractf128_ps(v, 1);
	const __m128 lo = _mm256_castps256_ps128(v);
	const __m128 sum128 = _mm_add_ps(lo, hi);
	const __m128 shuf = _mm_movehl_ps(sum128, sum128);
	const __m128 sums = _mm_add_ps(sum128, shuf);
	const __m128 s2 = _mm_add_ss(sums, _mm_shuffle_ps(sums, sums, 0x55));
	return _mm_cvtss_f32(s2);
}

float MathUtils::max_m256(__m256 v)
{
	const __m256 v_high = _mm256_permute2f128_ps(v, v, 0x31);
	const __m256 v_max1 = _mm256_max_ps(v, v_high);
	const __m256 v_shuf1 = _mm256_permute_ps(v_max1, _MM_SHUFFLE(2, 3, 0, 1));
	const __m256 v_max2 = _mm256_max_ps(v_max1, v_shuf1);
	const __m256 v_shuf2 = _mm256_permute_ps(v_max2, _MM_SHUFFLE(1, 0, 3, 2));
	const __m256 v_max3 = _mm256_max_ps(v_max2, v_shuf2);
	return _mm_cvtss_f32(_mm256_castps256_ps128(v_max3));
}

float MathUtils::min_m256(__m256 v)
{
	const __m256 v_high = _mm256_permute2f128_ps(v, v, 0x31);
	const __m256 v_min1 = _mm256_min_ps(v, v_high);
	const __m256 v_shuf1 = _mm256_permute_ps(v_min1, _MM_SHUFFLE(2, 3, 0, 1));
	const __m256 v_min2 = _mm256_min_ps(v_min1, v_shuf1);
	const __m256 v_shuf2 = _mm256_permute_ps(v_min2, _MM_SHUFFLE(1, 0, 3, 2));
	const __m256 v_min3 = _mm256_min_ps(v_min2, v_shuf2);
	return _mm_cvtss_f32(_mm256_castps256_ps128(v_min3));
}

/* Vector addition */

// Vectorizes the addition of two vectors and writes the result into the provided vector -> c = a + b
void MathUtils::vector_add(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_add_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_add_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_add_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_add_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_add_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] + b[i];
	}
}

void MathUtils::vector_add(float* const __restrict a, const float* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_add_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&a[i + 8], _mm256_add_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_add_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_add_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_add_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		a[i] += b[i];
	}
}

void MathUtils::vector_add(const float* const __restrict a, float b, float* const __restrict c, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_add_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&c[i + 8], _mm256_add_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&c[i + 16], _mm256_add_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&c[i + 24], _mm256_add_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_add_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] + b;
	}
}

void MathUtils::vector_add(float* const __restrict a, float b, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_add_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&a[i + 8], _mm256_add_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&a[i + 16], _mm256_add_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&a[i + 24], _mm256_add_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_add_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		a[i] += b;
	}
}

/* Vector subtraction */

// Vectorizes the subtraction of two vectors and writes the result into the provided vector -> c = a - b
void MathUtils::vector_sub(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_sub_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_sub_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_sub_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_sub_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_sub_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] - b[i];
	}
}

void MathUtils::vector_sub(float* const __restrict a, const float* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_sub_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&a[i + 8], _mm256_sub_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_sub_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_sub_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}
	
	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_sub_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		a[i] -= b[i];
	}
}

void MathUtils::vector_sub(const float* const __restrict a, float b, float* const __restrict c, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_sub_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&c[i + 8], _mm256_sub_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&c[i + 16], _mm256_sub_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&c[i + 24], _mm256_sub_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_sub_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] - b;
	}
}

void MathUtils::vector_sub(float* const __restrict a, float b, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_sub_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&a[i + 8], _mm256_sub_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&a[i + 16], _mm256_sub_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&a[i + 24], _mm256_sub_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_sub_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		a[i] -= b;
	}
}

void MathUtils::vector_sub(float a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	const __m256 reg_a = _mm256_set1_ps(a);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = a - b[i];
	}
}

void MathUtils::vector_sub(float a, float* const __restrict b, size_t n)
{
	const __m256 reg_a = _mm256_set1_ps(a);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&b[i], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i])));
		_mm256_store_ps(&b[i + 8], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&b[i + 16], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&b[i + 24], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&b[i], _mm256_sub_ps(reg_a, _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		b[i] = a - b[i];
	}
}

/* Vector multiplication */

// Vectorizes the multiplication of two vectors and writes the result into the provided vector -> c = a * b
void MathUtils::vector_mul(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] * b[i];
	}
}

void MathUtils::vector_mul(float* const __restrict a, const float* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 23 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&a[i + 8], _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		a[i] *= b[i];
	}
}

void MathUtils::vector_mul(const float* const __restrict a, float b, float* const __restrict c, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&c[i + 8], _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&c[i + 16], _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&c[i + 24], _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] * b;
	}
}

void MathUtils::vector_mul(float* const __restrict a, float b, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&a[i + 8], _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&a[i + 16], _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&a[i + 24], _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		a[i] *= b;
	}
}

/* Vector division */

// Vectorizes the division of two vectors and writes the result into the provided vector -> c = a / b
void MathUtils::vector_div(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_div_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_div_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_div_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_div_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_div_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] / b[i];
	}
}

void MathUtils::vector_div(float* const __restrict a, const float* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_div_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&a[i + 8], _mm256_div_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_div_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_div_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_div_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		a[i] /= b[i];
	}
}

void MathUtils::vector_div(const float* const __restrict a, float b, float* const __restrict c, size_t n)
{
	const float recip_b = 1.0f / b;
	const __m256 reg_recip_b = _mm256_set1_ps(recip_b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), reg_recip_b));
		_mm256_store_ps(&c[i + 8], _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), reg_recip_b));
		_mm256_store_ps(&c[i + 16], _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), reg_recip_b));
		_mm256_store_ps(&c[i + 24], _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), reg_recip_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), reg_recip_b));
	}

	for (; i < n; ++i)
	{
		c[i] = a[i] * recip_b;
	}
}

void MathUtils::vector_div(float* const __restrict a, float b, size_t n)
{
	const float recip_b = 1.0f / b;
	const __m256 reg_recip_b = _mm256_set1_ps(recip_b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), reg_recip_b));
		_mm256_store_ps(&a[i + 8], _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), reg_recip_b));
		_mm256_store_ps(&a[i + 16], _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), reg_recip_b));
		_mm256_store_ps(&a[i + 24], _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), reg_recip_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), reg_recip_b));
	}

	for (; i < n; ++i)
	{
		a[i] *= recip_b;
	}
}

void MathUtils::vector_div(float a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	const __m256 reg_a = _mm256_set1_ps(a);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = a / b[i];
	}
}

void MathUtils::vector_div(float a, float* const __restrict b, size_t n)
{
	const __m256 reg_a = _mm256_set1_ps(a);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&b[i], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i])));
		_mm256_store_ps(&b[i + 8], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&b[i + 16], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&b[i + 24], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&b[i], _mm256_div_ps(reg_a, _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		b[i] = a / b[i];
	}
}

/* Vector exponentiation */

// Vectorizes the exponentiation of two vectors and writes the result into the provided vector -> c = a ^ b
void MathUtils::vector_pow(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_pow_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_pow_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_pow_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_pow_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_pow_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = std::pow(a[i], b[i]);
	}
}

void MathUtils::vector_pow(float* const __restrict a, const float* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_pow_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&a[i + 8], _mm256_pow_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_pow_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_pow_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_pow_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		a[i] = std::pow(a[i], b[i]);
	}
}

void MathUtils::vector_pow(const float* const __restrict a, float b, float* const __restrict c, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_pow_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&c[i + 8], _mm256_pow_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&c[i + 16], _mm256_pow_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&c[i + 24], _mm256_pow_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_pow_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		c[i] = std::pow(a[i], b);
	}
}

void MathUtils::vector_pow(float* const __restrict a, float b, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_pow_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&a[i + 8], _mm256_pow_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&a[i + 16], _mm256_pow_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&a[i + 24], _mm256_pow_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_pow_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		a[i] = std::pow(a[i], b);
	}
}

void MathUtils::vector_pow(float a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	const __m256 reg_a = _mm256_set1_ps(a);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_pow_ps(reg_a, _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_pow_ps(reg_a, _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_pow_ps(reg_a, _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_pow_ps(reg_a, _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_pow_ps(reg_a, _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = std::pow(a, b[i]);
	}
}

/* Vector logarithm */

// Vectorizes the logarithm of a vector argument and base and writes the result into the provided vector -> r = log_base(arg)
void MathUtils::vector_log(const float* const __restrict arg, const float* const __restrict log_base, float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i])), _mm256_log_ps(_mm256_load_ps(&log_base[i]))));
		_mm256_store_ps(&r[i + 8], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 8])), _mm256_log_ps(_mm256_load_ps(&log_base[i + 8]))));
		_mm256_store_ps(&r[i + 16], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 16])), _mm256_log_ps(_mm256_load_ps(&log_base[i + 16]))));
		_mm256_store_ps(&r[i + 24], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 24])), _mm256_log_ps(_mm256_load_ps(&log_base[i + 24]))));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i])), _mm256_log_ps(_mm256_load_ps(&log_base[i]))));
	}

	for (; i < n; ++i)
	{
		r[i] = std::log(arg[i]) / std::log(log_base[i]);
	}
}

void MathUtils::vector_log(float* const __restrict arg, const float* const __restrict log_base, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&arg[i], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i])), _mm256_log_ps(_mm256_load_ps(&log_base[i]))));
		_mm256_store_ps(&arg[i + 8], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 8])), _mm256_log_ps(_mm256_load_ps(&log_base[i + 8]))));
		_mm256_store_ps(&arg[i + 16], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 16])), _mm256_log_ps(_mm256_load_ps(&log_base[i + 16]))));
		_mm256_store_ps(&arg[i + 24], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 24])), _mm256_log_ps(_mm256_load_ps(&log_base[i + 24]))));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&arg[i], _mm256_div_ps(_mm256_log_ps(_mm256_load_ps(&arg[i])), _mm256_log_ps(_mm256_load_ps(&log_base[i]))));
	}

	for (; i < n; ++i)
	{
		arg[i] = std::log(arg[i]) / std::log(log_base[i]);
	}
}

void MathUtils::vector_log(const float* const __restrict arg, float log_base, float* const __restrict r, size_t n)
{
	const float ln_base = 1.0f / std::log(log_base);
	const __m256 reg_ln_base = _mm256_set1_ps(ln_base);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i])), reg_ln_base));
		_mm256_store_ps(&r[i + 8], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 8])), reg_ln_base));
		_mm256_store_ps(&r[i + 16], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 16])), reg_ln_base));
		_mm256_store_ps(&r[i + 24], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 24])), reg_ln_base));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i])), reg_ln_base));
	}

	for (; i < n; ++i)
	{
		r[i] = std::log(arg[i]) * ln_base;
	}
}

void MathUtils::vector_log(float* const __restrict arg, float log_base, size_t n)
{
	const float ln_base = 1.0f / std::log(log_base);
	const __m256 reg_ln_base = _mm256_set1_ps(ln_base);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&arg[i], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i])), reg_ln_base));
		_mm256_store_ps(&arg[i + 8], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 8])), reg_ln_base));
		_mm256_store_ps(&arg[i + 16], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 16])), reg_ln_base));
		_mm256_store_ps(&arg[i + 24], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i + 24])), reg_ln_base));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&arg[i], _mm256_mul_ps(_mm256_log_ps(_mm256_load_ps(&arg[i])), reg_ln_base));
	}

	for (; i < n; ++i)
	{
		arg[i] = std::log(arg[i]) * ln_base;
	}
}

void MathUtils::vector_log(float arg, const float* const __restrict log_base, float* const __restrict r, size_t n)
{
	const float ln_arg = std::log(arg);
	const __m256 reg_ln_arg = _mm256_set1_ps(ln_arg);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_div_ps(reg_ln_arg, _mm256_log_ps(_mm256_load_ps(&log_base[i]))));
		_mm256_store_ps(&r[i + 8], _mm256_div_ps(reg_ln_arg, _mm256_log_ps(_mm256_load_ps(&log_base[i + 8]))));
		_mm256_store_ps(&r[i + 16], _mm256_div_ps(reg_ln_arg, _mm256_log_ps(_mm256_load_ps(&log_base[i + 16]))));
		_mm256_store_ps(&r[i + 24], _mm256_div_ps(reg_ln_arg, _mm256_log_ps(_mm256_load_ps(&log_base[i + 24]))));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_div_ps(reg_ln_arg, _mm256_log_ps(_mm256_load_ps(&log_base[i]))));
	}

	for (; i < n; ++i)
	{
		r[i] = ln_arg / std::log(log_base[i]);
	}
}

/* Vector fused multiply addition */

// Vectorizes the fused multiply addition of three vectors and writes the result into the provided vector -> r = a + b * c
void MathUtils::vector_fmadd(const float* const __restrict a, const float* const __restrict b, const float* const __restrict c,
	float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_fmadd_ps(_mm256_load_ps(&b[i]), _mm256_load_ps(&c[i]), _mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 8]), _mm256_load_ps(&c[i + 8]), _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 16]), _mm256_load_ps(&c[i + 16]), _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 24]), _mm256_load_ps(&c[i + 24]), _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_fmadd_ps(_mm256_load_ps(&b[i]), _mm256_load_ps(&c[i]), _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] + b[i] * c[i];
	}
}

void MathUtils::vector_fmadd(float* const __restrict a, const float* const __restrict b, const float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_fmadd_ps(_mm256_load_ps(&b[i]), _mm256_load_ps(&c[i]), _mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 8]), _mm256_load_ps(&c[i + 8]), _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 16]), _mm256_load_ps(&c[i + 16]), _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 24]), _mm256_load_ps(&c[i + 24]), _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_fmadd_ps(_mm256_load_ps(&b[i]), _mm256_load_ps(&c[i]), _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		a[i] += b[i] * c[i];
	}
}

void MathUtils::vector_fmadd(const float* const __restrict a, const float* const __restrict b, float c,
	float* const __restrict r, size_t n)
{
	const __m256 reg_c = _mm256_set1_ps(c);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_fmadd_ps(_mm256_load_ps(&b[i]), reg_c, _mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 8]), reg_c, _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 16]), reg_c, _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 24]), reg_c, _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_fmadd_ps(_mm256_load_ps(&b[i]), reg_c, _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] + b[i] * c;
	}
}

void MathUtils::vector_fmadd(float* const __restrict a, const float* const __restrict b, float c, size_t n)
{
	const __m256 reg_c = _mm256_set1_ps(c);

	size_t i = 0;
	for (; i + 32 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_fmadd_ps(_mm256_load_ps(&b[i]), reg_c, _mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 8]), reg_c, _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 16]), reg_c, _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_fmadd_ps(_mm256_load_ps(&b[i + 24]), reg_c, _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_fmadd_ps(_mm256_load_ps(&b[i]), reg_c, _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		a[i] += b[i] * c;
	}
}

/* Vector fused negative multiply addition */

// Vectorizes the fused negative multiply addition of three vectors and writes the result into the provided vector -> r = a - b * c
void MathUtils::vector_fnmadd(const float* const __restrict a, const float* const __restrict b, const float* const __restrict c,
	float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_fnmadd_ps(_mm256_load_ps(&b[i]), _mm256_load_ps(&c[i]), _mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 8]), _mm256_load_ps(&c[i + 8]), _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 16]), _mm256_load_ps(&c[i + 16]), _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 24]), _mm256_load_ps(&c[i + 24]), _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_fnmadd_ps(_mm256_load_ps(&b[i]), _mm256_load_ps(&c[i]), _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] - b[i] * c[i];
	}
}

void MathUtils::vector_fnmadd(float* const __restrict a, const float* const __restrict b, const float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_fnmadd_ps(_mm256_load_ps(&b[i]), _mm256_load_ps(&c[i]), _mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 8]), _mm256_load_ps(&c[i + 8]), _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 16]), _mm256_load_ps(&c[i + 16]), _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 24]), _mm256_load_ps(&c[i + 24]), _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_fnmadd_ps(_mm256_load_ps(&b[i]), _mm256_load_ps(&c[i]), _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		a[i] -= b[i] * c[i];
	}
}

void MathUtils::vector_fnmadd(const float* const __restrict a, const float* const __restrict b, float c,
	float* const __restrict r, size_t n)
{
	const __m256 reg_c = _mm256_set1_ps(c);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_fnmadd_ps(_mm256_load_ps(&b[i]), reg_c, _mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 8]), reg_c, _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 16]), reg_c, _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 24]), reg_c, _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_fnmadd_ps(_mm256_load_ps(&b[i]), reg_c, _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] - b[i] * c;
	}
}

void MathUtils::vector_fnmadd(float* const __restrict a, const float* const __restrict b, float c, size_t n)
{
	const __m256 reg_c = _mm256_set1_ps(c);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_fnmadd_ps(_mm256_load_ps(&b[i]), reg_c, _mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 8]), reg_c, _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 16]), reg_c, _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_fnmadd_ps(_mm256_load_ps(&b[i + 24]), reg_c, _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_fnmadd_ps(_mm256_load_ps(&b[i]), reg_c, _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		a[i] -= b[i] * c;
	}
}

/* Vector square */

// Vectorizes the square of a vector and writes the result into the provided vector -> r = a ^ 2
void MathUtils::vector_sq(const float* const __restrict a, float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = a[i] * a[i];
	}
}

void MathUtils::vector_sq(float* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		a[i] *= a[i];
	}
}

/* Vector square root */

// Vectorizes the square root of a vector and writes the result into the provided vector -> r = sqrt(a)
void MathUtils::vector_sqrt(const float* const __restrict a, float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_sqrt_ps(_mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_sqrt_ps(_mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_sqrt_ps(_mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_sqrt_ps(_mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_sqrt_ps(_mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = std::sqrt(a[i]);
	}
}

void MathUtils::vector_sqrt(float* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_sqrt_ps(_mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_sqrt_ps(_mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_sqrt_ps(_mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_sqrt_ps(_mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_sqrt_ps(_mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		a[i] = std::sqrt(a[i]);
	}
}

/* Vector natural exponentiation */

// Vectorizes the natural exponentiation of a vector and writes the result into the provided vector -> r = e ^ a
void MathUtils::vector_exp(const float* const __restrict a, float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_exp_ps(_mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_exp_ps(_mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_exp_ps(_mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_exp_ps(_mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_exp_ps(_mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = std::exp(a[i]);
	}
}

void MathUtils::vector_exp(float* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_exp_ps(_mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_exp_ps(_mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_exp_ps(_mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_exp_ps(_mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_exp_ps(_mm256_load_ps(&a[i])));
	}
	for (; i < n; ++i)
	{
		a[i] = std::exp(a[i]);
	}
}

/* Vector natural logarithm */

// Vectorizes the natural logarithm of a vector and writes the result into the provided vector -> r = ln(a)
void MathUtils::vector_ln(const float* const __restrict a, float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_log_ps(_mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_log_ps(_mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_log_ps(_mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_log_ps(_mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_log_ps(_mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = std::log(a[i]);
	}
}

void MathUtils::vector_ln(float* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_log_ps(_mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_log_ps(_mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_log_ps(_mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_log_ps(_mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_log_ps(_mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		a[i] = std::log(a[i]);
	}
}

/* Vector operations */

// Vectorizes the sum of a vector.
float MathUtils::vector_sum(const float* const __restrict a, size_t n)
{
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		acc0 = _mm256_add_ps(acc0, _mm256_load_ps(&a[i]));
		acc1 = _mm256_add_ps(acc1, _mm256_load_ps(&a[i + 8]));
		acc2 = _mm256_add_ps(acc2, _mm256_load_ps(&a[i + 16]));
		acc3 = _mm256_add_ps(acc3, _mm256_load_ps(&a[i + 24]));
	}

	acc0 = _mm256_add_ps(acc0, acc1);
	acc2 = _mm256_add_ps(acc2, acc3);
	acc0 = _mm256_add_ps(acc0, acc2);

	for (; i + 8 <= n; i += 8)
	{
		acc0 = _mm256_add_ps(acc0, _mm256_load_ps(&a[i]));
	}

	float sum = sum_m256(acc0);

	for (; i < n; ++i)
	{
		sum += a[i];
	}

	return sum;
}

float MathUtils::vector_max(const float* const __restrict a, size_t n)
{
	__m256 max0 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
	__m256 max1 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
	__m256 max2 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
	__m256 max3 = _mm256_set1_ps(std::numeric_limits<float>::lowest());

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		max0 = _mm256_max_ps(max0, _mm256_load_ps(&a[i]));
		max1 = _mm256_max_ps(max1, _mm256_load_ps(&a[i + 8]));
		max2 = _mm256_max_ps(max2, _mm256_load_ps(&a[i + 16]));
		max3 = _mm256_max_ps(max3, _mm256_load_ps(&a[i + 24]));
	}

	max0 = _mm256_max_ps(max0, max1);
	max2 = _mm256_max_ps(max2, max3);
	max0 = _mm256_max_ps(max0, max2);

	for (; i + 8 <= n; i += 8)
	{
		max0 = _mm256_max_ps(max0, _mm256_load_ps(&a[i]));
	}

	float max = max_m256(max0);

	for (; i < n; ++i)
	{
		if (a[i] > max) max = a[i];
	}

	return max;
}

float MathUtils::vector_min(const float* const __restrict a, size_t n)
{
	__m256 min0 = _mm256_set1_ps((std::numeric_limits<float>::max)());
	__m256 min1 = _mm256_set1_ps((std::numeric_limits<float>::max)());
	__m256 min2 = _mm256_set1_ps((std::numeric_limits<float>::max)());
	__m256 min3 = _mm256_set1_ps((std::numeric_limits<float>::max)());

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		min0 = _mm256_min_ps(min0, _mm256_load_ps(&a[i]));
		min1 = _mm256_min_ps(min1, _mm256_load_ps(&a[i + 8]));
		min2 = _mm256_min_ps(min2, _mm256_load_ps(&a[i + 16]));
		min3 = _mm256_min_ps(min3, _mm256_load_ps(&a[i + 24]));
	}

	min0 = _mm256_min_ps(min0, min1);
	min2 = _mm256_min_ps(min2, min3);
	min0 = _mm256_min_ps(min0, min2);

	for (; i + 8 <= n; i += 8)
	{
		min0 = _mm256_min_ps(min0, _mm256_load_ps(&a[i]));
	}

	float min = min_m256(min0);

	for (; i < n; ++i)
	{
		if (a[i] < min) min = a[i];
	}

	return min;
}

float MathUtils::vector_dot(const float* const __restrict a, const float* const __restrict b, size_t n)
{
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();

	size_t i = 0;
	for (; i + 32 <= n; i += 8)
	{
		acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	acc0 = _mm256_add_ps(acc0, acc1);
	acc2 = _mm256_add_ps(acc2, acc3);
	acc0 = _mm256_add_ps(acc0, acc2);

	for (; i + 8 <= n; i += 8)
	{
		acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	float dot = sum_m256(acc0);

	for (; i < n; ++i)
	{
		dot += a[i] * b[i];
	}

	return dot;
}

float MathUtils::vector_dot(const float* __restrict a, const float* __restrict b, size_t a_off, size_t b_off, size_t n)
{
	const float* const __restrict p_a = &a[a_off];
	const float* const __restrict p_b = &b[b_off];

	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	__m256 acc2 = _mm256_setzero_ps();
	__m256 acc3 = _mm256_setzero_ps();

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_load_ps(&p_a[i]), _mm256_load_ps(&p_b[i])));
		acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_load_ps(&p_a[i + 8]), _mm256_load_ps(&p_b[i + 8])));
		acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_load_ps(&p_a[i + 16]), _mm256_load_ps(&p_b[i + 16])));
		acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_load_ps(&p_a[i + 24]), _mm256_load_ps(&p_b[i + 24])));
	}

	acc0 = _mm256_add_ps(acc0, acc1);
	acc2 = _mm256_add_ps(acc2, acc3);
	acc0 = _mm256_add_ps(acc0, acc2);

	for (; i + 8 <= n; i += 8)
	{
		acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_load_ps(&p_a[i]), _mm256_load_ps(&p_b[i])));
	}

	float dot = sum_m256(acc0);

	for (; i < n; ++i)
	{
		dot += p_a[i] * p_b[i];
	}

	return dot;
}

/* Vector limiting functions */

// Vectorizes the max of two vectors and writes the result into the provided vector -> c = max(a, b)
void MathUtils::vector_max(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_max_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_max_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_max_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_max_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_max_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = max(a[i], b[i]);
	}
}

void MathUtils::vector_max(float* const __restrict a, const float* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&a[i + 8], _mm256_max_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_max_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_max_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		if (b[i] > a[i]) a[i] = b[i];
	}
}

void MathUtils::vector_max(const float* const __restrict a, float b, float* const __restrict c, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_max_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&c[i + 8], _mm256_max_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&c[i + 16], _mm256_max_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&c[i + 24], _mm256_max_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_max_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		c[i] = max(a[i], b);
	}
}

void MathUtils::vector_max(float* const __restrict a, float b, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&a[i + 8], _mm256_max_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&a[i + 16], _mm256_max_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&a[i + 24], _mm256_max_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		if (b > a[i]) a[i] = b;
	}
}

void MathUtils::vector_min(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&c[i + 8], _mm256_min_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&c[i + 16], _mm256_min_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&c[i + 24], _mm256_min_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		c[i] = min(a[i], b[i]);
	}
}

void MathUtils::vector_min(float* const __restrict a, const float* const __restrict b, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
		_mm256_store_ps(&a[i + 8], _mm256_min_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&b[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_min_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&b[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_min_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&b[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i])));
	}

	for (; i < n; ++i)
	{
		if (b[i] < a[i]) a[i] = b[i];
	}
}

void MathUtils::vector_min(const float* const __restrict a, float b, float* const __restrict c, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&c[i], _mm256_min_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&c[i + 8], _mm256_min_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&c[i + 16], _mm256_min_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&c[i + 24], _mm256_min_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&c[i], _mm256_min_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		c[i] = min(a[i], b);
	}
}

void MathUtils::vector_min(float* const __restrict a, float b, size_t n)
{
	const __m256 reg_b = _mm256_set1_ps(b);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_min_ps(_mm256_load_ps(&a[i]), reg_b));
		_mm256_store_ps(&a[i + 8], _mm256_min_ps(_mm256_load_ps(&a[i + 8]), reg_b));
		_mm256_store_ps(&a[i + 16], _mm256_min_ps(_mm256_load_ps(&a[i + 16]), reg_b));
		_mm256_store_ps(&a[i + 24], _mm256_min_ps(_mm256_load_ps(&a[i + 24]), reg_b));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_min_ps(_mm256_load_ps(&a[i]), reg_b));
	}

	for (; i < n; ++i)
	{
		if (b < a[i]) a[i] = b;
	}
}

void MathUtils::vector_clamp(const float* const __restrict a, const float* const __restrict min, const float* const __restrict max,
	float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&max[i])), _mm256_load_ps(&min[i])));
		_mm256_store_ps(&r[i + 8], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&max[i + 8])), _mm256_load_ps(&min[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&max[i + 16])), _mm256_load_ps(&min[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&max[i + 24])), _mm256_load_ps(&min[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&max[i])), _mm256_load_ps(&min[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = std::clamp(a[i], min[i], max[i]);
	}
}

void MathUtils::vector_clamp(float* const __restrict a, const float* const __restrict min, const float* const __restrict max, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&max[i])), _mm256_load_ps(&min[i])));
		_mm256_store_ps(&a[i + 8], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&max[i + 8])), _mm256_load_ps(&min[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&max[i + 16])), _mm256_load_ps(&min[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&max[i + 24])), _mm256_load_ps(&min[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&max[i])), _mm256_load_ps(&min[i])));
	}

	for (; i < n; ++i)
	{
		a[i] = std::clamp(a[i], min[i], max[i]);
	}
}

void MathUtils::vector_clamp(const float* const __restrict a, float min, const float* const __restrict max,
	float* const __restrict r, size_t n)
{
	const __m256 reg_min = _mm256_set1_ps(min);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&max[i])), reg_min));
		_mm256_store_ps(&r[i + 8], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&max[i + 8])), reg_min));
		_mm256_store_ps(&r[i + 16], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&max[i + 16])), reg_min));
		_mm256_store_ps(&r[i + 24], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&max[i + 24])), reg_min));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&max[i])), reg_min));
	}

	for (; i < n; ++i)
	{
		r[i] = std::clamp(a[i], min, max[i]);
	}
}

void MathUtils::vector_clamp(float* const __restrict a, float min, const float* const __restrict max, size_t n)
{
	const __m256 reg_min = _mm256_set1_ps(min);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&max[i])), reg_min));
		_mm256_store_ps(&a[i + 8], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 8]), _mm256_load_ps(&max[i + 8])), reg_min));
		_mm256_store_ps(&a[i + 16], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 16]), _mm256_load_ps(&max[i + 16])), reg_min));
		_mm256_store_ps(&a[i + 24], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 24]), _mm256_load_ps(&max[i + 24])), reg_min));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&max[i])), reg_min));
	}

	for (; i < n; ++i)
	{
		a[i] = std::clamp(a[i], min, max[i]);
	}
}

void MathUtils::vector_clamp(const float* const __restrict a, const float* const __restrict min, float max,
	float* const __restrict r, size_t n)
{
	const __m256 reg_max = _mm256_set1_ps(max);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), reg_max), _mm256_load_ps(&min[i])));
		_mm256_store_ps(&r[i + 8], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 8]), reg_max), _mm256_load_ps(&min[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 16]), reg_max), _mm256_load_ps(&min[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 24]), reg_max), _mm256_load_ps(&min[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), reg_max), _mm256_load_ps(&min[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = std::clamp(a[i], min[i], max);
	}
}

void MathUtils::vector_clamp(float* const __restrict a, const float* const __restrict min, float max, size_t n)
{
	const __m256 reg_max = _mm256_set1_ps(max);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), reg_max), _mm256_load_ps(&min[i])));
		_mm256_store_ps(&a[i + 8], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 8]), reg_max), _mm256_load_ps(&min[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 16]), reg_max), _mm256_load_ps(&min[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 24]), reg_max), _mm256_load_ps(&min[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), reg_max), _mm256_load_ps(&min[i])));
	}

	for (; i < n; ++i)
	{
		a[i] = std::clamp(a[i], min[i], max);
	}
}

void MathUtils::vector_clamp(const float* const __restrict a, float min, float max, float* const __restrict r, size_t n)
{
	const __m256 reg_min = _mm256_set1_ps(min);
	const __m256 reg_max = _mm256_set1_ps(max);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), reg_max), reg_min));
		_mm256_store_ps(&r[i + 8], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 8]), reg_max), reg_min));
		_mm256_store_ps(&r[i + 16], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 16]), reg_max), reg_min));
		_mm256_store_ps(&r[i + 24], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 24]), reg_max), reg_min));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), reg_max), reg_min));
	}

	for (; i < n; ++i)
	{
		r[i] = std::clamp(a[i], min, max);
	}
}

void MathUtils::vector_clamp(float* const __restrict a, float min, float max, size_t n)
{
	const __m256 reg_min = _mm256_set1_ps(min);
	const __m256 reg_max = _mm256_set1_ps(max);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), reg_max), reg_min));
		_mm256_store_ps(&a[i + 8], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 8]), reg_max), reg_min));
		_mm256_store_ps(&a[i + 16], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 16]), reg_max), reg_min));
		_mm256_store_ps(&a[i + 24], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i + 24]), reg_max), reg_min));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_max_ps(_mm256_min_ps(_mm256_load_ps(&a[i]), reg_max), reg_min));
	}

	for (; i < n; ++i)
	{
		a[i] = std::clamp(a[i], min, max);
	}
}

/* Vector activation functions */

// Vectorizes the sigmoid function applied to a vector and writes the result into the provided vector -> r = sigmoid(a)
void MathUtils::vector_sigmoid(const float* const __restrict a, float* const __restrict r, size_t n)
{
	const __m256 neg_mask = _mm256_set1_ps(-0.0f);
	const __m256 reg_one = _mm256_set1_ps(1.0f);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i]), neg_mask)), reg_one)));
		_mm256_store_ps(&r[i + 8], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i + 8]), neg_mask)), reg_one)));
		_mm256_store_ps(&r[i + 16], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i + 16]), neg_mask)), reg_one)));
		_mm256_store_ps(&r[i + 24], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i + 24]), neg_mask)), reg_one)));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i]), neg_mask)), reg_one)));
	}

	for (; i < n; ++i)
	{
		r[i] = 1.0f / (1.0f + std::exp(-a[i]));
	}
}

void MathUtils::vector_sigmoid(float* const __restrict a, size_t n)
{
	const __m256 neg_mask = _mm256_set1_ps(-0.0f);
	const __m256 reg_one = _mm256_set1_ps(1.0f);

	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i]), neg_mask)), reg_one)));
		_mm256_store_ps(&a[i + 8], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i + 8]), neg_mask)), reg_one)));
		_mm256_store_ps(&a[i + 16], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i + 16]), neg_mask)), reg_one)));
		_mm256_store_ps(&a[i + 24], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i + 24]), neg_mask)), reg_one)));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_div_ps(reg_one, _mm256_add_ps(_mm256_exp_ps(_mm256_xor_ps(_mm256_load_ps(&a[i]), neg_mask)), reg_one)));
	}

	for (; i < n; ++i)
	{
		a[i] = 1.0f / (1.0f + std::exp(-a[i]));
	}
}

void MathUtils::vector_tanh(const float* const __restrict a, float* const __restrict r, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&r[i], _mm256_tanh_ps(_mm256_load_ps(&a[i])));
		_mm256_store_ps(&r[i + 8], _mm256_tanh_ps(_mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&r[i + 16], _mm256_tanh_ps(_mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&r[i + 24], _mm256_tanh_ps(_mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&r[i], _mm256_tanh_ps(_mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		r[i] = std::tanh(a[i]);
	}
}

void MathUtils::vector_tanh(float* const __restrict a, size_t n)
{
	size_t i = 0;
	for (; i + 32 <= n; i += 32)
	{
		_mm256_store_ps(&a[i], _mm256_tanh_ps(_mm256_load_ps(&a[i])));
		_mm256_store_ps(&a[i + 8], _mm256_tanh_ps(_mm256_load_ps(&a[i + 8])));
		_mm256_store_ps(&a[i + 16], _mm256_tanh_ps(_mm256_load_ps(&a[i + 16])));
		_mm256_store_ps(&a[i + 24], _mm256_tanh_ps(_mm256_load_ps(&a[i + 24])));
	}

	for (; i + 8 <= n; i += 8)
	{
		_mm256_store_ps(&a[i], _mm256_tanh_ps(_mm256_load_ps(&a[i])));
	}

	for (; i < n; ++i)
	{
		a[i] = std::tanh(a[i]);
	}
}

/* Matrix operations */

// Computes the matrix multiplication of two vectors and writes the result into the provided vector -> r = a @ b_t
void MathUtils::matmul_raw(const float* __restrict a, const float* __restrict b_t, float* __restrict r, size_t batch_count,
	size_t m, size_t n, size_t p, size_t a_batch_stride, size_t b_t_batch_stride, size_t r_batch_stride, size_t a_off,
	size_t b_t_off, size_t r_off, bool use_parallel, bool accumulate)
{
	#pragma warning (disable : 6993)
	#pragma omp parallel for collapse(2) if(use_parallel)
	for (size_t batch = 0; batch < batch_count; ++batch)
	{
		for (size_t i = 0; i < m; ++i)
		{
			const size_t a_base = a_off + batch * a_batch_stride + i * n;
			const size_t b_t_batch_off = b_t_off + batch * b_t_batch_stride;
			const size_t r_base = r_off + batch * r_batch_stride + i * p;

			for (size_t j = 0; j < p; ++j)
			{
				const float val = vector_dot(a, b_t, a_base, b_t_batch_off + j * n, n);
				if (accumulate) r[r_base + j] += val;
				else r[r_base + j] = val;
			}
		}
	}
}

void MathUtils::matmul_reduce_raw(const float* __restrict a_t, const float* __restrict b_t, float* __restrict r, size_t batch_count,
	size_t m, size_t n, size_t p, size_t a_t_batch_stride, size_t b_t_batch_stride, size_t a_t_off, size_t b_t_off, size_t r_off,
	bool use_parallel, bool accumulate)
{
	#pragma omp parallel for if(use_parallel)
	for (size_t i = 0; i < m; ++i)
	{
		const size_t a_t_off_base = a_t_off + i * n;
		const size_t r_off_base = r_off + i * p;
		for (size_t j = 0; j < p; ++j)
		{
			float sum = 0.0;

			const size_t b_t_off_base = b_t_off + j * n;
			for (size_t batch = 0; batch < batch_count; ++batch)
			{
				sum += vector_dot(a_t, b_t, a_t_off_base + batch * a_t_batch_stride,
					b_t_off_base + batch * b_t_batch_stride, n);
			}

			if (accumulate) r[r_off_base + j] += sum;
			else r[r_off_base + j] = sum;
		}
	}
}

void MathUtils::transpose_matrix(const float* __restrict src, float* __restrict dst, size_t src_off, size_t dst_off,
	size_t rows, size_t cols)
{
	for (size_t r = 0; r < rows; ++r)
	{
		for (size_t c = 0; c < cols; ++c)
		{
			dst[dst_off + c * rows + r] = src[src_off + r * cols + c]; // switch row and column indices of data
		}
	}
}

size_t MathUtils::compute_output_position(size_t b, size_t op, const ConvGeometry& g)
{
	size_t offset = b * g.input_strides[0];
	for (size_t i = 0; i < g.spatial_rank; ++i)
	{
		size_t coord = (op / g.out_spatial_strides[i]) % g.out_dims[i];
		offset += coord * g.input_strides[i + 1];
	}
	return offset;
}

void MathUtils::im2col(const float* __restrict input, const ConvGeometry& g, float* __restrict input_col, bool use_parallel)
{
	#pragma omp parallel for if(use_parallel)
	for (size_t b = 0; b < g.batches; ++b)
	{
		const size_t row_base = b * g.out_spatial_size * g.kernel_volume_size;

		for (size_t op = 0; op < g.out_spatial_size; ++op)
		{
			const size_t row = row_base + op * g.kernel_volume_size;
			const size_t base_input = compute_output_position(b, op, g);

			for (size_t k = 0; k < g.kernel_volume_size; ++k)
			{
				input_col[row + k] = input[base_input + g.input_kernel_offset[k]];
			}
		}
	}
}

void MathUtils::col2im(const float* __restrict d_input_col, const ConvGeometry& g, float* __restrict d_input, bool use_parallel)
{
	#pragma omp parallel for if(use_parallel)
	for (int b = 0; b < g.batches; ++b)
	{
		const size_t row_base = b * g.out_spatial_size * g.kernel_volume_size;

		for (size_t op = 0; op < g.out_spatial_size; ++op)
		{
			const size_t row = row_base + op * g.kernel_volume_size;
			const size_t base_input = compute_output_position(b, op, g);

			for (size_t k = 0; k < g.kernel_volume_size; ++k)
			{
				d_input[base_input + g.input_kernel_offset[k]] += d_input_col[row + k];
			}
		}
	}
}

void MathUtils::kernels2matmul(const float* __restrict kernels, const ConvGeometry& g, float* __restrict kernels_mat)
{
	for (size_t f = 0; f < g.filter_count; ++f)
	{
		const size_t filter_offset = f * g.kernel_volume_size;

		for (size_t k = 0; k < g.kernel_volume_size; ++k)
		{
			kernels_mat[filter_offset + k] = kernels[filter_offset + g.kernel_kernel_offset[k]];
		}
	}
}

void MathUtils::matmul2kernels(const float* __restrict kernels_mat, const ConvGeometry& g, float* __restrict kernels, bool accumulate)
{
	for (size_t f = 0; f < g.filter_count; ++f)
	{
		const size_t filter_offset = f * g.kernel_volume_size;

		for (size_t k = 0; k < g.kernel_volume_size; ++k)
		{
			if (accumulate) kernels[filter_offset + g.kernel_kernel_offset[k]] += kernels_mat[filter_offset + k];
			else kernels[filter_offset + g.kernel_kernel_offset[k]] = kernels_mat[filter_offset + k];
		}
	}
}