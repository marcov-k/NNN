#pragma once

#include <immintrin.h>
#include <limits>
#include <omp.h>
#include <random>
#include <span>

#include "DataContainers.h"

// Collection of various math and vectorization utility functions.
class MathUtils
{
public:
	MathUtils() = delete;

	/* Random number generation */

	// Generates a random float in the given range using a uniform real distribution.
	static float get_random_float(float min = 0.0, float max = 1.0)
	{
		thread_local std::random_device rd;
		thread_local std::mt19937 gen(rd());

		std::uniform_real_distribution dis(min, max);

		return dis(gen);
	}

	// Generates a random float from a normal distribution with the given mean and standard deviation.
	static float next_gaussian(float mean = 0.0, float std_dev = 1.0)
	{
		thread_local std::random_device rd;
		thread_local std::mt19937 gen(rd());

		std::normal_distribution dis(mean, std_dev);

		return dis(gen);
	}

	/* Register operations */

	// Computes the sum of a 256-bit register of floats.
	static float sum_m256(__m256 v);

	// Computes the max of a 256-bit register of floats.
	static float max_m256(__m256 v);

	// Computes the min of a 256-bit register of floats.
	static float min_m256(__m256 v);

	/* Vector addition */

	// Vectorizes the addition of two vectors and writes the result into the provided vector -> c = a + b
	static void vector_add(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the addition of two vectors and writes the result into the provided vector -> c = a + b
	static void vector_add(std::span<const float> a, std::span<const float> b, std::span<float> c)
	{
		vector_add(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the addition of two vectors and writes the result into the first vector -> a += b
	static void vector_add(float* const __restrict a, const float* const __restrict b, size_t n);

	// Vectorizes the addition of two vectors and writes the result into the first vector -> a += b
	static void vector_add(std::span<float> a, std::span<const float> b)
	{
		vector_add(a.data(), b.data(), a.size());
	}

	// Vectorizes the addition of a vector and scalar and writes the result into the provided vector -> c = a + b
	static void vector_add(const float* const __restrict a, float b, float* const __restrict c, size_t n);

	// Vectorizes the addition of a vector and scalar and writes the result into the provided vector -> c = a + b
	static void vector_add(std::span<const float> a, float b, std::span<float> c)
	{
		vector_add(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the addition of a vector and scalar and writes the result into the vector -> a += b
	static void vector_add(float* const __restrict a, float b, size_t n);

	// Vectorizes the addition of a vector and scalar and writes the result into the vector -> a += b
	static void vector_add(std::span<float> a, float b)
	{
		vector_add(a.data(), b, a.size());
	}

	/* Vector subtraction */

	// Vectorizes the subtraction of two vectors and writes the result into the provided vector -> c = a - b
	static void vector_sub(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the subtraction of two vectors and writes the result into the provided vector -> c = a - b
	static void vector_sub(std::span<const float> a, std::span<const float> b, std::span<float> c)
	{
		vector_sub(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the subtraction of two vectors and writes the result into the first vector -> a -= b
	static void vector_sub(float* const __restrict a, const float* const __restrict b, size_t n);

	// Vectorizes the subtraction of two vectors and writes the result into the first vector -> a -= b
	static void vector_sub(std::span<float> a, std::span<const float> b)
	{
		vector_sub(a.data(), b.data(), a.size());
	}

	// Vectorizes the subtraction of a vector and scalar and writes the result into the provided vector -> c = a - b
	static void vector_sub(const float* const __restrict a, float b, float* const __restrict c, size_t n);

	// Vectorizes the subtraction of a vector and scalar and writes the result into the provided vector -> c = a - b
	static void vector_sub(std::span<const float> a, float b, std::span<float> c)
	{
		vector_sub(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the subtraction of a vector and scalar and writes the result into the vector -> a -= b
	static void vector_sub(float* const __restrict a, float b, size_t n);

	// Vectorizes the subtraction of a vector and scalar and writes the result into the vector -> a -= b
	static void vector_sub(std::span<float> a, float b)
	{
		vector_sub(a.data(), b, a.size());
	}

	// Vectorizes the subtraction of a scalar and vector and writes the result into the provided vector -> c = a - b
	static void vector_sub(float a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the subtraction of a scalar and vector and writes the result into the provided vector -> c = a - b
	static void vector_sub(float a, std::span<const float> b, std::span<float> c)
	{
		vector_sub(a, b.data(), c.data(), b.size());
	}

	// Vectorizes the subtraction of a scalar and vector and writes the result into the vector -> b = a - b
	static void vector_sub(float a, float* const __restrict b, size_t n);

	// Vectorizes the subtraction of a scalar and vector and writes the result into the vector -> b = a - b
	static void vector_sub(float a, std::span<float> b)
	{
		vector_sub(a, b.data(), b.size());
	}

	/* Vector multiplication */

	// Vectorizes the multiplication of two vectors and writes the result into the provided vector -> c = a * b
	static void vector_mul(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the multiplication of two vectors and writes the result into the provided vector -> c = a * b
	static void vector_mul(std::span<const float> a, std::span<const float> b, std::span<float> c)
	{
		vector_mul(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the multiplication of two vectors and writes the result into the first vector -> a *= b
	static void vector_mul(float* const __restrict a, const float* const __restrict b, size_t n);

	// Vectorizes the multiplication of two vectors and writes the result into the first vector -> a *= b
	static void vector_mul(std::span<float> a, std::span<const float> b)
	{
		vector_mul(a.data(), b.data(), a.size());
	}

	// Vectorizes the multiplication of a vector and scalar and writes the result into the provided vector -> c = a * b
	static void vector_mul(const float* const __restrict a, float b, float* const __restrict c, size_t n);

	// Vectorizes the multiplication of a vector and scalar and writes the result into the provided vector -> c = a * b
	static void vector_mul(std::span<const float> a, float b, std::span<float> c)
	{
		vector_mul(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the multiplication of a vector and scalar and writes the result into the vector -> a *= b
	static void vector_mul(float* const __restrict a, float b, size_t n);

	// Vectorizes the multiplication of a vector and scalar and writes the result into the vector -> a *= b
	static void vector_mul(std::span<float> a, float b)
	{
		vector_mul(a.data(), b, a.size());
	}

	/* Vector division */

	// Vectorizes the division of two vectors and writes the result into the provided vector -> c = a / b
	static void vector_div(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the division of two vectors and writes the result into the provided vector -> c = a / b
	static void vector_div(std::span<const float> a, std::span<const float> b, std::span<float> c)
	{
		vector_div(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the division of two vectors and writes the result into the first vector -> a /= b
	static void vector_div(float* const __restrict a, const float* const __restrict b, size_t n);

	// Vectorizes the division of two vectors and writes the result into the first vector -> a /= b
	static void vector_div(std::span<float> a, std::span<const float> b)
	{
		vector_div(a.data(), b.data(), a.size());
	}

	// Vectorizes the division of a vector and scalar and writes the result into the provided vector -> c = a / b
	static void vector_div(const float* const __restrict a, float b, float* const __restrict c, size_t n);

	// Vectorizes the division of a vector and scalar and writes the result into the provided vector -> c = a / b
	static void vector_div(std::span<const float> a, float b, std::span<float> c)
	{
		vector_div(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the division of a vector and scalar and writes the result into the vector -> a /= b
	static void vector_div(float* const __restrict a, float b, size_t n);

	// Vectorizes the division of a vector and scalar and writes the result into the vector -> a /= b
	static void vector_div(std::span<float> a, float b)
	{
		vector_div(a.data(), b, a.size());
	}

	// Vectorizes the division of a scalar and vector and writes the result into the provided vector -> c = a / b
	static void vector_div(float a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the division of a scalar and vector and writes the result into the provided vector -> c = a / b
	static void vector_div(float a, std::span<const float> b, std::span<float> c)
	{
		vector_div(a, b.data(), c.data(), b.size());
	}

	// Vectorizes the division of a scalar and vector and writes the result into the vector -> b = a / b
	static void vector_div(float a, float* const __restrict b, size_t n);

	// Vectorizes the division of a scalar and vector and writes the result into the vector -> b = a / b
	static void vector_div(float a, std::span<float> b)
	{
		vector_div(a, b.data(), b.size());
	}

	/* Vector exponentiation */

	// Vectorizes the exponentiation of two vectors and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the exponentiation of two vectors and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(std::span<const float> a, std::span<const float> b, std::span<float> c)
	{
		vector_pow(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the exponentiation of two vectors and writes the result into the first vector -> a = a ^ b
	static void vector_pow(float* const __restrict a, const float* const __restrict b, size_t n);

	// Vectorizes the exponentiation of two vectors and writes the result into the first vector -> a = a ^ b
	static void vector_pow(std::span<float> a, std::span<const float> b)
	{
		vector_pow(a.data(), b.data(), a.size());
	}

	// Vectorizes the exponentiation of a vector and scalar and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(const float* const __restrict a, float b, float* const __restrict c, size_t n);

	// Vectorizes the exponentiation of a vector and scalar and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(std::span<const float> a, float b, std::span<float> c)
	{
		vector_pow(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the exponentiation of a vector and scalar and writes the result into the vector -> a = a ^ b
	static void vector_pow(float* const __restrict a, float b, size_t n);

	// Vectorizes the exponentiation of a vector and scalar and writes the result into the vector -> a = a ^ b
	static void vector_pow(std::span<float> a, float b)
	{
		vector_pow(a.data(), b, a.size());
	}

	// Vectorizes the exponentiation of a scalar and vector and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(float a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the exponentiation of a scalar and vector and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(float a, std::span<const float> b, std::span<float> c)
	{
		vector_pow(a, b.data(), c.data(), b.size());
	}

	/* Vector logarithm */

	// Vectorizes the logarithm of a vector argument and base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(const float* const __restrict arg, const float* const __restrict log_base, float* const __restrict r, size_t n);

	// Vectorizes the logarithm of a vector argument and base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(std::span<const float> arg, std::span<const float> log_base, std::span<float> r)
	{
		vector_log(arg.data(), log_base.data(), r.data(), arg.size());
	}

	// Vectorizes the logarithm of a vector argument and base and writes the result into the argument vector -> arg = log_base(arg)
	static void vector_log(float* const __restrict arg, const float* const __restrict log_base, size_t n);

	// Vectorizes the logarithm of a vector argument and base and writes the result into the argument vector -> arg = log_base(arg)
	static void vector_log(std::span<float> arg, std::span<const float> log_base)
	{
		vector_log(arg.data(), log_base.data(), arg.size());
	}

	// Vectorizes the logarithm of a vector argument and scalar base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(const float* const __restrict arg, float log_base, float* const __restrict r, size_t n);

	// Vectorizes the logarithm of a vector argument and scalar base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(std::span<const float> arg, float log_base, std::span<float> r)
	{
		vector_log(arg.data(), log_base, r.data(), arg.size());
	}

	// Vectorizes the logarithm of a vector argument and scalar base and writes the result into the vector -> arg = log_base(arg)
	static void vector_log(float* const __restrict arg, float log_base, size_t n);

	// Vectorizes the logarithm of a vector argument and scalar base and writes the result into the vector -> arg = log_base(arg)
	static void vector_log(std::span<float> arg, float log_base)
	{
		vector_log(arg.data(), log_base, arg.size());
	}

	// Vectorizes the logarithm of a scalar argument and vector base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(float arg, const float* const __restrict log_base, float* const __restrict r, size_t n);

	// Vectorizes the logarithm of a scalar argument and vector base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(float arg, std::span<const float> log_base, std::span<float> r)
	{
		vector_log(arg, log_base.data(), r.data(), log_base.size());
	}

	/* Vector fused multiply addition */

	// Vectorizes the fused multiply addition of three vectors and writes the result into the provided vector -> r = a + b * c
	static void vector_fmadd(const float* const __restrict a, const float* const __restrict b, const float* const __restrict c,
		float* const __restrict r, size_t n);

	// Vectorizes the fused multiply addition of three vectors and writes the result into the provided vector -> r = a + b * c
	static void vector_fmadd(std::span<const float> a, std::span<const float> b, std::span<const float> c, std::span<float> r)
	{
		vector_fmadd(a.data(), b.data(), c.data(), r.data(), a.size());
	}

	// Vectorizes the fused multiply addition of three vectors and writes the result into the first vector -> a += b * c
	static void vector_fmadd(float* const __restrict a, const float* const __restrict b, const float* const __restrict c, size_t n);

	// Vectorizes the fused multiply addition of three vectors and writes the result into the first vector -> a += b * c
	static void vector_fmadd(std::span<float> a, std::span<const float> b, std::span<const float> c)
	{
		vector_fmadd(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the fused mutliply addition of two vectors and a scalar and writes the result into the provided vector -> r = a + b * c
	static void vector_fmadd(const float* const __restrict a, const float* const __restrict b, float c,
		float* const __restrict r, size_t n);

	// Vectorizes the fused multiply addition of two vectors and a scalar and writes the result into the provided vector -> r = a + b * c
	static void vector_fmadd(std::span<const float> a, std::span<const float> b, float c, std::span<float> r)
	{
		vector_fmadd(a.data(), b.data(), c, r.data(), a.size());
	}

	// Vectorizes the fused mutliply addition of two vectors and a scalar and writes the result into the first vector -> a += b * c
	static void vector_fmadd(float* const __restrict a, const float* const __restrict b, float c, size_t n);

	// Vectorizes the fused multiply addition of two vectors and a scalar and writes the result into the first vector -> a += b * c
	static void vector_fmadd(std::span<float> a, std::span<const float> b, float c)
	{
		vector_fmadd(a.data(), b.data(), c, a.size());
	}

	/* Vector fused negative multiply addition */

	// Vectorizes the fused negative multiply addition of three vectors and writes the result into the provided vector -> r = a - b * c
	static void vector_fnmadd(const float* const __restrict a, const float* const __restrict b, const float* const __restrict c,
		float* const __restrict r, size_t n);

	// Vectorizes the fused negative multiply addition of three vectors and writes the result into the provided vector -> r = a - b * c
	static void vector_fnmadd(std::span<const float> a, std::span<const float> b, std::span<const float> c, std::span<float> r)
	{
		vector_fnmadd(a.data(), b.data(), c.data(), r.data(), a.size());
	}

	// Vectorizes the fused negative multiply addition of three vectors and writes the result into the first vector -> a -= b * c
	static void vector_fnmadd(float* const __restrict a, const float* const __restrict b, const float* const __restrict c, size_t n);

	// Vectorizes the fused negative multiply addition of three vectors and writes the result into the first vector -> a -= b * c
	static void vector_fnmadd(std::span<float> a, std::span<const float> b, std::span<const float> c)
	{
		vector_fnmadd(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the fused negative multiply addition of two vectors and a scalar and writes the result into the provided vector -> r = a - b * c
	static void vector_fnmadd(const float* const __restrict a, const float* const __restrict b, float c,
		float* const __restrict r, size_t n);

	// Vectorizes the fused negative multiply addition of two vectors and a scalar and writes the result into the provided vector -> r = a - b * c
	static void vector_fnmadd(std::span<const float> a, std::span<const float> b, float c, std::span<float> r)
	{
		vector_fnmadd(a.data(), b.data(), c, r.data(), a.size());
	}

	// Vectorizes the fused negative multiply addition of two vectors and a scalar and writes the result into the first vector -> a -= b * c
	static void vector_fnmadd(float* const __restrict a, const float* const __restrict b, float c, size_t n);

	// Vectorizes the fused negative multiply addition of two vectors and a scalar and writes the reuslt into the first vector -> a -= b * c
	static void vector_fnmadd(std::span<float> a, std::span<const float> b, float c)
	{
		vector_fnmadd(a.data(), b.data(), c, a.size());
	}

	/* Vector square */

	// Vectorizes the square of a vector and writes the result into the provided vector -> r = a ^ 2
	static void vector_sq(const float* const __restrict a, float* const __restrict r, size_t n);

	// Vectorizes the square of a vector and writes the result into the provided vector -> r = a ^ 2
	static void vector_sq(std::span<const float> a, std::span<float> r)
	{
		vector_sq(a.data(), r.data(), a.size());
	}

	// Vectorizes the square of a vector and writes the result into the vector -> a = a ^ 2
	static void vector_sq(float* const __restrict a, size_t n);

	// Vectorizes the square of a vector and writes the result into the vector -> a = a ^ 2
	static void vector_sq(std::span<float> a)
	{
		vector_sq(a.data(), a.size());
	}

	/* Vector square root */

	// Vectorizes the square root of a vector and writes the result into the provided vector -> r = sqrt(a)
	static void vector_sqrt(const float* const __restrict a, float* const __restrict r, size_t n);

	// Vectorizes the square root of a vector and writes the result into the provided vector -> r = sqrt(a)
	static void vector_sqrt(std::span<const float> a, std::span<float> r)
	{
		vector_sqrt(a.data(), r.data(), a.size());
	}

	// Vectorizes the square root of a vector and writes the result into the vector -> a = sqrt(a)
	static void vector_sqrt(float* const __restrict a, size_t n);

	// Vectorizes the square root of a vector and writes the result into the vector -> a = sqrt(a)
	static void vector_sqrt(std::span<float> a)
	{
		vector_sqrt(a.data(), a.size());
	}

	/* Vector natural exponentiation */

	// Vectorizes the natural exponentiation of a vector and writes the result into the provided vector -> r = e ^ a
	static void vector_exp(const float* const __restrict a, float* const __restrict r, size_t n);

	// Vectorizes the natural exponentiation of a vector and writes the result into the provided vector -> r = e ^ a
	static void vector_exp(std::span<const float> a, std::span<float> r)
	{
		vector_exp(a.data(), r.data(), a.size());
	}

	// Vectorizes the natural exponentiation of a vector and writes the result into the vector -> a = e ^ a
	static void vector_exp(float* const __restrict a, size_t n);

	// Vectorizes the natural exponentiation of a vector and writes the result into the vector -> a = e ^ a
	static void vector_exp(std::span<float> a)
	{
		vector_exp(a.data(), a.size());
	}

	/* Vector natural logarithm */

	// Vectorizes the natural logarithm of a vector and writes the result into the provided vector -> r = ln(a)
	static void vector_ln(const float* const __restrict a, float* const __restrict r, size_t n);

	// Vectorizes the natural logarithm of a vector and writes the result into the provided vector -> r = ln(a)
	static void vector_ln(std::span<const float> a, std::span<float> r)
	{
		vector_ln(a.data(), r.data(), a.size());
	}

	// Vectorizes the natural logarithm of a vector and writes the result into the vector -> a = ln(a)
	static void vector_ln(float* const __restrict a, size_t n);

	// Vectorizes the natural logarithm of a vector and writes the result into the vector -> a = ln(a)
	static void vector_ln(std::span<float> a)
	{
		vector_ln(a.data(), a.size());
	}

	/* Vector operations */

	// Vectorizes the sum of a vector.
	static float vector_sum(const float* const __restrict a, size_t n);

	// Vectorizes the sum of a vector.
	static float vector_sum(std::span<const float> a)
	{
		return vector_sum(a.data(), a.size());
	}

	// Vectorizes the max of a vector.
	static float vector_max(const float* const __restrict a, size_t n);

	// Vectorizes the max of a vector.
	static float vector_max(std::span<const float> a)
	{
		return vector_max(a.data(), a.size());
	}

	// Vectorizes the min of a vector.
	static float vector_min(const float* const __restrict a, size_t n);

	// Vectorizes the min of a vector.
	static float vector_min(std::span<const float> a)
	{
		return vector_min(a.data(), a.size());
	}

	// Vectorizes the dot product of two vectors.
	static float vector_dot(const float* const __restrict a, const float* const __restrict b, size_t n);

	// Vectorizes the dot product of two vectors.
	static float vector_dot(std::span<const float> a, std::span<const float> b)
	{
		return vector_dot(a.data(), b.data(), a.size());
	}

	// Vectorizes the dot product of a subrange of two vectors.
	static float vector_dot(const float* __restrict a, const float* __restrict b, size_t a_off, size_t b_off, size_t n);

	/* Vector limiting functions */

	// Vectorizes the max of two vectors and writes the result into the provided vector -> c = max(a, b)
	static void vector_max(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the max of two vectors and writes the result into the provided vector -> c = max(a, b)
	static void vector_max(std::span<const float> a, std::span<const float> b, std::span<float> c)
	{
		vector_max(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the max of two vectors and writes the result into the first vector -> a = max(a, b)
	static void vector_max(float* const __restrict a, const float* const __restrict b, size_t n);

	// Vectorizes the max of two vectors and writes the result into the first vector -> a = max(a, b)
	static void vector_max(std::span<float> a, std::span<const float> b)
	{
		vector_max(a.data(), b.data(), a.size());
	}

	// Vectorizes the max of a vector and scalar and writes the result into the provided vector -> c = max(a, b)
	static void vector_max(const float* const __restrict a, float b, float* const __restrict c, size_t n);

	// Vectorizes the max of a vector and scalar and writes the result into the provided vector -> c = max(a, b)
	static void vector_max(std::span<const float> a, float b, std::span<float> c)
	{
		vector_max(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the max of a vector and scalar and writes the result into the vector -> a = max(a, b)
	static void vector_max(float* const __restrict a, float b, size_t n);

	// Vectorizes the max of a vector and scalar and writes the result into the vector -> a = max(a, b)
	static void vector_max(std::span<float> a, float b)
	{
		vector_max(a.data(), b, a.size());
	}

	// Vectorizes the min of two vectors and writes the result into the provided vector -> c = min(a, b)
	static void vector_min(const float* const __restrict a, const float* const __restrict b, float* const __restrict c, size_t n);

	// Vectorizes the min of two vectors and writes the result into the provided vector -> c = min(a, b)
	static void vector_min(std::span<const float> a, std::span<const float> b, std::span<float> c)
	{
		vector_min(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the min of two vectors and writes the result into the first vector -> a = min(a, b)
	static void vector_min(float* const __restrict a, const float* const __restrict b, size_t n);

	// Vectorizes the min of two vectors and writes the result into the first vector -> a = min(a, b)
	static void vector_min(std::span<float> a, std::span<const float> b)
	{
		vector_min(a.data(), b.data(), a.size());
	}

	// Vectorizes the min of a vector and scalar and writes the result into the provided vector -> c = min(a, b)
	static void vector_min(const float* const __restrict a, float b, float* const __restrict c, size_t n);

	// Vectorizes the min of a vector and scalar and writes the result into the provided vector -> c = min(a, b)
	static void vector_min(std::span<const float> a, float b, std::span<float> c)
	{
		vector_min(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the min of a vector and scalar and writes the result into the vector -> a = min(a, b)
	static void vector_min(float* const __restrict a, float b, size_t n);

	// Vectorizes the min of a vector and scalar and writes the result into the vector -> a = min(a, b)
	static void vector_min(std::span<float> a, float b)
	{
		vector_min(a.data(), b, a.size());
	}

	// Vectorizes the clamp of a vector and two limit vectors and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(const float* const __restrict a, const float* const __restrict min, const float* const __restrict max,
		float* const __restrict r, size_t n);

	// Vectorizes the clamp of a vector and two limit vectors and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(std::span<const float> a, std::span<const float> min, std::span<const float> max, std::span<float> r)
	{
		vector_clamp(a.data(), min.data(), max.data(), r.data(), a.size());
	}

	// Vectorizes the clamp of a vector and two limit vectors and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(float* const __restrict a, const float* const __restrict min, const float* const __restrict max, size_t n);

	// Vectorizes the clamp of a vector and two limit vectors and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(std::span<float> a, std::span<const float> min, std::span<const float> max)
	{
		vector_clamp(a.data(), min.data(), max.data(), a.size());
	}

	// Vectorizes the clamp of a vector and limit scalar and vector and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(const float* const __restrict a, float min, const float* const __restrict max,
		float* const __restrict r, size_t n);

	// Vectorizes the clamp of a vector and limit scalar and vector and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(std::span<const float> a, float min, std::span<const float> max, std::span<float> r)
	{
		vector_clamp(a.data(), min, max.data(), r.data(), a.size());
	}

	// Vectorizes the clamp of a vector and limit scalar and vector and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(float* const __restrict a, float min, const float* const __restrict max, size_t n);

	// Vectorizes the clamp of a vector and limit scalar and vector and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(std::span<float> a, float min, std::span<const float> max)
	{
		vector_clamp(a.data(), min, max.data(), a.size());
	}

	// Vectorizes the clamp of a vector and limit vector and scalar and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(const float* const __restrict a, const float* const __restrict min, float max,
		float* const __restrict r, size_t n);

	// Vectorizes the clamp of a vector and limit vector and scalar and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(std::span<const float> a, std::span<const float> min, float max, std::span<float> r)
	{
		vector_clamp(a.data(), min.data(), max, r.data(), a.size());
	}

	// Vectorizes the clamp of a vector and limit vector and scalar and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(float* const __restrict a, const float* const __restrict min, float max, size_t n);

	// Vectorizes the clamp of a vector and limit vector and scalar and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(std::span<float> a, std::span<const float> min, float max)
	{
		vector_clamp(a.data(), min.data(), max, a.size());
	}

	// Vectorizes the clamp of a vector and two limit scalars and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(const float* const __restrict a, float min, float max, float* const __restrict r, size_t n);

	// Vectorizes the clamp of a vector and two limit scalars and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(std::span<const float> a, float min, float max, std::span<float> r)
	{
		vector_clamp(a.data(), min, max, r.data(), a.size());
	}

	// Vectorizes the clamp of a vector and two limit scalars and writes the result into the vector -> a = clamp(a, min, max)
	static void vector_clamp(float* const __restrict a, float min, float max, size_t n);

	// Vectorizes the clamp of a vector and two limit scalars and writes the result into the vector -> a = clamp(a, min, max)
	static void vector_clamp(std::span<float> a, float min, float max)
	{
		vector_clamp(a.data(), min, max, a.size());
	}

	/* Vector activation functions */

	// Vectorizes the sigmoid function applied to a vector and writes the result into the provided vector -> r = sigmoid(a)
	static void vector_sigmoid(const float* const __restrict a, float* const __restrict r, size_t n);

	// Vectorizes the sigmoid function applied to a vector and writes the result into the provided vector -> r = sigmoid(a)
	static void vector_sigmoid(std::span<const float> a, std::span<float> r)
	{
		vector_sigmoid(a.data(), r.data(), a.size());
	}

	// Vectorizes the sigmoid function applied to a vector and writes the result into the vector -> a = sigmoid(a)
	static void vector_sigmoid(float* const __restrict a, size_t n);

	// Vectorizes the sigmoid function applied to a vector and writes the result into the vector -> a = sigmoid(a)
	static void vector_sigmoid(std::span<float> a)
	{
		vector_sigmoid(a.data(), a.size());
	}

	// Vectorizes the hyperbolic tangent function applied to a vector and writes the result into the provided vector -> r = tanh(a)
	static void vector_tanh(const float* const __restrict a, float* const __restrict r, size_t n);

	// Vectorizes the hyperbolic tangent function applied to a vector and writes the result into the provided vector -> r = tanh(a)
	static void vector_tanh(std::span<const float> a, std::span<float> r)
	{
		vector_tanh(a.data(), r.data(), a.size());
	}

	// Vectorizes the hyperbolic tangent function applied to a vector and writes the result into the vector -> a = tanh(a)
	static void vector_tanh(float* const __restrict a, size_t n);

	// Vectorizes the hyperbolic tangent function applied to a vector and writes the result into the vector -> a = tanh(a)
	static void vector_tanh(std::span<float> a)
	{
		vector_tanh(a.data(), a.size());
	}

	/* Matrix operations */

	// Computes the matrix multiplication of two vectors and writes the result into the provided vector -> r = a @ b_t
	static void matmul_raw(const float* __restrict a, const float* __restrict b_t, float* __restrict r, size_t batch_count,
		size_t m, size_t n, size_t p, size_t a_batch_stride, size_t b_t_batch_stride, size_t r_batch_stride, size_t a_off, size_t b_t_off,
		size_t r_off, bool use_parallel, bool accumulate);

	// Computes the matrix multiplication of two vectors and accumulates the reuslt into the provided vector.
	static void matmul_reduce_raw(const float* __restrict a_t, const float* __restrict b_t, float* __restrict r, size_t batch_count,
		size_t m, size_t n, size_t p, size_t a_t_batch_stride, size_t b_t_batch_stride, size_t a_t_off, size_t b_t_off, size_t r_off,
		bool use_parallel, bool accumulate);

	// Transposes the given matrix data vector and writes the result into the given vector.
	static void transpose_matrix(const float* __restrict src, float* __restrict dst, size_t src_off, size_t dst_off,
		size_t rows, size_t cols);

	// Computes the base_input position for im2col and col2im functions.
	static size_t compute_output_position(size_t b, size_t op, const ConvGeometry& g);

	// Performs the im2col transformation on a matrix vector and writes the result into the provided vector.
	static void im2col(const float* __restrict input, const ConvGeometry& g, float* __restrict input_col, bool use_parallel);

	// Performs the col2im transformation on a matrix vector and writes the result into the provided vector.
	static void col2im(const float* __restrict d_input_col, const ConvGeometry& g, float* __restrict d_input, bool use_parallel);

	// Transforms a kernels tensor vector into the layout required for matmul in convolution and writes the result into the provided vector.
	static void kernels2matmul(const float* __restrict kernels, const ConvGeometry& g, float* __restrict kernels_mat);

	// Inverts the kernels2matmul transformation of a kernels tensor vector and writes the result into the provided vector.
	static void matmul2kernels(const float* __restrict kernels_mat, const ConvGeometry& g, float* __restrict kernels, bool accumulate);
};