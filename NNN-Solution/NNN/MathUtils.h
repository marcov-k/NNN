#pragma once

#include <immintrin.h>
#include <limits>
#include <random>
#include <span>

// Collection of various math and vectorization utility functions.
class MathUtils
{
public:
	MathUtils() = delete;

	/* Random number generation */

	// Generates a random double in the given range using a uniform real distribution.
	static double get_random_double(double min = 0.0, double max = 1.0)
	{
		thread_local std::random_device rd;
		thread_local std::mt19937 gen(rd());

		std::uniform_real_distribution dis(min, max);

		return dis(gen);
	}

	// Generates a random double from a normal distribution with the given mean and standard deviation.
	static double next_gaussian(double mean = 0.0, double std_dev = 1.0)
	{
		thread_local std::random_device rd;
		thread_local std::mt19937 gen(rd());

		std::normal_distribution dis(mean, std_dev);

		return dis(gen);
	}

	/* Register operations */

	// Computes the sum of a 256-bit register of doubles.
	static double sum_m256d(__m256d v);

	// Computes the max of a 256-bit register of doubles.
	static double max_m256d(__m256d v);

	/* Vector addition */

	// Vectorizes the addition of two vectors and writes the result into the provided vector -> c = a + b
	static void vector_add(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the addition of two vectors and writes the result into the provided vector -> c = a + b
	static void vector_add(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_add(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the addition of two vectors and writes the result into the first vector -> a += b
	static void vector_add(double* const __restrict a, const double* const __restrict b, size_t n);

	// Vectorizes the addition of two vectors and writes the result into the first vector -> a += b
	static void vector_add(std::span<double> a, std::span<const double> b)
	{
		vector_add(a.data(), b.data(), a.size());
	}

	// Vectorizes the addition of a vector and scalar and writes the result into the provided vector -> c = a + b
	static void vector_add(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	// Vectorizes the addition of a vector and scalar and writes the result into the provided vector -> c = a + b
	static void vector_add(std::span<const double> a, double b, std::span<double> c)
	{
		vector_add(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the addition of a vector and scalar and writes the result into the vector -> a += b
	static void vector_add(double* const __restrict a, double b, size_t n);

	// Vectorizes the addition of a vector and scalar and writes the result into the vector -> a += b
	static void vector_add(std::span<double> a, double b)
	{
		vector_add(a.data(), b, a.size());
	}

	/* Vector subtraction */

	// Vectorizes the subtraction of two vectors and writes the result into the provided vector -> c = a - b
	static void vector_sub(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the subtraction of two vectors and writes the result into the provided vector -> c = a - b
	static void vector_sub(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_sub(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the subtraction of two vectors and writes the result into the first vector -> a -= b
	static void vector_sub(double* const __restrict a, const double* const __restrict b, size_t n);

	// Vectorizes the subtraction of two vectors and writes the result into the first vector -> a -= b
	static void vector_sub(std::span<double> a, std::span<const double> b)
	{
		vector_sub(a.data(), b.data(), a.size());
	}

	// Vectorizes the subtraction of a vector and scalar and writes the result into the provided vector -> c = a - b
	static void vector_sub(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	// Vectorizes the subtraction of a vector and scalar and writes the result into the provided vector -> c = a - b
	static void vector_sub(std::span<const double> a, double b, std::span<double> c)
	{
		vector_sub(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the subtraction of a vector and scalar and writes the result into the vector -> a -= b
	static void vector_sub(double* const __restrict a, double b, size_t n);

	// Vectorizes the subtraction of a vector and scalar and writes the result into the vector -> a -= b
	static void vector_sub(std::span<double> a, double b)
	{
		vector_sub(a.data(), b, a.size());
	}

	// Vectorizes the subtraction of a scalar and vector and writes the result into the provided vector -> c = a - b
	static void vector_sub(double a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the subtraction of a scalar and vector and writes the result into the provided vector -> c = a - b
	static void vector_sub(double a, std::span<const double> b, std::span<double> c)
	{
		vector_sub(a, b.data(), c.data(), b.size());
	}

	// Vectorizes the subtraction of a scalar and vector and writes the result into the vector -> b = a - b
	static void vector_sub(double a, double* const __restrict b, size_t n);

	// Vectorizes the subtraction of a scalar and vector and writes the result into the vector -> b = a - b
	static void vector_sub(double a, std::span<double> b)
	{
		vector_sub(a, b.data(), b.size());
	}

	/* Vector multiplication */

	// Vectorizes the multiplication of two vectors and writes the result into the provided vector -> c = a * b
	static void vector_mul(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the multiplication of two vectors and writes the result into the provided vector -> c = a * b
	static void vector_mul(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_mul(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the multiplication of two vectors and writes the result into the first vector -> a *= b
	static void vector_mul(double* const __restrict a, const double* const __restrict b, size_t n);

	// Vectorizes the multiplication of two vectors and writes the result into the first vector -> a *= b
	static void vector_mul(std::span<double> a, std::span<const double> b)
	{
		vector_mul(a.data(), b.data(), a.size());
	}

	// Vectorizes the multiplication of a vector and scalar and writes the result into the provided vector -> c = a * b
	static void vector_mul(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	// Vectorizes the multiplication of a vector and scalar and writes the result into the provided vector -> c = a * b
	static void vector_mul(std::span<const double> a, double b, std::span<double> c)
	{
		vector_mul(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the multiplication of a vector and scalar and writes the result into the vector -> a *= b
	static void vector_mul(double* const __restrict a, double b, size_t n);

	// Vectorizes the multiplication of a vector and scalar and writes the result into the vector -> a *= b
	static void vector_mul(std::span<double> a, double b)
	{
		vector_mul(a.data(), b, a.size());
	}

	/* Vector division */

	// Vectorizes the division of two vectors and writes the result into the provided vector -> c = a / b
	static void vector_div(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the division of two vectors and writes the result into the provided vector -> c = a / b
	static void vector_div(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_div(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the division of two vectors and writes the result into the first vector -> a /= b
	static void vector_div(double* const __restrict a, const double* const __restrict b, size_t n);

	// Vectorizes the division of two vectors and writes the result into the first vector -> a /= b
	static void vector_div(std::span<double> a, std::span<const double> b)
	{
		vector_div(a.data(), b.data(), a.size());
	}

	// Vectorizes the division of a vector and scalar and writes the result into the provided vector -> c = a / b
	static void vector_div(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	// Vectorizes the division of a vector and scalar and writes the result into the provided vector -> c = a / b
	static void vector_div(std::span<const double> a, double b, std::span<double> c)
	{
		vector_div(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the division of a vector and scalar and writes the result into the vector -> a /= b
	static void vector_div(double* const __restrict a, double b, size_t n);

	// Vectorizes the division of a vector and scalar and writes the result into the vector -> a /= b
	static void vector_div(std::span<double> a, double b)
	{
		vector_div(a.data(), b, a.size());
	}

	// Vectorizes the division of a scalar and vector and writes the result into the provided vector -> c = a / b
	static void vector_div(double a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the division of a scalar and vector and writes the result into the provided vector -> c = a / b
	static void vector_div(double a, std::span<const double> b, std::span<double> c)
	{
		vector_div(a, b.data(), c.data(), b.size());
	}

	// Vectorizes the division of a scalar and vector and writes the result into the vector -> b = a / b
	static void vector_div(double a, double* const __restrict b, size_t n);

	// Vectorizes the division of a scalar and vector and writes the result into the vector -> b = a / b
	static void vector_div(double a, std::span<double> b)
	{
		vector_div(a, b.data(), b.size());
	}

	/* Vector exponentiation */

	// Vectorizes the exponentiation of two vectors and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the exponentiation of two vectors and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_pow(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the exponentiation of two vectors and writes the result into the first vector -> a = a ^ b
	static void vector_pow(double* const __restrict a, const double* const __restrict b, size_t n);

	// Vectorizes the exponentiation of two vectors and writes the result into the first vector -> a = a ^ b
	static void vector_pow(std::span<double> a, std::span<const double> b)
	{
		vector_pow(a.data(), b.data(), a.size());
	}

	// Vectorizes the exponentiation of a vector and scalar and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	// Vectorizes the exponentiation of a vector and scalar and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(std::span<const double> a, double b, std::span<double> c)
	{
		vector_pow(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the exponentiation of a vector and scalar and writes the result into the vector -> a = a ^ b
	static void vector_pow(double* const __restrict a, double b, size_t n);

	// Vectorizes the exponentiation of a vector and scalar and writes the result into the vector -> a = a ^ b
	static void vector_pow(std::span<double> a, double b)
	{
		vector_pow(a.data(), b, a.size());
	}

	// Vectorizes the exponentiation of a scalar and vector and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(double a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the exponentiation of a scalar and vector and writes the result into the provided vector -> c = a ^ b
	static void vector_pow(double a, std::span<const double> b, std::span<double> c)
	{
		vector_pow(a, b.data(), c.data(), b.size());
	}

	/* Vector logarithm */

	// Vectorizes the logarithm of a vector argument and base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(const double* const __restrict arg, const double* const __restrict log_base, double* const __restrict r, size_t n);

	// Vectorizes the logarithm of a vector argument and base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(std::span<const double> arg, std::span<const double> log_base, std::span<double> r)
	{
		vector_log(arg.data(), log_base.data(), r.data(), arg.size());
	}

	// Vectorizes the logarithm of a vector argument and base and writes the result into the argument vector -> arg = log_base(arg)
	static void vector_log(double* const __restrict arg, const double* const __restrict log_base, size_t n);

	// Vectorizes the logarithm of a vector argument and base and writes the result into the argument vector -> arg = log_base(arg)
	static void vector_log(std::span<double> arg, std::span<const double> log_base)
	{
		vector_log(arg.data(), log_base.data(), arg.size());
	}

	// Vectorizes the logarithm of a vector argument and scalar base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(const double* const __restrict arg, double log_base, double* const __restrict r, size_t n);

	// Vectorizes the logarithm of a vector argument and scalar base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(std::span<const double> arg, double log_base, std::span<double> r)
	{
		vector_log(arg.data(), log_base, r.data(), arg.size());
	}

	// Vectorizes the logarithm of a vector argument and scalar base and writes the result into the vector -> arg = log_base(arg)
	static void vector_log(double* const __restrict arg, double log_base, size_t n);

	// Vectorizes the logarithm of a vector argument and scalar base and writes the result into the vector -> arg = log_base(arg)
	static void vector_log(std::span<double> arg, double log_base)
	{
		vector_log(arg.data(), log_base, arg.size());
	}

	// Vectorizes the logarithm of a scalar argument and vector base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(double arg, const double* const __restrict log_base, double* const __restrict r, size_t n);

	// Vectorizes the logarithm of a scalar argument and vector base and writes the result into the provided vector -> r = log_base(arg)
	static void vector_log(double arg, std::span<const double> log_base, std::span<double> r)
	{
		vector_log(arg, log_base.data(), r.data(), log_base.size());
	}

	/* Vector fused multiply addition */

	// Vectorizes the fused multiply addition of three vectors and writes the result into the provided vector -> r = a + b * c
	static void vector_fmadd(const double* const __restrict a, const double* const __restrict b, const double* const __restrict c,
		double* const __restrict r, size_t n);

	// Vectorizes the fused multiply addition of three vectors and writes the result into the provided vector -> r = a + b * c
	static void vector_fmadd(std::span<const double> a, std::span<const double> b, std::span<const double> c, std::span<double> r)
	{
		vector_fmadd(a.data(), b.data(), c.data(), r.data(), a.size());
	}

	// Vectorizes the fused multiply addition of three vectors and writes the result into the first vector -> a += b * c
	static void vector_fmadd(double* const __restrict a, const double* const __restrict b, const double* const __restrict c, size_t n);

	// Vectorizes the fused multiply addition of three vectors and writes the result into the first vector -> a += b * c
	static void vector_fmadd(std::span<double> a, std::span<const double> b, std::span<const double> c)
	{
		vector_fmadd(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the fused mutliply addition of two vectors and a scalar and writes the result into the provided vector -> r = a + b * c
	static void vector_fmadd(const double* const __restrict a, const double* const __restrict b, double c,
		double* const __restrict r, size_t n);

	// Vectorizes the fused multiply addition of two vectors and a scalar and writes the result into the provided vector -> r = a + b * c
	static void vector_fmadd(std::span<const double> a, std::span<const double> b, double c, std::span<double> r)
	{
		vector_fmadd(a.data(), b.data(), c, r.data(), a.size());
	}

	// Vectorizes the fused mutliply addition of two vectors and a scalar and writes the result into the first vector -> a += b * c
	static void vector_fmadd(double* const __restrict a, const double* const __restrict b, double c, size_t n);

	// Vectorizes the fused multiply addition of two vectors and a scalar and writes the result into the first vector -> a += b * c
	static void vector_fmadd(std::span<double> a, std::span<const double> b, double c)
	{
		vector_fmadd(a.data(), b.data(), c, a.size());
	}

	/* Vector fused negative multiply addition */

	// Vectorizes the fused negative multiply addition of three vectors and writes the result into the provided vector -> r = a - b * c
	static void vector_fnmadd(const double* const __restrict a, const double* const __restrict b, const double* const __restrict c,
		double* const __restrict r, size_t n);

	// Vectorizes the fused negative multiply addition of three vectors and writes the result into the provided vector -> r = a - b * c
	static void vector_fnmadd(std::span<const double> a, std::span<const double> b, std::span<const double> c, std::span<double> r)
	{
		vector_fnmadd(a.data(), b.data(), c.data(), r.data(), a.size());
	}

	// Vectorizes the fused negative multiply addition of three vectors and writes the result into the first vector -> a -= b * c
	static void vector_fnmadd(double* const __restrict a, const double* const __restrict b, const double* const __restrict c, size_t n);

	// Vectorizes the fused negative multiply addition of three vectors and writes the result into the first vector -> a -= b * c
	static void vector_fnmadd(std::span<double> a, std::span<const double> b, std::span<const double> c)
	{
		vector_fnmadd(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the fused negative multiply addition of two vectors and a scalar and writes the result into the provided vector -> r = a - b * c
	static void vector_fnmadd(const double* const __restrict a, const double* const __restrict b, double c,
		double* const __restrict r, size_t n);

	// Vectorizes the fused negative multiply addition of two vectors and a scalar and writes the result into the provided vector -> r = a - b * c
	static void vector_fnmadd(std::span<const double> a, std::span<const double> b, double c, std::span<double> r)
	{
		vector_fnmadd(a.data(), b.data(), c, r.data(), a.size());
	}

	// Vectorizes the fused negative multiply addition of two vectors and a scalar and writes the result into the first vector -> a -= b * c
	static void vector_fnmadd(double* const __restrict a, const double* const __restrict b, double c, size_t n);

	// Vectorizes the fused negative multiply addition of two vectors and a scalar and writes the reuslt into the first vector -> a -= b * c
	static void vector_fnmadd(std::span<double> a, std::span<const double> b, double c)
	{
		vector_fnmadd(a.data(), b.data(), c, a.size());
	}

	/* Vector square */

	// Vectorizes the square of a vector and writes the result into the provided vector -> r = a ^ 2
	static void vector_sq(const double* const __restrict a, double* const __restrict r, size_t n);

	// Vectorizes the square of a vector and writes the result into the provided vector -> r = a ^ 2
	static void vector_sq(std::span<const double> a, std::span<double> r)
	{
		vector_sq(a.data(), r.data(), a.size());
	}

	// Vectorizes the square of a vector and writes the result into the vector -> a = a ^ 2
	static void vector_sq(double* const __restrict a, size_t n);

	// Vectorizes the square of a vector and writes the result into the vector -> a = a ^ 2
	static void vector_sq(std::span<double> a)
	{
		vector_sq(a.data(), a.size());
	}

	/* Vector square root */

	// Vectorizes the square root of a vector and writes the result into the provided vector -> r = sqrt(a)
	static void vector_sqrt(const double* const __restrict a, double* const __restrict r, size_t n);

	// Vectorizes the square root of a vector and writes the result into the provided vector -> r = sqrt(a)
	static void vector_sqrt(std::span<const double> a, std::span<double> r)
	{
		vector_sqrt(a.data(), r.data(), a.size());
	}

	// Vectorizes the square root of a vector and writes the result into the vector -> a = sqrt(a)
	static void vector_sqrt(double* const __restrict a, size_t n);

	// Vectorizes the square root of a vector and writes the result into the vector -> a = sqrt(a)
	static void vector_sqrt(std::span<double> a)
	{
		vector_sqrt(a.data(), a.size());
	}

	/* Vector natural exponentiation */

	// Vectorizes the natural exponentiation of a vector and writes the result into the provided vector -> r = e ^ a
	static void vector_exp(const double* const __restrict a, double* const __restrict r, size_t n);

	// Vectorizes the natural exponentiation of a vector and writes the result into the provided vector -> r = e ^ a
	static void vector_exp(std::span<const double> a, std::span<double> r)
	{
		vector_exp(a.data(), r.data(), a.size());
	}

	// Vectorizes the natural exponentiation of a vector and writes the result into the vector -> a = e ^ a
	static void vector_exp(double* const __restrict a, size_t n);

	// Vectorizes the natural exponentiation of a vector and writes the result into the vector -> a = e ^ a
	static void vector_exp(std::span<double> a)
	{
		vector_exp(a.data(), a.size());
	}

	/* Vector natural logarithm */

	// Vectorizes the natural logarithm of a vector and writes the result into the provided vector -> r = ln(a)
	static void vector_ln(const double* const __restrict a, double* const __restrict r, size_t n);

	// Vectorizes the natural logarithm of a vector and writes the result into the provided vector -> r = ln(a)
	static void vector_ln(std::span<const double> a, std::span<double> r)
	{
		vector_ln(a.data(), r.data(), a.size());
	}

	// Vectorizes the natural logarithm of a vector and writes the result into the vector -> a = ln(a)
	static void vector_ln(double* const __restrict a, size_t n);

	// Vectorizes the natural logarithm of a vector and writes the result into the vector -> a = ln(a)
	static void vector_ln(std::span<double> a)
	{
		vector_ln(a.data(), a.size());
	}

	/* Vector operations */

	// Vectorizes the sum of a vector.
	static double vector_sum(const double* const __restrict a, size_t n);

	// Vectorizes the sum of a vector.
	static double vector_sum(std::span<const double> a)
	{
		return vector_sum(a.data(), a.size());
	}

	// Vectorizes the max of a vector.
	static double vector_max(const double* const __restrict a, size_t n);

	// Vectorizes the max of a vector.
	static double vector_max(std::span<const double> a)
	{
		return vector_max(a.data(), a.size());
	}

	// Vectorizes the dot product of two vectors.
	static double vector_dot(const double* const __restrict a, const double* const __restrict b, size_t n);

	// Vectorizes the dot product of two vectors.
	static double vector_dot(std::span<const double> a, std::span<const double> b)
	{
		return vector_dot(a.data(), b.data(), a.size());
	}

	// Vectorizes the dot product of a subrange of two vectors.
	static double vector_dot(const double* __restrict a, const double* __restrict b, int a_off, int b_off, int n);

	/* Vector limiting functions */

	// Vectorizes the max of two vectors and writes the result into the provided vector -> c = max(a, b)
	static void vector_max(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the max of two vectors and writes the result into the provided vector -> c = max(a, b)
	static void vector_max(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_max(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the max of two vectors and writes the result into the first vector -> a = max(a, b)
	static void vector_max(double* const __restrict a, const double* const __restrict b, size_t n);

	// Vectorizes the max of two vectors and writes the result into the first vector -> a = max(a, b)
	static void vector_max(std::span<double> a, std::span<const double> b)
	{
		vector_max(a.data(), b.data(), a.size());
	}

	// Vectorizes the max of a vector and scalar and writes the result into the provided vector -> c = max(a, b)
	static void vector_max(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	// Vectorizes the max of a vector and scalar and writes the result into the provided vector -> c = max(a, b)
	static void vector_max(std::span<const double> a, double b, std::span<double> c)
	{
		vector_max(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the max of a vector and scalar and writes the result into the vector -> a = max(a, b)
	static void vector_max(double* const __restrict a, double b, size_t n);

	// Vectorizes the max of a vector and scalar and writes the result into the vector -> a = max(a, b)
	static void vector_max(std::span<double> a, double b)
	{
		vector_max(a.data(), b, a.size());
	}

	// Vectorizes the min of two vectors and writes the result into the provided vector -> c = min(a, b)
	static void vector_min(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	// Vectorizes the min of two vectors and writes the result into the provided vector -> c = min(a, b)
	static void vector_min(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_min(a.data(), b.data(), c.data(), a.size());
	}

	// Vectorizes the min of two vectors and writes the result into the first vector -> a = min(a, b)
	static void vector_min(double* const __restrict a, const double* const __restrict b, size_t n);

	// Vectorizes the min of two vectors and writes the result into the first vector -> a = min(a, b)
	static void vector_min(std::span<double> a, std::span<const double> b)
	{
		vector_min(a.data(), b.data(), a.size());
	}

	// Vectorizes the min of a vector and scalar and writes the result into the provided vector -> c = min(a, b)
	static void vector_min(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	// Vectorizes the min of a vector and scalar and writes the result into the provided vector -> c = min(a, b)
	static void vector_min(std::span<const double> a, double b, std::span<double> c)
	{
		vector_min(a.data(), b, c.data(), a.size());
	}

	// Vectorizes the min of a vector and scalar and writes the result into the vector -> a = min(a, b)
	static void vector_min(double* const __restrict a, double b, size_t n);

	// Vectorizes the min of a vector and scalar and writes the result into the vector -> a = min(a, b)
	static void vector_min(std::span<double> a, double b)
	{
		vector_min(a.data(), b, a.size());
	}

	// Vectorizes the clamp of a vector and two limit vectors and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(const double* const __restrict a, const double* const __restrict min, const double* const __restrict max,
		double* const __restrict r, size_t n);

	// Vectorizes the clamp of a vector and two limit vectors and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(std::span<const double> a, std::span<const double> min, std::span<const double> max, std::span<double> r)
	{
		vector_clamp(a.data(), min.data(), max.data(), r.data(), a.size());
	}

	// Vectorizes the clamp of a vector and two limit vectors and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(double* const __restrict a, const double* const __restrict min, const double* const __restrict max, size_t n);

	// Vectorizes the clamp of a vector and two limit vectors and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(std::span<double> a, std::span<const double> min, std::span<const double> max)
	{
		vector_clamp(a.data(), min.data(), max.data(), a.size());
	}

	// Vectorizes the clamp of a vector and limit scalar and vector and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(const double* const __restrict a, double min, const double* const __restrict max,
		double* const __restrict r, size_t n);

	// Vectorizes the clamp of a vector and limit scalar and vector and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(std::span<const double> a, double min, std::span<const double> max, std::span<double> r)
	{
		vector_clamp(a.data(), min, max.data(), r.data(), a.size());
	}

	// Vectorizes the clamp of a vector and limit scalar and vector and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(double* const __restrict a, double min, const double* const __restrict max, size_t n);

	// Vectorizes the clamp of a vector and limit scalar and vector and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(std::span<double> a, double min, std::span<const double> max)
	{
		vector_clamp(a.data(), min, max.data(), a.size());
	}

	// Vectorizes the clamp of a vector and limit vector and scalar and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(const double* const __restrict a, const double* const __restrict min, double max,
		double* const __restrict r, size_t n);

	// Vectorizes the clamp of a vector and limit vector and scalar and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(std::span<const double> a, std::span<const double> min, double max, std::span<double> r)
	{
		vector_clamp(a.data(), min.data(), max, r.data(), a.size());
	}

	// Vectorizes the clamp of a vector and limit vector and scalar and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(double* const __restrict a, const double* const __restrict min, double max, size_t n);

	// Vectorizes the clamp of a vector and limit vector and scalar and writes the result into the first vector -> a = clamp(a, min, max)
	static void vector_clamp(std::span<double> a, std::span<const double> min, double max)
	{
		vector_clamp(a.data(), min.data(), max, a.size());
	}

	// Vectorizes the clamp of a vector and two limit scalars and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(const double* const __restrict a, double min, double max, double* const __restrict r, size_t n);

	// Vectorizes the clamp of a vector and two limit scalars and writes the result into the provided vector -> r = clamp(a, min, max)
	static void vector_clamp(std::span<const double> a, double min, double max, std::span<double> r)
	{
		vector_clamp(a.data(), min, max, r.data(), a.size());
	}

	// Vectorizes the clamp of a vector and two limit scalars and writes the result into the vector -> a = clamp(a, min, max)
	static void vector_clamp(double* const __restrict a, double min, double max, size_t n);

	// Vectorizes the clamp of a vector and two limit scalars and writes the result into the vector -> a = clamp(a, min, max)
	static void vector_clamp(std::span<double> a, double min, double max)
	{
		vector_clamp(a.data(), min, max, a.size());
	}

	/* Vector activation functions */

	// Vectorizes the sigmoid function applied to a vector and writes the result into the provided vector -> r = sigmoid(a)
	static void vector_sigmoid(const double* const __restrict a, double* const __restrict r, size_t n);

	// Vectorizes the sigmoid function applied to a vector and writes the result into the provided vector -> r = sigmoid(a)
	static void vector_sigmoid(std::span<const double> a, std::span<double> r)
	{
		vector_sigmoid(a.data(), r.data(), a.size());
	}

	// Vectorizes the sigmoid function applied to a vector and writes the result into the vector -> a = sigmoid(a)
	static void vector_sigmoid(double* const __restrict a, size_t n);

	// Vectorizes the sigmoid function applied to a vector and writes the result into the vector -> a = sigmoid(a)
	static void vector_sigmoid(std::span<double> a)
	{
		vector_sigmoid(a.data(), a.size());
	}

	// Vectorizes the hyperbolic tangent function applied to a vector and writes the result into the provided vector -> r = tanh(a)
	static void vector_tanh(const double* const __restrict a, double* const __restrict r, size_t n);

	// Vectorizes the hyperbolic tangent function applied to a vector and writes the result into the provided vector -> r = tanh(a)
	static void vector_tanh(std::span<const double> a, std::span<double> r)
	{
		vector_tanh(a.data(), r.data(), a.size());
	}

	// Vectorizes the hyperbolic tangent function applied to a vector and writes the result into the vector -> a = tanh(a)
	static void vector_tanh(double* const __restrict a, size_t n);

	// Vectorizes the hyperbolic tangent function applied to a vector and writes the result into the vector -> a = tanh(a)
	static void vector_tanh(std::span<double> a)
	{
		vector_tanh(a.data(), a.size());
	}
};