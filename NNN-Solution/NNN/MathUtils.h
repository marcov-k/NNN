#pragma once

#include <immintrin.h>
#include <limits>
#include <random>
#include <span>

class MathUtils
{
public:
	MathUtils() = delete;

	// Random number generation

	static double get_random_double(double min = 0.0, double max = 1.0)
	{
		thread_local std::random_device rd;
		thread_local std::mt19937 gen(rd());

		std::uniform_real_distribution dis(min, max);

		return dis(gen);
	}

	static double next_gaussian(double mean = 0.0, double std_dev = 1.0)
	{
		thread_local std::random_device rd;
		thread_local std::mt19937 gen(rd());

		std::normal_distribution dis(mean, std_dev);

		return dis(gen);
	}

	// Register operations

	static double sum_m256d(__m256d v);

	static double max_m256d(__m256d v);

	// Vector addition

	static void vector_add(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	static void vector_add(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_add(a.data(), b.data(), c.data(), a.size());
	}

	static void vector_add(double* const __restrict a, const double* const __restrict b, size_t n);

	static void vector_add(std::span<double> a, std::span<const double> b)
	{
		vector_add(a.data(), b.data(), a.size());
	}

	static void vector_add(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	static void vector_add(std::span<const double> a, double b, std::span<double> c)
	{
		vector_add(a.data(), b, c.data(), a.size());
	}

	static void vector_add(double* const __restrict a, double b, size_t n);

	static void vector_add(std::span<double> a, double b)
	{
		vector_add(a.data(), b, a.size());
	}

	// Vector subtraction

	static void vector_sub(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	static void vector_sub(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_sub(a.data(), b.data(), c.data(), a.size());
	}

	static void vector_sub(double* const __restrict a, const double* const __restrict b, size_t n);

	static void vector_sub(std::span<double> a, std::span<const double> b)
	{
		vector_sub(a.data(), b.data(), a.size());
	}

	static void vector_sub(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	static void vector_sub(std::span<const double> a, double b, std::span<double> c)
	{
		vector_sub(a.data(), b, c.data(), a.size());
	}

	static void vector_sub(double* const __restrict a, double b, size_t n);

	static void vector_sub(std::span<double> a, double b)
	{
		vector_sub(a.data(), b, a.size());
	}

	static void vector_sub(double a, const double* const __restrict b, double* const __restrict c, size_t n);

	static void vector_sub(double a, std::span<const double> b, std::span<double> c)
	{
		vector_sub(a, b.data(), c.data(), b.size());
	}

	// Vector multiplication

	static void vector_mul(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	static void vector_mul(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_mul(a.data(), b.data(), c.data(), a.size());
	}

	static void vector_mul(double* const __restrict a, const double* const __restrict b, size_t n);

	static void vector_mul(std::span<double> a, std::span<const double> b)
	{
		vector_mul(a.data(), b.data(), a.size());
	}

	static void vector_mul(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	static void vector_mul(std::span<const double> a, double b, std::span<double> c)
	{
		vector_mul(a.data(), b, c.data(), a.size());
	}

	static void vector_mul(double* const __restrict a, double b, size_t n);

	static void vector_mul(std::span<double> a, double b)
	{
		vector_mul(a.data(), b, a.size());
	}

	// Vector division

	static void vector_div(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	static void vector_div(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_div(a.data(), b.data(), c.data(), a.size());
	}

	static void vector_div(double* const __restrict a, const double* const __restrict b, size_t n);

	static void vector_div(std::span<double> a, std::span<const double> b)
	{
		vector_div(a.data(), b.data(), a.size());
	}

	static void vector_div(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	static void vector_div(std::span<const double> a, double b, std::span<double> c)
	{
		vector_div(a.data(), b, c.data(), a.size());
	}

	static void vector_div(double* const __restrict a, double b, size_t n);

	static void vector_div(std::span<double> a, double b)
	{
		vector_div(a.data(), b, a.size());
	}

	static void vector_div(double a, const double* const __restrict b, double* const __restrict c, size_t n);

	static void vector_div(double a, std::span<const double> b, std::span<double> c)
	{
		vector_div(a, b.data(), c.data(), b.size());
	}

	// Vector exponentiation

	static void vector_pow(const double* const __restrict a, const double* const __restrict b, double* const __restrict c, size_t n);

	static void vector_pow(std::span<const double> a, std::span<const double> b, std::span<double> c)
	{
		vector_pow(a.data(), b.data(), c.data(), a.size());
	}

	static void vector_pow(double* const __restrict a, const double* const __restrict b, size_t n);

	static void vector_pow(std::span<double> a, std::span<const double> b)
	{
		vector_pow(a.data(), b.data(), a.size());
	}

	static void vector_pow(const double* const __restrict a, double b, double* const __restrict c, size_t n);

	static void vector_pow(std::span<const double> a, double b, std::span<double> c)
	{
		vector_pow(a.data(), b, c.data(), a.size());
	}

	static void vector_pow(double* const __restrict a, double b, size_t n);

	static void vector_pow(std::span<double> a, double b)
	{
		vector_pow(a.data(), b, a.size());
	}

	static void vector_pow(double a, const double* const __restrict b, double* const __restrict c, size_t n);

	static void vector_pow(double a, std::span<const double> b, std::span<double> c)
	{
		vector_pow(a, b.data(), c.data(), b.size());
	}

	// Vector logarithm

	static void vector_log(const double* const __restrict arg, const double* const __restrict log_base, double* const __restrict r, size_t n);

	static void vector_log(std::span<const double> arg, std::span<const double> log_base, std::span<double> r)
	{
		vector_log(arg.data(), log_base.data(), r.data(), arg.size());
	}

	static void vector_log(double* const __restrict arg, const double* const __restrict log_base, size_t n);

	static void vector_log(std::span<double> arg, std::span<const double> log_base)
	{
		vector_log(arg.data(), log_base.data(), arg.size());
	}

	static void vector_log(const double* const __restrict arg, double log_base, double* const __restrict r, size_t n);

	static void vector_log(std::span<const double> arg, double log_base, std::span<double> r)
	{
		vector_log(arg.data(), log_base, r.data(), arg.size());
	}

	static void vector_log(double* const __restrict arg, double log_base, size_t n);

	static void vector_log(std::span<double> arg, double log_base)
	{
		vector_log(arg.data(), log_base, arg.size());
	}

	static void vector_log(double arg, const double* const __restrict log_base, double* const __restrict r, size_t n);

	static void vector_log(double arg, std::span<const double> log_base, std::span<double> r)
	{
		vector_log(arg, log_base.data(), r.data(), log_base.size());
	}

	// Vector fused multiply addition

	static void vector_fmadd(const double* const __restrict a, const double* const __restrict b, const double* const __restrict c,
		double* const __restrict r, size_t n);

	static void vector_fmadd(std::span<const double> a, std::span<const double> b, std::span<const double> c, std::span<double> r)
	{
		vector_fmadd(a.data(), b.data(), c.data(), r.data(), a.size());
	}

	static void vector_fmadd(double* const __restrict a, const double* const __restrict b, const double* const __restrict c, size_t n);

	static void vector_fmadd(std::span<double> a, std::span<const double> b, std::span<const double> c)
	{
		vector_fmadd(a.data(), b.data(), c.data(), a.size());
	}

	static void vector_fmadd(const double* const __restrict a, const double* const __restrict b, double c,
		double* const __restrict r, size_t n);

	static void vector_fmadd(std::span<const double> a, std::span<const double> b, double c, std::span<double> r)
	{
		vector_fmadd(a.data(), b.data(), c, r.data(), a.size());
	}

	static void vector_fmadd(double* const __restrict a, const double* const __restrict b, double c, size_t n);

	static void vector_fmadd(std::span<double> a, std::span<const double> b, double c)
	{
		vector_fmadd(a.data(), b.data(), c, a.size());
	}

	// Vector fused negative multiply addition

	static void vector_fnmadd(const double* const __restrict a, const double* const __restrict b, const double* const __restrict c,
		double* const __restrict r, size_t n);

	static void vector_fnmadd(std::span<const double> a, std::span<const double> b, std::span<const double> c, std::span<double> r)
	{
		vector_fnmadd(a.data(), b.data(), c.data(), r.data(), a.size());
	}

	static void vector_fnmadd(double* const __restrict a, const double* const __restrict b, const double* const __restrict c, size_t n);

	static void vector_fnmadd(std::span<double> a, std::span<const double> b, std::span<const double> c)
	{
		vector_fnmadd(a.data(), b.data(), c.data(), a.size());
	}

	static void vector_fnmadd(const double* const __restrict a, const double* const __restrict b, double c,
		double* const __restrict r, size_t n);

	static void vector_fnmadd(std::span<const double> a, std::span<const double> b, double c, std::span<double> r)
	{
		vector_fnmadd(a.data(), b.data(), c, r.data(), a.size());
	}

	static void vector_fnmadd(double* const __restrict a, const double* const __restrict b, double c, size_t n);

	static void vector_fnmadd(std::span<double> a, std::span<const double> b, double c)
	{
		vector_fnmadd(a.data(), b.data(), c, a.size());
	}

	// Vector square

	static void vector_sq(const double* const __restrict a, double* const __restrict r, size_t n);

	static void vector_sq(std::span<const double> a, std::span<double> r)
	{
		vector_sq(a.data(), r.data(), a.size());
	}

	static void vector_sq(double* const __restrict a, size_t n);

	static void vector_sq(std::span<double> a)
	{
		vector_sq(a.data(), a.size());
	}

	// Vector square root

	static void vector_sqrt(const double* const __restrict a, double* const __restrict r, size_t n);

	static void vector_sqrt(std::span<const double> a, std::span<double> r)
	{
		vector_sqrt(a.data(), r.data(), a.size());
	}

	static void vector_sqrt(double* const __restrict a, size_t n);

	static void vector_sqrt(std::span<double> a)
	{
		vector_sqrt(a.data(), a.size());
	}

	// Vector natural exponentiation

	static void vector_exp(const double* const __restrict a, double* const __restrict r, size_t n);

	static void vector_exp(std::span<const double> a, std::span<double> r)
	{
		vector_exp(a.data(), r.data(), a.size());
	}

	static void vector_exp(double* const __restrict a, size_t n);

	static void vector_exp(std::span<double> a)
	{
		vector_exp(a.data(), a.size());
	}

	// Vector logarithm (special)

	static void vector_ln(const double* const __restrict a, double* const __restrict r, size_t n);

	static void vector_ln(std::span<const double> a, std::span<double> r)
	{
		vector_ln(a.data(), r.data(), a.size());
	}

	static void vector_ln(double* const __restrict a, size_t n);

	static void vector_ln(std::span<double> a)
	{
		vector_ln(a.data(), a.size());
	}

	// Vector operations

	static double vector_sum(const double* const __restrict a, size_t n);

	static double vector_sum(std::span<const double> a)
	{
		return vector_sum(a.data(), a.size());
	}

	static double vector_max(const double* const __restrict a, size_t n);

	static double vector_max(std::span<const double> a)
	{
		return vector_max(a.data(), a.size());
	}

	static double vector_dot(const double* const __restrict a, const double* const __restrict b, size_t n);

	static double vector_dot(std::span<const double> a, std::span<const double> b)
	{
		return vector_dot(a.data(), b.data(), a.size());
	}

	static double vector_dot(const double* __restrict a, const double* __restrict b, int a_off, int b_off, int n);
};