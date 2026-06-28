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

	static void vector_add(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_add(std::span<double> a, std::span<const double> b);

	static void vector_add(std::span<const double> a, double b, std::span<double> c);

	static void vector_add(std::span<double> a, double b);

	// Vector subtraction

	static void vector_sub(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_sub(std::span<double> a, std::span<const double> b);

	static void vector_sub(std::span<const double> a, double b, std::span<double> c);

	static void vector_sub(std::span<double> a, double b);

	static void vector_sub(double a, std::span<const double> b, std::span<double> c);

	// Vector multiplication

	static void vector_mul(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_mul(std::span<double> a, std::span<const double> b);

	static void vector_mul(std::span<const double> a, double b, std::span<double> c);

	static void vector_mul(std::span<double> a, double b);

	// Vector division

	static void vector_div(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_div(std::span<double> a, std::span<const double> b);

	static void vector_div(std::span<const double> a, double b, std::span<double> c);

	static void vector_div(std::span<double> a, double b);

	static void vector_div(const double a, std::span<const double> b, std::span<double> c);

	// Vector exponentiation

	static void vector_pow(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_pow(std::span<double> a, std::span<const double> b);

	static void vector_pow(std::span<const double> a, double b, std::span<double> c);

	static void vector_pow(std::span<double> a, double b);

	static void vector_pow(double a, std::span<const double> b, std::span<double> c);

	// Vector logarithm

	static void vector_log(std::span<const double> arg, std::span<const double> log_base, std::span<double> r);

	static void vector_log(std::span<double> arg, std::span<const double> log_base);

	static void vector_log(std::span<const double> arg, double log_base, std::span<double> r);

	static void vector_log(std::span<double> arg, double log_base);

	static void vector_log(double arg, std::span<const double> log_base, std::span<double> r);

	// Vector fused multiply addition

	static void vector_fmadd(std::span<const double> a, std::span<const double> b, std::span<const double> c, std::span<double> r);

	static void vector_fmadd(std::span<double> a, std::span<const double> b, std::span<const double> c);

	static void vector_fmadd(std::span<const double> a, std::span<const double> b, double c, std::span<double> r);

	static void vector_fmadd(std::span<double> a, std::span<const double> b, double c);

	// Vector fused negative multiply addition

	static void vector_fnmadd(std::span<const double> a, std::span<const double> b, std::span<const double> c, std::span<double> r);

	static void vector_fnmadd(std::span<double> a, std::span<const double> b, std::span<const double> c);

	static void vector_fnmadd(std::span<const double> a, std::span<const double> b, double c, std::span<double> r);

	static void vector_fnmadd(std::span<double> a, std::span<const double> b, double c);

	// Vector square

	static void vector_sq(std::span<const double> a, std::span<double> r);

	static void vector_sq(std::span<double> a);

	// Vector square root

	static void vector_sqrt(std::span<const double> a, std::span<double> r);

	static void vector_sqrt(std::span<double> a);

	// Vector natural exponentiation

	static void vector_exp(std::span<const double> a, std::span<double> r);

	static void vector_exp(std::span<double> a);

	// Vector logarithm (special)

	static void vector_ln(std::span<const double> a, std::span<double> r);

	static void vector_ln(std::span<double> a);

	// Vector operations

	static double vector_sum(std::span<const double> a);

	static double vector_max(std::span<const double> a);

	static double vector_dot(std::span<const double> a, std::span<const double> b);

	static double vector_dot(const double* __restrict a, const double* __restrict b, int a_off, int b_off, int n);
};