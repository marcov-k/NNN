#pragma once

#include <immintrin.h>
#include <limits>
#include <random>
#include <span>

class MathUtils
{
public:
	MathUtils() = delete;

	static double get_random_double(double min = 0.0, double max = 1.0)
	{
		thread_local std::random_device rd;
		thread_local std::mt19937 gen(rd());

		std::uniform_real_distribution dis(min, max);

		return dis(gen);
	}

	static double sum_m256d(__m256d v);

	static double max_m256d(__m256d v);

	static void vector_add(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_add(std::span<double> a, std::span<const double> b);

	static void vector_add(std::span<const double> a, double b, std::span<double> c);

	static void vector_add(std::span<double> a, double b);

	static void vector_sub(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_sub(std::span<double> a, std::span<const double> b);

	static void vector_sub(std::span<const double> a, double b, std::span<double> c);

	static void vector_sub(std::span<double> a, double b);

	static void vector_sub(double a, std::span<const double> b, std::span<double> c);

	static void vector_mul(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_mul(std::span<double> a, std::span<const double> b);

	static void vector_mul(std::span<const double> a, double b, std::span<double> c);

	static void vector_mul(std::span<double> a, double b);

	static void vector_div(std::span<const double> a, std::span<const double> b, std::span<double> c);

	static void vector_div(std::span<double> a, std::span<const double> b);

	static void vector_div(std::span<const double> a, double b, std::span<double> c);

	static void vector_div(std::span<double> a, double b);

	static void vector_div(double a, std::span<const double> b, std::span<double> c);

	static double vector_sum(std::span<const double> a);

	static double vector_max(std::span<const double> a);

	static double vector_dot(std::span<const double> a, std::span<const double> b);
};