#pragma once

#include <immintrin.h>
#include <vector>

class MathUtils
{
public:
	MathUtils() = delete;

	static double sum_m256d(__m256d v);

	static void vector_add(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c);

	static void vector_add(std::vector<double>& a, const std::vector<double>& b);

	static void vector_add(const std::vector<double>& a, double b, std::vector<double>& c);

	static void vector_add(std::vector<double>& a, double b);

	static double vector_sum(const std::vector<double>& a);
};