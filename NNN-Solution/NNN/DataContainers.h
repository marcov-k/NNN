#pragma once

#include <vector>

// Container struct for convolution geometric data.
struct ConvGeometry
{
	size_t batches = 0;
	size_t spatial_rank = 0;
	size_t input_channels = 0;
	size_t filter_count = 0;

	std::vector<int> input_dims;
	std::vector<int> kernel_dims;
	std::vector<int> out_dims;

	std::vector<int> input_strides;
	std::vector<size_t> out_spatial_strides;
	std::vector<size_t> kernel_spatial_strides;

	std::vector<size_t> input_kernel_offset;
	std::vector<size_t> kernel_kernel_offset;

	size_t out_spatial_size = 0;
	size_t kernel_spatial_size = 0;
	size_t kernel_volume_size = 0;

	size_t im2col_rows = 0;
	size_t im2col_cols = 0;
};