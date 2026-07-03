#pragma once

#include <vector>

// Container struct for convolution geometric data.
struct ConvGeometry
{
	int batches = 0;
	int spatial_rank = 0;
	int input_channels = 0;
	int filter_count = 0;

	std::vector<int> input_dims;
	std::vector<int> kernel_dims;
	std::vector<int> out_dims;

	std::vector<int> input_strides;
	std::vector<int> out_spatial_strides;
	std::vector<int> kernel_spatial_strides;

	std::vector<int> input_kernel_offset;
	std::vector<int> kernel_kernel_offset;

	int out_spatial_size = 0;
	int kernel_spatial_size = 0;
	int kernel_volume_size = 0;

	int im2col_rows = 0;
	int im2col_cols = 0;
};