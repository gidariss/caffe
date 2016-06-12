// "RegionAdaptivePooling" layer implements the region max pooling layer described on Fast-RCNN[1] modified 
// however in order to support the rectangular rings regions that were described on the MR-CNN system[2]. Those 
// rectangular ring regions are defined in terms of an inner rectangle and and an outer rectangle. During the 
// region pooling operation, both the inner and the outer rectangles are projected on the activation maps and 
// the activations that lay inside the inner rectangle are ignored during the adaptive max pooling operation.
//
// With respect to the "RegionPooling" layer, "RegionAdaptivePooling" includes a faster implementation of the 
// backward operation and some bug fixes during the forward/backward operation (both thanks to Sergey Zagoruyko and Adam Lerer).
// Due to those bug fixes, the outcome of a forward/backward operation of the "RegionAdaptivePooling" layer  
// is not identical to the outcome of the same operations in the "RegionPooling" layer. Hence, for backward 
// compatibility with models trained with the "RegionPooling" layer I kept both layers.
// 
// [1] Ross Girshick. "Fast-RCNN"
// [2] Spyros Gidaris and Nikos Komodakis. "Object detection via a multi-region & semantic segmentation-aware CNN model"
// --------------------------------------------------------
// Author: Spyros Gidaris
// ---------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/region_adaptive_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"


using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__device__ void project_region_to_featuremap(const Dtype* region, 
	const Dtype offset, const Dtype spatial_scale, 
	int& x0, int& y0, int& x1, int& y1)
{
    x0 = round((region[0] + offset) * spatial_scale);
    y0 = round((region[1] + offset) * spatial_scale);
    x1 = round( (region[2] - offset) * spatial_scale) - 1;
    y1 = round( (region[3] - offset) * spatial_scale) - 1;

    if (x0 > x1) {
    	x0 = (x0 + x1) / 2;
	x1 = x0;
    }
    if (y0 > y1) {
	y0 = (y0 + y1) / 2;
	y1 = y0;
    } 
}

template <typename Dtype>
__device__ int get_bin_start_point(const int bin_index, Dtype bin_size, const int region_start_point, const int max_point_size) {
	return min(max_point_size, max(0, static_cast<int>(round(static_cast<Dtype>(bin_index) * bin_size)) + region_start_point));
}

template <typename Dtype>
__device__ int get_bin_stop_point(const int bin_index, Dtype bin_size, const int region_start_point, const int max_point_size) {
	return min(max_point_size, max(0, static_cast<int>(round(static_cast<Dtype>(bin_index) * bin_size)) + region_start_point));
}

template <typename Dtype>
__global__ void RegionAdaptivePoolForward(
    const int num_output_elems, 
    const Dtype* bottom_data,
    const Dtype spatial_scale, 
    const Dtype offset, 
    const int channels, 
    const int height,
    const int width, 
    const int pooled_height, 
    const int pooled_width,
    const int num_rois_dims,
    const Dtype* bottom_rois, 
    Dtype* top_data, 
    int* argmax_data) {

  CUDA_KERNEL_LOOP(index, num_output_elems) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw =  index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c  = (index / pooled_width / pooled_height) % channels;
    int n  =  index / pooled_width / pooled_height / channels;
    const bool there_is_inner_rectangle = num_rois_dims >= 9;

    // For each ROI R = [batch_index, x_outer_1, y_outer_1, x_outer_2, y_outer_2, x_inner_1, y_inner_1, x_inner_2, y_inner_2]: 
    // where R_outer = [x_outer_1, y_outer_1, x_outer_2, y_outer_2] is the outer rectangle of the region and 
    // R_inner = [x_inner_1, y_inner_1, x_inner_2, y_inner_2] is the inner rectangle of the region
    // max pooler over R by ignoring (setting to zero) the activations that lay inside the inner rectangle R_inner
	
    const Dtype* bottom_this_roi = bottom_rois + n * num_rois_dims;
    int roi_batch_ind = bottom_this_roi[0];
    
    // outer rectangle of the region
    int roi_start_w, roi_start_h, roi_end_w, roi_end_h;
    project_region_to_featuremap(bottom_this_roi+1, offset, spatial_scale, 
	roi_start_w, roi_start_h, roi_end_w, roi_end_h);
    int roi_width  = max(roi_end_w - roi_start_w + 1, 1); // Force malformed ROIs to be 1x1
    int roi_height = max(roi_end_h - roi_start_h + 1, 1); // Force malformed ROIs to be 1x1
 
    int roi_start_w_in, roi_start_h_in, roi_end_w_in, roi_end_h_in;
    if (there_is_inner_rectangle) {
    	// inner rectangle of the region
	project_region_to_featuremap(bottom_this_roi+5, offset, spatial_scale, 
		roi_start_w_in, roi_start_h_in, roi_end_w_in, roi_end_h_in);
    }

    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width);

    const int hstart = get_bin_start_point(ph,   bin_size_h, roi_start_h, height);
    const int hend   = get_bin_stop_point( ph+1, bin_size_h, roi_start_h, height);
    const int wstart = get_bin_start_point(pw,   bin_size_w, roi_start_w, width);
    const int wend   = get_bin_stop_point( pw+1, bin_size_w, roi_start_w, width);

    // Define an empty pooling region to be zero
    bool is_empty = (hend <= hstart) || (wend <= wstart);
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    //Dtype maxval = 0;  
    int maxidx = -1;
    const Dtype* bottom_data_this = bottom_data + (roi_batch_ind * channels + c) * height * width;
    if (there_is_inner_rectangle) {
 	bool is_empty_ring_cell = true;
    	for (int h = hstart; h < hend; ++h) {
	  const bool is_inside_h = h > roi_start_h_in && h < roi_end_h_in;
      	  for (int w = wstart; w < wend; ++w) {
	    const bool is_inside_w = w > roi_start_w_in && w < roi_end_w_in;
	    if (!(is_inside_w && is_inside_h)) {  // check if it is inside the inner rectange of the region
              int bottom_index = h * width + w;
	      is_empty_ring_cell = false;
              if (bottom_data_this[bottom_index] > maxval) {
                maxval = bottom_data_this[bottom_index];
                maxidx = bottom_index;
              }
            }
          }
        }
	if (is_empty_ring_cell) {maxval = 0;}
    } else {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = h * width + w;
          if (bottom_data_this[bottom_index] > maxval) {
            maxval = bottom_data_this[bottom_index];
            maxidx = bottom_index;
          }
        }
      }
    }

    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void RegionAdaptivePoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const int num_rois       = bottom[1]->num();
  const int num_rois_dims  = bottom[1]->count() / num_rois;
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  RegionAdaptivePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, offset_, channels_, height_, width_,
      pooled_height_, pooled_width_, num_rois_dims, bottom_rois, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void RegionAdaptivePoolBackward(
    const int num_output_elems, 
    const Dtype* top_diff,
    const int* argmax_data, 
    const Dtype spatial_scale,
    const Dtype offset, 
    const int channels, 
    const int height, 
    const int width,
    const int pooled_height, 
    const int pooled_width,
    const int num_rois_dims,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, num_output_elems) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype* bottom_roi_this = bottom_rois + n * num_rois_dims;
    int roi_batch_ind = bottom_roi_this[0];

    int top_offset                  = (n * channels + c) * pooled_height * pooled_width;
    Dtype* offset_bottom_diff       = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    const Dtype* offset_top_diff    = top_diff    + top_offset;
    
    const int spatial_offset_top    = ph*pooled_width + pw;
    const int spatial_offset_bottom = argmax_data[top_offset + spatial_offset_top];
    if(spatial_offset_bottom != -1) {
      caffe_gpu_atomic_add(offset_top_diff[spatial_offset_top], 
	offset_bottom_diff + spatial_offset_bottom);
    }
  }
}

template <typename Dtype>
void RegionAdaptivePoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  const int count     	   = bottom[0]->count();
  const int count_top      = top[0]->count();
  const int num_rois       = bottom[1]->num();
  const int num_rois_dims  = bottom[1]->count() / num_rois;

  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  RegionAdaptivePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count_top, top_diff, argmax_data, spatial_scale_, offset_, channels_,
      height_, width_, pooled_height_, pooled_width_, num_rois_dims, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionAdaptivePoolingLayer);

}  // namespace caffe
