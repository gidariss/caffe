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

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void RegionAdaptivePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  RegionPoolingParameter region_pool_param = this->layer_param_.region_pooling_param();
  CHECK_GT(region_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(region_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = region_pool_param.pooled_h();
  pooled_width_  = region_pool_param.pooled_w();
  spatial_scale_ = region_pool_param.spatial_scale();
  offset_ 	 = region_pool_param.offset();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void RegionAdaptivePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,  pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
}

template <typename Dtype>
void project_region_to_featuremap(const Dtype* region, const Dtype offset, const Dtype spatial_scale, int& x0, int& y0, int& x1, int& y1)
{
    x0 = round((region[0] + offset) * spatial_scale);
    y0 = round((region[1] + offset) * spatial_scale);
    x1 = round((region[2] - offset) * spatial_scale) - 1;
    y1 = round((region[3] - offset) * spatial_scale) - 1;

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
int get_bin_start_point(const int bin_index, Dtype bin_size, const int region_start_point, const int max_point_size) {
	return min(max_point_size, max(0, static_cast<int>(round(static_cast<Dtype>(bin_index) * bin_size)) + region_start_point));
}

template <typename Dtype>
int get_bin_stop_point(const int bin_index, Dtype bin_size, const int region_start_point, const int max_point_size) {
	return min(max_point_size, max(0, static_cast<int>(round(static_cast<Dtype>(bin_index) * bin_size)) + region_start_point));
}

template <typename Dtype>
void RegionAdaptivePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index, x_outer_1, y_outer_1, x_outer_2, y_outer_2, x_inner_1, y_inner_1, x_inner_2, y_inner_2]: 
  // where R_outer = [x_outer_1, y_outer_1, x_outer_2, y_outer_2] is the outer rectangle of the region and 
  // R_inner = [x_inner_1, y_inner_1, x_inner_2, y_inner_2] is the inner rectangle of the region
  // max pooler over R by ignoring (setting to zero) the activations that lay inside the inner rectangle R_inner
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind  = bottom_rois[0];
    // outer rectangle of the region
    int roi_start_w, roi_start_h, roi_end_w, roi_end_h;
    project_region_to_featuremap(bottom_rois+1, offset_, spatial_scale_, 
	roi_start_w, roi_start_h, roi_end_w, roi_end_h);
    const int roi_width  = max(roi_end_w - roi_start_w + 1, 1); // Force malformed ROIs to be 1x1
    const int roi_height = max(roi_end_h - roi_start_h + 1, 1); // Force malformed ROIs to be 1x1

    int roi_start_w_in, roi_start_h_in, roi_end_w_in, roi_end_h_in;
    project_region_to_featuremap(bottom_rois+5, offset_, spatial_scale_, 
	roi_start_w_in, roi_start_h_in, roi_end_w_in, roi_end_h_in);

    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width_);
   
    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)

     	  const int hstart = get_bin_start_point(ph,   bin_size_h, roi_start_h, height_);
          const int hend   = get_bin_stop_point( ph+1, bin_size_h, roi_start_h, height_);
          const int wstart = get_bin_start_point(pw,   bin_size_w, roi_start_w, width_);
          const int wend   = get_bin_stop_point( pw+1, bin_size_w, roi_start_w, width_);
          
          const int pool_index = ph * pooled_width_ + pw;
          top_data[pool_index] = 0;
          argmax_data[pool_index] = -1;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
	      if (!(w > roi_start_w_in && w < roi_end_w_in && h > roi_start_h_in && h < roi_end_h_in)) { 
                // if it is not inside the inner rectangle of the region
		const int index = h * width_ + w;
		if (batch_data[index] > top_data[pool_index]) {
		  top_data[pool_index] = batch_data[index];
		  argmax_data[pool_index] = index;
		}					
	      }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void RegionAdaptivePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RegionAdaptivePoolingLayer);
#endif

INSTANTIATE_CLASS(RegionAdaptivePoolingLayer);
REGISTER_LAYER_CLASS(RegionAdaptivePooling);

}  // namespace caffe
