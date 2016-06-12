// "RegionWarping" layer implements a region bilinear pooling layer similar to the one described on [*] 
// (there called region warping layer) WITHOUT following however the exaxt definition given there. 
// It functionality is:
// 
// FORWARD: Given the convolutional feature maps of an image (1st input blob) and a set of regions inside 
// the image of arbitraty size (2nd input blob), it extracts fixed size region feature maps (output blob)
// by: (1) projecting the regions to the domain of the input image conv. feature maps (2) assuming a 
// H_R x W_R regular grid in the place of each projected region (where W_R and H_R are the width and 
// height of the output region feature maps) and then (3) performing bilinear pooling on each point of the
// H_R x W_R regular grid in order create an output region feature of size C x H_R x W_R where C is the
// number of input feature maps (feature channels).
//
// BACKWARD: Given the top gradients w.r.t. the region features (top blob gradients) it computes:
// (A) the gradients w.r.t. the input image conv. feature maps (1st bottom blob gradients)
// (B) and optionally the gradients w.r.t. the input regions (2nd bottom blob gradients).
// In my work I only needed the gradients w.r.t. image conv. feature maps (case (A)) and this code  
// it worked for me just fine in all my experiments/projects. On the other hand, I never needed the
// gradients w.r.t. the input regions (case (B)) thus appart from a few code checks I have not 
// verified if the case (B) really works inside a CNN training scenario. 
// 
// [*] Jifeng Dai, Kaiming He and Jian Sun: Instance-aware Semantic Segmentation via Multi-task Network Cascades
// 
// --------------------------------------------------------
// Author: Spyros Gidaris
// ---------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>   

#include "caffe/layers/region_warping_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void RegionWarpingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  force_region_grads_ = region_pool_param.force_region_grads();
}

template <typename Dtype>
void RegionWarpingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,  pooled_width_);
}

template <typename Dtype>
void RegionWarpingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // const Dtype* bottom_data = bottom[0]->cpu_data();
  // Number of ROIs
  const int num_rois = bottom[1]->num();
  // const int num_rois_dims = bottom[1]->channels();
  const int batch_size = bottom[0]->num();
  const int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(0), top_data);

// For each ROI R = [batch_index, x0, y0, x1, y1]: 
for (int r = 0; r < num_rois; ++r) {
    // pointer to the n-th region of interest
    const Dtype* bottom_rois = bottom[1]->cpu_data() + bottom[1]->offset(r);
    const int roi_batch_ind  = bottom_rois[0];
    // project region on the feature map space
    const Dtype roi_x0 = (bottom_rois[1] + offset_) * spatial_scale_;
    const Dtype roi_y0 = (bottom_rois[2] + offset_) * spatial_scale_;
    const Dtype roi_x1 = (bottom_rois[3] - offset_) * spatial_scale_;
    const Dtype roi_y1 = (bottom_rois[4] - offset_) * spatial_scale_;

    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    // Are the following two equations correct? 
    const Dtype roi_height = max(Dtype(roi_y1 - roi_y0 + 1), Dtype(1)); 
    const Dtype roi_width  = max(Dtype(roi_x1 - roi_x0 + 1), Dtype(1));

    const Dtype scale_factor_y = static_cast<Dtype>(roi_height-1) / static_cast<Dtype>(pooled_height_-1+FLT_EPSILON);
    const Dtype scale_factor_x = static_cast<Dtype>(roi_width-1)  / static_cast<Dtype>(pooled_width_-1+FLT_EPSILON);

    for (int c = 0; c < channels_; ++c) {
        // pointer to the input feature map of the roi_batch_ind-th image and the c-th channel
	const Dtype* feature_map = bottom[0]->cpu_data() + bottom[0]->offset(roi_batch_ind, c);
	// pointer to the output region feature map of the n-th region and for the c-th channel
	Dtype* region_feature_map = top_data + top[0]->offset(r,c);
    	for (int vv = 0; vv < pooled_height_; ++vv) {
		for (int uu = 0; uu < pooled_height_; ++uu) {
            		// uu and vv coordinates w.r.t. the feature map
	    		const Dtype uu_wrt_feat = roi_x0 + uu * scale_factor_x;
            		const Dtype vv_wrt_feat = roi_y0 + vv * scale_factor_y;

            		// find the range [u_min, u_max] of u inside which the function k(uu_wrt_feat - u) > 0; k(x) = max(1, 1 - |x|);
            		const int u_min =  static_cast<int>(min(width_-1,  max(0,static_cast<int>( ceil(uu_wrt_feat-1)))));
            		const int u_max =  static_cast<int>(min(width_-1,  max(0,static_cast<int>(floor(uu_wrt_feat+1))))); 
	    		// find the range [v_min, v_max] of v inside which the function k(vv_wrt_feat - v) > 0;
            		const int v_min =  static_cast<int>(min(height_-1, max(0, static_cast<int>(ceil(vv_wrt_feat-1)))));
            		const int v_max =  static_cast<int>(min(height_-1, max(0,static_cast<int>(floor(vv_wrt_feat+1))))); 

	    		Dtype sampled_feat = Dtype(0);
            		for (int v = v_min; v <= v_max; v++) {
	        		const Dtype k_value_v = max(Dtype(0),Dtype(1-fabs(vv_wrt_feat - static_cast<Dtype>(v))));
	      			for (int u = u_min; u <= u_max; u++) {
					const int in_index = v * width_ + u;			
					const Dtype k_value_u = max(Dtype(0),Dtype(1-fabs(uu_wrt_feat - static_cast<Dtype>(u))));
					sampled_feat += k_value_v * k_value_u * feature_map[in_index];
				}
	    		}
			const int out_index = vv * pooled_width_ + uu;
			region_feature_map[out_index] = sampled_feat;
	    	}
        }
    }
  }
}

template <typename Dtype>
void RegionWarpingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RegionWarpingLayer);
#endif

INSTANTIATE_CLASS(RegionWarpingLayer);
REGISTER_LAYER_CLASS(RegionWarping);

}  // namespace caffe
