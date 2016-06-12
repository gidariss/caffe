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
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {


template <typename Dtype>
__device__ void get_min_point(Dtype u_sampled, int& u_min, Dtype& u_weight_min)
{
   u_min        = static_cast<int>(floor(u_sampled));
   u_weight_min = Dtype(1) - (u_sampled - Dtype(u_min));
}

__device__ bool between(const int value, const int lowerBound, const int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}

template <typename Dtype>
__global__ void RegionWarpingForward(const int nthreads, const Dtype* feature_maps, // feature_map = bottom_data
    const Dtype spatial_scale, const Dtype offset, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width, const int rois_num_dims,
    const Dtype* bottom_rois, Dtype* region_feature_maps) { // region_feature_maps = top_data
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (r, c, vv, uu) is an element in the pooled output
    int uu = index % pooled_width; // column in the pooled output
    int vv = (index / pooled_width) % pooled_height; // row in the pooled output
    int c  = (index / pooled_width / pooled_height) % channels; // channel in the pooled output
    int r  = index / pooled_width / pooled_height / channels; // region in the pooled output

    const Dtype* bottom_rois_this = bottom_rois + r * rois_num_dims;
    int roi_batch_ind = bottom_rois_this[0];

    // project region on the feature map space (real valued coordinates)
    const Dtype roi_x0 = (bottom_rois_this[1] + offset) * spatial_scale;
    const Dtype roi_y0 = (bottom_rois_this[2] + offset) * spatial_scale;
    const Dtype roi_x1 = (bottom_rois_this[3] - offset) * spatial_scale;
    const Dtype roi_y1 = (bottom_rois_this[4] - offset) * spatial_scale;

    // Force malformed ROIs to be 1x1; is this ???
    const Dtype roi_height = max(Dtype(roi_y1 - roi_y0 + 1), Dtype(1)); 
    const Dtype roi_width  = max(Dtype(roi_x1 - roi_x0 + 1), Dtype(1));

    const Dtype scale_factor_y = static_cast<Dtype>(roi_height-1) / static_cast<Dtype>(pooled_height-1+FLT_EPSILON);
    const Dtype scale_factor_x = static_cast<Dtype>(roi_width-1)  / static_cast<Dtype>(pooled_width-1+FLT_EPSILON);

    // uu and vv coordinates w.r.t. the feature map; uu and vv are spatial coordinates of the region pooled feature
    const Dtype uu_wrt_feat = roi_x0 + uu * scale_factor_x;
    const Dtype vv_wrt_feat = roi_y0 + vv * scale_factor_y;

    // find the range [u_min, u_max] of u inside which the function k(uu_wrt_feat - u) > 0; k(x) = max(1, 1 - |x|);
    // find the range [v_min, v_max] of v inside which the function k(vv_wrt_feat - v) > 0;
    int u_min, v_min;
    Dtype v_weight_min, u_weight_min;
    get_min_point(uu_wrt_feat, u_min, u_weight_min);
    get_min_point(vv_wrt_feat, v_min, v_weight_min);
    int u_max = u_min + 1;
    int v_max = v_min + 1;
    Dtype u_weight_max = Dtype(1) - u_weight_min;
    Dtype v_weight_max = Dtype(1) - v_weight_min; 

    // is u_min, u_max, v_min, or v_max inside the borders
    const bool u_min_is_inside = between(u_min, 0, width-1); 
    const bool u_max_is_inside = between(u_max, 0, width-1);
    const bool v_min_is_inside = between(v_min, 0, height-1);
    const bool v_max_is_inside = between(v_max, 0, height-1);

    // pointer to the input feature map of the roi_batch_ind-th image and the c-th channel
    const Dtype* feature_maps_offset = feature_maps + (roi_batch_ind * channels + c) * height * width;
    
    // here is the bilinear interpolation performed 
    Dtype sampled_feat = Dtype(0);
    if (u_min_is_inside && v_min_is_inside) {
	const int in_index = v_min * width + u_min;	
	sampled_feat += u_weight_min * v_weight_min * feature_maps_offset[in_index];
    }
    if (u_max_is_inside && v_min_is_inside) {
	const int in_index = v_min * width + u_max;	
	sampled_feat += u_weight_max * v_weight_min * feature_maps_offset[in_index];
    }
    if (u_min_is_inside && v_max_is_inside) {
	const int in_index = v_max * width + u_min;	
	sampled_feat += u_weight_min * v_weight_max * feature_maps_offset[in_index];
    }    
    if (u_max_is_inside && v_max_is_inside) {
	const int in_index = v_max * width + u_max;	
	sampled_feat += u_weight_max * v_weight_max * feature_maps_offset[in_index];
    }
    region_feature_maps[index] = sampled_feat;
  }
}

template <typename Dtype>
void RegionWarpingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  const int rois_num_dims = bottom[1]->count() / bottom[1]->num();
  // NOLINT_NEXT_LINE(whitespace/operators)
  RegionWarpingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, offset_, channels_, height_, width_,
      pooled_height_, pooled_width_, rois_num_dims, bottom_rois, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void RegionWarpingBackward(const int nthreads, const Dtype* region_feature_maps_diff, // region_feature_maps_diff = top_diff
    const int num_rois, const Dtype spatial_scale, const Dtype offset, 
    const int channels, const int height, const int width, const int rois_num_dims,
    const int pooled_height, const int pooled_width, Dtype* feature_maps_diff, // feature_maps_diff = bottom_diff
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (uu, vv, c, r) is an element in the pooled output
    int uu = index % pooled_width; // column in the pooled output
    int vv = (index / pooled_width) % pooled_height; // row in the pooled output
    int c  = (index / pooled_width / pooled_height) % channels; // channel in the pooled output
    int r  =  index / pooled_width / pooled_height / channels; // region in the pooled output

    const Dtype* bottom_rois_this = bottom_rois + r * rois_num_dims;
    int roi_batch_ind = bottom_rois_this[0];

    // project region on the feature map space (real valued coordinates)
    const Dtype roi_x0 = (bottom_rois_this[1] + offset) * spatial_scale;
    const Dtype roi_y0 = (bottom_rois_this[2] + offset) * spatial_scale;
    const Dtype roi_x1 = (bottom_rois_this[3] - offset) * spatial_scale;
    const Dtype roi_y1 = (bottom_rois_this[4] - offset) * spatial_scale;

    // Force malformed ROIs to be 1x1; is this ???
    const Dtype roi_height = max(Dtype(roi_y1 - roi_y0 + 1), Dtype(1)); 
    const Dtype roi_width  = max(Dtype(roi_x1 - roi_x0 + 1), Dtype(1));

    const Dtype scale_factor_y = static_cast<Dtype>(roi_height-1) / static_cast<Dtype>(pooled_height-1+FLT_EPSILON);
    const Dtype scale_factor_x = static_cast<Dtype>(roi_width-1)  / static_cast<Dtype>(pooled_width-1+FLT_EPSILON);

    // uu and vv coordinates w.r.t. the feature map; uu and vv are the spatial coordinates of the region pooled feature
    const Dtype uu_wrt_feat = roi_x0 + uu * scale_factor_x;
    const Dtype vv_wrt_feat = roi_y0 + vv * scale_factor_y;

    // find the range [u_min, u_max] of u inside which the function k(uu_wrt_feat - u) > 0; k(x) = max(1, 1 - |x|);
    // find the range [v_min, v_max] of v inside which the function k(vv_wrt_feat - v) > 0;
    int u_min, v_min;
    Dtype v_weight_min, u_weight_min;
    get_min_point(uu_wrt_feat, u_min, u_weight_min);
    get_min_point(vv_wrt_feat, v_min, v_weight_min);
    int u_max = u_min + 1;
    int v_max = v_min + 1;
    Dtype u_weight_max = Dtype(1) - u_weight_min;
    Dtype v_weight_max = Dtype(1) - v_weight_min; 

    // is u_min, u_max, v_min, or v_max inside the borders
    const bool u_min_is_inside = between(u_min, 0, width-1); 
    const bool u_max_is_inside = between(u_max, 0, width-1);
    const bool v_min_is_inside = between(v_min, 0, height-1);
    const bool v_max_is_inside = between(v_max, 0, height-1);

    // pointer to the feature map gradients of the roi_batch_ind-th image and the c-th channel
    Dtype* feature_maps_diff_offset = feature_maps_diff + (roi_batch_ind * channels + c) * height * width;

    // pointer to the input feature map of the roi_batch_ind-th image and the c-th channel
    // feature_maps += (roi_batch_ind * channels + c) * height * width;

    // gradient value w.r.t. the top region feature in the position index={uu,vv,c,r};
    const Dtype region_feature_maps_diff_this = region_feature_maps_diff[index]; 

    // here is the bilinear interpolation performed 
    if (u_min_is_inside && v_min_is_inside) {
	const int bottom_index = v_min * width + u_min;	
	caffe_gpu_atomic_add(u_weight_min * v_weight_min * region_feature_maps_diff_this, 
		feature_maps_diff_offset + bottom_index);
    }
    if (u_max_is_inside && v_min_is_inside) {
	const int bottom_index = v_min * width + u_max;	
	caffe_gpu_atomic_add(u_weight_max * v_weight_min * region_feature_maps_diff_this, 
		feature_maps_diff_offset + bottom_index);
    }
    if (u_min_is_inside && v_max_is_inside) {
	const int bottom_index = v_max * width + u_min;	
	caffe_gpu_atomic_add(u_weight_min * v_weight_max * region_feature_maps_diff_this, 
		feature_maps_diff_offset + bottom_index);
    }    
    if (u_max_is_inside && v_max_is_inside) {
	const int bottom_index = v_max * width + u_max;	
	caffe_gpu_atomic_add(u_weight_max * v_weight_max * region_feature_maps_diff_this, 
		feature_maps_diff_offset + bottom_index);
    }
  }
}

template <typename Dtype>
__global__ void RegionWarpingBackwardAlsoRegionGrad(const int nthreads, const Dtype* region_feature_maps_diff, // region_feature_maps_diff = top_diff
    const int num_rois, const Dtype spatial_scale, const Dtype offset, 
    const int channels, const int height, const int width, const int rois_num_dims,
    const int pooled_height, const int pooled_width, const Dtype* feature_maps, Dtype* feature_maps_diff, // feature_maps_diff = bottom_diff
    const Dtype* bottom_rois, Dtype* bottom_rois_diffs) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // 
    // index = (r * pooled_height + vv) * pooled_width + uu
    int uu = index % pooled_width; // column in the pooled output
    int vv = (index / pooled_width) % pooled_height; // row in the pooled output
    int r  =  index / pooled_width / pooled_height; // region in the pooled output

    const Dtype* bottom_rois_this = bottom_rois + r * rois_num_dims;
    int roi_batch_ind = bottom_rois_this[0];

    // project region on the feature map space (real valued coordinates)
    const Dtype roi_x0 = (bottom_rois_this[1] + offset) * spatial_scale;
    const Dtype roi_y0 = (bottom_rois_this[2] + offset) * spatial_scale;
    const Dtype roi_x1 = (bottom_rois_this[3] - offset) * spatial_scale;
    const Dtype roi_y1 = (bottom_rois_this[4] - offset) * spatial_scale;

    // Force malformed ROIs to be 1x1;
    const Dtype roi_height = max(Dtype(roi_y1 - roi_y0 + 1), Dtype(1)); 
    const Dtype roi_width  = max(Dtype(roi_x1 - roi_x0 + 1), Dtype(1));

    const Dtype scale_factor_y = static_cast<Dtype>(roi_height-1) / static_cast<Dtype>(pooled_height-1+FLT_EPSILON);
    const Dtype scale_factor_x = static_cast<Dtype>(roi_width-1)  / static_cast<Dtype>(pooled_width-1+FLT_EPSILON);

    // uu and vv coordinates w.r.t. the feature map; uu and vv are the spatial coordinates of the region pooled feature
    const Dtype uu_wrt_feat = roi_x0 + uu * scale_factor_x;
    const Dtype vv_wrt_feat = roi_y0 + vv * scale_factor_y;

    // find the range [u_min, u_max] of u inside which the function k(uu_wrt_feat - u) > 0; k(x) = max(1, 1 - |x|);
    // find the range [v_min, v_max] of v inside which the function k(vv_wrt_feat - v) > 0;
    int u_min, v_min;
    Dtype v_weight_min, u_weight_min;
    get_min_point(uu_wrt_feat, u_min, u_weight_min);
    get_min_point(vv_wrt_feat, v_min, v_weight_min);
    int u_max = u_min + 1;
    int v_max = v_min + 1;
    Dtype u_weight_max = Dtype(1) - u_weight_min;
    Dtype v_weight_max = Dtype(1) - v_weight_min; 

    // is u_min, u_max, v_min, or v_max inside the borders
    const bool u_min_is_inside = between(u_min, 0, width-1); 
    const bool u_max_is_inside = between(u_max, 0, width-1);
    const bool v_min_is_inside = between(v_min, 0, height-1);
    const bool v_max_is_inside = between(v_max, 0, height-1);

    // pointer to the feature map gradients of the roi_batch_ind-th image and the c-th channel
    // Dtype* feature_maps_diff_offset = feature_maps_diff + (roi_batch_ind * channels) * height * width;

    // pointer to the input feature map of the roi_batch_ind-th image and the c-th channel
    // const Dtype* feature_maps_offset = feature_maps + (roi_batch_ind * channels) * height * width;
    // const Dtype* region_feature_maps_diff_offset = region_feature_maps_diff + (r * channels * height + vv) * width + uu;

    // here is the bilinear interpolation performed 
    Dtype dot_umin_vmin = 0;
    Dtype dot_umax_vmin = 0;
    Dtype dot_umin_vmax = 0;
    Dtype dot_umax_vmax = 0;
    
    for (int c = 0; c < channels; c++) {
	    const Dtype* feature_maps_offset = feature_maps      + (roi_batch_ind * channels + c) * height * width;
	    Dtype* feature_maps_diff_offset  = feature_maps_diff + (roi_batch_ind * channels + c) * height * width;
            const int top_index = ((r * channels + c) * pooled_height + vv) * pooled_width + uu;
            // gradient value w.r.t. the top region feature in the position index={uu,vv,c,r};
	    const Dtype region_feature_maps_diff_this = region_feature_maps_diff[top_index]; 
	    if (u_min_is_inside && v_min_is_inside) {
		const int bottom_index = v_min * width + u_min;	
		caffe_gpu_atomic_add(u_weight_min * v_weight_min * region_feature_maps_diff_this, 
			feature_maps_diff_offset + bottom_index);

		//dot_umin_vmin += feature_maps_offset[bottom_index] * region_feature_maps_diff_this;
		dot_umin_vmin += feature_maps_offset[bottom_index] * region_feature_maps_diff_this;
	    }
	    if (u_max_is_inside && v_min_is_inside) {
		const int bottom_index = v_min * width + u_max;	
		caffe_gpu_atomic_add(u_weight_max * v_weight_min * region_feature_maps_diff_this, 
			feature_maps_diff_offset + bottom_index);

		dot_umax_vmin += feature_maps_offset[bottom_index] * region_feature_maps_diff_this;
	    }
	    if (u_min_is_inside && v_max_is_inside) {
		const int bottom_index = v_max * width + u_min;	
		caffe_gpu_atomic_add(u_weight_min * v_weight_max * region_feature_maps_diff_this, 
			feature_maps_diff_offset + bottom_index);

		dot_umin_vmax += feature_maps_offset[bottom_index] * region_feature_maps_diff_this;
	    }    
	    if (u_max_is_inside && v_max_is_inside) {
		const int bottom_index = v_max * width + u_max;	
		caffe_gpu_atomic_add(u_weight_max * v_weight_max * region_feature_maps_diff_this, 
			feature_maps_diff_offset + bottom_index);

		dot_umax_vmax += feature_maps_offset[bottom_index] * region_feature_maps_diff_this;
	    }
            //feature_maps_diff_offset += height * width;
	    //region_feature_maps_diff_offset += pooled_width * pooled_height;
    }
	
    const Dtype gradient_wrt_uu = (-v_weight_min * dot_umin_vmin) + (-v_weight_max * dot_umin_vmax) + 
	(+v_weight_min * dot_umax_vmin) + (+v_weight_max * dot_umax_vmax);
    const Dtype gradient_wrt_vv = (-u_weight_min * dot_umin_vmin) + (+u_weight_min * dot_umin_vmax) + 
        (-u_weight_max * dot_umax_vmin) + (+u_weight_max * dot_umax_vmax); 

    const Dtype x_diff = gradient_wrt_uu;
    const Dtype y_diff = gradient_wrt_vv;
    const Dtype w_diff = gradient_wrt_uu * Dtype(uu) /  (pooled_width-1+FLT_EPSILON);
    const Dtype h_diff = gradient_wrt_vv * Dtype(vv) / (pooled_height-1+FLT_EPSILON);

    const Dtype x0_diff = spatial_scale * (x_diff - w_diff);
    const Dtype y0_diff = spatial_scale * (y_diff - h_diff);
    const Dtype x1_diff = spatial_scale * w_diff;
    const Dtype y1_diff = spatial_scale * h_diff;

    Dtype* bottom_rois_diff_offset = bottom_rois_diffs + r * rois_num_dims;
    atomicAdd(reinterpret_cast<float*>(bottom_rois_diff_offset + 1), static_cast<float>(x0_diff));
    atomicAdd(reinterpret_cast<float*>(bottom_rois_diff_offset + 2), static_cast<float>(y0_diff));
    atomicAdd(reinterpret_cast<float*>(bottom_rois_diff_offset + 3), static_cast<float>(x1_diff));
    atomicAdd(reinterpret_cast<float*>(bottom_rois_diff_offset + 4), static_cast<float>(y1_diff));
  }
}


template <typename Dtype>
void RegionWarpingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!(propagate_down[0] || propagate_down[1])) { return; }

	const Dtype* bottom_rois = bottom[1]->gpu_data();
	const int num_rois       = bottom[1]->num();
	const int num_rois_dims  = bottom[1]->count() / num_rois;
	const int count_top      = top[0]->count();
	const Dtype* region_feat_diff = top[0]->gpu_diff();
	Dtype* feat_map_diff     = bottom[0]->mutable_gpu_diff();


  	const int feat_map_count = bottom[0]->count();
  	caffe_gpu_set(feat_map_count, Dtype(0.), feat_map_diff);

	if (propagate_down[1] || force_region_grads_) {
		const Dtype* feat_map_data = bottom[0]->gpu_data();
		Dtype* bottom_rois_diff = bottom[1]->mutable_gpu_diff();
		caffe_gpu_set(bottom[1]->count(), Dtype(0.), bottom_rois_diff);
		const int num_nthreads = count_top / top[0]->channels();
		//LOG(INFO) << "num_nthreads : "<< num_nthreads << " count_top " << count_top << " top[0]->channels() " << top[0]->channels();
	  	RegionWarpingBackwardAlsoRegionGrad<Dtype><<<CAFFE_GET_BLOCKS(num_nthreads), CAFFE_CUDA_NUM_THREADS>>>( 
		  num_nthreads, region_feat_diff, num_rois, spatial_scale_, offset_, channels_,
		  height_, width_, num_rois_dims, pooled_height_, pooled_width_, feat_map_data, feat_map_diff, // feature_maps_diff = bottom_diff
		  bottom_rois, bottom_rois_diff);
	} 
	else {
	  	RegionWarpingBackward<Dtype><<<CAFFE_GET_BLOCKS(count_top), CAFFE_CUDA_NUM_THREADS>>>(
		  count_top, region_feat_diff, num_rois, spatial_scale_, offset_, channels_,
		  height_, width_, num_rois_dims, pooled_height_, pooled_width_, feat_map_diff, bottom_rois);
        }

	CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(RegionWarpingLayer);

}  // namespace caffe
