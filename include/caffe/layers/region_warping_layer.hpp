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

#ifndef CAFFE_REGION_WARPING_LAYER_HPP_
#define CAFFE_REGION_WARPING_LAYER_HPP_

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class RegionWarpingLayer : public Layer<Dtype> {
 public:
  explicit RegionWarpingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionWarping"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Dtype spatial_scale_;
  Dtype offset_;
  bool force_region_grads_;
};

}  // namespace caffe

#endif  // CAFFE_REGION_WARPING_LAYER_HPP_
