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


#ifndef CAFFE_ADAPTIVE_REGION_POOLING_LAYER_HPP_
#define CAFFE_ADAPTIVE_REGION_POOLING_LAYER_HPP_

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class RegionAdaptivePoolingLayer : public Layer<Dtype> {
 public:
  explicit RegionAdaptivePoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionAdaptivePooling"; }

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
  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_ADAPTIVE_REGION_POOLING_LAYER_HPP_
