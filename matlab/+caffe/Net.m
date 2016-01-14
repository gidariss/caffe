classdef Net < handle
  % Wrapper class of caffe::Net in matlab
  
  properties (Access = private)
    hNet_self
    attributes
    % attribute fields
    %     hLayer_layers
    %     hBlob_blobs
    %     input_blob_indices
    %     output_blob_indices
    %     layer_names
    %     blob_names
    %     bottom_id_vecs
    %     top_id_vecs
  end
  properties (SetAccess = private)
    layer_vec
    blob_vec
    inputs
    outputs
    name2layer_index
    name2blob_index
    layer_names
    blob_names
    bottom_id_vecs_per_layer
    top_id_vecs_per_layer
  end
  
  methods
    function self = Net(varargin)
      % decide whether to construct a net from model_file or handle
      if ~(nargin == 1 && isstruct(varargin{1}))
        % construct a net from model_file
        self = caffe.get_net(varargin{:});
        return
      end
      % construct a net from handle
      hNet_net = varargin{1};
      CHECK(is_valid_handle(hNet_net), 'invalid Net handle');
      
      % setup self handle and attributes
      self.hNet_self = hNet_net;
      self.attributes = caffe_('net_get_attr', self.hNet_self);
      
      % setup layer_vec
      self.layer_vec = caffe.Layer.empty();
      for n = 1:length(self.attributes.hLayer_layers)
        self.layer_vec(n) = caffe.Layer(self.attributes.hLayer_layers(n));
      end
      
      % setup blob_vec
      self.blob_vec = caffe.Blob.empty();
      for n = 1:length(self.attributes.hBlob_blobs);
        self.blob_vec(n) = caffe.Blob(self.attributes.hBlob_blobs(n));
      end
      
      % setup input and output blob and their names
      % note: add 1 to indices as matlab is 1-indexed while C++ is 0-indexed
      self.inputs = ...
        self.attributes.blob_names(self.attributes.input_blob_indices + 1);
      self.outputs = ...
        self.attributes.blob_names(self.attributes.output_blob_indices + 1);
      
      % create map objects to map from name to layers and blobs
      self.name2layer_index = containers.Map(self.attributes.layer_names, ...
        1:length(self.attributes.layer_names));
      self.name2blob_index = containers.Map(self.attributes.blob_names, ...
        1:length(self.attributes.blob_names));
      
      % expose layer_names and blob_names for public read access
      self.layer_names = self.attributes.layer_names;
      self.blob_names = self.attributes.blob_names;

      % expose the bottom and top blob ids of each layer
      self.bottom_id_vecs_per_layer = self.attributes.bottom_id_vecs;
      self.top_id_vecs_per_layer = self.attributes.top_id_vecs;
    end
    function layer = layers(self, layer_name)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      layer = self.layer_vec(self.name2layer_index(layer_name));
    end
    function blob = blobs(self, blob_name)
      CHECK(ischar(blob_name), 'blob_name must be a string');
      blob = self.blob_vec(self.name2blob_index(blob_name));
    end
    function bottom_blob_ids = bottom_blob_ids_of(self, layer_name)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      layer_index = self.name2layer_index(layer_name);
      bottom_blob_ids = self.bottom_id_vecs_per_layer{layer_index}+1;
    end
    function top_blob_ids = top_blob_ids_of(self, layer_name)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      layer_index = self.name2layer_index(layer_name);
      top_blob_ids = self.top_id_vecs_per_layer{layer_index}+1;
    end
    function layer_ids = layer_ids_with_input_blob(self, blob_name)
      CHECK(ischar(blob_name), 'blob_name must be a string');
      blob_index = self.name2blob_index(blob_name) - 1;
      layer_ids = [];
      for i = 1:length(self.bottom_id_vecs_per_layer)
        if any(self.bottom_id_vecs_per_layer{i} == blob_index)
          layer_ids = [layer_ids; i];
        end
      end
    end 
    function blob = params(self, layer_name, blob_index)
      CHECK(ischar(layer_name), 'layer_name must be a string');
      CHECK(isscalar(blob_index), 'blob_index must be a scalar');
      blob = self.layer_vec(self.name2layer_index(layer_name)).params(blob_index);
    end
    function forward_prefilled(self)
      caffe_('net_forward', self.hNet_self);
    end
    function forward_prefilled_from(self, startLayerIdx)
      caffe_('net_forward_from', self.hNet_self, startLayerIdx-1);
    end
    function forward_prefilled_from_to(self, startLayerIdx, stopLayerIdx)
      caffe_('net_forward_from_to', self.hNet_self, startLayerIdx-1, stopLayerIdx-1);
    end
    function backward_prefilled(self)
      caffe_('net_backward', self.hNet_self);
    end
    function res = forward(self, input_data)
      CHECK(iscell(input_data), 'input_data must be a cell array');
      CHECK(length(input_data) == length(self.inputs), ...
        'input data cell length must match input blob number');
      % copy data to input blobs
      for n = 1:length(self.inputs)
        self.blobs(self.inputs{n}).set_data(input_data{n});
      end
      self.forward_prefilled();
      % retrieve data from output blobs
      res = cell(length(self.outputs), 1);
      for n = 1:length(self.outputs)
        res{n} = self.blobs(self.outputs{n}).get_data();
      end
    end
    function res = backward(self, output_diff)
      CHECK(iscell(output_diff), 'output_diff must be a cell array');
      CHECK(length(output_diff) == length(self.outputs), ...
        'output diff cell length must match output blob number');
      % copy diff to output blobs
      for n = 1:length(self.outputs)
        self.blobs(self.outputs{n}).set_diff(output_diff{n});
      end
      self.backward_prefilled();
      % retrieve diff from input blobs
      res = cell(length(self.inputs), 1);
      for n = 1:length(self.inputs)
        res{n} = self.blobs(self.inputs{n}).get_diff();
      end
    end
    function copy_from(self, weights_file)
      CHECK(ischar(weights_file), 'weights_file must be a string');
      CHECK_FILE_EXIST(weights_file);
      caffe_('net_copy_from', self.hNet_self, weights_file);
    end
    function reshape(self)
      caffe_('net_reshape', self.hNet_self);
    end
    function save(self, weights_file)
      CHECK(ischar(weights_file), 'weights_file must be a string');
      caffe_('net_save', self.hNet_self, weights_file);
    end
  end
end
