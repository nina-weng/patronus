import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from models.receptive_field import compute_proto_layer_rf_info



class BasicConvEncoder(nn.Module):
    '''
    Based on EncoderPatch at ./prototype_vae/ae_patch/model_patch.py
    '''
    def __init__(self, stddev=0.1,channel_inputs = None,layer_filter_sizes=None,
                                    layer_strides=None,
                                    layer_paddings=None,
                                    ):
        super(BasicConvEncoder, self).__init__()
        assert len(layer_filter_sizes) == len(layer_strides) == len(layer_paddings)
        assert len(channel_inputs) ==  len(layer_filter_sizes) + 1

        layers = []
        for i in range(len(layer_filter_sizes)):
            layers.append(nn.Conv2d(channel_inputs[i], channel_inputs[i+1], layer_filter_sizes[i], 
                                    stride=layer_strides[i],padding=layer_paddings[i]))
            layers.append(nn.ReLU())
        self.features = nn.Sequential(*layers)

        self._init_weights(stddev)

    def forward(self, x,):
        x = self.features(x)
        return x

    def _init_weights(self, stddev):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=stddev)


class ProtoActLearningBlock(nn.Module):
    def __init__(self, 
                 num_prototypes=30, 
                 encoder_type = 'basic_conv',
                 stddev=0.1,
                 prototype_activation_function = 'log',
                 device=None,
                 img_size=None,
                 channel_inputs:tuple[int] =(1, 32, 64, 64, 128),
                 layer_filter_sizes:tuple[int] =(3,3,3,3),
                 layer_strides:tuple[int] = (2,1,1,1),
                 layer_paddings:tuple[int] = (1,0,0,0),
                 input_channels: int = 1,
                 ):
        super(ProtoActLearningBlock, self).__init__()
        self.device=device
        self.channel_inputs = list(channel_inputs)
        self.input_channels = input_channels
        self.channel_inputs[0] = input_channels
        self.num_prototypes = num_prototypes
        self.prototype_activation_function = prototype_activation_function
        self.epsilon = 1e-4
        self.encoder_type = encoder_type
        assert encoder_type in ['basic_conv'],  f"Unsupported encoder type: {encoder_type}" 

        if self.encoder_type == 'basic_conv':
            self.encoder = BasicConvEncoder(stddev,
                                            channel_inputs = self.channel_inputs,
                                            layer_filter_sizes=layer_filter_sizes,
                                            layer_strides=layer_strides,
                                            layer_paddings=layer_paddings,
                                            )
            
            self.prototype_shape = (num_prototypes, channel_inputs[-1],1,1)
            self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                                requires_grad=True)
            self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)
            
            self.proto_layer_rf_info = compute_proto_layer_rf_info(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=self.prototype_shape[2])
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
            



    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances #shape = (BS,num_prototypes,7,7)


    def prototype_distances(self, conv_features):
        '''
        conv_features = self.conv_features(x) 
        '''
        distances = self._l2_convolution(conv_features)
        return distances
    
    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x, ts=None):
        hidden_rep = self.encoder(x) # (BS, channel_inputs[-1], hidden_rep_size, hidden_rep_size)
        hidden_rep_size = hidden_rep.shape[2]

        # --> k2: now the hidden representation of the corresponding prototype, shape (BS, channel_inputs[-1]) 
        # rename k2 to hidden_rep_reshape, as this implies what it is
        hidden_rep_reshape = hidden_rep.permute(0, 2, 3, 1) # (BS, hidden_rep_size, hidden_rep_size, channel_inputs[-1])
        hidden_rep_reshape = hidden_rep_reshape.view(hidden_rep_reshape.shape[0], hidden_rep_size*hidden_rep_size,self.channel_inputs[-1]).detach() # (BS, hidden_rep_size*hidden_rep_size, channel_inputs[-1])
        
        distances = self.prototype_distances(hidden_rep) 
        neg_distances = -distances
        min_distances,min_indices = F.max_pool2d(neg_distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]),
                                                   return_indices=True)
        min_distances = -min_distances # distances.shape=torch.Size([BS, num_p]) 
        min_indices = min_indices.view(min_indices.shape[0], -1) # (BS, num_prototypes)

        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances) # (BS, num_prototypes)

        return prototype_activations




