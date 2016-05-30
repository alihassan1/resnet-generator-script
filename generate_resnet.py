from itertools import islice

#
def make_data_layer(num_layers, train_lmdb, test_lmdb, mean_file):
    str = '''name: "ResNet%d"
layer {
  name: "resnet_%d"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 32
    mean_file: "%s"
  }
  data_param {
    source: "%s"
    batch_size: 16
    backend: LMDB
  }
}
layer {
  name: "resnet_%d"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: true
    crop_size: 32
    mean_file: "%s"
  }
  data_param {
    source: "%s"
    batch_size: 32
    backend: LMDB
  }
}''' % (num_layers, num_layers, mean_file, train_lmdb, num_layers, mean_file, test_lmdb)
    return str

# Defines Convolution layer
def make_conv_layer(lname, lbottom, ltop, num_output, pad, ksize, stride, wf_type = 'msra', bias_term_flag = True):
    if bias_term_flag == True:
        str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  convolution_param {
    num_output: %d
    pad: %d
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "%s"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}''' % (lname, lbottom, ltop, num_output, pad, ksize, stride, wf_type)
    else:
        str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  convolution_param {
    num_output: %d
    bias_term: false
    pad: %d
    kernel_size: %d
    stride: %d
    weight_filler {
      type: "%s"
    }
    bias_filler {
      type: "constant"
      value: 0.2
    }
  }
}''' % (lname, lbottom, ltop, num_output, pad, ksize, stride, wf_type)

    return str

# Defines batch normalization layer
def make_batchNorm_layer(lname, lbottom, ltop):
    str = '''layer {
  name: "%s"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"
  batch_norm_param {
  }
}''' % (lname, lbottom, ltop)
    return str

# Defines scaling layer
def make_scale_layer(lname, lbottom, ltop):
    str = '''layer {
  name: "%s"
  type: "Scale"
  bottom: "%s"
  top: "%s"
  scale_param {
    bias_term: true
  }
}''' % (lname, lbottom, ltop)
    return str

# Defines activation layer
def make_activation_layer(lname, lbottom, ltop, activation_function = 'ReLU'):
    str = '''layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}''' % (lname, activation_function, lbottom, ltop)
    return str

# Defines a pooling layer
def make_pooling_layer(lname, lbottom, ltop, ksize, stride, pool_function = 'MAX'):
    str = '''layer {
  name: "%s"
  type: "Pooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_size: %d
    stride: %d
  }
}''' % (lname, lbottom, ltop, pool_function, ksize, stride)
    return str

def make_eltwise_layer(lname, lbottom1, lbottom2, ltop):
    str = '''layer {
  name: "%s"
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
}''' % (lname, lbottom1, lbottom2, ltop)
    return str

def make_block_unit(name_str, prev_top, num_output, pad, ksize, stride, wf_type, activation_flag = True):
    # nstr = '%s_branch%s' % (stage, branch)
    lname = 'res' + name_str
    unit_str = '\n' + make_conv_layer(lname, prev_top, lname, num_output, pad, ksize, stride, wf_type, False)
    prev_top = lname
    unit_str += '\n' + make_batchNorm_layer('bn' + name_str, prev_top, prev_top)
    unit_str += '\n' + make_scale_layer('scale' + name_str, prev_top, prev_top)
    if activation_flag == True:
        unit_str += '\n' + make_activation_layer('res' + name_str + '_relu', prev_top, prev_top, 'ReLU')
    return unit_str, prev_top

def make_resnet_block(stage_num, branch_alphabet, stage_num_outputs, prev_top, wf_type, firstUnitStride = 1):
    nstr = '%s_branch%s' % (str(stage_num) + branch_alphabet, '2a')
    num_output = stage_num_outputs[0]
    unit_str, prev_top = make_block_unit(nstr, prev_top, num_output, 0, 1, firstUnitStride, wf_type, True)
    branch_str = unit_str

    nstr = '%s_branch%s' % (str(stage_num) + branch_alphabet, '2b')
    unit_str, prev_top = make_block_unit(nstr, prev_top, num_output, 1, 3, 1, wf_type, True)
    branch_str += unit_str

    nstr = '%s_branch%s' % (str(stage_num) + branch_alphabet, '2c')
    num_output = stage_num_outputs[1]
    unit_str, prev_top = make_block_unit(nstr, prev_top, num_output, 0, 1, 1, wf_type, False)
    branch_str += unit_str

    return branch_str, prev_top

def make_fc_layer(lname, lbottom, ltop, num_output, wf_type = 'msra'):
    str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  inner_product_param {
    num_output: %d
    weight_filler {
      type: "%s"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}''' % (lname, lbottom, ltop, num_output, wf_type)
    return str

def make_loss_layer(lname, type, lbottom, ltop, phase):
    str = '''layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  bottom: "label"
  top: "%s"
  include {
    phase: %s
  }
}''' % (lname, type, lbottom, ltop, phase)
    return str

def make_train_test_net(stages, num_categories, num_layers, train_lmdb, test_lmdb, mean_file):
    # generate data layer
    net_str = make_data_layer(num_layers, train_lmdb, test_lmdb, mean_file)

    # Set weight_filler
    wf_type = 'xavier'  # 'msra'

    # Set parameters for the first ResNet unit/block
    pad = 1
    ksize = 3
    stride = 1

    net_str += '\n' + make_conv_layer('conv1', 'data', 'conv1', 64, pad, ksize, stride, wf_type)
    net_str += '\n' + make_batchNorm_layer('bn_conv1', 'conv1', 'conv1')
    net_str += '\n' + make_scale_layer('scale_conv1', 'conv1', 'conv1')
    net_str += '\n' + make_activation_layer('conv1_relu', 'conv1', 'conv1', 'ReLU')
    net_str += '\n' + make_pooling_layer('pool1', 'conv1', 'pool1', 3, 1, 'MAX')

    # iterate over stages
    stage_num_outputs = [(64, 256), (128, 512), (256, 1024), (512, 2048)]
    strides = [1, 2, 2, 2] # strides for 1st unit at each stage
    abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    prev_top = 'pool1'
    for i in range(len(stages)):
        stage_num = i+2
        nstr = '%s_branch1' % (str(stage_num) + abc[0])
        num_output = stage_num_outputs[i][1]
        unit_str, prev_top_s1 = make_block_unit(nstr, prev_top, num_output, 0, 1, strides[i], wf_type, False)
        net_str += unit_str

        branch_str, prev_top_b1 = make_resnet_block(stage_num, abc[0], stage_num_outputs[i], prev_top, wf_type, strides[i])
        net_str += branch_str

        lname = 'res%d%s' % (stage_num, abc[0])
        eltwise_lstr = make_eltwise_layer(lname, prev_top_s1, prev_top_b1, lname)
        net_str += '\n' + eltwise_lstr
        act_str = make_activation_layer(lname + '_relu', lname, lname)
        net_str += '\n' + act_str

        prev_res_top = lname

        for j in range(1, stages[i]):
            branch_str, prev_top = make_resnet_block(stage_num, abc[j], stage_num_outputs[i], prev_res_top, wf_type)
            net_str += branch_str
            lname = 'res%d%s' % (stage_num, abc[j])
            eltwise_lstr = make_eltwise_layer(lname, prev_res_top, prev_top, lname)
            net_str += '\n' + eltwise_lstr
            act_str = make_activation_layer(lname + '_relu', lname, lname)
            net_str += '\n' + act_str
            prev_res_top = lname

        prev_top = prev_res_top

    net_str += '\n' + make_pooling_layer('pool5', prev_top, 'pool5', 3, 1, 'AVE')

    prev_top = 'pool5'
    lname = 'fc%d' % num_categories
    fc_str = make_fc_layer(lname, prev_top, lname, num_categories, wf_type)
    prev_top = lname
    net_str += '\n' + fc_str

    loss_tr_str = make_loss_layer('prob', 'SoftmaxWithLoss', prev_top, 'prob', 'TRAIN')
    net_str += '\n' + loss_tr_str

    loss_ts_str = make_loss_layer('test', 'Accuracy', prev_top, 'test', 'TEST')
    net_str += '\n' + loss_ts_str

    return net_str

def solver(train_file, snapshot_prefix):
    str = '''# The train/test net protocol buffer definition
net: "%s"
# test_iter specifies how many forward passes the test should carry out.
test_iter: 100
# Carry out testing every 500 training iterations.
test_interval: 500
test_initialization: false
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0001
# The learning rate policy
lr_policy: "step"
stepsize: 80000
gamma: 0.0001
power: 0.75
# Display every 100 iterations
display: 100
average_loss: 100
# The maximum number of iterations
max_iter: 10000
# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "%s"
# solver mode: CPU or GPU
solver_mode: GPU''' % (train_file, snapshot_prefix)
    return str

def main():

    # There are 4 stages in a typical ResNet, each stage can have multiple blocks, each block have three or four units.
    # if it s the first block of a stage then it will have four units; and three for all subsequent blocks.
    # See this awesome visualization for reference (http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006).

    # number of layers at each stage
    stages = [3, 2, 2, 2]
    num_categories = 10

    # total number of layers can be computed as follows
    num_layers = sum(stages)*3 + len(stages) + 1

    # NOTE: Convolution layer parameters can be modified from make_train_test_net function

    # Apologies for not writing an argument parser, I feel comfortable making modifications here when the argument
    # count is greater than 3-4.

    # add paths to the input data
    root_dir = 'C:/AH/ResNet/'
    train_lmdb = root_dir + 'data/cifar10_train_lmdb'
    test_lmdb = root_dir + 'data/cifar10_test_lmdb'
    mean_file = root_dir + 'data/cifar10_mean.binaryproto'

    # set path to the out file (train.prototxt)
    out_train_prototxt = root_dir + 'models/cifar10_train_' + str(num_layers) + '.prototxt'
    train_str = make_train_test_net(stages, num_categories, num_layers, train_lmdb, test_lmdb, mean_file)
    fp = open(out_train_prototxt, 'w')
    fp.write(train_str)
    fp.close()

    # set path for the solver file
    out_solver_prototxt = root_dir + 'models/cifar10_solver_' + str(num_layers) + '.prototxt'
    solver_str = solver(out_train_prototxt, root_dir + 'models/out/ah_resnet' + str(num_layers))
    fp = open(out_solver_prototxt, 'w')
    fp.write(solver_str)
    fp.close()

if __name__ == '__main__':
    main()

