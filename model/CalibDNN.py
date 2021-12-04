from model.DepthNet import *
from model.ResNet import *
import math
import os


class AggregationBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(AggregationBlock, self).__init__()
        self.conv0 = set_conv(in_channels, out_channels, kernel=1, strides=stride, padding=0)
        self.bn0 = set_batch_normalization(out_channels)

        self.conv1 = set_conv(in_channels, out_channels, kernel=3, strides=stride, padding=1)
        self.bn1 = set_batch_normalization(out_channels)

        self.conv2 = set_conv(out_channels, out_channels, kernel=3, padding=1)
        self.bn2 = set_batch_normalization(out_channels)

        self.relu = set_relu(True)

    def forward(self, x):
        residual = self.conv0(x)
        residual = self.bn0(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class CalibDNN(nn.Module):
    # ResNet18 + DepthNet18
    def __init__(self, layer_num):
        super(CalibDNN, self).__init__()

        self.channels = (512, 256, 128)
        self.model_name = 'CalibDNN_{}'.format(layer_num)

        ########################### Featrue Extraction #################################
        self.resnet = ResNet18(layer_num, 1000)

        self.firstmaxpool = set_max_pool(kernel=5, strides=1, padding=2)
        self.depthnet = DepthNet18(layer_num, 1000)
        ########################### Featrue Extraction #################################

        ########################### Featrue Aggregation #################################
        self.feature_aggregation1 = AggregationBlock(512 + 512, self.channels[0], stride=2)
        self.feature_aggregation2 = AggregationBlock(self.channels[0], self.channels[1], stride=2)

        self.conv0 = set_conv(self.channels[1], self.channels[2], kernel=3, strides=2, padding=1)
        self.bn0 = set_batch_normalization(self.channels[2])
        ########################### Decouple 2 Branch #################################
        self.conv_rot = set_conv(self.channels[2], self.channels[2], kernel=1, padding=0)
        self.bn_rot = set_batch_normalization(self.channels[2])
        self.fcl_rot = set_dense(self.channels[2], 4)

        self.conv_tr = set_conv(self.channels[2], self.channels[2], kernel=1, padding=0)
        self.bn_tr = set_batch_normalization(self.channels[2])
        self.fcl_tr = set_dense(self.channels[2], 3)
        ########################### Decouple 2 Branch #################################
        self.gap = set_global_average_pooling()
        self.dropout = set_dropout(0.5)
        self.relu = set_relu(True)
        ########################### Featrue Aggregation #################################

    def forward(self, x1, x2):
        x1 = self.resnet(x1)

        max_pool = self.firstmaxpool(x2)
        x2 = self.depthnet(max_pool)

        x = set_concat([x1, x2], axis=1)

        x = self.feature_aggregation1(x)
        x = self.feature_aggregation2(x)

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        rot = self.conv_rot(x)
        rot = self.bn_rot(rot)
        rot = self.relu(rot)
        rot = self.gap(rot)
        rot = rot.view(rot.size(0), -1)
        rot = self.dropout(rot)
        rot = self.fcl_rot(rot)

        tr = self.conv_tr(x)
        tr = self.bn_tr(tr)
        tr = self.relu(tr)
        tr = self.gap(tr)
        tr = tr.view(tr.size(0), -1)
        tr = self.dropout(tr)
        tr = self.fcl_tr(tr)

        return rot, tr

    def get_name(self):
        return self.model_name

    def initialize_weights(self, init_weights):
        # Feature Aggregation에서는 He-Normal 로 Initialization을 진행해아하지만, 여기서는 그냥 진행
        if init_weights is True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)

                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)


def CalibDNN18(layer_num, pretrained):
    model = CalibDNN(layer_num)

    if os.path.isfile(pretrained):
        print('CalibDNN18 : Pretrained Model!')
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(pretrained)
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        print('CalibDNN18 : No Pretrained Model!')
        model.initialize_weights(init_weights=True)

    return model

# devices = torch.device("cuda") if is_gpu_avaliable() else torch.device("cpu")
# model = CalibDNN18(18)
# summary(model, [(1, 3, 375, 1242), (1, 3, 375, 1242)], devices)
