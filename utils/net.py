from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
import os

class BaseFeatureExtractor(nn.Module):
    '''
    From https://github.com/thuml/Universal-Domain-Adaptation
    a base class for feature extractor
    '''
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)


class ResNet50Fc(BaseFeatureExtractor):
    """
    modefied from https://github.com/thuml/Universal-Domain-Adaptation
    implement ResNet50 as backbone, but the last fc layer is removed
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                model_resnet = models.resnet50(pretrained=False)
                model_resnet.load_state_dict(torch.load(model_path, weights_only=False))
            else:
                raise Exception('invalid model path!')
        else:
            model_resnet = models.resnet50(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        mod = list(model_resnet.children())
        mod.pop()
        self.feature_extractor = nn.Sequential(*mod)
        self.output_dim = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x


class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp 
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """
    def __init__(self, in_dim, out_dim, hidden_mlp=2048, feat_dim=256, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
                            nn.Linear(in_dim, hidden_mlp),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_mlp, feat_dim))
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls

class DynamicPrototypeLinear(nn.Module):
    """
    prototypeを動的に追加できるlinear layer。

    通常のnn.Linearのように
    weightをK x Dとして扱えるが、
    内部ではprototypeごとに
    nn.Parameterとして保持する。
    """

    def __init__(
        self,
        in_dim,
        out_dim
    ):

        super(
            DynamicPrototypeLinear,
            self
        ).__init__()

        self.in_dim = int(
            in_dim
        )


        # ----------------------------------------------------
        # nn.Linearと同じ初期化を利用
        # ----------------------------------------------------

        initial_linear = nn.Linear(
            in_dim,
            out_dim,
            bias=False
        )


        initial_weight = (
            initial_linear
            .weight
            .detach()
            .clone()
        )


        self.weight_list = (
            nn.ParameterList()
        )


        for i in range(
            out_dim
        ):

            parameter = nn.Parameter(
                initial_weight[
                    i
                ].clone()
            )

            self.weight_list.append(
                parameter
            )


    @property
    def weight(
        self
    ):
        """
        K x D のweight matrixを返す。
        """

        return torch.stack(
            list(
                self.weight_list
            ),
            dim=0
        )


    @property
    def out_features(
        self
    ):

        return len(
            self.weight_list
        )


    def forward(
        self,
        x
    ):

        return F.linear(
            x,
            self.weight
        )


    def add_prototype(
        self,
        weight
    ):
        """
        prototypeを1つ追加する。
        """

        new_parameter = nn.Parameter(
            weight.detach().clone()
        )

        self.weight_list.append(
            new_parameter
        )

        return new_parameter


class DynamicProtoCLS(nn.Module):
    """
    提案手法用の動的target prototype classifier。
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        temp=0.05
    ):

        super(
            DynamicProtoCLS,
            self
        ).__init__()

        self.fc = DynamicPrototypeLinear(
            in_dim,
            out_dim
        )

        self.tmp = temp

        self.weight_norm()


    @property
    def num_prototypes(
        self
    ):

        return (
            self.fc.out_features
        )


    def forward(
        self,
        x
    ):

        x = F.normalize(
            x,
            dim=1
        )

        x = (
            self.fc(x)
            / self.tmp
        )

        return x


    def weight_norm(
        self
    ):

        with torch.no_grad():

            for parameter in (
                self.fc.weight_list
            ):

                norm = parameter.data.norm(
                    p=2
                ).clamp_min(
                    1e-12
                )

                parameter.data.div_(
                    norm
                )


    def split_prototype(
        self,
        prototype_id,
        child_weight_1,
        child_weight_2
    ):
        """
        prototype kを2つへ分割。

        元のprototype:
            child_weight_1へ置換

        新prototype:
            child_weight_2として追加
        """

        prototype_id = int(
            prototype_id
        )


        child_weight_1 = F.normalize(
            child_weight_1.reshape(
                1,
                -1
            ),
            dim=1
        ).reshape(-1)


        child_weight_2 = F.normalize(
            child_weight_2.reshape(
                1,
                -1
            ),
            dim=1
        ).reshape(-1)


        with torch.no_grad():

            self.fc.weight_list[
                prototype_id
            ].copy_(
                child_weight_1
            )


        new_parameter = (
            self.fc.add_prototype(
                child_weight_2
            )
        )


        new_prototype_id = (
            self.num_prototypes
            - 1
        )


        return (
            new_parameter,
            new_prototype_id
        )
