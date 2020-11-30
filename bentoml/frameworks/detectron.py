import os
from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv
from detectron2.config import CfgNode as CN

def _add_imagr_config(cfg):
    _C = cfg
    _C.INPUT.ROT90 = CN()
    _C.INPUT.ROT90.ENABLED = False
    _C.INPUT.CROP.ENABLED = False

    # Color distortion
    _C.INPUT.COLOR_DISTORTION = CN()
    _C.INPUT.COLOR_DISTORTION.ENABLED = False

    _C.INPUT.COLOR_DISTORTION.RANDOM_BRIGHTNESS = CN()
    _C.INPUT.COLOR_DISTORTION.RANDOM_BRIGHTNESS.ENABLED = False
    _C.INPUT.COLOR_DISTORTION.RANDOM_BRIGHTNESS.RANGE = [0.9, 1.1]

    _C.INPUT.COLOR_DISTORTION.RANDOM_CONTRAST = CN()
    _C.INPUT.COLOR_DISTORTION.RANDOM_CONTRAST.ENABLED = False
    _C.INPUT.COLOR_DISTORTION.RANDOM_CONTRAST.RANGE = [0.9, 1.1]

    _C.INPUT.COLOR_DISTORTION.RANDOM_SATURATION = CN()
    _C.INPUT.COLOR_DISTORTION.RANDOM_SATURATION.ENABLED = False
    _C.INPUT.COLOR_DISTORTION.RANDOM_SATURATION.RANGE = [0.9, 1.1]

    _C.INPUT.COLOR_DISTORTION.RANDOM_LIGHTING = CN()
    _C.INPUT.COLOR_DISTORTION.RANDOM_LIGHTING.ENABLED = False
    _C.INPUT.COLOR_DISTORTION.RANDOM_LIGHTING.SCALE = 0.05

    _C.INPUT.RANDOM_ERASING = CN()
    _C.INPUT.RANDOM_ERASING.ENABLED = False

    _C.INPUT.TIGHT_CROP = False  # If set, training AND validation will be a tight crop around the mask

    _C.ANOMALY_DETECTION = CN()
    _C.ANOMALY_DETECTION.PERCENTILE = 99.9
    _C.ANOMALY_DETECTION.FUNC = "linear"
    _C.ANOMALY_DETECTION.REFIT_PERIOD = 1000
    _C.ANOMALY_DETECTION.STEP_WINDOW = 1000
    _C.ANOMALY_DETECTION.LOSS_TO_MONITOR = "loss_mask"


def _add_pointrend_config(cfg):
    """
    Add config for PointRend.
    """
    # Names of the input feature maps to be used by a coarse mask head.
    cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.ROI_MASK_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_MASK_HEAD.NUM_FC = 2
    # The side size of a coarse mask head prediction.
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 7
    # True if point head is used.
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = False

    cfg.MODEL.POINT_HEAD = CN()
    cfg.MODEL.POINT_HEAD.NAME = "StandardPointHead"
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 80
    # Names of the input feature maps to be used by a mask point head.
    cfg.MODEL.POINT_HEAD.IN_FEATURES = ("p2",)
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 14 * 14
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = 3
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO = 0.75
    # Number of subdivision steps during inference.
    cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
    # Maximum number of points selected at each subdivision step (N).
    cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 28 * 28
    cfg.MODEL.POINT_HEAD.FC_DIM = 256
    cfg.MODEL.POINT_HEAD.NUM_FC = 3
    cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK = False
    # If True, then coarse prediction features are used as inout for each layer in PointRend's MLP.
    cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER = True


class DetectronModelArtifact(BentoServiceArtifact):
    def __init__(self, name):
        super(DetectronModelArtifact, self).__init__(name)
        self._file_name = name
        self._model = None
        self._aug = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(['torch', "detectron2"])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, detectron_model):  # pylint:disable=arguments-differ
        try:
            import detectron2  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronModelArtifact"
            )
        self._model = detectron_model
        return self

    def load(self, path):
        try:
            from detectron2.checkpoint import (
                DetectionCheckpointer,
            )  # noqa # pylint: disable=unused-import
            from detectron2.modeling import META_ARCH_REGISTRY
            from centermask2.centermask.config import get_cfg
            from detectron2.data import transforms as T
            import json
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )

        cfg = get_cfg()
        _add_imagr_config(cfg)
        _add_pointrend_config(cfg)  # setup YACS ConfigNodes used to configure model
        cfg.merge_from_file(f"{path}/{self._file_name}.yaml")
        meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
        self._model = meta_arch(cfg)
        self._model.eval()

        device = os.environ.get('BENTOML_DEVICE')
        if device == "GPU":
            device = "cuda:0"
        else:
            device = "cpu"
        self._model.to(device)
        checkpointer = DetectionCheckpointer(self._model)
        checkpointer.load(f"{path}/{self._file_name}.pth")
        self._aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        return self.pack(self._model)

    def get(self):
        return self._model

    def save(self, dst):
        try:
            from detectron2.checkpoint import (
                DetectionCheckpointer,
            )  # noqa # pylint: disable=unused-import
            from centermask2.centermask.config import get_cfg
            import shutil
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )
        os.makedirs(dst, exist_ok=True)
        # checkpointer = DetectionCheckpointer(self._model, save_dir=dst)
        # checkpointer.save(self._file_name)
        shutil.copy("model.pth", dst)
        shutil.copy("model.yaml", dst)
        # cfg = get_cfg()
        # cfg.merge_from_file("input_model.yaml")
        # with open(
        #     os.path.join(dst, "model.yaml"), 'w', encoding='utf-8'
        # ) as output_file:
        #     output_file.write(cfg.dump())
