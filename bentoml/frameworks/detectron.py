import os

from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

class DetectronArtifact(BentoServiceArtifact):

    def __init__(self, name):
        super(DetectronArtifact, self).__init__(name)

        self._model = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(['torch', "detectron2"])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, detectron_model):  # pylint:disable=arguments-differ
        try:
            import detectron2  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )
        self._model = detectron_model
        return self

    def load(self, path):
        try:
            from detectron2.checkpoint import DetectionCheckpointer  # noqa # pylint: disable=unused-import
            from detectron2.modeling import META_ARCH_REGISTRY
            from detectron2.config import get_cfg
            import json
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )

        json_object = json.loads(path)
        cfg = get_cfg()
        cfg.merge_from_file(json_object['cfg'])        
        meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
        model = meta_arch(cfg)
        model.eval()
        device = "cuda:{}".format(json_object['device'])
        model.to(device)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(json_object['ckpt'])
        return self.pack(model)


    def get(self):
        return self._model

    def save(self, dst):
        try:
            from detectron2.checkpoint import DetectionCheckpointer  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )
        checkpointer = DetectionCheckpointer(self._model, save_dir="output")
        checkpointer.save("model")  # save to output/model_999.pth

