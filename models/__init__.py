def create_model(opt):
    from .grasp_net import GraspNetModel
    model = GraspNetModel(opt)
    return model
