

global_controller = None  # 初始化为空

def set_predictor_controller(controller):
    global global_controller
    global_controller = controller

def get_predictor_controller():
    global global_controller
    if global_controller is None:
        raise RuntimeError("PredictorController has not been initialized yet.")
    return global_controller
