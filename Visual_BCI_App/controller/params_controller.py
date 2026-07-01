class ParamsController:
    def __init__(self):
        self.current_window = 'home'

    def interfaceChange(self, a0):
        if a0 == 0:
            paramsController.current_window = 'home'
        else:
            paramsController.current_window = a0

paramsController = ParamsController()