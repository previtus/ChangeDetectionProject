import Model1_SkipSiamFCN

class ModelHandler(object):
    """
    An intermediate between the code and bunch of tester models.
    """

    def __init__(self, settings, dataset):
        self.settings = settings
        self.model = Model1_SkipSiamFCN.Model1_SkipSiamFCN(settings, dataset)