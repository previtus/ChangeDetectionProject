import Model2_SiamUnet_Encoder

class ModelHandler(object):
    """
    An intermediate between the code and bunch of tester models.
    """

    def __init__(self, settings, dataset):
        self.settings = settings
        self.model = Model2_SiamUnet_Encoder.Model2_SiamUnet_Encoder(settings, dataset)