
class Settings(object):
    """
    Project wide management of settings
    """

    def __init__(self, args):
        self.args = args

        self.run_name = args.name
        self.default_raster_shape = (256,256,4)
        self.default_vector_shape = (256,256)
        self.large_file_folder = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"