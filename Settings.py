
class Settings(object):
    """
    Project wide management of settings
    """

    def __init__(self, args):
        self.args = args

        self.run_name = args.name
        self.large_file_folder = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"