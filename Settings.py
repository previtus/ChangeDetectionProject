
class Settings(object):
    """
    Project wide management of settings
    """

    def __init__(self, args):
        self.args = args

        self.run_name = args.name
        self.large_file_folder = "/scratch/ruzicka/python_projects_large/ChangeDetectionProject_files/"

        self.verbose = 1
        # 0 = zip it!
        # 1 = print important info for run
        # 2 = print a lot of stuff
        # 3 = debugger level of details and checks