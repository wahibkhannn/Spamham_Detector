import os


def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


# class SpamhamException(Exception):
#     def __init__(self, error_message, error_detail):
#         """
#         :param error_message: error message in string format
#         """
#         super().__init__(error_message)
#         self.error_message = error_message_detail(
#             error_message, error_detail=error_detail
#         )

import traceback

class SpamhamException(Exception):
    def __init__(self, error_message: Exception, sys_module):
        super().__init__(str(error_message))
        self.error_message = str(error_message)
        self.traceback = traceback.format_exc()  # safe string of full traceback

    def __str__(self):
        return self.error_message
