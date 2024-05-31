import pandas as pd

def handle_exceptions(func):
    """ Decorator to handle exceptions in functions """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
            print(f"Error in function {func.__name__}: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred in function '{func.__name__}': {e}")
            raise
    return wrapper