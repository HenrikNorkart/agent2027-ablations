from __future__ import annotations


class Singleton(type):
    """Metaclass that ensures only one instance of a class is created.
    
    Implements the Singleton pattern at the metaclass level. Any class using this
    metaclass will maintain a single instance across all instantiation calls.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Create or return the singleton instance.
        
        Args:
            *args: Variable length argument list for class initialization.
            **kwargs: Arbitrary keyword arguments for class initialization.
            
        Returns:
            The singleton instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
