class CongfigDict(dict):
    def __init__(self):
        super().__init__(self)

    def __setitem__(self, key, value) -> None:
        super().__setattr__(key, value)
        return super().__setitem__(key, value)
    
    def get(self,key):
        try:
            return super().__getattribute__(key)
        except:
            return None
    
    def __getitem__(self, key):
        return super().__getattribute__(key)
    
    def __delitem__(self, key) -> None:
        super().__delattr__(key)
        return super().__delitem__(key)