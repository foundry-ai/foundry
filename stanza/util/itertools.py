class BackIterator:
    def __init__(self, iterator, buffer=None):
        self.iterator = iterator
        self.buffer = buffer or []
    
    def put_back(self, *items):
        items = list(items)
        items.reverse()
        self.buffer.extend(items)
    
    def __iter__(self):
        return self

    def __next__(self):
        if len(self.buffer) > 0:
            return self.buffer.pop()
        return next(self.iterator)

def make_put_backable(iterator):
    if isinstance(iterator, BackIterator):
        return iterator
    return BackIterator(iterator)

def put_back(iterator, *items):
    if not hasattr(iterator, 'put_back'):
        raise ValueError("Iterator must support put_back, use make_put_backable()")
    iterator.put_back(*items)