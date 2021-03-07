from typing import List, Union


def cache(depend_attr: Union[List[str], str]):
    def decorator(method):
        return Cache(method, depend_attr)
    return decorator


class Cache(object):
    """
    Computes attribute value and caches it in the instance.
    """
    def __init__(self, method, depend_attr, name=None):
        # record the unbound-method and the name
        self.depend_attr = depend_attr
        self.method = method
        self.name = name or method.__name__
        self.last_id = []
        self.result = None
        self.__doc__ = method.__doc__

    def __get__(self, inst, cls):
        # self: <__main__.cache object at 0xb781340c>
        # inst: <__main__.Foo object at 0xb781348c>
        # cls: <class '__main__.Foo'>
        if inst is None:
            # instance attribute accessed on class, return self
            # You get here if you write `Foo.bar`
            return self

        # compute, cache and return the instance's attribute value
        if isinstance(self.depend_attr, list):
            new_id = [id(getattr(inst, da)) for da in self.depend_attr]
        else:
            new_id = [id(getattr(inst, self.depend_attr))]

        if new_id != self.last_id:
            result = self.method(inst)
            self.result = result
        else:
            result = self.result

        return result
