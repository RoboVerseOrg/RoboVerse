class AttrDict(dict):
    """A dict that also lets you do obj.key in addition to obj['key']."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"{name!r} not found") from e

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(f"{name!r} not found") from e

    @classmethod
    def from_dict(cls, d):
        """Recursively convert dicts (and lists of dicts) into AttrDict."""
        if not isinstance(d, dict):
            raise TypeError(f"from_dict expects a dict, got {type(d)}")
        def _convert(obj):
            if isinstance(obj, dict):
                # recurse into dict
                return cls({k: _convert(v) for k, v in obj.items()})
            elif isinstance(obj, list):
                # recurse into list items
                return [_convert(v) for v in obj]
            else:
                return obj
        return _convert(d)
